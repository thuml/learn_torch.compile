from __future__ import annotations



def forward(self, primals_1: "f32[960]", primals_2: "f32[960]", primals_3: "f32[1000, 1280]", primals_4: "f32[1000]", primals_5: "f32[16, 3, 3, 3]", primals_6: "f32[16]", primals_7: "f32[16]", primals_8: "f32[8, 16, 1, 1]", primals_9: "f32[8]", primals_10: "f32[8]", primals_11: "f32[8, 1, 3, 3]", primals_12: "f32[8]", primals_13: "f32[8]", primals_14: "f32[8, 16, 1, 1]", primals_15: "f32[8]", primals_16: "f32[8]", primals_17: "f32[8, 1, 3, 3]", primals_18: "f32[8]", primals_19: "f32[8]", primals_20: "f32[24, 16, 1, 1]", primals_21: "f32[24]", primals_22: "f32[24]", primals_23: "f32[24, 1, 3, 3]", primals_24: "f32[24]", primals_25: "f32[24]", primals_26: "f32[48, 1, 3, 3]", primals_27: "f32[48]", primals_28: "f32[48]", primals_29: "f32[12, 48, 1, 1]", primals_30: "f32[12]", primals_31: "f32[12]", primals_32: "f32[12, 1, 3, 3]", primals_33: "f32[12]", primals_34: "f32[12]", primals_35: "f32[16, 1, 3, 3]", primals_36: "f32[16]", primals_37: "f32[16]", primals_38: "f32[24, 16, 1, 1]", primals_39: "f32[24]", primals_40: "f32[24]", primals_41: "f32[36, 24, 1, 1]", primals_42: "f32[36]", primals_43: "f32[36]", primals_44: "f32[36, 1, 3, 3]", primals_45: "f32[36]", primals_46: "f32[36]", primals_47: "f32[12, 72, 1, 1]", primals_48: "f32[12]", primals_49: "f32[12]", primals_50: "f32[12, 1, 3, 3]", primals_51: "f32[12]", primals_52: "f32[12]", primals_53: "f32[36, 24, 1, 1]", primals_54: "f32[36]", primals_55: "f32[36]", primals_56: "f32[36, 1, 3, 3]", primals_57: "f32[36]", primals_58: "f32[36]", primals_59: "f32[72, 1, 5, 5]", primals_60: "f32[72]", primals_61: "f32[72]", primals_62: "f32[20, 72, 1, 1]", primals_63: "f32[20]", primals_64: "f32[72, 20, 1, 1]", primals_65: "f32[72]", primals_66: "f32[20, 72, 1, 1]", primals_67: "f32[20]", primals_68: "f32[20]", primals_69: "f32[20, 1, 3, 3]", primals_70: "f32[20]", primals_71: "f32[20]", primals_72: "f32[24, 1, 5, 5]", primals_73: "f32[24]", primals_74: "f32[24]", primals_75: "f32[40, 24, 1, 1]", primals_76: "f32[40]", primals_77: "f32[40]", primals_78: "f32[60, 40, 1, 1]", primals_79: "f32[60]", primals_80: "f32[60]", primals_81: "f32[60, 1, 3, 3]", primals_82: "f32[60]", primals_83: "f32[60]", primals_84: "f32[32, 120, 1, 1]", primals_85: "f32[32]", primals_86: "f32[120, 32, 1, 1]", primals_87: "f32[120]", primals_88: "f32[20, 120, 1, 1]", primals_89: "f32[20]", primals_90: "f32[20]", primals_91: "f32[20, 1, 3, 3]", primals_92: "f32[20]", primals_93: "f32[20]", primals_94: "f32[120, 40, 1, 1]", primals_95: "f32[120]", primals_96: "f32[120]", primals_97: "f32[120, 1, 3, 3]", primals_98: "f32[120]", primals_99: "f32[120]", primals_100: "f32[240, 1, 3, 3]", primals_101: "f32[240]", primals_102: "f32[240]", primals_103: "f32[40, 240, 1, 1]", primals_104: "f32[40]", primals_105: "f32[40]", primals_106: "f32[40, 1, 3, 3]", primals_107: "f32[40]", primals_108: "f32[40]", primals_109: "f32[40, 1, 3, 3]", primals_110: "f32[40]", primals_111: "f32[40]", primals_112: "f32[80, 40, 1, 1]", primals_113: "f32[80]", primals_114: "f32[80]", primals_115: "f32[100, 80, 1, 1]", primals_116: "f32[100]", primals_117: "f32[100]", primals_118: "f32[100, 1, 3, 3]", primals_119: "f32[100]", primals_120: "f32[100]", primals_121: "f32[40, 200, 1, 1]", primals_122: "f32[40]", primals_123: "f32[40]", primals_124: "f32[40, 1, 3, 3]", primals_125: "f32[40]", primals_126: "f32[40]", primals_127: "f32[92, 80, 1, 1]", primals_128: "f32[92]", primals_129: "f32[92]", primals_130: "f32[92, 1, 3, 3]", primals_131: "f32[92]", primals_132: "f32[92]", primals_133: "f32[40, 184, 1, 1]", primals_134: "f32[40]", primals_135: "f32[40]", primals_136: "f32[40, 1, 3, 3]", primals_137: "f32[40]", primals_138: "f32[40]", primals_139: "f32[92, 80, 1, 1]", primals_140: "f32[92]", primals_141: "f32[92]", primals_142: "f32[92, 1, 3, 3]", primals_143: "f32[92]", primals_144: "f32[92]", primals_145: "f32[40, 184, 1, 1]", primals_146: "f32[40]", primals_147: "f32[40]", primals_148: "f32[40, 1, 3, 3]", primals_149: "f32[40]", primals_150: "f32[40]", primals_151: "f32[240, 80, 1, 1]", primals_152: "f32[240]", primals_153: "f32[240]", primals_154: "f32[240, 1, 3, 3]", primals_155: "f32[240]", primals_156: "f32[240]", primals_157: "f32[120, 480, 1, 1]", primals_158: "f32[120]", primals_159: "f32[480, 120, 1, 1]", primals_160: "f32[480]", primals_161: "f32[56, 480, 1, 1]", primals_162: "f32[56]", primals_163: "f32[56]", primals_164: "f32[56, 1, 3, 3]", primals_165: "f32[56]", primals_166: "f32[56]", primals_167: "f32[80, 1, 3, 3]", primals_168: "f32[80]", primals_169: "f32[80]", primals_170: "f32[112, 80, 1, 1]", primals_171: "f32[112]", primals_172: "f32[112]", primals_173: "f32[336, 112, 1, 1]", primals_174: "f32[336]", primals_175: "f32[336]", primals_176: "f32[336, 1, 3, 3]", primals_177: "f32[336]", primals_178: "f32[336]", primals_179: "f32[168, 672, 1, 1]", primals_180: "f32[168]", primals_181: "f32[672, 168, 1, 1]", primals_182: "f32[672]", primals_183: "f32[56, 672, 1, 1]", primals_184: "f32[56]", primals_185: "f32[56]", primals_186: "f32[56, 1, 3, 3]", primals_187: "f32[56]", primals_188: "f32[56]", primals_189: "f32[336, 112, 1, 1]", primals_190: "f32[336]", primals_191: "f32[336]", primals_192: "f32[336, 1, 3, 3]", primals_193: "f32[336]", primals_194: "f32[336]", primals_195: "f32[672, 1, 5, 5]", primals_196: "f32[672]", primals_197: "f32[672]", primals_198: "f32[168, 672, 1, 1]", primals_199: "f32[168]", primals_200: "f32[672, 168, 1, 1]", primals_201: "f32[672]", primals_202: "f32[80, 672, 1, 1]", primals_203: "f32[80]", primals_204: "f32[80]", primals_205: "f32[80, 1, 3, 3]", primals_206: "f32[80]", primals_207: "f32[80]", primals_208: "f32[112, 1, 5, 5]", primals_209: "f32[112]", primals_210: "f32[112]", primals_211: "f32[160, 112, 1, 1]", primals_212: "f32[160]", primals_213: "f32[160]", primals_214: "f32[480, 160, 1, 1]", primals_215: "f32[480]", primals_216: "f32[480]", primals_217: "f32[480, 1, 3, 3]", primals_218: "f32[480]", primals_219: "f32[480]", primals_220: "f32[80, 960, 1, 1]", primals_221: "f32[80]", primals_222: "f32[80]", primals_223: "f32[80, 1, 3, 3]", primals_224: "f32[80]", primals_225: "f32[80]", primals_226: "f32[480, 160, 1, 1]", primals_227: "f32[480]", primals_228: "f32[480]", primals_229: "f32[480, 1, 3, 3]", primals_230: "f32[480]", primals_231: "f32[480]", primals_232: "f32[240, 960, 1, 1]", primals_233: "f32[240]", primals_234: "f32[960, 240, 1, 1]", primals_235: "f32[960]", primals_236: "f32[80, 960, 1, 1]", primals_237: "f32[80]", primals_238: "f32[80]", primals_239: "f32[80, 1, 3, 3]", primals_240: "f32[80]", primals_241: "f32[80]", primals_242: "f32[480, 160, 1, 1]", primals_243: "f32[480]", primals_244: "f32[480]", primals_245: "f32[480, 1, 3, 3]", primals_246: "f32[480]", primals_247: "f32[480]", primals_248: "f32[80, 960, 1, 1]", primals_249: "f32[80]", primals_250: "f32[80]", primals_251: "f32[80, 1, 3, 3]", primals_252: "f32[80]", primals_253: "f32[80]", primals_254: "f32[480, 160, 1, 1]", primals_255: "f32[480]", primals_256: "f32[480]", primals_257: "f32[480, 1, 3, 3]", primals_258: "f32[480]", primals_259: "f32[480]", primals_260: "f32[240, 960, 1, 1]", primals_261: "f32[240]", primals_262: "f32[960, 240, 1, 1]", primals_263: "f32[960]", primals_264: "f32[80, 960, 1, 1]", primals_265: "f32[80]", primals_266: "f32[80]", primals_267: "f32[80, 1, 3, 3]", primals_268: "f32[80]", primals_269: "f32[80]", primals_270: "f32[960, 160, 1, 1]", primals_271: "f32[1280, 960, 1, 1]", primals_272: "f32[1280]", primals_273: "i64[]", primals_274: "f32[960]", primals_275: "f32[960]", primals_276: "f32[16]", primals_277: "f32[16]", primals_278: "i64[]", primals_279: "f32[8]", primals_280: "f32[8]", primals_281: "i64[]", primals_282: "f32[8]", primals_283: "f32[8]", primals_284: "i64[]", primals_285: "f32[8]", primals_286: "f32[8]", primals_287: "i64[]", primals_288: "f32[8]", primals_289: "f32[8]", primals_290: "i64[]", primals_291: "f32[24]", primals_292: "f32[24]", primals_293: "i64[]", primals_294: "f32[24]", primals_295: "f32[24]", primals_296: "i64[]", primals_297: "f32[48]", primals_298: "f32[48]", primals_299: "i64[]", primals_300: "f32[12]", primals_301: "f32[12]", primals_302: "i64[]", primals_303: "f32[12]", primals_304: "f32[12]", primals_305: "i64[]", primals_306: "f32[16]", primals_307: "f32[16]", primals_308: "i64[]", primals_309: "f32[24]", primals_310: "f32[24]", primals_311: "i64[]", primals_312: "f32[36]", primals_313: "f32[36]", primals_314: "i64[]", primals_315: "f32[36]", primals_316: "f32[36]", primals_317: "i64[]", primals_318: "f32[12]", primals_319: "f32[12]", primals_320: "i64[]", primals_321: "f32[12]", primals_322: "f32[12]", primals_323: "i64[]", primals_324: "f32[36]", primals_325: "f32[36]", primals_326: "i64[]", primals_327: "f32[36]", primals_328: "f32[36]", primals_329: "i64[]", primals_330: "f32[72]", primals_331: "f32[72]", primals_332: "i64[]", primals_333: "f32[20]", primals_334: "f32[20]", primals_335: "i64[]", primals_336: "f32[20]", primals_337: "f32[20]", primals_338: "i64[]", primals_339: "f32[24]", primals_340: "f32[24]", primals_341: "i64[]", primals_342: "f32[40]", primals_343: "f32[40]", primals_344: "i64[]", primals_345: "f32[60]", primals_346: "f32[60]", primals_347: "i64[]", primals_348: "f32[60]", primals_349: "f32[60]", primals_350: "i64[]", primals_351: "f32[20]", primals_352: "f32[20]", primals_353: "i64[]", primals_354: "f32[20]", primals_355: "f32[20]", primals_356: "i64[]", primals_357: "f32[120]", primals_358: "f32[120]", primals_359: "i64[]", primals_360: "f32[120]", primals_361: "f32[120]", primals_362: "i64[]", primals_363: "f32[240]", primals_364: "f32[240]", primals_365: "i64[]", primals_366: "f32[40]", primals_367: "f32[40]", primals_368: "i64[]", primals_369: "f32[40]", primals_370: "f32[40]", primals_371: "i64[]", primals_372: "f32[40]", primals_373: "f32[40]", primals_374: "i64[]", primals_375: "f32[80]", primals_376: "f32[80]", primals_377: "i64[]", primals_378: "f32[100]", primals_379: "f32[100]", primals_380: "i64[]", primals_381: "f32[100]", primals_382: "f32[100]", primals_383: "i64[]", primals_384: "f32[40]", primals_385: "f32[40]", primals_386: "i64[]", primals_387: "f32[40]", primals_388: "f32[40]", primals_389: "i64[]", primals_390: "f32[92]", primals_391: "f32[92]", primals_392: "i64[]", primals_393: "f32[92]", primals_394: "f32[92]", primals_395: "i64[]", primals_396: "f32[40]", primals_397: "f32[40]", primals_398: "i64[]", primals_399: "f32[40]", primals_400: "f32[40]", primals_401: "i64[]", primals_402: "f32[92]", primals_403: "f32[92]", primals_404: "i64[]", primals_405: "f32[92]", primals_406: "f32[92]", primals_407: "i64[]", primals_408: "f32[40]", primals_409: "f32[40]", primals_410: "i64[]", primals_411: "f32[40]", primals_412: "f32[40]", primals_413: "i64[]", primals_414: "f32[240]", primals_415: "f32[240]", primals_416: "i64[]", primals_417: "f32[240]", primals_418: "f32[240]", primals_419: "i64[]", primals_420: "f32[56]", primals_421: "f32[56]", primals_422: "i64[]", primals_423: "f32[56]", primals_424: "f32[56]", primals_425: "i64[]", primals_426: "f32[80]", primals_427: "f32[80]", primals_428: "i64[]", primals_429: "f32[112]", primals_430: "f32[112]", primals_431: "i64[]", primals_432: "f32[336]", primals_433: "f32[336]", primals_434: "i64[]", primals_435: "f32[336]", primals_436: "f32[336]", primals_437: "i64[]", primals_438: "f32[56]", primals_439: "f32[56]", primals_440: "i64[]", primals_441: "f32[56]", primals_442: "f32[56]", primals_443: "i64[]", primals_444: "f32[336]", primals_445: "f32[336]", primals_446: "i64[]", primals_447: "f32[336]", primals_448: "f32[336]", primals_449: "i64[]", primals_450: "f32[672]", primals_451: "f32[672]", primals_452: "i64[]", primals_453: "f32[80]", primals_454: "f32[80]", primals_455: "i64[]", primals_456: "f32[80]", primals_457: "f32[80]", primals_458: "i64[]", primals_459: "f32[112]", primals_460: "f32[112]", primals_461: "i64[]", primals_462: "f32[160]", primals_463: "f32[160]", primals_464: "i64[]", primals_465: "f32[480]", primals_466: "f32[480]", primals_467: "i64[]", primals_468: "f32[480]", primals_469: "f32[480]", primals_470: "i64[]", primals_471: "f32[80]", primals_472: "f32[80]", primals_473: "i64[]", primals_474: "f32[80]", primals_475: "f32[80]", primals_476: "i64[]", primals_477: "f32[480]", primals_478: "f32[480]", primals_479: "i64[]", primals_480: "f32[480]", primals_481: "f32[480]", primals_482: "i64[]", primals_483: "f32[80]", primals_484: "f32[80]", primals_485: "i64[]", primals_486: "f32[80]", primals_487: "f32[80]", primals_488: "i64[]", primals_489: "f32[480]", primals_490: "f32[480]", primals_491: "i64[]", primals_492: "f32[480]", primals_493: "f32[480]", primals_494: "i64[]", primals_495: "f32[80]", primals_496: "f32[80]", primals_497: "i64[]", primals_498: "f32[80]", primals_499: "f32[80]", primals_500: "i64[]", primals_501: "f32[480]", primals_502: "f32[480]", primals_503: "i64[]", primals_504: "f32[480]", primals_505: "f32[480]", primals_506: "i64[]", primals_507: "f32[80]", primals_508: "f32[80]", primals_509: "i64[]", primals_510: "f32[80]", primals_511: "f32[80]", primals_512: "i64[]", primals_513: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:282, code: x = self.conv_stem(x)
    convolution: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(primals_513, primals_5, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:283, code: x = self.bn1(x)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_278, 1)
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
    mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(primals_276, 0.9)
    add_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[16]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
    add_3: "f32[16]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:284, code: x = self.act1(x)
    relu: "f32[8, 16, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_1: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_8, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_281, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 8, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 8, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 8, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[8]" = torch.ops.aten.mul.Tensor(primals_279, 0.9)
    add_7: "f32[8]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_11: "f32[8]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[8]" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
    add_8: "f32[8]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_5: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_7: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    relu_1: "f32[8, 8, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_2: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(relu_1, primals_11, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_284, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 8, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 8, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 8, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[8]" = torch.ops.aten.mul.Tensor(primals_282, 0.9)
    add_12: "f32[8]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
    mul_18: "f32[8]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[8]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
    add_13: "f32[8]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1)
    unsqueeze_9: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
    unsqueeze_11: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    relu_2: "f32[8, 8, 112, 112]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat: "f32[8, 16, 112, 112]" = torch.ops.aten.cat.default([relu_1, relu_2], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_1: "f32[8, 16, 112, 112]" = torch.ops.aten.slice.Tensor(cat, 0, 0, 9223372036854775807);  cat = None
    slice_2: "f32[8, 16, 112, 112]" = torch.ops.aten.slice.Tensor(slice_1, 2, 0, 9223372036854775807);  slice_1 = None
    slice_3: "f32[8, 16, 112, 112]" = torch.ops.aten.slice.Tensor(slice_2, 3, 0, 9223372036854775807);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_3: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(slice_3, primals_14, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_287, 1)
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 8, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 8, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 8, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_21: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[8]" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
    add_17: "f32[8]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_24: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
    mul_25: "f32[8]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[8]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
    add_18: "f32[8]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_13: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_15: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_4: "f32[8, 8, 112, 112]" = torch.ops.aten.convolution.default(add_19, primals_17, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_290, 1)
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 8, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 8, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 8, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 8, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
    mul_28: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[8]" = torch.ops.aten.mul.Tensor(primals_288, 0.9)
    add_22: "f32[8]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_31: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.00000996502277);  squeeze_14 = None
    mul_32: "f32[8]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[8]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
    add_23: "f32[8]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
    unsqueeze_17: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 8, 112, 112]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[8, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
    unsqueeze_19: "f32[8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 8, 112, 112]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_1: "f32[8, 16, 112, 112]" = torch.ops.aten.cat.default([add_19, add_24], 1);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_4: "f32[8, 16, 112, 112]" = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
    slice_5: "f32[8, 16, 112, 112]" = torch.ops.aten.slice.Tensor(slice_4, 2, 0, 9223372036854775807)
    slice_6: "f32[8, 16, 112, 112]" = torch.ops.aten.slice.Tensor(slice_5, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_25: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(slice_6, relu);  slice_6 = None
    slice_scatter: "f32[8, 16, 112, 112]" = torch.ops.aten.slice_scatter.default(slice_5, add_25, 3, 0, 9223372036854775807);  slice_5 = add_25 = None
    slice_scatter_1: "f32[8, 16, 112, 112]" = torch.ops.aten.slice_scatter.default(slice_4, slice_scatter, 2, 0, 9223372036854775807);  slice_4 = slice_scatter = None
    slice_scatter_2: "f32[8, 16, 112, 112]" = torch.ops.aten.slice_scatter.default(cat_1, slice_scatter_1, 0, 0, 9223372036854775807);  cat_1 = slice_scatter_1 = None
    slice_9: "f32[8, 16, 112, 112]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 0, 0, 9223372036854775807);  slice_scatter_2 = None
    slice_10: "f32[8, 16, 112, 112]" = torch.ops.aten.slice.Tensor(slice_9, 2, 0, 9223372036854775807);  slice_9 = None
    slice_11: "f32[8, 16, 112, 112]" = torch.ops.aten.slice.Tensor(slice_10, 3, 0, 9223372036854775807);  slice_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_5: "f32[8, 24, 112, 112]" = torch.ops.aten.convolution.default(slice_11, primals_20, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_293, 1)
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 24, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 24, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_5: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
    mul_35: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[24]" = torch.ops.aten.mul.Tensor(primals_291, 0.9)
    add_28: "f32[24]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_38: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.00000996502277);  squeeze_17 = None
    mul_39: "f32[24]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[24]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
    add_29: "f32[24]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_21: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_23: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_30: "f32[8, 24, 112, 112]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    relu_3: "f32[8, 24, 112, 112]" = torch.ops.aten.relu.default(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_6: "f32[8, 24, 112, 112]" = torch.ops.aten.convolution.default(relu_3, primals_23, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 24)
    add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_296, 1)
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 24, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 24, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_32: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_6: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
    mul_42: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[24]" = torch.ops.aten.mul.Tensor(primals_294, 0.9)
    add_33: "f32[24]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_45: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.00000996502277);  squeeze_20 = None
    mul_46: "f32[24]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[24]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
    add_34: "f32[24]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1)
    unsqueeze_25: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
    unsqueeze_27: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_35: "f32[8, 24, 112, 112]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    relu_4: "f32[8, 24, 112, 112]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_2: "f32[8, 48, 112, 112]" = torch.ops.aten.cat.default([relu_3, relu_4], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_12: "f32[8, 48, 112, 112]" = torch.ops.aten.slice.Tensor(cat_2, 0, 0, 9223372036854775807);  cat_2 = None
    slice_13: "f32[8, 48, 112, 112]" = torch.ops.aten.slice.Tensor(slice_12, 2, 0, 9223372036854775807);  slice_12 = None
    slice_14: "f32[8, 48, 112, 112]" = torch.ops.aten.slice.Tensor(slice_13, 3, 0, 9223372036854775807);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    convolution_7: "f32[8, 48, 56, 56]" = torch.ops.aten.convolution.default(slice_14, primals_26, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_299, 1)
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 48, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 48, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
    mul_49: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[48]" = torch.ops.aten.mul.Tensor(primals_297, 0.9)
    add_38: "f32[48]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_52: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_53: "f32[48]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[48]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
    add_39: "f32[48]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_29: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_31: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_8: "f32[8, 12, 56, 56]" = torch.ops.aten.convolution.default(add_40, primals_29, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_302, 1)
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 12, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 12, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 12, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 12, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
    mul_56: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[12]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[12]" = torch.ops.aten.mul.Tensor(primals_300, 0.9)
    add_43: "f32[12]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_59: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_60: "f32[12]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[12]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
    add_44: "f32[12]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1)
    unsqueeze_33: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
    unsqueeze_35: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_9: "f32[8, 12, 56, 56]" = torch.ops.aten.convolution.default(add_45, primals_32, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 12)
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_305, 1)
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 12, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 12, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 12, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 12, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
    mul_63: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[12]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[12]" = torch.ops.aten.mul.Tensor(primals_303, 0.9)
    add_48: "f32[12]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_66: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_67: "f32[12]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[12]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
    add_49: "f32[12]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_37: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_39: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_3: "f32[8, 24, 56, 56]" = torch.ops.aten.cat.default([add_45, add_50], 1);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_15: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(cat_3, 0, 0, 9223372036854775807)
    slice_16: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(slice_15, 2, 0, 9223372036854775807)
    slice_17: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(slice_16, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    convolution_10: "f32[8, 16, 56, 56]" = torch.ops.aten.convolution.default(slice_11, primals_35, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16)
    add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_308, 1)
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 16, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 16, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_52: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_10: "f32[8, 16, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
    mul_70: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_71: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_72: "f32[16]" = torch.ops.aten.mul.Tensor(primals_306, 0.9)
    add_53: "f32[16]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_32: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_73: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000398612827361);  squeeze_32 = None
    mul_74: "f32[16]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[16]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_54: "f32[16]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_40: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1)
    unsqueeze_41: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_76: "f32[8, 16, 56, 56]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
    unsqueeze_42: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
    unsqueeze_43: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_55: "f32[8, 16, 56, 56]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
    convolution_11: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(add_55, primals_38, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_56: "i64[]" = torch.ops.aten.add.Tensor(primals_311, 1)
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 24, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 24, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_11: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
    mul_77: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_78: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_79: "f32[24]" = torch.ops.aten.mul.Tensor(primals_309, 0.9)
    add_58: "f32[24]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_35: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_80: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000398612827361);  squeeze_35 = None
    mul_81: "f32[24]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[24]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_59: "f32[24]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_44: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_45: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_83: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
    unsqueeze_46: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_47: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_60: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
    add_61: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(slice_17, add_60);  slice_17 = add_60 = None
    slice_scatter_3: "f32[8, 24, 56, 56]" = torch.ops.aten.slice_scatter.default(slice_16, add_61, 3, 0, 9223372036854775807);  slice_16 = add_61 = None
    slice_scatter_4: "f32[8, 24, 56, 56]" = torch.ops.aten.slice_scatter.default(slice_15, slice_scatter_3, 2, 0, 9223372036854775807);  slice_15 = slice_scatter_3 = None
    slice_scatter_5: "f32[8, 24, 56, 56]" = torch.ops.aten.slice_scatter.default(cat_3, slice_scatter_4, 0, 0, 9223372036854775807);  cat_3 = slice_scatter_4 = None
    slice_20: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 0, 0, 9223372036854775807);  slice_scatter_5 = None
    slice_21: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(slice_20, 2, 0, 9223372036854775807);  slice_20 = None
    slice_22: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(slice_21, 3, 0, 9223372036854775807);  slice_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_12: "f32[8, 36, 56, 56]" = torch.ops.aten.convolution.default(slice_22, primals_41, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_314, 1)
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 36, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 36, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 36, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 36, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_12: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
    mul_84: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[36]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_85: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_86: "f32[36]" = torch.ops.aten.mul.Tensor(primals_312, 0.9)
    add_64: "f32[36]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_38: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_87: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000398612827361);  squeeze_38 = None
    mul_88: "f32[36]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[36]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
    add_65: "f32[36]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_48: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1)
    unsqueeze_49: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_90: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
    unsqueeze_50: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
    unsqueeze_51: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
    relu_5: "f32[8, 36, 56, 56]" = torch.ops.aten.relu.default(add_66);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_13: "f32[8, 36, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_44, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 36)
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_317, 1)
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 36, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 36, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_68: "f32[1, 36, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 36, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_13: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
    mul_91: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[36]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_92: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_93: "f32[36]" = torch.ops.aten.mul.Tensor(primals_315, 0.9)
    add_69: "f32[36]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    squeeze_41: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_94: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000398612827361);  squeeze_41 = None
    mul_95: "f32[36]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
    mul_96: "f32[36]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
    add_70: "f32[36]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    unsqueeze_52: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_53: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_97: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
    unsqueeze_54: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_55: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_71: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
    relu_6: "f32[8, 36, 56, 56]" = torch.ops.aten.relu.default(add_71);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_4: "f32[8, 72, 56, 56]" = torch.ops.aten.cat.default([relu_5, relu_6], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_23: "f32[8, 72, 56, 56]" = torch.ops.aten.slice.Tensor(cat_4, 0, 0, 9223372036854775807);  cat_4 = None
    slice_24: "f32[8, 72, 56, 56]" = torch.ops.aten.slice.Tensor(slice_23, 2, 0, 9223372036854775807);  slice_23 = None
    slice_25: "f32[8, 72, 56, 56]" = torch.ops.aten.slice.Tensor(slice_24, 3, 0, 9223372036854775807);  slice_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_14: "f32[8, 12, 56, 56]" = torch.ops.aten.convolution.default(slice_25, primals_47, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_72: "i64[]" = torch.ops.aten.add.Tensor(primals_320, 1)
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 12, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 12, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_73: "f32[1, 12, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 12, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_14: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
    mul_98: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[12]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_99: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_100: "f32[12]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
    add_74: "f32[12]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_44: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_101: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000398612827361);  squeeze_44 = None
    mul_102: "f32[12]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[12]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_75: "f32[12]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_56: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1)
    unsqueeze_57: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_104: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
    unsqueeze_58: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
    unsqueeze_59: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_76: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_15: "f32[8, 12, 56, 56]" = torch.ops.aten.convolution.default(add_76, primals_50, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 12)
    add_77: "i64[]" = torch.ops.aten.add.Tensor(primals_323, 1)
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 12, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 12, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_78: "f32[1, 12, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 12, 1, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_15: "f32[8, 12, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
    mul_105: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[12]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_106: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_107: "f32[12]" = torch.ops.aten.mul.Tensor(primals_321, 0.9)
    add_79: "f32[12]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_47: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_108: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000398612827361);  squeeze_47 = None
    mul_109: "f32[12]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[12]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_80: "f32[12]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_60: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_61: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_111: "f32[8, 12, 56, 56]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
    unsqueeze_62: "f32[12, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_63: "f32[12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_81: "f32[8, 12, 56, 56]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_5: "f32[8, 24, 56, 56]" = torch.ops.aten.cat.default([add_76, add_81], 1);  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_26: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(cat_5, 0, 0, 9223372036854775807)
    slice_27: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(slice_26, 2, 0, 9223372036854775807)
    slice_28: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(slice_27, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_82: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(slice_28, slice_22);  slice_28 = None
    slice_scatter_6: "f32[8, 24, 56, 56]" = torch.ops.aten.slice_scatter.default(slice_27, add_82, 3, 0, 9223372036854775807);  slice_27 = add_82 = None
    slice_scatter_7: "f32[8, 24, 56, 56]" = torch.ops.aten.slice_scatter.default(slice_26, slice_scatter_6, 2, 0, 9223372036854775807);  slice_26 = slice_scatter_6 = None
    slice_scatter_8: "f32[8, 24, 56, 56]" = torch.ops.aten.slice_scatter.default(cat_5, slice_scatter_7, 0, 0, 9223372036854775807);  cat_5 = slice_scatter_7 = None
    slice_31: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 0, 0, 9223372036854775807);  slice_scatter_8 = None
    slice_32: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(slice_31, 2, 0, 9223372036854775807);  slice_31 = None
    slice_33: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(slice_32, 3, 0, 9223372036854775807);  slice_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_16: "f32[8, 36, 56, 56]" = torch.ops.aten.convolution.default(slice_33, primals_53, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_326, 1)
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 36, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 36, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_84: "f32[1, 36, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 36, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_16: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
    mul_112: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[36]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_113: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_114: "f32[36]" = torch.ops.aten.mul.Tensor(primals_324, 0.9)
    add_85: "f32[36]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_50: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_115: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000398612827361);  squeeze_50 = None
    mul_116: "f32[36]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[36]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_86: "f32[36]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_64: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1)
    unsqueeze_65: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_118: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
    unsqueeze_66: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
    unsqueeze_67: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_87: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
    relu_7: "f32[8, 36, 56, 56]" = torch.ops.aten.relu.default(add_87);  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_17: "f32[8, 36, 56, 56]" = torch.ops.aten.convolution.default(relu_7, primals_56, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 36)
    add_88: "i64[]" = torch.ops.aten.add.Tensor(primals_329, 1)
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 36, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 36, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_89: "f32[1, 36, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 36, 1, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_17: "f32[8, 36, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_35)
    mul_119: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[36]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_120: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_121: "f32[36]" = torch.ops.aten.mul.Tensor(primals_327, 0.9)
    add_90: "f32[36]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_53: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_122: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000398612827361);  squeeze_53 = None
    mul_123: "f32[36]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[36]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_91: "f32[36]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_68: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_69: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_125: "f32[8, 36, 56, 56]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
    unsqueeze_70: "f32[36, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_71: "f32[36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_92: "f32[8, 36, 56, 56]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
    relu_8: "f32[8, 36, 56, 56]" = torch.ops.aten.relu.default(add_92);  add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_6: "f32[8, 72, 56, 56]" = torch.ops.aten.cat.default([relu_7, relu_8], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_34: "f32[8, 72, 56, 56]" = torch.ops.aten.slice.Tensor(cat_6, 0, 0, 9223372036854775807);  cat_6 = None
    slice_35: "f32[8, 72, 56, 56]" = torch.ops.aten.slice.Tensor(slice_34, 2, 0, 9223372036854775807);  slice_34 = None
    slice_36: "f32[8, 72, 56, 56]" = torch.ops.aten.slice.Tensor(slice_35, 3, 0, 9223372036854775807);  slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    convolution_18: "f32[8, 72, 28, 28]" = torch.ops.aten.convolution.default(slice_36, primals_59, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    add_93: "i64[]" = torch.ops.aten.add.Tensor(primals_332, 1)
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 72, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 72, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_94: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_18: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_18: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_37)
    mul_126: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_127: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_128: "f32[72]" = torch.ops.aten.mul.Tensor(primals_330, 0.9)
    add_95: "f32[72]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_56: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_129: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
    mul_130: "f32[72]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[72]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_96: "f32[72]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_72: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1)
    unsqueeze_73: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_132: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
    unsqueeze_74: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
    unsqueeze_75: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_97: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 72, 1, 1]" = torch.ops.aten.mean.dim(add_97, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_19: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_62, primals_63, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_9: "f32[8, 20, 1, 1]" = torch.ops.aten.relu.default(convolution_19);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_20: "f32[8, 72, 1, 1]" = torch.ops.aten.convolution.default(relu_9, primals_64, primals_65, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_98: "f32[8, 72, 1, 1]" = torch.ops.aten.add.Tensor(convolution_20, 3)
    clamp_min: "f32[8, 72, 1, 1]" = torch.ops.aten.clamp_min.default(add_98, 0);  add_98 = None
    clamp_max: "f32[8, 72, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    div: "f32[8, 72, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max, 6);  clamp_max = None
    mul_133: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(add_97, div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_21: "f32[8, 20, 28, 28]" = torch.ops.aten.convolution.default(mul_133, primals_66, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_99: "i64[]" = torch.ops.aten.add.Tensor(primals_335, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 20, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 20, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_100: "f32[1, 20, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_19: "f32[1, 20, 1, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_19: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_39)
    mul_134: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[20]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_135: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_136: "f32[20]" = torch.ops.aten.mul.Tensor(primals_333, 0.9)
    add_101: "f32[20]" = torch.ops.aten.add.Tensor(mul_135, mul_136);  mul_135 = mul_136 = None
    squeeze_59: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_137: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
    mul_138: "f32[20]" = torch.ops.aten.mul.Tensor(mul_137, 0.1);  mul_137 = None
    mul_139: "f32[20]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_102: "f32[20]" = torch.ops.aten.add.Tensor(mul_138, mul_139);  mul_138 = mul_139 = None
    unsqueeze_76: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_77: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_140: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_134, unsqueeze_77);  mul_134 = unsqueeze_77 = None
    unsqueeze_78: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_79: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_103: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_79);  mul_140 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_22: "f32[8, 20, 28, 28]" = torch.ops.aten.convolution.default(add_103, primals_69, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 20)
    add_104: "i64[]" = torch.ops.aten.add.Tensor(primals_338, 1)
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 20, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 20, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_105: "f32[1, 20, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_20: "f32[1, 20, 1, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_20: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_41)
    mul_141: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[20]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_142: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_143: "f32[20]" = torch.ops.aten.mul.Tensor(primals_336, 0.9)
    add_106: "f32[20]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    squeeze_62: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_144: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
    mul_145: "f32[20]" = torch.ops.aten.mul.Tensor(mul_144, 0.1);  mul_144 = None
    mul_146: "f32[20]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_107: "f32[20]" = torch.ops.aten.add.Tensor(mul_145, mul_146);  mul_145 = mul_146 = None
    unsqueeze_80: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1)
    unsqueeze_81: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_147: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_141, unsqueeze_81);  mul_141 = unsqueeze_81 = None
    unsqueeze_82: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1);  primals_71 = None
    unsqueeze_83: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_108: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_147, unsqueeze_83);  mul_147 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_7: "f32[8, 40, 28, 28]" = torch.ops.aten.cat.default([add_103, add_108], 1);  add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_37: "f32[8, 40, 28, 28]" = torch.ops.aten.slice.Tensor(cat_7, 0, 0, 9223372036854775807)
    slice_38: "f32[8, 40, 28, 28]" = torch.ops.aten.slice.Tensor(slice_37, 2, 0, 9223372036854775807)
    slice_39: "f32[8, 40, 28, 28]" = torch.ops.aten.slice.Tensor(slice_38, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    convolution_23: "f32[8, 24, 28, 28]" = torch.ops.aten.convolution.default(slice_33, primals_72, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 24)
    add_109: "i64[]" = torch.ops.aten.add.Tensor(primals_341, 1)
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 24, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 24, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_110: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_21: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_21: "f32[8, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_43)
    mul_148: "f32[8, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_149: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_150: "f32[24]" = torch.ops.aten.mul.Tensor(primals_339, 0.9)
    add_111: "f32[24]" = torch.ops.aten.add.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
    squeeze_65: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_151: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
    mul_152: "f32[24]" = torch.ops.aten.mul.Tensor(mul_151, 0.1);  mul_151 = None
    mul_153: "f32[24]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_112: "f32[24]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
    unsqueeze_84: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_85: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_154: "f32[8, 24, 28, 28]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_85);  mul_148 = unsqueeze_85 = None
    unsqueeze_86: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_87: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_113: "f32[8, 24, 28, 28]" = torch.ops.aten.add.Tensor(mul_154, unsqueeze_87);  mul_154 = unsqueeze_87 = None
    convolution_24: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(add_113, primals_75, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_114: "i64[]" = torch.ops.aten.add.Tensor(primals_344, 1)
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 40, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 40, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_115: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_22: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_22: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_45)
    mul_155: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_156: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_157: "f32[40]" = torch.ops.aten.mul.Tensor(primals_342, 0.9)
    add_116: "f32[40]" = torch.ops.aten.add.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
    squeeze_68: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_158: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001594642002871);  squeeze_68 = None
    mul_159: "f32[40]" = torch.ops.aten.mul.Tensor(mul_158, 0.1);  mul_158 = None
    mul_160: "f32[40]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_117: "f32[40]" = torch.ops.aten.add.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    unsqueeze_88: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1)
    unsqueeze_89: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_161: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_155, unsqueeze_89);  mul_155 = unsqueeze_89 = None
    unsqueeze_90: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1);  primals_77 = None
    unsqueeze_91: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_118: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_91);  mul_161 = unsqueeze_91 = None
    add_119: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(slice_39, add_118);  slice_39 = add_118 = None
    slice_scatter_9: "f32[8, 40, 28, 28]" = torch.ops.aten.slice_scatter.default(slice_38, add_119, 3, 0, 9223372036854775807);  slice_38 = add_119 = None
    slice_scatter_10: "f32[8, 40, 28, 28]" = torch.ops.aten.slice_scatter.default(slice_37, slice_scatter_9, 2, 0, 9223372036854775807);  slice_37 = slice_scatter_9 = None
    slice_scatter_11: "f32[8, 40, 28, 28]" = torch.ops.aten.slice_scatter.default(cat_7, slice_scatter_10, 0, 0, 9223372036854775807);  cat_7 = slice_scatter_10 = None
    slice_42: "f32[8, 40, 28, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 0, 0, 9223372036854775807);  slice_scatter_11 = None
    slice_43: "f32[8, 40, 28, 28]" = torch.ops.aten.slice.Tensor(slice_42, 2, 0, 9223372036854775807);  slice_42 = None
    slice_44: "f32[8, 40, 28, 28]" = torch.ops.aten.slice.Tensor(slice_43, 3, 0, 9223372036854775807);  slice_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_25: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(slice_44, primals_78, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_120: "i64[]" = torch.ops.aten.add.Tensor(primals_347, 1)
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 60, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 60, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_121: "f32[1, 60, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_23: "f32[1, 60, 1, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_23: "f32[8, 60, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_47)
    mul_162: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[60]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[60]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_163: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_164: "f32[60]" = torch.ops.aten.mul.Tensor(primals_345, 0.9)
    add_122: "f32[60]" = torch.ops.aten.add.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None
    squeeze_71: "f32[60]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_165: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001594642002871);  squeeze_71 = None
    mul_166: "f32[60]" = torch.ops.aten.mul.Tensor(mul_165, 0.1);  mul_165 = None
    mul_167: "f32[60]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
    add_123: "f32[60]" = torch.ops.aten.add.Tensor(mul_166, mul_167);  mul_166 = mul_167 = None
    unsqueeze_92: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_93: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_168: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(mul_162, unsqueeze_93);  mul_162 = unsqueeze_93 = None
    unsqueeze_94: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_95: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_124: "f32[8, 60, 28, 28]" = torch.ops.aten.add.Tensor(mul_168, unsqueeze_95);  mul_168 = unsqueeze_95 = None
    relu_10: "f32[8, 60, 28, 28]" = torch.ops.aten.relu.default(add_124);  add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_26: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(relu_10, primals_81, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 60)
    add_125: "i64[]" = torch.ops.aten.add.Tensor(primals_350, 1)
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 60, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 60, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_126: "f32[1, 60, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_24: "f32[1, 60, 1, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_24: "f32[8, 60, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_49)
    mul_169: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[60]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[60]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_170: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_171: "f32[60]" = torch.ops.aten.mul.Tensor(primals_348, 0.9)
    add_127: "f32[60]" = torch.ops.aten.add.Tensor(mul_170, mul_171);  mul_170 = mul_171 = None
    squeeze_74: "f32[60]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_172: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001594642002871);  squeeze_74 = None
    mul_173: "f32[60]" = torch.ops.aten.mul.Tensor(mul_172, 0.1);  mul_172 = None
    mul_174: "f32[60]" = torch.ops.aten.mul.Tensor(primals_349, 0.9)
    add_128: "f32[60]" = torch.ops.aten.add.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None
    unsqueeze_96: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1)
    unsqueeze_97: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_175: "f32[8, 60, 28, 28]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_97);  mul_169 = unsqueeze_97 = None
    unsqueeze_98: "f32[60, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1);  primals_83 = None
    unsqueeze_99: "f32[60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_129: "f32[8, 60, 28, 28]" = torch.ops.aten.add.Tensor(mul_175, unsqueeze_99);  mul_175 = unsqueeze_99 = None
    relu_11: "f32[8, 60, 28, 28]" = torch.ops.aten.relu.default(add_129);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_8: "f32[8, 120, 28, 28]" = torch.ops.aten.cat.default([relu_10, relu_11], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_45: "f32[8, 120, 28, 28]" = torch.ops.aten.slice.Tensor(cat_8, 0, 0, 9223372036854775807)
    slice_46: "f32[8, 120, 28, 28]" = torch.ops.aten.slice.Tensor(slice_45, 2, 0, 9223372036854775807);  slice_45 = None
    slice_47: "f32[8, 120, 28, 28]" = torch.ops.aten.slice.Tensor(slice_46, 3, 0, 9223372036854775807);  slice_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(slice_47, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_27: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_84, primals_85, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_12: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(convolution_27);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_28: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(relu_12, primals_86, primals_87, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_130: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_28, 3)
    clamp_min_1: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_130, 0);  add_130 = None
    clamp_max_1: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    div_1: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_1, 6);  clamp_max_1 = None
    mul_176: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(slice_47, div_1);  slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_29: "f32[8, 20, 28, 28]" = torch.ops.aten.convolution.default(mul_176, primals_88, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_131: "i64[]" = torch.ops.aten.add.Tensor(primals_353, 1)
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 20, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 20, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_132: "f32[1, 20, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_25: "f32[1, 20, 1, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    sub_25: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_51)
    mul_177: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[20]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_178: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_179: "f32[20]" = torch.ops.aten.mul.Tensor(primals_351, 0.9)
    add_133: "f32[20]" = torch.ops.aten.add.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None
    squeeze_77: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_180: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001594642002871);  squeeze_77 = None
    mul_181: "f32[20]" = torch.ops.aten.mul.Tensor(mul_180, 0.1);  mul_180 = None
    mul_182: "f32[20]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
    add_134: "f32[20]" = torch.ops.aten.add.Tensor(mul_181, mul_182);  mul_181 = mul_182 = None
    unsqueeze_100: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_101: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_183: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_177, unsqueeze_101);  mul_177 = unsqueeze_101 = None
    unsqueeze_102: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_103: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_135: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_183, unsqueeze_103);  mul_183 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_30: "f32[8, 20, 28, 28]" = torch.ops.aten.convolution.default(add_135, primals_91, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 20)
    add_136: "i64[]" = torch.ops.aten.add.Tensor(primals_356, 1)
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 20, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 20, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_137: "f32[1, 20, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_26: "f32[1, 20, 1, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_26: "f32[8, 20, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_53)
    mul_184: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[20]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_185: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_186: "f32[20]" = torch.ops.aten.mul.Tensor(primals_354, 0.9)
    add_138: "f32[20]" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
    squeeze_80: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_187: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001594642002871);  squeeze_80 = None
    mul_188: "f32[20]" = torch.ops.aten.mul.Tensor(mul_187, 0.1);  mul_187 = None
    mul_189: "f32[20]" = torch.ops.aten.mul.Tensor(primals_355, 0.9)
    add_139: "f32[20]" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    unsqueeze_104: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_105: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_190: "f32[8, 20, 28, 28]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_105);  mul_184 = unsqueeze_105 = None
    unsqueeze_106: "f32[20, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_107: "f32[20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_140: "f32[8, 20, 28, 28]" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_107);  mul_190 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_9: "f32[8, 40, 28, 28]" = torch.ops.aten.cat.default([add_135, add_140], 1);  add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_48: "f32[8, 40, 28, 28]" = torch.ops.aten.slice.Tensor(cat_9, 0, 0, 9223372036854775807)
    slice_49: "f32[8, 40, 28, 28]" = torch.ops.aten.slice.Tensor(slice_48, 2, 0, 9223372036854775807)
    slice_50: "f32[8, 40, 28, 28]" = torch.ops.aten.slice.Tensor(slice_49, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_141: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(slice_50, slice_44);  slice_50 = None
    slice_scatter_12: "f32[8, 40, 28, 28]" = torch.ops.aten.slice_scatter.default(slice_49, add_141, 3, 0, 9223372036854775807);  slice_49 = add_141 = None
    slice_scatter_13: "f32[8, 40, 28, 28]" = torch.ops.aten.slice_scatter.default(slice_48, slice_scatter_12, 2, 0, 9223372036854775807);  slice_48 = slice_scatter_12 = None
    slice_scatter_14: "f32[8, 40, 28, 28]" = torch.ops.aten.slice_scatter.default(cat_9, slice_scatter_13, 0, 0, 9223372036854775807);  cat_9 = slice_scatter_13 = None
    slice_53: "f32[8, 40, 28, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_14, 0, 0, 9223372036854775807);  slice_scatter_14 = None
    slice_54: "f32[8, 40, 28, 28]" = torch.ops.aten.slice.Tensor(slice_53, 2, 0, 9223372036854775807);  slice_53 = None
    slice_55: "f32[8, 40, 28, 28]" = torch.ops.aten.slice.Tensor(slice_54, 3, 0, 9223372036854775807);  slice_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_31: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(slice_55, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_142: "i64[]" = torch.ops.aten.add.Tensor(primals_359, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 120, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 120, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_143: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_27: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_27: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_55)
    mul_191: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_82: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_192: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_193: "f32[120]" = torch.ops.aten.mul.Tensor(primals_357, 0.9)
    add_144: "f32[120]" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
    squeeze_83: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_194: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0001594642002871);  squeeze_83 = None
    mul_195: "f32[120]" = torch.ops.aten.mul.Tensor(mul_194, 0.1);  mul_194 = None
    mul_196: "f32[120]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
    add_145: "f32[120]" = torch.ops.aten.add.Tensor(mul_195, mul_196);  mul_195 = mul_196 = None
    unsqueeze_108: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_109: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_197: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_191, unsqueeze_109);  mul_191 = unsqueeze_109 = None
    unsqueeze_110: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_111: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_146: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_111);  mul_197 = unsqueeze_111 = None
    relu_13: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_146);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_32: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(relu_13, primals_97, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120)
    add_147: "i64[]" = torch.ops.aten.add.Tensor(primals_362, 1)
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 120, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 120, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_148: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_28: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    sub_28: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_57)
    mul_198: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_85: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_199: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_200: "f32[120]" = torch.ops.aten.mul.Tensor(primals_360, 0.9)
    add_149: "f32[120]" = torch.ops.aten.add.Tensor(mul_199, mul_200);  mul_199 = mul_200 = None
    squeeze_86: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_201: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0001594642002871);  squeeze_86 = None
    mul_202: "f32[120]" = torch.ops.aten.mul.Tensor(mul_201, 0.1);  mul_201 = None
    mul_203: "f32[120]" = torch.ops.aten.mul.Tensor(primals_361, 0.9)
    add_150: "f32[120]" = torch.ops.aten.add.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None
    unsqueeze_112: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_113: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_204: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_198, unsqueeze_113);  mul_198 = unsqueeze_113 = None
    unsqueeze_114: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_115: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_151: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_204, unsqueeze_115);  mul_204 = unsqueeze_115 = None
    relu_14: "f32[8, 120, 28, 28]" = torch.ops.aten.relu.default(add_151);  add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_10: "f32[8, 240, 28, 28]" = torch.ops.aten.cat.default([relu_13, relu_14], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_56: "f32[8, 240, 28, 28]" = torch.ops.aten.slice.Tensor(cat_10, 0, 0, 9223372036854775807);  cat_10 = None
    slice_57: "f32[8, 240, 28, 28]" = torch.ops.aten.slice.Tensor(slice_56, 2, 0, 9223372036854775807);  slice_56 = None
    slice_58: "f32[8, 240, 28, 28]" = torch.ops.aten.slice.Tensor(slice_57, 3, 0, 9223372036854775807);  slice_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    convolution_33: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(slice_58, primals_100, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    add_152: "i64[]" = torch.ops.aten.add.Tensor(primals_365, 1)
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 240, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 240, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_153: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_29: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    sub_29: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_59)
    mul_205: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_88: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_206: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_207: "f32[240]" = torch.ops.aten.mul.Tensor(primals_363, 0.9)
    add_154: "f32[240]" = torch.ops.aten.add.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
    squeeze_89: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_208: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0006381620931717);  squeeze_89 = None
    mul_209: "f32[240]" = torch.ops.aten.mul.Tensor(mul_208, 0.1);  mul_208 = None
    mul_210: "f32[240]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
    add_155: "f32[240]" = torch.ops.aten.add.Tensor(mul_209, mul_210);  mul_209 = mul_210 = None
    unsqueeze_116: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_117: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_211: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_117);  mul_205 = unsqueeze_117 = None
    unsqueeze_118: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_119: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_156: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_211, unsqueeze_119);  mul_211 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_34: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_156, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_157: "i64[]" = torch.ops.aten.add.Tensor(primals_368, 1)
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 40, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 40, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_158: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_30: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_30: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_61)
    mul_212: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_91: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_213: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_214: "f32[40]" = torch.ops.aten.mul.Tensor(primals_366, 0.9)
    add_159: "f32[40]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
    squeeze_92: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_215: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0006381620931717);  squeeze_92 = None
    mul_216: "f32[40]" = torch.ops.aten.mul.Tensor(mul_215, 0.1);  mul_215 = None
    mul_217: "f32[40]" = torch.ops.aten.mul.Tensor(primals_367, 0.9)
    add_160: "f32[40]" = torch.ops.aten.add.Tensor(mul_216, mul_217);  mul_216 = mul_217 = None
    unsqueeze_120: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
    unsqueeze_121: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_218: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_212, unsqueeze_121);  mul_212 = unsqueeze_121 = None
    unsqueeze_122: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
    unsqueeze_123: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_161: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_123);  mul_218 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_35: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_161, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40)
    add_162: "i64[]" = torch.ops.aten.add.Tensor(primals_371, 1)
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 40, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 40, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_163: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_31: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_31: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_63)
    mul_219: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_94: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_220: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_221: "f32[40]" = torch.ops.aten.mul.Tensor(primals_369, 0.9)
    add_164: "f32[40]" = torch.ops.aten.add.Tensor(mul_220, mul_221);  mul_220 = mul_221 = None
    squeeze_95: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_222: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0006381620931717);  squeeze_95 = None
    mul_223: "f32[40]" = torch.ops.aten.mul.Tensor(mul_222, 0.1);  mul_222 = None
    mul_224: "f32[40]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
    add_165: "f32[40]" = torch.ops.aten.add.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None
    unsqueeze_124: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_125: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_225: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_219, unsqueeze_125);  mul_219 = unsqueeze_125 = None
    unsqueeze_126: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_127: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_166: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_225, unsqueeze_127);  mul_225 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_11: "f32[8, 80, 14, 14]" = torch.ops.aten.cat.default([add_161, add_166], 1);  add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_59: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(cat_11, 0, 0, 9223372036854775807)
    slice_60: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_59, 2, 0, 9223372036854775807)
    slice_61: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_60, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    convolution_36: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(slice_55, primals_109, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 40)
    add_167: "i64[]" = torch.ops.aten.add.Tensor(primals_374, 1)
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 40, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 40, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_168: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_32: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_32: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_65)
    mul_226: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_97: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_227: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_228: "f32[40]" = torch.ops.aten.mul.Tensor(primals_372, 0.9)
    add_169: "f32[40]" = torch.ops.aten.add.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
    squeeze_98: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_229: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0006381620931717);  squeeze_98 = None
    mul_230: "f32[40]" = torch.ops.aten.mul.Tensor(mul_229, 0.1);  mul_229 = None
    mul_231: "f32[40]" = torch.ops.aten.mul.Tensor(primals_373, 0.9)
    add_170: "f32[40]" = torch.ops.aten.add.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None
    unsqueeze_128: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_129: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_232: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_129);  mul_226 = unsqueeze_129 = None
    unsqueeze_130: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_131: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_171: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_232, unsqueeze_131);  mul_232 = unsqueeze_131 = None
    convolution_37: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(add_171, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_172: "i64[]" = torch.ops.aten.add.Tensor(primals_377, 1)
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 80, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 80, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_173: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_33: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    sub_33: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_67)
    mul_233: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_100: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_234: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_235: "f32[80]" = torch.ops.aten.mul.Tensor(primals_375, 0.9)
    add_174: "f32[80]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    squeeze_101: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_236: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0006381620931717);  squeeze_101 = None
    mul_237: "f32[80]" = torch.ops.aten.mul.Tensor(mul_236, 0.1);  mul_236 = None
    mul_238: "f32[80]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
    add_175: "f32[80]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
    unsqueeze_132: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_133: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_239: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_233, unsqueeze_133);  mul_233 = unsqueeze_133 = None
    unsqueeze_134: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_135: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_176: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_135);  mul_239 = unsqueeze_135 = None
    add_177: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(slice_61, add_176);  slice_61 = add_176 = None
    slice_scatter_15: "f32[8, 80, 14, 14]" = torch.ops.aten.slice_scatter.default(slice_60, add_177, 3, 0, 9223372036854775807);  slice_60 = add_177 = None
    slice_scatter_16: "f32[8, 80, 14, 14]" = torch.ops.aten.slice_scatter.default(slice_59, slice_scatter_15, 2, 0, 9223372036854775807);  slice_59 = slice_scatter_15 = None
    slice_scatter_17: "f32[8, 80, 14, 14]" = torch.ops.aten.slice_scatter.default(cat_11, slice_scatter_16, 0, 0, 9223372036854775807);  cat_11 = slice_scatter_16 = None
    slice_64: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_17, 0, 0, 9223372036854775807);  slice_scatter_17 = None
    slice_65: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_64, 2, 0, 9223372036854775807);  slice_64 = None
    slice_66: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_65, 3, 0, 9223372036854775807);  slice_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_38: "f32[8, 100, 14, 14]" = torch.ops.aten.convolution.default(slice_66, primals_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_178: "i64[]" = torch.ops.aten.add.Tensor(primals_380, 1)
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 100, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 100, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_179: "f32[1, 100, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_34: "f32[1, 100, 1, 1]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
    sub_34: "f32[8, 100, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_69)
    mul_240: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[100]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_103: "f32[100]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_241: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_242: "f32[100]" = torch.ops.aten.mul.Tensor(primals_378, 0.9)
    add_180: "f32[100]" = torch.ops.aten.add.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
    squeeze_104: "f32[100]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_243: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0006381620931717);  squeeze_104 = None
    mul_244: "f32[100]" = torch.ops.aten.mul.Tensor(mul_243, 0.1);  mul_243 = None
    mul_245: "f32[100]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
    add_181: "f32[100]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    unsqueeze_136: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
    unsqueeze_137: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_246: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(mul_240, unsqueeze_137);  mul_240 = unsqueeze_137 = None
    unsqueeze_138: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
    unsqueeze_139: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_182: "f32[8, 100, 14, 14]" = torch.ops.aten.add.Tensor(mul_246, unsqueeze_139);  mul_246 = unsqueeze_139 = None
    relu_15: "f32[8, 100, 14, 14]" = torch.ops.aten.relu.default(add_182);  add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_39: "f32[8, 100, 14, 14]" = torch.ops.aten.convolution.default(relu_15, primals_118, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 100)
    add_183: "i64[]" = torch.ops.aten.add.Tensor(primals_383, 1)
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 100, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 100, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_184: "f32[1, 100, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_35: "f32[1, 100, 1, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    sub_35: "f32[8, 100, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_71)
    mul_247: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[100]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_106: "f32[100]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_248: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_249: "f32[100]" = torch.ops.aten.mul.Tensor(primals_381, 0.9)
    add_185: "f32[100]" = torch.ops.aten.add.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None
    squeeze_107: "f32[100]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_250: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0006381620931717);  squeeze_107 = None
    mul_251: "f32[100]" = torch.ops.aten.mul.Tensor(mul_250, 0.1);  mul_250 = None
    mul_252: "f32[100]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
    add_186: "f32[100]" = torch.ops.aten.add.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
    unsqueeze_140: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_141: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_253: "f32[8, 100, 14, 14]" = torch.ops.aten.mul.Tensor(mul_247, unsqueeze_141);  mul_247 = unsqueeze_141 = None
    unsqueeze_142: "f32[100, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_143: "f32[100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_187: "f32[8, 100, 14, 14]" = torch.ops.aten.add.Tensor(mul_253, unsqueeze_143);  mul_253 = unsqueeze_143 = None
    relu_16: "f32[8, 100, 14, 14]" = torch.ops.aten.relu.default(add_187);  add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_12: "f32[8, 200, 14, 14]" = torch.ops.aten.cat.default([relu_15, relu_16], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_67: "f32[8, 200, 14, 14]" = torch.ops.aten.slice.Tensor(cat_12, 0, 0, 9223372036854775807);  cat_12 = None
    slice_68: "f32[8, 200, 14, 14]" = torch.ops.aten.slice.Tensor(slice_67, 2, 0, 9223372036854775807);  slice_67 = None
    slice_69: "f32[8, 200, 14, 14]" = torch.ops.aten.slice.Tensor(slice_68, 3, 0, 9223372036854775807);  slice_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_40: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(slice_69, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_188: "i64[]" = torch.ops.aten.add.Tensor(primals_386, 1)
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 40, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 40, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_189: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_36: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_36: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_73)
    mul_254: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_109: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_255: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_256: "f32[40]" = torch.ops.aten.mul.Tensor(primals_384, 0.9)
    add_190: "f32[40]" = torch.ops.aten.add.Tensor(mul_255, mul_256);  mul_255 = mul_256 = None
    squeeze_110: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_257: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0006381620931717);  squeeze_110 = None
    mul_258: "f32[40]" = torch.ops.aten.mul.Tensor(mul_257, 0.1);  mul_257 = None
    mul_259: "f32[40]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
    add_191: "f32[40]" = torch.ops.aten.add.Tensor(mul_258, mul_259);  mul_258 = mul_259 = None
    unsqueeze_144: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
    unsqueeze_145: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_260: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_145);  mul_254 = unsqueeze_145 = None
    unsqueeze_146: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
    unsqueeze_147: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_192: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_147);  mul_260 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_41: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_192, primals_124, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40)
    add_193: "i64[]" = torch.ops.aten.add.Tensor(primals_389, 1)
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 40, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 40, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_194: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_37: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_37: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_75)
    mul_261: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_112: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_262: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_263: "f32[40]" = torch.ops.aten.mul.Tensor(primals_387, 0.9)
    add_195: "f32[40]" = torch.ops.aten.add.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
    squeeze_113: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_264: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0006381620931717);  squeeze_113 = None
    mul_265: "f32[40]" = torch.ops.aten.mul.Tensor(mul_264, 0.1);  mul_264 = None
    mul_266: "f32[40]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
    add_196: "f32[40]" = torch.ops.aten.add.Tensor(mul_265, mul_266);  mul_265 = mul_266 = None
    unsqueeze_148: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_149: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_267: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_261, unsqueeze_149);  mul_261 = unsqueeze_149 = None
    unsqueeze_150: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_151: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_197: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_267, unsqueeze_151);  mul_267 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_13: "f32[8, 80, 14, 14]" = torch.ops.aten.cat.default([add_192, add_197], 1);  add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_70: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(cat_13, 0, 0, 9223372036854775807)
    slice_71: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_70, 2, 0, 9223372036854775807)
    slice_72: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_71, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_198: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(slice_72, slice_66);  slice_72 = None
    slice_scatter_18: "f32[8, 80, 14, 14]" = torch.ops.aten.slice_scatter.default(slice_71, add_198, 3, 0, 9223372036854775807);  slice_71 = add_198 = None
    slice_scatter_19: "f32[8, 80, 14, 14]" = torch.ops.aten.slice_scatter.default(slice_70, slice_scatter_18, 2, 0, 9223372036854775807);  slice_70 = slice_scatter_18 = None
    slice_scatter_20: "f32[8, 80, 14, 14]" = torch.ops.aten.slice_scatter.default(cat_13, slice_scatter_19, 0, 0, 9223372036854775807);  cat_13 = slice_scatter_19 = None
    slice_75: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_20, 0, 0, 9223372036854775807);  slice_scatter_20 = None
    slice_76: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_75, 2, 0, 9223372036854775807);  slice_75 = None
    slice_77: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_76, 3, 0, 9223372036854775807);  slice_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_42: "f32[8, 92, 14, 14]" = torch.ops.aten.convolution.default(slice_77, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_199: "i64[]" = torch.ops.aten.add.Tensor(primals_392, 1)
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 92, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 92, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_200: "f32[1, 92, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_38: "f32[1, 92, 1, 1]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
    sub_38: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_77)
    mul_268: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_115: "f32[92]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_269: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_270: "f32[92]" = torch.ops.aten.mul.Tensor(primals_390, 0.9)
    add_201: "f32[92]" = torch.ops.aten.add.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
    squeeze_116: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_271: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0006381620931717);  squeeze_116 = None
    mul_272: "f32[92]" = torch.ops.aten.mul.Tensor(mul_271, 0.1);  mul_271 = None
    mul_273: "f32[92]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
    add_202: "f32[92]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
    unsqueeze_152: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
    unsqueeze_153: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_274: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_153);  mul_268 = unsqueeze_153 = None
    unsqueeze_154: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
    unsqueeze_155: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_203: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_274, unsqueeze_155);  mul_274 = unsqueeze_155 = None
    relu_17: "f32[8, 92, 14, 14]" = torch.ops.aten.relu.default(add_203);  add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_43: "f32[8, 92, 14, 14]" = torch.ops.aten.convolution.default(relu_17, primals_130, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 92)
    add_204: "i64[]" = torch.ops.aten.add.Tensor(primals_395, 1)
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 92, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 92, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_205: "f32[1, 92, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_39: "f32[1, 92, 1, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    sub_39: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_79)
    mul_275: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_118: "f32[92]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_276: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_277: "f32[92]" = torch.ops.aten.mul.Tensor(primals_393, 0.9)
    add_206: "f32[92]" = torch.ops.aten.add.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
    squeeze_119: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_278: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0006381620931717);  squeeze_119 = None
    mul_279: "f32[92]" = torch.ops.aten.mul.Tensor(mul_278, 0.1);  mul_278 = None
    mul_280: "f32[92]" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
    add_207: "f32[92]" = torch.ops.aten.add.Tensor(mul_279, mul_280);  mul_279 = mul_280 = None
    unsqueeze_156: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_157: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_281: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_275, unsqueeze_157);  mul_275 = unsqueeze_157 = None
    unsqueeze_158: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_159: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_208: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_281, unsqueeze_159);  mul_281 = unsqueeze_159 = None
    relu_18: "f32[8, 92, 14, 14]" = torch.ops.aten.relu.default(add_208);  add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_14: "f32[8, 184, 14, 14]" = torch.ops.aten.cat.default([relu_17, relu_18], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_78: "f32[8, 184, 14, 14]" = torch.ops.aten.slice.Tensor(cat_14, 0, 0, 9223372036854775807);  cat_14 = None
    slice_79: "f32[8, 184, 14, 14]" = torch.ops.aten.slice.Tensor(slice_78, 2, 0, 9223372036854775807);  slice_78 = None
    slice_80: "f32[8, 184, 14, 14]" = torch.ops.aten.slice.Tensor(slice_79, 3, 0, 9223372036854775807);  slice_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_44: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(slice_80, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_209: "i64[]" = torch.ops.aten.add.Tensor(primals_398, 1)
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 40, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 40, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_210: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_40: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_40: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_81)
    mul_282: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_121: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_283: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_284: "f32[40]" = torch.ops.aten.mul.Tensor(primals_396, 0.9)
    add_211: "f32[40]" = torch.ops.aten.add.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
    squeeze_122: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_285: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0006381620931717);  squeeze_122 = None
    mul_286: "f32[40]" = torch.ops.aten.mul.Tensor(mul_285, 0.1);  mul_285 = None
    mul_287: "f32[40]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
    add_212: "f32[40]" = torch.ops.aten.add.Tensor(mul_286, mul_287);  mul_286 = mul_287 = None
    unsqueeze_160: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
    unsqueeze_161: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_288: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_282, unsqueeze_161);  mul_282 = unsqueeze_161 = None
    unsqueeze_162: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
    unsqueeze_163: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_213: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_288, unsqueeze_163);  mul_288 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_45: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_213, primals_136, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40)
    add_214: "i64[]" = torch.ops.aten.add.Tensor(primals_401, 1)
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 40, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 40, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_215: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_41: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
    sub_41: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_83)
    mul_289: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_124: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_290: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_291: "f32[40]" = torch.ops.aten.mul.Tensor(primals_399, 0.9)
    add_216: "f32[40]" = torch.ops.aten.add.Tensor(mul_290, mul_291);  mul_290 = mul_291 = None
    squeeze_125: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_292: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0006381620931717);  squeeze_125 = None
    mul_293: "f32[40]" = torch.ops.aten.mul.Tensor(mul_292, 0.1);  mul_292 = None
    mul_294: "f32[40]" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
    add_217: "f32[40]" = torch.ops.aten.add.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
    unsqueeze_164: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
    unsqueeze_165: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_295: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_165);  mul_289 = unsqueeze_165 = None
    unsqueeze_166: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
    unsqueeze_167: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_218: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_295, unsqueeze_167);  mul_295 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_15: "f32[8, 80, 14, 14]" = torch.ops.aten.cat.default([add_213, add_218], 1);  add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_81: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(cat_15, 0, 0, 9223372036854775807)
    slice_82: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_81, 2, 0, 9223372036854775807)
    slice_83: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_82, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_219: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(slice_83, slice_77);  slice_83 = None
    slice_scatter_21: "f32[8, 80, 14, 14]" = torch.ops.aten.slice_scatter.default(slice_82, add_219, 3, 0, 9223372036854775807);  slice_82 = add_219 = None
    slice_scatter_22: "f32[8, 80, 14, 14]" = torch.ops.aten.slice_scatter.default(slice_81, slice_scatter_21, 2, 0, 9223372036854775807);  slice_81 = slice_scatter_21 = None
    slice_scatter_23: "f32[8, 80, 14, 14]" = torch.ops.aten.slice_scatter.default(cat_15, slice_scatter_22, 0, 0, 9223372036854775807);  cat_15 = slice_scatter_22 = None
    slice_86: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 0, 0, 9223372036854775807);  slice_scatter_23 = None
    slice_87: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_86, 2, 0, 9223372036854775807);  slice_86 = None
    slice_88: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_87, 3, 0, 9223372036854775807);  slice_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_46: "f32[8, 92, 14, 14]" = torch.ops.aten.convolution.default(slice_88, primals_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_220: "i64[]" = torch.ops.aten.add.Tensor(primals_404, 1)
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 92, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 92, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_221: "f32[1, 92, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_42: "f32[1, 92, 1, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
    sub_42: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_85)
    mul_296: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_127: "f32[92]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_297: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_298: "f32[92]" = torch.ops.aten.mul.Tensor(primals_402, 0.9)
    add_222: "f32[92]" = torch.ops.aten.add.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
    squeeze_128: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_299: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0006381620931717);  squeeze_128 = None
    mul_300: "f32[92]" = torch.ops.aten.mul.Tensor(mul_299, 0.1);  mul_299 = None
    mul_301: "f32[92]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
    add_223: "f32[92]" = torch.ops.aten.add.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    unsqueeze_168: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
    unsqueeze_169: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_302: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_169);  mul_296 = unsqueeze_169 = None
    unsqueeze_170: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
    unsqueeze_171: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_224: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_171);  mul_302 = unsqueeze_171 = None
    relu_19: "f32[8, 92, 14, 14]" = torch.ops.aten.relu.default(add_224);  add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_47: "f32[8, 92, 14, 14]" = torch.ops.aten.convolution.default(relu_19, primals_142, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 92)
    add_225: "i64[]" = torch.ops.aten.add.Tensor(primals_407, 1)
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 92, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 92, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_226: "f32[1, 92, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_43: "f32[1, 92, 1, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
    sub_43: "f32[8, 92, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_87)
    mul_303: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_130: "f32[92]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_304: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_305: "f32[92]" = torch.ops.aten.mul.Tensor(primals_405, 0.9)
    add_227: "f32[92]" = torch.ops.aten.add.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
    squeeze_131: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_306: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0006381620931717);  squeeze_131 = None
    mul_307: "f32[92]" = torch.ops.aten.mul.Tensor(mul_306, 0.1);  mul_306 = None
    mul_308: "f32[92]" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
    add_228: "f32[92]" = torch.ops.aten.add.Tensor(mul_307, mul_308);  mul_307 = mul_308 = None
    unsqueeze_172: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_173: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_309: "f32[8, 92, 14, 14]" = torch.ops.aten.mul.Tensor(mul_303, unsqueeze_173);  mul_303 = unsqueeze_173 = None
    unsqueeze_174: "f32[92, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_175: "f32[92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_229: "f32[8, 92, 14, 14]" = torch.ops.aten.add.Tensor(mul_309, unsqueeze_175);  mul_309 = unsqueeze_175 = None
    relu_20: "f32[8, 92, 14, 14]" = torch.ops.aten.relu.default(add_229);  add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_16: "f32[8, 184, 14, 14]" = torch.ops.aten.cat.default([relu_19, relu_20], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_89: "f32[8, 184, 14, 14]" = torch.ops.aten.slice.Tensor(cat_16, 0, 0, 9223372036854775807);  cat_16 = None
    slice_90: "f32[8, 184, 14, 14]" = torch.ops.aten.slice.Tensor(slice_89, 2, 0, 9223372036854775807);  slice_89 = None
    slice_91: "f32[8, 184, 14, 14]" = torch.ops.aten.slice.Tensor(slice_90, 3, 0, 9223372036854775807);  slice_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_48: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(slice_91, primals_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_230: "i64[]" = torch.ops.aten.add.Tensor(primals_410, 1)
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 40, 1, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 40, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_231: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_44: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_44: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_89)
    mul_310: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_133: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_311: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_312: "f32[40]" = torch.ops.aten.mul.Tensor(primals_408, 0.9)
    add_232: "f32[40]" = torch.ops.aten.add.Tensor(mul_311, mul_312);  mul_311 = mul_312 = None
    squeeze_134: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_313: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0006381620931717);  squeeze_134 = None
    mul_314: "f32[40]" = torch.ops.aten.mul.Tensor(mul_313, 0.1);  mul_313 = None
    mul_315: "f32[40]" = torch.ops.aten.mul.Tensor(primals_409, 0.9)
    add_233: "f32[40]" = torch.ops.aten.add.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
    unsqueeze_176: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1)
    unsqueeze_177: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_316: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_177);  mul_310 = unsqueeze_177 = None
    unsqueeze_178: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
    unsqueeze_179: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_234: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_316, unsqueeze_179);  mul_316 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_49: "f32[8, 40, 14, 14]" = torch.ops.aten.convolution.default(add_234, primals_148, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 40)
    add_235: "i64[]" = torch.ops.aten.add.Tensor(primals_413, 1)
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 40, 1, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 40, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_236: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
    rsqrt_45: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
    sub_45: "f32[8, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_91)
    mul_317: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_136: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_318: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_319: "f32[40]" = torch.ops.aten.mul.Tensor(primals_411, 0.9)
    add_237: "f32[40]" = torch.ops.aten.add.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
    squeeze_137: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_320: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0006381620931717);  squeeze_137 = None
    mul_321: "f32[40]" = torch.ops.aten.mul.Tensor(mul_320, 0.1);  mul_320 = None
    mul_322: "f32[40]" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
    add_238: "f32[40]" = torch.ops.aten.add.Tensor(mul_321, mul_322);  mul_321 = mul_322 = None
    unsqueeze_180: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1)
    unsqueeze_181: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_323: "f32[8, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_317, unsqueeze_181);  mul_317 = unsqueeze_181 = None
    unsqueeze_182: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
    unsqueeze_183: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_239: "f32[8, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_323, unsqueeze_183);  mul_323 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_17: "f32[8, 80, 14, 14]" = torch.ops.aten.cat.default([add_234, add_239], 1);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_92: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(cat_17, 0, 0, 9223372036854775807)
    slice_93: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_92, 2, 0, 9223372036854775807)
    slice_94: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_93, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_240: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(slice_94, slice_88);  slice_94 = None
    slice_scatter_24: "f32[8, 80, 14, 14]" = torch.ops.aten.slice_scatter.default(slice_93, add_240, 3, 0, 9223372036854775807);  slice_93 = add_240 = None
    slice_scatter_25: "f32[8, 80, 14, 14]" = torch.ops.aten.slice_scatter.default(slice_92, slice_scatter_24, 2, 0, 9223372036854775807);  slice_92 = slice_scatter_24 = None
    slice_scatter_26: "f32[8, 80, 14, 14]" = torch.ops.aten.slice_scatter.default(cat_17, slice_scatter_25, 0, 0, 9223372036854775807);  cat_17 = slice_scatter_25 = None
    slice_97: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_26, 0, 0, 9223372036854775807);  slice_scatter_26 = None
    slice_98: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_97, 2, 0, 9223372036854775807);  slice_97 = None
    slice_99: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(slice_98, 3, 0, 9223372036854775807);  slice_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_50: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(slice_99, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_241: "i64[]" = torch.ops.aten.add.Tensor(primals_416, 1)
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[1, 240, 1, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 240, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_242: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
    rsqrt_46: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
    sub_46: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_93)
    mul_324: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
    squeeze_139: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_325: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_326: "f32[240]" = torch.ops.aten.mul.Tensor(primals_414, 0.9)
    add_243: "f32[240]" = torch.ops.aten.add.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
    squeeze_140: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
    mul_327: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0006381620931717);  squeeze_140 = None
    mul_328: "f32[240]" = torch.ops.aten.mul.Tensor(mul_327, 0.1);  mul_327 = None
    mul_329: "f32[240]" = torch.ops.aten.mul.Tensor(primals_415, 0.9)
    add_244: "f32[240]" = torch.ops.aten.add.Tensor(mul_328, mul_329);  mul_328 = mul_329 = None
    unsqueeze_184: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1)
    unsqueeze_185: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_330: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_324, unsqueeze_185);  mul_324 = unsqueeze_185 = None
    unsqueeze_186: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
    unsqueeze_187: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_245: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_330, unsqueeze_187);  mul_330 = unsqueeze_187 = None
    relu_21: "f32[8, 240, 14, 14]" = torch.ops.aten.relu.default(add_245);  add_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_51: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(relu_21, primals_154, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 240)
    add_246: "i64[]" = torch.ops.aten.add.Tensor(primals_419, 1)
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 240, 1, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 240, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_247: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
    rsqrt_47: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
    sub_47: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_95)
    mul_331: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_142: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_332: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_333: "f32[240]" = torch.ops.aten.mul.Tensor(primals_417, 0.9)
    add_248: "f32[240]" = torch.ops.aten.add.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
    squeeze_143: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_334: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0006381620931717);  squeeze_143 = None
    mul_335: "f32[240]" = torch.ops.aten.mul.Tensor(mul_334, 0.1);  mul_334 = None
    mul_336: "f32[240]" = torch.ops.aten.mul.Tensor(primals_418, 0.9)
    add_249: "f32[240]" = torch.ops.aten.add.Tensor(mul_335, mul_336);  mul_335 = mul_336 = None
    unsqueeze_188: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
    unsqueeze_189: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_337: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_189);  mul_331 = unsqueeze_189 = None
    unsqueeze_190: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
    unsqueeze_191: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_250: "f32[8, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_337, unsqueeze_191);  mul_337 = unsqueeze_191 = None
    relu_22: "f32[8, 240, 14, 14]" = torch.ops.aten.relu.default(add_250);  add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_18: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([relu_21, relu_22], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_100: "f32[8, 480, 14, 14]" = torch.ops.aten.slice.Tensor(cat_18, 0, 0, 9223372036854775807)
    slice_101: "f32[8, 480, 14, 14]" = torch.ops.aten.slice.Tensor(slice_100, 2, 0, 9223372036854775807);  slice_100 = None
    slice_102: "f32[8, 480, 14, 14]" = torch.ops.aten.slice.Tensor(slice_101, 3, 0, 9223372036854775807);  slice_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(slice_102, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_52: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_157, primals_158, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_23: "f32[8, 120, 1, 1]" = torch.ops.aten.relu.default(convolution_52);  convolution_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_53: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(relu_23, primals_159, primals_160, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_251: "f32[8, 480, 1, 1]" = torch.ops.aten.add.Tensor(convolution_53, 3)
    clamp_min_2: "f32[8, 480, 1, 1]" = torch.ops.aten.clamp_min.default(add_251, 0);  add_251 = None
    clamp_max_2: "f32[8, 480, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    div_2: "f32[8, 480, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_2, 6);  clamp_max_2 = None
    mul_338: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(slice_102, div_2);  slice_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_54: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(mul_338, primals_161, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_252: "i64[]" = torch.ops.aten.add.Tensor(primals_422, 1)
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 56, 1, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 56, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_253: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_48: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_253);  add_253 = None
    sub_48: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_97)
    mul_339: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_145: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_340: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_341: "f32[56]" = torch.ops.aten.mul.Tensor(primals_420, 0.9)
    add_254: "f32[56]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    squeeze_146: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_342: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0006381620931717);  squeeze_146 = None
    mul_343: "f32[56]" = torch.ops.aten.mul.Tensor(mul_342, 0.1);  mul_342 = None
    mul_344: "f32[56]" = torch.ops.aten.mul.Tensor(primals_421, 0.9)
    add_255: "f32[56]" = torch.ops.aten.add.Tensor(mul_343, mul_344);  mul_343 = mul_344 = None
    unsqueeze_192: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1)
    unsqueeze_193: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_345: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_339, unsqueeze_193);  mul_339 = unsqueeze_193 = None
    unsqueeze_194: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1);  primals_163 = None
    unsqueeze_195: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_256: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_345, unsqueeze_195);  mul_345 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_55: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_256, primals_164, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 56)
    add_257: "i64[]" = torch.ops.aten.add.Tensor(primals_425, 1)
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 56, 1, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 56, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_258: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
    rsqrt_49: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
    sub_49: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_99)
    mul_346: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_148: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_347: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_348: "f32[56]" = torch.ops.aten.mul.Tensor(primals_423, 0.9)
    add_259: "f32[56]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    squeeze_149: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_349: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0006381620931717);  squeeze_149 = None
    mul_350: "f32[56]" = torch.ops.aten.mul.Tensor(mul_349, 0.1);  mul_349 = None
    mul_351: "f32[56]" = torch.ops.aten.mul.Tensor(primals_424, 0.9)
    add_260: "f32[56]" = torch.ops.aten.add.Tensor(mul_350, mul_351);  mul_350 = mul_351 = None
    unsqueeze_196: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_165, -1)
    unsqueeze_197: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_352: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_197);  mul_346 = unsqueeze_197 = None
    unsqueeze_198: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_166, -1);  primals_166 = None
    unsqueeze_199: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_261: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_352, unsqueeze_199);  mul_352 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_19: "f32[8, 112, 14, 14]" = torch.ops.aten.cat.default([add_256, add_261], 1);  add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_103: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(cat_19, 0, 0, 9223372036854775807)
    slice_104: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(slice_103, 2, 0, 9223372036854775807)
    slice_105: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(slice_104, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    convolution_56: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(slice_99, primals_167, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80)
    add_262: "i64[]" = torch.ops.aten.add.Tensor(primals_428, 1)
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 80, 1, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 80, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_263: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
    rsqrt_50: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
    sub_50: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_101)
    mul_353: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_151: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_354: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_355: "f32[80]" = torch.ops.aten.mul.Tensor(primals_426, 0.9)
    add_264: "f32[80]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    squeeze_152: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_356: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0006381620931717);  squeeze_152 = None
    mul_357: "f32[80]" = torch.ops.aten.mul.Tensor(mul_356, 0.1);  mul_356 = None
    mul_358: "f32[80]" = torch.ops.aten.mul.Tensor(primals_427, 0.9)
    add_265: "f32[80]" = torch.ops.aten.add.Tensor(mul_357, mul_358);  mul_357 = mul_358 = None
    unsqueeze_200: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1)
    unsqueeze_201: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_359: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(mul_353, unsqueeze_201);  mul_353 = unsqueeze_201 = None
    unsqueeze_202: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_169, -1);  primals_169 = None
    unsqueeze_203: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_266: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(mul_359, unsqueeze_203);  mul_359 = unsqueeze_203 = None
    convolution_57: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(add_266, primals_170, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_267: "i64[]" = torch.ops.aten.add.Tensor(primals_431, 1)
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 112, 1, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 112, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_268: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_51: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
    sub_51: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_103)
    mul_360: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_154: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_361: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_362: "f32[112]" = torch.ops.aten.mul.Tensor(primals_429, 0.9)
    add_269: "f32[112]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    squeeze_155: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_363: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0006381620931717);  squeeze_155 = None
    mul_364: "f32[112]" = torch.ops.aten.mul.Tensor(mul_363, 0.1);  mul_363 = None
    mul_365: "f32[112]" = torch.ops.aten.mul.Tensor(primals_430, 0.9)
    add_270: "f32[112]" = torch.ops.aten.add.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
    unsqueeze_204: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_171, -1)
    unsqueeze_205: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_366: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_360, unsqueeze_205);  mul_360 = unsqueeze_205 = None
    unsqueeze_206: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_172, -1);  primals_172 = None
    unsqueeze_207: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_271: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_366, unsqueeze_207);  mul_366 = unsqueeze_207 = None
    add_272: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(slice_105, add_271);  slice_105 = add_271 = None
    slice_scatter_27: "f32[8, 112, 14, 14]" = torch.ops.aten.slice_scatter.default(slice_104, add_272, 3, 0, 9223372036854775807);  slice_104 = add_272 = None
    slice_scatter_28: "f32[8, 112, 14, 14]" = torch.ops.aten.slice_scatter.default(slice_103, slice_scatter_27, 2, 0, 9223372036854775807);  slice_103 = slice_scatter_27 = None
    slice_scatter_29: "f32[8, 112, 14, 14]" = torch.ops.aten.slice_scatter.default(cat_19, slice_scatter_28, 0, 0, 9223372036854775807);  cat_19 = slice_scatter_28 = None
    slice_108: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_29, 0, 0, 9223372036854775807);  slice_scatter_29 = None
    slice_109: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(slice_108, 2, 0, 9223372036854775807);  slice_108 = None
    slice_110: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(slice_109, 3, 0, 9223372036854775807);  slice_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_58: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(slice_110, primals_173, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_273: "i64[]" = torch.ops.aten.add.Tensor(primals_434, 1)
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 336, 1, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 336, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_274: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
    rsqrt_52: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_274);  add_274 = None
    sub_52: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_105)
    mul_367: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_157: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_368: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_369: "f32[336]" = torch.ops.aten.mul.Tensor(primals_432, 0.9)
    add_275: "f32[336]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    squeeze_158: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_370: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0006381620931717);  squeeze_158 = None
    mul_371: "f32[336]" = torch.ops.aten.mul.Tensor(mul_370, 0.1);  mul_370 = None
    mul_372: "f32[336]" = torch.ops.aten.mul.Tensor(primals_433, 0.9)
    add_276: "f32[336]" = torch.ops.aten.add.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
    unsqueeze_208: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_174, -1)
    unsqueeze_209: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_373: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_209);  mul_367 = unsqueeze_209 = None
    unsqueeze_210: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_175, -1);  primals_175 = None
    unsqueeze_211: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_277: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_373, unsqueeze_211);  mul_373 = unsqueeze_211 = None
    relu_24: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_277);  add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_59: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(relu_24, primals_176, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 336)
    add_278: "i64[]" = torch.ops.aten.add.Tensor(primals_437, 1)
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 336, 1, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 336, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_279: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_53: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_279);  add_279 = None
    sub_53: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_107)
    mul_374: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_160: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_375: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_376: "f32[336]" = torch.ops.aten.mul.Tensor(primals_435, 0.9)
    add_280: "f32[336]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    squeeze_161: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_377: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0006381620931717);  squeeze_161 = None
    mul_378: "f32[336]" = torch.ops.aten.mul.Tensor(mul_377, 0.1);  mul_377 = None
    mul_379: "f32[336]" = torch.ops.aten.mul.Tensor(primals_436, 0.9)
    add_281: "f32[336]" = torch.ops.aten.add.Tensor(mul_378, mul_379);  mul_378 = mul_379 = None
    unsqueeze_212: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_177, -1)
    unsqueeze_213: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_380: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_374, unsqueeze_213);  mul_374 = unsqueeze_213 = None
    unsqueeze_214: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_178, -1);  primals_178 = None
    unsqueeze_215: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_282: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_380, unsqueeze_215);  mul_380 = unsqueeze_215 = None
    relu_25: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_282);  add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_20: "f32[8, 672, 14, 14]" = torch.ops.aten.cat.default([relu_24, relu_25], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_111: "f32[8, 672, 14, 14]" = torch.ops.aten.slice.Tensor(cat_20, 0, 0, 9223372036854775807)
    slice_112: "f32[8, 672, 14, 14]" = torch.ops.aten.slice.Tensor(slice_111, 2, 0, 9223372036854775807);  slice_111 = None
    slice_113: "f32[8, 672, 14, 14]" = torch.ops.aten.slice.Tensor(slice_112, 3, 0, 9223372036854775807);  slice_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(slice_113, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_60: "f32[8, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_179, primals_180, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_26: "f32[8, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_60);  convolution_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_61: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_26, primals_181, primals_182, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_283: "f32[8, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_61, 3)
    clamp_min_3: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_283, 0);  add_283 = None
    clamp_max_3: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    div_3: "f32[8, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_3, 6);  clamp_max_3 = None
    mul_381: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(slice_113, div_3);  slice_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_62: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(mul_381, primals_183, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_284: "i64[]" = torch.ops.aten.add.Tensor(primals_440, 1)
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 56, 1, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 56, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_285: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
    rsqrt_54: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
    sub_54: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_109)
    mul_382: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_163: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_383: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_384: "f32[56]" = torch.ops.aten.mul.Tensor(primals_438, 0.9)
    add_286: "f32[56]" = torch.ops.aten.add.Tensor(mul_383, mul_384);  mul_383 = mul_384 = None
    squeeze_164: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_385: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0006381620931717);  squeeze_164 = None
    mul_386: "f32[56]" = torch.ops.aten.mul.Tensor(mul_385, 0.1);  mul_385 = None
    mul_387: "f32[56]" = torch.ops.aten.mul.Tensor(primals_439, 0.9)
    add_287: "f32[56]" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
    unsqueeze_216: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_184, -1)
    unsqueeze_217: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_388: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_217);  mul_382 = unsqueeze_217 = None
    unsqueeze_218: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_185, -1);  primals_185 = None
    unsqueeze_219: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_288: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_388, unsqueeze_219);  mul_388 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_63: "f32[8, 56, 14, 14]" = torch.ops.aten.convolution.default(add_288, primals_186, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 56)
    add_289: "i64[]" = torch.ops.aten.add.Tensor(primals_443, 1)
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 56, 1, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 56, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_290: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
    rsqrt_55: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
    sub_55: "f32[8, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_111)
    mul_389: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_166: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_390: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_391: "f32[56]" = torch.ops.aten.mul.Tensor(primals_441, 0.9)
    add_291: "f32[56]" = torch.ops.aten.add.Tensor(mul_390, mul_391);  mul_390 = mul_391 = None
    squeeze_167: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_392: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0006381620931717);  squeeze_167 = None
    mul_393: "f32[56]" = torch.ops.aten.mul.Tensor(mul_392, 0.1);  mul_392 = None
    mul_394: "f32[56]" = torch.ops.aten.mul.Tensor(primals_442, 0.9)
    add_292: "f32[56]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    unsqueeze_220: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_187, -1)
    unsqueeze_221: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_395: "f32[8, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_389, unsqueeze_221);  mul_389 = unsqueeze_221 = None
    unsqueeze_222: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_188, -1);  primals_188 = None
    unsqueeze_223: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_293: "f32[8, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_395, unsqueeze_223);  mul_395 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_21: "f32[8, 112, 14, 14]" = torch.ops.aten.cat.default([add_288, add_293], 1);  add_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_114: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(cat_21, 0, 0, 9223372036854775807)
    slice_115: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(slice_114, 2, 0, 9223372036854775807)
    slice_116: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(slice_115, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_294: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(slice_116, slice_110);  slice_116 = None
    slice_scatter_30: "f32[8, 112, 14, 14]" = torch.ops.aten.slice_scatter.default(slice_115, add_294, 3, 0, 9223372036854775807);  slice_115 = add_294 = None
    slice_scatter_31: "f32[8, 112, 14, 14]" = torch.ops.aten.slice_scatter.default(slice_114, slice_scatter_30, 2, 0, 9223372036854775807);  slice_114 = slice_scatter_30 = None
    slice_scatter_32: "f32[8, 112, 14, 14]" = torch.ops.aten.slice_scatter.default(cat_21, slice_scatter_31, 0, 0, 9223372036854775807);  cat_21 = slice_scatter_31 = None
    slice_119: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_32, 0, 0, 9223372036854775807);  slice_scatter_32 = None
    slice_120: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(slice_119, 2, 0, 9223372036854775807);  slice_119 = None
    slice_121: "f32[8, 112, 14, 14]" = torch.ops.aten.slice.Tensor(slice_120, 3, 0, 9223372036854775807);  slice_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_64: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(slice_121, primals_189, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_295: "i64[]" = torch.ops.aten.add.Tensor(primals_446, 1)
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 336, 1, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 336, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_296: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_56: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
    sub_56: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_113)
    mul_396: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_169: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_397: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_398: "f32[336]" = torch.ops.aten.mul.Tensor(primals_444, 0.9)
    add_297: "f32[336]" = torch.ops.aten.add.Tensor(mul_397, mul_398);  mul_397 = mul_398 = None
    squeeze_170: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_399: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0006381620931717);  squeeze_170 = None
    mul_400: "f32[336]" = torch.ops.aten.mul.Tensor(mul_399, 0.1);  mul_399 = None
    mul_401: "f32[336]" = torch.ops.aten.mul.Tensor(primals_445, 0.9)
    add_298: "f32[336]" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
    unsqueeze_224: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_190, -1)
    unsqueeze_225: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_402: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_396, unsqueeze_225);  mul_396 = unsqueeze_225 = None
    unsqueeze_226: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_191, -1);  primals_191 = None
    unsqueeze_227: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_299: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_402, unsqueeze_227);  mul_402 = unsqueeze_227 = None
    relu_27: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_299);  add_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_65: "f32[8, 336, 14, 14]" = torch.ops.aten.convolution.default(relu_27, primals_192, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 336)
    add_300: "i64[]" = torch.ops.aten.add.Tensor(primals_449, 1)
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 336, 1, 1]" = var_mean_57[0]
    getitem_115: "f32[1, 336, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_301: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
    rsqrt_57: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_301);  add_301 = None
    sub_57: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_115)
    mul_403: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_172: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_404: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_405: "f32[336]" = torch.ops.aten.mul.Tensor(primals_447, 0.9)
    add_302: "f32[336]" = torch.ops.aten.add.Tensor(mul_404, mul_405);  mul_404 = mul_405 = None
    squeeze_173: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_406: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0006381620931717);  squeeze_173 = None
    mul_407: "f32[336]" = torch.ops.aten.mul.Tensor(mul_406, 0.1);  mul_406 = None
    mul_408: "f32[336]" = torch.ops.aten.mul.Tensor(primals_448, 0.9)
    add_303: "f32[336]" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    unsqueeze_228: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_193, -1)
    unsqueeze_229: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_409: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_403, unsqueeze_229);  mul_403 = unsqueeze_229 = None
    unsqueeze_230: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_194, -1);  primals_194 = None
    unsqueeze_231: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_304: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_409, unsqueeze_231);  mul_409 = unsqueeze_231 = None
    relu_28: "f32[8, 336, 14, 14]" = torch.ops.aten.relu.default(add_304);  add_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_22: "f32[8, 672, 14, 14]" = torch.ops.aten.cat.default([relu_27, relu_28], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_122: "f32[8, 672, 14, 14]" = torch.ops.aten.slice.Tensor(cat_22, 0, 0, 9223372036854775807);  cat_22 = None
    slice_123: "f32[8, 672, 14, 14]" = torch.ops.aten.slice.Tensor(slice_122, 2, 0, 9223372036854775807);  slice_122 = None
    slice_124: "f32[8, 672, 14, 14]" = torch.ops.aten.slice.Tensor(slice_123, 3, 0, 9223372036854775807);  slice_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:172, code: x = self.conv_dw(x)
    convolution_66: "f32[8, 672, 7, 7]" = torch.ops.aten.convolution.default(slice_124, primals_195, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 672)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    add_305: "i64[]" = torch.ops.aten.add.Tensor(primals_452, 1)
    var_mean_58 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[1, 672, 1, 1]" = var_mean_58[0]
    getitem_117: "f32[1, 672, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_306: "f32[1, 672, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
    rsqrt_58: "f32[1, 672, 1, 1]" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
    sub_58: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_117)
    mul_410: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    squeeze_174: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
    squeeze_175: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_411: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_412: "f32[672]" = torch.ops.aten.mul.Tensor(primals_450, 0.9)
    add_307: "f32[672]" = torch.ops.aten.add.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
    squeeze_176: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
    mul_413: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0025575447570332);  squeeze_176 = None
    mul_414: "f32[672]" = torch.ops.aten.mul.Tensor(mul_413, 0.1);  mul_413 = None
    mul_415: "f32[672]" = torch.ops.aten.mul.Tensor(primals_451, 0.9)
    add_308: "f32[672]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    unsqueeze_232: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_196, -1)
    unsqueeze_233: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_416: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_410, unsqueeze_233);  mul_410 = unsqueeze_233 = None
    unsqueeze_234: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_197, -1);  primals_197 = None
    unsqueeze_235: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_309: "f32[8, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_416, unsqueeze_235);  mul_416 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 672, 1, 1]" = torch.ops.aten.mean.dim(add_309, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_67: "f32[8, 168, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_198, primals_199, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_29: "f32[8, 168, 1, 1]" = torch.ops.aten.relu.default(convolution_67);  convolution_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_68: "f32[8, 672, 1, 1]" = torch.ops.aten.convolution.default(relu_29, primals_200, primals_201, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_310: "f32[8, 672, 1, 1]" = torch.ops.aten.add.Tensor(convolution_68, 3)
    clamp_min_4: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_min.default(add_310, 0);  add_310 = None
    clamp_max_4: "f32[8, 672, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
    div_4: "f32[8, 672, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_4, 6);  clamp_max_4 = None
    mul_417: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_309, div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_69: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(mul_417, primals_202, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_311: "i64[]" = torch.ops.aten.add.Tensor(primals_455, 1)
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 80, 1, 1]" = var_mean_59[0]
    getitem_119: "f32[1, 80, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_312: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_59: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_312);  add_312 = None
    sub_59: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_119)
    mul_418: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    squeeze_177: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_178: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    mul_419: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_420: "f32[80]" = torch.ops.aten.mul.Tensor(primals_453, 0.9)
    add_313: "f32[80]" = torch.ops.aten.add.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
    squeeze_179: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_421: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0025575447570332);  squeeze_179 = None
    mul_422: "f32[80]" = torch.ops.aten.mul.Tensor(mul_421, 0.1);  mul_421 = None
    mul_423: "f32[80]" = torch.ops.aten.mul.Tensor(primals_454, 0.9)
    add_314: "f32[80]" = torch.ops.aten.add.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
    unsqueeze_236: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1)
    unsqueeze_237: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_424: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_418, unsqueeze_237);  mul_418 = unsqueeze_237 = None
    unsqueeze_238: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
    unsqueeze_239: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_315: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_424, unsqueeze_239);  mul_424 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_70: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_315, primals_205, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80)
    add_316: "i64[]" = torch.ops.aten.add.Tensor(primals_458, 1)
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 80, 1, 1]" = var_mean_60[0]
    getitem_121: "f32[1, 80, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_317: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
    rsqrt_60: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
    sub_60: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_121)
    mul_425: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    squeeze_180: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_181: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    mul_426: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_427: "f32[80]" = torch.ops.aten.mul.Tensor(primals_456, 0.9)
    add_318: "f32[80]" = torch.ops.aten.add.Tensor(mul_426, mul_427);  mul_426 = mul_427 = None
    squeeze_182: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_428: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0025575447570332);  squeeze_182 = None
    mul_429: "f32[80]" = torch.ops.aten.mul.Tensor(mul_428, 0.1);  mul_428 = None
    mul_430: "f32[80]" = torch.ops.aten.mul.Tensor(primals_457, 0.9)
    add_319: "f32[80]" = torch.ops.aten.add.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
    unsqueeze_240: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_206, -1)
    unsqueeze_241: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_431: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_425, unsqueeze_241);  mul_425 = unsqueeze_241 = None
    unsqueeze_242: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_207, -1);  primals_207 = None
    unsqueeze_243: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_320: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_431, unsqueeze_243);  mul_431 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_23: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_315, add_320], 1);  add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_125: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(cat_23, 0, 0, 9223372036854775807)
    slice_126: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_125, 2, 0, 9223372036854775807)
    slice_127: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_126, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    convolution_71: "f32[8, 112, 7, 7]" = torch.ops.aten.convolution.default(slice_121, primals_208, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 112)
    add_321: "i64[]" = torch.ops.aten.add.Tensor(primals_461, 1)
    var_mean_61 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[1, 112, 1, 1]" = var_mean_61[0]
    getitem_123: "f32[1, 112, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_322: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
    rsqrt_61: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_322);  add_322 = None
    sub_61: "f32[8, 112, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_123)
    mul_432: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    squeeze_183: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    squeeze_184: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_433: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_434: "f32[112]" = torch.ops.aten.mul.Tensor(primals_459, 0.9)
    add_323: "f32[112]" = torch.ops.aten.add.Tensor(mul_433, mul_434);  mul_433 = mul_434 = None
    squeeze_185: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
    mul_435: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0025575447570332);  squeeze_185 = None
    mul_436: "f32[112]" = torch.ops.aten.mul.Tensor(mul_435, 0.1);  mul_435 = None
    mul_437: "f32[112]" = torch.ops.aten.mul.Tensor(primals_460, 0.9)
    add_324: "f32[112]" = torch.ops.aten.add.Tensor(mul_436, mul_437);  mul_436 = mul_437 = None
    unsqueeze_244: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_209, -1)
    unsqueeze_245: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_438: "f32[8, 112, 7, 7]" = torch.ops.aten.mul.Tensor(mul_432, unsqueeze_245);  mul_432 = unsqueeze_245 = None
    unsqueeze_246: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_210, -1);  primals_210 = None
    unsqueeze_247: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_325: "f32[8, 112, 7, 7]" = torch.ops.aten.add.Tensor(mul_438, unsqueeze_247);  mul_438 = unsqueeze_247 = None
    convolution_72: "f32[8, 160, 7, 7]" = torch.ops.aten.convolution.default(add_325, primals_211, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_326: "i64[]" = torch.ops.aten.add.Tensor(primals_464, 1)
    var_mean_62 = torch.ops.aten.var_mean.correction(convolution_72, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 160, 1, 1]" = var_mean_62[0]
    getitem_125: "f32[1, 160, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_327: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
    rsqrt_62: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_327);  add_327 = None
    sub_62: "f32[8, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_72, getitem_125)
    mul_439: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
    squeeze_186: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_187: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
    mul_440: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_441: "f32[160]" = torch.ops.aten.mul.Tensor(primals_462, 0.9)
    add_328: "f32[160]" = torch.ops.aten.add.Tensor(mul_440, mul_441);  mul_440 = mul_441 = None
    squeeze_188: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_442: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0025575447570332);  squeeze_188 = None
    mul_443: "f32[160]" = torch.ops.aten.mul.Tensor(mul_442, 0.1);  mul_442 = None
    mul_444: "f32[160]" = torch.ops.aten.mul.Tensor(primals_463, 0.9)
    add_329: "f32[160]" = torch.ops.aten.add.Tensor(mul_443, mul_444);  mul_443 = mul_444 = None
    unsqueeze_248: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_212, -1)
    unsqueeze_249: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    mul_445: "f32[8, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_439, unsqueeze_249);  mul_439 = unsqueeze_249 = None
    unsqueeze_250: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_213, -1);  primals_213 = None
    unsqueeze_251: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    add_330: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_445, unsqueeze_251);  mul_445 = unsqueeze_251 = None
    add_331: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(slice_127, add_330);  slice_127 = add_330 = None
    slice_scatter_33: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(slice_126, add_331, 3, 0, 9223372036854775807);  slice_126 = add_331 = None
    slice_scatter_34: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(slice_125, slice_scatter_33, 2, 0, 9223372036854775807);  slice_125 = slice_scatter_33 = None
    slice_scatter_35: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(cat_23, slice_scatter_34, 0, 0, 9223372036854775807);  cat_23 = slice_scatter_34 = None
    slice_130: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_35, 0, 0, 9223372036854775807);  slice_scatter_35 = None
    slice_131: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_130, 2, 0, 9223372036854775807);  slice_130 = None
    slice_132: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_131, 3, 0, 9223372036854775807);  slice_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_73: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(slice_132, primals_214, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_332: "i64[]" = torch.ops.aten.add.Tensor(primals_467, 1)
    var_mean_63 = torch.ops.aten.var_mean.correction(convolution_73, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 480, 1, 1]" = var_mean_63[0]
    getitem_127: "f32[1, 480, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_333: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
    rsqrt_63: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_333);  add_333 = None
    sub_63: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_73, getitem_127)
    mul_446: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
    squeeze_189: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_190: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
    mul_447: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_448: "f32[480]" = torch.ops.aten.mul.Tensor(primals_465, 0.9)
    add_334: "f32[480]" = torch.ops.aten.add.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
    squeeze_191: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_449: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0025575447570332);  squeeze_191 = None
    mul_450: "f32[480]" = torch.ops.aten.mul.Tensor(mul_449, 0.1);  mul_449 = None
    mul_451: "f32[480]" = torch.ops.aten.mul.Tensor(primals_466, 0.9)
    add_335: "f32[480]" = torch.ops.aten.add.Tensor(mul_450, mul_451);  mul_450 = mul_451 = None
    unsqueeze_252: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_215, -1)
    unsqueeze_253: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_452: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_446, unsqueeze_253);  mul_446 = unsqueeze_253 = None
    unsqueeze_254: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_216, -1);  primals_216 = None
    unsqueeze_255: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_336: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_452, unsqueeze_255);  mul_452 = unsqueeze_255 = None
    relu_30: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_336);  add_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_74: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(relu_30, primals_217, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    add_337: "i64[]" = torch.ops.aten.add.Tensor(primals_470, 1)
    var_mean_64 = torch.ops.aten.var_mean.correction(convolution_74, [0, 2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[1, 480, 1, 1]" = var_mean_64[0]
    getitem_129: "f32[1, 480, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_338: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
    rsqrt_64: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_338);  add_338 = None
    sub_64: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, getitem_129)
    mul_453: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
    squeeze_192: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
    squeeze_193: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
    mul_454: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
    mul_455: "f32[480]" = torch.ops.aten.mul.Tensor(primals_468, 0.9)
    add_339: "f32[480]" = torch.ops.aten.add.Tensor(mul_454, mul_455);  mul_454 = mul_455 = None
    squeeze_194: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
    mul_456: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0025575447570332);  squeeze_194 = None
    mul_457: "f32[480]" = torch.ops.aten.mul.Tensor(mul_456, 0.1);  mul_456 = None
    mul_458: "f32[480]" = torch.ops.aten.mul.Tensor(primals_469, 0.9)
    add_340: "f32[480]" = torch.ops.aten.add.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
    unsqueeze_256: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_218, -1)
    unsqueeze_257: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    mul_459: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_453, unsqueeze_257);  mul_453 = unsqueeze_257 = None
    unsqueeze_258: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_219, -1);  primals_219 = None
    unsqueeze_259: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    add_341: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_459, unsqueeze_259);  mul_459 = unsqueeze_259 = None
    relu_31: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_341);  add_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_24: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([relu_30, relu_31], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_133: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(cat_24, 0, 0, 9223372036854775807);  cat_24 = None
    slice_134: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(slice_133, 2, 0, 9223372036854775807);  slice_133 = None
    slice_135: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(slice_134, 3, 0, 9223372036854775807);  slice_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_75: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(slice_135, primals_220, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_342: "i64[]" = torch.ops.aten.add.Tensor(primals_473, 1)
    var_mean_65 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
    getitem_130: "f32[1, 80, 1, 1]" = var_mean_65[0]
    getitem_131: "f32[1, 80, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_343: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05)
    rsqrt_65: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_343);  add_343 = None
    sub_65: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_131)
    mul_460: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
    squeeze_195: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
    squeeze_196: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
    mul_461: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
    mul_462: "f32[80]" = torch.ops.aten.mul.Tensor(primals_471, 0.9)
    add_344: "f32[80]" = torch.ops.aten.add.Tensor(mul_461, mul_462);  mul_461 = mul_462 = None
    squeeze_197: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
    mul_463: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0025575447570332);  squeeze_197 = None
    mul_464: "f32[80]" = torch.ops.aten.mul.Tensor(mul_463, 0.1);  mul_463 = None
    mul_465: "f32[80]" = torch.ops.aten.mul.Tensor(primals_472, 0.9)
    add_345: "f32[80]" = torch.ops.aten.add.Tensor(mul_464, mul_465);  mul_464 = mul_465 = None
    unsqueeze_260: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_221, -1)
    unsqueeze_261: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_466: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_460, unsqueeze_261);  mul_460 = unsqueeze_261 = None
    unsqueeze_262: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_222, -1);  primals_222 = None
    unsqueeze_263: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_346: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_466, unsqueeze_263);  mul_466 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_76: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_346, primals_223, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80)
    add_347: "i64[]" = torch.ops.aten.add.Tensor(primals_476, 1)
    var_mean_66 = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
    getitem_132: "f32[1, 80, 1, 1]" = var_mean_66[0]
    getitem_133: "f32[1, 80, 1, 1]" = var_mean_66[1];  var_mean_66 = None
    add_348: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05)
    rsqrt_66: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_348);  add_348 = None
    sub_66: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_133)
    mul_467: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
    squeeze_198: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
    squeeze_199: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
    mul_468: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
    mul_469: "f32[80]" = torch.ops.aten.mul.Tensor(primals_474, 0.9)
    add_349: "f32[80]" = torch.ops.aten.add.Tensor(mul_468, mul_469);  mul_468 = mul_469 = None
    squeeze_200: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
    mul_470: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0025575447570332);  squeeze_200 = None
    mul_471: "f32[80]" = torch.ops.aten.mul.Tensor(mul_470, 0.1);  mul_470 = None
    mul_472: "f32[80]" = torch.ops.aten.mul.Tensor(primals_475, 0.9)
    add_350: "f32[80]" = torch.ops.aten.add.Tensor(mul_471, mul_472);  mul_471 = mul_472 = None
    unsqueeze_264: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_224, -1)
    unsqueeze_265: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    mul_473: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_467, unsqueeze_265);  mul_467 = unsqueeze_265 = None
    unsqueeze_266: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_225, -1);  primals_225 = None
    unsqueeze_267: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    add_351: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_473, unsqueeze_267);  mul_473 = unsqueeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_25: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_346, add_351], 1);  add_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_136: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(cat_25, 0, 0, 9223372036854775807)
    slice_137: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_136, 2, 0, 9223372036854775807)
    slice_138: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_137, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_352: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(slice_138, slice_132);  slice_138 = None
    slice_scatter_36: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(slice_137, add_352, 3, 0, 9223372036854775807);  slice_137 = add_352 = None
    slice_scatter_37: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(slice_136, slice_scatter_36, 2, 0, 9223372036854775807);  slice_136 = slice_scatter_36 = None
    slice_scatter_38: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(cat_25, slice_scatter_37, 0, 0, 9223372036854775807);  cat_25 = slice_scatter_37 = None
    slice_141: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_38, 0, 0, 9223372036854775807);  slice_scatter_38 = None
    slice_142: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_141, 2, 0, 9223372036854775807);  slice_141 = None
    slice_143: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_142, 3, 0, 9223372036854775807);  slice_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_77: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(slice_143, primals_226, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_353: "i64[]" = torch.ops.aten.add.Tensor(primals_479, 1)
    var_mean_67 = torch.ops.aten.var_mean.correction(convolution_77, [0, 2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[1, 480, 1, 1]" = var_mean_67[0]
    getitem_135: "f32[1, 480, 1, 1]" = var_mean_67[1];  var_mean_67 = None
    add_354: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05)
    rsqrt_67: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_354);  add_354 = None
    sub_67: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_77, getitem_135)
    mul_474: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
    squeeze_201: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
    squeeze_202: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
    mul_475: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
    mul_476: "f32[480]" = torch.ops.aten.mul.Tensor(primals_477, 0.9)
    add_355: "f32[480]" = torch.ops.aten.add.Tensor(mul_475, mul_476);  mul_475 = mul_476 = None
    squeeze_203: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
    mul_477: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_203, 1.0025575447570332);  squeeze_203 = None
    mul_478: "f32[480]" = torch.ops.aten.mul.Tensor(mul_477, 0.1);  mul_477 = None
    mul_479: "f32[480]" = torch.ops.aten.mul.Tensor(primals_478, 0.9)
    add_356: "f32[480]" = torch.ops.aten.add.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
    unsqueeze_268: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_227, -1)
    unsqueeze_269: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_480: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_474, unsqueeze_269);  mul_474 = unsqueeze_269 = None
    unsqueeze_270: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1);  primals_228 = None
    unsqueeze_271: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_357: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_480, unsqueeze_271);  mul_480 = unsqueeze_271 = None
    relu_32: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_357);  add_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_78: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(relu_32, primals_229, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    add_358: "i64[]" = torch.ops.aten.add.Tensor(primals_482, 1)
    var_mean_68 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
    getitem_136: "f32[1, 480, 1, 1]" = var_mean_68[0]
    getitem_137: "f32[1, 480, 1, 1]" = var_mean_68[1];  var_mean_68 = None
    add_359: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05)
    rsqrt_68: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_359);  add_359 = None
    sub_68: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_137)
    mul_481: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
    squeeze_204: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
    squeeze_205: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
    mul_482: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1)
    mul_483: "f32[480]" = torch.ops.aten.mul.Tensor(primals_480, 0.9)
    add_360: "f32[480]" = torch.ops.aten.add.Tensor(mul_482, mul_483);  mul_482 = mul_483 = None
    squeeze_206: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
    mul_484: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_206, 1.0025575447570332);  squeeze_206 = None
    mul_485: "f32[480]" = torch.ops.aten.mul.Tensor(mul_484, 0.1);  mul_484 = None
    mul_486: "f32[480]" = torch.ops.aten.mul.Tensor(primals_481, 0.9)
    add_361: "f32[480]" = torch.ops.aten.add.Tensor(mul_485, mul_486);  mul_485 = mul_486 = None
    unsqueeze_272: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_230, -1)
    unsqueeze_273: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    mul_487: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_481, unsqueeze_273);  mul_481 = unsqueeze_273 = None
    unsqueeze_274: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_231, -1);  primals_231 = None
    unsqueeze_275: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    add_362: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_487, unsqueeze_275);  mul_487 = unsqueeze_275 = None
    relu_33: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_362);  add_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_26: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([relu_32, relu_33], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_144: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(cat_26, 0, 0, 9223372036854775807)
    slice_145: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(slice_144, 2, 0, 9223372036854775807);  slice_144 = None
    slice_146: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(slice_145, 3, 0, 9223372036854775807);  slice_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(slice_146, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_79: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_232, primals_233, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_34: "f32[8, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_79);  convolution_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_80: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_34, primals_234, primals_235, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_363: "f32[8, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_80, 3)
    clamp_min_5: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_363, 0);  add_363 = None
    clamp_max_5: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
    div_5: "f32[8, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_5, 6);  clamp_max_5 = None
    mul_488: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(slice_146, div_5);  slice_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_81: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(mul_488, primals_236, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_364: "i64[]" = torch.ops.aten.add.Tensor(primals_485, 1)
    var_mean_69 = torch.ops.aten.var_mean.correction(convolution_81, [0, 2, 3], correction = 0, keepdim = True)
    getitem_138: "f32[1, 80, 1, 1]" = var_mean_69[0]
    getitem_139: "f32[1, 80, 1, 1]" = var_mean_69[1];  var_mean_69 = None
    add_365: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05)
    rsqrt_69: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_365);  add_365 = None
    sub_69: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_81, getitem_139)
    mul_489: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
    squeeze_207: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
    squeeze_208: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
    mul_490: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1)
    mul_491: "f32[80]" = torch.ops.aten.mul.Tensor(primals_483, 0.9)
    add_366: "f32[80]" = torch.ops.aten.add.Tensor(mul_490, mul_491);  mul_490 = mul_491 = None
    squeeze_209: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
    mul_492: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_209, 1.0025575447570332);  squeeze_209 = None
    mul_493: "f32[80]" = torch.ops.aten.mul.Tensor(mul_492, 0.1);  mul_492 = None
    mul_494: "f32[80]" = torch.ops.aten.mul.Tensor(primals_484, 0.9)
    add_367: "f32[80]" = torch.ops.aten.add.Tensor(mul_493, mul_494);  mul_493 = mul_494 = None
    unsqueeze_276: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_237, -1)
    unsqueeze_277: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_495: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_489, unsqueeze_277);  mul_489 = unsqueeze_277 = None
    unsqueeze_278: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_238, -1);  primals_238 = None
    unsqueeze_279: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_368: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_495, unsqueeze_279);  mul_495 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_82: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_368, primals_239, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80)
    add_369: "i64[]" = torch.ops.aten.add.Tensor(primals_488, 1)
    var_mean_70 = torch.ops.aten.var_mean.correction(convolution_82, [0, 2, 3], correction = 0, keepdim = True)
    getitem_140: "f32[1, 80, 1, 1]" = var_mean_70[0]
    getitem_141: "f32[1, 80, 1, 1]" = var_mean_70[1];  var_mean_70 = None
    add_370: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05)
    rsqrt_70: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_370);  add_370 = None
    sub_70: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_82, getitem_141)
    mul_496: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
    squeeze_210: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
    squeeze_211: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
    mul_497: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
    mul_498: "f32[80]" = torch.ops.aten.mul.Tensor(primals_486, 0.9)
    add_371: "f32[80]" = torch.ops.aten.add.Tensor(mul_497, mul_498);  mul_497 = mul_498 = None
    squeeze_212: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
    mul_499: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_212, 1.0025575447570332);  squeeze_212 = None
    mul_500: "f32[80]" = torch.ops.aten.mul.Tensor(mul_499, 0.1);  mul_499 = None
    mul_501: "f32[80]" = torch.ops.aten.mul.Tensor(primals_487, 0.9)
    add_372: "f32[80]" = torch.ops.aten.add.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
    unsqueeze_280: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_240, -1)
    unsqueeze_281: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    mul_502: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_496, unsqueeze_281);  mul_496 = unsqueeze_281 = None
    unsqueeze_282: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_241, -1);  primals_241 = None
    unsqueeze_283: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    add_373: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_502, unsqueeze_283);  mul_502 = unsqueeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_27: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_368, add_373], 1);  add_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_147: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(cat_27, 0, 0, 9223372036854775807)
    slice_148: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_147, 2, 0, 9223372036854775807)
    slice_149: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_148, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_374: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(slice_149, slice_143);  slice_149 = None
    slice_scatter_39: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(slice_148, add_374, 3, 0, 9223372036854775807);  slice_148 = add_374 = None
    slice_scatter_40: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(slice_147, slice_scatter_39, 2, 0, 9223372036854775807);  slice_147 = slice_scatter_39 = None
    slice_scatter_41: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(cat_27, slice_scatter_40, 0, 0, 9223372036854775807);  cat_27 = slice_scatter_40 = None
    slice_152: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_41, 0, 0, 9223372036854775807);  slice_scatter_41 = None
    slice_153: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_152, 2, 0, 9223372036854775807);  slice_152 = None
    slice_154: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_153, 3, 0, 9223372036854775807);  slice_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_83: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(slice_154, primals_242, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_375: "i64[]" = torch.ops.aten.add.Tensor(primals_491, 1)
    var_mean_71 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
    getitem_142: "f32[1, 480, 1, 1]" = var_mean_71[0]
    getitem_143: "f32[1, 480, 1, 1]" = var_mean_71[1];  var_mean_71 = None
    add_376: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05)
    rsqrt_71: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
    sub_71: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_143)
    mul_503: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
    squeeze_213: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
    squeeze_214: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
    mul_504: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1)
    mul_505: "f32[480]" = torch.ops.aten.mul.Tensor(primals_489, 0.9)
    add_377: "f32[480]" = torch.ops.aten.add.Tensor(mul_504, mul_505);  mul_504 = mul_505 = None
    squeeze_215: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
    mul_506: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_215, 1.0025575447570332);  squeeze_215 = None
    mul_507: "f32[480]" = torch.ops.aten.mul.Tensor(mul_506, 0.1);  mul_506 = None
    mul_508: "f32[480]" = torch.ops.aten.mul.Tensor(primals_490, 0.9)
    add_378: "f32[480]" = torch.ops.aten.add.Tensor(mul_507, mul_508);  mul_507 = mul_508 = None
    unsqueeze_284: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_243, -1)
    unsqueeze_285: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_509: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_503, unsqueeze_285);  mul_503 = unsqueeze_285 = None
    unsqueeze_286: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_244, -1);  primals_244 = None
    unsqueeze_287: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_379: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_509, unsqueeze_287);  mul_509 = unsqueeze_287 = None
    relu_35: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_379);  add_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_84: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(relu_35, primals_245, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    add_380: "i64[]" = torch.ops.aten.add.Tensor(primals_494, 1)
    var_mean_72 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
    getitem_144: "f32[1, 480, 1, 1]" = var_mean_72[0]
    getitem_145: "f32[1, 480, 1, 1]" = var_mean_72[1];  var_mean_72 = None
    add_381: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05)
    rsqrt_72: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_381);  add_381 = None
    sub_72: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_145)
    mul_510: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
    squeeze_216: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
    squeeze_217: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
    mul_511: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1)
    mul_512: "f32[480]" = torch.ops.aten.mul.Tensor(primals_492, 0.9)
    add_382: "f32[480]" = torch.ops.aten.add.Tensor(mul_511, mul_512);  mul_511 = mul_512 = None
    squeeze_218: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
    mul_513: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_218, 1.0025575447570332);  squeeze_218 = None
    mul_514: "f32[480]" = torch.ops.aten.mul.Tensor(mul_513, 0.1);  mul_513 = None
    mul_515: "f32[480]" = torch.ops.aten.mul.Tensor(primals_493, 0.9)
    add_383: "f32[480]" = torch.ops.aten.add.Tensor(mul_514, mul_515);  mul_514 = mul_515 = None
    unsqueeze_288: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_246, -1)
    unsqueeze_289: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    mul_516: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_510, unsqueeze_289);  mul_510 = unsqueeze_289 = None
    unsqueeze_290: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_247, -1);  primals_247 = None
    unsqueeze_291: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    add_384: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_516, unsqueeze_291);  mul_516 = unsqueeze_291 = None
    relu_36: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_384);  add_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_28: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([relu_35, relu_36], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_155: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(cat_28, 0, 0, 9223372036854775807);  cat_28 = None
    slice_156: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(slice_155, 2, 0, 9223372036854775807);  slice_155 = None
    slice_157: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(slice_156, 3, 0, 9223372036854775807);  slice_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_85: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(slice_157, primals_248, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_385: "i64[]" = torch.ops.aten.add.Tensor(primals_497, 1)
    var_mean_73 = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True)
    getitem_146: "f32[1, 80, 1, 1]" = var_mean_73[0]
    getitem_147: "f32[1, 80, 1, 1]" = var_mean_73[1];  var_mean_73 = None
    add_386: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05)
    rsqrt_73: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_386);  add_386 = None
    sub_73: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_147)
    mul_517: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
    squeeze_219: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
    squeeze_220: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
    mul_518: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
    mul_519: "f32[80]" = torch.ops.aten.mul.Tensor(primals_495, 0.9)
    add_387: "f32[80]" = torch.ops.aten.add.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
    squeeze_221: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
    mul_520: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_221, 1.0025575447570332);  squeeze_221 = None
    mul_521: "f32[80]" = torch.ops.aten.mul.Tensor(mul_520, 0.1);  mul_520 = None
    mul_522: "f32[80]" = torch.ops.aten.mul.Tensor(primals_496, 0.9)
    add_388: "f32[80]" = torch.ops.aten.add.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
    unsqueeze_292: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_249, -1)
    unsqueeze_293: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_523: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_517, unsqueeze_293);  mul_517 = unsqueeze_293 = None
    unsqueeze_294: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_250, -1);  primals_250 = None
    unsqueeze_295: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_389: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_523, unsqueeze_295);  mul_523 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_86: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_389, primals_251, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80)
    add_390: "i64[]" = torch.ops.aten.add.Tensor(primals_500, 1)
    var_mean_74 = torch.ops.aten.var_mean.correction(convolution_86, [0, 2, 3], correction = 0, keepdim = True)
    getitem_148: "f32[1, 80, 1, 1]" = var_mean_74[0]
    getitem_149: "f32[1, 80, 1, 1]" = var_mean_74[1];  var_mean_74 = None
    add_391: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05)
    rsqrt_74: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_391);  add_391 = None
    sub_74: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_86, getitem_149)
    mul_524: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
    squeeze_222: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
    squeeze_223: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
    mul_525: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
    mul_526: "f32[80]" = torch.ops.aten.mul.Tensor(primals_498, 0.9)
    add_392: "f32[80]" = torch.ops.aten.add.Tensor(mul_525, mul_526);  mul_525 = mul_526 = None
    squeeze_224: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
    mul_527: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_224, 1.0025575447570332);  squeeze_224 = None
    mul_528: "f32[80]" = torch.ops.aten.mul.Tensor(mul_527, 0.1);  mul_527 = None
    mul_529: "f32[80]" = torch.ops.aten.mul.Tensor(primals_499, 0.9)
    add_393: "f32[80]" = torch.ops.aten.add.Tensor(mul_528, mul_529);  mul_528 = mul_529 = None
    unsqueeze_296: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_252, -1)
    unsqueeze_297: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    mul_530: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_524, unsqueeze_297);  mul_524 = unsqueeze_297 = None
    unsqueeze_298: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_253, -1);  primals_253 = None
    unsqueeze_299: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    add_394: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_530, unsqueeze_299);  mul_530 = unsqueeze_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_29: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_389, add_394], 1);  add_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_158: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(cat_29, 0, 0, 9223372036854775807)
    slice_159: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_158, 2, 0, 9223372036854775807)
    slice_160: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_159, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_395: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(slice_160, slice_154);  slice_160 = None
    slice_scatter_42: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(slice_159, add_395, 3, 0, 9223372036854775807);  slice_159 = add_395 = None
    slice_scatter_43: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(slice_158, slice_scatter_42, 2, 0, 9223372036854775807);  slice_158 = slice_scatter_42 = None
    slice_scatter_44: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(cat_29, slice_scatter_43, 0, 0, 9223372036854775807);  cat_29 = slice_scatter_43 = None
    slice_163: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_44, 0, 0, 9223372036854775807);  slice_scatter_44 = None
    slice_164: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_163, 2, 0, 9223372036854775807);  slice_163 = None
    slice_165: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_164, 3, 0, 9223372036854775807);  slice_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_87: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(slice_165, primals_254, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_396: "i64[]" = torch.ops.aten.add.Tensor(primals_503, 1)
    var_mean_75 = torch.ops.aten.var_mean.correction(convolution_87, [0, 2, 3], correction = 0, keepdim = True)
    getitem_150: "f32[1, 480, 1, 1]" = var_mean_75[0]
    getitem_151: "f32[1, 480, 1, 1]" = var_mean_75[1];  var_mean_75 = None
    add_397: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
    rsqrt_75: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_397);  add_397 = None
    sub_75: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_87, getitem_151)
    mul_531: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
    squeeze_225: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
    squeeze_226: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
    mul_532: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1)
    mul_533: "f32[480]" = torch.ops.aten.mul.Tensor(primals_501, 0.9)
    add_398: "f32[480]" = torch.ops.aten.add.Tensor(mul_532, mul_533);  mul_532 = mul_533 = None
    squeeze_227: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
    mul_534: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_227, 1.0025575447570332);  squeeze_227 = None
    mul_535: "f32[480]" = torch.ops.aten.mul.Tensor(mul_534, 0.1);  mul_534 = None
    mul_536: "f32[480]" = torch.ops.aten.mul.Tensor(primals_502, 0.9)
    add_399: "f32[480]" = torch.ops.aten.add.Tensor(mul_535, mul_536);  mul_535 = mul_536 = None
    unsqueeze_300: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_255, -1)
    unsqueeze_301: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_537: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_531, unsqueeze_301);  mul_531 = unsqueeze_301 = None
    unsqueeze_302: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_256, -1);  primals_256 = None
    unsqueeze_303: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_400: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_537, unsqueeze_303);  mul_537 = unsqueeze_303 = None
    relu_37: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_400);  add_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_88: "f32[8, 480, 7, 7]" = torch.ops.aten.convolution.default(relu_37, primals_257, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 480)
    add_401: "i64[]" = torch.ops.aten.add.Tensor(primals_506, 1)
    var_mean_76 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
    getitem_152: "f32[1, 480, 1, 1]" = var_mean_76[0]
    getitem_153: "f32[1, 480, 1, 1]" = var_mean_76[1];  var_mean_76 = None
    add_402: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05)
    rsqrt_76: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_402);  add_402 = None
    sub_76: "f32[8, 480, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_153)
    mul_538: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
    squeeze_228: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
    squeeze_229: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
    mul_539: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1)
    mul_540: "f32[480]" = torch.ops.aten.mul.Tensor(primals_504, 0.9)
    add_403: "f32[480]" = torch.ops.aten.add.Tensor(mul_539, mul_540);  mul_539 = mul_540 = None
    squeeze_230: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
    mul_541: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_230, 1.0025575447570332);  squeeze_230 = None
    mul_542: "f32[480]" = torch.ops.aten.mul.Tensor(mul_541, 0.1);  mul_541 = None
    mul_543: "f32[480]" = torch.ops.aten.mul.Tensor(primals_505, 0.9)
    add_404: "f32[480]" = torch.ops.aten.add.Tensor(mul_542, mul_543);  mul_542 = mul_543 = None
    unsqueeze_304: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_258, -1)
    unsqueeze_305: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    mul_544: "f32[8, 480, 7, 7]" = torch.ops.aten.mul.Tensor(mul_538, unsqueeze_305);  mul_538 = unsqueeze_305 = None
    unsqueeze_306: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_259, -1);  primals_259 = None
    unsqueeze_307: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    add_405: "f32[8, 480, 7, 7]" = torch.ops.aten.add.Tensor(mul_544, unsqueeze_307);  mul_544 = unsqueeze_307 = None
    relu_38: "f32[8, 480, 7, 7]" = torch.ops.aten.relu.default(add_405);  add_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_30: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([relu_37, relu_38], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_166: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(cat_30, 0, 0, 9223372036854775807)
    slice_167: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(slice_166, 2, 0, 9223372036854775807);  slice_166 = None
    slice_168: "f32[8, 960, 7, 7]" = torch.ops.aten.slice.Tensor(slice_167, 3, 0, 9223372036854775807);  slice_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(slice_168, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_89: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_260, primals_261, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    relu_39: "f32[8, 240, 1, 1]" = torch.ops.aten.relu.default(convolution_89);  convolution_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_90: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(relu_39, primals_262, primals_263, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_406: "f32[8, 960, 1, 1]" = torch.ops.aten.add.Tensor(convolution_90, 3)
    clamp_min_6: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_406, 0);  add_406 = None
    clamp_max_6: "f32[8, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
    div_6: "f32[8, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_6, 6);  clamp_max_6 = None
    mul_545: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(slice_168, div_6);  slice_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    convolution_91: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(mul_545, primals_264, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_407: "i64[]" = torch.ops.aten.add.Tensor(primals_509, 1)
    var_mean_77 = torch.ops.aten.var_mean.correction(convolution_91, [0, 2, 3], correction = 0, keepdim = True)
    getitem_154: "f32[1, 80, 1, 1]" = var_mean_77[0]
    getitem_155: "f32[1, 80, 1, 1]" = var_mean_77[1];  var_mean_77 = None
    add_408: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05)
    rsqrt_77: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_408);  add_408 = None
    sub_77: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_91, getitem_155)
    mul_546: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
    squeeze_231: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
    squeeze_232: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
    mul_547: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
    mul_548: "f32[80]" = torch.ops.aten.mul.Tensor(primals_507, 0.9)
    add_409: "f32[80]" = torch.ops.aten.add.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
    squeeze_233: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
    mul_549: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_233, 1.0025575447570332);  squeeze_233 = None
    mul_550: "f32[80]" = torch.ops.aten.mul.Tensor(mul_549, 0.1);  mul_549 = None
    mul_551: "f32[80]" = torch.ops.aten.mul.Tensor(primals_508, 0.9)
    add_410: "f32[80]" = torch.ops.aten.add.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
    unsqueeze_308: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_265, -1)
    unsqueeze_309: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_552: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_546, unsqueeze_309);  mul_546 = unsqueeze_309 = None
    unsqueeze_310: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_266, -1);  primals_266 = None
    unsqueeze_311: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_411: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_552, unsqueeze_311);  mul_552 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    convolution_92: "f32[8, 80, 7, 7]" = torch.ops.aten.convolution.default(add_411, primals_267, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 80)
    add_412: "i64[]" = torch.ops.aten.add.Tensor(primals_512, 1)
    var_mean_78 = torch.ops.aten.var_mean.correction(convolution_92, [0, 2, 3], correction = 0, keepdim = True)
    getitem_156: "f32[1, 80, 1, 1]" = var_mean_78[0]
    getitem_157: "f32[1, 80, 1, 1]" = var_mean_78[1];  var_mean_78 = None
    add_413: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
    rsqrt_78: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_413);  add_413 = None
    sub_78: "f32[8, 80, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_157)
    mul_553: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
    squeeze_234: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
    squeeze_235: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
    mul_554: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1)
    mul_555: "f32[80]" = torch.ops.aten.mul.Tensor(primals_510, 0.9)
    add_414: "f32[80]" = torch.ops.aten.add.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
    squeeze_236: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
    mul_556: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_236, 1.0025575447570332);  squeeze_236 = None
    mul_557: "f32[80]" = torch.ops.aten.mul.Tensor(mul_556, 0.1);  mul_556 = None
    mul_558: "f32[80]" = torch.ops.aten.mul.Tensor(primals_511, 0.9)
    add_415: "f32[80]" = torch.ops.aten.add.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
    unsqueeze_312: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_268, -1)
    unsqueeze_313: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    mul_559: "f32[8, 80, 7, 7]" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_313);  mul_553 = unsqueeze_313 = None
    unsqueeze_314: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_269, -1);  primals_269 = None
    unsqueeze_315: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    add_416: "f32[8, 80, 7, 7]" = torch.ops.aten.add.Tensor(mul_559, unsqueeze_315);  mul_559 = unsqueeze_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:62, code: out = torch.cat([x1, x2], dim=1)
    cat_31: "f32[8, 160, 7, 7]" = torch.ops.aten.cat.default([add_411, add_416], 1);  add_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:63, code: return out[:, :self.out_chs, :, :]
    slice_169: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(cat_31, 0, 0, 9223372036854775807)
    slice_170: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_169, 2, 0, 9223372036854775807)
    slice_171: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_170, 3, 0, 9223372036854775807)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    add_417: "f32[8, 160, 7, 7]" = torch.ops.aten.add.Tensor(slice_171, slice_165);  slice_171 = None
    slice_scatter_45: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(slice_170, add_417, 3, 0, 9223372036854775807);  slice_170 = add_417 = None
    slice_scatter_46: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(slice_169, slice_scatter_45, 2, 0, 9223372036854775807);  slice_169 = slice_scatter_45 = None
    slice_scatter_47: "f32[8, 160, 7, 7]" = torch.ops.aten.slice_scatter.default(cat_31, slice_scatter_46, 0, 0, 9223372036854775807);  cat_31 = slice_scatter_46 = None
    slice_174: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 0, 0, 9223372036854775807);  slice_scatter_47 = None
    slice_175: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_174, 2, 0, 9223372036854775807);  slice_174 = None
    slice_176: "f32[8, 160, 7, 7]" = torch.ops.aten.slice.Tensor(slice_175, 3, 0, 9223372036854775807);  slice_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:82, code: x = self.conv(x)
    convolution_93: "f32[8, 960, 7, 7]" = torch.ops.aten.convolution.default(slice_176, primals_270, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_418: "i64[]" = torch.ops.aten.add.Tensor(primals_273, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_79 = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True)
    getitem_158: "f32[1, 960, 1, 1]" = var_mean_79[0]
    getitem_159: "f32[1, 960, 1, 1]" = var_mean_79[1];  var_mean_79 = None
    add_419: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
    rsqrt_79: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_419);  add_419 = None
    sub_79: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_159)
    mul_560: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
    squeeze_237: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
    squeeze_238: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
    mul_561: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1)
    mul_562: "f32[960]" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
    add_420: "f32[960]" = torch.ops.aten.add.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    squeeze_239: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
    mul_563: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_239, 1.0025575447570332);  squeeze_239 = None
    mul_564: "f32[960]" = torch.ops.aten.mul.Tensor(mul_563, 0.1);  mul_563 = None
    mul_565: "f32[960]" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
    add_421: "f32[960]" = torch.ops.aten.add.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_316: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_317: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_566: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_317);  mul_560 = unsqueeze_317 = None
    unsqueeze_318: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_319: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_422: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_319);  mul_566 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_40: "f32[8, 960, 7, 7]" = torch.ops.aten.relu.default(add_422);  add_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_7: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(relu_40, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:293, code: x = self.conv_head(x)
    convolution_94: "f32[8, 1280, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_271, primals_272, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:294, code: x = self.act2(x)
    relu_41: "f32[8, 1280, 1, 1]" = torch.ops.aten.relu.default(convolution_94);  convolution_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/linear.py:19, code: return F.linear(input, self.weight, self.bias)
    permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    view_1: "f32[8, 1280]" = torch.ops.aten.view.default(relu_41, [8, 1280])
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_4, view_1, permute);  primals_4 = None
    permute_1: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:294, code: x = self.act2(x)
    alias_43: "f32[8, 1280, 1, 1]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_44: "f32[8, 1280, 1, 1]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    le: "b8[8, 1280, 1, 1]" = torch.ops.aten.le.Scalar(alias_44, 0);  alias_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_46: "f32[8, 960, 7, 7]" = torch.ops.aten.alias.default(relu_40);  relu_40 = None
    alias_47: "f32[8, 960, 7, 7]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    le_1: "b8[8, 960, 7, 7]" = torch.ops.aten.le.Scalar(alias_47, 0);  alias_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_320: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
    unsqueeze_321: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_332: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
    unsqueeze_333: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
    unsqueeze_334: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_344: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
    unsqueeze_345: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
    unsqueeze_346: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    gt: "b8[8, 960, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_90, -3.0)
    lt: "b8[8, 960, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_90, 3.0);  convolution_90 = None
    bitwise_and: "b8[8, 960, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt, lt);  gt = lt = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_52: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_53: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    le_3: "b8[8, 480, 7, 7]" = torch.ops.aten.le.Scalar(alias_53, 0);  alias_53 = None
    unsqueeze_356: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
    unsqueeze_357: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
    unsqueeze_358: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_368: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
    unsqueeze_369: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
    unsqueeze_370: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_380: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
    unsqueeze_381: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
    unsqueeze_382: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_392: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
    unsqueeze_393: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_58: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_59: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    le_5: "b8[8, 480, 7, 7]" = torch.ops.aten.le.Scalar(alias_59, 0);  alias_59 = None
    unsqueeze_404: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
    unsqueeze_405: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_416: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
    unsqueeze_417: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_428: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
    unsqueeze_429: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_440: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
    unsqueeze_441: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    gt_1: "b8[8, 960, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_80, -3.0)
    lt_1: "b8[8, 960, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_80, 3.0);  convolution_80 = None
    bitwise_and_1: "b8[8, 960, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_1, lt_1);  gt_1 = lt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_67: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_68: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(alias_67);  alias_67 = None
    le_8: "b8[8, 480, 7, 7]" = torch.ops.aten.le.Scalar(alias_68, 0);  alias_68 = None
    unsqueeze_452: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
    unsqueeze_453: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_464: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
    unsqueeze_465: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_476: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
    unsqueeze_477: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_488: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
    unsqueeze_489: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_73: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_74: "f32[8, 480, 7, 7]" = torch.ops.aten.alias.default(alias_73);  alias_73 = None
    le_10: "b8[8, 480, 7, 7]" = torch.ops.aten.le.Scalar(alias_74, 0);  alias_74 = None
    unsqueeze_500: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_501: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_512: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_513: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    unsqueeze_524: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_525: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    unsqueeze_536: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_537: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_548: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_549: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_560: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_561: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    gt_2: "b8[8, 672, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_68, -3.0)
    lt_2: "b8[8, 672, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_68, 3.0);  convolution_68 = None
    bitwise_and_2: "b8[8, 672, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_2, lt_2);  gt_2 = lt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    unsqueeze_572: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_573: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_82: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_83: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(alias_82);  alias_82 = None
    le_13: "b8[8, 336, 14, 14]" = torch.ops.aten.le.Scalar(alias_83, 0);  alias_83 = None
    unsqueeze_584: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_585: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_596: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_597: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_608: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_609: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_620: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_621: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    gt_3: "b8[8, 672, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_61, -3.0)
    lt_3: "b8[8, 672, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_61, 3.0);  convolution_61 = None
    bitwise_and_3: "b8[8, 672, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_3, lt_3);  gt_3 = lt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_91: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_92: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(alias_91);  alias_91 = None
    le_16: "b8[8, 336, 14, 14]" = torch.ops.aten.le.Scalar(alias_92, 0);  alias_92 = None
    unsqueeze_632: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_633: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_644: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_645: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    unsqueeze_656: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_657: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    unsqueeze_668: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_669: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_680: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_681: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_692: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_693: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    gt_4: "b8[8, 480, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_53, -3.0)
    lt_4: "b8[8, 480, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_53, 3.0);  convolution_53 = None
    bitwise_and_4: "b8[8, 480, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_4, lt_4);  gt_4 = lt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_100: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_101: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(alias_100);  alias_100 = None
    le_19: "b8[8, 240, 14, 14]" = torch.ops.aten.le.Scalar(alias_101, 0);  alias_101 = None
    unsqueeze_704: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_705: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
    unsqueeze_706: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_716: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_717: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
    unsqueeze_718: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_728: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_729: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
    unsqueeze_730: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_740: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_741: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
    unsqueeze_742: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_106: "f32[8, 92, 14, 14]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_107: "f32[8, 92, 14, 14]" = torch.ops.aten.alias.default(alias_106);  alias_106 = None
    le_21: "b8[8, 92, 14, 14]" = torch.ops.aten.le.Scalar(alias_107, 0);  alias_107 = None
    unsqueeze_752: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_753: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
    unsqueeze_754: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_764: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_765: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
    unsqueeze_766: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_776: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_777: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
    unsqueeze_778: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_788: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_789: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
    unsqueeze_790: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_112: "f32[8, 92, 14, 14]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_113: "f32[8, 92, 14, 14]" = torch.ops.aten.alias.default(alias_112);  alias_112 = None
    le_23: "b8[8, 92, 14, 14]" = torch.ops.aten.le.Scalar(alias_113, 0);  alias_113 = None
    unsqueeze_800: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_801: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
    unsqueeze_802: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_812: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_813: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
    unsqueeze_814: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_824: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_825: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
    unsqueeze_826: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_836: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_837: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
    unsqueeze_838: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_118: "f32[8, 100, 14, 14]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_119: "f32[8, 100, 14, 14]" = torch.ops.aten.alias.default(alias_118);  alias_118 = None
    le_25: "b8[8, 100, 14, 14]" = torch.ops.aten.le.Scalar(alias_119, 0);  alias_119 = None
    unsqueeze_848: "f32[1, 100]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_849: "f32[1, 100, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
    unsqueeze_850: "f32[1, 100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_860: "f32[1, 100]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_861: "f32[1, 100, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
    unsqueeze_862: "f32[1, 100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    unsqueeze_872: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_873: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
    unsqueeze_874: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None
    unsqueeze_884: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_885: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
    unsqueeze_886: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_896: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_897: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
    unsqueeze_898: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_908: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_909: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
    unsqueeze_910: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    unsqueeze_920: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_921: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 2);  unsqueeze_920 = None
    unsqueeze_922: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 3);  unsqueeze_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_124: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_125: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_124);  alias_124 = None
    le_27: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_125, 0);  alias_125 = None
    unsqueeze_932: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_933: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 2);  unsqueeze_932 = None
    unsqueeze_934: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 3);  unsqueeze_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_944: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_945: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 2);  unsqueeze_944 = None
    unsqueeze_946: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 3);  unsqueeze_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_956: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_957: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 2);  unsqueeze_956 = None
    unsqueeze_958: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 3);  unsqueeze_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_968: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_969: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
    unsqueeze_970: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    gt_5: "b8[8, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_28, -3.0)
    lt_5: "b8[8, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_28, 3.0);  convolution_28 = None
    bitwise_and_5: "b8[8, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_5, lt_5);  gt_5 = lt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_133: "f32[8, 60, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_134: "f32[8, 60, 28, 28]" = torch.ops.aten.alias.default(alias_133);  alias_133 = None
    le_30: "b8[8, 60, 28, 28]" = torch.ops.aten.le.Scalar(alias_134, 0);  alias_134 = None
    unsqueeze_980: "f32[1, 60]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_981: "f32[1, 60, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 2);  unsqueeze_980 = None
    unsqueeze_982: "f32[1, 60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 3);  unsqueeze_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_992: "f32[1, 60]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_993: "f32[1, 60, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 2);  unsqueeze_992 = None
    unsqueeze_994: "f32[1, 60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 3);  unsqueeze_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    unsqueeze_1004: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_1005: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 2);  unsqueeze_1004 = None
    unsqueeze_1006: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 3);  unsqueeze_1005 = None
    unsqueeze_1016: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_1017: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 2);  unsqueeze_1016 = None
    unsqueeze_1018: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 3);  unsqueeze_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_1028: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_1029: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 2);  unsqueeze_1028 = None
    unsqueeze_1030: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 3);  unsqueeze_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_1040: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_1041: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 2);  unsqueeze_1040 = None
    unsqueeze_1042: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 3);  unsqueeze_1041 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    gt_6: "b8[8, 72, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_20, -3.0)
    lt_6: "b8[8, 72, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_20, 3.0);  convolution_20 = None
    bitwise_and_6: "b8[8, 72, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_6, lt_6);  gt_6 = lt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    unsqueeze_1052: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_1053: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 2);  unsqueeze_1052 = None
    unsqueeze_1054: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 3);  unsqueeze_1053 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_142: "f32[8, 36, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_143: "f32[8, 36, 56, 56]" = torch.ops.aten.alias.default(alias_142);  alias_142 = None
    le_33: "b8[8, 36, 56, 56]" = torch.ops.aten.le.Scalar(alias_143, 0);  alias_143 = None
    unsqueeze_1064: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_1065: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 2);  unsqueeze_1064 = None
    unsqueeze_1066: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 3);  unsqueeze_1065 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_1076: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_1077: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, 2);  unsqueeze_1076 = None
    unsqueeze_1078: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 3);  unsqueeze_1077 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_1088: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_1089: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, 2);  unsqueeze_1088 = None
    unsqueeze_1090: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 3);  unsqueeze_1089 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_1100: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_1101: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, 2);  unsqueeze_1100 = None
    unsqueeze_1102: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 3);  unsqueeze_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_148: "f32[8, 36, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_149: "f32[8, 36, 56, 56]" = torch.ops.aten.alias.default(alias_148);  alias_148 = None
    le_35: "b8[8, 36, 56, 56]" = torch.ops.aten.le.Scalar(alias_149, 0);  alias_149 = None
    unsqueeze_1112: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_1113: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, 2);  unsqueeze_1112 = None
    unsqueeze_1114: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 3);  unsqueeze_1113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_1124: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_1125: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, 2);  unsqueeze_1124 = None
    unsqueeze_1126: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 3);  unsqueeze_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:182, code: x += self.shortcut(shortcut)
    unsqueeze_1136: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_1137: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, 2);  unsqueeze_1136 = None
    unsqueeze_1138: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 3);  unsqueeze_1137 = None
    unsqueeze_1148: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_1149: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, 2);  unsqueeze_1148 = None
    unsqueeze_1150: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 3);  unsqueeze_1149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_1160: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_1161: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, 2);  unsqueeze_1160 = None
    unsqueeze_1162: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 3);  unsqueeze_1161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_1172: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_1173: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, 2);  unsqueeze_1172 = None
    unsqueeze_1174: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 3);  unsqueeze_1173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:173, code: x = self.bn_dw(x)
    unsqueeze_1184: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_1185: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, 2);  unsqueeze_1184 = None
    unsqueeze_1186: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 3);  unsqueeze_1185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_154: "f32[8, 24, 112, 112]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_155: "f32[8, 24, 112, 112]" = torch.ops.aten.alias.default(alias_154);  alias_154 = None
    le_37: "b8[8, 24, 112, 112]" = torch.ops.aten.le.Scalar(alias_155, 0);  alias_155 = None
    unsqueeze_1196: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_1197: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, 2);  unsqueeze_1196 = None
    unsqueeze_1198: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 3);  unsqueeze_1197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_1208: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_1209: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, 2);  unsqueeze_1208 = None
    unsqueeze_1210: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 3);  unsqueeze_1209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    unsqueeze_1220: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_1221: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, 2);  unsqueeze_1220 = None
    unsqueeze_1222: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 3);  unsqueeze_1221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_1232: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_1233: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, 2);  unsqueeze_1232 = None
    unsqueeze_1234: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 3);  unsqueeze_1233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:61, code: x2 = self.cheap_operation(x1)
    alias_160: "f32[8, 8, 112, 112]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_161: "f32[8, 8, 112, 112]" = torch.ops.aten.alias.default(alias_160);  alias_160 = None
    le_39: "b8[8, 8, 112, 112]" = torch.ops.aten.le.Scalar(alias_161, 0);  alias_161 = None
    unsqueeze_1244: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_1245: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, 2);  unsqueeze_1244 = None
    unsqueeze_1246: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 3);  unsqueeze_1245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:60, code: x1 = self.primary_conv(x)
    unsqueeze_1256: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_1257: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, 2);  unsqueeze_1256 = None
    unsqueeze_1258: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 3);  unsqueeze_1257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/ghostnet.py:283, code: x = self.bn1(x)
    unsqueeze_1268: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_1269: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, 2);  unsqueeze_1268 = None
    unsqueeze_1270: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 3);  unsqueeze_1269 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_273, add_418);  primals_273 = add_418 = None
    copy__1: "f32[960]" = torch.ops.aten.copy_.default(primals_274, add_420);  primals_274 = add_420 = None
    copy__2: "f32[960]" = torch.ops.aten.copy_.default(primals_275, add_421);  primals_275 = add_421 = None
    copy__3: "f32[16]" = torch.ops.aten.copy_.default(primals_276, add_2);  primals_276 = add_2 = None
    copy__4: "f32[16]" = torch.ops.aten.copy_.default(primals_277, add_3);  primals_277 = add_3 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_278, add);  primals_278 = add = None
    copy__6: "f32[8]" = torch.ops.aten.copy_.default(primals_279, add_7);  primals_279 = add_7 = None
    copy__7: "f32[8]" = torch.ops.aten.copy_.default(primals_280, add_8);  primals_280 = add_8 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_281, add_5);  primals_281 = add_5 = None
    copy__9: "f32[8]" = torch.ops.aten.copy_.default(primals_282, add_12);  primals_282 = add_12 = None
    copy__10: "f32[8]" = torch.ops.aten.copy_.default(primals_283, add_13);  primals_283 = add_13 = None
    copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_284, add_10);  primals_284 = add_10 = None
    copy__12: "f32[8]" = torch.ops.aten.copy_.default(primals_285, add_17);  primals_285 = add_17 = None
    copy__13: "f32[8]" = torch.ops.aten.copy_.default(primals_286, add_18);  primals_286 = add_18 = None
    copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_287, add_15);  primals_287 = add_15 = None
    copy__15: "f32[8]" = torch.ops.aten.copy_.default(primals_288, add_22);  primals_288 = add_22 = None
    copy__16: "f32[8]" = torch.ops.aten.copy_.default(primals_289, add_23);  primals_289 = add_23 = None
    copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_290, add_20);  primals_290 = add_20 = None
    copy__18: "f32[24]" = torch.ops.aten.copy_.default(primals_291, add_28);  primals_291 = add_28 = None
    copy__19: "f32[24]" = torch.ops.aten.copy_.default(primals_292, add_29);  primals_292 = add_29 = None
    copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_293, add_26);  primals_293 = add_26 = None
    copy__21: "f32[24]" = torch.ops.aten.copy_.default(primals_294, add_33);  primals_294 = add_33 = None
    copy__22: "f32[24]" = torch.ops.aten.copy_.default(primals_295, add_34);  primals_295 = add_34 = None
    copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_296, add_31);  primals_296 = add_31 = None
    copy__24: "f32[48]" = torch.ops.aten.copy_.default(primals_297, add_38);  primals_297 = add_38 = None
    copy__25: "f32[48]" = torch.ops.aten.copy_.default(primals_298, add_39);  primals_298 = add_39 = None
    copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_299, add_36);  primals_299 = add_36 = None
    copy__27: "f32[12]" = torch.ops.aten.copy_.default(primals_300, add_43);  primals_300 = add_43 = None
    copy__28: "f32[12]" = torch.ops.aten.copy_.default(primals_301, add_44);  primals_301 = add_44 = None
    copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_302, add_41);  primals_302 = add_41 = None
    copy__30: "f32[12]" = torch.ops.aten.copy_.default(primals_303, add_48);  primals_303 = add_48 = None
    copy__31: "f32[12]" = torch.ops.aten.copy_.default(primals_304, add_49);  primals_304 = add_49 = None
    copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_305, add_46);  primals_305 = add_46 = None
    copy__33: "f32[16]" = torch.ops.aten.copy_.default(primals_306, add_53);  primals_306 = add_53 = None
    copy__34: "f32[16]" = torch.ops.aten.copy_.default(primals_307, add_54);  primals_307 = add_54 = None
    copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_308, add_51);  primals_308 = add_51 = None
    copy__36: "f32[24]" = torch.ops.aten.copy_.default(primals_309, add_58);  primals_309 = add_58 = None
    copy__37: "f32[24]" = torch.ops.aten.copy_.default(primals_310, add_59);  primals_310 = add_59 = None
    copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_311, add_56);  primals_311 = add_56 = None
    copy__39: "f32[36]" = torch.ops.aten.copy_.default(primals_312, add_64);  primals_312 = add_64 = None
    copy__40: "f32[36]" = torch.ops.aten.copy_.default(primals_313, add_65);  primals_313 = add_65 = None
    copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_314, add_62);  primals_314 = add_62 = None
    copy__42: "f32[36]" = torch.ops.aten.copy_.default(primals_315, add_69);  primals_315 = add_69 = None
    copy__43: "f32[36]" = torch.ops.aten.copy_.default(primals_316, add_70);  primals_316 = add_70 = None
    copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_317, add_67);  primals_317 = add_67 = None
    copy__45: "f32[12]" = torch.ops.aten.copy_.default(primals_318, add_74);  primals_318 = add_74 = None
    copy__46: "f32[12]" = torch.ops.aten.copy_.default(primals_319, add_75);  primals_319 = add_75 = None
    copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_320, add_72);  primals_320 = add_72 = None
    copy__48: "f32[12]" = torch.ops.aten.copy_.default(primals_321, add_79);  primals_321 = add_79 = None
    copy__49: "f32[12]" = torch.ops.aten.copy_.default(primals_322, add_80);  primals_322 = add_80 = None
    copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_323, add_77);  primals_323 = add_77 = None
    copy__51: "f32[36]" = torch.ops.aten.copy_.default(primals_324, add_85);  primals_324 = add_85 = None
    copy__52: "f32[36]" = torch.ops.aten.copy_.default(primals_325, add_86);  primals_325 = add_86 = None
    copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_326, add_83);  primals_326 = add_83 = None
    copy__54: "f32[36]" = torch.ops.aten.copy_.default(primals_327, add_90);  primals_327 = add_90 = None
    copy__55: "f32[36]" = torch.ops.aten.copy_.default(primals_328, add_91);  primals_328 = add_91 = None
    copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_329, add_88);  primals_329 = add_88 = None
    copy__57: "f32[72]" = torch.ops.aten.copy_.default(primals_330, add_95);  primals_330 = add_95 = None
    copy__58: "f32[72]" = torch.ops.aten.copy_.default(primals_331, add_96);  primals_331 = add_96 = None
    copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_332, add_93);  primals_332 = add_93 = None
    copy__60: "f32[20]" = torch.ops.aten.copy_.default(primals_333, add_101);  primals_333 = add_101 = None
    copy__61: "f32[20]" = torch.ops.aten.copy_.default(primals_334, add_102);  primals_334 = add_102 = None
    copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_335, add_99);  primals_335 = add_99 = None
    copy__63: "f32[20]" = torch.ops.aten.copy_.default(primals_336, add_106);  primals_336 = add_106 = None
    copy__64: "f32[20]" = torch.ops.aten.copy_.default(primals_337, add_107);  primals_337 = add_107 = None
    copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_338, add_104);  primals_338 = add_104 = None
    copy__66: "f32[24]" = torch.ops.aten.copy_.default(primals_339, add_111);  primals_339 = add_111 = None
    copy__67: "f32[24]" = torch.ops.aten.copy_.default(primals_340, add_112);  primals_340 = add_112 = None
    copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_341, add_109);  primals_341 = add_109 = None
    copy__69: "f32[40]" = torch.ops.aten.copy_.default(primals_342, add_116);  primals_342 = add_116 = None
    copy__70: "f32[40]" = torch.ops.aten.copy_.default(primals_343, add_117);  primals_343 = add_117 = None
    copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_344, add_114);  primals_344 = add_114 = None
    copy__72: "f32[60]" = torch.ops.aten.copy_.default(primals_345, add_122);  primals_345 = add_122 = None
    copy__73: "f32[60]" = torch.ops.aten.copy_.default(primals_346, add_123);  primals_346 = add_123 = None
    copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_347, add_120);  primals_347 = add_120 = None
    copy__75: "f32[60]" = torch.ops.aten.copy_.default(primals_348, add_127);  primals_348 = add_127 = None
    copy__76: "f32[60]" = torch.ops.aten.copy_.default(primals_349, add_128);  primals_349 = add_128 = None
    copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_350, add_125);  primals_350 = add_125 = None
    copy__78: "f32[20]" = torch.ops.aten.copy_.default(primals_351, add_133);  primals_351 = add_133 = None
    copy__79: "f32[20]" = torch.ops.aten.copy_.default(primals_352, add_134);  primals_352 = add_134 = None
    copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_353, add_131);  primals_353 = add_131 = None
    copy__81: "f32[20]" = torch.ops.aten.copy_.default(primals_354, add_138);  primals_354 = add_138 = None
    copy__82: "f32[20]" = torch.ops.aten.copy_.default(primals_355, add_139);  primals_355 = add_139 = None
    copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_356, add_136);  primals_356 = add_136 = None
    copy__84: "f32[120]" = torch.ops.aten.copy_.default(primals_357, add_144);  primals_357 = add_144 = None
    copy__85: "f32[120]" = torch.ops.aten.copy_.default(primals_358, add_145);  primals_358 = add_145 = None
    copy__86: "i64[]" = torch.ops.aten.copy_.default(primals_359, add_142);  primals_359 = add_142 = None
    copy__87: "f32[120]" = torch.ops.aten.copy_.default(primals_360, add_149);  primals_360 = add_149 = None
    copy__88: "f32[120]" = torch.ops.aten.copy_.default(primals_361, add_150);  primals_361 = add_150 = None
    copy__89: "i64[]" = torch.ops.aten.copy_.default(primals_362, add_147);  primals_362 = add_147 = None
    copy__90: "f32[240]" = torch.ops.aten.copy_.default(primals_363, add_154);  primals_363 = add_154 = None
    copy__91: "f32[240]" = torch.ops.aten.copy_.default(primals_364, add_155);  primals_364 = add_155 = None
    copy__92: "i64[]" = torch.ops.aten.copy_.default(primals_365, add_152);  primals_365 = add_152 = None
    copy__93: "f32[40]" = torch.ops.aten.copy_.default(primals_366, add_159);  primals_366 = add_159 = None
    copy__94: "f32[40]" = torch.ops.aten.copy_.default(primals_367, add_160);  primals_367 = add_160 = None
    copy__95: "i64[]" = torch.ops.aten.copy_.default(primals_368, add_157);  primals_368 = add_157 = None
    copy__96: "f32[40]" = torch.ops.aten.copy_.default(primals_369, add_164);  primals_369 = add_164 = None
    copy__97: "f32[40]" = torch.ops.aten.copy_.default(primals_370, add_165);  primals_370 = add_165 = None
    copy__98: "i64[]" = torch.ops.aten.copy_.default(primals_371, add_162);  primals_371 = add_162 = None
    copy__99: "f32[40]" = torch.ops.aten.copy_.default(primals_372, add_169);  primals_372 = add_169 = None
    copy__100: "f32[40]" = torch.ops.aten.copy_.default(primals_373, add_170);  primals_373 = add_170 = None
    copy__101: "i64[]" = torch.ops.aten.copy_.default(primals_374, add_167);  primals_374 = add_167 = None
    copy__102: "f32[80]" = torch.ops.aten.copy_.default(primals_375, add_174);  primals_375 = add_174 = None
    copy__103: "f32[80]" = torch.ops.aten.copy_.default(primals_376, add_175);  primals_376 = add_175 = None
    copy__104: "i64[]" = torch.ops.aten.copy_.default(primals_377, add_172);  primals_377 = add_172 = None
    copy__105: "f32[100]" = torch.ops.aten.copy_.default(primals_378, add_180);  primals_378 = add_180 = None
    copy__106: "f32[100]" = torch.ops.aten.copy_.default(primals_379, add_181);  primals_379 = add_181 = None
    copy__107: "i64[]" = torch.ops.aten.copy_.default(primals_380, add_178);  primals_380 = add_178 = None
    copy__108: "f32[100]" = torch.ops.aten.copy_.default(primals_381, add_185);  primals_381 = add_185 = None
    copy__109: "f32[100]" = torch.ops.aten.copy_.default(primals_382, add_186);  primals_382 = add_186 = None
    copy__110: "i64[]" = torch.ops.aten.copy_.default(primals_383, add_183);  primals_383 = add_183 = None
    copy__111: "f32[40]" = torch.ops.aten.copy_.default(primals_384, add_190);  primals_384 = add_190 = None
    copy__112: "f32[40]" = torch.ops.aten.copy_.default(primals_385, add_191);  primals_385 = add_191 = None
    copy__113: "i64[]" = torch.ops.aten.copy_.default(primals_386, add_188);  primals_386 = add_188 = None
    copy__114: "f32[40]" = torch.ops.aten.copy_.default(primals_387, add_195);  primals_387 = add_195 = None
    copy__115: "f32[40]" = torch.ops.aten.copy_.default(primals_388, add_196);  primals_388 = add_196 = None
    copy__116: "i64[]" = torch.ops.aten.copy_.default(primals_389, add_193);  primals_389 = add_193 = None
    copy__117: "f32[92]" = torch.ops.aten.copy_.default(primals_390, add_201);  primals_390 = add_201 = None
    copy__118: "f32[92]" = torch.ops.aten.copy_.default(primals_391, add_202);  primals_391 = add_202 = None
    copy__119: "i64[]" = torch.ops.aten.copy_.default(primals_392, add_199);  primals_392 = add_199 = None
    copy__120: "f32[92]" = torch.ops.aten.copy_.default(primals_393, add_206);  primals_393 = add_206 = None
    copy__121: "f32[92]" = torch.ops.aten.copy_.default(primals_394, add_207);  primals_394 = add_207 = None
    copy__122: "i64[]" = torch.ops.aten.copy_.default(primals_395, add_204);  primals_395 = add_204 = None
    copy__123: "f32[40]" = torch.ops.aten.copy_.default(primals_396, add_211);  primals_396 = add_211 = None
    copy__124: "f32[40]" = torch.ops.aten.copy_.default(primals_397, add_212);  primals_397 = add_212 = None
    copy__125: "i64[]" = torch.ops.aten.copy_.default(primals_398, add_209);  primals_398 = add_209 = None
    copy__126: "f32[40]" = torch.ops.aten.copy_.default(primals_399, add_216);  primals_399 = add_216 = None
    copy__127: "f32[40]" = torch.ops.aten.copy_.default(primals_400, add_217);  primals_400 = add_217 = None
    copy__128: "i64[]" = torch.ops.aten.copy_.default(primals_401, add_214);  primals_401 = add_214 = None
    copy__129: "f32[92]" = torch.ops.aten.copy_.default(primals_402, add_222);  primals_402 = add_222 = None
    copy__130: "f32[92]" = torch.ops.aten.copy_.default(primals_403, add_223);  primals_403 = add_223 = None
    copy__131: "i64[]" = torch.ops.aten.copy_.default(primals_404, add_220);  primals_404 = add_220 = None
    copy__132: "f32[92]" = torch.ops.aten.copy_.default(primals_405, add_227);  primals_405 = add_227 = None
    copy__133: "f32[92]" = torch.ops.aten.copy_.default(primals_406, add_228);  primals_406 = add_228 = None
    copy__134: "i64[]" = torch.ops.aten.copy_.default(primals_407, add_225);  primals_407 = add_225 = None
    copy__135: "f32[40]" = torch.ops.aten.copy_.default(primals_408, add_232);  primals_408 = add_232 = None
    copy__136: "f32[40]" = torch.ops.aten.copy_.default(primals_409, add_233);  primals_409 = add_233 = None
    copy__137: "i64[]" = torch.ops.aten.copy_.default(primals_410, add_230);  primals_410 = add_230 = None
    copy__138: "f32[40]" = torch.ops.aten.copy_.default(primals_411, add_237);  primals_411 = add_237 = None
    copy__139: "f32[40]" = torch.ops.aten.copy_.default(primals_412, add_238);  primals_412 = add_238 = None
    copy__140: "i64[]" = torch.ops.aten.copy_.default(primals_413, add_235);  primals_413 = add_235 = None
    copy__141: "f32[240]" = torch.ops.aten.copy_.default(primals_414, add_243);  primals_414 = add_243 = None
    copy__142: "f32[240]" = torch.ops.aten.copy_.default(primals_415, add_244);  primals_415 = add_244 = None
    copy__143: "i64[]" = torch.ops.aten.copy_.default(primals_416, add_241);  primals_416 = add_241 = None
    copy__144: "f32[240]" = torch.ops.aten.copy_.default(primals_417, add_248);  primals_417 = add_248 = None
    copy__145: "f32[240]" = torch.ops.aten.copy_.default(primals_418, add_249);  primals_418 = add_249 = None
    copy__146: "i64[]" = torch.ops.aten.copy_.default(primals_419, add_246);  primals_419 = add_246 = None
    copy__147: "f32[56]" = torch.ops.aten.copy_.default(primals_420, add_254);  primals_420 = add_254 = None
    copy__148: "f32[56]" = torch.ops.aten.copy_.default(primals_421, add_255);  primals_421 = add_255 = None
    copy__149: "i64[]" = torch.ops.aten.copy_.default(primals_422, add_252);  primals_422 = add_252 = None
    copy__150: "f32[56]" = torch.ops.aten.copy_.default(primals_423, add_259);  primals_423 = add_259 = None
    copy__151: "f32[56]" = torch.ops.aten.copy_.default(primals_424, add_260);  primals_424 = add_260 = None
    copy__152: "i64[]" = torch.ops.aten.copy_.default(primals_425, add_257);  primals_425 = add_257 = None
    copy__153: "f32[80]" = torch.ops.aten.copy_.default(primals_426, add_264);  primals_426 = add_264 = None
    copy__154: "f32[80]" = torch.ops.aten.copy_.default(primals_427, add_265);  primals_427 = add_265 = None
    copy__155: "i64[]" = torch.ops.aten.copy_.default(primals_428, add_262);  primals_428 = add_262 = None
    copy__156: "f32[112]" = torch.ops.aten.copy_.default(primals_429, add_269);  primals_429 = add_269 = None
    copy__157: "f32[112]" = torch.ops.aten.copy_.default(primals_430, add_270);  primals_430 = add_270 = None
    copy__158: "i64[]" = torch.ops.aten.copy_.default(primals_431, add_267);  primals_431 = add_267 = None
    copy__159: "f32[336]" = torch.ops.aten.copy_.default(primals_432, add_275);  primals_432 = add_275 = None
    copy__160: "f32[336]" = torch.ops.aten.copy_.default(primals_433, add_276);  primals_433 = add_276 = None
    copy__161: "i64[]" = torch.ops.aten.copy_.default(primals_434, add_273);  primals_434 = add_273 = None
    copy__162: "f32[336]" = torch.ops.aten.copy_.default(primals_435, add_280);  primals_435 = add_280 = None
    copy__163: "f32[336]" = torch.ops.aten.copy_.default(primals_436, add_281);  primals_436 = add_281 = None
    copy__164: "i64[]" = torch.ops.aten.copy_.default(primals_437, add_278);  primals_437 = add_278 = None
    copy__165: "f32[56]" = torch.ops.aten.copy_.default(primals_438, add_286);  primals_438 = add_286 = None
    copy__166: "f32[56]" = torch.ops.aten.copy_.default(primals_439, add_287);  primals_439 = add_287 = None
    copy__167: "i64[]" = torch.ops.aten.copy_.default(primals_440, add_284);  primals_440 = add_284 = None
    copy__168: "f32[56]" = torch.ops.aten.copy_.default(primals_441, add_291);  primals_441 = add_291 = None
    copy__169: "f32[56]" = torch.ops.aten.copy_.default(primals_442, add_292);  primals_442 = add_292 = None
    copy__170: "i64[]" = torch.ops.aten.copy_.default(primals_443, add_289);  primals_443 = add_289 = None
    copy__171: "f32[336]" = torch.ops.aten.copy_.default(primals_444, add_297);  primals_444 = add_297 = None
    copy__172: "f32[336]" = torch.ops.aten.copy_.default(primals_445, add_298);  primals_445 = add_298 = None
    copy__173: "i64[]" = torch.ops.aten.copy_.default(primals_446, add_295);  primals_446 = add_295 = None
    copy__174: "f32[336]" = torch.ops.aten.copy_.default(primals_447, add_302);  primals_447 = add_302 = None
    copy__175: "f32[336]" = torch.ops.aten.copy_.default(primals_448, add_303);  primals_448 = add_303 = None
    copy__176: "i64[]" = torch.ops.aten.copy_.default(primals_449, add_300);  primals_449 = add_300 = None
    copy__177: "f32[672]" = torch.ops.aten.copy_.default(primals_450, add_307);  primals_450 = add_307 = None
    copy__178: "f32[672]" = torch.ops.aten.copy_.default(primals_451, add_308);  primals_451 = add_308 = None
    copy__179: "i64[]" = torch.ops.aten.copy_.default(primals_452, add_305);  primals_452 = add_305 = None
    copy__180: "f32[80]" = torch.ops.aten.copy_.default(primals_453, add_313);  primals_453 = add_313 = None
    copy__181: "f32[80]" = torch.ops.aten.copy_.default(primals_454, add_314);  primals_454 = add_314 = None
    copy__182: "i64[]" = torch.ops.aten.copy_.default(primals_455, add_311);  primals_455 = add_311 = None
    copy__183: "f32[80]" = torch.ops.aten.copy_.default(primals_456, add_318);  primals_456 = add_318 = None
    copy__184: "f32[80]" = torch.ops.aten.copy_.default(primals_457, add_319);  primals_457 = add_319 = None
    copy__185: "i64[]" = torch.ops.aten.copy_.default(primals_458, add_316);  primals_458 = add_316 = None
    copy__186: "f32[112]" = torch.ops.aten.copy_.default(primals_459, add_323);  primals_459 = add_323 = None
    copy__187: "f32[112]" = torch.ops.aten.copy_.default(primals_460, add_324);  primals_460 = add_324 = None
    copy__188: "i64[]" = torch.ops.aten.copy_.default(primals_461, add_321);  primals_461 = add_321 = None
    copy__189: "f32[160]" = torch.ops.aten.copy_.default(primals_462, add_328);  primals_462 = add_328 = None
    copy__190: "f32[160]" = torch.ops.aten.copy_.default(primals_463, add_329);  primals_463 = add_329 = None
    copy__191: "i64[]" = torch.ops.aten.copy_.default(primals_464, add_326);  primals_464 = add_326 = None
    copy__192: "f32[480]" = torch.ops.aten.copy_.default(primals_465, add_334);  primals_465 = add_334 = None
    copy__193: "f32[480]" = torch.ops.aten.copy_.default(primals_466, add_335);  primals_466 = add_335 = None
    copy__194: "i64[]" = torch.ops.aten.copy_.default(primals_467, add_332);  primals_467 = add_332 = None
    copy__195: "f32[480]" = torch.ops.aten.copy_.default(primals_468, add_339);  primals_468 = add_339 = None
    copy__196: "f32[480]" = torch.ops.aten.copy_.default(primals_469, add_340);  primals_469 = add_340 = None
    copy__197: "i64[]" = torch.ops.aten.copy_.default(primals_470, add_337);  primals_470 = add_337 = None
    copy__198: "f32[80]" = torch.ops.aten.copy_.default(primals_471, add_344);  primals_471 = add_344 = None
    copy__199: "f32[80]" = torch.ops.aten.copy_.default(primals_472, add_345);  primals_472 = add_345 = None
    copy__200: "i64[]" = torch.ops.aten.copy_.default(primals_473, add_342);  primals_473 = add_342 = None
    copy__201: "f32[80]" = torch.ops.aten.copy_.default(primals_474, add_349);  primals_474 = add_349 = None
    copy__202: "f32[80]" = torch.ops.aten.copy_.default(primals_475, add_350);  primals_475 = add_350 = None
    copy__203: "i64[]" = torch.ops.aten.copy_.default(primals_476, add_347);  primals_476 = add_347 = None
    copy__204: "f32[480]" = torch.ops.aten.copy_.default(primals_477, add_355);  primals_477 = add_355 = None
    copy__205: "f32[480]" = torch.ops.aten.copy_.default(primals_478, add_356);  primals_478 = add_356 = None
    copy__206: "i64[]" = torch.ops.aten.copy_.default(primals_479, add_353);  primals_479 = add_353 = None
    copy__207: "f32[480]" = torch.ops.aten.copy_.default(primals_480, add_360);  primals_480 = add_360 = None
    copy__208: "f32[480]" = torch.ops.aten.copy_.default(primals_481, add_361);  primals_481 = add_361 = None
    copy__209: "i64[]" = torch.ops.aten.copy_.default(primals_482, add_358);  primals_482 = add_358 = None
    copy__210: "f32[80]" = torch.ops.aten.copy_.default(primals_483, add_366);  primals_483 = add_366 = None
    copy__211: "f32[80]" = torch.ops.aten.copy_.default(primals_484, add_367);  primals_484 = add_367 = None
    copy__212: "i64[]" = torch.ops.aten.copy_.default(primals_485, add_364);  primals_485 = add_364 = None
    copy__213: "f32[80]" = torch.ops.aten.copy_.default(primals_486, add_371);  primals_486 = add_371 = None
    copy__214: "f32[80]" = torch.ops.aten.copy_.default(primals_487, add_372);  primals_487 = add_372 = None
    copy__215: "i64[]" = torch.ops.aten.copy_.default(primals_488, add_369);  primals_488 = add_369 = None
    copy__216: "f32[480]" = torch.ops.aten.copy_.default(primals_489, add_377);  primals_489 = add_377 = None
    copy__217: "f32[480]" = torch.ops.aten.copy_.default(primals_490, add_378);  primals_490 = add_378 = None
    copy__218: "i64[]" = torch.ops.aten.copy_.default(primals_491, add_375);  primals_491 = add_375 = None
    copy__219: "f32[480]" = torch.ops.aten.copy_.default(primals_492, add_382);  primals_492 = add_382 = None
    copy__220: "f32[480]" = torch.ops.aten.copy_.default(primals_493, add_383);  primals_493 = add_383 = None
    copy__221: "i64[]" = torch.ops.aten.copy_.default(primals_494, add_380);  primals_494 = add_380 = None
    copy__222: "f32[80]" = torch.ops.aten.copy_.default(primals_495, add_387);  primals_495 = add_387 = None
    copy__223: "f32[80]" = torch.ops.aten.copy_.default(primals_496, add_388);  primals_496 = add_388 = None
    copy__224: "i64[]" = torch.ops.aten.copy_.default(primals_497, add_385);  primals_497 = add_385 = None
    copy__225: "f32[80]" = torch.ops.aten.copy_.default(primals_498, add_392);  primals_498 = add_392 = None
    copy__226: "f32[80]" = torch.ops.aten.copy_.default(primals_499, add_393);  primals_499 = add_393 = None
    copy__227: "i64[]" = torch.ops.aten.copy_.default(primals_500, add_390);  primals_500 = add_390 = None
    copy__228: "f32[480]" = torch.ops.aten.copy_.default(primals_501, add_398);  primals_501 = add_398 = None
    copy__229: "f32[480]" = torch.ops.aten.copy_.default(primals_502, add_399);  primals_502 = add_399 = None
    copy__230: "i64[]" = torch.ops.aten.copy_.default(primals_503, add_396);  primals_503 = add_396 = None
    copy__231: "f32[480]" = torch.ops.aten.copy_.default(primals_504, add_403);  primals_504 = add_403 = None
    copy__232: "f32[480]" = torch.ops.aten.copy_.default(primals_505, add_404);  primals_505 = add_404 = None
    copy__233: "i64[]" = torch.ops.aten.copy_.default(primals_506, add_401);  primals_506 = add_401 = None
    copy__234: "f32[80]" = torch.ops.aten.copy_.default(primals_507, add_409);  primals_507 = add_409 = None
    copy__235: "f32[80]" = torch.ops.aten.copy_.default(primals_508, add_410);  primals_508 = add_410 = None
    copy__236: "i64[]" = torch.ops.aten.copy_.default(primals_509, add_407);  primals_509 = add_407 = None
    copy__237: "f32[80]" = torch.ops.aten.copy_.default(primals_510, add_414);  primals_510 = add_414 = None
    copy__238: "f32[80]" = torch.ops.aten.copy_.default(primals_511, add_415);  primals_511 = add_415 = None
    copy__239: "i64[]" = torch.ops.aten.copy_.default(primals_512, add_412);  primals_512 = add_412 = None
    return [addmm, primals_1, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, primals_20, primals_21, primals_23, primals_24, primals_26, primals_27, primals_29, primals_30, primals_32, primals_33, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_48, primals_50, primals_51, primals_53, primals_54, primals_56, primals_57, primals_59, primals_60, primals_62, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_159, primals_161, primals_162, primals_164, primals_165, primals_167, primals_168, primals_170, primals_171, primals_173, primals_174, primals_176, primals_177, primals_179, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_234, primals_236, primals_237, primals_239, primals_240, primals_242, primals_243, primals_245, primals_246, primals_248, primals_249, primals_251, primals_252, primals_254, primals_255, primals_257, primals_258, primals_260, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_513, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, slice_3, convolution_3, squeeze_10, add_19, convolution_4, squeeze_13, slice_11, convolution_5, squeeze_16, relu_3, convolution_6, squeeze_19, slice_14, convolution_7, squeeze_22, add_40, convolution_8, squeeze_25, add_45, convolution_9, squeeze_28, convolution_10, squeeze_31, add_55, convolution_11, squeeze_34, slice_22, convolution_12, squeeze_37, relu_5, convolution_13, squeeze_40, slice_25, convolution_14, squeeze_43, add_76, convolution_15, squeeze_46, slice_33, convolution_16, squeeze_49, relu_7, convolution_17, squeeze_52, slice_36, convolution_18, squeeze_55, add_97, mean, relu_9, div, mul_133, convolution_21, squeeze_58, add_103, convolution_22, squeeze_61, convolution_23, squeeze_64, add_113, convolution_24, squeeze_67, slice_44, convolution_25, squeeze_70, relu_10, convolution_26, squeeze_73, cat_8, mean_1, relu_12, div_1, mul_176, convolution_29, squeeze_76, add_135, convolution_30, squeeze_79, slice_55, convolution_31, squeeze_82, relu_13, convolution_32, squeeze_85, slice_58, convolution_33, squeeze_88, add_156, convolution_34, squeeze_91, add_161, convolution_35, squeeze_94, convolution_36, squeeze_97, add_171, convolution_37, squeeze_100, slice_66, convolution_38, squeeze_103, relu_15, convolution_39, squeeze_106, slice_69, convolution_40, squeeze_109, add_192, convolution_41, squeeze_112, slice_77, convolution_42, squeeze_115, relu_17, convolution_43, squeeze_118, slice_80, convolution_44, squeeze_121, add_213, convolution_45, squeeze_124, slice_88, convolution_46, squeeze_127, relu_19, convolution_47, squeeze_130, slice_91, convolution_48, squeeze_133, add_234, convolution_49, squeeze_136, slice_99, convolution_50, squeeze_139, relu_21, convolution_51, squeeze_142, cat_18, mean_2, relu_23, div_2, mul_338, convolution_54, squeeze_145, add_256, convolution_55, squeeze_148, convolution_56, squeeze_151, add_266, convolution_57, squeeze_154, slice_110, convolution_58, squeeze_157, relu_24, convolution_59, squeeze_160, cat_20, mean_3, relu_26, div_3, mul_381, convolution_62, squeeze_163, add_288, convolution_63, squeeze_166, slice_121, convolution_64, squeeze_169, relu_27, convolution_65, squeeze_172, slice_124, convolution_66, squeeze_175, add_309, mean_4, relu_29, div_4, mul_417, convolution_69, squeeze_178, add_315, convolution_70, squeeze_181, convolution_71, squeeze_184, add_325, convolution_72, squeeze_187, slice_132, convolution_73, squeeze_190, relu_30, convolution_74, squeeze_193, slice_135, convolution_75, squeeze_196, add_346, convolution_76, squeeze_199, slice_143, convolution_77, squeeze_202, relu_32, convolution_78, squeeze_205, cat_26, mean_5, relu_34, div_5, mul_488, convolution_81, squeeze_208, add_368, convolution_82, squeeze_211, slice_154, convolution_83, squeeze_214, relu_35, convolution_84, squeeze_217, slice_157, convolution_85, squeeze_220, add_389, convolution_86, squeeze_223, slice_165, convolution_87, squeeze_226, relu_37, convolution_88, squeeze_229, cat_30, mean_6, relu_39, div_6, mul_545, convolution_91, squeeze_232, add_411, convolution_92, squeeze_235, slice_176, convolution_93, squeeze_238, mean_7, view_1, permute_1, le, le_1, unsqueeze_322, unsqueeze_334, unsqueeze_346, bitwise_and, le_3, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, le_5, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, bitwise_and_1, le_8, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, le_10, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, bitwise_and_2, unsqueeze_574, le_13, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, bitwise_and_3, le_16, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, bitwise_and_4, le_19, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, le_21, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, le_23, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, le_25, unsqueeze_850, unsqueeze_862, unsqueeze_874, unsqueeze_886, unsqueeze_898, unsqueeze_910, unsqueeze_922, le_27, unsqueeze_934, unsqueeze_946, unsqueeze_958, unsqueeze_970, bitwise_and_5, le_30, unsqueeze_982, unsqueeze_994, unsqueeze_1006, unsqueeze_1018, unsqueeze_1030, unsqueeze_1042, bitwise_and_6, unsqueeze_1054, le_33, unsqueeze_1066, unsqueeze_1078, unsqueeze_1090, unsqueeze_1102, le_35, unsqueeze_1114, unsqueeze_1126, unsqueeze_1138, unsqueeze_1150, unsqueeze_1162, unsqueeze_1174, unsqueeze_1186, le_37, unsqueeze_1198, unsqueeze_1210, unsqueeze_1222, unsqueeze_1234, le_39, unsqueeze_1246, unsqueeze_1258, unsqueeze_1270]
    