from __future__ import annotations



def forward(self, primals_1: "f32[32, 3, 3, 3]", primals_2: "f32[32]", primals_3: "f32[32]", primals_4: "f32[32]", primals_5: "f32[32]", primals_6: "f32[32]", primals_7: "f32[32]", primals_8: "f32[192]", primals_9: "f32[192]", primals_10: "f32[64, 1, 3, 3]", primals_11: "f32[64, 1, 5, 5]", primals_12: "f32[64, 1, 7, 7]", primals_13: "f32[192]", primals_14: "f32[192]", primals_15: "f32[40]", primals_16: "f32[40]", primals_17: "f32[120]", primals_18: "f32[120]", primals_19: "f32[120]", primals_20: "f32[120]", primals_21: "f32[40]", primals_22: "f32[40]", primals_23: "f32[240]", primals_24: "f32[240]", primals_25: "f32[60, 1, 3, 3]", primals_26: "f32[60, 1, 5, 5]", primals_27: "f32[60, 1, 7, 7]", primals_28: "f32[60, 1, 9, 9]", primals_29: "f32[240]", primals_30: "f32[240]", primals_31: "f32[56]", primals_32: "f32[56]", primals_33: "f32[336]", primals_34: "f32[336]", primals_35: "f32[336]", primals_36: "f32[336]", primals_37: "f32[56]", primals_38: "f32[56]", primals_39: "f32[336]", primals_40: "f32[336]", primals_41: "f32[336]", primals_42: "f32[336]", primals_43: "f32[56]", primals_44: "f32[56]", primals_45: "f32[336]", primals_46: "f32[336]", primals_47: "f32[336]", primals_48: "f32[336]", primals_49: "f32[56]", primals_50: "f32[56]", primals_51: "f32[336]", primals_52: "f32[336]", primals_53: "f32[112, 1, 3, 3]", primals_54: "f32[112, 1, 5, 5]", primals_55: "f32[112, 1, 7, 7]", primals_56: "f32[336]", primals_57: "f32[336]", primals_58: "f32[104]", primals_59: "f32[104]", primals_60: "f32[624]", primals_61: "f32[624]", primals_62: "f32[624]", primals_63: "f32[624]", primals_64: "f32[104]", primals_65: "f32[104]", primals_66: "f32[624]", primals_67: "f32[624]", primals_68: "f32[624]", primals_69: "f32[624]", primals_70: "f32[104]", primals_71: "f32[104]", primals_72: "f32[624]", primals_73: "f32[624]", primals_74: "f32[624]", primals_75: "f32[624]", primals_76: "f32[104]", primals_77: "f32[104]", primals_78: "f32[624]", primals_79: "f32[624]", primals_80: "f32[624]", primals_81: "f32[624]", primals_82: "f32[160]", primals_83: "f32[160]", primals_84: "f32[480]", primals_85: "f32[480]", primals_86: "f32[480]", primals_87: "f32[480]", primals_88: "f32[160]", primals_89: "f32[160]", primals_90: "f32[480]", primals_91: "f32[480]", primals_92: "f32[480]", primals_93: "f32[480]", primals_94: "f32[160]", primals_95: "f32[160]", primals_96: "f32[480]", primals_97: "f32[480]", primals_98: "f32[480]", primals_99: "f32[480]", primals_100: "f32[160]", primals_101: "f32[160]", primals_102: "f32[960]", primals_103: "f32[960]", primals_104: "f32[240, 1, 3, 3]", primals_105: "f32[240, 1, 5, 5]", primals_106: "f32[240, 1, 7, 7]", primals_107: "f32[240, 1, 9, 9]", primals_108: "f32[960]", primals_109: "f32[960]", primals_110: "f32[264]", primals_111: "f32[264]", primals_112: "f32[1584]", primals_113: "f32[1584]", primals_114: "f32[1584]", primals_115: "f32[1584]", primals_116: "f32[264]", primals_117: "f32[264]", primals_118: "f32[1584]", primals_119: "f32[1584]", primals_120: "f32[1584]", primals_121: "f32[1584]", primals_122: "f32[264]", primals_123: "f32[264]", primals_124: "f32[1584]", primals_125: "f32[1584]", primals_126: "f32[1584]", primals_127: "f32[1584]", primals_128: "f32[264]", primals_129: "f32[264]", primals_130: "f32[1536]", primals_131: "f32[1536]", primals_132: "f32[32, 1, 3, 3]", primals_133: "f32[32, 32, 1, 1]", primals_134: "f32[96, 16, 1, 1]", primals_135: "f32[96, 16, 1, 1]", primals_136: "f32[20, 96, 1, 1]", primals_137: "f32[20, 96, 1, 1]", primals_138: "f32[60, 20, 1, 1]", primals_139: "f32[60, 20, 1, 1]", primals_140: "f32[120, 1, 3, 3]", primals_141: "f32[20, 60, 1, 1]", primals_142: "f32[20, 60, 1, 1]", primals_143: "f32[240, 40, 1, 1]", primals_144: "f32[20, 240, 1, 1]", primals_145: "f32[20]", primals_146: "f32[240, 20, 1, 1]", primals_147: "f32[240]", primals_148: "f32[56, 240, 1, 1]", primals_149: "f32[168, 28, 1, 1]", primals_150: "f32[168, 28, 1, 1]", primals_151: "f32[168, 1, 3, 3]", primals_152: "f32[168, 1, 5, 5]", primals_153: "f32[28, 336, 1, 1]", primals_154: "f32[28]", primals_155: "f32[336, 28, 1, 1]", primals_156: "f32[336]", primals_157: "f32[28, 168, 1, 1]", primals_158: "f32[28, 168, 1, 1]", primals_159: "f32[168, 28, 1, 1]", primals_160: "f32[168, 28, 1, 1]", primals_161: "f32[168, 1, 3, 3]", primals_162: "f32[168, 1, 5, 5]", primals_163: "f32[28, 336, 1, 1]", primals_164: "f32[28]", primals_165: "f32[336, 28, 1, 1]", primals_166: "f32[336]", primals_167: "f32[28, 168, 1, 1]", primals_168: "f32[28, 168, 1, 1]", primals_169: "f32[168, 28, 1, 1]", primals_170: "f32[168, 28, 1, 1]", primals_171: "f32[168, 1, 3, 3]", primals_172: "f32[168, 1, 5, 5]", primals_173: "f32[28, 336, 1, 1]", primals_174: "f32[28]", primals_175: "f32[336, 28, 1, 1]", primals_176: "f32[336]", primals_177: "f32[28, 168, 1, 1]", primals_178: "f32[28, 168, 1, 1]", primals_179: "f32[336, 56, 1, 1]", primals_180: "f32[14, 336, 1, 1]", primals_181: "f32[14]", primals_182: "f32[336, 14, 1, 1]", primals_183: "f32[336]", primals_184: "f32[104, 336, 1, 1]", primals_185: "f32[312, 52, 1, 1]", primals_186: "f32[312, 52, 1, 1]", primals_187: "f32[156, 1, 3, 3]", primals_188: "f32[156, 1, 5, 5]", primals_189: "f32[156, 1, 7, 7]", primals_190: "f32[156, 1, 9, 9]", primals_191: "f32[26, 624, 1, 1]", primals_192: "f32[26]", primals_193: "f32[624, 26, 1, 1]", primals_194: "f32[624]", primals_195: "f32[52, 312, 1, 1]", primals_196: "f32[52, 312, 1, 1]", primals_197: "f32[312, 52, 1, 1]", primals_198: "f32[312, 52, 1, 1]", primals_199: "f32[156, 1, 3, 3]", primals_200: "f32[156, 1, 5, 5]", primals_201: "f32[156, 1, 7, 7]", primals_202: "f32[156, 1, 9, 9]", primals_203: "f32[26, 624, 1, 1]", primals_204: "f32[26]", primals_205: "f32[624, 26, 1, 1]", primals_206: "f32[624]", primals_207: "f32[52, 312, 1, 1]", primals_208: "f32[52, 312, 1, 1]", primals_209: "f32[312, 52, 1, 1]", primals_210: "f32[312, 52, 1, 1]", primals_211: "f32[156, 1, 3, 3]", primals_212: "f32[156, 1, 5, 5]", primals_213: "f32[156, 1, 7, 7]", primals_214: "f32[156, 1, 9, 9]", primals_215: "f32[26, 624, 1, 1]", primals_216: "f32[26]", primals_217: "f32[624, 26, 1, 1]", primals_218: "f32[624]", primals_219: "f32[52, 312, 1, 1]", primals_220: "f32[52, 312, 1, 1]", primals_221: "f32[624, 104, 1, 1]", primals_222: "f32[624, 1, 3, 3]", primals_223: "f32[52, 624, 1, 1]", primals_224: "f32[52]", primals_225: "f32[624, 52, 1, 1]", primals_226: "f32[624]", primals_227: "f32[160, 624, 1, 1]", primals_228: "f32[240, 80, 1, 1]", primals_229: "f32[240, 80, 1, 1]", primals_230: "f32[120, 1, 3, 3]", primals_231: "f32[120, 1, 5, 5]", primals_232: "f32[120, 1, 7, 7]", primals_233: "f32[120, 1, 9, 9]", primals_234: "f32[80, 480, 1, 1]", primals_235: "f32[80]", primals_236: "f32[480, 80, 1, 1]", primals_237: "f32[480]", primals_238: "f32[80, 240, 1, 1]", primals_239: "f32[80, 240, 1, 1]", primals_240: "f32[240, 80, 1, 1]", primals_241: "f32[240, 80, 1, 1]", primals_242: "f32[120, 1, 3, 3]", primals_243: "f32[120, 1, 5, 5]", primals_244: "f32[120, 1, 7, 7]", primals_245: "f32[120, 1, 9, 9]", primals_246: "f32[80, 480, 1, 1]", primals_247: "f32[80]", primals_248: "f32[480, 80, 1, 1]", primals_249: "f32[480]", primals_250: "f32[80, 240, 1, 1]", primals_251: "f32[80, 240, 1, 1]", primals_252: "f32[240, 80, 1, 1]", primals_253: "f32[240, 80, 1, 1]", primals_254: "f32[120, 1, 3, 3]", primals_255: "f32[120, 1, 5, 5]", primals_256: "f32[120, 1, 7, 7]", primals_257: "f32[120, 1, 9, 9]", primals_258: "f32[80, 480, 1, 1]", primals_259: "f32[80]", primals_260: "f32[480, 80, 1, 1]", primals_261: "f32[480]", primals_262: "f32[80, 240, 1, 1]", primals_263: "f32[80, 240, 1, 1]", primals_264: "f32[960, 160, 1, 1]", primals_265: "f32[80, 960, 1, 1]", primals_266: "f32[80]", primals_267: "f32[960, 80, 1, 1]", primals_268: "f32[960]", primals_269: "f32[264, 960, 1, 1]", primals_270: "f32[1584, 264, 1, 1]", primals_271: "f32[396, 1, 3, 3]", primals_272: "f32[396, 1, 5, 5]", primals_273: "f32[396, 1, 7, 7]", primals_274: "f32[396, 1, 9, 9]", primals_275: "f32[132, 1584, 1, 1]", primals_276: "f32[132]", primals_277: "f32[1584, 132, 1, 1]", primals_278: "f32[1584]", primals_279: "f32[132, 792, 1, 1]", primals_280: "f32[132, 792, 1, 1]", primals_281: "f32[1584, 264, 1, 1]", primals_282: "f32[396, 1, 3, 3]", primals_283: "f32[396, 1, 5, 5]", primals_284: "f32[396, 1, 7, 7]", primals_285: "f32[396, 1, 9, 9]", primals_286: "f32[132, 1584, 1, 1]", primals_287: "f32[132]", primals_288: "f32[1584, 132, 1, 1]", primals_289: "f32[1584]", primals_290: "f32[132, 792, 1, 1]", primals_291: "f32[132, 792, 1, 1]", primals_292: "f32[1584, 264, 1, 1]", primals_293: "f32[396, 1, 3, 3]", primals_294: "f32[396, 1, 5, 5]", primals_295: "f32[396, 1, 7, 7]", primals_296: "f32[396, 1, 9, 9]", primals_297: "f32[132, 1584, 1, 1]", primals_298: "f32[132]", primals_299: "f32[1584, 132, 1, 1]", primals_300: "f32[1584]", primals_301: "f32[132, 792, 1, 1]", primals_302: "f32[132, 792, 1, 1]", primals_303: "f32[1536, 264, 1, 1]", primals_304: "f32[1000, 1536]", primals_305: "f32[1000]", primals_306: "i64[]", primals_307: "f32[32]", primals_308: "f32[32]", primals_309: "i64[]", primals_310: "f32[32]", primals_311: "f32[32]", primals_312: "i64[]", primals_313: "f32[32]", primals_314: "f32[32]", primals_315: "i64[]", primals_316: "f32[192]", primals_317: "f32[192]", primals_318: "i64[]", primals_319: "f32[192]", primals_320: "f32[192]", primals_321: "i64[]", primals_322: "f32[40]", primals_323: "f32[40]", primals_324: "i64[]", primals_325: "f32[120]", primals_326: "f32[120]", primals_327: "i64[]", primals_328: "f32[120]", primals_329: "f32[120]", primals_330: "i64[]", primals_331: "f32[40]", primals_332: "f32[40]", primals_333: "i64[]", primals_334: "f32[240]", primals_335: "f32[240]", primals_336: "i64[]", primals_337: "f32[240]", primals_338: "f32[240]", primals_339: "i64[]", primals_340: "f32[56]", primals_341: "f32[56]", primals_342: "i64[]", primals_343: "f32[336]", primals_344: "f32[336]", primals_345: "i64[]", primals_346: "f32[336]", primals_347: "f32[336]", primals_348: "i64[]", primals_349: "f32[56]", primals_350: "f32[56]", primals_351: "i64[]", primals_352: "f32[336]", primals_353: "f32[336]", primals_354: "i64[]", primals_355: "f32[336]", primals_356: "f32[336]", primals_357: "i64[]", primals_358: "f32[56]", primals_359: "f32[56]", primals_360: "i64[]", primals_361: "f32[336]", primals_362: "f32[336]", primals_363: "i64[]", primals_364: "f32[336]", primals_365: "f32[336]", primals_366: "i64[]", primals_367: "f32[56]", primals_368: "f32[56]", primals_369: "i64[]", primals_370: "f32[336]", primals_371: "f32[336]", primals_372: "i64[]", primals_373: "f32[336]", primals_374: "f32[336]", primals_375: "i64[]", primals_376: "f32[104]", primals_377: "f32[104]", primals_378: "i64[]", primals_379: "f32[624]", primals_380: "f32[624]", primals_381: "i64[]", primals_382: "f32[624]", primals_383: "f32[624]", primals_384: "i64[]", primals_385: "f32[104]", primals_386: "f32[104]", primals_387: "i64[]", primals_388: "f32[624]", primals_389: "f32[624]", primals_390: "i64[]", primals_391: "f32[624]", primals_392: "f32[624]", primals_393: "i64[]", primals_394: "f32[104]", primals_395: "f32[104]", primals_396: "i64[]", primals_397: "f32[624]", primals_398: "f32[624]", primals_399: "i64[]", primals_400: "f32[624]", primals_401: "f32[624]", primals_402: "i64[]", primals_403: "f32[104]", primals_404: "f32[104]", primals_405: "i64[]", primals_406: "f32[624]", primals_407: "f32[624]", primals_408: "i64[]", primals_409: "f32[624]", primals_410: "f32[624]", primals_411: "i64[]", primals_412: "f32[160]", primals_413: "f32[160]", primals_414: "i64[]", primals_415: "f32[480]", primals_416: "f32[480]", primals_417: "i64[]", primals_418: "f32[480]", primals_419: "f32[480]", primals_420: "i64[]", primals_421: "f32[160]", primals_422: "f32[160]", primals_423: "i64[]", primals_424: "f32[480]", primals_425: "f32[480]", primals_426: "i64[]", primals_427: "f32[480]", primals_428: "f32[480]", primals_429: "i64[]", primals_430: "f32[160]", primals_431: "f32[160]", primals_432: "i64[]", primals_433: "f32[480]", primals_434: "f32[480]", primals_435: "i64[]", primals_436: "f32[480]", primals_437: "f32[480]", primals_438: "i64[]", primals_439: "f32[160]", primals_440: "f32[160]", primals_441: "i64[]", primals_442: "f32[960]", primals_443: "f32[960]", primals_444: "i64[]", primals_445: "f32[960]", primals_446: "f32[960]", primals_447: "i64[]", primals_448: "f32[264]", primals_449: "f32[264]", primals_450: "i64[]", primals_451: "f32[1584]", primals_452: "f32[1584]", primals_453: "i64[]", primals_454: "f32[1584]", primals_455: "f32[1584]", primals_456: "i64[]", primals_457: "f32[264]", primals_458: "f32[264]", primals_459: "i64[]", primals_460: "f32[1584]", primals_461: "f32[1584]", primals_462: "i64[]", primals_463: "f32[1584]", primals_464: "f32[1584]", primals_465: "i64[]", primals_466: "f32[264]", primals_467: "f32[264]", primals_468: "i64[]", primals_469: "f32[1584]", primals_470: "f32[1584]", primals_471: "i64[]", primals_472: "f32[1584]", primals_473: "f32[1584]", primals_474: "i64[]", primals_475: "f32[264]", primals_476: "f32[264]", primals_477: "i64[]", primals_478: "f32[1536]", primals_479: "f32[1536]", primals_480: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    constant_pad_nd: "f32[8, 3, 225, 225]" = torch.ops.aten.constant_pad_nd.default(primals_480, [0, 1, 0, 1], 0.0);  primals_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(constant_pad_nd, primals_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 32, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
    rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_132, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
    rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_7: "f32[32]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_8: "f32[32]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1);  primals_5 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_2: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(relu_1, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_312, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 32, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 32, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 0.001)
    rsqrt_2: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[32]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
    add_12: "f32[32]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
    mul_18: "f32[32]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[32]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_13: "f32[32]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
    unsqueeze_9: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
    unsqueeze_11: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:129, code: x = self.drop_path(x) + shortcut
    add_15: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(add_14, relu);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(add_15, [16, 16], 1);  add_15 = None
    getitem_6: "f32[8, 16, 112, 112]" = split_with_sizes[0]
    getitem_7: "f32[8, 16, 112, 112]" = split_with_sizes[1];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_3: "f32[8, 96, 112, 112]" = torch.ops.aten.convolution.default(getitem_6, primals_134, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_4: "f32[8, 96, 112, 112]" = torch.ops.aten.convolution.default(getitem_7, primals_135, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat: "f32[8, 192, 112, 112]" = torch.ops.aten.cat.default([convolution_3, convolution_4], 1);  convolution_3 = convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_16: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(cat, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 192, 1, 1]" = var_mean_3[0]
    getitem_9: "f32[1, 192, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_17: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 0.001)
    rsqrt_3: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_3: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(cat, getitem_9)
    mul_21: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_10: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[192]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
    add_18: "f32[192]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_24: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
    mul_25: "f32[192]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[192]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_19: "f32[192]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_13: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_15: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_20: "f32[8, 192, 112, 112]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[8, 192, 112, 112]" = torch.ops.aten.relu.default(add_20);  add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(relu_2, [64, 64, 64], 1)
    getitem_13: "f32[8, 64, 112, 112]" = split_with_sizes_2[0]
    constant_pad_nd_1: "f32[8, 64, 113, 113]" = torch.ops.aten.constant_pad_nd.default(getitem_13, [0, 1, 0, 1], 0.0);  getitem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_5: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(constant_pad_nd_1, primals_10, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    getitem_17: "f32[8, 64, 112, 112]" = split_with_sizes_2[1]
    constant_pad_nd_2: "f32[8, 64, 115, 115]" = torch.ops.aten.constant_pad_nd.default(getitem_17, [1, 2, 1, 2], 0.0);  getitem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_6: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(constant_pad_nd_2, primals_11, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    getitem_21: "f32[8, 64, 112, 112]" = split_with_sizes_2[2];  split_with_sizes_2 = None
    constant_pad_nd_3: "f32[8, 64, 117, 117]" = torch.ops.aten.constant_pad_nd.default(getitem_21, [2, 3, 2, 3], 0.0);  getitem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_7: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(constant_pad_nd_3, primals_12, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_1: "f32[8, 192, 56, 56]" = torch.ops.aten.cat.default([convolution_5, convolution_6, convolution_7], 1);  convolution_5 = convolution_6 = convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_21: "i64[]" = torch.ops.aten.add.Tensor(primals_318, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(cat_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 192, 1, 1]" = var_mean_4[0]
    getitem_23: "f32[1, 192, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_22: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 0.001)
    rsqrt_4: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_4: "f32[8, 192, 56, 56]" = torch.ops.aten.sub.Tensor(cat_1, getitem_23)
    mul_28: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_13: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[192]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_23: "f32[192]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_31: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_32: "f32[192]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[192]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
    add_24: "f32[192]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_17: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 192, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_19: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_25: "f32[8, 192, 56, 56]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_3: "f32[8, 192, 56, 56]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(relu_3, [96, 96], 1)
    getitem_26: "f32[8, 96, 56, 56]" = split_with_sizes_6[0]
    convolution_8: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_26, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    getitem_29: "f32[8, 96, 56, 56]" = split_with_sizes_6[1];  split_with_sizes_6 = None
    convolution_9: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_29, primals_137, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_2: "f32[8, 40, 56, 56]" = torch.ops.aten.cat.default([convolution_8, convolution_9], 1);  convolution_8 = convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_321, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(cat_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 40, 1, 1]" = var_mean_5[0]
    getitem_31: "f32[1, 40, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 0.001)
    rsqrt_5: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_5: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_2, getitem_31)
    mul_35: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_16: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[40]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_28: "f32[40]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_38: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_39: "f32[40]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[40]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
    add_29: "f32[40]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_21: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_23: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_30: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(add_30, [20, 20], 1)
    getitem_32: "f32[8, 20, 56, 56]" = split_with_sizes_8[0]
    getitem_33: "f32[8, 20, 56, 56]" = split_with_sizes_8[1];  split_with_sizes_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_10: "f32[8, 60, 56, 56]" = torch.ops.aten.convolution.default(getitem_32, primals_138, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_11: "f32[8, 60, 56, 56]" = torch.ops.aten.convolution.default(getitem_33, primals_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_3: "f32[8, 120, 56, 56]" = torch.ops.aten.cat.default([convolution_10, convolution_11], 1);  convolution_10 = convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_324, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(cat_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 120, 1, 1]" = var_mean_6[0]
    getitem_35: "f32[1, 120, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_32: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 0.001)
    rsqrt_6: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_6: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(cat_3, getitem_35)
    mul_42: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_19: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[120]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_33: "f32[120]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_45: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_46: "f32[120]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[120]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
    add_34: "f32[120]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_25: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_27: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_35: "f32[8, 120, 56, 56]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_4: "f32[8, 120, 56, 56]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_12: "f32[8, 120, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_140, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_327, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 120, 1, 1]" = var_mean_7[0]
    getitem_37: "f32[1, 120, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 0.001)
    rsqrt_7: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_37)
    mul_49: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_22: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[120]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_38: "f32[120]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_52: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_53: "f32[120]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[120]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
    add_39: "f32[120]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_29: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_31: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 120, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[8, 120, 56, 56]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(relu_5, [60, 60], 1)
    getitem_40: "f32[8, 60, 56, 56]" = split_with_sizes_10[0]
    convolution_13: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_40, primals_141, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    getitem_43: "f32[8, 60, 56, 56]" = split_with_sizes_10[1];  split_with_sizes_10 = None
    convolution_14: "f32[8, 20, 56, 56]" = torch.ops.aten.convolution.default(getitem_43, primals_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_4: "f32[8, 40, 56, 56]" = torch.ops.aten.cat.default([convolution_13, convolution_14], 1);  convolution_13 = convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_330, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(cat_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 40, 1, 1]" = var_mean_8[0]
    getitem_45: "f32[1, 40, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 0.001)
    rsqrt_8: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 40, 56, 56]" = torch.ops.aten.sub.Tensor(cat_4, getitem_45)
    mul_56: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_25: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[40]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_43: "f32[40]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_59: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_60: "f32[40]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[40]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
    add_44: "f32[40]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_33: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 40, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_35: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_46: "f32[8, 40, 56, 56]" = torch.ops.aten.add.Tensor(add_45, add_30);  add_45 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_15: "f32[8, 240, 56, 56]" = torch.ops.aten.convolution.default(add_46, primals_143, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_333, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 240, 1, 1]" = var_mean_9[0]
    getitem_47: "f32[1, 240, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_48: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 0.001)
    rsqrt_9: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_9: "f32[8, 240, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_47)
    mul_63: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_28: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[240]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_49: "f32[240]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_66: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_67: "f32[240]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[240]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
    add_50: "f32[240]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_37: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_39: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_51: "f32[8, 240, 56, 56]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid: "f32[8, 240, 56, 56]" = torch.ops.aten.sigmoid.default(add_51)
    mul_70: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(add_51, sigmoid);  sigmoid = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(mul_70, [60, 60, 60, 60], 1);  mul_70 = None
    getitem_52: "f32[8, 60, 56, 56]" = split_with_sizes_13[0]
    constant_pad_nd_4: "f32[8, 60, 57, 57]" = torch.ops.aten.constant_pad_nd.default(getitem_52, [0, 1, 0, 1], 0.0);  getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_16: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_4, primals_25, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    getitem_57: "f32[8, 60, 56, 56]" = split_with_sizes_13[1]
    constant_pad_nd_5: "f32[8, 60, 59, 59]" = torch.ops.aten.constant_pad_nd.default(getitem_57, [1, 2, 1, 2], 0.0);  getitem_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_17: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_5, primals_26, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    getitem_62: "f32[8, 60, 56, 56]" = split_with_sizes_13[2]
    constant_pad_nd_6: "f32[8, 60, 61, 61]" = torch.ops.aten.constant_pad_nd.default(getitem_62, [2, 3, 2, 3], 0.0);  getitem_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_18: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_6, primals_27, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    getitem_67: "f32[8, 60, 56, 56]" = split_with_sizes_13[3];  split_with_sizes_13 = None
    constant_pad_nd_7: "f32[8, 60, 63, 63]" = torch.ops.aten.constant_pad_nd.default(getitem_67, [3, 4, 3, 4], 0.0);  getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_19: "f32[8, 60, 28, 28]" = torch.ops.aten.convolution.default(constant_pad_nd_7, primals_28, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_5: "f32[8, 240, 28, 28]" = torch.ops.aten.cat.default([convolution_16, convolution_17, convolution_18, convolution_19], 1);  convolution_16 = convolution_17 = convolution_18 = convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_336, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(cat_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 240, 1, 1]" = var_mean_10[0]
    getitem_69: "f32[1, 240, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_53: "f32[1, 240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 0.001)
    rsqrt_10: "f32[1, 240, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_10: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(cat_5, getitem_69)
    mul_71: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_31: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_72: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_73: "f32[240]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_54: "f32[240]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
    squeeze_32: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_74: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0001594642002871);  squeeze_32 = None
    mul_75: "f32[240]" = torch.ops.aten.mul.Tensor(mul_74, 0.1);  mul_74 = None
    mul_76: "f32[240]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_55: "f32[240]" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
    unsqueeze_40: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_41: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_77: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_41);  mul_71 = unsqueeze_41 = None
    unsqueeze_42: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_43: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_56: "f32[8, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_43);  mul_77 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_1: "f32[8, 240, 28, 28]" = torch.ops.aten.sigmoid.default(add_56)
    mul_78: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_56, sigmoid_1);  sigmoid_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 240, 1, 1]" = torch.ops.aten.mean.dim(mul_78, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_20: "f32[8, 20, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_144, primals_145, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_2: "f32[8, 20, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_20)
    mul_79: "f32[8, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_20, sigmoid_2);  sigmoid_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_21: "f32[8, 240, 1, 1]" = torch.ops.aten.convolution.default(mul_79, primals_146, primals_147, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_3: "f32[8, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_21)
    mul_80: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_78, sigmoid_3);  mul_78 = sigmoid_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_22: "f32[8, 56, 28, 28]" = torch.ops.aten.convolution.default(mul_80, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_57: "i64[]" = torch.ops.aten.add.Tensor(primals_339, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 56, 1, 1]" = var_mean_11[0]
    getitem_71: "f32[1, 56, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_58: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 0.001)
    rsqrt_11: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_11: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_71)
    mul_81: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_34: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_82: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_83: "f32[56]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_59: "f32[56]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
    squeeze_35: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_84: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001594642002871);  squeeze_35 = None
    mul_85: "f32[56]" = torch.ops.aten.mul.Tensor(mul_84, 0.1);  mul_84 = None
    mul_86: "f32[56]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_60: "f32[56]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    unsqueeze_44: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_45: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_87: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_81, unsqueeze_45);  mul_81 = unsqueeze_45 = None
    unsqueeze_46: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_47: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_61: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_87, unsqueeze_47);  mul_87 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_17 = torch.ops.aten.split_with_sizes.default(add_61, [28, 28], 1)
    getitem_72: "f32[8, 28, 28, 28]" = split_with_sizes_17[0]
    getitem_73: "f32[8, 28, 28, 28]" = split_with_sizes_17[1];  split_with_sizes_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_23: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_72, primals_149, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_24: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_73, primals_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_6: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_23, convolution_24], 1);  convolution_23 = convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_342, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(cat_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 336, 1, 1]" = var_mean_12[0]
    getitem_75: "f32[1, 336, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 0.001)
    rsqrt_12: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_12: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_6, getitem_75)
    mul_88: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_37: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_89: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_90: "f32[336]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_64: "f32[336]" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
    squeeze_38: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_91: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001594642002871);  squeeze_38 = None
    mul_92: "f32[336]" = torch.ops.aten.mul.Tensor(mul_91, 0.1);  mul_91 = None
    mul_93: "f32[336]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_65: "f32[336]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    unsqueeze_48: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_49: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_94: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_49);  mul_88 = unsqueeze_49 = None
    unsqueeze_50: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_51: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_51);  mul_94 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_4: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_66)
    mul_95: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_66, sigmoid_4);  sigmoid_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_19 = torch.ops.aten.split_with_sizes.default(mul_95, [168, 168], 1);  mul_95 = None
    getitem_78: "f32[8, 168, 28, 28]" = split_with_sizes_19[0]
    convolution_25: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_78, primals_151, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168)
    getitem_81: "f32[8, 168, 28, 28]" = split_with_sizes_19[1];  split_with_sizes_19 = None
    convolution_26: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_81, primals_152, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_7: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_25, convolution_26], 1);  convolution_25 = convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_345, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(cat_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 336, 1, 1]" = var_mean_13[0]
    getitem_83: "f32[1, 336, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_68: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 0.001)
    rsqrt_13: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_13: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_7, getitem_83)
    mul_96: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_40: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_97: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_98: "f32[336]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
    add_69: "f32[336]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    squeeze_41: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_99: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001594642002871);  squeeze_41 = None
    mul_100: "f32[336]" = torch.ops.aten.mul.Tensor(mul_99, 0.1);  mul_99 = None
    mul_101: "f32[336]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
    add_70: "f32[336]" = torch.ops.aten.add.Tensor(mul_100, mul_101);  mul_100 = mul_101 = None
    unsqueeze_52: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_53: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_102: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_96, unsqueeze_53);  mul_96 = unsqueeze_53 = None
    unsqueeze_54: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_55: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_71: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_102, unsqueeze_55);  mul_102 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_5: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_71)
    mul_103: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_71, sigmoid_5);  sigmoid_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_103, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_27: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_153, primals_154, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_6: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_27)
    mul_104: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_27, sigmoid_6);  sigmoid_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_28: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_104, primals_155, primals_156, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_7: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_28)
    mul_105: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_103, sigmoid_7);  mul_103 = sigmoid_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_21 = torch.ops.aten.split_with_sizes.default(mul_105, [168, 168], 1);  mul_105 = None
    getitem_84: "f32[8, 168, 28, 28]" = split_with_sizes_21[0]
    getitem_85: "f32[8, 168, 28, 28]" = split_with_sizes_21[1];  split_with_sizes_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_29: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_84, primals_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_30: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_85, primals_158, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_8: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([convolution_29, convolution_30], 1);  convolution_29 = convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_72: "i64[]" = torch.ops.aten.add.Tensor(primals_348, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(cat_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 56, 1, 1]" = var_mean_14[0]
    getitem_87: "f32[1, 56, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_73: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 0.001)
    rsqrt_14: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_14: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_8, getitem_87)
    mul_106: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_43: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_107: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_108: "f32[56]" = torch.ops.aten.mul.Tensor(primals_349, 0.9)
    add_74: "f32[56]" = torch.ops.aten.add.Tensor(mul_107, mul_108);  mul_107 = mul_108 = None
    squeeze_44: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_109: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001594642002871);  squeeze_44 = None
    mul_110: "f32[56]" = torch.ops.aten.mul.Tensor(mul_109, 0.1);  mul_109 = None
    mul_111: "f32[56]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
    add_75: "f32[56]" = torch.ops.aten.add.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
    unsqueeze_56: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_57: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_112: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_57);  mul_106 = unsqueeze_57 = None
    unsqueeze_58: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_59: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_76: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_112, unsqueeze_59);  mul_112 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_77: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_76, add_61);  add_76 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_22 = torch.ops.aten.split_with_sizes.default(add_77, [28, 28], 1)
    getitem_88: "f32[8, 28, 28, 28]" = split_with_sizes_22[0]
    getitem_89: "f32[8, 28, 28, 28]" = split_with_sizes_22[1];  split_with_sizes_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_31: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_88, primals_159, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_32: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_89, primals_160, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_9: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_31, convolution_32], 1);  convolution_31 = convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_351, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(cat_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 336, 1, 1]" = var_mean_15[0]
    getitem_91: "f32[1, 336, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_79: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 0.001)
    rsqrt_15: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_15: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_9, getitem_91)
    mul_113: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_46: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_114: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_115: "f32[336]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
    add_80: "f32[336]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
    squeeze_47: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_116: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001594642002871);  squeeze_47 = None
    mul_117: "f32[336]" = torch.ops.aten.mul.Tensor(mul_116, 0.1);  mul_116 = None
    mul_118: "f32[336]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
    add_81: "f32[336]" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    unsqueeze_60: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_61: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_119: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_113, unsqueeze_61);  mul_113 = unsqueeze_61 = None
    unsqueeze_62: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_63: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_82: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_63);  mul_119 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_8: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_82)
    mul_120: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_82, sigmoid_8);  sigmoid_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_24 = torch.ops.aten.split_with_sizes.default(mul_120, [168, 168], 1);  mul_120 = None
    getitem_94: "f32[8, 168, 28, 28]" = split_with_sizes_24[0]
    convolution_33: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_94, primals_161, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168)
    getitem_97: "f32[8, 168, 28, 28]" = split_with_sizes_24[1];  split_with_sizes_24 = None
    convolution_34: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_97, primals_162, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_10: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_33, convolution_34], 1);  convolution_33 = convolution_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_354, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(cat_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 336, 1, 1]" = var_mean_16[0]
    getitem_99: "f32[1, 336, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_84: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 0.001)
    rsqrt_16: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_16: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_10, getitem_99)
    mul_121: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_49: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_122: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_123: "f32[336]" = torch.ops.aten.mul.Tensor(primals_355, 0.9)
    add_85: "f32[336]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
    squeeze_50: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_124: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
    mul_125: "f32[336]" = torch.ops.aten.mul.Tensor(mul_124, 0.1);  mul_124 = None
    mul_126: "f32[336]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
    add_86: "f32[336]" = torch.ops.aten.add.Tensor(mul_125, mul_126);  mul_125 = mul_126 = None
    unsqueeze_64: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_65: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_127: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_65);  mul_121 = unsqueeze_65 = None
    unsqueeze_66: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_67: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_87: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_127, unsqueeze_67);  mul_127 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_9: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_87)
    mul_128: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_87, sigmoid_9);  sigmoid_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_128, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_35: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_163, primals_164, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_10: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_35)
    mul_129: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_35, sigmoid_10);  sigmoid_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_36: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_129, primals_165, primals_166, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_11: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36)
    mul_130: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_128, sigmoid_11);  mul_128 = sigmoid_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_26 = torch.ops.aten.split_with_sizes.default(mul_130, [168, 168], 1);  mul_130 = None
    getitem_100: "f32[8, 168, 28, 28]" = split_with_sizes_26[0]
    getitem_101: "f32[8, 168, 28, 28]" = split_with_sizes_26[1];  split_with_sizes_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_37: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_100, primals_167, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_38: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_101, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_11: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([convolution_37, convolution_38], 1);  convolution_37 = convolution_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_88: "i64[]" = torch.ops.aten.add.Tensor(primals_357, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(cat_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 56, 1, 1]" = var_mean_17[0]
    getitem_103: "f32[1, 56, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_89: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 0.001)
    rsqrt_17: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_17: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_11, getitem_103)
    mul_131: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_52: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_132: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_133: "f32[56]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
    add_90: "f32[56]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
    squeeze_53: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_134: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001594642002871);  squeeze_53 = None
    mul_135: "f32[56]" = torch.ops.aten.mul.Tensor(mul_134, 0.1);  mul_134 = None
    mul_136: "f32[56]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
    add_91: "f32[56]" = torch.ops.aten.add.Tensor(mul_135, mul_136);  mul_135 = mul_136 = None
    unsqueeze_68: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_69: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_137: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_69);  mul_131 = unsqueeze_69 = None
    unsqueeze_70: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_71: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_92: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_71);  mul_137 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_93: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_92, add_77);  add_92 = add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_27 = torch.ops.aten.split_with_sizes.default(add_93, [28, 28], 1)
    getitem_104: "f32[8, 28, 28, 28]" = split_with_sizes_27[0]
    getitem_105: "f32[8, 28, 28, 28]" = split_with_sizes_27[1];  split_with_sizes_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_39: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_104, primals_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_40: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_105, primals_170, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_12: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_39, convolution_40], 1);  convolution_39 = convolution_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_94: "i64[]" = torch.ops.aten.add.Tensor(primals_360, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(cat_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 336, 1, 1]" = var_mean_18[0]
    getitem_107: "f32[1, 336, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_95: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 0.001)
    rsqrt_18: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_18: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_12, getitem_107)
    mul_138: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_55: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_139: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_140: "f32[336]" = torch.ops.aten.mul.Tensor(primals_361, 0.9)
    add_96: "f32[336]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
    squeeze_56: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_141: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
    mul_142: "f32[336]" = torch.ops.aten.mul.Tensor(mul_141, 0.1);  mul_141 = None
    mul_143: "f32[336]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
    add_97: "f32[336]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    unsqueeze_72: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_73: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_144: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_73);  mul_138 = unsqueeze_73 = None
    unsqueeze_74: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_75: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_98: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_75);  mul_144 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_12: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_98)
    mul_145: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_98, sigmoid_12);  sigmoid_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_29 = torch.ops.aten.split_with_sizes.default(mul_145, [168, 168], 1);  mul_145 = None
    getitem_110: "f32[8, 168, 28, 28]" = split_with_sizes_29[0]
    convolution_41: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_110, primals_171, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 168)
    getitem_113: "f32[8, 168, 28, 28]" = split_with_sizes_29[1];  split_with_sizes_29 = None
    convolution_42: "f32[8, 168, 28, 28]" = torch.ops.aten.convolution.default(getitem_113, primals_172, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 168)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_13: "f32[8, 336, 28, 28]" = torch.ops.aten.cat.default([convolution_41, convolution_42], 1);  convolution_41 = convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_99: "i64[]" = torch.ops.aten.add.Tensor(primals_363, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(cat_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 336, 1, 1]" = var_mean_19[0]
    getitem_115: "f32[1, 336, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_100: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 0.001)
    rsqrt_19: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_19: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(cat_13, getitem_115)
    mul_146: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_58: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_147: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_148: "f32[336]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
    add_101: "f32[336]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
    squeeze_59: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_149: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
    mul_150: "f32[336]" = torch.ops.aten.mul.Tensor(mul_149, 0.1);  mul_149 = None
    mul_151: "f32[336]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
    add_102: "f32[336]" = torch.ops.aten.add.Tensor(mul_150, mul_151);  mul_150 = mul_151 = None
    unsqueeze_76: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_77: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_152: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_146, unsqueeze_77);  mul_146 = unsqueeze_77 = None
    unsqueeze_78: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_79: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_103: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_79);  mul_152 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_13: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_103)
    mul_153: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_103, sigmoid_13);  sigmoid_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_153, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_43: "f32[8, 28, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_173, primals_174, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_14: "f32[8, 28, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43)
    mul_154: "f32[8, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_43, sigmoid_14);  sigmoid_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_44: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_154, primals_175, primals_176, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_15: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_44)
    mul_155: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_153, sigmoid_15);  mul_153 = sigmoid_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_31 = torch.ops.aten.split_with_sizes.default(mul_155, [168, 168], 1);  mul_155 = None
    getitem_116: "f32[8, 168, 28, 28]" = split_with_sizes_31[0]
    getitem_117: "f32[8, 168, 28, 28]" = split_with_sizes_31[1];  split_with_sizes_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_45: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_116, primals_177, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_46: "f32[8, 28, 28, 28]" = torch.ops.aten.convolution.default(getitem_117, primals_178, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_14: "f32[8, 56, 28, 28]" = torch.ops.aten.cat.default([convolution_45, convolution_46], 1);  convolution_45 = convolution_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_104: "i64[]" = torch.ops.aten.add.Tensor(primals_366, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(cat_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 56, 1, 1]" = var_mean_20[0]
    getitem_119: "f32[1, 56, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_105: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 0.001)
    rsqrt_20: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_20: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(cat_14, getitem_119)
    mul_156: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_61: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_157: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_158: "f32[56]" = torch.ops.aten.mul.Tensor(primals_367, 0.9)
    add_106: "f32[56]" = torch.ops.aten.add.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
    squeeze_62: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_159: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
    mul_160: "f32[56]" = torch.ops.aten.mul.Tensor(mul_159, 0.1);  mul_159 = None
    mul_161: "f32[56]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
    add_107: "f32[56]" = torch.ops.aten.add.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None
    unsqueeze_80: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_81: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_162: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(mul_156, unsqueeze_81);  mul_156 = unsqueeze_81 = None
    unsqueeze_82: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_83: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_108: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_162, unsqueeze_83);  mul_162 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_109: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(add_108, add_93);  add_108 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_47: "f32[8, 336, 28, 28]" = torch.ops.aten.convolution.default(add_109, primals_179, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_110: "i64[]" = torch.ops.aten.add.Tensor(primals_369, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 336, 1, 1]" = var_mean_21[0]
    getitem_121: "f32[1, 336, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_111: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 0.001)
    rsqrt_21: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_21: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_121)
    mul_163: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_64: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_164: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_165: "f32[336]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
    add_112: "f32[336]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    squeeze_65: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_166: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
    mul_167: "f32[336]" = torch.ops.aten.mul.Tensor(mul_166, 0.1);  mul_166 = None
    mul_168: "f32[336]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
    add_113: "f32[336]" = torch.ops.aten.add.Tensor(mul_167, mul_168);  mul_167 = mul_168 = None
    unsqueeze_84: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_85: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_169: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_85);  mul_163 = unsqueeze_85 = None
    unsqueeze_86: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_87: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_114: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Tensor(mul_169, unsqueeze_87);  mul_169 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_16: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_114)
    mul_170: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_114, sigmoid_16);  sigmoid_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_33 = torch.ops.aten.split_with_sizes.default(mul_170, [112, 112, 112], 1);  mul_170 = None
    getitem_125: "f32[8, 112, 28, 28]" = split_with_sizes_33[0]
    constant_pad_nd_8: "f32[8, 112, 29, 29]" = torch.ops.aten.constant_pad_nd.default(getitem_125, [0, 1, 0, 1], 0.0);  getitem_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_48: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(constant_pad_nd_8, primals_53, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    getitem_129: "f32[8, 112, 28, 28]" = split_with_sizes_33[1]
    constant_pad_nd_9: "f32[8, 112, 31, 31]" = torch.ops.aten.constant_pad_nd.default(getitem_129, [1, 2, 1, 2], 0.0);  getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_49: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(constant_pad_nd_9, primals_54, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    getitem_133: "f32[8, 112, 28, 28]" = split_with_sizes_33[2];  split_with_sizes_33 = None
    constant_pad_nd_10: "f32[8, 112, 33, 33]" = torch.ops.aten.constant_pad_nd.default(getitem_133, [2, 3, 2, 3], 0.0);  getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_50: "f32[8, 112, 14, 14]" = torch.ops.aten.convolution.default(constant_pad_nd_10, primals_55, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 112)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_15: "f32[8, 336, 14, 14]" = torch.ops.aten.cat.default([convolution_48, convolution_49, convolution_50], 1);  convolution_48 = convolution_49 = convolution_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_372, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(cat_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[1, 336, 1, 1]" = var_mean_22[0]
    getitem_135: "f32[1, 336, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_116: "f32[1, 336, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 0.001)
    rsqrt_22: "f32[1, 336, 1, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_22: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(cat_15, getitem_135)
    mul_171: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
    squeeze_67: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_172: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_173: "f32[336]" = torch.ops.aten.mul.Tensor(primals_373, 0.9)
    add_117: "f32[336]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    squeeze_68: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
    mul_174: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0006381620931717);  squeeze_68 = None
    mul_175: "f32[336]" = torch.ops.aten.mul.Tensor(mul_174, 0.1);  mul_174 = None
    mul_176: "f32[336]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
    add_118: "f32[336]" = torch.ops.aten.add.Tensor(mul_175, mul_176);  mul_175 = mul_176 = None
    unsqueeze_88: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_89: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_177: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_89);  mul_171 = unsqueeze_89 = None
    unsqueeze_90: "f32[336, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_91: "f32[336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_119: "f32[8, 336, 14, 14]" = torch.ops.aten.add.Tensor(mul_177, unsqueeze_91);  mul_177 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_17: "f32[8, 336, 14, 14]" = torch.ops.aten.sigmoid.default(add_119)
    mul_178: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(add_119, sigmoid_17);  sigmoid_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 336, 1, 1]" = torch.ops.aten.mean.dim(mul_178, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_51: "f32[8, 14, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_180, primals_181, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_18: "f32[8, 14, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_51)
    mul_179: "f32[8, 14, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_51, sigmoid_18);  sigmoid_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_52: "f32[8, 336, 1, 1]" = torch.ops.aten.convolution.default(mul_179, primals_182, primals_183, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_19: "f32[8, 336, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52)
    mul_180: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(mul_178, sigmoid_19);  mul_178 = sigmoid_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_53: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(mul_180, primals_184, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_120: "i64[]" = torch.ops.aten.add.Tensor(primals_375, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_136: "f32[1, 104, 1, 1]" = var_mean_23[0]
    getitem_137: "f32[1, 104, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_121: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 0.001)
    rsqrt_23: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_23: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_137)
    mul_181: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
    squeeze_70: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_182: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_183: "f32[104]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
    add_122: "f32[104]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
    squeeze_71: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
    mul_184: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0006381620931717);  squeeze_71 = None
    mul_185: "f32[104]" = torch.ops.aten.mul.Tensor(mul_184, 0.1);  mul_184 = None
    mul_186: "f32[104]" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
    add_123: "f32[104]" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
    unsqueeze_92: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1)
    unsqueeze_93: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_187: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_93);  mul_181 = unsqueeze_93 = None
    unsqueeze_94: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1);  primals_59 = None
    unsqueeze_95: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_124: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_187, unsqueeze_95);  mul_187 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_36 = torch.ops.aten.split_with_sizes.default(add_124, [52, 52], 1)
    getitem_138: "f32[8, 52, 14, 14]" = split_with_sizes_36[0]
    getitem_139: "f32[8, 52, 14, 14]" = split_with_sizes_36[1];  split_with_sizes_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_54: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_138, primals_185, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_55: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_139, primals_186, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_16: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_54, convolution_55], 1);  convolution_54 = convolution_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_125: "i64[]" = torch.ops.aten.add.Tensor(primals_378, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(cat_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_140: "f32[1, 624, 1, 1]" = var_mean_24[0]
    getitem_141: "f32[1, 624, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_126: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 0.001)
    rsqrt_24: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_24: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_16, getitem_141)
    mul_188: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
    squeeze_73: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_189: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_190: "f32[624]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
    add_127: "f32[624]" = torch.ops.aten.add.Tensor(mul_189, mul_190);  mul_189 = mul_190 = None
    squeeze_74: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
    mul_191: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0006381620931717);  squeeze_74 = None
    mul_192: "f32[624]" = torch.ops.aten.mul.Tensor(mul_191, 0.1);  mul_191 = None
    mul_193: "f32[624]" = torch.ops.aten.mul.Tensor(primals_380, 0.9)
    add_128: "f32[624]" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
    unsqueeze_96: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1)
    unsqueeze_97: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_194: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_188, unsqueeze_97);  mul_188 = unsqueeze_97 = None
    unsqueeze_98: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
    unsqueeze_99: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_129: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_99);  mul_194 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_20: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_129)
    mul_195: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_129, sigmoid_20);  sigmoid_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_38 = torch.ops.aten.split_with_sizes.default(mul_195, [156, 156, 156, 156], 1);  mul_195 = None
    getitem_146: "f32[8, 156, 14, 14]" = split_with_sizes_38[0]
    convolution_56: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_146, primals_187, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156)
    getitem_151: "f32[8, 156, 14, 14]" = split_with_sizes_38[1]
    convolution_57: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_151, primals_188, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156)
    getitem_156: "f32[8, 156, 14, 14]" = split_with_sizes_38[2]
    convolution_58: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_156, primals_189, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156)
    getitem_161: "f32[8, 156, 14, 14]" = split_with_sizes_38[3];  split_with_sizes_38 = None
    convolution_59: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_161, primals_190, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_17: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_56, convolution_57, convolution_58, convolution_59], 1);  convolution_56 = convolution_57 = convolution_58 = convolution_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_130: "i64[]" = torch.ops.aten.add.Tensor(primals_381, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(cat_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_162: "f32[1, 624, 1, 1]" = var_mean_25[0]
    getitem_163: "f32[1, 624, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_131: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 0.001)
    rsqrt_25: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    sub_25: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_17, getitem_163)
    mul_196: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
    squeeze_76: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_197: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_198: "f32[624]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
    add_132: "f32[624]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_77: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
    mul_199: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0006381620931717);  squeeze_77 = None
    mul_200: "f32[624]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[624]" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
    add_133: "f32[624]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_100: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_101: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_202: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_101);  mul_196 = unsqueeze_101 = None
    unsqueeze_102: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_103: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_134: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_103);  mul_202 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_21: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_134)
    mul_203: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_134, sigmoid_21);  sigmoid_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_203, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_60: "f32[8, 26, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_191, primals_192, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_22: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_60)
    mul_204: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_60, sigmoid_22);  sigmoid_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_61: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_204, primals_193, primals_194, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_23: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_61)
    mul_205: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_203, sigmoid_23);  mul_203 = sigmoid_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_42 = torch.ops.aten.split_with_sizes.default(mul_205, [312, 312], 1);  mul_205 = None
    getitem_164: "f32[8, 312, 14, 14]" = split_with_sizes_42[0]
    getitem_165: "f32[8, 312, 14, 14]" = split_with_sizes_42[1];  split_with_sizes_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_62: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_164, primals_195, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_63: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_165, primals_196, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_18: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([convolution_62, convolution_63], 1);  convolution_62 = convolution_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_135: "i64[]" = torch.ops.aten.add.Tensor(primals_384, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(cat_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_166: "f32[1, 104, 1, 1]" = var_mean_26[0]
    getitem_167: "f32[1, 104, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_136: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_166, 0.001)
    rsqrt_26: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_26: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_18, getitem_167)
    mul_206: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
    squeeze_79: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_207: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_208: "f32[104]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
    add_137: "f32[104]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    squeeze_80: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
    mul_209: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0006381620931717);  squeeze_80 = None
    mul_210: "f32[104]" = torch.ops.aten.mul.Tensor(mul_209, 0.1);  mul_209 = None
    mul_211: "f32[104]" = torch.ops.aten.mul.Tensor(primals_386, 0.9)
    add_138: "f32[104]" = torch.ops.aten.add.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
    unsqueeze_104: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1)
    unsqueeze_105: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_212: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_206, unsqueeze_105);  mul_206 = unsqueeze_105 = None
    unsqueeze_106: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1);  primals_65 = None
    unsqueeze_107: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_139: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_107);  mul_212 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_140: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_139, add_124);  add_139 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_43 = torch.ops.aten.split_with_sizes.default(add_140, [52, 52], 1)
    getitem_168: "f32[8, 52, 14, 14]" = split_with_sizes_43[0]
    getitem_169: "f32[8, 52, 14, 14]" = split_with_sizes_43[1];  split_with_sizes_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_64: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_168, primals_197, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_65: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_169, primals_198, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_19: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_64, convolution_65], 1);  convolution_64 = convolution_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_141: "i64[]" = torch.ops.aten.add.Tensor(primals_387, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(cat_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_170: "f32[1, 624, 1, 1]" = var_mean_27[0]
    getitem_171: "f32[1, 624, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_142: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_170, 0.001)
    rsqrt_27: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    sub_27: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_19, getitem_171)
    mul_213: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
    squeeze_82: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_214: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_215: "f32[624]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
    add_143: "f32[624]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    squeeze_83: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_170, [0, 2, 3]);  getitem_170 = None
    mul_216: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0006381620931717);  squeeze_83 = None
    mul_217: "f32[624]" = torch.ops.aten.mul.Tensor(mul_216, 0.1);  mul_216 = None
    mul_218: "f32[624]" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
    add_144: "f32[624]" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
    unsqueeze_108: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1)
    unsqueeze_109: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_219: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_213, unsqueeze_109);  mul_213 = unsqueeze_109 = None
    unsqueeze_110: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
    unsqueeze_111: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_145: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_219, unsqueeze_111);  mul_219 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_24: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_145)
    mul_220: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_145, sigmoid_24);  sigmoid_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_45 = torch.ops.aten.split_with_sizes.default(mul_220, [156, 156, 156, 156], 1);  mul_220 = None
    getitem_176: "f32[8, 156, 14, 14]" = split_with_sizes_45[0]
    convolution_66: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_176, primals_199, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156)
    getitem_181: "f32[8, 156, 14, 14]" = split_with_sizes_45[1]
    convolution_67: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_181, primals_200, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156)
    getitem_186: "f32[8, 156, 14, 14]" = split_with_sizes_45[2]
    convolution_68: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_186, primals_201, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156)
    getitem_191: "f32[8, 156, 14, 14]" = split_with_sizes_45[3];  split_with_sizes_45 = None
    convolution_69: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_191, primals_202, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_20: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_66, convolution_67, convolution_68, convolution_69], 1);  convolution_66 = convolution_67 = convolution_68 = convolution_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_146: "i64[]" = torch.ops.aten.add.Tensor(primals_390, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(cat_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_192: "f32[1, 624, 1, 1]" = var_mean_28[0]
    getitem_193: "f32[1, 624, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_147: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_192, 0.001)
    rsqrt_28: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_28: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_20, getitem_193)
    mul_221: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
    squeeze_85: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_222: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_223: "f32[624]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
    add_148: "f32[624]" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
    squeeze_86: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_192, [0, 2, 3]);  getitem_192 = None
    mul_224: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0006381620931717);  squeeze_86 = None
    mul_225: "f32[624]" = torch.ops.aten.mul.Tensor(mul_224, 0.1);  mul_224 = None
    mul_226: "f32[624]" = torch.ops.aten.mul.Tensor(primals_392, 0.9)
    add_149: "f32[624]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    unsqueeze_112: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_113: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_227: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_221, unsqueeze_113);  mul_221 = unsqueeze_113 = None
    unsqueeze_114: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_115: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_150: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_227, unsqueeze_115);  mul_227 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_25: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_150)
    mul_228: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_150, sigmoid_25);  sigmoid_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_228, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_70: "f32[8, 26, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_203, primals_204, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_26: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_70)
    mul_229: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_70, sigmoid_26);  sigmoid_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_71: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_229, primals_205, primals_206, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_27: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_71)
    mul_230: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_228, sigmoid_27);  mul_228 = sigmoid_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_49 = torch.ops.aten.split_with_sizes.default(mul_230, [312, 312], 1);  mul_230 = None
    getitem_194: "f32[8, 312, 14, 14]" = split_with_sizes_49[0]
    getitem_195: "f32[8, 312, 14, 14]" = split_with_sizes_49[1];  split_with_sizes_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_72: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_194, primals_207, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_73: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_195, primals_208, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_21: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([convolution_72, convolution_73], 1);  convolution_72 = convolution_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_151: "i64[]" = torch.ops.aten.add.Tensor(primals_393, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(cat_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_196: "f32[1, 104, 1, 1]" = var_mean_29[0]
    getitem_197: "f32[1, 104, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_152: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_196, 0.001)
    rsqrt_29: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_29: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_21, getitem_197)
    mul_231: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_197, [0, 2, 3]);  getitem_197 = None
    squeeze_88: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_232: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_233: "f32[104]" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
    add_153: "f32[104]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_89: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_196, [0, 2, 3]);  getitem_196 = None
    mul_234: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0006381620931717);  squeeze_89 = None
    mul_235: "f32[104]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[104]" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
    add_154: "f32[104]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_116: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1)
    unsqueeze_117: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_237: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_117);  mul_231 = unsqueeze_117 = None
    unsqueeze_118: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1);  primals_71 = None
    unsqueeze_119: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_155: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_119);  mul_237 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_156: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_155, add_140);  add_155 = add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_50 = torch.ops.aten.split_with_sizes.default(add_156, [52, 52], 1)
    getitem_198: "f32[8, 52, 14, 14]" = split_with_sizes_50[0]
    getitem_199: "f32[8, 52, 14, 14]" = split_with_sizes_50[1];  split_with_sizes_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_74: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_198, primals_209, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_75: "f32[8, 312, 14, 14]" = torch.ops.aten.convolution.default(getitem_199, primals_210, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_22: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_74, convolution_75], 1);  convolution_74 = convolution_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_157: "i64[]" = torch.ops.aten.add.Tensor(primals_396, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(cat_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_200: "f32[1, 624, 1, 1]" = var_mean_30[0]
    getitem_201: "f32[1, 624, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_158: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_200, 0.001)
    rsqrt_30: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_30: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_22, getitem_201)
    mul_238: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_201, [0, 2, 3]);  getitem_201 = None
    squeeze_91: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_239: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_240: "f32[624]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
    add_159: "f32[624]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_92: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_200, [0, 2, 3]);  getitem_200 = None
    mul_241: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0006381620931717);  squeeze_92 = None
    mul_242: "f32[624]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[624]" = torch.ops.aten.mul.Tensor(primals_398, 0.9)
    add_160: "f32[624]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_120: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1)
    unsqueeze_121: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_244: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_121);  mul_238 = unsqueeze_121 = None
    unsqueeze_122: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
    unsqueeze_123: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_161: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_123);  mul_244 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_28: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_161)
    mul_245: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_161, sigmoid_28);  sigmoid_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_52 = torch.ops.aten.split_with_sizes.default(mul_245, [156, 156, 156, 156], 1);  mul_245 = None
    getitem_206: "f32[8, 156, 14, 14]" = split_with_sizes_52[0]
    convolution_76: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_206, primals_211, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 156)
    getitem_211: "f32[8, 156, 14, 14]" = split_with_sizes_52[1]
    convolution_77: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_211, primals_212, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 156)
    getitem_216: "f32[8, 156, 14, 14]" = split_with_sizes_52[2]
    convolution_78: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_216, primals_213, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 156)
    getitem_221: "f32[8, 156, 14, 14]" = split_with_sizes_52[3];  split_with_sizes_52 = None
    convolution_79: "f32[8, 156, 14, 14]" = torch.ops.aten.convolution.default(getitem_221, primals_214, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_23: "f32[8, 624, 14, 14]" = torch.ops.aten.cat.default([convolution_76, convolution_77, convolution_78, convolution_79], 1);  convolution_76 = convolution_77 = convolution_78 = convolution_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_162: "i64[]" = torch.ops.aten.add.Tensor(primals_399, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(cat_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_222: "f32[1, 624, 1, 1]" = var_mean_31[0]
    getitem_223: "f32[1, 624, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_163: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_222, 0.001)
    rsqrt_31: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_31: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(cat_23, getitem_223)
    mul_246: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_223, [0, 2, 3]);  getitem_223 = None
    squeeze_94: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_247: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_248: "f32[624]" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
    add_164: "f32[624]" = torch.ops.aten.add.Tensor(mul_247, mul_248);  mul_247 = mul_248 = None
    squeeze_95: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_222, [0, 2, 3]);  getitem_222 = None
    mul_249: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0006381620931717);  squeeze_95 = None
    mul_250: "f32[624]" = torch.ops.aten.mul.Tensor(mul_249, 0.1);  mul_249 = None
    mul_251: "f32[624]" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
    add_165: "f32[624]" = torch.ops.aten.add.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
    unsqueeze_124: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_125: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_252: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_246, unsqueeze_125);  mul_246 = unsqueeze_125 = None
    unsqueeze_126: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_127: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_166: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_252, unsqueeze_127);  mul_252 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_29: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_166)
    mul_253: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_166, sigmoid_29);  sigmoid_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_253, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_80: "f32[8, 26, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_215, primals_216, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_30: "f32[8, 26, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_80)
    mul_254: "f32[8, 26, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_80, sigmoid_30);  sigmoid_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_81: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_254, primals_217, primals_218, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_31: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_81)
    mul_255: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_253, sigmoid_31);  mul_253 = sigmoid_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_56 = torch.ops.aten.split_with_sizes.default(mul_255, [312, 312], 1);  mul_255 = None
    getitem_224: "f32[8, 312, 14, 14]" = split_with_sizes_56[0]
    getitem_225: "f32[8, 312, 14, 14]" = split_with_sizes_56[1];  split_with_sizes_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_82: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_224, primals_219, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_83: "f32[8, 52, 14, 14]" = torch.ops.aten.convolution.default(getitem_225, primals_220, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_24: "f32[8, 104, 14, 14]" = torch.ops.aten.cat.default([convolution_82, convolution_83], 1);  convolution_82 = convolution_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_167: "i64[]" = torch.ops.aten.add.Tensor(primals_402, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(cat_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_226: "f32[1, 104, 1, 1]" = var_mean_32[0]
    getitem_227: "f32[1, 104, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_168: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_226, 0.001)
    rsqrt_32: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_32: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(cat_24, getitem_227)
    mul_256: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_227, [0, 2, 3]);  getitem_227 = None
    squeeze_97: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_257: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_258: "f32[104]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
    add_169: "f32[104]" = torch.ops.aten.add.Tensor(mul_257, mul_258);  mul_257 = mul_258 = None
    squeeze_98: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_226, [0, 2, 3]);  getitem_226 = None
    mul_259: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0006381620931717);  squeeze_98 = None
    mul_260: "f32[104]" = torch.ops.aten.mul.Tensor(mul_259, 0.1);  mul_259 = None
    mul_261: "f32[104]" = torch.ops.aten.mul.Tensor(primals_404, 0.9)
    add_170: "f32[104]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    unsqueeze_128: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1)
    unsqueeze_129: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_262: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_129);  mul_256 = unsqueeze_129 = None
    unsqueeze_130: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1);  primals_77 = None
    unsqueeze_131: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_171: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_262, unsqueeze_131);  mul_262 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_172: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(add_171, add_156);  add_171 = add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_84: "f32[8, 624, 14, 14]" = torch.ops.aten.convolution.default(add_172, primals_221, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_173: "i64[]" = torch.ops.aten.add.Tensor(primals_405, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
    getitem_228: "f32[1, 624, 1, 1]" = var_mean_33[0]
    getitem_229: "f32[1, 624, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_174: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_228, 0.001)
    rsqrt_33: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    sub_33: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_229)
    mul_263: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_229, [0, 2, 3]);  getitem_229 = None
    squeeze_100: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_264: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_265: "f32[624]" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
    add_175: "f32[624]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
    squeeze_101: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_228, [0, 2, 3]);  getitem_228 = None
    mul_266: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0006381620931717);  squeeze_101 = None
    mul_267: "f32[624]" = torch.ops.aten.mul.Tensor(mul_266, 0.1);  mul_266 = None
    mul_268: "f32[624]" = torch.ops.aten.mul.Tensor(primals_407, 0.9)
    add_176: "f32[624]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    unsqueeze_132: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1)
    unsqueeze_133: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_269: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_133);  mul_263 = unsqueeze_133 = None
    unsqueeze_134: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
    unsqueeze_135: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_177: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_135);  mul_269 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_32: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_177)
    mul_270: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_177, sigmoid_32);  sigmoid_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_85: "f32[8, 624, 14, 14]" = torch.ops.aten.convolution.default(mul_270, primals_222, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 624)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_178: "i64[]" = torch.ops.aten.add.Tensor(primals_408, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True)
    getitem_230: "f32[1, 624, 1, 1]" = var_mean_34[0]
    getitem_231: "f32[1, 624, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_179: "f32[1, 624, 1, 1]" = torch.ops.aten.add.Tensor(getitem_230, 0.001)
    rsqrt_34: "f32[1, 624, 1, 1]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
    sub_34: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_231)
    mul_271: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_231, [0, 2, 3]);  getitem_231 = None
    squeeze_103: "f32[624]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_272: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_273: "f32[624]" = torch.ops.aten.mul.Tensor(primals_409, 0.9)
    add_180: "f32[624]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
    squeeze_104: "f32[624]" = torch.ops.aten.squeeze.dims(getitem_230, [0, 2, 3]);  getitem_230 = None
    mul_274: "f32[624]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0006381620931717);  squeeze_104 = None
    mul_275: "f32[624]" = torch.ops.aten.mul.Tensor(mul_274, 0.1);  mul_274 = None
    mul_276: "f32[624]" = torch.ops.aten.mul.Tensor(primals_410, 0.9)
    add_181: "f32[624]" = torch.ops.aten.add.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    unsqueeze_136: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_137: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_277: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_137);  mul_271 = unsqueeze_137 = None
    unsqueeze_138: "f32[624, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_139: "f32[624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_182: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Tensor(mul_277, unsqueeze_139);  mul_277 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_33: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_182)
    mul_278: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_182, sigmoid_33);  sigmoid_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[8, 624, 1, 1]" = torch.ops.aten.mean.dim(mul_278, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_86: "f32[8, 52, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_223, primals_224, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_34: "f32[8, 52, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_86)
    mul_279: "f32[8, 52, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_86, sigmoid_34);  sigmoid_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_87: "f32[8, 624, 1, 1]" = torch.ops.aten.convolution.default(mul_279, primals_225, primals_226, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_35: "f32[8, 624, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_87)
    mul_280: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(mul_278, sigmoid_35);  mul_278 = sigmoid_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_88: "f32[8, 160, 14, 14]" = torch.ops.aten.convolution.default(mul_280, primals_227, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_183: "i64[]" = torch.ops.aten.add.Tensor(primals_411, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
    getitem_232: "f32[1, 160, 1, 1]" = var_mean_35[0]
    getitem_233: "f32[1, 160, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_184: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_232, 0.001)
    rsqrt_35: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    sub_35: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_233)
    mul_281: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_233, [0, 2, 3]);  getitem_233 = None
    squeeze_106: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_282: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_283: "f32[160]" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
    add_185: "f32[160]" = torch.ops.aten.add.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
    squeeze_107: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_232, [0, 2, 3]);  getitem_232 = None
    mul_284: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0006381620931717);  squeeze_107 = None
    mul_285: "f32[160]" = torch.ops.aten.mul.Tensor(mul_284, 0.1);  mul_284 = None
    mul_286: "f32[160]" = torch.ops.aten.mul.Tensor(primals_413, 0.9)
    add_186: "f32[160]" = torch.ops.aten.add.Tensor(mul_285, mul_286);  mul_285 = mul_286 = None
    unsqueeze_140: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1)
    unsqueeze_141: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_287: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_281, unsqueeze_141);  mul_281 = unsqueeze_141 = None
    unsqueeze_142: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1);  primals_83 = None
    unsqueeze_143: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_187: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_287, unsqueeze_143);  mul_287 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_57 = torch.ops.aten.split_with_sizes.default(add_187, [80, 80], 1)
    getitem_234: "f32[8, 80, 14, 14]" = split_with_sizes_57[0]
    getitem_235: "f32[8, 80, 14, 14]" = split_with_sizes_57[1];  split_with_sizes_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_89: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_234, primals_228, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_90: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_235, primals_229, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_25: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_89, convolution_90], 1);  convolution_89 = convolution_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_188: "i64[]" = torch.ops.aten.add.Tensor(primals_414, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(cat_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_236: "f32[1, 480, 1, 1]" = var_mean_36[0]
    getitem_237: "f32[1, 480, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_189: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_236, 0.001)
    rsqrt_36: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_36: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_25, getitem_237)
    mul_288: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_237, [0, 2, 3]);  getitem_237 = None
    squeeze_109: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_289: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_290: "f32[480]" = torch.ops.aten.mul.Tensor(primals_415, 0.9)
    add_190: "f32[480]" = torch.ops.aten.add.Tensor(mul_289, mul_290);  mul_289 = mul_290 = None
    squeeze_110: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_236, [0, 2, 3]);  getitem_236 = None
    mul_291: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0006381620931717);  squeeze_110 = None
    mul_292: "f32[480]" = torch.ops.aten.mul.Tensor(mul_291, 0.1);  mul_291 = None
    mul_293: "f32[480]" = torch.ops.aten.mul.Tensor(primals_416, 0.9)
    add_191: "f32[480]" = torch.ops.aten.add.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
    unsqueeze_144: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
    unsqueeze_145: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_294: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_288, unsqueeze_145);  mul_288 = unsqueeze_145 = None
    unsqueeze_146: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
    unsqueeze_147: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_192: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_294, unsqueeze_147);  mul_294 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_36: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_192)
    mul_295: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_192, sigmoid_36);  sigmoid_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_59 = torch.ops.aten.split_with_sizes.default(mul_295, [120, 120, 120, 120], 1);  mul_295 = None
    getitem_242: "f32[8, 120, 14, 14]" = split_with_sizes_59[0]
    convolution_91: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_242, primals_230, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120)
    getitem_247: "f32[8, 120, 14, 14]" = split_with_sizes_59[1]
    convolution_92: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_247, primals_231, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    getitem_252: "f32[8, 120, 14, 14]" = split_with_sizes_59[2]
    convolution_93: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_252, primals_232, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120)
    getitem_257: "f32[8, 120, 14, 14]" = split_with_sizes_59[3];  split_with_sizes_59 = None
    convolution_94: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_257, primals_233, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_26: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_91, convolution_92, convolution_93, convolution_94], 1);  convolution_91 = convolution_92 = convolution_93 = convolution_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_193: "i64[]" = torch.ops.aten.add.Tensor(primals_417, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(cat_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_258: "f32[1, 480, 1, 1]" = var_mean_37[0]
    getitem_259: "f32[1, 480, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_194: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_258, 0.001)
    rsqrt_37: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_37: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_26, getitem_259)
    mul_296: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_259, [0, 2, 3]);  getitem_259 = None
    squeeze_112: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_297: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_298: "f32[480]" = torch.ops.aten.mul.Tensor(primals_418, 0.9)
    add_195: "f32[480]" = torch.ops.aten.add.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
    squeeze_113: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_258, [0, 2, 3]);  getitem_258 = None
    mul_299: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0006381620931717);  squeeze_113 = None
    mul_300: "f32[480]" = torch.ops.aten.mul.Tensor(mul_299, 0.1);  mul_299 = None
    mul_301: "f32[480]" = torch.ops.aten.mul.Tensor(primals_419, 0.9)
    add_196: "f32[480]" = torch.ops.aten.add.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    unsqueeze_148: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_149: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_302: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_149);  mul_296 = unsqueeze_149 = None
    unsqueeze_150: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_151: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_197: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_151);  mul_302 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_37: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_197)
    mul_303: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_197, sigmoid_37);  sigmoid_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_303, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_95: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_234, primals_235, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_38: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_95)
    mul_304: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_95, sigmoid_38);  sigmoid_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_96: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_304, primals_236, primals_237, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_39: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_96)
    mul_305: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_303, sigmoid_39);  mul_303 = sigmoid_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_63 = torch.ops.aten.split_with_sizes.default(mul_305, [240, 240], 1);  mul_305 = None
    getitem_260: "f32[8, 240, 14, 14]" = split_with_sizes_63[0]
    getitem_261: "f32[8, 240, 14, 14]" = split_with_sizes_63[1];  split_with_sizes_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_97: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_260, primals_238, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_98: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_261, primals_239, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_27: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([convolution_97, convolution_98], 1);  convolution_97 = convolution_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_198: "i64[]" = torch.ops.aten.add.Tensor(primals_420, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(cat_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_262: "f32[1, 160, 1, 1]" = var_mean_38[0]
    getitem_263: "f32[1, 160, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_199: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_262, 0.001)
    rsqrt_38: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_38: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_27, getitem_263)
    mul_306: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_263, [0, 2, 3]);  getitem_263 = None
    squeeze_115: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_307: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_308: "f32[160]" = torch.ops.aten.mul.Tensor(primals_421, 0.9)
    add_200: "f32[160]" = torch.ops.aten.add.Tensor(mul_307, mul_308);  mul_307 = mul_308 = None
    squeeze_116: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_262, [0, 2, 3]);  getitem_262 = None
    mul_309: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0006381620931717);  squeeze_116 = None
    mul_310: "f32[160]" = torch.ops.aten.mul.Tensor(mul_309, 0.1);  mul_309 = None
    mul_311: "f32[160]" = torch.ops.aten.mul.Tensor(primals_422, 0.9)
    add_201: "f32[160]" = torch.ops.aten.add.Tensor(mul_310, mul_311);  mul_310 = mul_311 = None
    unsqueeze_152: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1)
    unsqueeze_153: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_312: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_306, unsqueeze_153);  mul_306 = unsqueeze_153 = None
    unsqueeze_154: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1);  primals_89 = None
    unsqueeze_155: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_202: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_312, unsqueeze_155);  mul_312 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_203: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_202, add_187);  add_202 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_64 = torch.ops.aten.split_with_sizes.default(add_203, [80, 80], 1)
    getitem_264: "f32[8, 80, 14, 14]" = split_with_sizes_64[0]
    getitem_265: "f32[8, 80, 14, 14]" = split_with_sizes_64[1];  split_with_sizes_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_99: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_264, primals_240, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_100: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_265, primals_241, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_28: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_99, convolution_100], 1);  convolution_99 = convolution_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_204: "i64[]" = torch.ops.aten.add.Tensor(primals_423, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(cat_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_266: "f32[1, 480, 1, 1]" = var_mean_39[0]
    getitem_267: "f32[1, 480, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_205: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_266, 0.001)
    rsqrt_39: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    sub_39: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_28, getitem_267)
    mul_313: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_267, [0, 2, 3]);  getitem_267 = None
    squeeze_118: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_314: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_315: "f32[480]" = torch.ops.aten.mul.Tensor(primals_424, 0.9)
    add_206: "f32[480]" = torch.ops.aten.add.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
    squeeze_119: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_266, [0, 2, 3]);  getitem_266 = None
    mul_316: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0006381620931717);  squeeze_119 = None
    mul_317: "f32[480]" = torch.ops.aten.mul.Tensor(mul_316, 0.1);  mul_316 = None
    mul_318: "f32[480]" = torch.ops.aten.mul.Tensor(primals_425, 0.9)
    add_207: "f32[480]" = torch.ops.aten.add.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
    unsqueeze_156: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1)
    unsqueeze_157: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_319: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_157);  mul_313 = unsqueeze_157 = None
    unsqueeze_158: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
    unsqueeze_159: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_208: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_319, unsqueeze_159);  mul_319 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_40: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_208)
    mul_320: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_208, sigmoid_40);  sigmoid_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_66 = torch.ops.aten.split_with_sizes.default(mul_320, [120, 120, 120, 120], 1);  mul_320 = None
    getitem_272: "f32[8, 120, 14, 14]" = split_with_sizes_66[0]
    convolution_101: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_272, primals_242, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120)
    getitem_277: "f32[8, 120, 14, 14]" = split_with_sizes_66[1]
    convolution_102: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_277, primals_243, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    getitem_282: "f32[8, 120, 14, 14]" = split_with_sizes_66[2]
    convolution_103: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_282, primals_244, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120)
    getitem_287: "f32[8, 120, 14, 14]" = split_with_sizes_66[3];  split_with_sizes_66 = None
    convolution_104: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_287, primals_245, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_29: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_101, convolution_102, convolution_103, convolution_104], 1);  convolution_101 = convolution_102 = convolution_103 = convolution_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_209: "i64[]" = torch.ops.aten.add.Tensor(primals_426, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(cat_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_288: "f32[1, 480, 1, 1]" = var_mean_40[0]
    getitem_289: "f32[1, 480, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_210: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_288, 0.001)
    rsqrt_40: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_40: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_29, getitem_289)
    mul_321: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_289, [0, 2, 3]);  getitem_289 = None
    squeeze_121: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_322: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_323: "f32[480]" = torch.ops.aten.mul.Tensor(primals_427, 0.9)
    add_211: "f32[480]" = torch.ops.aten.add.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None
    squeeze_122: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_288, [0, 2, 3]);  getitem_288 = None
    mul_324: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0006381620931717);  squeeze_122 = None
    mul_325: "f32[480]" = torch.ops.aten.mul.Tensor(mul_324, 0.1);  mul_324 = None
    mul_326: "f32[480]" = torch.ops.aten.mul.Tensor(primals_428, 0.9)
    add_212: "f32[480]" = torch.ops.aten.add.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
    unsqueeze_160: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_161: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_327: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_321, unsqueeze_161);  mul_321 = unsqueeze_161 = None
    unsqueeze_162: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_163: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_213: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_327, unsqueeze_163);  mul_327 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_41: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_213)
    mul_328: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_213, sigmoid_41);  sigmoid_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_328, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_105: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_246, primals_247, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_42: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_105)
    mul_329: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_105, sigmoid_42);  sigmoid_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_106: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_329, primals_248, primals_249, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_43: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_106)
    mul_330: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_328, sigmoid_43);  mul_328 = sigmoid_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_70 = torch.ops.aten.split_with_sizes.default(mul_330, [240, 240], 1);  mul_330 = None
    getitem_290: "f32[8, 240, 14, 14]" = split_with_sizes_70[0]
    getitem_291: "f32[8, 240, 14, 14]" = split_with_sizes_70[1];  split_with_sizes_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_107: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_290, primals_250, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_108: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_291, primals_251, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_30: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([convolution_107, convolution_108], 1);  convolution_107 = convolution_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_214: "i64[]" = torch.ops.aten.add.Tensor(primals_429, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(cat_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_292: "f32[1, 160, 1, 1]" = var_mean_41[0]
    getitem_293: "f32[1, 160, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_215: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_292, 0.001)
    rsqrt_41: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
    sub_41: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_30, getitem_293)
    mul_331: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_293, [0, 2, 3]);  getitem_293 = None
    squeeze_124: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_332: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_333: "f32[160]" = torch.ops.aten.mul.Tensor(primals_430, 0.9)
    add_216: "f32[160]" = torch.ops.aten.add.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
    squeeze_125: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_292, [0, 2, 3]);  getitem_292 = None
    mul_334: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0006381620931717);  squeeze_125 = None
    mul_335: "f32[160]" = torch.ops.aten.mul.Tensor(mul_334, 0.1);  mul_334 = None
    mul_336: "f32[160]" = torch.ops.aten.mul.Tensor(primals_431, 0.9)
    add_217: "f32[160]" = torch.ops.aten.add.Tensor(mul_335, mul_336);  mul_335 = mul_336 = None
    unsqueeze_164: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1)
    unsqueeze_165: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_337: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_165);  mul_331 = unsqueeze_165 = None
    unsqueeze_166: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1);  primals_95 = None
    unsqueeze_167: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_218: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_337, unsqueeze_167);  mul_337 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_219: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_218, add_203);  add_218 = add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_71 = torch.ops.aten.split_with_sizes.default(add_219, [80, 80], 1)
    getitem_294: "f32[8, 80, 14, 14]" = split_with_sizes_71[0]
    getitem_295: "f32[8, 80, 14, 14]" = split_with_sizes_71[1];  split_with_sizes_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_109: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_294, primals_252, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_110: "f32[8, 240, 14, 14]" = torch.ops.aten.convolution.default(getitem_295, primals_253, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_31: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_109, convolution_110], 1);  convolution_109 = convolution_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_220: "i64[]" = torch.ops.aten.add.Tensor(primals_432, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(cat_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_296: "f32[1, 480, 1, 1]" = var_mean_42[0]
    getitem_297: "f32[1, 480, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_221: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_296, 0.001)
    rsqrt_42: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
    sub_42: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_31, getitem_297)
    mul_338: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_297, [0, 2, 3]);  getitem_297 = None
    squeeze_127: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_339: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_340: "f32[480]" = torch.ops.aten.mul.Tensor(primals_433, 0.9)
    add_222: "f32[480]" = torch.ops.aten.add.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
    squeeze_128: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_296, [0, 2, 3]);  getitem_296 = None
    mul_341: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0006381620931717);  squeeze_128 = None
    mul_342: "f32[480]" = torch.ops.aten.mul.Tensor(mul_341, 0.1);  mul_341 = None
    mul_343: "f32[480]" = torch.ops.aten.mul.Tensor(primals_434, 0.9)
    add_223: "f32[480]" = torch.ops.aten.add.Tensor(mul_342, mul_343);  mul_342 = mul_343 = None
    unsqueeze_168: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1)
    unsqueeze_169: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_344: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_338, unsqueeze_169);  mul_338 = unsqueeze_169 = None
    unsqueeze_170: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
    unsqueeze_171: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_224: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_344, unsqueeze_171);  mul_344 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_44: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_224)
    mul_345: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_224, sigmoid_44);  sigmoid_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_73 = torch.ops.aten.split_with_sizes.default(mul_345, [120, 120, 120, 120], 1);  mul_345 = None
    getitem_302: "f32[8, 120, 14, 14]" = split_with_sizes_73[0]
    convolution_111: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_302, primals_254, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120)
    getitem_307: "f32[8, 120, 14, 14]" = split_with_sizes_73[1]
    convolution_112: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_307, primals_255, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    getitem_312: "f32[8, 120, 14, 14]" = split_with_sizes_73[2]
    convolution_113: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_312, primals_256, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 120)
    getitem_317: "f32[8, 120, 14, 14]" = split_with_sizes_73[3];  split_with_sizes_73 = None
    convolution_114: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(getitem_317, primals_257, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_32: "f32[8, 480, 14, 14]" = torch.ops.aten.cat.default([convolution_111, convolution_112, convolution_113, convolution_114], 1);  convolution_111 = convolution_112 = convolution_113 = convolution_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_225: "i64[]" = torch.ops.aten.add.Tensor(primals_435, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(cat_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_318: "f32[1, 480, 1, 1]" = var_mean_43[0]
    getitem_319: "f32[1, 480, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_226: "f32[1, 480, 1, 1]" = torch.ops.aten.add.Tensor(getitem_318, 0.001)
    rsqrt_43: "f32[1, 480, 1, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
    sub_43: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_32, getitem_319)
    mul_346: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_319, [0, 2, 3]);  getitem_319 = None
    squeeze_130: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_347: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_348: "f32[480]" = torch.ops.aten.mul.Tensor(primals_436, 0.9)
    add_227: "f32[480]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    squeeze_131: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_318, [0, 2, 3]);  getitem_318 = None
    mul_349: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0006381620931717);  squeeze_131 = None
    mul_350: "f32[480]" = torch.ops.aten.mul.Tensor(mul_349, 0.1);  mul_349 = None
    mul_351: "f32[480]" = torch.ops.aten.mul.Tensor(primals_437, 0.9)
    add_228: "f32[480]" = torch.ops.aten.add.Tensor(mul_350, mul_351);  mul_350 = mul_351 = None
    unsqueeze_172: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_173: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_352: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_173);  mul_346 = unsqueeze_173 = None
    unsqueeze_174: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_175: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_229: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_352, unsqueeze_175);  mul_352 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_45: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_229)
    mul_353: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_229, sigmoid_45);  sigmoid_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[8, 480, 1, 1]" = torch.ops.aten.mean.dim(mul_353, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_115: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_258, primals_259, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_46: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_115)
    mul_354: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_115, sigmoid_46);  sigmoid_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_116: "f32[8, 480, 1, 1]" = torch.ops.aten.convolution.default(mul_354, primals_260, primals_261, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_47: "f32[8, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_116)
    mul_355: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_353, sigmoid_47);  mul_353 = sigmoid_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_77 = torch.ops.aten.split_with_sizes.default(mul_355, [240, 240], 1);  mul_355 = None
    getitem_320: "f32[8, 240, 14, 14]" = split_with_sizes_77[0]
    getitem_321: "f32[8, 240, 14, 14]" = split_with_sizes_77[1];  split_with_sizes_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_117: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_320, primals_262, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_118: "f32[8, 80, 14, 14]" = torch.ops.aten.convolution.default(getitem_321, primals_263, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_33: "f32[8, 160, 14, 14]" = torch.ops.aten.cat.default([convolution_117, convolution_118], 1);  convolution_117 = convolution_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_230: "i64[]" = torch.ops.aten.add.Tensor(primals_438, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(cat_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_322: "f32[1, 160, 1, 1]" = var_mean_44[0]
    getitem_323: "f32[1, 160, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_231: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_322, 0.001)
    rsqrt_44: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_44: "f32[8, 160, 14, 14]" = torch.ops.aten.sub.Tensor(cat_33, getitem_323)
    mul_356: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_323, [0, 2, 3]);  getitem_323 = None
    squeeze_133: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_357: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_358: "f32[160]" = torch.ops.aten.mul.Tensor(primals_439, 0.9)
    add_232: "f32[160]" = torch.ops.aten.add.Tensor(mul_357, mul_358);  mul_357 = mul_358 = None
    squeeze_134: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_322, [0, 2, 3]);  getitem_322 = None
    mul_359: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0006381620931717);  squeeze_134 = None
    mul_360: "f32[160]" = torch.ops.aten.mul.Tensor(mul_359, 0.1);  mul_359 = None
    mul_361: "f32[160]" = torch.ops.aten.mul.Tensor(primals_440, 0.9)
    add_233: "f32[160]" = torch.ops.aten.add.Tensor(mul_360, mul_361);  mul_360 = mul_361 = None
    unsqueeze_176: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1)
    unsqueeze_177: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_362: "f32[8, 160, 14, 14]" = torch.ops.aten.mul.Tensor(mul_356, unsqueeze_177);  mul_356 = unsqueeze_177 = None
    unsqueeze_178: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1);  primals_101 = None
    unsqueeze_179: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_234: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(mul_362, unsqueeze_179);  mul_362 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_235: "f32[8, 160, 14, 14]" = torch.ops.aten.add.Tensor(add_234, add_219);  add_234 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_119: "f32[8, 960, 14, 14]" = torch.ops.aten.convolution.default(add_235, primals_264, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_236: "i64[]" = torch.ops.aten.add.Tensor(primals_441, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_119, [0, 2, 3], correction = 0, keepdim = True)
    getitem_324: "f32[1, 960, 1, 1]" = var_mean_45[0]
    getitem_325: "f32[1, 960, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_237: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem_324, 0.001)
    rsqrt_45: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
    sub_45: "f32[8, 960, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, getitem_325)
    mul_363: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_325, [0, 2, 3]);  getitem_325 = None
    squeeze_136: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_364: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_365: "f32[960]" = torch.ops.aten.mul.Tensor(primals_442, 0.9)
    add_238: "f32[960]" = torch.ops.aten.add.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
    squeeze_137: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_324, [0, 2, 3]);  getitem_324 = None
    mul_366: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0006381620931717);  squeeze_137 = None
    mul_367: "f32[960]" = torch.ops.aten.mul.Tensor(mul_366, 0.1);  mul_366 = None
    mul_368: "f32[960]" = torch.ops.aten.mul.Tensor(primals_443, 0.9)
    add_239: "f32[960]" = torch.ops.aten.add.Tensor(mul_367, mul_368);  mul_367 = mul_368 = None
    unsqueeze_180: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1)
    unsqueeze_181: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_369: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(mul_363, unsqueeze_181);  mul_363 = unsqueeze_181 = None
    unsqueeze_182: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
    unsqueeze_183: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_240: "f32[8, 960, 14, 14]" = torch.ops.aten.add.Tensor(mul_369, unsqueeze_183);  mul_369 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_48: "f32[8, 960, 14, 14]" = torch.ops.aten.sigmoid.default(add_240)
    mul_370: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(add_240, sigmoid_48);  sigmoid_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    split_with_sizes_79 = torch.ops.aten.split_with_sizes.default(mul_370, [240, 240, 240, 240], 1);  mul_370 = None
    getitem_330: "f32[8, 240, 14, 14]" = split_with_sizes_79[0]
    constant_pad_nd_11: "f32[8, 240, 15, 15]" = torch.ops.aten.constant_pad_nd.default(getitem_330, [0, 1, 0, 1], 0.0);  getitem_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_120: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_11, primals_104, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    getitem_335: "f32[8, 240, 14, 14]" = split_with_sizes_79[1]
    constant_pad_nd_12: "f32[8, 240, 17, 17]" = torch.ops.aten.constant_pad_nd.default(getitem_335, [1, 2, 1, 2], 0.0);  getitem_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_121: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_12, primals_105, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    getitem_340: "f32[8, 240, 14, 14]" = split_with_sizes_79[2]
    constant_pad_nd_13: "f32[8, 240, 19, 19]" = torch.ops.aten.constant_pad_nd.default(getitem_340, [2, 3, 2, 3], 0.0);  getitem_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_122: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_13, primals_106, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/padding.py:55, code: x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=value)
    getitem_345: "f32[8, 240, 14, 14]" = split_with_sizes_79[3];  split_with_sizes_79 = None
    constant_pad_nd_14: "f32[8, 240, 21, 21]" = torch.ops.aten.constant_pad_nd.default(getitem_345, [3, 4, 3, 4], 0.0);  getitem_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv2d_same.py:27, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    convolution_123: "f32[8, 240, 7, 7]" = torch.ops.aten.convolution.default(constant_pad_nd_14, primals_107, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_34: "f32[8, 960, 7, 7]" = torch.ops.aten.cat.default([convolution_120, convolution_121, convolution_122, convolution_123], 1);  convolution_120 = convolution_121 = convolution_122 = convolution_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_241: "i64[]" = torch.ops.aten.add.Tensor(primals_444, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(cat_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_346: "f32[1, 960, 1, 1]" = var_mean_46[0]
    getitem_347: "f32[1, 960, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_242: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem_346, 0.001)
    rsqrt_46: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
    sub_46: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(cat_34, getitem_347)
    mul_371: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_347, [0, 2, 3]);  getitem_347 = None
    squeeze_139: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_372: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_373: "f32[960]" = torch.ops.aten.mul.Tensor(primals_445, 0.9)
    add_243: "f32[960]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    squeeze_140: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_346, [0, 2, 3]);  getitem_346 = None
    mul_374: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0025575447570332);  squeeze_140 = None
    mul_375: "f32[960]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
    mul_376: "f32[960]" = torch.ops.aten.mul.Tensor(primals_446, 0.9)
    add_244: "f32[960]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_184: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1)
    unsqueeze_185: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_377: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_185);  mul_371 = unsqueeze_185 = None
    unsqueeze_186: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
    unsqueeze_187: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_245: "f32[8, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_187);  mul_377 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_49: "f32[8, 960, 7, 7]" = torch.ops.aten.sigmoid.default(add_245)
    mul_378: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_245, sigmoid_49);  sigmoid_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[8, 960, 1, 1]" = torch.ops.aten.mean.dim(mul_378, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_124: "f32[8, 80, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_265, primals_266, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_50: "f32[8, 80, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_124)
    mul_379: "f32[8, 80, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_124, sigmoid_50);  sigmoid_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_125: "f32[8, 960, 1, 1]" = torch.ops.aten.convolution.default(mul_379, primals_267, primals_268, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_51: "f32[8, 960, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_125)
    mul_380: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_378, sigmoid_51);  mul_378 = sigmoid_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_126: "f32[8, 264, 7, 7]" = torch.ops.aten.convolution.default(mul_380, primals_269, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_246: "i64[]" = torch.ops.aten.add.Tensor(primals_447, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_126, [0, 2, 3], correction = 0, keepdim = True)
    getitem_348: "f32[1, 264, 1, 1]" = var_mean_47[0]
    getitem_349: "f32[1, 264, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_247: "f32[1, 264, 1, 1]" = torch.ops.aten.add.Tensor(getitem_348, 0.001)
    rsqrt_47: "f32[1, 264, 1, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
    sub_47: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_126, getitem_349)
    mul_381: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_349, [0, 2, 3]);  getitem_349 = None
    squeeze_142: "f32[264]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_382: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_383: "f32[264]" = torch.ops.aten.mul.Tensor(primals_448, 0.9)
    add_248: "f32[264]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    squeeze_143: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_348, [0, 2, 3]);  getitem_348 = None
    mul_384: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0025575447570332);  squeeze_143 = None
    mul_385: "f32[264]" = torch.ops.aten.mul.Tensor(mul_384, 0.1);  mul_384 = None
    mul_386: "f32[264]" = torch.ops.aten.mul.Tensor(primals_449, 0.9)
    add_249: "f32[264]" = torch.ops.aten.add.Tensor(mul_385, mul_386);  mul_385 = mul_386 = None
    unsqueeze_188: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_189: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_387: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_381, unsqueeze_189);  mul_381 = unsqueeze_189 = None
    unsqueeze_190: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_191: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_250: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_387, unsqueeze_191);  mul_387 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_127: "f32[8, 1584, 7, 7]" = torch.ops.aten.convolution.default(add_250, primals_270, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_251: "i64[]" = torch.ops.aten.add.Tensor(primals_450, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_127, [0, 2, 3], correction = 0, keepdim = True)
    getitem_350: "f32[1, 1584, 1, 1]" = var_mean_48[0]
    getitem_351: "f32[1, 1584, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_252: "f32[1, 1584, 1, 1]" = torch.ops.aten.add.Tensor(getitem_350, 0.001)
    rsqrt_48: "f32[1, 1584, 1, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
    sub_48: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_127, getitem_351)
    mul_388: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_351, [0, 2, 3]);  getitem_351 = None
    squeeze_145: "f32[1584]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_389: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_390: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_451, 0.9)
    add_253: "f32[1584]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    squeeze_146: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_350, [0, 2, 3]);  getitem_350 = None
    mul_391: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0025575447570332);  squeeze_146 = None
    mul_392: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_391, 0.1);  mul_391 = None
    mul_393: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_452, 0.9)
    add_254: "f32[1584]" = torch.ops.aten.add.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
    unsqueeze_192: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1)
    unsqueeze_193: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_394: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_193);  mul_388 = unsqueeze_193 = None
    unsqueeze_194: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1);  primals_113 = None
    unsqueeze_195: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_255: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_394, unsqueeze_195);  mul_394 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_52: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_255)
    mul_395: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_255, sigmoid_52);  sigmoid_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_84 = torch.ops.aten.split_with_sizes.default(mul_395, [396, 396, 396, 396], 1);  mul_395 = None
    getitem_356: "f32[8, 396, 7, 7]" = split_with_sizes_84[0]
    convolution_128: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_356, primals_271, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396)
    getitem_361: "f32[8, 396, 7, 7]" = split_with_sizes_84[1]
    convolution_129: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_361, primals_272, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396)
    getitem_366: "f32[8, 396, 7, 7]" = split_with_sizes_84[2]
    convolution_130: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_366, primals_273, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396)
    getitem_371: "f32[8, 396, 7, 7]" = split_with_sizes_84[3];  split_with_sizes_84 = None
    convolution_131: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_371, primals_274, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_35: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([convolution_128, convolution_129, convolution_130, convolution_131], 1);  convolution_128 = convolution_129 = convolution_130 = convolution_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_256: "i64[]" = torch.ops.aten.add.Tensor(primals_453, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(cat_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_372: "f32[1, 1584, 1, 1]" = var_mean_49[0]
    getitem_373: "f32[1, 1584, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_257: "f32[1, 1584, 1, 1]" = torch.ops.aten.add.Tensor(getitem_372, 0.001)
    rsqrt_49: "f32[1, 1584, 1, 1]" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
    sub_49: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_35, getitem_373)
    mul_396: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_373, [0, 2, 3]);  getitem_373 = None
    squeeze_148: "f32[1584]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_397: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_398: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_454, 0.9)
    add_258: "f32[1584]" = torch.ops.aten.add.Tensor(mul_397, mul_398);  mul_397 = mul_398 = None
    squeeze_149: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_372, [0, 2, 3]);  getitem_372 = None
    mul_399: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0025575447570332);  squeeze_149 = None
    mul_400: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_399, 0.1);  mul_399 = None
    mul_401: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_455, 0.9)
    add_259: "f32[1584]" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
    unsqueeze_196: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1)
    unsqueeze_197: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_402: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_396, unsqueeze_197);  mul_396 = unsqueeze_197 = None
    unsqueeze_198: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
    unsqueeze_199: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_260: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_402, unsqueeze_199);  mul_402 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_53: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_260)
    mul_403: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_260, sigmoid_53);  sigmoid_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_13: "f32[8, 1584, 1, 1]" = torch.ops.aten.mean.dim(mul_403, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_132: "f32[8, 132, 1, 1]" = torch.ops.aten.convolution.default(mean_13, primals_275, primals_276, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_54: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_132)
    mul_404: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_132, sigmoid_54);  sigmoid_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_133: "f32[8, 1584, 1, 1]" = torch.ops.aten.convolution.default(mul_404, primals_277, primals_278, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_55: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_133)
    mul_405: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_403, sigmoid_55);  mul_403 = sigmoid_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_88 = torch.ops.aten.split_with_sizes.default(mul_405, [792, 792], 1);  mul_405 = None
    getitem_374: "f32[8, 792, 7, 7]" = split_with_sizes_88[0]
    getitem_375: "f32[8, 792, 7, 7]" = split_with_sizes_88[1];  split_with_sizes_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_134: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_374, primals_279, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_135: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_375, primals_280, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_36: "f32[8, 264, 7, 7]" = torch.ops.aten.cat.default([convolution_134, convolution_135], 1);  convolution_134 = convolution_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_261: "i64[]" = torch.ops.aten.add.Tensor(primals_456, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(cat_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_376: "f32[1, 264, 1, 1]" = var_mean_50[0]
    getitem_377: "f32[1, 264, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_262: "f32[1, 264, 1, 1]" = torch.ops.aten.add.Tensor(getitem_376, 0.001)
    rsqrt_50: "f32[1, 264, 1, 1]" = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
    sub_50: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_36, getitem_377)
    mul_406: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_377, [0, 2, 3]);  getitem_377 = None
    squeeze_151: "f32[264]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_407: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_408: "f32[264]" = torch.ops.aten.mul.Tensor(primals_457, 0.9)
    add_263: "f32[264]" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    squeeze_152: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_376, [0, 2, 3]);  getitem_376 = None
    mul_409: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0025575447570332);  squeeze_152 = None
    mul_410: "f32[264]" = torch.ops.aten.mul.Tensor(mul_409, 0.1);  mul_409 = None
    mul_411: "f32[264]" = torch.ops.aten.mul.Tensor(primals_458, 0.9)
    add_264: "f32[264]" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    unsqueeze_200: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
    unsqueeze_201: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_412: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_201);  mul_406 = unsqueeze_201 = None
    unsqueeze_202: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
    unsqueeze_203: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_265: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_203);  mul_412 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_266: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_265, add_250);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_136: "f32[8, 1584, 7, 7]" = torch.ops.aten.convolution.default(add_266, primals_281, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_267: "i64[]" = torch.ops.aten.add.Tensor(primals_459, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_136, [0, 2, 3], correction = 0, keepdim = True)
    getitem_378: "f32[1, 1584, 1, 1]" = var_mean_51[0]
    getitem_379: "f32[1, 1584, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_268: "f32[1, 1584, 1, 1]" = torch.ops.aten.add.Tensor(getitem_378, 0.001)
    rsqrt_51: "f32[1, 1584, 1, 1]" = torch.ops.aten.rsqrt.default(add_268);  add_268 = None
    sub_51: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_136, getitem_379)
    mul_413: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_379, [0, 2, 3]);  getitem_379 = None
    squeeze_154: "f32[1584]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_414: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_415: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_460, 0.9)
    add_269: "f32[1584]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    squeeze_155: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_378, [0, 2, 3]);  getitem_378 = None
    mul_416: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0025575447570332);  squeeze_155 = None
    mul_417: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
    mul_418: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_461, 0.9)
    add_270: "f32[1584]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    unsqueeze_204: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1)
    unsqueeze_205: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_419: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_205);  mul_413 = unsqueeze_205 = None
    unsqueeze_206: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1);  primals_119 = None
    unsqueeze_207: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_271: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_207);  mul_419 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_56: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_271)
    mul_420: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_271, sigmoid_56);  sigmoid_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_90 = torch.ops.aten.split_with_sizes.default(mul_420, [396, 396, 396, 396], 1);  mul_420 = None
    getitem_384: "f32[8, 396, 7, 7]" = split_with_sizes_90[0]
    convolution_137: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_384, primals_282, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396)
    getitem_389: "f32[8, 396, 7, 7]" = split_with_sizes_90[1]
    convolution_138: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_389, primals_283, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396)
    getitem_394: "f32[8, 396, 7, 7]" = split_with_sizes_90[2]
    convolution_139: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_394, primals_284, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396)
    getitem_399: "f32[8, 396, 7, 7]" = split_with_sizes_90[3];  split_with_sizes_90 = None
    convolution_140: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_399, primals_285, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_37: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([convolution_137, convolution_138, convolution_139, convolution_140], 1);  convolution_137 = convolution_138 = convolution_139 = convolution_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_272: "i64[]" = torch.ops.aten.add.Tensor(primals_462, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(cat_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_400: "f32[1, 1584, 1, 1]" = var_mean_52[0]
    getitem_401: "f32[1, 1584, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_273: "f32[1, 1584, 1, 1]" = torch.ops.aten.add.Tensor(getitem_400, 0.001)
    rsqrt_52: "f32[1, 1584, 1, 1]" = torch.ops.aten.rsqrt.default(add_273);  add_273 = None
    sub_52: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_37, getitem_401)
    mul_421: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_401, [0, 2, 3]);  getitem_401 = None
    squeeze_157: "f32[1584]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_422: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_423: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_463, 0.9)
    add_274: "f32[1584]" = torch.ops.aten.add.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
    squeeze_158: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_400, [0, 2, 3]);  getitem_400 = None
    mul_424: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0025575447570332);  squeeze_158 = None
    mul_425: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_424, 0.1);  mul_424 = None
    mul_426: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_464, 0.9)
    add_275: "f32[1584]" = torch.ops.aten.add.Tensor(mul_425, mul_426);  mul_425 = mul_426 = None
    unsqueeze_208: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1)
    unsqueeze_209: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_427: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_421, unsqueeze_209);  mul_421 = unsqueeze_209 = None
    unsqueeze_210: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
    unsqueeze_211: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_276: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_427, unsqueeze_211);  mul_427 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_57: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_276)
    mul_428: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_276, sigmoid_57);  sigmoid_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_14: "f32[8, 1584, 1, 1]" = torch.ops.aten.mean.dim(mul_428, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_141: "f32[8, 132, 1, 1]" = torch.ops.aten.convolution.default(mean_14, primals_286, primals_287, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_58: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_141)
    mul_429: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_141, sigmoid_58);  sigmoid_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_142: "f32[8, 1584, 1, 1]" = torch.ops.aten.convolution.default(mul_429, primals_288, primals_289, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_59: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_142)
    mul_430: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_428, sigmoid_59);  mul_428 = sigmoid_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_94 = torch.ops.aten.split_with_sizes.default(mul_430, [792, 792], 1);  mul_430 = None
    getitem_402: "f32[8, 792, 7, 7]" = split_with_sizes_94[0]
    getitem_403: "f32[8, 792, 7, 7]" = split_with_sizes_94[1];  split_with_sizes_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_143: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_402, primals_290, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_144: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_403, primals_291, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_38: "f32[8, 264, 7, 7]" = torch.ops.aten.cat.default([convolution_143, convolution_144], 1);  convolution_143 = convolution_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_277: "i64[]" = torch.ops.aten.add.Tensor(primals_465, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(cat_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_404: "f32[1, 264, 1, 1]" = var_mean_53[0]
    getitem_405: "f32[1, 264, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_278: "f32[1, 264, 1, 1]" = torch.ops.aten.add.Tensor(getitem_404, 0.001)
    rsqrt_53: "f32[1, 264, 1, 1]" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
    sub_53: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_38, getitem_405)
    mul_431: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_405, [0, 2, 3]);  getitem_405 = None
    squeeze_160: "f32[264]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_432: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_433: "f32[264]" = torch.ops.aten.mul.Tensor(primals_466, 0.9)
    add_279: "f32[264]" = torch.ops.aten.add.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
    squeeze_161: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_404, [0, 2, 3]);  getitem_404 = None
    mul_434: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0025575447570332);  squeeze_161 = None
    mul_435: "f32[264]" = torch.ops.aten.mul.Tensor(mul_434, 0.1);  mul_434 = None
    mul_436: "f32[264]" = torch.ops.aten.mul.Tensor(primals_467, 0.9)
    add_280: "f32[264]" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
    unsqueeze_212: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
    unsqueeze_213: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_437: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_431, unsqueeze_213);  mul_431 = unsqueeze_213 = None
    unsqueeze_214: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
    unsqueeze_215: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_281: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_437, unsqueeze_215);  mul_437 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_282: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_281, add_266);  add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_145: "f32[8, 1584, 7, 7]" = torch.ops.aten.convolution.default(add_282, primals_292, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_283: "i64[]" = torch.ops.aten.add.Tensor(primals_468, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_145, [0, 2, 3], correction = 0, keepdim = True)
    getitem_406: "f32[1, 1584, 1, 1]" = var_mean_54[0]
    getitem_407: "f32[1, 1584, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_284: "f32[1, 1584, 1, 1]" = torch.ops.aten.add.Tensor(getitem_406, 0.001)
    rsqrt_54: "f32[1, 1584, 1, 1]" = torch.ops.aten.rsqrt.default(add_284);  add_284 = None
    sub_54: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_145, getitem_407)
    mul_438: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_407, [0, 2, 3]);  getitem_407 = None
    squeeze_163: "f32[1584]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_439: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_440: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_469, 0.9)
    add_285: "f32[1584]" = torch.ops.aten.add.Tensor(mul_439, mul_440);  mul_439 = mul_440 = None
    squeeze_164: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_406, [0, 2, 3]);  getitem_406 = None
    mul_441: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0025575447570332);  squeeze_164 = None
    mul_442: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_441, 0.1);  mul_441 = None
    mul_443: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_470, 0.9)
    add_286: "f32[1584]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    unsqueeze_216: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1)
    unsqueeze_217: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_444: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_438, unsqueeze_217);  mul_438 = unsqueeze_217 = None
    unsqueeze_218: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1);  primals_125 = None
    unsqueeze_219: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_287: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_444, unsqueeze_219);  mul_444 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_60: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_287)
    mul_445: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_287, sigmoid_60);  sigmoid_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    split_with_sizes_96 = torch.ops.aten.split_with_sizes.default(mul_445, [396, 396, 396, 396], 1);  mul_445 = None
    getitem_412: "f32[8, 396, 7, 7]" = split_with_sizes_96[0]
    convolution_146: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_412, primals_293, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 396)
    getitem_417: "f32[8, 396, 7, 7]" = split_with_sizes_96[1]
    convolution_147: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_417, primals_294, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 396)
    getitem_422: "f32[8, 396, 7, 7]" = split_with_sizes_96[2]
    convolution_148: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_422, primals_295, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 396)
    getitem_427: "f32[8, 396, 7, 7]" = split_with_sizes_96[3];  split_with_sizes_96 = None
    convolution_149: "f32[8, 396, 7, 7]" = torch.ops.aten.convolution.default(getitem_427, primals_296, None, [1, 1], [4, 4], [1, 1], False, [0, 0], 396)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_39: "f32[8, 1584, 7, 7]" = torch.ops.aten.cat.default([convolution_146, convolution_147, convolution_148, convolution_149], 1);  convolution_146 = convolution_147 = convolution_148 = convolution_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_288: "i64[]" = torch.ops.aten.add.Tensor(primals_471, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_55 = torch.ops.aten.var_mean.correction(cat_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_428: "f32[1, 1584, 1, 1]" = var_mean_55[0]
    getitem_429: "f32[1, 1584, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_289: "f32[1, 1584, 1, 1]" = torch.ops.aten.add.Tensor(getitem_428, 0.001)
    rsqrt_55: "f32[1, 1584, 1, 1]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
    sub_55: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(cat_39, getitem_429)
    mul_446: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_429, [0, 2, 3]);  getitem_429 = None
    squeeze_166: "f32[1584]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_447: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_448: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_472, 0.9)
    add_290: "f32[1584]" = torch.ops.aten.add.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
    squeeze_167: "f32[1584]" = torch.ops.aten.squeeze.dims(getitem_428, [0, 2, 3]);  getitem_428 = None
    mul_449: "f32[1584]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0025575447570332);  squeeze_167 = None
    mul_450: "f32[1584]" = torch.ops.aten.mul.Tensor(mul_449, 0.1);  mul_449 = None
    mul_451: "f32[1584]" = torch.ops.aten.mul.Tensor(primals_473, 0.9)
    add_291: "f32[1584]" = torch.ops.aten.add.Tensor(mul_450, mul_451);  mul_450 = mul_451 = None
    unsqueeze_220: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1)
    unsqueeze_221: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_452: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_446, unsqueeze_221);  mul_446 = unsqueeze_221 = None
    unsqueeze_222: "f32[1584, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
    unsqueeze_223: "f32[1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_292: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Tensor(mul_452, unsqueeze_223);  mul_452 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_61: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_292)
    mul_453: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_292, sigmoid_61);  sigmoid_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_15: "f32[8, 1584, 1, 1]" = torch.ops.aten.mean.dim(mul_453, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_150: "f32[8, 132, 1, 1]" = torch.ops.aten.convolution.default(mean_15, primals_297, primals_298, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    sigmoid_62: "f32[8, 132, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_150)
    mul_454: "f32[8, 132, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_150, sigmoid_62);  sigmoid_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_151: "f32[8, 1584, 1, 1]" = torch.ops.aten.convolution.default(mul_454, primals_299, primals_300, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    sigmoid_63: "f32[8, 1584, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_151)
    mul_455: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(mul_453, sigmoid_63);  mul_453 = sigmoid_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:48, code: x_split = torch.split(x, self.splits, 1)
    split_with_sizes_100 = torch.ops.aten.split_with_sizes.default(mul_455, [792, 792], 1);  mul_455 = None
    getitem_430: "f32[8, 792, 7, 7]" = split_with_sizes_100[0]
    getitem_431: "f32[8, 792, 7, 7]" = split_with_sizes_100[1];  split_with_sizes_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:49, code: x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
    convolution_152: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_430, primals_301, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    convolution_153: "f32[8, 132, 7, 7]" = torch.ops.aten.convolution.default(getitem_431, primals_302, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mixed_conv2d.py:50, code: x = torch.cat(x_out, 1)
    cat_40: "f32[8, 264, 7, 7]" = torch.ops.aten.cat.default([convolution_152, convolution_153], 1);  convolution_152 = convolution_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_293: "i64[]" = torch.ops.aten.add.Tensor(primals_474, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(cat_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_432: "f32[1, 264, 1, 1]" = var_mean_56[0]
    getitem_433: "f32[1, 264, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_294: "f32[1, 264, 1, 1]" = torch.ops.aten.add.Tensor(getitem_432, 0.001)
    rsqrt_56: "f32[1, 264, 1, 1]" = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
    sub_56: "f32[8, 264, 7, 7]" = torch.ops.aten.sub.Tensor(cat_40, getitem_433)
    mul_456: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_433, [0, 2, 3]);  getitem_433 = None
    squeeze_169: "f32[264]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_457: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_458: "f32[264]" = torch.ops.aten.mul.Tensor(primals_475, 0.9)
    add_295: "f32[264]" = torch.ops.aten.add.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
    squeeze_170: "f32[264]" = torch.ops.aten.squeeze.dims(getitem_432, [0, 2, 3]);  getitem_432 = None
    mul_459: "f32[264]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0025575447570332);  squeeze_170 = None
    mul_460: "f32[264]" = torch.ops.aten.mul.Tensor(mul_459, 0.1);  mul_459 = None
    mul_461: "f32[264]" = torch.ops.aten.mul.Tensor(primals_476, 0.9)
    add_296: "f32[264]" = torch.ops.aten.add.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    unsqueeze_224: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
    unsqueeze_225: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_462: "f32[8, 264, 7, 7]" = torch.ops.aten.mul.Tensor(mul_456, unsqueeze_225);  mul_456 = unsqueeze_225 = None
    unsqueeze_226: "f32[264, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
    unsqueeze_227: "f32[264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_297: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(mul_462, unsqueeze_227);  mul_462 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_298: "f32[8, 264, 7, 7]" = torch.ops.aten.add.Tensor(add_297, add_282);  add_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_154: "f32[8, 1536, 7, 7]" = torch.ops.aten.convolution.default(add_298, primals_303, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_299: "i64[]" = torch.ops.aten.add.Tensor(primals_477, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_154, [0, 2, 3], correction = 0, keepdim = True)
    getitem_434: "f32[1, 1536, 1, 1]" = var_mean_57[0]
    getitem_435: "f32[1, 1536, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_300: "f32[1, 1536, 1, 1]" = torch.ops.aten.add.Tensor(getitem_434, 0.001)
    rsqrt_57: "f32[1, 1536, 1, 1]" = torch.ops.aten.rsqrt.default(add_300);  add_300 = None
    sub_57: "f32[8, 1536, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_154, getitem_435)
    mul_463: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_435, [0, 2, 3]);  getitem_435 = None
    squeeze_172: "f32[1536]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_464: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_465: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_478, 0.9)
    add_301: "f32[1536]" = torch.ops.aten.add.Tensor(mul_464, mul_465);  mul_464 = mul_465 = None
    squeeze_173: "f32[1536]" = torch.ops.aten.squeeze.dims(getitem_434, [0, 2, 3]);  getitem_434 = None
    mul_466: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0025575447570332);  squeeze_173 = None
    mul_467: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_466, 0.1);  mul_466 = None
    mul_468: "f32[1536]" = torch.ops.aten.mul.Tensor(primals_479, 0.9)
    add_302: "f32[1536]" = torch.ops.aten.add.Tensor(mul_467, mul_468);  mul_467 = mul_468 = None
    unsqueeze_228: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1)
    unsqueeze_229: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_469: "f32[8, 1536, 7, 7]" = torch.ops.aten.mul.Tensor(mul_463, unsqueeze_229);  mul_463 = unsqueeze_229 = None
    unsqueeze_230: "f32[1536, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1);  primals_131 = None
    unsqueeze_231: "f32[1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_303: "f32[8, 1536, 7, 7]" = torch.ops.aten.add.Tensor(mul_469, unsqueeze_231);  mul_469 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[8, 1536, 7, 7]" = torch.ops.aten.relu.default(add_303);  add_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_16: "f32[8, 1536, 1, 1]" = torch.ops.aten.mean.dim(relu_6, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 1536]" = torch.ops.aten.reshape.default(mean_16, [8, 1536]);  mean_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    permute: "f32[1536, 1000]" = torch.ops.aten.permute.default(primals_304, [1, 0]);  primals_304 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_305, view, permute);  primals_305 = None
    permute_1: "f32[1000, 1536]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le: "b8[8, 1536, 7, 7]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_232: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_233: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
    unsqueeze_234: "f32[1, 1536, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_244: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_245: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
    unsqueeze_246: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_2: "f32[8, 1584, 7, 7]" = torch.ops.aten.full.default([8, 1584, 7, 7], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_256: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_257: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
    unsqueeze_258: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_66: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_287)
    sub_73: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_66)
    mul_507: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_287, sub_73);  add_287 = sub_73 = None
    add_307: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Scalar(mul_507, 1);  mul_507 = None
    mul_508: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_66, add_307);  sigmoid_66 = add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_268: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_269: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
    unsqueeze_270: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_280: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_281: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
    unsqueeze_282: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_292: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_293: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
    unsqueeze_294: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_69: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_271)
    sub_89: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_69)
    mul_547: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_271, sub_89);  add_271 = sub_89 = None
    add_312: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Scalar(mul_547, 1);  mul_547 = None
    mul_548: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_69, add_312);  sigmoid_69 = add_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_304: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_305: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
    unsqueeze_306: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_316: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_317: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
    unsqueeze_318: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_328: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_329: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
    unsqueeze_330: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_72: "f32[8, 1584, 7, 7]" = torch.ops.aten.sigmoid.default(add_255)
    sub_105: "f32[8, 1584, 7, 7]" = torch.ops.aten.sub.Tensor(full_default_2, sigmoid_72);  full_default_2 = None
    mul_587: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(add_255, sub_105);  add_255 = sub_105 = None
    add_317: "f32[8, 1584, 7, 7]" = torch.ops.aten.add.Scalar(mul_587, 1);  mul_587 = None
    mul_588: "f32[8, 1584, 7, 7]" = torch.ops.aten.mul.Tensor(sigmoid_72, add_317);  sigmoid_72 = add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_340: "f32[1, 1584]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_341: "f32[1, 1584, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
    unsqueeze_342: "f32[1, 1584, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_352: "f32[1, 264]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_353: "f32[1, 264, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
    unsqueeze_354: "f32[1, 264, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_364: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_365: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
    unsqueeze_366: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_75: "f32[8, 960, 14, 14]" = torch.ops.aten.sigmoid.default(add_240)
    full_default_12: "f32[8, 960, 14, 14]" = torch.ops.aten.full.default([8, 960, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_121: "f32[8, 960, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_12, sigmoid_75);  full_default_12 = None
    mul_627: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(add_240, sub_121);  add_240 = sub_121 = None
    add_322: "f32[8, 960, 14, 14]" = torch.ops.aten.add.Scalar(mul_627, 1);  mul_627 = None
    mul_628: "f32[8, 960, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_75, add_322);  sigmoid_75 = add_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_376: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_377: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_388: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_389: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_14: "f32[8, 480, 14, 14]" = torch.ops.aten.full.default([8, 480, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_400: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_401: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
    unsqueeze_402: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_78: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_224)
    sub_137: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_78)
    mul_667: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_224, sub_137);  add_224 = sub_137 = None
    add_326: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_667, 1);  mul_667 = None
    mul_668: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_78, add_326);  sigmoid_78 = add_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_412: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_413: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
    unsqueeze_414: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_424: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_425: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_436: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_437: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_81: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_208)
    sub_153: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_81)
    mul_707: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_208, sub_153);  add_208 = sub_153 = None
    add_331: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_707, 1);  mul_707 = None
    mul_708: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_81, add_331);  sigmoid_81 = add_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_448: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_449: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
    unsqueeze_450: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_460: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_461: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
    unsqueeze_462: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_472: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_473: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
    unsqueeze_474: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_84: "f32[8, 480, 14, 14]" = torch.ops.aten.sigmoid.default(add_192)
    sub_169: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_14, sigmoid_84);  full_default_14 = None
    mul_747: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_192, sub_169);  add_192 = sub_169 = None
    add_336: "f32[8, 480, 14, 14]" = torch.ops.aten.add.Scalar(mul_747, 1);  mul_747 = None
    mul_748: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_84, add_336);  sigmoid_84 = add_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_484: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_485: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
    unsqueeze_486: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_496: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_497: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default_23: "f32[8, 624, 14, 14]" = torch.ops.aten.full.default([8, 624, 14, 14], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_508: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_509: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_87: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_177)
    sub_185: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_87)
    mul_787: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_177, sub_185);  add_177 = sub_185 = None
    add_341: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_787, 1);  mul_787 = None
    mul_788: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_87, add_341);  sigmoid_87 = add_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_520: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_521: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_532: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_533: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_544: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_545: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_90: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_161)
    sub_201: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_90)
    mul_827: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_161, sub_201);  add_161 = sub_201 = None
    add_345: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_827, 1);  mul_827 = None
    mul_828: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_90, add_345);  sigmoid_90 = add_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_556: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_557: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_568: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_569: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_580: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_581: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_93: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_145)
    sub_217: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_93)
    mul_867: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_145, sub_217);  add_145 = sub_217 = None
    add_350: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_867, 1);  mul_867 = None
    mul_868: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_93, add_350);  sigmoid_93 = add_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_592: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_593: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_604: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_605: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_616: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_617: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_96: "f32[8, 624, 14, 14]" = torch.ops.aten.sigmoid.default(add_129)
    sub_233: "f32[8, 624, 14, 14]" = torch.ops.aten.sub.Tensor(full_default_23, sigmoid_96);  full_default_23 = None
    mul_907: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(add_129, sub_233);  add_129 = sub_233 = None
    add_355: "f32[8, 624, 14, 14]" = torch.ops.aten.add.Scalar(mul_907, 1);  mul_907 = None
    mul_908: "f32[8, 624, 14, 14]" = torch.ops.aten.mul.Tensor(sigmoid_96, add_355);  sigmoid_96 = add_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_628: "f32[1, 624]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_629: "f32[1, 624, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 624, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_640: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_641: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_652: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_653: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_99: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_114)
    full_default_36: "f32[8, 336, 28, 28]" = torch.ops.aten.full.default([8, 336, 28, 28], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_249: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_36, sigmoid_99)
    mul_947: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_114, sub_249);  add_114 = sub_249 = None
    add_360: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_947, 1);  mul_947 = None
    mul_948: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_99, add_360);  sigmoid_99 = add_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_664: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_665: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_676: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_677: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_688: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_689: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_102: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_98)
    sub_265: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_36, sigmoid_102)
    mul_987: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_98, sub_265);  add_98 = sub_265 = None
    add_364: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_987, 1);  mul_987 = None
    mul_988: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_102, add_364);  sigmoid_102 = add_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_700: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_701: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_712: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_713: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_724: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_725: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_105: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_82)
    sub_281: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_36, sigmoid_105)
    mul_1027: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_82, sub_281);  add_82 = sub_281 = None
    add_369: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_1027, 1);  mul_1027 = None
    mul_1028: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_105, add_369);  sigmoid_105 = add_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_736: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_737: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
    unsqueeze_738: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_748: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_749: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
    unsqueeze_750: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_760: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_761: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
    unsqueeze_762: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_108: "f32[8, 336, 28, 28]" = torch.ops.aten.sigmoid.default(add_66)
    sub_297: "f32[8, 336, 28, 28]" = torch.ops.aten.sub.Tensor(full_default_36, sigmoid_108);  full_default_36 = None
    mul_1067: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(add_66, sub_297);  add_66 = sub_297 = None
    add_374: "f32[8, 336, 28, 28]" = torch.ops.aten.add.Scalar(mul_1067, 1);  mul_1067 = None
    mul_1068: "f32[8, 336, 28, 28]" = torch.ops.aten.mul.Tensor(sigmoid_108, add_374);  sigmoid_108 = add_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_772: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_773: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
    unsqueeze_774: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_784: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_785: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
    unsqueeze_786: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_796: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_797: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
    unsqueeze_798: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    sigmoid_111: "f32[8, 240, 56, 56]" = torch.ops.aten.sigmoid.default(add_51)
    full_default_48: "f32[8, 240, 56, 56]" = torch.ops.aten.full.default([8, 240, 56, 56], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    sub_313: "f32[8, 240, 56, 56]" = torch.ops.aten.sub.Tensor(full_default_48, sigmoid_111);  full_default_48 = None
    mul_1107: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(add_51, sub_313);  add_51 = sub_313 = None
    add_379: "f32[8, 240, 56, 56]" = torch.ops.aten.add.Scalar(mul_1107, 1);  mul_1107 = None
    mul_1108: "f32[8, 240, 56, 56]" = torch.ops.aten.mul.Tensor(sigmoid_111, add_379);  sigmoid_111 = add_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_808: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_809: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
    unsqueeze_810: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_820: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_821: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
    unsqueeze_822: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_43: "f32[8, 120, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_44: "f32[8, 120, 56, 56]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    le_1: "b8[8, 120, 56, 56]" = torch.ops.aten.le.Scalar(alias_44, 0);  alias_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_832: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_833: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
    unsqueeze_834: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_844: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_845: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
    unsqueeze_846: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_856: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_857: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
    unsqueeze_858: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_49: "f32[8, 192, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_50: "f32[8, 192, 56, 56]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    le_3: "b8[8, 192, 56, 56]" = torch.ops.aten.le.Scalar(alias_50, 0);  alias_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_868: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_869: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
    unsqueeze_870: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_4: "b8[8, 192, 112, 112]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_880: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_881: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
    unsqueeze_882: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_892: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_893: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
    unsqueeze_894: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_904: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_905: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
    unsqueeze_906: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_916: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_917: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
    unsqueeze_918: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_306, add);  primals_306 = add = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_307, add_2);  primals_307 = add_2 = None
    copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_308, add_3);  primals_308 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_5);  primals_309 = add_5 = None
    copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_310, add_7);  primals_310 = add_7 = None
    copy__5: "f32[32]" = torch.ops.aten.copy_.default(primals_311, add_8);  primals_311 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_312, add_10);  primals_312 = add_10 = None
    copy__7: "f32[32]" = torch.ops.aten.copy_.default(primals_313, add_12);  primals_313 = add_12 = None
    copy__8: "f32[32]" = torch.ops.aten.copy_.default(primals_314, add_13);  primals_314 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_315, add_16);  primals_315 = add_16 = None
    copy__10: "f32[192]" = torch.ops.aten.copy_.default(primals_316, add_18);  primals_316 = add_18 = None
    copy__11: "f32[192]" = torch.ops.aten.copy_.default(primals_317, add_19);  primals_317 = add_19 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_318, add_21);  primals_318 = add_21 = None
    copy__13: "f32[192]" = torch.ops.aten.copy_.default(primals_319, add_23);  primals_319 = add_23 = None
    copy__14: "f32[192]" = torch.ops.aten.copy_.default(primals_320, add_24);  primals_320 = add_24 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_321, add_26);  primals_321 = add_26 = None
    copy__16: "f32[40]" = torch.ops.aten.copy_.default(primals_322, add_28);  primals_322 = add_28 = None
    copy__17: "f32[40]" = torch.ops.aten.copy_.default(primals_323, add_29);  primals_323 = add_29 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_324, add_31);  primals_324 = add_31 = None
    copy__19: "f32[120]" = torch.ops.aten.copy_.default(primals_325, add_33);  primals_325 = add_33 = None
    copy__20: "f32[120]" = torch.ops.aten.copy_.default(primals_326, add_34);  primals_326 = add_34 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_327, add_36);  primals_327 = add_36 = None
    copy__22: "f32[120]" = torch.ops.aten.copy_.default(primals_328, add_38);  primals_328 = add_38 = None
    copy__23: "f32[120]" = torch.ops.aten.copy_.default(primals_329, add_39);  primals_329 = add_39 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_330, add_41);  primals_330 = add_41 = None
    copy__25: "f32[40]" = torch.ops.aten.copy_.default(primals_331, add_43);  primals_331 = add_43 = None
    copy__26: "f32[40]" = torch.ops.aten.copy_.default(primals_332, add_44);  primals_332 = add_44 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_333, add_47);  primals_333 = add_47 = None
    copy__28: "f32[240]" = torch.ops.aten.copy_.default(primals_334, add_49);  primals_334 = add_49 = None
    copy__29: "f32[240]" = torch.ops.aten.copy_.default(primals_335, add_50);  primals_335 = add_50 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_336, add_52);  primals_336 = add_52 = None
    copy__31: "f32[240]" = torch.ops.aten.copy_.default(primals_337, add_54);  primals_337 = add_54 = None
    copy__32: "f32[240]" = torch.ops.aten.copy_.default(primals_338, add_55);  primals_338 = add_55 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_339, add_57);  primals_339 = add_57 = None
    copy__34: "f32[56]" = torch.ops.aten.copy_.default(primals_340, add_59);  primals_340 = add_59 = None
    copy__35: "f32[56]" = torch.ops.aten.copy_.default(primals_341, add_60);  primals_341 = add_60 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_342, add_62);  primals_342 = add_62 = None
    copy__37: "f32[336]" = torch.ops.aten.copy_.default(primals_343, add_64);  primals_343 = add_64 = None
    copy__38: "f32[336]" = torch.ops.aten.copy_.default(primals_344, add_65);  primals_344 = add_65 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_345, add_67);  primals_345 = add_67 = None
    copy__40: "f32[336]" = torch.ops.aten.copy_.default(primals_346, add_69);  primals_346 = add_69 = None
    copy__41: "f32[336]" = torch.ops.aten.copy_.default(primals_347, add_70);  primals_347 = add_70 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_348, add_72);  primals_348 = add_72 = None
    copy__43: "f32[56]" = torch.ops.aten.copy_.default(primals_349, add_74);  primals_349 = add_74 = None
    copy__44: "f32[56]" = torch.ops.aten.copy_.default(primals_350, add_75);  primals_350 = add_75 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_351, add_78);  primals_351 = add_78 = None
    copy__46: "f32[336]" = torch.ops.aten.copy_.default(primals_352, add_80);  primals_352 = add_80 = None
    copy__47: "f32[336]" = torch.ops.aten.copy_.default(primals_353, add_81);  primals_353 = add_81 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_354, add_83);  primals_354 = add_83 = None
    copy__49: "f32[336]" = torch.ops.aten.copy_.default(primals_355, add_85);  primals_355 = add_85 = None
    copy__50: "f32[336]" = torch.ops.aten.copy_.default(primals_356, add_86);  primals_356 = add_86 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_357, add_88);  primals_357 = add_88 = None
    copy__52: "f32[56]" = torch.ops.aten.copy_.default(primals_358, add_90);  primals_358 = add_90 = None
    copy__53: "f32[56]" = torch.ops.aten.copy_.default(primals_359, add_91);  primals_359 = add_91 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_360, add_94);  primals_360 = add_94 = None
    copy__55: "f32[336]" = torch.ops.aten.copy_.default(primals_361, add_96);  primals_361 = add_96 = None
    copy__56: "f32[336]" = torch.ops.aten.copy_.default(primals_362, add_97);  primals_362 = add_97 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_363, add_99);  primals_363 = add_99 = None
    copy__58: "f32[336]" = torch.ops.aten.copy_.default(primals_364, add_101);  primals_364 = add_101 = None
    copy__59: "f32[336]" = torch.ops.aten.copy_.default(primals_365, add_102);  primals_365 = add_102 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_366, add_104);  primals_366 = add_104 = None
    copy__61: "f32[56]" = torch.ops.aten.copy_.default(primals_367, add_106);  primals_367 = add_106 = None
    copy__62: "f32[56]" = torch.ops.aten.copy_.default(primals_368, add_107);  primals_368 = add_107 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_369, add_110);  primals_369 = add_110 = None
    copy__64: "f32[336]" = torch.ops.aten.copy_.default(primals_370, add_112);  primals_370 = add_112 = None
    copy__65: "f32[336]" = torch.ops.aten.copy_.default(primals_371, add_113);  primals_371 = add_113 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_372, add_115);  primals_372 = add_115 = None
    copy__67: "f32[336]" = torch.ops.aten.copy_.default(primals_373, add_117);  primals_373 = add_117 = None
    copy__68: "f32[336]" = torch.ops.aten.copy_.default(primals_374, add_118);  primals_374 = add_118 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_375, add_120);  primals_375 = add_120 = None
    copy__70: "f32[104]" = torch.ops.aten.copy_.default(primals_376, add_122);  primals_376 = add_122 = None
    copy__71: "f32[104]" = torch.ops.aten.copy_.default(primals_377, add_123);  primals_377 = add_123 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_378, add_125);  primals_378 = add_125 = None
    copy__73: "f32[624]" = torch.ops.aten.copy_.default(primals_379, add_127);  primals_379 = add_127 = None
    copy__74: "f32[624]" = torch.ops.aten.copy_.default(primals_380, add_128);  primals_380 = add_128 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_381, add_130);  primals_381 = add_130 = None
    copy__76: "f32[624]" = torch.ops.aten.copy_.default(primals_382, add_132);  primals_382 = add_132 = None
    copy__77: "f32[624]" = torch.ops.aten.copy_.default(primals_383, add_133);  primals_383 = add_133 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_384, add_135);  primals_384 = add_135 = None
    copy__79: "f32[104]" = torch.ops.aten.copy_.default(primals_385, add_137);  primals_385 = add_137 = None
    copy__80: "f32[104]" = torch.ops.aten.copy_.default(primals_386, add_138);  primals_386 = add_138 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_387, add_141);  primals_387 = add_141 = None
    copy__82: "f32[624]" = torch.ops.aten.copy_.default(primals_388, add_143);  primals_388 = add_143 = None
    copy__83: "f32[624]" = torch.ops.aten.copy_.default(primals_389, add_144);  primals_389 = add_144 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_390, add_146);  primals_390 = add_146 = None
    copy__85: "f32[624]" = torch.ops.aten.copy_.default(primals_391, add_148);  primals_391 = add_148 = None
    copy__86: "f32[624]" = torch.ops.aten.copy_.default(primals_392, add_149);  primals_392 = add_149 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_393, add_151);  primals_393 = add_151 = None
    copy__88: "f32[104]" = torch.ops.aten.copy_.default(primals_394, add_153);  primals_394 = add_153 = None
    copy__89: "f32[104]" = torch.ops.aten.copy_.default(primals_395, add_154);  primals_395 = add_154 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_396, add_157);  primals_396 = add_157 = None
    copy__91: "f32[624]" = torch.ops.aten.copy_.default(primals_397, add_159);  primals_397 = add_159 = None
    copy__92: "f32[624]" = torch.ops.aten.copy_.default(primals_398, add_160);  primals_398 = add_160 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_399, add_162);  primals_399 = add_162 = None
    copy__94: "f32[624]" = torch.ops.aten.copy_.default(primals_400, add_164);  primals_400 = add_164 = None
    copy__95: "f32[624]" = torch.ops.aten.copy_.default(primals_401, add_165);  primals_401 = add_165 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_402, add_167);  primals_402 = add_167 = None
    copy__97: "f32[104]" = torch.ops.aten.copy_.default(primals_403, add_169);  primals_403 = add_169 = None
    copy__98: "f32[104]" = torch.ops.aten.copy_.default(primals_404, add_170);  primals_404 = add_170 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_405, add_173);  primals_405 = add_173 = None
    copy__100: "f32[624]" = torch.ops.aten.copy_.default(primals_406, add_175);  primals_406 = add_175 = None
    copy__101: "f32[624]" = torch.ops.aten.copy_.default(primals_407, add_176);  primals_407 = add_176 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_408, add_178);  primals_408 = add_178 = None
    copy__103: "f32[624]" = torch.ops.aten.copy_.default(primals_409, add_180);  primals_409 = add_180 = None
    copy__104: "f32[624]" = torch.ops.aten.copy_.default(primals_410, add_181);  primals_410 = add_181 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_411, add_183);  primals_411 = add_183 = None
    copy__106: "f32[160]" = torch.ops.aten.copy_.default(primals_412, add_185);  primals_412 = add_185 = None
    copy__107: "f32[160]" = torch.ops.aten.copy_.default(primals_413, add_186);  primals_413 = add_186 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_414, add_188);  primals_414 = add_188 = None
    copy__109: "f32[480]" = torch.ops.aten.copy_.default(primals_415, add_190);  primals_415 = add_190 = None
    copy__110: "f32[480]" = torch.ops.aten.copy_.default(primals_416, add_191);  primals_416 = add_191 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_417, add_193);  primals_417 = add_193 = None
    copy__112: "f32[480]" = torch.ops.aten.copy_.default(primals_418, add_195);  primals_418 = add_195 = None
    copy__113: "f32[480]" = torch.ops.aten.copy_.default(primals_419, add_196);  primals_419 = add_196 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_420, add_198);  primals_420 = add_198 = None
    copy__115: "f32[160]" = torch.ops.aten.copy_.default(primals_421, add_200);  primals_421 = add_200 = None
    copy__116: "f32[160]" = torch.ops.aten.copy_.default(primals_422, add_201);  primals_422 = add_201 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_423, add_204);  primals_423 = add_204 = None
    copy__118: "f32[480]" = torch.ops.aten.copy_.default(primals_424, add_206);  primals_424 = add_206 = None
    copy__119: "f32[480]" = torch.ops.aten.copy_.default(primals_425, add_207);  primals_425 = add_207 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_426, add_209);  primals_426 = add_209 = None
    copy__121: "f32[480]" = torch.ops.aten.copy_.default(primals_427, add_211);  primals_427 = add_211 = None
    copy__122: "f32[480]" = torch.ops.aten.copy_.default(primals_428, add_212);  primals_428 = add_212 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_429, add_214);  primals_429 = add_214 = None
    copy__124: "f32[160]" = torch.ops.aten.copy_.default(primals_430, add_216);  primals_430 = add_216 = None
    copy__125: "f32[160]" = torch.ops.aten.copy_.default(primals_431, add_217);  primals_431 = add_217 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_432, add_220);  primals_432 = add_220 = None
    copy__127: "f32[480]" = torch.ops.aten.copy_.default(primals_433, add_222);  primals_433 = add_222 = None
    copy__128: "f32[480]" = torch.ops.aten.copy_.default(primals_434, add_223);  primals_434 = add_223 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_435, add_225);  primals_435 = add_225 = None
    copy__130: "f32[480]" = torch.ops.aten.copy_.default(primals_436, add_227);  primals_436 = add_227 = None
    copy__131: "f32[480]" = torch.ops.aten.copy_.default(primals_437, add_228);  primals_437 = add_228 = None
    copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_438, add_230);  primals_438 = add_230 = None
    copy__133: "f32[160]" = torch.ops.aten.copy_.default(primals_439, add_232);  primals_439 = add_232 = None
    copy__134: "f32[160]" = torch.ops.aten.copy_.default(primals_440, add_233);  primals_440 = add_233 = None
    copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_441, add_236);  primals_441 = add_236 = None
    copy__136: "f32[960]" = torch.ops.aten.copy_.default(primals_442, add_238);  primals_442 = add_238 = None
    copy__137: "f32[960]" = torch.ops.aten.copy_.default(primals_443, add_239);  primals_443 = add_239 = None
    copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_444, add_241);  primals_444 = add_241 = None
    copy__139: "f32[960]" = torch.ops.aten.copy_.default(primals_445, add_243);  primals_445 = add_243 = None
    copy__140: "f32[960]" = torch.ops.aten.copy_.default(primals_446, add_244);  primals_446 = add_244 = None
    copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_447, add_246);  primals_447 = add_246 = None
    copy__142: "f32[264]" = torch.ops.aten.copy_.default(primals_448, add_248);  primals_448 = add_248 = None
    copy__143: "f32[264]" = torch.ops.aten.copy_.default(primals_449, add_249);  primals_449 = add_249 = None
    copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_450, add_251);  primals_450 = add_251 = None
    copy__145: "f32[1584]" = torch.ops.aten.copy_.default(primals_451, add_253);  primals_451 = add_253 = None
    copy__146: "f32[1584]" = torch.ops.aten.copy_.default(primals_452, add_254);  primals_452 = add_254 = None
    copy__147: "i64[]" = torch.ops.aten.copy_.default(primals_453, add_256);  primals_453 = add_256 = None
    copy__148: "f32[1584]" = torch.ops.aten.copy_.default(primals_454, add_258);  primals_454 = add_258 = None
    copy__149: "f32[1584]" = torch.ops.aten.copy_.default(primals_455, add_259);  primals_455 = add_259 = None
    copy__150: "i64[]" = torch.ops.aten.copy_.default(primals_456, add_261);  primals_456 = add_261 = None
    copy__151: "f32[264]" = torch.ops.aten.copy_.default(primals_457, add_263);  primals_457 = add_263 = None
    copy__152: "f32[264]" = torch.ops.aten.copy_.default(primals_458, add_264);  primals_458 = add_264 = None
    copy__153: "i64[]" = torch.ops.aten.copy_.default(primals_459, add_267);  primals_459 = add_267 = None
    copy__154: "f32[1584]" = torch.ops.aten.copy_.default(primals_460, add_269);  primals_460 = add_269 = None
    copy__155: "f32[1584]" = torch.ops.aten.copy_.default(primals_461, add_270);  primals_461 = add_270 = None
    copy__156: "i64[]" = torch.ops.aten.copy_.default(primals_462, add_272);  primals_462 = add_272 = None
    copy__157: "f32[1584]" = torch.ops.aten.copy_.default(primals_463, add_274);  primals_463 = add_274 = None
    copy__158: "f32[1584]" = torch.ops.aten.copy_.default(primals_464, add_275);  primals_464 = add_275 = None
    copy__159: "i64[]" = torch.ops.aten.copy_.default(primals_465, add_277);  primals_465 = add_277 = None
    copy__160: "f32[264]" = torch.ops.aten.copy_.default(primals_466, add_279);  primals_466 = add_279 = None
    copy__161: "f32[264]" = torch.ops.aten.copy_.default(primals_467, add_280);  primals_467 = add_280 = None
    copy__162: "i64[]" = torch.ops.aten.copy_.default(primals_468, add_283);  primals_468 = add_283 = None
    copy__163: "f32[1584]" = torch.ops.aten.copy_.default(primals_469, add_285);  primals_469 = add_285 = None
    copy__164: "f32[1584]" = torch.ops.aten.copy_.default(primals_470, add_286);  primals_470 = add_286 = None
    copy__165: "i64[]" = torch.ops.aten.copy_.default(primals_471, add_288);  primals_471 = add_288 = None
    copy__166: "f32[1584]" = torch.ops.aten.copy_.default(primals_472, add_290);  primals_472 = add_290 = None
    copy__167: "f32[1584]" = torch.ops.aten.copy_.default(primals_473, add_291);  primals_473 = add_291 = None
    copy__168: "i64[]" = torch.ops.aten.copy_.default(primals_474, add_293);  primals_474 = add_293 = None
    copy__169: "f32[264]" = torch.ops.aten.copy_.default(primals_475, add_295);  primals_475 = add_295 = None
    copy__170: "f32[264]" = torch.ops.aten.copy_.default(primals_476, add_296);  primals_476 = add_296 = None
    copy__171: "i64[]" = torch.ops.aten.copy_.default(primals_477, add_299);  primals_477 = add_299 = None
    copy__172: "f32[1536]" = torch.ops.aten.copy_.default(primals_478, add_301);  primals_478 = add_301 = None
    copy__173: "f32[1536]" = torch.ops.aten.copy_.default(primals_479, add_302);  primals_479 = add_302 = None
    return [addmm, primals_1, primals_2, primals_4, primals_6, primals_8, primals_10, primals_11, primals_12, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_26, primals_27, primals_28, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_54, primals_55, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_146, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_155, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_193, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_205, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_217, primals_219, primals_220, primals_221, primals_222, primals_223, primals_225, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_236, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_248, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_260, primals_262, primals_263, primals_264, primals_265, primals_267, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_288, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_299, primals_301, primals_302, primals_303, constant_pad_nd, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, getitem_6, getitem_7, cat, squeeze_10, constant_pad_nd_1, constant_pad_nd_2, constant_pad_nd_3, cat_1, squeeze_13, getitem_26, getitem_29, cat_2, squeeze_16, getitem_32, getitem_33, cat_3, squeeze_19, relu_4, convolution_12, squeeze_22, getitem_40, getitem_43, cat_4, squeeze_25, add_46, convolution_15, squeeze_28, constant_pad_nd_4, constant_pad_nd_5, constant_pad_nd_6, constant_pad_nd_7, cat_5, squeeze_31, add_56, mean, convolution_20, mul_79, convolution_21, mul_80, convolution_22, squeeze_34, getitem_72, getitem_73, cat_6, squeeze_37, getitem_78, getitem_81, cat_7, squeeze_40, add_71, mean_1, convolution_27, mul_104, convolution_28, getitem_84, getitem_85, cat_8, squeeze_43, getitem_88, getitem_89, cat_9, squeeze_46, getitem_94, getitem_97, cat_10, squeeze_49, add_87, mean_2, convolution_35, mul_129, convolution_36, getitem_100, getitem_101, cat_11, squeeze_52, getitem_104, getitem_105, cat_12, squeeze_55, getitem_110, getitem_113, cat_13, squeeze_58, add_103, mean_3, convolution_43, mul_154, convolution_44, getitem_116, getitem_117, cat_14, squeeze_61, add_109, convolution_47, squeeze_64, constant_pad_nd_8, constant_pad_nd_9, constant_pad_nd_10, cat_15, squeeze_67, add_119, mean_4, convolution_51, mul_179, convolution_52, mul_180, convolution_53, squeeze_70, getitem_138, getitem_139, cat_16, squeeze_73, getitem_146, getitem_151, getitem_156, getitem_161, cat_17, squeeze_76, add_134, mean_5, convolution_60, mul_204, convolution_61, getitem_164, getitem_165, cat_18, squeeze_79, getitem_168, getitem_169, cat_19, squeeze_82, getitem_176, getitem_181, getitem_186, getitem_191, cat_20, squeeze_85, add_150, mean_6, convolution_70, mul_229, convolution_71, getitem_194, getitem_195, cat_21, squeeze_88, getitem_198, getitem_199, cat_22, squeeze_91, getitem_206, getitem_211, getitem_216, getitem_221, cat_23, squeeze_94, add_166, mean_7, convolution_80, mul_254, convolution_81, getitem_224, getitem_225, cat_24, squeeze_97, add_172, convolution_84, squeeze_100, mul_270, convolution_85, squeeze_103, add_182, mean_8, convolution_86, mul_279, convolution_87, mul_280, convolution_88, squeeze_106, getitem_234, getitem_235, cat_25, squeeze_109, getitem_242, getitem_247, getitem_252, getitem_257, cat_26, squeeze_112, add_197, mean_9, convolution_95, mul_304, convolution_96, getitem_260, getitem_261, cat_27, squeeze_115, getitem_264, getitem_265, cat_28, squeeze_118, getitem_272, getitem_277, getitem_282, getitem_287, cat_29, squeeze_121, add_213, mean_10, convolution_105, mul_329, convolution_106, getitem_290, getitem_291, cat_30, squeeze_124, getitem_294, getitem_295, cat_31, squeeze_127, getitem_302, getitem_307, getitem_312, getitem_317, cat_32, squeeze_130, add_229, mean_11, convolution_115, mul_354, convolution_116, getitem_320, getitem_321, cat_33, squeeze_133, add_235, convolution_119, squeeze_136, constant_pad_nd_11, constant_pad_nd_12, constant_pad_nd_13, constant_pad_nd_14, cat_34, squeeze_139, add_245, mean_12, convolution_124, mul_379, convolution_125, mul_380, convolution_126, squeeze_142, add_250, convolution_127, squeeze_145, getitem_356, getitem_361, getitem_366, getitem_371, cat_35, squeeze_148, add_260, mean_13, convolution_132, mul_404, convolution_133, getitem_374, getitem_375, cat_36, squeeze_151, add_266, convolution_136, squeeze_154, getitem_384, getitem_389, getitem_394, getitem_399, cat_37, squeeze_157, add_276, mean_14, convolution_141, mul_429, convolution_142, getitem_402, getitem_403, cat_38, squeeze_160, add_282, convolution_145, squeeze_163, getitem_412, getitem_417, getitem_422, getitem_427, cat_39, squeeze_166, add_292, mean_15, convolution_150, mul_454, convolution_151, getitem_430, getitem_431, cat_40, squeeze_169, add_298, convolution_154, squeeze_172, view, permute_1, le, unsqueeze_234, unsqueeze_246, unsqueeze_258, mul_508, unsqueeze_270, unsqueeze_282, unsqueeze_294, mul_548, unsqueeze_306, unsqueeze_318, unsqueeze_330, mul_588, unsqueeze_342, unsqueeze_354, unsqueeze_366, mul_628, unsqueeze_378, unsqueeze_390, unsqueeze_402, mul_668, unsqueeze_414, unsqueeze_426, unsqueeze_438, mul_708, unsqueeze_450, unsqueeze_462, unsqueeze_474, mul_748, unsqueeze_486, unsqueeze_498, unsqueeze_510, mul_788, unsqueeze_522, unsqueeze_534, unsqueeze_546, mul_828, unsqueeze_558, unsqueeze_570, unsqueeze_582, mul_868, unsqueeze_594, unsqueeze_606, unsqueeze_618, mul_908, unsqueeze_630, unsqueeze_642, unsqueeze_654, mul_948, unsqueeze_666, unsqueeze_678, unsqueeze_690, mul_988, unsqueeze_702, unsqueeze_714, unsqueeze_726, mul_1028, unsqueeze_738, unsqueeze_750, unsqueeze_762, mul_1068, unsqueeze_774, unsqueeze_786, unsqueeze_798, mul_1108, unsqueeze_810, unsqueeze_822, le_1, unsqueeze_834, unsqueeze_846, unsqueeze_858, le_3, unsqueeze_870, le_4, unsqueeze_882, unsqueeze_894, unsqueeze_906, unsqueeze_918]
    