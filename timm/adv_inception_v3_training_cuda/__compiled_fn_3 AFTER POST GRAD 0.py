from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_2: "f32[32]", primals_3: "f32[32]", primals_4: "f32[32]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[80]", primals_8: "f32[80]", primals_9: "f32[192]", primals_10: "f32[192]", primals_11: "f32[64]", primals_12: "f32[64]", primals_13: "f32[48]", primals_14: "f32[48]", primals_15: "f32[64]", primals_16: "f32[64]", primals_17: "f32[64]", primals_18: "f32[64]", primals_19: "f32[96]", primals_20: "f32[96]", primals_21: "f32[96]", primals_22: "f32[96]", primals_23: "f32[32]", primals_24: "f32[32]", primals_25: "f32[64]", primals_26: "f32[64]", primals_27: "f32[48]", primals_28: "f32[48]", primals_29: "f32[64]", primals_30: "f32[64]", primals_31: "f32[64]", primals_32: "f32[64]", primals_33: "f32[96]", primals_34: "f32[96]", primals_35: "f32[96]", primals_36: "f32[96]", primals_37: "f32[64]", primals_38: "f32[64]", primals_39: "f32[64]", primals_40: "f32[64]", primals_41: "f32[48]", primals_42: "f32[48]", primals_43: "f32[64]", primals_44: "f32[64]", primals_45: "f32[64]", primals_46: "f32[64]", primals_47: "f32[96]", primals_48: "f32[96]", primals_49: "f32[96]", primals_50: "f32[96]", primals_51: "f32[64]", primals_52: "f32[64]", primals_53: "f32[384]", primals_54: "f32[384]", primals_55: "f32[64]", primals_56: "f32[64]", primals_57: "f32[96]", primals_58: "f32[96]", primals_59: "f32[96]", primals_60: "f32[96]", primals_61: "f32[192]", primals_62: "f32[192]", primals_63: "f32[128]", primals_64: "f32[128]", primals_65: "f32[128]", primals_66: "f32[128]", primals_67: "f32[192]", primals_68: "f32[192]", primals_69: "f32[128]", primals_70: "f32[128]", primals_71: "f32[128]", primals_72: "f32[128]", primals_73: "f32[128]", primals_74: "f32[128]", primals_75: "f32[128]", primals_76: "f32[128]", primals_77: "f32[192]", primals_78: "f32[192]", primals_79: "f32[192]", primals_80: "f32[192]", primals_81: "f32[192]", primals_82: "f32[192]", primals_83: "f32[160]", primals_84: "f32[160]", primals_85: "f32[160]", primals_86: "f32[160]", primals_87: "f32[192]", primals_88: "f32[192]", primals_89: "f32[160]", primals_90: "f32[160]", primals_91: "f32[160]", primals_92: "f32[160]", primals_93: "f32[160]", primals_94: "f32[160]", primals_95: "f32[160]", primals_96: "f32[160]", primals_97: "f32[192]", primals_98: "f32[192]", primals_99: "f32[192]", primals_100: "f32[192]", primals_101: "f32[192]", primals_102: "f32[192]", primals_103: "f32[160]", primals_104: "f32[160]", primals_105: "f32[160]", primals_106: "f32[160]", primals_107: "f32[192]", primals_108: "f32[192]", primals_109: "f32[160]", primals_110: "f32[160]", primals_111: "f32[160]", primals_112: "f32[160]", primals_113: "f32[160]", primals_114: "f32[160]", primals_115: "f32[160]", primals_116: "f32[160]", primals_117: "f32[192]", primals_118: "f32[192]", primals_119: "f32[192]", primals_120: "f32[192]", primals_121: "f32[192]", primals_122: "f32[192]", primals_123: "f32[192]", primals_124: "f32[192]", primals_125: "f32[192]", primals_126: "f32[192]", primals_127: "f32[192]", primals_128: "f32[192]", primals_129: "f32[192]", primals_130: "f32[192]", primals_131: "f32[192]", primals_132: "f32[192]", primals_133: "f32[192]", primals_134: "f32[192]", primals_135: "f32[192]", primals_136: "f32[192]", primals_137: "f32[192]", primals_138: "f32[192]", primals_139: "f32[192]", primals_140: "f32[192]", primals_141: "f32[192]", primals_142: "f32[192]", primals_143: "f32[320]", primals_144: "f32[320]", primals_145: "f32[192]", primals_146: "f32[192]", primals_147: "f32[192]", primals_148: "f32[192]", primals_149: "f32[192]", primals_150: "f32[192]", primals_151: "f32[192]", primals_152: "f32[192]", primals_153: "f32[320]", primals_154: "f32[320]", primals_155: "f32[384]", primals_156: "f32[384]", primals_157: "f32[384]", primals_158: "f32[384]", primals_159: "f32[384]", primals_160: "f32[384]", primals_161: "f32[448]", primals_162: "f32[448]", primals_163: "f32[384]", primals_164: "f32[384]", primals_165: "f32[384]", primals_166: "f32[384]", primals_167: "f32[384]", primals_168: "f32[384]", primals_169: "f32[192]", primals_170: "f32[192]", primals_171: "f32[320]", primals_172: "f32[320]", primals_173: "f32[384]", primals_174: "f32[384]", primals_175: "f32[384]", primals_176: "f32[384]", primals_177: "f32[384]", primals_178: "f32[384]", primals_179: "f32[448]", primals_180: "f32[448]", primals_181: "f32[384]", primals_182: "f32[384]", primals_183: "f32[384]", primals_184: "f32[384]", primals_185: "f32[384]", primals_186: "f32[384]", primals_187: "f32[192]", primals_188: "f32[192]", primals_189: "f32[32, 3, 3, 3]", primals_190: "f32[32, 32, 3, 3]", primals_191: "f32[64, 32, 3, 3]", primals_192: "f32[80, 64, 1, 1]", primals_193: "f32[192, 80, 3, 3]", primals_194: "f32[64, 192, 1, 1]", primals_195: "f32[48, 192, 1, 1]", primals_196: "f32[64, 48, 5, 5]", primals_197: "f32[64, 192, 1, 1]", primals_198: "f32[96, 64, 3, 3]", primals_199: "f32[96, 96, 3, 3]", primals_200: "f32[32, 192, 1, 1]", primals_201: "f32[64, 256, 1, 1]", primals_202: "f32[48, 256, 1, 1]", primals_203: "f32[64, 48, 5, 5]", primals_204: "f32[64, 256, 1, 1]", primals_205: "f32[96, 64, 3, 3]", primals_206: "f32[96, 96, 3, 3]", primals_207: "f32[64, 256, 1, 1]", primals_208: "f32[64, 288, 1, 1]", primals_209: "f32[48, 288, 1, 1]", primals_210: "f32[64, 48, 5, 5]", primals_211: "f32[64, 288, 1, 1]", primals_212: "f32[96, 64, 3, 3]", primals_213: "f32[96, 96, 3, 3]", primals_214: "f32[64, 288, 1, 1]", primals_215: "f32[384, 288, 3, 3]", primals_216: "f32[64, 288, 1, 1]", primals_217: "f32[96, 64, 3, 3]", primals_218: "f32[96, 96, 3, 3]", primals_219: "f32[192, 768, 1, 1]", primals_220: "f32[128, 768, 1, 1]", primals_221: "f32[128, 128, 1, 7]", primals_222: "f32[192, 128, 7, 1]", primals_223: "f32[128, 768, 1, 1]", primals_224: "f32[128, 128, 7, 1]", primals_225: "f32[128, 128, 1, 7]", primals_226: "f32[128, 128, 7, 1]", primals_227: "f32[192, 128, 1, 7]", primals_228: "f32[192, 768, 1, 1]", primals_229: "f32[192, 768, 1, 1]", primals_230: "f32[160, 768, 1, 1]", primals_231: "f32[160, 160, 1, 7]", primals_232: "f32[192, 160, 7, 1]", primals_233: "f32[160, 768, 1, 1]", primals_234: "f32[160, 160, 7, 1]", primals_235: "f32[160, 160, 1, 7]", primals_236: "f32[160, 160, 7, 1]", primals_237: "f32[192, 160, 1, 7]", primals_238: "f32[192, 768, 1, 1]", primals_239: "f32[192, 768, 1, 1]", primals_240: "f32[160, 768, 1, 1]", primals_241: "f32[160, 160, 1, 7]", primals_242: "f32[192, 160, 7, 1]", primals_243: "f32[160, 768, 1, 1]", primals_244: "f32[160, 160, 7, 1]", primals_245: "f32[160, 160, 1, 7]", primals_246: "f32[160, 160, 7, 1]", primals_247: "f32[192, 160, 1, 7]", primals_248: "f32[192, 768, 1, 1]", primals_249: "f32[192, 768, 1, 1]", primals_250: "f32[192, 768, 1, 1]", primals_251: "f32[192, 192, 1, 7]", primals_252: "f32[192, 192, 7, 1]", primals_253: "f32[192, 768, 1, 1]", primals_254: "f32[192, 192, 7, 1]", primals_255: "f32[192, 192, 1, 7]", primals_256: "f32[192, 192, 7, 1]", primals_257: "f32[192, 192, 1, 7]", primals_258: "f32[192, 768, 1, 1]", primals_259: "f32[192, 768, 1, 1]", primals_260: "f32[320, 192, 3, 3]", primals_261: "f32[192, 768, 1, 1]", primals_262: "f32[192, 192, 1, 7]", primals_263: "f32[192, 192, 7, 1]", primals_264: "f32[192, 192, 3, 3]", primals_265: "f32[320, 1280, 1, 1]", primals_266: "f32[384, 1280, 1, 1]", primals_267: "f32[384, 384, 1, 3]", primals_268: "f32[384, 384, 3, 1]", primals_269: "f32[448, 1280, 1, 1]", primals_270: "f32[384, 448, 3, 3]", primals_271: "f32[384, 384, 1, 3]", primals_272: "f32[384, 384, 3, 1]", primals_273: "f32[192, 1280, 1, 1]", primals_274: "f32[320, 2048, 1, 1]", primals_275: "f32[384, 2048, 1, 1]", primals_276: "f32[384, 384, 1, 3]", primals_277: "f32[384, 384, 3, 1]", primals_278: "f32[448, 2048, 1, 1]", primals_279: "f32[384, 448, 3, 3]", primals_280: "f32[384, 384, 1, 3]", primals_281: "f32[384, 384, 3, 1]", primals_282: "f32[192, 2048, 1, 1]", primals_283: "f32[1000, 2048]", primals_284: "f32[1000]", primals_285: "i64[]", primals_286: "f32[32]", primals_287: "f32[32]", primals_288: "i64[]", primals_289: "f32[32]", primals_290: "f32[32]", primals_291: "i64[]", primals_292: "f32[64]", primals_293: "f32[64]", primals_294: "i64[]", primals_295: "f32[80]", primals_296: "f32[80]", primals_297: "i64[]", primals_298: "f32[192]", primals_299: "f32[192]", primals_300: "i64[]", primals_301: "f32[64]", primals_302: "f32[64]", primals_303: "i64[]", primals_304: "f32[48]", primals_305: "f32[48]", primals_306: "i64[]", primals_307: "f32[64]", primals_308: "f32[64]", primals_309: "i64[]", primals_310: "f32[64]", primals_311: "f32[64]", primals_312: "i64[]", primals_313: "f32[96]", primals_314: "f32[96]", primals_315: "i64[]", primals_316: "f32[96]", primals_317: "f32[96]", primals_318: "i64[]", primals_319: "f32[32]", primals_320: "f32[32]", primals_321: "i64[]", primals_322: "f32[64]", primals_323: "f32[64]", primals_324: "i64[]", primals_325: "f32[48]", primals_326: "f32[48]", primals_327: "i64[]", primals_328: "f32[64]", primals_329: "f32[64]", primals_330: "i64[]", primals_331: "f32[64]", primals_332: "f32[64]", primals_333: "i64[]", primals_334: "f32[96]", primals_335: "f32[96]", primals_336: "i64[]", primals_337: "f32[96]", primals_338: "f32[96]", primals_339: "i64[]", primals_340: "f32[64]", primals_341: "f32[64]", primals_342: "i64[]", primals_343: "f32[64]", primals_344: "f32[64]", primals_345: "i64[]", primals_346: "f32[48]", primals_347: "f32[48]", primals_348: "i64[]", primals_349: "f32[64]", primals_350: "f32[64]", primals_351: "i64[]", primals_352: "f32[64]", primals_353: "f32[64]", primals_354: "i64[]", primals_355: "f32[96]", primals_356: "f32[96]", primals_357: "i64[]", primals_358: "f32[96]", primals_359: "f32[96]", primals_360: "i64[]", primals_361: "f32[64]", primals_362: "f32[64]", primals_363: "i64[]", primals_364: "f32[384]", primals_365: "f32[384]", primals_366: "i64[]", primals_367: "f32[64]", primals_368: "f32[64]", primals_369: "i64[]", primals_370: "f32[96]", primals_371: "f32[96]", primals_372: "i64[]", primals_373: "f32[96]", primals_374: "f32[96]", primals_375: "i64[]", primals_376: "f32[192]", primals_377: "f32[192]", primals_378: "i64[]", primals_379: "f32[128]", primals_380: "f32[128]", primals_381: "i64[]", primals_382: "f32[128]", primals_383: "f32[128]", primals_384: "i64[]", primals_385: "f32[192]", primals_386: "f32[192]", primals_387: "i64[]", primals_388: "f32[128]", primals_389: "f32[128]", primals_390: "i64[]", primals_391: "f32[128]", primals_392: "f32[128]", primals_393: "i64[]", primals_394: "f32[128]", primals_395: "f32[128]", primals_396: "i64[]", primals_397: "f32[128]", primals_398: "f32[128]", primals_399: "i64[]", primals_400: "f32[192]", primals_401: "f32[192]", primals_402: "i64[]", primals_403: "f32[192]", primals_404: "f32[192]", primals_405: "i64[]", primals_406: "f32[192]", primals_407: "f32[192]", primals_408: "i64[]", primals_409: "f32[160]", primals_410: "f32[160]", primals_411: "i64[]", primals_412: "f32[160]", primals_413: "f32[160]", primals_414: "i64[]", primals_415: "f32[192]", primals_416: "f32[192]", primals_417: "i64[]", primals_418: "f32[160]", primals_419: "f32[160]", primals_420: "i64[]", primals_421: "f32[160]", primals_422: "f32[160]", primals_423: "i64[]", primals_424: "f32[160]", primals_425: "f32[160]", primals_426: "i64[]", primals_427: "f32[160]", primals_428: "f32[160]", primals_429: "i64[]", primals_430: "f32[192]", primals_431: "f32[192]", primals_432: "i64[]", primals_433: "f32[192]", primals_434: "f32[192]", primals_435: "i64[]", primals_436: "f32[192]", primals_437: "f32[192]", primals_438: "i64[]", primals_439: "f32[160]", primals_440: "f32[160]", primals_441: "i64[]", primals_442: "f32[160]", primals_443: "f32[160]", primals_444: "i64[]", primals_445: "f32[192]", primals_446: "f32[192]", primals_447: "i64[]", primals_448: "f32[160]", primals_449: "f32[160]", primals_450: "i64[]", primals_451: "f32[160]", primals_452: "f32[160]", primals_453: "i64[]", primals_454: "f32[160]", primals_455: "f32[160]", primals_456: "i64[]", primals_457: "f32[160]", primals_458: "f32[160]", primals_459: "i64[]", primals_460: "f32[192]", primals_461: "f32[192]", primals_462: "i64[]", primals_463: "f32[192]", primals_464: "f32[192]", primals_465: "i64[]", primals_466: "f32[192]", primals_467: "f32[192]", primals_468: "i64[]", primals_469: "f32[192]", primals_470: "f32[192]", primals_471: "i64[]", primals_472: "f32[192]", primals_473: "f32[192]", primals_474: "i64[]", primals_475: "f32[192]", primals_476: "f32[192]", primals_477: "i64[]", primals_478: "f32[192]", primals_479: "f32[192]", primals_480: "i64[]", primals_481: "f32[192]", primals_482: "f32[192]", primals_483: "i64[]", primals_484: "f32[192]", primals_485: "f32[192]", primals_486: "i64[]", primals_487: "f32[192]", primals_488: "f32[192]", primals_489: "i64[]", primals_490: "f32[192]", primals_491: "f32[192]", primals_492: "i64[]", primals_493: "f32[192]", primals_494: "f32[192]", primals_495: "i64[]", primals_496: "f32[192]", primals_497: "f32[192]", primals_498: "i64[]", primals_499: "f32[320]", primals_500: "f32[320]", primals_501: "i64[]", primals_502: "f32[192]", primals_503: "f32[192]", primals_504: "i64[]", primals_505: "f32[192]", primals_506: "f32[192]", primals_507: "i64[]", primals_508: "f32[192]", primals_509: "f32[192]", primals_510: "i64[]", primals_511: "f32[192]", primals_512: "f32[192]", primals_513: "i64[]", primals_514: "f32[320]", primals_515: "f32[320]", primals_516: "i64[]", primals_517: "f32[384]", primals_518: "f32[384]", primals_519: "i64[]", primals_520: "f32[384]", primals_521: "f32[384]", primals_522: "i64[]", primals_523: "f32[384]", primals_524: "f32[384]", primals_525: "i64[]", primals_526: "f32[448]", primals_527: "f32[448]", primals_528: "i64[]", primals_529: "f32[384]", primals_530: "f32[384]", primals_531: "i64[]", primals_532: "f32[384]", primals_533: "f32[384]", primals_534: "i64[]", primals_535: "f32[384]", primals_536: "f32[384]", primals_537: "i64[]", primals_538: "f32[192]", primals_539: "f32[192]", primals_540: "i64[]", primals_541: "f32[320]", primals_542: "f32[320]", primals_543: "i64[]", primals_544: "f32[384]", primals_545: "f32[384]", primals_546: "i64[]", primals_547: "f32[384]", primals_548: "f32[384]", primals_549: "i64[]", primals_550: "f32[384]", primals_551: "f32[384]", primals_552: "i64[]", primals_553: "f32[448]", primals_554: "f32[448]", primals_555: "i64[]", primals_556: "f32[384]", primals_557: "f32[384]", primals_558: "i64[]", primals_559: "f32[384]", primals_560: "f32[384]", primals_561: "i64[]", primals_562: "f32[384]", primals_563: "f32[384]", primals_564: "i64[]", primals_565: "f32[192]", primals_566: "f32[192]", primals_567: "f32[8, 3, 299, 299]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution: "f32[8, 32, 149, 149]" = torch.ops.aten.convolution.default(primals_567, primals_189, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 32, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
    rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 32, 149, 149]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 32, 149, 149]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
    add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000056304087113);  squeeze_2 = None
    mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
    add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 32, 149, 149]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 32, 149, 149]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu: "f32[8, 32, 149, 149]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_1: "f32[8, 32, 147, 147]" = torch.ops.aten.convolution.default(relu, primals_190, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_288, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 32, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 32, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
    rsqrt_1: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 32, 147, 147]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 32, 147, 147]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[32]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
    add_7: "f32[32]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.000005784660238);  squeeze_5 = None
    mul_11: "f32[32]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[32]" = torch.ops.aten.mul.Tensor(primals_290, 0.9)
    add_8: "f32[32]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 32, 147, 147]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 32, 147, 147]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_1: "f32[8, 32, 147, 147]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_2: "f32[8, 64, 147, 147]" = torch.ops.aten.convolution.default(relu_1, primals_191, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 64, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 64, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 0.001)
    rsqrt_2: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 64, 147, 147]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 64, 147, 147]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[64]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
    add_12: "f32[64]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.000005784660238);  squeeze_8 = None
    mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[64]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
    add_13: "f32[64]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 64, 147, 147]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 64, 147, 147]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_2: "f32[8, 64, 147, 147]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:345, code: x = self.Pool1(x)  # N x 64 x 73 x 73
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu_2, [3, 3], [2, 2])
    getitem_6: "f32[8, 64, 73, 73]" = max_pool2d_with_indices[0]
    getitem_7: "i64[8, 64, 73, 73]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_3: "f32[8, 80, 73, 73]" = torch.ops.aten.convolution.default(getitem_6, primals_192, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_294, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 80, 1, 1]" = var_mean_3[0]
    getitem_9: "f32[1, 80, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 80, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 0.001)
    rsqrt_3: "f32[1, 80, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 80, 73, 73]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
    mul_21: "f32[8, 80, 73, 73]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_10: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[80]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
    add_17: "f32[80]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_24: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000234571086768);  squeeze_11 = None
    mul_25: "f32[80]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[80]" = torch.ops.aten.mul.Tensor(primals_296, 0.9)
    add_18: "f32[80]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 80, 73, 73]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[80, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 80, 73, 73]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_3: "f32[8, 80, 73, 73]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_4: "f32[8, 192, 71, 71]" = torch.ops.aten.convolution.default(relu_3, primals_193, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 192, 1, 1]" = var_mean_4[0]
    getitem_11: "f32[1, 192, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 0.001)
    rsqrt_4: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 192, 71, 71]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
    mul_28: "f32[8, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_13: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[192]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
    add_22: "f32[192]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_31: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000247972822178);  squeeze_14 = None
    mul_32: "f32[192]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[192]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
    add_23: "f32[192]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 192, 71, 71]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 192, 71, 71]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_4: "f32[8, 192, 71, 71]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:348, code: x = self.Pool2(x)  # N x 192 x 35 x 35
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(relu_4, [3, 3], [2, 2])
    getitem_12: "f32[8, 192, 35, 35]" = max_pool2d_with_indices_1[0]
    getitem_13: "i64[8, 192, 35, 35]" = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_5: "f32[8, 64, 35, 35]" = torch.ops.aten.convolution.default(getitem_12, primals_194, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_300, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 64, 1, 1]" = var_mean_5[0]
    getitem_15: "f32[1, 64, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 0.001)
    rsqrt_5: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_15)
    mul_35: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_16: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[64]" = torch.ops.aten.mul.Tensor(primals_301, 0.9)
    add_27: "f32[64]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_38: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0001020512297174);  squeeze_17 = None
    mul_39: "f32[64]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[64]" = torch.ops.aten.mul.Tensor(primals_302, 0.9)
    add_28: "f32[64]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_5: "f32[8, 64, 35, 35]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_6: "f32[8, 48, 35, 35]" = torch.ops.aten.convolution.default(getitem_12, primals_195, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 48, 1, 1]" = var_mean_6[0]
    getitem_17: "f32[1, 48, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 0.001)
    rsqrt_6: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_17)
    mul_42: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_19: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[48]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
    add_32: "f32[48]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_45: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0001020512297174);  squeeze_20 = None
    mul_46: "f32[48]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[48]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
    add_33: "f32[48]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 48, 35, 35]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_6: "f32[8, 48, 35, 35]" = torch.ops.aten.relu.default(add_34);  add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_7: "f32[8, 64, 35, 35]" = torch.ops.aten.convolution.default(relu_6, primals_196, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_306, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 64, 1, 1]" = var_mean_7[0]
    getitem_19: "f32[1, 64, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_36: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 0.001)
    rsqrt_7: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_7: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_19)
    mul_49: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_22: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[64]" = torch.ops.aten.mul.Tensor(primals_307, 0.9)
    add_37: "f32[64]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_52: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0001020512297174);  squeeze_23 = None
    mul_53: "f32[64]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[64]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
    add_38: "f32[64]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_39: "f32[8, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_7: "f32[8, 64, 35, 35]" = torch.ops.aten.relu.default(add_39);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_8: "f32[8, 64, 35, 35]" = torch.ops.aten.convolution.default(getitem_12, primals_197, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_40: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 64, 1, 1]" = var_mean_8[0]
    getitem_21: "f32[1, 64, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_41: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 0.001)
    rsqrt_8: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    sub_8: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_21)
    mul_56: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_25: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[64]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
    add_42: "f32[64]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_59: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0001020512297174);  squeeze_26 = None
    mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[64]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
    add_43: "f32[64]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_44: "f32[8, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_8: "f32[8, 64, 35, 35]" = torch.ops.aten.relu.default(add_44);  add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_9: "f32[8, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_8, primals_198, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_45: "i64[]" = torch.ops.aten.add.Tensor(primals_312, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 96, 1, 1]" = var_mean_9[0]
    getitem_23: "f32[1, 96, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_46: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 0.001)
    rsqrt_9: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_9: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_23)
    mul_63: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_28: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[96]" = torch.ops.aten.mul.Tensor(primals_313, 0.9)
    add_47: "f32[96]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_66: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0001020512297174);  squeeze_29 = None
    mul_67: "f32[96]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[96]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
    add_48: "f32[96]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_49: "f32[8, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_9: "f32[8, 96, 35, 35]" = torch.ops.aten.relu.default(add_49);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_10: "f32[8, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_9, primals_199, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_50: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 96, 1, 1]" = var_mean_10[0]
    getitem_25: "f32[1, 96, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_51: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 0.001)
    rsqrt_10: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_10: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_25)
    mul_70: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_31: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_71: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_72: "f32[96]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
    add_52: "f32[96]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_32: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_73: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0001020512297174);  squeeze_32 = None
    mul_74: "f32[96]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[96]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
    add_53: "f32[96]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_40: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_76: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
    unsqueeze_42: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_54: "f32[8, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_10: "f32[8, 96, 35, 35]" = torch.ops.aten.relu.default(add_54);  add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:48, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d: "f32[8, 192, 35, 35]" = torch.ops.aten.avg_pool2d.default(getitem_12, [3, 3], [1, 1], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_11: "f32[8, 32, 35, 35]" = torch.ops.aten.convolution.default(avg_pool2d, primals_200, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_55: "i64[]" = torch.ops.aten.add.Tensor(primals_318, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 32, 1, 1]" = var_mean_11[0]
    getitem_27: "f32[1, 32, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_56: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 0.001)
    rsqrt_11: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_11: "f32[8, 32, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_27)
    mul_77: "f32[8, 32, 35, 35]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_34: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_78: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_79: "f32[32]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_57: "f32[32]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_35: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_80: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001020512297174);  squeeze_35 = None
    mul_81: "f32[32]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[32]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
    add_58: "f32[32]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_44: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_83: "f32[8, 32, 35, 35]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
    unsqueeze_46: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_59: "f32[8, 32, 35, 35]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_11: "f32[8, 32, 35, 35]" = torch.ops.aten.relu.default(add_59);  add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:56, code: return torch.cat(outputs, 1)
    cat: "f32[8, 256, 35, 35]" = torch.ops.aten.cat.default([relu_5, relu_7, relu_10, relu_11], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_12: "f32[8, 64, 35, 35]" = torch.ops.aten.convolution.default(cat, primals_201, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_60: "i64[]" = torch.ops.aten.add.Tensor(primals_321, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 64, 1, 1]" = var_mean_12[0]
    getitem_29: "f32[1, 64, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_61: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 0.001)
    rsqrt_12: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_12: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_29)
    mul_84: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_37: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_85: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_86: "f32[64]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_62: "f32[64]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_38: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_87: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001020512297174);  squeeze_38 = None
    mul_88: "f32[64]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[64]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
    add_63: "f32[64]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_90: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
    unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_64: "f32[8, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_12: "f32[8, 64, 35, 35]" = torch.ops.aten.relu.default(add_64);  add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_13: "f32[8, 48, 35, 35]" = torch.ops.aten.convolution.default(cat, primals_202, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_65: "i64[]" = torch.ops.aten.add.Tensor(primals_324, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 48, 1, 1]" = var_mean_13[0]
    getitem_31: "f32[1, 48, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_66: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 0.001)
    rsqrt_13: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_13: "f32[8, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_31)
    mul_91: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_40: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_92: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_93: "f32[48]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_67: "f32[48]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    squeeze_41: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_94: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001020512297174);  squeeze_41 = None
    mul_95: "f32[48]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
    mul_96: "f32[48]" = torch.ops.aten.mul.Tensor(primals_326, 0.9)
    add_68: "f32[48]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    unsqueeze_52: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_97: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
    unsqueeze_54: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_69: "f32[8, 48, 35, 35]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_13: "f32[8, 48, 35, 35]" = torch.ops.aten.relu.default(add_69);  add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_14: "f32[8, 64, 35, 35]" = torch.ops.aten.convolution.default(relu_13, primals_203, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_70: "i64[]" = torch.ops.aten.add.Tensor(primals_327, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 64, 1, 1]" = var_mean_14[0]
    getitem_33: "f32[1, 64, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_71: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 0.001)
    rsqrt_14: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_14: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_33)
    mul_98: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_43: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_99: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_100: "f32[64]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_72: "f32[64]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_44: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_101: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001020512297174);  squeeze_44 = None
    mul_102: "f32[64]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[64]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
    add_73: "f32[64]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_56: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_104: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
    unsqueeze_58: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_74: "f32[8, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_14: "f32[8, 64, 35, 35]" = torch.ops.aten.relu.default(add_74);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_15: "f32[8, 64, 35, 35]" = torch.ops.aten.convolution.default(cat, primals_204, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_75: "i64[]" = torch.ops.aten.add.Tensor(primals_330, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 64, 1, 1]" = var_mean_15[0]
    getitem_35: "f32[1, 64, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_76: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 0.001)
    rsqrt_15: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_15: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_35)
    mul_105: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_46: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_106: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_107: "f32[64]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_77: "f32[64]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_47: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_108: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001020512297174);  squeeze_47 = None
    mul_109: "f32[64]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[64]" = torch.ops.aten.mul.Tensor(primals_332, 0.9)
    add_78: "f32[64]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_60: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_111: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
    unsqueeze_62: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_79: "f32[8, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_15: "f32[8, 64, 35, 35]" = torch.ops.aten.relu.default(add_79);  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_16: "f32[8, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_15, primals_205, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_80: "i64[]" = torch.ops.aten.add.Tensor(primals_333, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 96, 1, 1]" = var_mean_16[0]
    getitem_37: "f32[1, 96, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_81: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 0.001)
    rsqrt_16: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_16: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_37)
    mul_112: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_49: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_113: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_114: "f32[96]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_82: "f32[96]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_50: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_115: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001020512297174);  squeeze_50 = None
    mul_116: "f32[96]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[96]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
    add_83: "f32[96]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_64: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_118: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
    unsqueeze_66: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_84: "f32[8, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_16: "f32[8, 96, 35, 35]" = torch.ops.aten.relu.default(add_84);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_17: "f32[8, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_16, primals_206, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_85: "i64[]" = torch.ops.aten.add.Tensor(primals_336, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 96, 1, 1]" = var_mean_17[0]
    getitem_39: "f32[1, 96, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_86: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 0.001)
    rsqrt_17: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_17: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_39)
    mul_119: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_52: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_120: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_121: "f32[96]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_87: "f32[96]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_53: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_122: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001020512297174);  squeeze_53 = None
    mul_123: "f32[96]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[96]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_88: "f32[96]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_68: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_125: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
    unsqueeze_70: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_89: "f32[8, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_17: "f32[8, 96, 35, 35]" = torch.ops.aten.relu.default(add_89);  add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:48, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_1: "f32[8, 256, 35, 35]" = torch.ops.aten.avg_pool2d.default(cat, [3, 3], [1, 1], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_18: "f32[8, 64, 35, 35]" = torch.ops.aten.convolution.default(avg_pool2d_1, primals_207, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_90: "i64[]" = torch.ops.aten.add.Tensor(primals_339, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 64, 1, 1]" = var_mean_18[0]
    getitem_41: "f32[1, 64, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_91: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 0.001)
    rsqrt_18: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_18: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_41)
    mul_126: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_55: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_127: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_128: "f32[64]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_92: "f32[64]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_56: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_129: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001020512297174);  squeeze_56 = None
    mul_130: "f32[64]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[64]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_93: "f32[64]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_72: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_132: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
    unsqueeze_74: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_94: "f32[8, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_18: "f32[8, 64, 35, 35]" = torch.ops.aten.relu.default(add_94);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:56, code: return torch.cat(outputs, 1)
    cat_1: "f32[8, 288, 35, 35]" = torch.ops.aten.cat.default([relu_12, relu_14, relu_17, relu_18], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_19: "f32[8, 64, 35, 35]" = torch.ops.aten.convolution.default(cat_1, primals_208, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_95: "i64[]" = torch.ops.aten.add.Tensor(primals_342, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 64, 1, 1]" = var_mean_19[0]
    getitem_43: "f32[1, 64, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_96: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 0.001)
    rsqrt_19: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_19: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_43)
    mul_133: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_58: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_134: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_135: "f32[64]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_97: "f32[64]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_59: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_136: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001020512297174);  squeeze_59 = None
    mul_137: "f32[64]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[64]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_98: "f32[64]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_76: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_139: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
    unsqueeze_78: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_99: "f32[8, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_19: "f32[8, 64, 35, 35]" = torch.ops.aten.relu.default(add_99);  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_20: "f32[8, 48, 35, 35]" = torch.ops.aten.convolution.default(cat_1, primals_209, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_100: "i64[]" = torch.ops.aten.add.Tensor(primals_345, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 48, 1, 1]" = var_mean_20[0]
    getitem_45: "f32[1, 48, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_101: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 0.001)
    rsqrt_20: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_20: "f32[8, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_45)
    mul_140: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_61: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_141: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_142: "f32[48]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
    add_102: "f32[48]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_62: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_143: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001020512297174);  squeeze_62 = None
    mul_144: "f32[48]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[48]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
    add_103: "f32[48]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    unsqueeze_80: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_146: "f32[8, 48, 35, 35]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
    unsqueeze_82: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_104: "f32[8, 48, 35, 35]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_20: "f32[8, 48, 35, 35]" = torch.ops.aten.relu.default(add_104);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_21: "f32[8, 64, 35, 35]" = torch.ops.aten.convolution.default(relu_20, primals_210, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_105: "i64[]" = torch.ops.aten.add.Tensor(primals_348, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 64, 1, 1]" = var_mean_21[0]
    getitem_47: "f32[1, 64, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_106: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 0.001)
    rsqrt_21: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_21: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_47)
    mul_147: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_64: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_148: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_149: "f32[64]" = torch.ops.aten.mul.Tensor(primals_349, 0.9)
    add_107: "f32[64]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_65: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_150: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001020512297174);  squeeze_65 = None
    mul_151: "f32[64]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[64]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
    add_108: "f32[64]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_84: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_153: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
    unsqueeze_86: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_109: "f32[8, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_21: "f32[8, 64, 35, 35]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_22: "f32[8, 64, 35, 35]" = torch.ops.aten.convolution.default(cat_1, primals_211, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_110: "i64[]" = torch.ops.aten.add.Tensor(primals_351, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 64, 1, 1]" = var_mean_22[0]
    getitem_49: "f32[1, 64, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_111: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 0.001)
    rsqrt_22: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_22: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_49)
    mul_154: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_67: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_155: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_156: "f32[64]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
    add_112: "f32[64]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    squeeze_68: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_157: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001020512297174);  squeeze_68 = None
    mul_158: "f32[64]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
    mul_159: "f32[64]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
    add_113: "f32[64]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    unsqueeze_88: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_160: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
    unsqueeze_90: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_114: "f32[8, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_22: "f32[8, 64, 35, 35]" = torch.ops.aten.relu.default(add_114);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_23: "f32[8, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_22, primals_212, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_354, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 96, 1, 1]" = var_mean_23[0]
    getitem_51: "f32[1, 96, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_116: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 0.001)
    rsqrt_23: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_23: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_51)
    mul_161: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_70: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_162: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_163: "f32[96]" = torch.ops.aten.mul.Tensor(primals_355, 0.9)
    add_117: "f32[96]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    squeeze_71: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_164: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001020512297174);  squeeze_71 = None
    mul_165: "f32[96]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
    mul_166: "f32[96]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
    add_118: "f32[96]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    unsqueeze_92: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_167: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
    unsqueeze_94: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_119: "f32[8, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_23: "f32[8, 96, 35, 35]" = torch.ops.aten.relu.default(add_119);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_24: "f32[8, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_23, primals_213, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_120: "i64[]" = torch.ops.aten.add.Tensor(primals_357, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 96, 1, 1]" = var_mean_24[0]
    getitem_53: "f32[1, 96, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_121: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 0.001)
    rsqrt_24: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_24: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_53)
    mul_168: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_73: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_169: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_170: "f32[96]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
    add_122: "f32[96]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_74: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_171: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001020512297174);  squeeze_74 = None
    mul_172: "f32[96]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[96]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
    add_123: "f32[96]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_96: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_174: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
    unsqueeze_98: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_124: "f32[8, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_24: "f32[8, 96, 35, 35]" = torch.ops.aten.relu.default(add_124);  add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:48, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_2: "f32[8, 288, 35, 35]" = torch.ops.aten.avg_pool2d.default(cat_1, [3, 3], [1, 1], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_25: "f32[8, 64, 35, 35]" = torch.ops.aten.convolution.default(avg_pool2d_2, primals_214, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_125: "i64[]" = torch.ops.aten.add.Tensor(primals_360, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 64, 1, 1]" = var_mean_25[0]
    getitem_55: "f32[1, 64, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_126: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 0.001)
    rsqrt_25: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
    sub_25: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_55)
    mul_175: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_76: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_176: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_177: "f32[64]" = torch.ops.aten.mul.Tensor(primals_361, 0.9)
    add_127: "f32[64]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_77: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_178: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001020512297174);  squeeze_77 = None
    mul_179: "f32[64]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[64]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
    add_128: "f32[64]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_100: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_181: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
    unsqueeze_102: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_129: "f32[8, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_25: "f32[8, 64, 35, 35]" = torch.ops.aten.relu.default(add_129);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:56, code: return torch.cat(outputs, 1)
    cat_2: "f32[8, 288, 35, 35]" = torch.ops.aten.cat.default([relu_19, relu_21, relu_24, relu_25], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_26: "f32[8, 384, 17, 17]" = torch.ops.aten.convolution.default(cat_2, primals_215, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_130: "i64[]" = torch.ops.aten.add.Tensor(primals_363, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 384, 1, 1]" = var_mean_26[0]
    getitem_57: "f32[1, 384, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_131: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 0.001)
    rsqrt_26: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    sub_26: "f32[8, 384, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_57)
    mul_182: "f32[8, 384, 17, 17]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_79: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_183: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_184: "f32[384]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
    add_132: "f32[384]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_80: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_185: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0004327131112072);  squeeze_80 = None
    mul_186: "f32[384]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[384]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
    add_133: "f32[384]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_104: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_188: "f32[8, 384, 17, 17]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
    unsqueeze_106: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_134: "f32[8, 384, 17, 17]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_26: "f32[8, 384, 17, 17]" = torch.ops.aten.relu.default(add_134);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_27: "f32[8, 64, 35, 35]" = torch.ops.aten.convolution.default(cat_2, primals_216, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_135: "i64[]" = torch.ops.aten.add.Tensor(primals_366, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 64, 1, 1]" = var_mean_27[0]
    getitem_59: "f32[1, 64, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_136: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 0.001)
    rsqrt_27: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_27: "f32[8, 64, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_59)
    mul_189: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_82: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_190: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_191: "f32[64]" = torch.ops.aten.mul.Tensor(primals_367, 0.9)
    add_137: "f32[64]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    squeeze_83: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_192: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0001020512297174);  squeeze_83 = None
    mul_193: "f32[64]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
    mul_194: "f32[64]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
    add_138: "f32[64]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    unsqueeze_108: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_109: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_195: "f32[8, 64, 35, 35]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
    unsqueeze_110: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_111: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_139: "f32[8, 64, 35, 35]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_27: "f32[8, 64, 35, 35]" = torch.ops.aten.relu.default(add_139);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_28: "f32[8, 96, 35, 35]" = torch.ops.aten.convolution.default(relu_27, primals_217, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_140: "i64[]" = torch.ops.aten.add.Tensor(primals_369, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 96, 1, 1]" = var_mean_28[0]
    getitem_61: "f32[1, 96, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_141: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 0.001)
    rsqrt_28: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_28: "f32[8, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_61)
    mul_196: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_85: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_197: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_198: "f32[96]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
    add_142: "f32[96]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_86: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_199: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0001020512297174);  squeeze_86 = None
    mul_200: "f32[96]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[96]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
    add_143: "f32[96]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_112: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_113: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_202: "f32[8, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
    unsqueeze_114: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_115: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_144: "f32[8, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_28: "f32[8, 96, 35, 35]" = torch.ops.aten.relu.default(add_144);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_29: "f32[8, 96, 17, 17]" = torch.ops.aten.convolution.default(relu_28, primals_218, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_145: "i64[]" = torch.ops.aten.add.Tensor(primals_372, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 96, 1, 1]" = var_mean_29[0]
    getitem_63: "f32[1, 96, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_146: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 0.001)
    rsqrt_29: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_29: "f32[8, 96, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_63)
    mul_203: "f32[8, 96, 17, 17]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_88: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_204: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_205: "f32[96]" = torch.ops.aten.mul.Tensor(primals_373, 0.9)
    add_147: "f32[96]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_89: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_206: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0004327131112072);  squeeze_89 = None
    mul_207: "f32[96]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[96]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
    add_148: "f32[96]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_116: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_117: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_209: "f32[8, 96, 17, 17]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
    unsqueeze_118: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_119: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_149: "f32[8, 96, 17, 17]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_29: "f32[8, 96, 17, 17]" = torch.ops.aten.relu.default(add_149);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:77, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
    max_pool2d_with_indices_2 = torch.ops.aten.max_pool2d_with_indices.default(cat_2, [3, 3], [2, 2])
    getitem_64: "f32[8, 288, 17, 17]" = max_pool2d_with_indices_2[0]
    getitem_65: "i64[8, 288, 17, 17]" = max_pool2d_with_indices_2[1];  max_pool2d_with_indices_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:84, code: return torch.cat(outputs, 1)
    cat_3: "f32[8, 768, 17, 17]" = torch.ops.aten.cat.default([relu_26, relu_29, getitem_64], 1);  getitem_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_30: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_3, primals_219, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_150: "i64[]" = torch.ops.aten.add.Tensor(primals_375, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 192, 1, 1]" = var_mean_30[0]
    getitem_67: "f32[1, 192, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_151: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 0.001)
    rsqrt_30: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_30: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_67)
    mul_210: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_91: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_211: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_212: "f32[192]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
    add_152: "f32[192]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    squeeze_92: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_213: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0004327131112072);  squeeze_92 = None
    mul_214: "f32[192]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
    mul_215: "f32[192]" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
    add_153: "f32[192]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    unsqueeze_120: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_121: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_216: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
    unsqueeze_122: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_123: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_154: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_30: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_154);  add_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_31: "f32[8, 128, 17, 17]" = torch.ops.aten.convolution.default(cat_3, primals_220, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_155: "i64[]" = torch.ops.aten.add.Tensor(primals_378, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1, 1]" = var_mean_31[0]
    getitem_69: "f32[1, 128, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_156: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 0.001)
    rsqrt_31: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    sub_31: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_69)
    mul_217: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_94: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_218: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_219: "f32[128]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
    add_157: "f32[128]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_95: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_220: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0004327131112072);  squeeze_95 = None
    mul_221: "f32[128]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[128]" = torch.ops.aten.mul.Tensor(primals_380, 0.9)
    add_158: "f32[128]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    unsqueeze_124: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_125: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_223: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
    unsqueeze_126: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_127: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_159: "f32[8, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_31: "f32[8, 128, 17, 17]" = torch.ops.aten.relu.default(add_159);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_32: "f32[8, 128, 17, 17]" = torch.ops.aten.convolution.default(relu_31, primals_221, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_160: "i64[]" = torch.ops.aten.add.Tensor(primals_381, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 128, 1, 1]" = var_mean_32[0]
    getitem_71: "f32[1, 128, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_161: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 0.001)
    rsqrt_32: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_32: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_71)
    mul_224: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_97: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_225: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_226: "f32[128]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
    add_162: "f32[128]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_98: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_227: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0004327131112072);  squeeze_98 = None
    mul_228: "f32[128]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[128]" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
    add_163: "f32[128]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_128: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_129: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_230: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
    unsqueeze_130: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_131: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_164: "f32[8, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_32: "f32[8, 128, 17, 17]" = torch.ops.aten.relu.default(add_164);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_33: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_32, primals_222, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_165: "i64[]" = torch.ops.aten.add.Tensor(primals_384, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 192, 1, 1]" = var_mean_33[0]
    getitem_73: "f32[1, 192, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_166: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 0.001)
    rsqrt_33: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    sub_33: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_73)
    mul_231: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_100: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_232: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_233: "f32[192]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
    add_167: "f32[192]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_101: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_234: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0004327131112072);  squeeze_101 = None
    mul_235: "f32[192]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[192]" = torch.ops.aten.mul.Tensor(primals_386, 0.9)
    add_168: "f32[192]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_132: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_133: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_237: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
    unsqueeze_134: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_135: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_169: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_33: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_169);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_34: "f32[8, 128, 17, 17]" = torch.ops.aten.convolution.default(cat_3, primals_223, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_170: "i64[]" = torch.ops.aten.add.Tensor(primals_387, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 128, 1, 1]" = var_mean_34[0]
    getitem_75: "f32[1, 128, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_171: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 0.001)
    rsqrt_34: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_34: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_75)
    mul_238: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_103: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_239: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_240: "f32[128]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
    add_172: "f32[128]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_104: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_241: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0004327131112072);  squeeze_104 = None
    mul_242: "f32[128]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[128]" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
    add_173: "f32[128]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_136: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_137: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_244: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
    unsqueeze_138: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_139: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_174: "f32[8, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_34: "f32[8, 128, 17, 17]" = torch.ops.aten.relu.default(add_174);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_35: "f32[8, 128, 17, 17]" = torch.ops.aten.convolution.default(relu_34, primals_224, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_175: "i64[]" = torch.ops.aten.add.Tensor(primals_390, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1, 1]" = var_mean_35[0]
    getitem_77: "f32[1, 128, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_176: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 0.001)
    rsqrt_35: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_35: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_77)
    mul_245: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_106: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_246: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_247: "f32[128]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
    add_177: "f32[128]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_107: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_248: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0004327131112072);  squeeze_107 = None
    mul_249: "f32[128]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[128]" = torch.ops.aten.mul.Tensor(primals_392, 0.9)
    add_178: "f32[128]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_140: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_141: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_251: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
    unsqueeze_142: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_143: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_179: "f32[8, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_35: "f32[8, 128, 17, 17]" = torch.ops.aten.relu.default(add_179);  add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_36: "f32[8, 128, 17, 17]" = torch.ops.aten.convolution.default(relu_35, primals_225, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_393, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 128, 1, 1]" = var_mean_36[0]
    getitem_79: "f32[1, 128, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_181: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 0.001)
    rsqrt_36: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    sub_36: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_79)
    mul_252: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_109: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_253: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_254: "f32[128]" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
    add_182: "f32[128]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    squeeze_110: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_255: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0004327131112072);  squeeze_110 = None
    mul_256: "f32[128]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
    mul_257: "f32[128]" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
    add_183: "f32[128]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    unsqueeze_144: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_145: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_258: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
    unsqueeze_146: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_147: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_184: "f32[8, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_36: "f32[8, 128, 17, 17]" = torch.ops.aten.relu.default(add_184);  add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_37: "f32[8, 128, 17, 17]" = torch.ops.aten.convolution.default(relu_36, primals_226, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_396, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1, 1]" = var_mean_37[0]
    getitem_81: "f32[1, 128, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_186: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 0.001)
    rsqrt_37: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    sub_37: "f32[8, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_81)
    mul_259: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_112: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_260: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_261: "f32[128]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
    add_187: "f32[128]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    squeeze_113: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_262: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0004327131112072);  squeeze_113 = None
    mul_263: "f32[128]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
    mul_264: "f32[128]" = torch.ops.aten.mul.Tensor(primals_398, 0.9)
    add_188: "f32[128]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    unsqueeze_148: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_149: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_265: "f32[8, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
    unsqueeze_150: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_151: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_189: "f32[8, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_37: "f32[8, 128, 17, 17]" = torch.ops.aten.relu.default(add_189);  add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_38: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_37, primals_227, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_190: "i64[]" = torch.ops.aten.add.Tensor(primals_399, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 192, 1, 1]" = var_mean_38[0]
    getitem_83: "f32[1, 192, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_191: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 0.001)
    rsqrt_38: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_38: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_83)
    mul_266: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_115: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_267: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_268: "f32[192]" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
    add_192: "f32[192]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    squeeze_116: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_269: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0004327131112072);  squeeze_116 = None
    mul_270: "f32[192]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
    mul_271: "f32[192]" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
    add_193: "f32[192]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    unsqueeze_152: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_153: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_272: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
    unsqueeze_154: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_155: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_194: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_38: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_194);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:120, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_3: "f32[8, 768, 17, 17]" = torch.ops.aten.avg_pool2d.default(cat_3, [3, 3], [1, 1], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_39: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(avg_pool2d_3, primals_228, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_195: "i64[]" = torch.ops.aten.add.Tensor(primals_402, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 192, 1, 1]" = var_mean_39[0]
    getitem_85: "f32[1, 192, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_196: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 0.001)
    rsqrt_39: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
    sub_39: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_85)
    mul_273: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_118: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_274: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_275: "f32[192]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
    add_197: "f32[192]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    squeeze_119: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_276: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0004327131112072);  squeeze_119 = None
    mul_277: "f32[192]" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
    mul_278: "f32[192]" = torch.ops.aten.mul.Tensor(primals_404, 0.9)
    add_198: "f32[192]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    unsqueeze_156: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_157: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_279: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
    unsqueeze_158: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_159: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_199: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_39: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_199);  add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:128, code: return torch.cat(outputs, 1)
    cat_4: "f32[8, 768, 17, 17]" = torch.ops.aten.cat.default([relu_30, relu_33, relu_38, relu_39], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_40: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_4, primals_229, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_200: "i64[]" = torch.ops.aten.add.Tensor(primals_405, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 192, 1, 1]" = var_mean_40[0]
    getitem_87: "f32[1, 192, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_201: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 0.001)
    rsqrt_40: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
    sub_40: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_87)
    mul_280: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_121: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_281: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_282: "f32[192]" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
    add_202: "f32[192]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    squeeze_122: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_283: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0004327131112072);  squeeze_122 = None
    mul_284: "f32[192]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
    mul_285: "f32[192]" = torch.ops.aten.mul.Tensor(primals_407, 0.9)
    add_203: "f32[192]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_160: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_161: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_286: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
    unsqueeze_162: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_163: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_204: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_40: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_204);  add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_41: "f32[8, 160, 17, 17]" = torch.ops.aten.convolution.default(cat_4, primals_230, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_205: "i64[]" = torch.ops.aten.add.Tensor(primals_408, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 160, 1, 1]" = var_mean_41[0]
    getitem_89: "f32[1, 160, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_206: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 0.001)
    rsqrt_41: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
    sub_41: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_89)
    mul_287: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_124: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_288: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_289: "f32[160]" = torch.ops.aten.mul.Tensor(primals_409, 0.9)
    add_207: "f32[160]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    squeeze_125: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_290: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0004327131112072);  squeeze_125 = None
    mul_291: "f32[160]" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
    mul_292: "f32[160]" = torch.ops.aten.mul.Tensor(primals_410, 0.9)
    add_208: "f32[160]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    unsqueeze_164: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_165: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_293: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
    unsqueeze_166: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_167: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_209: "f32[8, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_41: "f32[8, 160, 17, 17]" = torch.ops.aten.relu.default(add_209);  add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_42: "f32[8, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_41, primals_231, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_210: "i64[]" = torch.ops.aten.add.Tensor(primals_411, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 160, 1, 1]" = var_mean_42[0]
    getitem_91: "f32[1, 160, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_211: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 0.001)
    rsqrt_42: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
    sub_42: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_91)
    mul_294: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_127: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_295: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_296: "f32[160]" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
    add_212: "f32[160]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_128: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_297: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0004327131112072);  squeeze_128 = None
    mul_298: "f32[160]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[160]" = torch.ops.aten.mul.Tensor(primals_413, 0.9)
    add_213: "f32[160]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_168: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_169: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_300: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
    unsqueeze_170: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_171: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_214: "f32[8, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_42: "f32[8, 160, 17, 17]" = torch.ops.aten.relu.default(add_214);  add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_43: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_42, primals_232, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_215: "i64[]" = torch.ops.aten.add.Tensor(primals_414, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[1, 192, 1, 1]" = var_mean_43[0]
    getitem_93: "f32[1, 192, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_216: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 0.001)
    rsqrt_43: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
    sub_43: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_93)
    mul_301: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
    squeeze_130: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_302: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_303: "f32[192]" = torch.ops.aten.mul.Tensor(primals_415, 0.9)
    add_217: "f32[192]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    squeeze_131: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
    mul_304: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0004327131112072);  squeeze_131 = None
    mul_305: "f32[192]" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
    mul_306: "f32[192]" = torch.ops.aten.mul.Tensor(primals_416, 0.9)
    add_218: "f32[192]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_172: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_173: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_307: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
    unsqueeze_174: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_175: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_219: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_43: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_219);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_44: "f32[8, 160, 17, 17]" = torch.ops.aten.convolution.default(cat_4, primals_233, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_220: "i64[]" = torch.ops.aten.add.Tensor(primals_417, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 160, 1, 1]" = var_mean_44[0]
    getitem_95: "f32[1, 160, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_221: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 0.001)
    rsqrt_44: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
    sub_44: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_95)
    mul_308: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_133: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_309: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_310: "f32[160]" = torch.ops.aten.mul.Tensor(primals_418, 0.9)
    add_222: "f32[160]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    squeeze_134: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_311: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0004327131112072);  squeeze_134 = None
    mul_312: "f32[160]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
    mul_313: "f32[160]" = torch.ops.aten.mul.Tensor(primals_419, 0.9)
    add_223: "f32[160]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_176: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_177: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_314: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
    unsqueeze_178: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_179: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_224: "f32[8, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_44: "f32[8, 160, 17, 17]" = torch.ops.aten.relu.default(add_224);  add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_45: "f32[8, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_44, primals_234, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_225: "i64[]" = torch.ops.aten.add.Tensor(primals_420, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 160, 1, 1]" = var_mean_45[0]
    getitem_97: "f32[1, 160, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_226: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 0.001)
    rsqrt_45: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
    sub_45: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_97)
    mul_315: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_136: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_316: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_317: "f32[160]" = torch.ops.aten.mul.Tensor(primals_421, 0.9)
    add_227: "f32[160]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    squeeze_137: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_318: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0004327131112072);  squeeze_137 = None
    mul_319: "f32[160]" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
    mul_320: "f32[160]" = torch.ops.aten.mul.Tensor(primals_422, 0.9)
    add_228: "f32[160]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    unsqueeze_180: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_181: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_321: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
    unsqueeze_182: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_183: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_229: "f32[8, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_45: "f32[8, 160, 17, 17]" = torch.ops.aten.relu.default(add_229);  add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_46: "f32[8, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_45, primals_235, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_230: "i64[]" = torch.ops.aten.add.Tensor(primals_423, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 160, 1, 1]" = var_mean_46[0]
    getitem_99: "f32[1, 160, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_231: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 0.001)
    rsqrt_46: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_46: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_99)
    mul_322: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_139: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_323: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_324: "f32[160]" = torch.ops.aten.mul.Tensor(primals_424, 0.9)
    add_232: "f32[160]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    squeeze_140: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_325: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0004327131112072);  squeeze_140 = None
    mul_326: "f32[160]" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
    mul_327: "f32[160]" = torch.ops.aten.mul.Tensor(primals_425, 0.9)
    add_233: "f32[160]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    unsqueeze_184: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1)
    unsqueeze_185: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_328: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
    unsqueeze_186: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1);  primals_94 = None
    unsqueeze_187: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_234: "f32[8, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_46: "f32[8, 160, 17, 17]" = torch.ops.aten.relu.default(add_234);  add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_47: "f32[8, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_46, primals_236, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_235: "i64[]" = torch.ops.aten.add.Tensor(primals_426, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 160, 1, 1]" = var_mean_47[0]
    getitem_101: "f32[1, 160, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_236: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 0.001)
    rsqrt_47: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
    sub_47: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_101)
    mul_329: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_142: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_330: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_331: "f32[160]" = torch.ops.aten.mul.Tensor(primals_427, 0.9)
    add_237: "f32[160]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    squeeze_143: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_332: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0004327131112072);  squeeze_143 = None
    mul_333: "f32[160]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
    mul_334: "f32[160]" = torch.ops.aten.mul.Tensor(primals_428, 0.9)
    add_238: "f32[160]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    unsqueeze_188: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_189: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_335: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
    unsqueeze_190: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_191: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_239: "f32[8, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_47: "f32[8, 160, 17, 17]" = torch.ops.aten.relu.default(add_239);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_48: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_47, primals_237, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_240: "i64[]" = torch.ops.aten.add.Tensor(primals_429, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 192, 1, 1]" = var_mean_48[0]
    getitem_103: "f32[1, 192, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_241: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 0.001)
    rsqrt_48: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
    sub_48: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_103)
    mul_336: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_145: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_337: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_338: "f32[192]" = torch.ops.aten.mul.Tensor(primals_430, 0.9)
    add_242: "f32[192]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    squeeze_146: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_339: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0004327131112072);  squeeze_146 = None
    mul_340: "f32[192]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
    mul_341: "f32[192]" = torch.ops.aten.mul.Tensor(primals_431, 0.9)
    add_243: "f32[192]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    unsqueeze_192: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_193: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_342: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
    unsqueeze_194: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_195: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_244: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_48: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_244);  add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:120, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_4: "f32[8, 768, 17, 17]" = torch.ops.aten.avg_pool2d.default(cat_4, [3, 3], [1, 1], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_49: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(avg_pool2d_4, primals_238, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_245: "i64[]" = torch.ops.aten.add.Tensor(primals_432, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 192, 1, 1]" = var_mean_49[0]
    getitem_105: "f32[1, 192, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_246: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 0.001)
    rsqrt_49: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
    sub_49: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_105)
    mul_343: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_148: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_344: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_345: "f32[192]" = torch.ops.aten.mul.Tensor(primals_433, 0.9)
    add_247: "f32[192]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    squeeze_149: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_346: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0004327131112072);  squeeze_149 = None
    mul_347: "f32[192]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
    mul_348: "f32[192]" = torch.ops.aten.mul.Tensor(primals_434, 0.9)
    add_248: "f32[192]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    unsqueeze_196: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1)
    unsqueeze_197: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_349: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
    unsqueeze_198: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1);  primals_100 = None
    unsqueeze_199: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_249: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_49: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_249);  add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:128, code: return torch.cat(outputs, 1)
    cat_5: "f32[8, 768, 17, 17]" = torch.ops.aten.cat.default([relu_40, relu_43, relu_48, relu_49], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_50: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_5, primals_239, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_250: "i64[]" = torch.ops.aten.add.Tensor(primals_435, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 192, 1, 1]" = var_mean_50[0]
    getitem_107: "f32[1, 192, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_251: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 0.001)
    rsqrt_50: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
    sub_50: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_107)
    mul_350: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_151: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_351: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_352: "f32[192]" = torch.ops.aten.mul.Tensor(primals_436, 0.9)
    add_252: "f32[192]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    squeeze_152: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_353: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0004327131112072);  squeeze_152 = None
    mul_354: "f32[192]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
    mul_355: "f32[192]" = torch.ops.aten.mul.Tensor(primals_437, 0.9)
    add_253: "f32[192]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_200: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_201: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_356: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
    unsqueeze_202: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_203: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_254: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_50: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_254);  add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_51: "f32[8, 160, 17, 17]" = torch.ops.aten.convolution.default(cat_5, primals_240, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_255: "i64[]" = torch.ops.aten.add.Tensor(primals_438, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 160, 1, 1]" = var_mean_51[0]
    getitem_109: "f32[1, 160, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_256: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 0.001)
    rsqrt_51: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
    sub_51: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_109)
    mul_357: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_154: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_358: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_359: "f32[160]" = torch.ops.aten.mul.Tensor(primals_439, 0.9)
    add_257: "f32[160]" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    squeeze_155: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_360: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0004327131112072);  squeeze_155 = None
    mul_361: "f32[160]" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
    mul_362: "f32[160]" = torch.ops.aten.mul.Tensor(primals_440, 0.9)
    add_258: "f32[160]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    unsqueeze_204: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
    unsqueeze_205: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_363: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
    unsqueeze_206: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
    unsqueeze_207: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_259: "f32[8, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_51: "f32[8, 160, 17, 17]" = torch.ops.aten.relu.default(add_259);  add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_52: "f32[8, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_51, primals_241, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_260: "i64[]" = torch.ops.aten.add.Tensor(primals_441, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 160, 1, 1]" = var_mean_52[0]
    getitem_111: "f32[1, 160, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_261: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 0.001)
    rsqrt_52: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
    sub_52: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_111)
    mul_364: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_157: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_365: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_366: "f32[160]" = torch.ops.aten.mul.Tensor(primals_442, 0.9)
    add_262: "f32[160]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    squeeze_158: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_367: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0004327131112072);  squeeze_158 = None
    mul_368: "f32[160]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
    mul_369: "f32[160]" = torch.ops.aten.mul.Tensor(primals_443, 0.9)
    add_263: "f32[160]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    unsqueeze_208: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1)
    unsqueeze_209: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_370: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
    unsqueeze_210: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1);  primals_106 = None
    unsqueeze_211: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_264: "f32[8, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_52: "f32[8, 160, 17, 17]" = torch.ops.aten.relu.default(add_264);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_53: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_52, primals_242, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_265: "i64[]" = torch.ops.aten.add.Tensor(primals_444, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 192, 1, 1]" = var_mean_53[0]
    getitem_113: "f32[1, 192, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_266: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 0.001)
    rsqrt_53: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
    sub_53: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_113)
    mul_371: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_160: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_372: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_373: "f32[192]" = torch.ops.aten.mul.Tensor(primals_445, 0.9)
    add_267: "f32[192]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    squeeze_161: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_374: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0004327131112072);  squeeze_161 = None
    mul_375: "f32[192]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
    mul_376: "f32[192]" = torch.ops.aten.mul.Tensor(primals_446, 0.9)
    add_268: "f32[192]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_212: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_213: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_377: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
    unsqueeze_214: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_215: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_269: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_53: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_269);  add_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_54: "f32[8, 160, 17, 17]" = torch.ops.aten.convolution.default(cat_5, primals_243, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_270: "i64[]" = torch.ops.aten.add.Tensor(primals_447, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 160, 1, 1]" = var_mean_54[0]
    getitem_115: "f32[1, 160, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_271: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 0.001)
    rsqrt_54: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
    sub_54: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_115)
    mul_378: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_163: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_379: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_380: "f32[160]" = torch.ops.aten.mul.Tensor(primals_448, 0.9)
    add_272: "f32[160]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    squeeze_164: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_381: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0004327131112072);  squeeze_164 = None
    mul_382: "f32[160]" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
    mul_383: "f32[160]" = torch.ops.aten.mul.Tensor(primals_449, 0.9)
    add_273: "f32[160]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    unsqueeze_216: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1)
    unsqueeze_217: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_384: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
    unsqueeze_218: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
    unsqueeze_219: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_274: "f32[8, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_54: "f32[8, 160, 17, 17]" = torch.ops.aten.relu.default(add_274);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_55: "f32[8, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_54, primals_244, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_275: "i64[]" = torch.ops.aten.add.Tensor(primals_450, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[1, 160, 1, 1]" = var_mean_55[0]
    getitem_117: "f32[1, 160, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_276: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 0.001)
    rsqrt_55: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
    sub_55: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_117)
    mul_385: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
    squeeze_166: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_386: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_387: "f32[160]" = torch.ops.aten.mul.Tensor(primals_451, 0.9)
    add_277: "f32[160]" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
    squeeze_167: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
    mul_388: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0004327131112072);  squeeze_167 = None
    mul_389: "f32[160]" = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
    mul_390: "f32[160]" = torch.ops.aten.mul.Tensor(primals_452, 0.9)
    add_278: "f32[160]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    unsqueeze_220: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1)
    unsqueeze_221: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_391: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
    unsqueeze_222: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
    unsqueeze_223: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_279: "f32[8, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_55: "f32[8, 160, 17, 17]" = torch.ops.aten.relu.default(add_279);  add_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_56: "f32[8, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_55, primals_245, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_280: "i64[]" = torch.ops.aten.add.Tensor(primals_453, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 160, 1, 1]" = var_mean_56[0]
    getitem_119: "f32[1, 160, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_281: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 0.001)
    rsqrt_56: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_281);  add_281 = None
    sub_56: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_119)
    mul_392: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_169: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_393: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_394: "f32[160]" = torch.ops.aten.mul.Tensor(primals_454, 0.9)
    add_282: "f32[160]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    squeeze_170: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_395: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0004327131112072);  squeeze_170 = None
    mul_396: "f32[160]" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
    mul_397: "f32[160]" = torch.ops.aten.mul.Tensor(primals_455, 0.9)
    add_283: "f32[160]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    unsqueeze_224: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_225: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_398: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_225);  mul_392 = unsqueeze_225 = None
    unsqueeze_226: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_227: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_284: "f32[8, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_227);  mul_398 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_56: "f32[8, 160, 17, 17]" = torch.ops.aten.relu.default(add_284);  add_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_57: "f32[8, 160, 17, 17]" = torch.ops.aten.convolution.default(relu_56, primals_246, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_285: "i64[]" = torch.ops.aten.add.Tensor(primals_456, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 160, 1, 1]" = var_mean_57[0]
    getitem_121: "f32[1, 160, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_286: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 0.001)
    rsqrt_57: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
    sub_57: "f32[8, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_121)
    mul_399: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_172: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_400: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_401: "f32[160]" = torch.ops.aten.mul.Tensor(primals_457, 0.9)
    add_287: "f32[160]" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
    squeeze_173: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_402: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0004327131112072);  squeeze_173 = None
    mul_403: "f32[160]" = torch.ops.aten.mul.Tensor(mul_402, 0.1);  mul_402 = None
    mul_404: "f32[160]" = torch.ops.aten.mul.Tensor(primals_458, 0.9)
    add_288: "f32[160]" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
    unsqueeze_228: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1)
    unsqueeze_229: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_405: "f32[8, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_229);  mul_399 = unsqueeze_229 = None
    unsqueeze_230: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1);  primals_116 = None
    unsqueeze_231: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_289: "f32[8, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_405, unsqueeze_231);  mul_405 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_57: "f32[8, 160, 17, 17]" = torch.ops.aten.relu.default(add_289);  add_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_58: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_57, primals_247, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_290: "i64[]" = torch.ops.aten.add.Tensor(primals_459, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_58 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[1, 192, 1, 1]" = var_mean_58[0]
    getitem_123: "f32[1, 192, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_291: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 0.001)
    rsqrt_58: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_291);  add_291 = None
    sub_58: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_123)
    mul_406: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    squeeze_174: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    squeeze_175: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_407: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_408: "f32[192]" = torch.ops.aten.mul.Tensor(primals_460, 0.9)
    add_292: "f32[192]" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    squeeze_176: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
    mul_409: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0004327131112072);  squeeze_176 = None
    mul_410: "f32[192]" = torch.ops.aten.mul.Tensor(mul_409, 0.1);  mul_409 = None
    mul_411: "f32[192]" = torch.ops.aten.mul.Tensor(primals_461, 0.9)
    add_293: "f32[192]" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    unsqueeze_232: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1)
    unsqueeze_233: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_412: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_233);  mul_406 = unsqueeze_233 = None
    unsqueeze_234: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1);  primals_118 = None
    unsqueeze_235: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_294: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_235);  mul_412 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_58: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_294);  add_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:120, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_5: "f32[8, 768, 17, 17]" = torch.ops.aten.avg_pool2d.default(cat_5, [3, 3], [1, 1], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_59: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(avg_pool2d_5, primals_248, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_295: "i64[]" = torch.ops.aten.add.Tensor(primals_462, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 192, 1, 1]" = var_mean_59[0]
    getitem_125: "f32[1, 192, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_296: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 0.001)
    rsqrt_59: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
    sub_59: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_125)
    mul_413: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    squeeze_177: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_178: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    mul_414: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_415: "f32[192]" = torch.ops.aten.mul.Tensor(primals_463, 0.9)
    add_297: "f32[192]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    squeeze_179: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_416: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0004327131112072);  squeeze_179 = None
    mul_417: "f32[192]" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
    mul_418: "f32[192]" = torch.ops.aten.mul.Tensor(primals_464, 0.9)
    add_298: "f32[192]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    unsqueeze_236: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_237: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_419: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_237);  mul_413 = unsqueeze_237 = None
    unsqueeze_238: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_239: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_299: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_239);  mul_419 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_59: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_299);  add_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:128, code: return torch.cat(outputs, 1)
    cat_6: "f32[8, 768, 17, 17]" = torch.ops.aten.cat.default([relu_50, relu_53, relu_58, relu_59], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_60: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_6, primals_249, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_300: "i64[]" = torch.ops.aten.add.Tensor(primals_465, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 192, 1, 1]" = var_mean_60[0]
    getitem_127: "f32[1, 192, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_301: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 0.001)
    rsqrt_60: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_301);  add_301 = None
    sub_60: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_127)
    mul_420: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    squeeze_180: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_181: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    mul_421: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_422: "f32[192]" = torch.ops.aten.mul.Tensor(primals_466, 0.9)
    add_302: "f32[192]" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    squeeze_182: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_423: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0004327131112072);  squeeze_182 = None
    mul_424: "f32[192]" = torch.ops.aten.mul.Tensor(mul_423, 0.1);  mul_423 = None
    mul_425: "f32[192]" = torch.ops.aten.mul.Tensor(primals_467, 0.9)
    add_303: "f32[192]" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
    unsqueeze_240: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1)
    unsqueeze_241: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_426: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_241);  mul_420 = unsqueeze_241 = None
    unsqueeze_242: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1);  primals_122 = None
    unsqueeze_243: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_304: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_243);  mul_426 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_60: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_304);  add_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_61: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_6, primals_250, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_305: "i64[]" = torch.ops.aten.add.Tensor(primals_468, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_61 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[1, 192, 1, 1]" = var_mean_61[0]
    getitem_129: "f32[1, 192, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_306: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 0.001)
    rsqrt_61: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
    sub_61: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_129)
    mul_427: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    squeeze_183: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
    squeeze_184: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_428: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_429: "f32[192]" = torch.ops.aten.mul.Tensor(primals_469, 0.9)
    add_307: "f32[192]" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    squeeze_185: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
    mul_430: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0004327131112072);  squeeze_185 = None
    mul_431: "f32[192]" = torch.ops.aten.mul.Tensor(mul_430, 0.1);  mul_430 = None
    mul_432: "f32[192]" = torch.ops.aten.mul.Tensor(primals_470, 0.9)
    add_308: "f32[192]" = torch.ops.aten.add.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
    unsqueeze_244: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1)
    unsqueeze_245: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_433: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_245);  mul_427 = unsqueeze_245 = None
    unsqueeze_246: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1);  primals_124 = None
    unsqueeze_247: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_309: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_433, unsqueeze_247);  mul_433 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_61: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_309);  add_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_62: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_61, primals_251, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_310: "i64[]" = torch.ops.aten.add.Tensor(primals_471, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_62 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
    getitem_130: "f32[1, 192, 1, 1]" = var_mean_62[0]
    getitem_131: "f32[1, 192, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_311: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 0.001)
    rsqrt_62: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_311);  add_311 = None
    sub_62: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_131)
    mul_434: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
    squeeze_186: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
    squeeze_187: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
    mul_435: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_436: "f32[192]" = torch.ops.aten.mul.Tensor(primals_472, 0.9)
    add_312: "f32[192]" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
    squeeze_188: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
    mul_437: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0004327131112072);  squeeze_188 = None
    mul_438: "f32[192]" = torch.ops.aten.mul.Tensor(mul_437, 0.1);  mul_437 = None
    mul_439: "f32[192]" = torch.ops.aten.mul.Tensor(primals_473, 0.9)
    add_313: "f32[192]" = torch.ops.aten.add.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_248: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_249: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    mul_440: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_249);  mul_434 = unsqueeze_249 = None
    unsqueeze_250: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_251: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    add_314: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_251);  mul_440 = unsqueeze_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_62: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_314);  add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_63: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_62, primals_252, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_315: "i64[]" = torch.ops.aten.add.Tensor(primals_474, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_63 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_132: "f32[1, 192, 1, 1]" = var_mean_63[0]
    getitem_133: "f32[1, 192, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_316: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 0.001)
    rsqrt_63: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_316);  add_316 = None
    sub_63: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_133)
    mul_441: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
    squeeze_189: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
    squeeze_190: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
    mul_442: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_443: "f32[192]" = torch.ops.aten.mul.Tensor(primals_475, 0.9)
    add_317: "f32[192]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    squeeze_191: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
    mul_444: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0004327131112072);  squeeze_191 = None
    mul_445: "f32[192]" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
    mul_446: "f32[192]" = torch.ops.aten.mul.Tensor(primals_476, 0.9)
    add_318: "f32[192]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
    unsqueeze_252: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1)
    unsqueeze_253: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_447: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_253);  mul_441 = unsqueeze_253 = None
    unsqueeze_254: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1);  primals_128 = None
    unsqueeze_255: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_319: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_255);  mul_447 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_63: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_319);  add_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_64: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_6, primals_253, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_320: "i64[]" = torch.ops.aten.add.Tensor(primals_477, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_64 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[1, 192, 1, 1]" = var_mean_64[0]
    getitem_135: "f32[1, 192, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_321: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 0.001)
    rsqrt_64: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_321);  add_321 = None
    sub_64: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_135)
    mul_448: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
    squeeze_192: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
    squeeze_193: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
    mul_449: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
    mul_450: "f32[192]" = torch.ops.aten.mul.Tensor(primals_478, 0.9)
    add_322: "f32[192]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    squeeze_194: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
    mul_451: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0004327131112072);  squeeze_194 = None
    mul_452: "f32[192]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
    mul_453: "f32[192]" = torch.ops.aten.mul.Tensor(primals_479, 0.9)
    add_323: "f32[192]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    unsqueeze_256: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1)
    unsqueeze_257: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    mul_454: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_257);  mul_448 = unsqueeze_257 = None
    unsqueeze_258: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1);  primals_130 = None
    unsqueeze_259: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    add_324: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_259);  mul_454 = unsqueeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_64: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_324);  add_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_65: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_64, primals_254, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_325: "i64[]" = torch.ops.aten.add.Tensor(primals_480, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_65 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_136: "f32[1, 192, 1, 1]" = var_mean_65[0]
    getitem_137: "f32[1, 192, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_326: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 0.001)
    rsqrt_65: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_326);  add_326 = None
    sub_65: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_137)
    mul_455: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
    squeeze_195: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
    squeeze_196: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
    mul_456: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
    mul_457: "f32[192]" = torch.ops.aten.mul.Tensor(primals_481, 0.9)
    add_327: "f32[192]" = torch.ops.aten.add.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    squeeze_197: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
    mul_458: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0004327131112072);  squeeze_197 = None
    mul_459: "f32[192]" = torch.ops.aten.mul.Tensor(mul_458, 0.1);  mul_458 = None
    mul_460: "f32[192]" = torch.ops.aten.mul.Tensor(primals_482, 0.9)
    add_328: "f32[192]" = torch.ops.aten.add.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
    unsqueeze_260: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_261: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_461: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_455, unsqueeze_261);  mul_455 = unsqueeze_261 = None
    unsqueeze_262: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_263: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_329: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_263);  mul_461 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_65: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_329);  add_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_66: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_65, primals_255, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_330: "i64[]" = torch.ops.aten.add.Tensor(primals_483, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_66 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_138: "f32[1, 192, 1, 1]" = var_mean_66[0]
    getitem_139: "f32[1, 192, 1, 1]" = var_mean_66[1];  var_mean_66 = None
    add_331: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 0.001)
    rsqrt_66: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
    sub_66: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_139)
    mul_462: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
    squeeze_198: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
    squeeze_199: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
    mul_463: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
    mul_464: "f32[192]" = torch.ops.aten.mul.Tensor(primals_484, 0.9)
    add_332: "f32[192]" = torch.ops.aten.add.Tensor(mul_463, mul_464);  mul_463 = mul_464 = None
    squeeze_200: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
    mul_465: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0004327131112072);  squeeze_200 = None
    mul_466: "f32[192]" = torch.ops.aten.mul.Tensor(mul_465, 0.1);  mul_465 = None
    mul_467: "f32[192]" = torch.ops.aten.mul.Tensor(primals_485, 0.9)
    add_333: "f32[192]" = torch.ops.aten.add.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
    unsqueeze_264: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_133, -1)
    unsqueeze_265: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    mul_468: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_265);  mul_462 = unsqueeze_265 = None
    unsqueeze_266: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1);  primals_134 = None
    unsqueeze_267: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    add_334: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_468, unsqueeze_267);  mul_468 = unsqueeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_66: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_334);  add_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_67: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_66, primals_256, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_335: "i64[]" = torch.ops.aten.add.Tensor(primals_486, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_67 = torch.ops.aten.var_mean.correction(convolution_67, [0, 2, 3], correction = 0, keepdim = True)
    getitem_140: "f32[1, 192, 1, 1]" = var_mean_67[0]
    getitem_141: "f32[1, 192, 1, 1]" = var_mean_67[1];  var_mean_67 = None
    add_336: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 0.001)
    rsqrt_67: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_336);  add_336 = None
    sub_67: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_67, getitem_141)
    mul_469: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
    squeeze_201: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
    squeeze_202: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
    mul_470: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
    mul_471: "f32[192]" = torch.ops.aten.mul.Tensor(primals_487, 0.9)
    add_337: "f32[192]" = torch.ops.aten.add.Tensor(mul_470, mul_471);  mul_470 = mul_471 = None
    squeeze_203: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
    mul_472: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_203, 1.0004327131112072);  squeeze_203 = None
    mul_473: "f32[192]" = torch.ops.aten.mul.Tensor(mul_472, 0.1);  mul_472 = None
    mul_474: "f32[192]" = torch.ops.aten.mul.Tensor(primals_488, 0.9)
    add_338: "f32[192]" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    unsqueeze_268: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1)
    unsqueeze_269: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_475: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_269);  mul_469 = unsqueeze_269 = None
    unsqueeze_270: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_136, -1);  primals_136 = None
    unsqueeze_271: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_339: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_475, unsqueeze_271);  mul_475 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_67: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_339);  add_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_68: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_67, primals_257, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_340: "i64[]" = torch.ops.aten.add.Tensor(primals_489, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_68 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True)
    getitem_142: "f32[1, 192, 1, 1]" = var_mean_68[0]
    getitem_143: "f32[1, 192, 1, 1]" = var_mean_68[1];  var_mean_68 = None
    add_341: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 0.001)
    rsqrt_68: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
    sub_68: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_143)
    mul_476: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
    squeeze_204: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
    squeeze_205: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
    mul_477: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1)
    mul_478: "f32[192]" = torch.ops.aten.mul.Tensor(primals_490, 0.9)
    add_342: "f32[192]" = torch.ops.aten.add.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
    squeeze_206: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
    mul_479: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_206, 1.0004327131112072);  squeeze_206 = None
    mul_480: "f32[192]" = torch.ops.aten.mul.Tensor(mul_479, 0.1);  mul_479 = None
    mul_481: "f32[192]" = torch.ops.aten.mul.Tensor(primals_491, 0.9)
    add_343: "f32[192]" = torch.ops.aten.add.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
    unsqueeze_272: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
    unsqueeze_273: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    mul_482: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_273);  mul_476 = unsqueeze_273 = None
    unsqueeze_274: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
    unsqueeze_275: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    add_344: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_275);  mul_482 = unsqueeze_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_68: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_344);  add_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:120, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_6: "f32[8, 768, 17, 17]" = torch.ops.aten.avg_pool2d.default(cat_6, [3, 3], [1, 1], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_69: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(avg_pool2d_6, primals_258, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_345: "i64[]" = torch.ops.aten.add.Tensor(primals_492, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_69 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
    getitem_144: "f32[1, 192, 1, 1]" = var_mean_69[0]
    getitem_145: "f32[1, 192, 1, 1]" = var_mean_69[1];  var_mean_69 = None
    add_346: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 0.001)
    rsqrt_69: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_346);  add_346 = None
    sub_69: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_145)
    mul_483: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
    squeeze_207: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
    squeeze_208: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
    mul_484: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1)
    mul_485: "f32[192]" = torch.ops.aten.mul.Tensor(primals_493, 0.9)
    add_347: "f32[192]" = torch.ops.aten.add.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
    squeeze_209: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
    mul_486: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_209, 1.0004327131112072);  squeeze_209 = None
    mul_487: "f32[192]" = torch.ops.aten.mul.Tensor(mul_486, 0.1);  mul_486 = None
    mul_488: "f32[192]" = torch.ops.aten.mul.Tensor(primals_494, 0.9)
    add_348: "f32[192]" = torch.ops.aten.add.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
    unsqueeze_276: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_139, -1)
    unsqueeze_277: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_489: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_483, unsqueeze_277);  mul_483 = unsqueeze_277 = None
    unsqueeze_278: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1);  primals_140 = None
    unsqueeze_279: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_349: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_489, unsqueeze_279);  mul_489 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_69: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_349);  add_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:128, code: return torch.cat(outputs, 1)
    cat_7: "f32[8, 768, 17, 17]" = torch.ops.aten.cat.default([relu_60, relu_63, relu_68, relu_69], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_70: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_7, primals_259, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_350: "i64[]" = torch.ops.aten.add.Tensor(primals_495, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_70 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
    getitem_146: "f32[1, 192, 1, 1]" = var_mean_70[0]
    getitem_147: "f32[1, 192, 1, 1]" = var_mean_70[1];  var_mean_70 = None
    add_351: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 0.001)
    rsqrt_70: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
    sub_70: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_147)
    mul_490: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
    squeeze_210: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
    squeeze_211: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
    mul_491: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
    mul_492: "f32[192]" = torch.ops.aten.mul.Tensor(primals_496, 0.9)
    add_352: "f32[192]" = torch.ops.aten.add.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
    squeeze_212: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
    mul_493: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_212, 1.0004327131112072);  squeeze_212 = None
    mul_494: "f32[192]" = torch.ops.aten.mul.Tensor(mul_493, 0.1);  mul_493 = None
    mul_495: "f32[192]" = torch.ops.aten.mul.Tensor(primals_497, 0.9)
    add_353: "f32[192]" = torch.ops.aten.add.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
    unsqueeze_280: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1)
    unsqueeze_281: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    mul_496: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_281);  mul_490 = unsqueeze_281 = None
    unsqueeze_282: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_142, -1);  primals_142 = None
    unsqueeze_283: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    add_354: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_496, unsqueeze_283);  mul_496 = unsqueeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_70: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_354);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_71: "f32[8, 320, 8, 8]" = torch.ops.aten.convolution.default(relu_70, primals_260, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_355: "i64[]" = torch.ops.aten.add.Tensor(primals_498, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_71 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
    getitem_148: "f32[1, 320, 1, 1]" = var_mean_71[0]
    getitem_149: "f32[1, 320, 1, 1]" = var_mean_71[1];  var_mean_71 = None
    add_356: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem_148, 0.001)
    rsqrt_71: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_356);  add_356 = None
    sub_71: "f32[8, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_149)
    mul_497: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
    squeeze_213: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
    squeeze_214: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
    mul_498: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1)
    mul_499: "f32[320]" = torch.ops.aten.mul.Tensor(primals_499, 0.9)
    add_357: "f32[320]" = torch.ops.aten.add.Tensor(mul_498, mul_499);  mul_498 = mul_499 = None
    squeeze_215: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
    mul_500: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_215, 1.0019569471624266);  squeeze_215 = None
    mul_501: "f32[320]" = torch.ops.aten.mul.Tensor(mul_500, 0.1);  mul_500 = None
    mul_502: "f32[320]" = torch.ops.aten.mul.Tensor(primals_500, 0.9)
    add_358: "f32[320]" = torch.ops.aten.add.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
    unsqueeze_284: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_285: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_503: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_497, unsqueeze_285);  mul_497 = unsqueeze_285 = None
    unsqueeze_286: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_287: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_359: "f32[8, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_287);  mul_503 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_71: "f32[8, 320, 8, 8]" = torch.ops.aten.relu.default(add_359);  add_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_72: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(cat_7, primals_261, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_360: "i64[]" = torch.ops.aten.add.Tensor(primals_501, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_72 = torch.ops.aten.var_mean.correction(convolution_72, [0, 2, 3], correction = 0, keepdim = True)
    getitem_150: "f32[1, 192, 1, 1]" = var_mean_72[0]
    getitem_151: "f32[1, 192, 1, 1]" = var_mean_72[1];  var_mean_72 = None
    add_361: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 0.001)
    rsqrt_72: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_361);  add_361 = None
    sub_72: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_72, getitem_151)
    mul_504: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
    squeeze_216: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
    squeeze_217: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
    mul_505: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1)
    mul_506: "f32[192]" = torch.ops.aten.mul.Tensor(primals_502, 0.9)
    add_362: "f32[192]" = torch.ops.aten.add.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    squeeze_218: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
    mul_507: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_218, 1.0004327131112072);  squeeze_218 = None
    mul_508: "f32[192]" = torch.ops.aten.mul.Tensor(mul_507, 0.1);  mul_507 = None
    mul_509: "f32[192]" = torch.ops.aten.mul.Tensor(primals_503, 0.9)
    add_363: "f32[192]" = torch.ops.aten.add.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
    unsqueeze_288: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_145, -1)
    unsqueeze_289: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    mul_510: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_504, unsqueeze_289);  mul_504 = unsqueeze_289 = None
    unsqueeze_290: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1);  primals_146 = None
    unsqueeze_291: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    add_364: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_510, unsqueeze_291);  mul_510 = unsqueeze_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_72: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_364);  add_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_73: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_72, primals_262, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_365: "i64[]" = torch.ops.aten.add.Tensor(primals_504, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_73 = torch.ops.aten.var_mean.correction(convolution_73, [0, 2, 3], correction = 0, keepdim = True)
    getitem_152: "f32[1, 192, 1, 1]" = var_mean_73[0]
    getitem_153: "f32[1, 192, 1, 1]" = var_mean_73[1];  var_mean_73 = None
    add_366: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_152, 0.001)
    rsqrt_73: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_366);  add_366 = None
    sub_73: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_73, getitem_153)
    mul_511: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
    squeeze_219: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
    squeeze_220: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
    mul_512: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
    mul_513: "f32[192]" = torch.ops.aten.mul.Tensor(primals_505, 0.9)
    add_367: "f32[192]" = torch.ops.aten.add.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
    squeeze_221: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
    mul_514: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_221, 1.0004327131112072);  squeeze_221 = None
    mul_515: "f32[192]" = torch.ops.aten.mul.Tensor(mul_514, 0.1);  mul_514 = None
    mul_516: "f32[192]" = torch.ops.aten.mul.Tensor(primals_506, 0.9)
    add_368: "f32[192]" = torch.ops.aten.add.Tensor(mul_515, mul_516);  mul_515 = mul_516 = None
    unsqueeze_292: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1)
    unsqueeze_293: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_517: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_511, unsqueeze_293);  mul_511 = unsqueeze_293 = None
    unsqueeze_294: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_148, -1);  primals_148 = None
    unsqueeze_295: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_369: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_517, unsqueeze_295);  mul_517 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_73: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_369);  add_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_74: "f32[8, 192, 17, 17]" = torch.ops.aten.convolution.default(relu_73, primals_263, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_370: "i64[]" = torch.ops.aten.add.Tensor(primals_507, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_74 = torch.ops.aten.var_mean.correction(convolution_74, [0, 2, 3], correction = 0, keepdim = True)
    getitem_154: "f32[1, 192, 1, 1]" = var_mean_74[0]
    getitem_155: "f32[1, 192, 1, 1]" = var_mean_74[1];  var_mean_74 = None
    add_371: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 0.001)
    rsqrt_74: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_371);  add_371 = None
    sub_74: "f32[8, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_74, getitem_155)
    mul_518: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
    squeeze_222: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
    squeeze_223: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
    mul_519: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
    mul_520: "f32[192]" = torch.ops.aten.mul.Tensor(primals_508, 0.9)
    add_372: "f32[192]" = torch.ops.aten.add.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
    squeeze_224: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
    mul_521: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_224, 1.0004327131112072);  squeeze_224 = None
    mul_522: "f32[192]" = torch.ops.aten.mul.Tensor(mul_521, 0.1);  mul_521 = None
    mul_523: "f32[192]" = torch.ops.aten.mul.Tensor(primals_509, 0.9)
    add_373: "f32[192]" = torch.ops.aten.add.Tensor(mul_522, mul_523);  mul_522 = mul_523 = None
    unsqueeze_296: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1)
    unsqueeze_297: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    mul_524: "f32[8, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_518, unsqueeze_297);  mul_518 = unsqueeze_297 = None
    unsqueeze_298: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
    unsqueeze_299: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    add_374: "f32[8, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_524, unsqueeze_299);  mul_524 = unsqueeze_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_74: "f32[8, 192, 17, 17]" = torch.ops.aten.relu.default(add_374);  add_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_75: "f32[8, 192, 8, 8]" = torch.ops.aten.convolution.default(relu_74, primals_264, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_375: "i64[]" = torch.ops.aten.add.Tensor(primals_510, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_75 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
    getitem_156: "f32[1, 192, 1, 1]" = var_mean_75[0]
    getitem_157: "f32[1, 192, 1, 1]" = var_mean_75[1];  var_mean_75 = None
    add_376: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 0.001)
    rsqrt_75: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
    sub_75: "f32[8, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_157)
    mul_525: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
    squeeze_225: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
    squeeze_226: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
    mul_526: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1)
    mul_527: "f32[192]" = torch.ops.aten.mul.Tensor(primals_511, 0.9)
    add_377: "f32[192]" = torch.ops.aten.add.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
    squeeze_227: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
    mul_528: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_227, 1.0019569471624266);  squeeze_227 = None
    mul_529: "f32[192]" = torch.ops.aten.mul.Tensor(mul_528, 0.1);  mul_528 = None
    mul_530: "f32[192]" = torch.ops.aten.mul.Tensor(primals_512, 0.9)
    add_378: "f32[192]" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    unsqueeze_300: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_151, -1)
    unsqueeze_301: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_531: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_301);  mul_525 = unsqueeze_301 = None
    unsqueeze_302: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1);  primals_152 = None
    unsqueeze_303: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_379: "f32[8, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_531, unsqueeze_303);  mul_531 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_75: "f32[8, 192, 8, 8]" = torch.ops.aten.relu.default(add_379);  add_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:153, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
    max_pool2d_with_indices_3 = torch.ops.aten.max_pool2d_with_indices.default(cat_7, [3, 3], [2, 2])
    getitem_158: "f32[8, 768, 8, 8]" = max_pool2d_with_indices_3[0]
    getitem_159: "i64[8, 768, 8, 8]" = max_pool2d_with_indices_3[1];  max_pool2d_with_indices_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:159, code: return torch.cat(outputs, 1)
    cat_8: "f32[8, 1280, 8, 8]" = torch.ops.aten.cat.default([relu_71, relu_75, getitem_158], 1);  getitem_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_76: "f32[8, 320, 8, 8]" = torch.ops.aten.convolution.default(cat_8, primals_265, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_380: "i64[]" = torch.ops.aten.add.Tensor(primals_513, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_76 = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
    getitem_160: "f32[1, 320, 1, 1]" = var_mean_76[0]
    getitem_161: "f32[1, 320, 1, 1]" = var_mean_76[1];  var_mean_76 = None
    add_381: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem_160, 0.001)
    rsqrt_76: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_381);  add_381 = None
    sub_76: "f32[8, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_161)
    mul_532: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
    squeeze_228: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3]);  getitem_161 = None
    squeeze_229: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
    mul_533: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1)
    mul_534: "f32[320]" = torch.ops.aten.mul.Tensor(primals_514, 0.9)
    add_382: "f32[320]" = torch.ops.aten.add.Tensor(mul_533, mul_534);  mul_533 = mul_534 = None
    squeeze_230: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_160, [0, 2, 3]);  getitem_160 = None
    mul_535: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_230, 1.0019569471624266);  squeeze_230 = None
    mul_536: "f32[320]" = torch.ops.aten.mul.Tensor(mul_535, 0.1);  mul_535 = None
    mul_537: "f32[320]" = torch.ops.aten.mul.Tensor(primals_515, 0.9)
    add_383: "f32[320]" = torch.ops.aten.add.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
    unsqueeze_304: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1)
    unsqueeze_305: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    mul_538: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_305);  mul_532 = unsqueeze_305 = None
    unsqueeze_306: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_154, -1);  primals_154 = None
    unsqueeze_307: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    add_384: "f32[8, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_538, unsqueeze_307);  mul_538 = unsqueeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_76: "f32[8, 320, 8, 8]" = torch.ops.aten.relu.default(add_384);  add_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_77: "f32[8, 384, 8, 8]" = torch.ops.aten.convolution.default(cat_8, primals_266, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_385: "i64[]" = torch.ops.aten.add.Tensor(primals_516, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_77 = torch.ops.aten.var_mean.correction(convolution_77, [0, 2, 3], correction = 0, keepdim = True)
    getitem_162: "f32[1, 384, 1, 1]" = var_mean_77[0]
    getitem_163: "f32[1, 384, 1, 1]" = var_mean_77[1];  var_mean_77 = None
    add_386: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 0.001)
    rsqrt_77: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_386);  add_386 = None
    sub_77: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_77, getitem_163)
    mul_539: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
    squeeze_231: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
    squeeze_232: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
    mul_540: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
    mul_541: "f32[384]" = torch.ops.aten.mul.Tensor(primals_517, 0.9)
    add_387: "f32[384]" = torch.ops.aten.add.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
    squeeze_233: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
    mul_542: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_233, 1.0019569471624266);  squeeze_233 = None
    mul_543: "f32[384]" = torch.ops.aten.mul.Tensor(mul_542, 0.1);  mul_542 = None
    mul_544: "f32[384]" = torch.ops.aten.mul.Tensor(primals_518, 0.9)
    add_388: "f32[384]" = torch.ops.aten.add.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
    unsqueeze_308: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
    unsqueeze_309: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_545: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_539, unsqueeze_309);  mul_539 = unsqueeze_309 = None
    unsqueeze_310: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
    unsqueeze_311: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_389: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_311);  mul_545 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_77: "f32[8, 384, 8, 8]" = torch.ops.aten.relu.default(add_389);  add_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_78: "f32[8, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_77, primals_267, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_390: "i64[]" = torch.ops.aten.add.Tensor(primals_519, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_78 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
    getitem_164: "f32[1, 384, 1, 1]" = var_mean_78[0]
    getitem_165: "f32[1, 384, 1, 1]" = var_mean_78[1];  var_mean_78 = None
    add_391: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_164, 0.001)
    rsqrt_78: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_391);  add_391 = None
    sub_78: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_165)
    mul_546: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
    squeeze_234: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
    squeeze_235: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
    mul_547: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1)
    mul_548: "f32[384]" = torch.ops.aten.mul.Tensor(primals_520, 0.9)
    add_392: "f32[384]" = torch.ops.aten.add.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
    squeeze_236: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
    mul_549: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_236, 1.0019569471624266);  squeeze_236 = None
    mul_550: "f32[384]" = torch.ops.aten.mul.Tensor(mul_549, 0.1);  mul_549 = None
    mul_551: "f32[384]" = torch.ops.aten.mul.Tensor(primals_521, 0.9)
    add_393: "f32[384]" = torch.ops.aten.add.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
    unsqueeze_312: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_157, -1)
    unsqueeze_313: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    mul_552: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_546, unsqueeze_313);  mul_546 = unsqueeze_313 = None
    unsqueeze_314: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_158, -1);  primals_158 = None
    unsqueeze_315: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    add_394: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_552, unsqueeze_315);  mul_552 = unsqueeze_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_78: "f32[8, 384, 8, 8]" = torch.ops.aten.relu.default(add_394);  add_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_79: "f32[8, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_77, primals_268, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_395: "i64[]" = torch.ops.aten.add.Tensor(primals_522, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_79 = torch.ops.aten.var_mean.correction(convolution_79, [0, 2, 3], correction = 0, keepdim = True)
    getitem_166: "f32[1, 384, 1, 1]" = var_mean_79[0]
    getitem_167: "f32[1, 384, 1, 1]" = var_mean_79[1];  var_mean_79 = None
    add_396: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_166, 0.001)
    rsqrt_79: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_396);  add_396 = None
    sub_79: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_79, getitem_167)
    mul_553: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
    squeeze_237: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
    squeeze_238: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
    mul_554: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1)
    mul_555: "f32[384]" = torch.ops.aten.mul.Tensor(primals_523, 0.9)
    add_397: "f32[384]" = torch.ops.aten.add.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
    squeeze_239: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
    mul_556: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_239, 1.0019569471624266);  squeeze_239 = None
    mul_557: "f32[384]" = torch.ops.aten.mul.Tensor(mul_556, 0.1);  mul_556 = None
    mul_558: "f32[384]" = torch.ops.aten.mul.Tensor(primals_524, 0.9)
    add_398: "f32[384]" = torch.ops.aten.add.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
    unsqueeze_316: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1)
    unsqueeze_317: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_559: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_317);  mul_553 = unsqueeze_317 = None
    unsqueeze_318: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_160, -1);  primals_160 = None
    unsqueeze_319: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_399: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_559, unsqueeze_319);  mul_559 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_79: "f32[8, 384, 8, 8]" = torch.ops.aten.relu.default(add_399);  add_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:188, code: branch3x3 = torch.cat(branch3x3, 1)
    cat_9: "f32[8, 768, 8, 8]" = torch.ops.aten.cat.default([relu_78, relu_79], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_80: "f32[8, 448, 8, 8]" = torch.ops.aten.convolution.default(cat_8, primals_269, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_400: "i64[]" = torch.ops.aten.add.Tensor(primals_525, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_80 = torch.ops.aten.var_mean.correction(convolution_80, [0, 2, 3], correction = 0, keepdim = True)
    getitem_168: "f32[1, 448, 1, 1]" = var_mean_80[0]
    getitem_169: "f32[1, 448, 1, 1]" = var_mean_80[1];  var_mean_80 = None
    add_401: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_168, 0.001)
    rsqrt_80: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_401);  add_401 = None
    sub_80: "f32[8, 448, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_169)
    mul_560: "f32[8, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = None
    squeeze_240: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
    squeeze_241: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
    mul_561: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_240, 0.1)
    mul_562: "f32[448]" = torch.ops.aten.mul.Tensor(primals_526, 0.9)
    add_402: "f32[448]" = torch.ops.aten.add.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    squeeze_242: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
    mul_563: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_242, 1.0019569471624266);  squeeze_242 = None
    mul_564: "f32[448]" = torch.ops.aten.mul.Tensor(mul_563, 0.1);  mul_563 = None
    mul_565: "f32[448]" = torch.ops.aten.mul.Tensor(primals_527, 0.9)
    add_403: "f32[448]" = torch.ops.aten.add.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_320: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_161, -1)
    unsqueeze_321: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    mul_566: "f32[8, 448, 8, 8]" = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_321);  mul_560 = unsqueeze_321 = None
    unsqueeze_322: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1);  primals_162 = None
    unsqueeze_323: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    add_404: "f32[8, 448, 8, 8]" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_323);  mul_566 = unsqueeze_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_80: "f32[8, 448, 8, 8]" = torch.ops.aten.relu.default(add_404);  add_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_81: "f32[8, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_80, primals_270, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_405: "i64[]" = torch.ops.aten.add.Tensor(primals_528, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_81 = torch.ops.aten.var_mean.correction(convolution_81, [0, 2, 3], correction = 0, keepdim = True)
    getitem_170: "f32[1, 384, 1, 1]" = var_mean_81[0]
    getitem_171: "f32[1, 384, 1, 1]" = var_mean_81[1];  var_mean_81 = None
    add_406: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_170, 0.001)
    rsqrt_81: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_406);  add_406 = None
    sub_81: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_81, getitem_171)
    mul_567: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = None
    squeeze_243: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
    squeeze_244: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
    mul_568: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_243, 0.1)
    mul_569: "f32[384]" = torch.ops.aten.mul.Tensor(primals_529, 0.9)
    add_407: "f32[384]" = torch.ops.aten.add.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
    squeeze_245: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_170, [0, 2, 3]);  getitem_170 = None
    mul_570: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_245, 1.0019569471624266);  squeeze_245 = None
    mul_571: "f32[384]" = torch.ops.aten.mul.Tensor(mul_570, 0.1);  mul_570 = None
    mul_572: "f32[384]" = torch.ops.aten.mul.Tensor(primals_530, 0.9)
    add_408: "f32[384]" = torch.ops.aten.add.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
    unsqueeze_324: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1)
    unsqueeze_325: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_573: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_567, unsqueeze_325);  mul_567 = unsqueeze_325 = None
    unsqueeze_326: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_164, -1);  primals_164 = None
    unsqueeze_327: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_409: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_573, unsqueeze_327);  mul_573 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_81: "f32[8, 384, 8, 8]" = torch.ops.aten.relu.default(add_409);  add_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_82: "f32[8, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_81, primals_271, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_410: "i64[]" = torch.ops.aten.add.Tensor(primals_531, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_82 = torch.ops.aten.var_mean.correction(convolution_82, [0, 2, 3], correction = 0, keepdim = True)
    getitem_172: "f32[1, 384, 1, 1]" = var_mean_82[0]
    getitem_173: "f32[1, 384, 1, 1]" = var_mean_82[1];  var_mean_82 = None
    add_411: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_172, 0.001)
    rsqrt_82: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_411);  add_411 = None
    sub_82: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_82, getitem_173)
    mul_574: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
    squeeze_246: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3]);  getitem_173 = None
    squeeze_247: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
    mul_575: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_246, 0.1)
    mul_576: "f32[384]" = torch.ops.aten.mul.Tensor(primals_532, 0.9)
    add_412: "f32[384]" = torch.ops.aten.add.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
    squeeze_248: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_172, [0, 2, 3]);  getitem_172 = None
    mul_577: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_248, 1.0019569471624266);  squeeze_248 = None
    mul_578: "f32[384]" = torch.ops.aten.mul.Tensor(mul_577, 0.1);  mul_577 = None
    mul_579: "f32[384]" = torch.ops.aten.mul.Tensor(primals_533, 0.9)
    add_413: "f32[384]" = torch.ops.aten.add.Tensor(mul_578, mul_579);  mul_578 = mul_579 = None
    unsqueeze_328: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_165, -1)
    unsqueeze_329: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    mul_580: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_329);  mul_574 = unsqueeze_329 = None
    unsqueeze_330: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_166, -1);  primals_166 = None
    unsqueeze_331: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    add_414: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_580, unsqueeze_331);  mul_580 = unsqueeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_82: "f32[8, 384, 8, 8]" = torch.ops.aten.relu.default(add_414);  add_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_83: "f32[8, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_81, primals_272, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_415: "i64[]" = torch.ops.aten.add.Tensor(primals_534, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_83 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
    getitem_174: "f32[1, 384, 1, 1]" = var_mean_83[0]
    getitem_175: "f32[1, 384, 1, 1]" = var_mean_83[1];  var_mean_83 = None
    add_416: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_174, 0.001)
    rsqrt_83: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_416);  add_416 = None
    sub_83: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_175)
    mul_581: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
    squeeze_249: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
    squeeze_250: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
    mul_582: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_249, 0.1)
    mul_583: "f32[384]" = torch.ops.aten.mul.Tensor(primals_535, 0.9)
    add_417: "f32[384]" = torch.ops.aten.add.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    squeeze_251: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_174, [0, 2, 3]);  getitem_174 = None
    mul_584: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_251, 1.0019569471624266);  squeeze_251 = None
    mul_585: "f32[384]" = torch.ops.aten.mul.Tensor(mul_584, 0.1);  mul_584 = None
    mul_586: "f32[384]" = torch.ops.aten.mul.Tensor(primals_536, 0.9)
    add_418: "f32[384]" = torch.ops.aten.add.Tensor(mul_585, mul_586);  mul_585 = mul_586 = None
    unsqueeze_332: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1)
    unsqueeze_333: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_587: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_581, unsqueeze_333);  mul_581 = unsqueeze_333 = None
    unsqueeze_334: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1);  primals_168 = None
    unsqueeze_335: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_419: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_335);  mul_587 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_83: "f32[8, 384, 8, 8]" = torch.ops.aten.relu.default(add_419);  add_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:196, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
    cat_10: "f32[8, 768, 8, 8]" = torch.ops.aten.cat.default([relu_82, relu_83], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:198, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_7: "f32[8, 1280, 8, 8]" = torch.ops.aten.avg_pool2d.default(cat_8, [3, 3], [1, 1], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_84: "f32[8, 192, 8, 8]" = torch.ops.aten.convolution.default(avg_pool2d_7, primals_273, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_420: "i64[]" = torch.ops.aten.add.Tensor(primals_537, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_84 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
    getitem_176: "f32[1, 192, 1, 1]" = var_mean_84[0]
    getitem_177: "f32[1, 192, 1, 1]" = var_mean_84[1];  var_mean_84 = None
    add_421: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_176, 0.001)
    rsqrt_84: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_421);  add_421 = None
    sub_84: "f32[8, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_177)
    mul_588: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
    squeeze_252: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_177, [0, 2, 3]);  getitem_177 = None
    squeeze_253: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
    mul_589: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_252, 0.1)
    mul_590: "f32[192]" = torch.ops.aten.mul.Tensor(primals_538, 0.9)
    add_422: "f32[192]" = torch.ops.aten.add.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
    squeeze_254: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_176, [0, 2, 3]);  getitem_176 = None
    mul_591: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_254, 1.0019569471624266);  squeeze_254 = None
    mul_592: "f32[192]" = torch.ops.aten.mul.Tensor(mul_591, 0.1);  mul_591 = None
    mul_593: "f32[192]" = torch.ops.aten.mul.Tensor(primals_539, 0.9)
    add_423: "f32[192]" = torch.ops.aten.add.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
    unsqueeze_336: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_169, -1)
    unsqueeze_337: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    mul_594: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_588, unsqueeze_337);  mul_588 = unsqueeze_337 = None
    unsqueeze_338: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_170, -1);  primals_170 = None
    unsqueeze_339: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    add_424: "f32[8, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_594, unsqueeze_339);  mul_594 = unsqueeze_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_84: "f32[8, 192, 8, 8]" = torch.ops.aten.relu.default(add_424);  add_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:206, code: return torch.cat(outputs, 1)
    cat_11: "f32[8, 2048, 8, 8]" = torch.ops.aten.cat.default([relu_76, cat_9, cat_10, relu_84], 1);  cat_9 = cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_85: "f32[8, 320, 8, 8]" = torch.ops.aten.convolution.default(cat_11, primals_274, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_425: "i64[]" = torch.ops.aten.add.Tensor(primals_540, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_85 = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True)
    getitem_178: "f32[1, 320, 1, 1]" = var_mean_85[0]
    getitem_179: "f32[1, 320, 1, 1]" = var_mean_85[1];  var_mean_85 = None
    add_426: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem_178, 0.001)
    rsqrt_85: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_426);  add_426 = None
    sub_85: "f32[8, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_179)
    mul_595: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = None
    squeeze_255: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
    squeeze_256: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
    mul_596: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_255, 0.1)
    mul_597: "f32[320]" = torch.ops.aten.mul.Tensor(primals_541, 0.9)
    add_427: "f32[320]" = torch.ops.aten.add.Tensor(mul_596, mul_597);  mul_596 = mul_597 = None
    squeeze_257: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_178, [0, 2, 3]);  getitem_178 = None
    mul_598: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_257, 1.0019569471624266);  squeeze_257 = None
    mul_599: "f32[320]" = torch.ops.aten.mul.Tensor(mul_598, 0.1);  mul_598 = None
    mul_600: "f32[320]" = torch.ops.aten.mul.Tensor(primals_542, 0.9)
    add_428: "f32[320]" = torch.ops.aten.add.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
    unsqueeze_340: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_171, -1)
    unsqueeze_341: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_601: "f32[8, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_341);  mul_595 = unsqueeze_341 = None
    unsqueeze_342: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_172, -1);  primals_172 = None
    unsqueeze_343: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_429: "f32[8, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_601, unsqueeze_343);  mul_601 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_85: "f32[8, 320, 8, 8]" = torch.ops.aten.relu.default(add_429);  add_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_86: "f32[8, 384, 8, 8]" = torch.ops.aten.convolution.default(cat_11, primals_275, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_430: "i64[]" = torch.ops.aten.add.Tensor(primals_543, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_86 = torch.ops.aten.var_mean.correction(convolution_86, [0, 2, 3], correction = 0, keepdim = True)
    getitem_180: "f32[1, 384, 1, 1]" = var_mean_86[0]
    getitem_181: "f32[1, 384, 1, 1]" = var_mean_86[1];  var_mean_86 = None
    add_431: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_180, 0.001)
    rsqrt_86: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_431);  add_431 = None
    sub_86: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_86, getitem_181)
    mul_602: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = None
    squeeze_258: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_181, [0, 2, 3]);  getitem_181 = None
    squeeze_259: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_86, [0, 2, 3]);  rsqrt_86 = None
    mul_603: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_258, 0.1)
    mul_604: "f32[384]" = torch.ops.aten.mul.Tensor(primals_544, 0.9)
    add_432: "f32[384]" = torch.ops.aten.add.Tensor(mul_603, mul_604);  mul_603 = mul_604 = None
    squeeze_260: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_180, [0, 2, 3]);  getitem_180 = None
    mul_605: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_260, 1.0019569471624266);  squeeze_260 = None
    mul_606: "f32[384]" = torch.ops.aten.mul.Tensor(mul_605, 0.1);  mul_605 = None
    mul_607: "f32[384]" = torch.ops.aten.mul.Tensor(primals_545, 0.9)
    add_433: "f32[384]" = torch.ops.aten.add.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
    unsqueeze_344: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_173, -1)
    unsqueeze_345: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    mul_608: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_602, unsqueeze_345);  mul_602 = unsqueeze_345 = None
    unsqueeze_346: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_174, -1);  primals_174 = None
    unsqueeze_347: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    add_434: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_608, unsqueeze_347);  mul_608 = unsqueeze_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_86: "f32[8, 384, 8, 8]" = torch.ops.aten.relu.default(add_434);  add_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_87: "f32[8, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_86, primals_276, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_435: "i64[]" = torch.ops.aten.add.Tensor(primals_546, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_87 = torch.ops.aten.var_mean.correction(convolution_87, [0, 2, 3], correction = 0, keepdim = True)
    getitem_182: "f32[1, 384, 1, 1]" = var_mean_87[0]
    getitem_183: "f32[1, 384, 1, 1]" = var_mean_87[1];  var_mean_87 = None
    add_436: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_182, 0.001)
    rsqrt_87: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_436);  add_436 = None
    sub_87: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_87, getitem_183)
    mul_609: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = None
    squeeze_261: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3]);  getitem_183 = None
    squeeze_262: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
    mul_610: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_261, 0.1)
    mul_611: "f32[384]" = torch.ops.aten.mul.Tensor(primals_547, 0.9)
    add_437: "f32[384]" = torch.ops.aten.add.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
    squeeze_263: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_182, [0, 2, 3]);  getitem_182 = None
    mul_612: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_263, 1.0019569471624266);  squeeze_263 = None
    mul_613: "f32[384]" = torch.ops.aten.mul.Tensor(mul_612, 0.1);  mul_612 = None
    mul_614: "f32[384]" = torch.ops.aten.mul.Tensor(primals_548, 0.9)
    add_438: "f32[384]" = torch.ops.aten.add.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
    unsqueeze_348: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_175, -1)
    unsqueeze_349: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_615: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_609, unsqueeze_349);  mul_609 = unsqueeze_349 = None
    unsqueeze_350: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_176, -1);  primals_176 = None
    unsqueeze_351: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_439: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_615, unsqueeze_351);  mul_615 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_87: "f32[8, 384, 8, 8]" = torch.ops.aten.relu.default(add_439);  add_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_88: "f32[8, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_86, primals_277, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_440: "i64[]" = torch.ops.aten.add.Tensor(primals_549, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_88 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
    getitem_184: "f32[1, 384, 1, 1]" = var_mean_88[0]
    getitem_185: "f32[1, 384, 1, 1]" = var_mean_88[1];  var_mean_88 = None
    add_441: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_184, 0.001)
    rsqrt_88: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_441);  add_441 = None
    sub_88: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_185)
    mul_616: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = None
    squeeze_264: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_185, [0, 2, 3]);  getitem_185 = None
    squeeze_265: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
    mul_617: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_264, 0.1)
    mul_618: "f32[384]" = torch.ops.aten.mul.Tensor(primals_550, 0.9)
    add_442: "f32[384]" = torch.ops.aten.add.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
    squeeze_266: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_184, [0, 2, 3]);  getitem_184 = None
    mul_619: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_266, 1.0019569471624266);  squeeze_266 = None
    mul_620: "f32[384]" = torch.ops.aten.mul.Tensor(mul_619, 0.1);  mul_619 = None
    mul_621: "f32[384]" = torch.ops.aten.mul.Tensor(primals_551, 0.9)
    add_443: "f32[384]" = torch.ops.aten.add.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
    unsqueeze_352: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_177, -1)
    unsqueeze_353: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    mul_622: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_353);  mul_616 = unsqueeze_353 = None
    unsqueeze_354: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_178, -1);  primals_178 = None
    unsqueeze_355: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    add_444: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_622, unsqueeze_355);  mul_622 = unsqueeze_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_88: "f32[8, 384, 8, 8]" = torch.ops.aten.relu.default(add_444);  add_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:188, code: branch3x3 = torch.cat(branch3x3, 1)
    cat_12: "f32[8, 768, 8, 8]" = torch.ops.aten.cat.default([relu_87, relu_88], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_89: "f32[8, 448, 8, 8]" = torch.ops.aten.convolution.default(cat_11, primals_278, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_445: "i64[]" = torch.ops.aten.add.Tensor(primals_552, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_89 = torch.ops.aten.var_mean.correction(convolution_89, [0, 2, 3], correction = 0, keepdim = True)
    getitem_186: "f32[1, 448, 1, 1]" = var_mean_89[0]
    getitem_187: "f32[1, 448, 1, 1]" = var_mean_89[1];  var_mean_89 = None
    add_446: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_186, 0.001)
    rsqrt_89: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_446);  add_446 = None
    sub_89: "f32[8, 448, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_89, getitem_187)
    mul_623: "f32[8, 448, 8, 8]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = None
    squeeze_267: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
    squeeze_268: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_89, [0, 2, 3]);  rsqrt_89 = None
    mul_624: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_267, 0.1)
    mul_625: "f32[448]" = torch.ops.aten.mul.Tensor(primals_553, 0.9)
    add_447: "f32[448]" = torch.ops.aten.add.Tensor(mul_624, mul_625);  mul_624 = mul_625 = None
    squeeze_269: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_186, [0, 2, 3]);  getitem_186 = None
    mul_626: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_269, 1.0019569471624266);  squeeze_269 = None
    mul_627: "f32[448]" = torch.ops.aten.mul.Tensor(mul_626, 0.1);  mul_626 = None
    mul_628: "f32[448]" = torch.ops.aten.mul.Tensor(primals_554, 0.9)
    add_448: "f32[448]" = torch.ops.aten.add.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
    unsqueeze_356: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_179, -1)
    unsqueeze_357: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_629: "f32[8, 448, 8, 8]" = torch.ops.aten.mul.Tensor(mul_623, unsqueeze_357);  mul_623 = unsqueeze_357 = None
    unsqueeze_358: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_180, -1);  primals_180 = None
    unsqueeze_359: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_449: "f32[8, 448, 8, 8]" = torch.ops.aten.add.Tensor(mul_629, unsqueeze_359);  mul_629 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_89: "f32[8, 448, 8, 8]" = torch.ops.aten.relu.default(add_449);  add_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_90: "f32[8, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_89, primals_279, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_450: "i64[]" = torch.ops.aten.add.Tensor(primals_555, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_90 = torch.ops.aten.var_mean.correction(convolution_90, [0, 2, 3], correction = 0, keepdim = True)
    getitem_188: "f32[1, 384, 1, 1]" = var_mean_90[0]
    getitem_189: "f32[1, 384, 1, 1]" = var_mean_90[1];  var_mean_90 = None
    add_451: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_188, 0.001)
    rsqrt_90: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_451);  add_451 = None
    sub_90: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_90, getitem_189)
    mul_630: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = None
    squeeze_270: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
    squeeze_271: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_90, [0, 2, 3]);  rsqrt_90 = None
    mul_631: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_270, 0.1)
    mul_632: "f32[384]" = torch.ops.aten.mul.Tensor(primals_556, 0.9)
    add_452: "f32[384]" = torch.ops.aten.add.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
    squeeze_272: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_188, [0, 2, 3]);  getitem_188 = None
    mul_633: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_272, 1.0019569471624266);  squeeze_272 = None
    mul_634: "f32[384]" = torch.ops.aten.mul.Tensor(mul_633, 0.1);  mul_633 = None
    mul_635: "f32[384]" = torch.ops.aten.mul.Tensor(primals_557, 0.9)
    add_453: "f32[384]" = torch.ops.aten.add.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
    unsqueeze_360: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_181, -1)
    unsqueeze_361: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    mul_636: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_630, unsqueeze_361);  mul_630 = unsqueeze_361 = None
    unsqueeze_362: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_182, -1);  primals_182 = None
    unsqueeze_363: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    add_454: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_636, unsqueeze_363);  mul_636 = unsqueeze_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_90: "f32[8, 384, 8, 8]" = torch.ops.aten.relu.default(add_454);  add_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_91: "f32[8, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_90, primals_280, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_455: "i64[]" = torch.ops.aten.add.Tensor(primals_558, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_91 = torch.ops.aten.var_mean.correction(convolution_91, [0, 2, 3], correction = 0, keepdim = True)
    getitem_190: "f32[1, 384, 1, 1]" = var_mean_91[0]
    getitem_191: "f32[1, 384, 1, 1]" = var_mean_91[1];  var_mean_91 = None
    add_456: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_190, 0.001)
    rsqrt_91: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_456);  add_456 = None
    sub_91: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_91, getitem_191)
    mul_637: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = None
    squeeze_273: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_191, [0, 2, 3]);  getitem_191 = None
    squeeze_274: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_91, [0, 2, 3]);  rsqrt_91 = None
    mul_638: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_273, 0.1)
    mul_639: "f32[384]" = torch.ops.aten.mul.Tensor(primals_559, 0.9)
    add_457: "f32[384]" = torch.ops.aten.add.Tensor(mul_638, mul_639);  mul_638 = mul_639 = None
    squeeze_275: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_190, [0, 2, 3]);  getitem_190 = None
    mul_640: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_275, 1.0019569471624266);  squeeze_275 = None
    mul_641: "f32[384]" = torch.ops.aten.mul.Tensor(mul_640, 0.1);  mul_640 = None
    mul_642: "f32[384]" = torch.ops.aten.mul.Tensor(primals_560, 0.9)
    add_458: "f32[384]" = torch.ops.aten.add.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
    unsqueeze_364: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_183, -1)
    unsqueeze_365: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_643: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_365);  mul_637 = unsqueeze_365 = None
    unsqueeze_366: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_184, -1);  primals_184 = None
    unsqueeze_367: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_459: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_643, unsqueeze_367);  mul_643 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_91: "f32[8, 384, 8, 8]" = torch.ops.aten.relu.default(add_459);  add_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_92: "f32[8, 384, 8, 8]" = torch.ops.aten.convolution.default(relu_90, primals_281, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_460: "i64[]" = torch.ops.aten.add.Tensor(primals_561, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_92 = torch.ops.aten.var_mean.correction(convolution_92, [0, 2, 3], correction = 0, keepdim = True)
    getitem_192: "f32[1, 384, 1, 1]" = var_mean_92[0]
    getitem_193: "f32[1, 384, 1, 1]" = var_mean_92[1];  var_mean_92 = None
    add_461: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_192, 0.001)
    rsqrt_92: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_461);  add_461 = None
    sub_92: "f32[8, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_193)
    mul_644: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = None
    squeeze_276: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
    squeeze_277: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_92, [0, 2, 3]);  rsqrt_92 = None
    mul_645: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_276, 0.1)
    mul_646: "f32[384]" = torch.ops.aten.mul.Tensor(primals_562, 0.9)
    add_462: "f32[384]" = torch.ops.aten.add.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
    squeeze_278: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_192, [0, 2, 3]);  getitem_192 = None
    mul_647: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_278, 1.0019569471624266);  squeeze_278 = None
    mul_648: "f32[384]" = torch.ops.aten.mul.Tensor(mul_647, 0.1);  mul_647 = None
    mul_649: "f32[384]" = torch.ops.aten.mul.Tensor(primals_563, 0.9)
    add_463: "f32[384]" = torch.ops.aten.add.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
    unsqueeze_368: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_185, -1)
    unsqueeze_369: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    mul_650: "f32[8, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_644, unsqueeze_369);  mul_644 = unsqueeze_369 = None
    unsqueeze_370: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_186, -1);  primals_186 = None
    unsqueeze_371: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    add_464: "f32[8, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_650, unsqueeze_371);  mul_650 = unsqueeze_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_92: "f32[8, 384, 8, 8]" = torch.ops.aten.relu.default(add_464);  add_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:196, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
    cat_13: "f32[8, 768, 8, 8]" = torch.ops.aten.cat.default([relu_91, relu_92], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:198, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    avg_pool2d_8: "f32[8, 2048, 8, 8]" = torch.ops.aten.avg_pool2d.default(cat_11, [3, 3], [1, 1], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_93: "f32[8, 192, 8, 8]" = torch.ops.aten.convolution.default(avg_pool2d_8, primals_282, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_465: "i64[]" = torch.ops.aten.add.Tensor(primals_564, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_93 = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True)
    getitem_194: "f32[1, 192, 1, 1]" = var_mean_93[0]
    getitem_195: "f32[1, 192, 1, 1]" = var_mean_93[1];  var_mean_93 = None
    add_466: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_194, 0.001)
    rsqrt_93: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_466);  add_466 = None
    sub_93: "f32[8, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_195)
    mul_651: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = None
    squeeze_279: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3]);  getitem_195 = None
    squeeze_280: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
    mul_652: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_279, 0.1)
    mul_653: "f32[192]" = torch.ops.aten.mul.Tensor(primals_565, 0.9)
    add_467: "f32[192]" = torch.ops.aten.add.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
    squeeze_281: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_194, [0, 2, 3]);  getitem_194 = None
    mul_654: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_281, 1.0019569471624266);  squeeze_281 = None
    mul_655: "f32[192]" = torch.ops.aten.mul.Tensor(mul_654, 0.1);  mul_654 = None
    mul_656: "f32[192]" = torch.ops.aten.mul.Tensor(primals_566, 0.9)
    add_468: "f32[192]" = torch.ops.aten.add.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
    unsqueeze_372: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_187, -1)
    unsqueeze_373: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_657: "f32[8, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_651, unsqueeze_373);  mul_651 = unsqueeze_373 = None
    unsqueeze_374: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_188, -1);  primals_188 = None
    unsqueeze_375: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_469: "f32[8, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_657, unsqueeze_375);  mul_657 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    relu_93: "f32[8, 192, 8, 8]" = torch.ops.aten.relu.default(add_469);  add_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:206, code: return torch.cat(outputs, 1)
    cat_14: "f32[8, 2048, 8, 8]" = torch.ops.aten.cat.default([relu_85, cat_12, cat_13, relu_93], 1);  cat_12 = cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(cat_14, [-1, -2], True);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 2048]" = torch.ops.aten.reshape.default(mean, [8, 2048]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/inception_v3.py:377, code: x = self.fc(x)
    permute: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_284, view, permute);  primals_284 = None
    permute_1: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le: "b8[8, 192, 8, 8]" = torch.ops.aten.le.Scalar(relu_93, 0);  relu_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_376: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_279, 0);  squeeze_279 = None
    unsqueeze_377: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
    unsqueeze_378: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_1: "b8[8, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_92, 0);  relu_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_388: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_276, 0);  squeeze_276 = None
    unsqueeze_389: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
    unsqueeze_390: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_2: "b8[8, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_91, 0);  relu_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_400: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_273, 0);  squeeze_273 = None
    unsqueeze_401: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
    unsqueeze_402: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_412: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_270, 0);  squeeze_270 = None
    unsqueeze_413: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
    unsqueeze_414: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_424: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_267, 0);  squeeze_267 = None
    unsqueeze_425: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
    unsqueeze_426: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_5: "b8[8, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_88, 0);  relu_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_436: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_264, 0);  squeeze_264 = None
    unsqueeze_437: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
    unsqueeze_438: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_6: "b8[8, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_87, 0);  relu_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_448: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_261, 0);  squeeze_261 = None
    unsqueeze_449: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
    unsqueeze_450: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_460: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_258, 0);  squeeze_258 = None
    unsqueeze_461: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
    unsqueeze_462: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_8: "b8[8, 320, 8, 8]" = torch.ops.aten.le.Scalar(relu_85, 0);  relu_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_472: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_255, 0);  squeeze_255 = None
    unsqueeze_473: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
    unsqueeze_474: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_9: "b8[8, 192, 8, 8]" = torch.ops.aten.le.Scalar(relu_84, 0);  relu_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_484: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
    unsqueeze_485: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
    unsqueeze_486: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_10: "b8[8, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_83, 0);  relu_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_496: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
    unsqueeze_497: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
    unsqueeze_498: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_11: "b8[8, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_82, 0);  relu_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_508: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
    unsqueeze_509: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
    unsqueeze_510: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_520: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
    unsqueeze_521: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
    unsqueeze_522: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_532: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
    unsqueeze_533: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
    unsqueeze_534: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_14: "b8[8, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_79, 0);  relu_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_544: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
    unsqueeze_545: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
    unsqueeze_546: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_15: "b8[8, 384, 8, 8]" = torch.ops.aten.le.Scalar(relu_78, 0);  relu_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_556: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
    unsqueeze_557: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_568: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
    unsqueeze_569: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
    unsqueeze_570: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_17: "b8[8, 320, 8, 8]" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_580: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
    unsqueeze_581: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
    unsqueeze_582: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_18: "b8[8, 192, 8, 8]" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_592: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
    unsqueeze_593: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
    unsqueeze_594: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_604: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
    unsqueeze_605: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
    unsqueeze_606: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_616: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
    unsqueeze_617: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
    unsqueeze_618: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_628: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
    unsqueeze_629: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
    unsqueeze_630: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_22: "b8[8, 320, 8, 8]" = torch.ops.aten.le.Scalar(relu_71, 0);  relu_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_640: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
    unsqueeze_641: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
    unsqueeze_642: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_652: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
    unsqueeze_653: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
    unsqueeze_654: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_24: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_69, 0);  relu_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_664: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
    unsqueeze_665: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
    unsqueeze_666: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_25: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_676: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
    unsqueeze_677: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
    unsqueeze_678: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_688: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
    unsqueeze_689: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
    unsqueeze_690: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_700: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
    unsqueeze_701: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
    unsqueeze_702: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_712: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
    unsqueeze_713: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
    unsqueeze_714: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_724: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_725: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
    unsqueeze_726: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_30: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_736: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_737: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
    unsqueeze_738: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_748: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_749: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
    unsqueeze_750: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_760: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_761: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
    unsqueeze_762: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_33: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_772: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_773: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
    unsqueeze_774: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_34: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_784: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_785: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
    unsqueeze_786: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_35: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_796: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_797: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
    unsqueeze_798: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_808: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_809: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
    unsqueeze_810: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_820: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_821: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
    unsqueeze_822: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_832: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_833: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
    unsqueeze_834: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_844: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_845: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
    unsqueeze_846: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_40: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_856: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_857: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
    unsqueeze_858: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_868: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_869: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
    unsqueeze_870: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_880: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_881: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
    unsqueeze_882: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_43: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_892: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_893: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
    unsqueeze_894: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_44: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_904: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_905: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
    unsqueeze_906: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_45: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_916: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_917: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
    unsqueeze_918: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_928: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_929: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 2);  unsqueeze_928 = None
    unsqueeze_930: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 3);  unsqueeze_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_940: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_941: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 2);  unsqueeze_940 = None
    unsqueeze_942: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 3);  unsqueeze_941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_952: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_953: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 2);  unsqueeze_952 = None
    unsqueeze_954: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 3);  unsqueeze_953 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_964: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_965: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 2);  unsqueeze_964 = None
    unsqueeze_966: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 3);  unsqueeze_965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_50: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_976: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_977: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 2);  unsqueeze_976 = None
    unsqueeze_978: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 3);  unsqueeze_977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_988: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_989: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 2);  unsqueeze_988 = None
    unsqueeze_990: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 3);  unsqueeze_989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1000: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_1001: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 2);  unsqueeze_1000 = None
    unsqueeze_1002: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 3);  unsqueeze_1001 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_53: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1012: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_1013: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 2);  unsqueeze_1012 = None
    unsqueeze_1014: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 3);  unsqueeze_1013 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_54: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1024: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_1025: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 2);  unsqueeze_1024 = None
    unsqueeze_1026: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 3);  unsqueeze_1025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_55: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1036: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_1037: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 2);  unsqueeze_1036 = None
    unsqueeze_1038: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 3);  unsqueeze_1037 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1048: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_1049: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 2);  unsqueeze_1048 = None
    unsqueeze_1050: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 3);  unsqueeze_1049 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1060: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_1061: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 2);  unsqueeze_1060 = None
    unsqueeze_1062: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 3);  unsqueeze_1061 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1072: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_1073: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 2);  unsqueeze_1072 = None
    unsqueeze_1074: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 3);  unsqueeze_1073 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1084: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_1085: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 2);  unsqueeze_1084 = None
    unsqueeze_1086: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 3);  unsqueeze_1085 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_60: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1096: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_1097: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 2);  unsqueeze_1096 = None
    unsqueeze_1098: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 3);  unsqueeze_1097 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1108: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_1109: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 2);  unsqueeze_1108 = None
    unsqueeze_1110: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 3);  unsqueeze_1109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1120: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_1121: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 2);  unsqueeze_1120 = None
    unsqueeze_1122: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 3);  unsqueeze_1121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_63: "b8[8, 192, 17, 17]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1132: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_1133: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 2);  unsqueeze_1132 = None
    unsqueeze_1134: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 3);  unsqueeze_1133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_64: "b8[8, 96, 17, 17]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1144: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_1145: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 2);  unsqueeze_1144 = None
    unsqueeze_1146: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 3);  unsqueeze_1145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1156: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_1157: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 2);  unsqueeze_1156 = None
    unsqueeze_1158: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 3);  unsqueeze_1157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1168: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_1169: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 2);  unsqueeze_1168 = None
    unsqueeze_1170: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 3);  unsqueeze_1169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_67: "b8[8, 384, 17, 17]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1180: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_1181: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 2);  unsqueeze_1180 = None
    unsqueeze_1182: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 3);  unsqueeze_1181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_68: "b8[8, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1192: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_1193: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, 2);  unsqueeze_1192 = None
    unsqueeze_1194: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 3);  unsqueeze_1193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_69: "b8[8, 96, 35, 35]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1204: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_1205: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 2);  unsqueeze_1204 = None
    unsqueeze_1206: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 3);  unsqueeze_1205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1216: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_1217: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 2);  unsqueeze_1216 = None
    unsqueeze_1218: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 3);  unsqueeze_1217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1228: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_1229: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 2);  unsqueeze_1228 = None
    unsqueeze_1230: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 3);  unsqueeze_1229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_72: "b8[8, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1240: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_1241: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, 2);  unsqueeze_1240 = None
    unsqueeze_1242: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 3);  unsqueeze_1241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1252: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_1253: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, 2);  unsqueeze_1252 = None
    unsqueeze_1254: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 3);  unsqueeze_1253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_74: "b8[8, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1264: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_1265: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, 2);  unsqueeze_1264 = None
    unsqueeze_1266: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 3);  unsqueeze_1265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_75: "b8[8, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1276: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_1277: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, 2);  unsqueeze_1276 = None
    unsqueeze_1278: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 3);  unsqueeze_1277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_76: "b8[8, 96, 35, 35]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1288: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_1289: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, 2);  unsqueeze_1288 = None
    unsqueeze_1290: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1289, 3);  unsqueeze_1289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1300: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_1301: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, 2);  unsqueeze_1300 = None
    unsqueeze_1302: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1301, 3);  unsqueeze_1301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1312: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_1313: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, 2);  unsqueeze_1312 = None
    unsqueeze_1314: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1313, 3);  unsqueeze_1313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_79: "b8[8, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1324: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_1325: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, 2);  unsqueeze_1324 = None
    unsqueeze_1326: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1325, 3);  unsqueeze_1325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1336: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_1337: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, 2);  unsqueeze_1336 = None
    unsqueeze_1338: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1337, 3);  unsqueeze_1337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_81: "b8[8, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1348: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_1349: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, 2);  unsqueeze_1348 = None
    unsqueeze_1350: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1349, 3);  unsqueeze_1349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_82: "b8[8, 32, 35, 35]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1360: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_1361: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, 2);  unsqueeze_1360 = None
    unsqueeze_1362: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1361, 3);  unsqueeze_1361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_83: "b8[8, 96, 35, 35]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1372: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_1373: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, 2);  unsqueeze_1372 = None
    unsqueeze_1374: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1373, 3);  unsqueeze_1373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1384: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_1385: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, 2);  unsqueeze_1384 = None
    unsqueeze_1386: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1385, 3);  unsqueeze_1385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1396: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_1397: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1396, 2);  unsqueeze_1396 = None
    unsqueeze_1398: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1397, 3);  unsqueeze_1397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_86: "b8[8, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1408: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_1409: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1408, 2);  unsqueeze_1408 = None
    unsqueeze_1410: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1409, 3);  unsqueeze_1409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1420: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_1421: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1420, 2);  unsqueeze_1420 = None
    unsqueeze_1422: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1421, 3);  unsqueeze_1421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_88: "b8[8, 64, 35, 35]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1432: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_1433: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, 2);  unsqueeze_1432 = None
    unsqueeze_1434: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1433, 3);  unsqueeze_1433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1444: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_1445: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, 2);  unsqueeze_1444 = None
    unsqueeze_1446: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1445, 3);  unsqueeze_1445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1456: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_1457: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, 2);  unsqueeze_1456 = None
    unsqueeze_1458: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1457, 3);  unsqueeze_1457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1468: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_1469: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, 2);  unsqueeze_1468 = None
    unsqueeze_1470: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1469, 3);  unsqueeze_1469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1480: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_1481: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1480, 2);  unsqueeze_1480 = None
    unsqueeze_1482: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1481, 3);  unsqueeze_1481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1492: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_1493: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1492, 2);  unsqueeze_1492 = None
    unsqueeze_1494: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1493, 3);  unsqueeze_1493 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_285, add);  primals_285 = add = None
    copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_286, add_2);  primals_286 = add_2 = None
    copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_287, add_3);  primals_287 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_288, add_5);  primals_288 = add_5 = None
    copy__4: "f32[32]" = torch.ops.aten.copy_.default(primals_289, add_7);  primals_289 = add_7 = None
    copy__5: "f32[32]" = torch.ops.aten.copy_.default(primals_290, add_8);  primals_290 = add_8 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_10);  primals_291 = add_10 = None
    copy__7: "f32[64]" = torch.ops.aten.copy_.default(primals_292, add_12);  primals_292 = add_12 = None
    copy__8: "f32[64]" = torch.ops.aten.copy_.default(primals_293, add_13);  primals_293 = add_13 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_294, add_15);  primals_294 = add_15 = None
    copy__10: "f32[80]" = torch.ops.aten.copy_.default(primals_295, add_17);  primals_295 = add_17 = None
    copy__11: "f32[80]" = torch.ops.aten.copy_.default(primals_296, add_18);  primals_296 = add_18 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_20);  primals_297 = add_20 = None
    copy__13: "f32[192]" = torch.ops.aten.copy_.default(primals_298, add_22);  primals_298 = add_22 = None
    copy__14: "f32[192]" = torch.ops.aten.copy_.default(primals_299, add_23);  primals_299 = add_23 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_300, add_25);  primals_300 = add_25 = None
    copy__16: "f32[64]" = torch.ops.aten.copy_.default(primals_301, add_27);  primals_301 = add_27 = None
    copy__17: "f32[64]" = torch.ops.aten.copy_.default(primals_302, add_28);  primals_302 = add_28 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_30);  primals_303 = add_30 = None
    copy__19: "f32[48]" = torch.ops.aten.copy_.default(primals_304, add_32);  primals_304 = add_32 = None
    copy__20: "f32[48]" = torch.ops.aten.copy_.default(primals_305, add_33);  primals_305 = add_33 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_306, add_35);  primals_306 = add_35 = None
    copy__22: "f32[64]" = torch.ops.aten.copy_.default(primals_307, add_37);  primals_307 = add_37 = None
    copy__23: "f32[64]" = torch.ops.aten.copy_.default(primals_308, add_38);  primals_308 = add_38 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_40);  primals_309 = add_40 = None
    copy__25: "f32[64]" = torch.ops.aten.copy_.default(primals_310, add_42);  primals_310 = add_42 = None
    copy__26: "f32[64]" = torch.ops.aten.copy_.default(primals_311, add_43);  primals_311 = add_43 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_312, add_45);  primals_312 = add_45 = None
    copy__28: "f32[96]" = torch.ops.aten.copy_.default(primals_313, add_47);  primals_313 = add_47 = None
    copy__29: "f32[96]" = torch.ops.aten.copy_.default(primals_314, add_48);  primals_314 = add_48 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_315, add_50);  primals_315 = add_50 = None
    copy__31: "f32[96]" = torch.ops.aten.copy_.default(primals_316, add_52);  primals_316 = add_52 = None
    copy__32: "f32[96]" = torch.ops.aten.copy_.default(primals_317, add_53);  primals_317 = add_53 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_318, add_55);  primals_318 = add_55 = None
    copy__34: "f32[32]" = torch.ops.aten.copy_.default(primals_319, add_57);  primals_319 = add_57 = None
    copy__35: "f32[32]" = torch.ops.aten.copy_.default(primals_320, add_58);  primals_320 = add_58 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_321, add_60);  primals_321 = add_60 = None
    copy__37: "f32[64]" = torch.ops.aten.copy_.default(primals_322, add_62);  primals_322 = add_62 = None
    copy__38: "f32[64]" = torch.ops.aten.copy_.default(primals_323, add_63);  primals_323 = add_63 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_324, add_65);  primals_324 = add_65 = None
    copy__40: "f32[48]" = torch.ops.aten.copy_.default(primals_325, add_67);  primals_325 = add_67 = None
    copy__41: "f32[48]" = torch.ops.aten.copy_.default(primals_326, add_68);  primals_326 = add_68 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_327, add_70);  primals_327 = add_70 = None
    copy__43: "f32[64]" = torch.ops.aten.copy_.default(primals_328, add_72);  primals_328 = add_72 = None
    copy__44: "f32[64]" = torch.ops.aten.copy_.default(primals_329, add_73);  primals_329 = add_73 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_330, add_75);  primals_330 = add_75 = None
    copy__46: "f32[64]" = torch.ops.aten.copy_.default(primals_331, add_77);  primals_331 = add_77 = None
    copy__47: "f32[64]" = torch.ops.aten.copy_.default(primals_332, add_78);  primals_332 = add_78 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_333, add_80);  primals_333 = add_80 = None
    copy__49: "f32[96]" = torch.ops.aten.copy_.default(primals_334, add_82);  primals_334 = add_82 = None
    copy__50: "f32[96]" = torch.ops.aten.copy_.default(primals_335, add_83);  primals_335 = add_83 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_336, add_85);  primals_336 = add_85 = None
    copy__52: "f32[96]" = torch.ops.aten.copy_.default(primals_337, add_87);  primals_337 = add_87 = None
    copy__53: "f32[96]" = torch.ops.aten.copy_.default(primals_338, add_88);  primals_338 = add_88 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_339, add_90);  primals_339 = add_90 = None
    copy__55: "f32[64]" = torch.ops.aten.copy_.default(primals_340, add_92);  primals_340 = add_92 = None
    copy__56: "f32[64]" = torch.ops.aten.copy_.default(primals_341, add_93);  primals_341 = add_93 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_342, add_95);  primals_342 = add_95 = None
    copy__58: "f32[64]" = torch.ops.aten.copy_.default(primals_343, add_97);  primals_343 = add_97 = None
    copy__59: "f32[64]" = torch.ops.aten.copy_.default(primals_344, add_98);  primals_344 = add_98 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_345, add_100);  primals_345 = add_100 = None
    copy__61: "f32[48]" = torch.ops.aten.copy_.default(primals_346, add_102);  primals_346 = add_102 = None
    copy__62: "f32[48]" = torch.ops.aten.copy_.default(primals_347, add_103);  primals_347 = add_103 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_348, add_105);  primals_348 = add_105 = None
    copy__64: "f32[64]" = torch.ops.aten.copy_.default(primals_349, add_107);  primals_349 = add_107 = None
    copy__65: "f32[64]" = torch.ops.aten.copy_.default(primals_350, add_108);  primals_350 = add_108 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_351, add_110);  primals_351 = add_110 = None
    copy__67: "f32[64]" = torch.ops.aten.copy_.default(primals_352, add_112);  primals_352 = add_112 = None
    copy__68: "f32[64]" = torch.ops.aten.copy_.default(primals_353, add_113);  primals_353 = add_113 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_354, add_115);  primals_354 = add_115 = None
    copy__70: "f32[96]" = torch.ops.aten.copy_.default(primals_355, add_117);  primals_355 = add_117 = None
    copy__71: "f32[96]" = torch.ops.aten.copy_.default(primals_356, add_118);  primals_356 = add_118 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_357, add_120);  primals_357 = add_120 = None
    copy__73: "f32[96]" = torch.ops.aten.copy_.default(primals_358, add_122);  primals_358 = add_122 = None
    copy__74: "f32[96]" = torch.ops.aten.copy_.default(primals_359, add_123);  primals_359 = add_123 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_360, add_125);  primals_360 = add_125 = None
    copy__76: "f32[64]" = torch.ops.aten.copy_.default(primals_361, add_127);  primals_361 = add_127 = None
    copy__77: "f32[64]" = torch.ops.aten.copy_.default(primals_362, add_128);  primals_362 = add_128 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_363, add_130);  primals_363 = add_130 = None
    copy__79: "f32[384]" = torch.ops.aten.copy_.default(primals_364, add_132);  primals_364 = add_132 = None
    copy__80: "f32[384]" = torch.ops.aten.copy_.default(primals_365, add_133);  primals_365 = add_133 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_366, add_135);  primals_366 = add_135 = None
    copy__82: "f32[64]" = torch.ops.aten.copy_.default(primals_367, add_137);  primals_367 = add_137 = None
    copy__83: "f32[64]" = torch.ops.aten.copy_.default(primals_368, add_138);  primals_368 = add_138 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_369, add_140);  primals_369 = add_140 = None
    copy__85: "f32[96]" = torch.ops.aten.copy_.default(primals_370, add_142);  primals_370 = add_142 = None
    copy__86: "f32[96]" = torch.ops.aten.copy_.default(primals_371, add_143);  primals_371 = add_143 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_372, add_145);  primals_372 = add_145 = None
    copy__88: "f32[96]" = torch.ops.aten.copy_.default(primals_373, add_147);  primals_373 = add_147 = None
    copy__89: "f32[96]" = torch.ops.aten.copy_.default(primals_374, add_148);  primals_374 = add_148 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_375, add_150);  primals_375 = add_150 = None
    copy__91: "f32[192]" = torch.ops.aten.copy_.default(primals_376, add_152);  primals_376 = add_152 = None
    copy__92: "f32[192]" = torch.ops.aten.copy_.default(primals_377, add_153);  primals_377 = add_153 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_378, add_155);  primals_378 = add_155 = None
    copy__94: "f32[128]" = torch.ops.aten.copy_.default(primals_379, add_157);  primals_379 = add_157 = None
    copy__95: "f32[128]" = torch.ops.aten.copy_.default(primals_380, add_158);  primals_380 = add_158 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_381, add_160);  primals_381 = add_160 = None
    copy__97: "f32[128]" = torch.ops.aten.copy_.default(primals_382, add_162);  primals_382 = add_162 = None
    copy__98: "f32[128]" = torch.ops.aten.copy_.default(primals_383, add_163);  primals_383 = add_163 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_384, add_165);  primals_384 = add_165 = None
    copy__100: "f32[192]" = torch.ops.aten.copy_.default(primals_385, add_167);  primals_385 = add_167 = None
    copy__101: "f32[192]" = torch.ops.aten.copy_.default(primals_386, add_168);  primals_386 = add_168 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_387, add_170);  primals_387 = add_170 = None
    copy__103: "f32[128]" = torch.ops.aten.copy_.default(primals_388, add_172);  primals_388 = add_172 = None
    copy__104: "f32[128]" = torch.ops.aten.copy_.default(primals_389, add_173);  primals_389 = add_173 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_390, add_175);  primals_390 = add_175 = None
    copy__106: "f32[128]" = torch.ops.aten.copy_.default(primals_391, add_177);  primals_391 = add_177 = None
    copy__107: "f32[128]" = torch.ops.aten.copy_.default(primals_392, add_178);  primals_392 = add_178 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_393, add_180);  primals_393 = add_180 = None
    copy__109: "f32[128]" = torch.ops.aten.copy_.default(primals_394, add_182);  primals_394 = add_182 = None
    copy__110: "f32[128]" = torch.ops.aten.copy_.default(primals_395, add_183);  primals_395 = add_183 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_396, add_185);  primals_396 = add_185 = None
    copy__112: "f32[128]" = torch.ops.aten.copy_.default(primals_397, add_187);  primals_397 = add_187 = None
    copy__113: "f32[128]" = torch.ops.aten.copy_.default(primals_398, add_188);  primals_398 = add_188 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_399, add_190);  primals_399 = add_190 = None
    copy__115: "f32[192]" = torch.ops.aten.copy_.default(primals_400, add_192);  primals_400 = add_192 = None
    copy__116: "f32[192]" = torch.ops.aten.copy_.default(primals_401, add_193);  primals_401 = add_193 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_402, add_195);  primals_402 = add_195 = None
    copy__118: "f32[192]" = torch.ops.aten.copy_.default(primals_403, add_197);  primals_403 = add_197 = None
    copy__119: "f32[192]" = torch.ops.aten.copy_.default(primals_404, add_198);  primals_404 = add_198 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_405, add_200);  primals_405 = add_200 = None
    copy__121: "f32[192]" = torch.ops.aten.copy_.default(primals_406, add_202);  primals_406 = add_202 = None
    copy__122: "f32[192]" = torch.ops.aten.copy_.default(primals_407, add_203);  primals_407 = add_203 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_408, add_205);  primals_408 = add_205 = None
    copy__124: "f32[160]" = torch.ops.aten.copy_.default(primals_409, add_207);  primals_409 = add_207 = None
    copy__125: "f32[160]" = torch.ops.aten.copy_.default(primals_410, add_208);  primals_410 = add_208 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_411, add_210);  primals_411 = add_210 = None
    copy__127: "f32[160]" = torch.ops.aten.copy_.default(primals_412, add_212);  primals_412 = add_212 = None
    copy__128: "f32[160]" = torch.ops.aten.copy_.default(primals_413, add_213);  primals_413 = add_213 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_414, add_215);  primals_414 = add_215 = None
    copy__130: "f32[192]" = torch.ops.aten.copy_.default(primals_415, add_217);  primals_415 = add_217 = None
    copy__131: "f32[192]" = torch.ops.aten.copy_.default(primals_416, add_218);  primals_416 = add_218 = None
    copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_417, add_220);  primals_417 = add_220 = None
    copy__133: "f32[160]" = torch.ops.aten.copy_.default(primals_418, add_222);  primals_418 = add_222 = None
    copy__134: "f32[160]" = torch.ops.aten.copy_.default(primals_419, add_223);  primals_419 = add_223 = None
    copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_420, add_225);  primals_420 = add_225 = None
    copy__136: "f32[160]" = torch.ops.aten.copy_.default(primals_421, add_227);  primals_421 = add_227 = None
    copy__137: "f32[160]" = torch.ops.aten.copy_.default(primals_422, add_228);  primals_422 = add_228 = None
    copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_423, add_230);  primals_423 = add_230 = None
    copy__139: "f32[160]" = torch.ops.aten.copy_.default(primals_424, add_232);  primals_424 = add_232 = None
    copy__140: "f32[160]" = torch.ops.aten.copy_.default(primals_425, add_233);  primals_425 = add_233 = None
    copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_426, add_235);  primals_426 = add_235 = None
    copy__142: "f32[160]" = torch.ops.aten.copy_.default(primals_427, add_237);  primals_427 = add_237 = None
    copy__143: "f32[160]" = torch.ops.aten.copy_.default(primals_428, add_238);  primals_428 = add_238 = None
    copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_429, add_240);  primals_429 = add_240 = None
    copy__145: "f32[192]" = torch.ops.aten.copy_.default(primals_430, add_242);  primals_430 = add_242 = None
    copy__146: "f32[192]" = torch.ops.aten.copy_.default(primals_431, add_243);  primals_431 = add_243 = None
    copy__147: "i64[]" = torch.ops.aten.copy_.default(primals_432, add_245);  primals_432 = add_245 = None
    copy__148: "f32[192]" = torch.ops.aten.copy_.default(primals_433, add_247);  primals_433 = add_247 = None
    copy__149: "f32[192]" = torch.ops.aten.copy_.default(primals_434, add_248);  primals_434 = add_248 = None
    copy__150: "i64[]" = torch.ops.aten.copy_.default(primals_435, add_250);  primals_435 = add_250 = None
    copy__151: "f32[192]" = torch.ops.aten.copy_.default(primals_436, add_252);  primals_436 = add_252 = None
    copy__152: "f32[192]" = torch.ops.aten.copy_.default(primals_437, add_253);  primals_437 = add_253 = None
    copy__153: "i64[]" = torch.ops.aten.copy_.default(primals_438, add_255);  primals_438 = add_255 = None
    copy__154: "f32[160]" = torch.ops.aten.copy_.default(primals_439, add_257);  primals_439 = add_257 = None
    copy__155: "f32[160]" = torch.ops.aten.copy_.default(primals_440, add_258);  primals_440 = add_258 = None
    copy__156: "i64[]" = torch.ops.aten.copy_.default(primals_441, add_260);  primals_441 = add_260 = None
    copy__157: "f32[160]" = torch.ops.aten.copy_.default(primals_442, add_262);  primals_442 = add_262 = None
    copy__158: "f32[160]" = torch.ops.aten.copy_.default(primals_443, add_263);  primals_443 = add_263 = None
    copy__159: "i64[]" = torch.ops.aten.copy_.default(primals_444, add_265);  primals_444 = add_265 = None
    copy__160: "f32[192]" = torch.ops.aten.copy_.default(primals_445, add_267);  primals_445 = add_267 = None
    copy__161: "f32[192]" = torch.ops.aten.copy_.default(primals_446, add_268);  primals_446 = add_268 = None
    copy__162: "i64[]" = torch.ops.aten.copy_.default(primals_447, add_270);  primals_447 = add_270 = None
    copy__163: "f32[160]" = torch.ops.aten.copy_.default(primals_448, add_272);  primals_448 = add_272 = None
    copy__164: "f32[160]" = torch.ops.aten.copy_.default(primals_449, add_273);  primals_449 = add_273 = None
    copy__165: "i64[]" = torch.ops.aten.copy_.default(primals_450, add_275);  primals_450 = add_275 = None
    copy__166: "f32[160]" = torch.ops.aten.copy_.default(primals_451, add_277);  primals_451 = add_277 = None
    copy__167: "f32[160]" = torch.ops.aten.copy_.default(primals_452, add_278);  primals_452 = add_278 = None
    copy__168: "i64[]" = torch.ops.aten.copy_.default(primals_453, add_280);  primals_453 = add_280 = None
    copy__169: "f32[160]" = torch.ops.aten.copy_.default(primals_454, add_282);  primals_454 = add_282 = None
    copy__170: "f32[160]" = torch.ops.aten.copy_.default(primals_455, add_283);  primals_455 = add_283 = None
    copy__171: "i64[]" = torch.ops.aten.copy_.default(primals_456, add_285);  primals_456 = add_285 = None
    copy__172: "f32[160]" = torch.ops.aten.copy_.default(primals_457, add_287);  primals_457 = add_287 = None
    copy__173: "f32[160]" = torch.ops.aten.copy_.default(primals_458, add_288);  primals_458 = add_288 = None
    copy__174: "i64[]" = torch.ops.aten.copy_.default(primals_459, add_290);  primals_459 = add_290 = None
    copy__175: "f32[192]" = torch.ops.aten.copy_.default(primals_460, add_292);  primals_460 = add_292 = None
    copy__176: "f32[192]" = torch.ops.aten.copy_.default(primals_461, add_293);  primals_461 = add_293 = None
    copy__177: "i64[]" = torch.ops.aten.copy_.default(primals_462, add_295);  primals_462 = add_295 = None
    copy__178: "f32[192]" = torch.ops.aten.copy_.default(primals_463, add_297);  primals_463 = add_297 = None
    copy__179: "f32[192]" = torch.ops.aten.copy_.default(primals_464, add_298);  primals_464 = add_298 = None
    copy__180: "i64[]" = torch.ops.aten.copy_.default(primals_465, add_300);  primals_465 = add_300 = None
    copy__181: "f32[192]" = torch.ops.aten.copy_.default(primals_466, add_302);  primals_466 = add_302 = None
    copy__182: "f32[192]" = torch.ops.aten.copy_.default(primals_467, add_303);  primals_467 = add_303 = None
    copy__183: "i64[]" = torch.ops.aten.copy_.default(primals_468, add_305);  primals_468 = add_305 = None
    copy__184: "f32[192]" = torch.ops.aten.copy_.default(primals_469, add_307);  primals_469 = add_307 = None
    copy__185: "f32[192]" = torch.ops.aten.copy_.default(primals_470, add_308);  primals_470 = add_308 = None
    copy__186: "i64[]" = torch.ops.aten.copy_.default(primals_471, add_310);  primals_471 = add_310 = None
    copy__187: "f32[192]" = torch.ops.aten.copy_.default(primals_472, add_312);  primals_472 = add_312 = None
    copy__188: "f32[192]" = torch.ops.aten.copy_.default(primals_473, add_313);  primals_473 = add_313 = None
    copy__189: "i64[]" = torch.ops.aten.copy_.default(primals_474, add_315);  primals_474 = add_315 = None
    copy__190: "f32[192]" = torch.ops.aten.copy_.default(primals_475, add_317);  primals_475 = add_317 = None
    copy__191: "f32[192]" = torch.ops.aten.copy_.default(primals_476, add_318);  primals_476 = add_318 = None
    copy__192: "i64[]" = torch.ops.aten.copy_.default(primals_477, add_320);  primals_477 = add_320 = None
    copy__193: "f32[192]" = torch.ops.aten.copy_.default(primals_478, add_322);  primals_478 = add_322 = None
    copy__194: "f32[192]" = torch.ops.aten.copy_.default(primals_479, add_323);  primals_479 = add_323 = None
    copy__195: "i64[]" = torch.ops.aten.copy_.default(primals_480, add_325);  primals_480 = add_325 = None
    copy__196: "f32[192]" = torch.ops.aten.copy_.default(primals_481, add_327);  primals_481 = add_327 = None
    copy__197: "f32[192]" = torch.ops.aten.copy_.default(primals_482, add_328);  primals_482 = add_328 = None
    copy__198: "i64[]" = torch.ops.aten.copy_.default(primals_483, add_330);  primals_483 = add_330 = None
    copy__199: "f32[192]" = torch.ops.aten.copy_.default(primals_484, add_332);  primals_484 = add_332 = None
    copy__200: "f32[192]" = torch.ops.aten.copy_.default(primals_485, add_333);  primals_485 = add_333 = None
    copy__201: "i64[]" = torch.ops.aten.copy_.default(primals_486, add_335);  primals_486 = add_335 = None
    copy__202: "f32[192]" = torch.ops.aten.copy_.default(primals_487, add_337);  primals_487 = add_337 = None
    copy__203: "f32[192]" = torch.ops.aten.copy_.default(primals_488, add_338);  primals_488 = add_338 = None
    copy__204: "i64[]" = torch.ops.aten.copy_.default(primals_489, add_340);  primals_489 = add_340 = None
    copy__205: "f32[192]" = torch.ops.aten.copy_.default(primals_490, add_342);  primals_490 = add_342 = None
    copy__206: "f32[192]" = torch.ops.aten.copy_.default(primals_491, add_343);  primals_491 = add_343 = None
    copy__207: "i64[]" = torch.ops.aten.copy_.default(primals_492, add_345);  primals_492 = add_345 = None
    copy__208: "f32[192]" = torch.ops.aten.copy_.default(primals_493, add_347);  primals_493 = add_347 = None
    copy__209: "f32[192]" = torch.ops.aten.copy_.default(primals_494, add_348);  primals_494 = add_348 = None
    copy__210: "i64[]" = torch.ops.aten.copy_.default(primals_495, add_350);  primals_495 = add_350 = None
    copy__211: "f32[192]" = torch.ops.aten.copy_.default(primals_496, add_352);  primals_496 = add_352 = None
    copy__212: "f32[192]" = torch.ops.aten.copy_.default(primals_497, add_353);  primals_497 = add_353 = None
    copy__213: "i64[]" = torch.ops.aten.copy_.default(primals_498, add_355);  primals_498 = add_355 = None
    copy__214: "f32[320]" = torch.ops.aten.copy_.default(primals_499, add_357);  primals_499 = add_357 = None
    copy__215: "f32[320]" = torch.ops.aten.copy_.default(primals_500, add_358);  primals_500 = add_358 = None
    copy__216: "i64[]" = torch.ops.aten.copy_.default(primals_501, add_360);  primals_501 = add_360 = None
    copy__217: "f32[192]" = torch.ops.aten.copy_.default(primals_502, add_362);  primals_502 = add_362 = None
    copy__218: "f32[192]" = torch.ops.aten.copy_.default(primals_503, add_363);  primals_503 = add_363 = None
    copy__219: "i64[]" = torch.ops.aten.copy_.default(primals_504, add_365);  primals_504 = add_365 = None
    copy__220: "f32[192]" = torch.ops.aten.copy_.default(primals_505, add_367);  primals_505 = add_367 = None
    copy__221: "f32[192]" = torch.ops.aten.copy_.default(primals_506, add_368);  primals_506 = add_368 = None
    copy__222: "i64[]" = torch.ops.aten.copy_.default(primals_507, add_370);  primals_507 = add_370 = None
    copy__223: "f32[192]" = torch.ops.aten.copy_.default(primals_508, add_372);  primals_508 = add_372 = None
    copy__224: "f32[192]" = torch.ops.aten.copy_.default(primals_509, add_373);  primals_509 = add_373 = None
    copy__225: "i64[]" = torch.ops.aten.copy_.default(primals_510, add_375);  primals_510 = add_375 = None
    copy__226: "f32[192]" = torch.ops.aten.copy_.default(primals_511, add_377);  primals_511 = add_377 = None
    copy__227: "f32[192]" = torch.ops.aten.copy_.default(primals_512, add_378);  primals_512 = add_378 = None
    copy__228: "i64[]" = torch.ops.aten.copy_.default(primals_513, add_380);  primals_513 = add_380 = None
    copy__229: "f32[320]" = torch.ops.aten.copy_.default(primals_514, add_382);  primals_514 = add_382 = None
    copy__230: "f32[320]" = torch.ops.aten.copy_.default(primals_515, add_383);  primals_515 = add_383 = None
    copy__231: "i64[]" = torch.ops.aten.copy_.default(primals_516, add_385);  primals_516 = add_385 = None
    copy__232: "f32[384]" = torch.ops.aten.copy_.default(primals_517, add_387);  primals_517 = add_387 = None
    copy__233: "f32[384]" = torch.ops.aten.copy_.default(primals_518, add_388);  primals_518 = add_388 = None
    copy__234: "i64[]" = torch.ops.aten.copy_.default(primals_519, add_390);  primals_519 = add_390 = None
    copy__235: "f32[384]" = torch.ops.aten.copy_.default(primals_520, add_392);  primals_520 = add_392 = None
    copy__236: "f32[384]" = torch.ops.aten.copy_.default(primals_521, add_393);  primals_521 = add_393 = None
    copy__237: "i64[]" = torch.ops.aten.copy_.default(primals_522, add_395);  primals_522 = add_395 = None
    copy__238: "f32[384]" = torch.ops.aten.copy_.default(primals_523, add_397);  primals_523 = add_397 = None
    copy__239: "f32[384]" = torch.ops.aten.copy_.default(primals_524, add_398);  primals_524 = add_398 = None
    copy__240: "i64[]" = torch.ops.aten.copy_.default(primals_525, add_400);  primals_525 = add_400 = None
    copy__241: "f32[448]" = torch.ops.aten.copy_.default(primals_526, add_402);  primals_526 = add_402 = None
    copy__242: "f32[448]" = torch.ops.aten.copy_.default(primals_527, add_403);  primals_527 = add_403 = None
    copy__243: "i64[]" = torch.ops.aten.copy_.default(primals_528, add_405);  primals_528 = add_405 = None
    copy__244: "f32[384]" = torch.ops.aten.copy_.default(primals_529, add_407);  primals_529 = add_407 = None
    copy__245: "f32[384]" = torch.ops.aten.copy_.default(primals_530, add_408);  primals_530 = add_408 = None
    copy__246: "i64[]" = torch.ops.aten.copy_.default(primals_531, add_410);  primals_531 = add_410 = None
    copy__247: "f32[384]" = torch.ops.aten.copy_.default(primals_532, add_412);  primals_532 = add_412 = None
    copy__248: "f32[384]" = torch.ops.aten.copy_.default(primals_533, add_413);  primals_533 = add_413 = None
    copy__249: "i64[]" = torch.ops.aten.copy_.default(primals_534, add_415);  primals_534 = add_415 = None
    copy__250: "f32[384]" = torch.ops.aten.copy_.default(primals_535, add_417);  primals_535 = add_417 = None
    copy__251: "f32[384]" = torch.ops.aten.copy_.default(primals_536, add_418);  primals_536 = add_418 = None
    copy__252: "i64[]" = torch.ops.aten.copy_.default(primals_537, add_420);  primals_537 = add_420 = None
    copy__253: "f32[192]" = torch.ops.aten.copy_.default(primals_538, add_422);  primals_538 = add_422 = None
    copy__254: "f32[192]" = torch.ops.aten.copy_.default(primals_539, add_423);  primals_539 = add_423 = None
    copy__255: "i64[]" = torch.ops.aten.copy_.default(primals_540, add_425);  primals_540 = add_425 = None
    copy__256: "f32[320]" = torch.ops.aten.copy_.default(primals_541, add_427);  primals_541 = add_427 = None
    copy__257: "f32[320]" = torch.ops.aten.copy_.default(primals_542, add_428);  primals_542 = add_428 = None
    copy__258: "i64[]" = torch.ops.aten.copy_.default(primals_543, add_430);  primals_543 = add_430 = None
    copy__259: "f32[384]" = torch.ops.aten.copy_.default(primals_544, add_432);  primals_544 = add_432 = None
    copy__260: "f32[384]" = torch.ops.aten.copy_.default(primals_545, add_433);  primals_545 = add_433 = None
    copy__261: "i64[]" = torch.ops.aten.copy_.default(primals_546, add_435);  primals_546 = add_435 = None
    copy__262: "f32[384]" = torch.ops.aten.copy_.default(primals_547, add_437);  primals_547 = add_437 = None
    copy__263: "f32[384]" = torch.ops.aten.copy_.default(primals_548, add_438);  primals_548 = add_438 = None
    copy__264: "i64[]" = torch.ops.aten.copy_.default(primals_549, add_440);  primals_549 = add_440 = None
    copy__265: "f32[384]" = torch.ops.aten.copy_.default(primals_550, add_442);  primals_550 = add_442 = None
    copy__266: "f32[384]" = torch.ops.aten.copy_.default(primals_551, add_443);  primals_551 = add_443 = None
    copy__267: "i64[]" = torch.ops.aten.copy_.default(primals_552, add_445);  primals_552 = add_445 = None
    copy__268: "f32[448]" = torch.ops.aten.copy_.default(primals_553, add_447);  primals_553 = add_447 = None
    copy__269: "f32[448]" = torch.ops.aten.copy_.default(primals_554, add_448);  primals_554 = add_448 = None
    copy__270: "i64[]" = torch.ops.aten.copy_.default(primals_555, add_450);  primals_555 = add_450 = None
    copy__271: "f32[384]" = torch.ops.aten.copy_.default(primals_556, add_452);  primals_556 = add_452 = None
    copy__272: "f32[384]" = torch.ops.aten.copy_.default(primals_557, add_453);  primals_557 = add_453 = None
    copy__273: "i64[]" = torch.ops.aten.copy_.default(primals_558, add_455);  primals_558 = add_455 = None
    copy__274: "f32[384]" = torch.ops.aten.copy_.default(primals_559, add_457);  primals_559 = add_457 = None
    copy__275: "f32[384]" = torch.ops.aten.copy_.default(primals_560, add_458);  primals_560 = add_458 = None
    copy__276: "i64[]" = torch.ops.aten.copy_.default(primals_561, add_460);  primals_561 = add_460 = None
    copy__277: "f32[384]" = torch.ops.aten.copy_.default(primals_562, add_462);  primals_562 = add_462 = None
    copy__278: "f32[384]" = torch.ops.aten.copy_.default(primals_563, add_463);  primals_563 = add_463 = None
    copy__279: "i64[]" = torch.ops.aten.copy_.default(primals_564, add_465);  primals_564 = add_465 = None
    copy__280: "f32[192]" = torch.ops.aten.copy_.default(primals_565, add_467);  primals_565 = add_467 = None
    copy__281: "f32[192]" = torch.ops.aten.copy_.default(primals_566, add_468);  primals_566 = add_468 = None
    return [addmm, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_567, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, getitem_12, getitem_13, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_6, convolution_7, squeeze_22, convolution_8, squeeze_25, relu_8, convolution_9, squeeze_28, relu_9, convolution_10, squeeze_31, avg_pool2d, convolution_11, squeeze_34, cat, convolution_12, squeeze_37, convolution_13, squeeze_40, relu_13, convolution_14, squeeze_43, convolution_15, squeeze_46, relu_15, convolution_16, squeeze_49, relu_16, convolution_17, squeeze_52, avg_pool2d_1, convolution_18, squeeze_55, cat_1, convolution_19, squeeze_58, convolution_20, squeeze_61, relu_20, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_22, convolution_23, squeeze_70, relu_23, convolution_24, squeeze_73, avg_pool2d_2, convolution_25, squeeze_76, cat_2, convolution_26, squeeze_79, convolution_27, squeeze_82, relu_27, convolution_28, squeeze_85, relu_28, convolution_29, squeeze_88, getitem_65, cat_3, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_31, convolution_32, squeeze_97, relu_32, convolution_33, squeeze_100, convolution_34, squeeze_103, relu_34, convolution_35, squeeze_106, relu_35, convolution_36, squeeze_109, relu_36, convolution_37, squeeze_112, relu_37, convolution_38, squeeze_115, avg_pool2d_3, convolution_39, squeeze_118, cat_4, convolution_40, squeeze_121, convolution_41, squeeze_124, relu_41, convolution_42, squeeze_127, relu_42, convolution_43, squeeze_130, convolution_44, squeeze_133, relu_44, convolution_45, squeeze_136, relu_45, convolution_46, squeeze_139, relu_46, convolution_47, squeeze_142, relu_47, convolution_48, squeeze_145, avg_pool2d_4, convolution_49, squeeze_148, cat_5, convolution_50, squeeze_151, convolution_51, squeeze_154, relu_51, convolution_52, squeeze_157, relu_52, convolution_53, squeeze_160, convolution_54, squeeze_163, relu_54, convolution_55, squeeze_166, relu_55, convolution_56, squeeze_169, relu_56, convolution_57, squeeze_172, relu_57, convolution_58, squeeze_175, avg_pool2d_5, convolution_59, squeeze_178, cat_6, convolution_60, squeeze_181, convolution_61, squeeze_184, relu_61, convolution_62, squeeze_187, relu_62, convolution_63, squeeze_190, convolution_64, squeeze_193, relu_64, convolution_65, squeeze_196, relu_65, convolution_66, squeeze_199, relu_66, convolution_67, squeeze_202, relu_67, convolution_68, squeeze_205, avg_pool2d_6, convolution_69, squeeze_208, cat_7, convolution_70, squeeze_211, relu_70, convolution_71, squeeze_214, convolution_72, squeeze_217, relu_72, convolution_73, squeeze_220, relu_73, convolution_74, squeeze_223, relu_74, convolution_75, squeeze_226, getitem_159, cat_8, convolution_76, squeeze_229, convolution_77, squeeze_232, relu_77, convolution_78, squeeze_235, convolution_79, squeeze_238, convolution_80, squeeze_241, relu_80, convolution_81, squeeze_244, relu_81, convolution_82, squeeze_247, convolution_83, squeeze_250, avg_pool2d_7, convolution_84, squeeze_253, cat_11, convolution_85, squeeze_256, convolution_86, squeeze_259, relu_86, convolution_87, squeeze_262, convolution_88, squeeze_265, convolution_89, squeeze_268, relu_89, convolution_90, squeeze_271, relu_90, convolution_91, squeeze_274, convolution_92, squeeze_277, avg_pool2d_8, convolution_93, squeeze_280, view, permute_1, le, unsqueeze_378, le_1, unsqueeze_390, le_2, unsqueeze_402, unsqueeze_414, unsqueeze_426, le_5, unsqueeze_438, le_6, unsqueeze_450, unsqueeze_462, le_8, unsqueeze_474, le_9, unsqueeze_486, le_10, unsqueeze_498, le_11, unsqueeze_510, unsqueeze_522, unsqueeze_534, le_14, unsqueeze_546, le_15, unsqueeze_558, unsqueeze_570, le_17, unsqueeze_582, le_18, unsqueeze_594, unsqueeze_606, unsqueeze_618, unsqueeze_630, le_22, unsqueeze_642, unsqueeze_654, le_24, unsqueeze_666, le_25, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, le_30, unsqueeze_738, unsqueeze_750, unsqueeze_762, le_33, unsqueeze_774, le_34, unsqueeze_786, le_35, unsqueeze_798, unsqueeze_810, unsqueeze_822, unsqueeze_834, unsqueeze_846, le_40, unsqueeze_858, unsqueeze_870, unsqueeze_882, le_43, unsqueeze_894, le_44, unsqueeze_906, le_45, unsqueeze_918, unsqueeze_930, unsqueeze_942, unsqueeze_954, unsqueeze_966, le_50, unsqueeze_978, unsqueeze_990, unsqueeze_1002, le_53, unsqueeze_1014, le_54, unsqueeze_1026, le_55, unsqueeze_1038, unsqueeze_1050, unsqueeze_1062, unsqueeze_1074, unsqueeze_1086, le_60, unsqueeze_1098, unsqueeze_1110, unsqueeze_1122, le_63, unsqueeze_1134, le_64, unsqueeze_1146, unsqueeze_1158, unsqueeze_1170, le_67, unsqueeze_1182, le_68, unsqueeze_1194, le_69, unsqueeze_1206, unsqueeze_1218, unsqueeze_1230, le_72, unsqueeze_1242, unsqueeze_1254, le_74, unsqueeze_1266, le_75, unsqueeze_1278, le_76, unsqueeze_1290, unsqueeze_1302, unsqueeze_1314, le_79, unsqueeze_1326, unsqueeze_1338, le_81, unsqueeze_1350, le_82, unsqueeze_1362, le_83, unsqueeze_1374, unsqueeze_1386, unsqueeze_1398, le_86, unsqueeze_1410, unsqueeze_1422, le_88, unsqueeze_1434, unsqueeze_1446, unsqueeze_1458, unsqueeze_1470, unsqueeze_1482, unsqueeze_1494]
    