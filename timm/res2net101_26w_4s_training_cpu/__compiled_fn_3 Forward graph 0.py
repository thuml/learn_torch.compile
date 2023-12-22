from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[64]", primals_3: "f32[64]", primals_4: "f32[104, 64, 1, 1]", primals_5: "f32[104]", primals_6: "f32[104]", primals_7: "f32[26, 26, 3, 3]", primals_8: "f32[26]", primals_9: "f32[26]", primals_10: "f32[26, 26, 3, 3]", primals_11: "f32[26]", primals_12: "f32[26]", primals_13: "f32[26, 26, 3, 3]", primals_14: "f32[26]", primals_15: "f32[26]", primals_16: "f32[256, 104, 1, 1]", primals_17: "f32[256]", primals_18: "f32[256]", primals_19: "f32[256, 64, 1, 1]", primals_20: "f32[256]", primals_21: "f32[256]", primals_22: "f32[104, 256, 1, 1]", primals_23: "f32[104]", primals_24: "f32[104]", primals_25: "f32[26, 26, 3, 3]", primals_26: "f32[26]", primals_27: "f32[26]", primals_28: "f32[26, 26, 3, 3]", primals_29: "f32[26]", primals_30: "f32[26]", primals_31: "f32[26, 26, 3, 3]", primals_32: "f32[26]", primals_33: "f32[26]", primals_34: "f32[256, 104, 1, 1]", primals_35: "f32[256]", primals_36: "f32[256]", primals_37: "f32[104, 256, 1, 1]", primals_38: "f32[104]", primals_39: "f32[104]", primals_40: "f32[26, 26, 3, 3]", primals_41: "f32[26]", primals_42: "f32[26]", primals_43: "f32[26, 26, 3, 3]", primals_44: "f32[26]", primals_45: "f32[26]", primals_46: "f32[26, 26, 3, 3]", primals_47: "f32[26]", primals_48: "f32[26]", primals_49: "f32[256, 104, 1, 1]", primals_50: "f32[256]", primals_51: "f32[256]", primals_52: "f32[208, 256, 1, 1]", primals_53: "f32[208]", primals_54: "f32[208]", primals_55: "f32[52, 52, 3, 3]", primals_56: "f32[52]", primals_57: "f32[52]", primals_58: "f32[52, 52, 3, 3]", primals_59: "f32[52]", primals_60: "f32[52]", primals_61: "f32[52, 52, 3, 3]", primals_62: "f32[52]", primals_63: "f32[52]", primals_64: "f32[512, 208, 1, 1]", primals_65: "f32[512]", primals_66: "f32[512]", primals_67: "f32[512, 256, 1, 1]", primals_68: "f32[512]", primals_69: "f32[512]", primals_70: "f32[208, 512, 1, 1]", primals_71: "f32[208]", primals_72: "f32[208]", primals_73: "f32[52, 52, 3, 3]", primals_74: "f32[52]", primals_75: "f32[52]", primals_76: "f32[52, 52, 3, 3]", primals_77: "f32[52]", primals_78: "f32[52]", primals_79: "f32[52, 52, 3, 3]", primals_80: "f32[52]", primals_81: "f32[52]", primals_82: "f32[512, 208, 1, 1]", primals_83: "f32[512]", primals_84: "f32[512]", primals_85: "f32[208, 512, 1, 1]", primals_86: "f32[208]", primals_87: "f32[208]", primals_88: "f32[52, 52, 3, 3]", primals_89: "f32[52]", primals_90: "f32[52]", primals_91: "f32[52, 52, 3, 3]", primals_92: "f32[52]", primals_93: "f32[52]", primals_94: "f32[52, 52, 3, 3]", primals_95: "f32[52]", primals_96: "f32[52]", primals_97: "f32[512, 208, 1, 1]", primals_98: "f32[512]", primals_99: "f32[512]", primals_100: "f32[208, 512, 1, 1]", primals_101: "f32[208]", primals_102: "f32[208]", primals_103: "f32[52, 52, 3, 3]", primals_104: "f32[52]", primals_105: "f32[52]", primals_106: "f32[52, 52, 3, 3]", primals_107: "f32[52]", primals_108: "f32[52]", primals_109: "f32[52, 52, 3, 3]", primals_110: "f32[52]", primals_111: "f32[52]", primals_112: "f32[512, 208, 1, 1]", primals_113: "f32[512]", primals_114: "f32[512]", primals_115: "f32[416, 512, 1, 1]", primals_116: "f32[416]", primals_117: "f32[416]", primals_118: "f32[104, 104, 3, 3]", primals_119: "f32[104]", primals_120: "f32[104]", primals_121: "f32[104, 104, 3, 3]", primals_122: "f32[104]", primals_123: "f32[104]", primals_124: "f32[104, 104, 3, 3]", primals_125: "f32[104]", primals_126: "f32[104]", primals_127: "f32[1024, 416, 1, 1]", primals_128: "f32[1024]", primals_129: "f32[1024]", primals_130: "f32[1024, 512, 1, 1]", primals_131: "f32[1024]", primals_132: "f32[1024]", primals_133: "f32[416, 1024, 1, 1]", primals_134: "f32[416]", primals_135: "f32[416]", primals_136: "f32[104, 104, 3, 3]", primals_137: "f32[104]", primals_138: "f32[104]", primals_139: "f32[104, 104, 3, 3]", primals_140: "f32[104]", primals_141: "f32[104]", primals_142: "f32[104, 104, 3, 3]", primals_143: "f32[104]", primals_144: "f32[104]", primals_145: "f32[1024, 416, 1, 1]", primals_146: "f32[1024]", primals_147: "f32[1024]", primals_148: "f32[416, 1024, 1, 1]", primals_149: "f32[416]", primals_150: "f32[416]", primals_151: "f32[104, 104, 3, 3]", primals_152: "f32[104]", primals_153: "f32[104]", primals_154: "f32[104, 104, 3, 3]", primals_155: "f32[104]", primals_156: "f32[104]", primals_157: "f32[104, 104, 3, 3]", primals_158: "f32[104]", primals_159: "f32[104]", primals_160: "f32[1024, 416, 1, 1]", primals_161: "f32[1024]", primals_162: "f32[1024]", primals_163: "f32[416, 1024, 1, 1]", primals_164: "f32[416]", primals_165: "f32[416]", primals_166: "f32[104, 104, 3, 3]", primals_167: "f32[104]", primals_168: "f32[104]", primals_169: "f32[104, 104, 3, 3]", primals_170: "f32[104]", primals_171: "f32[104]", primals_172: "f32[104, 104, 3, 3]", primals_173: "f32[104]", primals_174: "f32[104]", primals_175: "f32[1024, 416, 1, 1]", primals_176: "f32[1024]", primals_177: "f32[1024]", primals_178: "f32[416, 1024, 1, 1]", primals_179: "f32[416]", primals_180: "f32[416]", primals_181: "f32[104, 104, 3, 3]", primals_182: "f32[104]", primals_183: "f32[104]", primals_184: "f32[104, 104, 3, 3]", primals_185: "f32[104]", primals_186: "f32[104]", primals_187: "f32[104, 104, 3, 3]", primals_188: "f32[104]", primals_189: "f32[104]", primals_190: "f32[1024, 416, 1, 1]", primals_191: "f32[1024]", primals_192: "f32[1024]", primals_193: "f32[416, 1024, 1, 1]", primals_194: "f32[416]", primals_195: "f32[416]", primals_196: "f32[104, 104, 3, 3]", primals_197: "f32[104]", primals_198: "f32[104]", primals_199: "f32[104, 104, 3, 3]", primals_200: "f32[104]", primals_201: "f32[104]", primals_202: "f32[104, 104, 3, 3]", primals_203: "f32[104]", primals_204: "f32[104]", primals_205: "f32[1024, 416, 1, 1]", primals_206: "f32[1024]", primals_207: "f32[1024]", primals_208: "f32[416, 1024, 1, 1]", primals_209: "f32[416]", primals_210: "f32[416]", primals_211: "f32[104, 104, 3, 3]", primals_212: "f32[104]", primals_213: "f32[104]", primals_214: "f32[104, 104, 3, 3]", primals_215: "f32[104]", primals_216: "f32[104]", primals_217: "f32[104, 104, 3, 3]", primals_218: "f32[104]", primals_219: "f32[104]", primals_220: "f32[1024, 416, 1, 1]", primals_221: "f32[1024]", primals_222: "f32[1024]", primals_223: "f32[416, 1024, 1, 1]", primals_224: "f32[416]", primals_225: "f32[416]", primals_226: "f32[104, 104, 3, 3]", primals_227: "f32[104]", primals_228: "f32[104]", primals_229: "f32[104, 104, 3, 3]", primals_230: "f32[104]", primals_231: "f32[104]", primals_232: "f32[104, 104, 3, 3]", primals_233: "f32[104]", primals_234: "f32[104]", primals_235: "f32[1024, 416, 1, 1]", primals_236: "f32[1024]", primals_237: "f32[1024]", primals_238: "f32[416, 1024, 1, 1]", primals_239: "f32[416]", primals_240: "f32[416]", primals_241: "f32[104, 104, 3, 3]", primals_242: "f32[104]", primals_243: "f32[104]", primals_244: "f32[104, 104, 3, 3]", primals_245: "f32[104]", primals_246: "f32[104]", primals_247: "f32[104, 104, 3, 3]", primals_248: "f32[104]", primals_249: "f32[104]", primals_250: "f32[1024, 416, 1, 1]", primals_251: "f32[1024]", primals_252: "f32[1024]", primals_253: "f32[416, 1024, 1, 1]", primals_254: "f32[416]", primals_255: "f32[416]", primals_256: "f32[104, 104, 3, 3]", primals_257: "f32[104]", primals_258: "f32[104]", primals_259: "f32[104, 104, 3, 3]", primals_260: "f32[104]", primals_261: "f32[104]", primals_262: "f32[104, 104, 3, 3]", primals_263: "f32[104]", primals_264: "f32[104]", primals_265: "f32[1024, 416, 1, 1]", primals_266: "f32[1024]", primals_267: "f32[1024]", primals_268: "f32[416, 1024, 1, 1]", primals_269: "f32[416]", primals_270: "f32[416]", primals_271: "f32[104, 104, 3, 3]", primals_272: "f32[104]", primals_273: "f32[104]", primals_274: "f32[104, 104, 3, 3]", primals_275: "f32[104]", primals_276: "f32[104]", primals_277: "f32[104, 104, 3, 3]", primals_278: "f32[104]", primals_279: "f32[104]", primals_280: "f32[1024, 416, 1, 1]", primals_281: "f32[1024]", primals_282: "f32[1024]", primals_283: "f32[416, 1024, 1, 1]", primals_284: "f32[416]", primals_285: "f32[416]", primals_286: "f32[104, 104, 3, 3]", primals_287: "f32[104]", primals_288: "f32[104]", primals_289: "f32[104, 104, 3, 3]", primals_290: "f32[104]", primals_291: "f32[104]", primals_292: "f32[104, 104, 3, 3]", primals_293: "f32[104]", primals_294: "f32[104]", primals_295: "f32[1024, 416, 1, 1]", primals_296: "f32[1024]", primals_297: "f32[1024]", primals_298: "f32[416, 1024, 1, 1]", primals_299: "f32[416]", primals_300: "f32[416]", primals_301: "f32[104, 104, 3, 3]", primals_302: "f32[104]", primals_303: "f32[104]", primals_304: "f32[104, 104, 3, 3]", primals_305: "f32[104]", primals_306: "f32[104]", primals_307: "f32[104, 104, 3, 3]", primals_308: "f32[104]", primals_309: "f32[104]", primals_310: "f32[1024, 416, 1, 1]", primals_311: "f32[1024]", primals_312: "f32[1024]", primals_313: "f32[416, 1024, 1, 1]", primals_314: "f32[416]", primals_315: "f32[416]", primals_316: "f32[104, 104, 3, 3]", primals_317: "f32[104]", primals_318: "f32[104]", primals_319: "f32[104, 104, 3, 3]", primals_320: "f32[104]", primals_321: "f32[104]", primals_322: "f32[104, 104, 3, 3]", primals_323: "f32[104]", primals_324: "f32[104]", primals_325: "f32[1024, 416, 1, 1]", primals_326: "f32[1024]", primals_327: "f32[1024]", primals_328: "f32[416, 1024, 1, 1]", primals_329: "f32[416]", primals_330: "f32[416]", primals_331: "f32[104, 104, 3, 3]", primals_332: "f32[104]", primals_333: "f32[104]", primals_334: "f32[104, 104, 3, 3]", primals_335: "f32[104]", primals_336: "f32[104]", primals_337: "f32[104, 104, 3, 3]", primals_338: "f32[104]", primals_339: "f32[104]", primals_340: "f32[1024, 416, 1, 1]", primals_341: "f32[1024]", primals_342: "f32[1024]", primals_343: "f32[416, 1024, 1, 1]", primals_344: "f32[416]", primals_345: "f32[416]", primals_346: "f32[104, 104, 3, 3]", primals_347: "f32[104]", primals_348: "f32[104]", primals_349: "f32[104, 104, 3, 3]", primals_350: "f32[104]", primals_351: "f32[104]", primals_352: "f32[104, 104, 3, 3]", primals_353: "f32[104]", primals_354: "f32[104]", primals_355: "f32[1024, 416, 1, 1]", primals_356: "f32[1024]", primals_357: "f32[1024]", primals_358: "f32[416, 1024, 1, 1]", primals_359: "f32[416]", primals_360: "f32[416]", primals_361: "f32[104, 104, 3, 3]", primals_362: "f32[104]", primals_363: "f32[104]", primals_364: "f32[104, 104, 3, 3]", primals_365: "f32[104]", primals_366: "f32[104]", primals_367: "f32[104, 104, 3, 3]", primals_368: "f32[104]", primals_369: "f32[104]", primals_370: "f32[1024, 416, 1, 1]", primals_371: "f32[1024]", primals_372: "f32[1024]", primals_373: "f32[416, 1024, 1, 1]", primals_374: "f32[416]", primals_375: "f32[416]", primals_376: "f32[104, 104, 3, 3]", primals_377: "f32[104]", primals_378: "f32[104]", primals_379: "f32[104, 104, 3, 3]", primals_380: "f32[104]", primals_381: "f32[104]", primals_382: "f32[104, 104, 3, 3]", primals_383: "f32[104]", primals_384: "f32[104]", primals_385: "f32[1024, 416, 1, 1]", primals_386: "f32[1024]", primals_387: "f32[1024]", primals_388: "f32[416, 1024, 1, 1]", primals_389: "f32[416]", primals_390: "f32[416]", primals_391: "f32[104, 104, 3, 3]", primals_392: "f32[104]", primals_393: "f32[104]", primals_394: "f32[104, 104, 3, 3]", primals_395: "f32[104]", primals_396: "f32[104]", primals_397: "f32[104, 104, 3, 3]", primals_398: "f32[104]", primals_399: "f32[104]", primals_400: "f32[1024, 416, 1, 1]", primals_401: "f32[1024]", primals_402: "f32[1024]", primals_403: "f32[416, 1024, 1, 1]", primals_404: "f32[416]", primals_405: "f32[416]", primals_406: "f32[104, 104, 3, 3]", primals_407: "f32[104]", primals_408: "f32[104]", primals_409: "f32[104, 104, 3, 3]", primals_410: "f32[104]", primals_411: "f32[104]", primals_412: "f32[104, 104, 3, 3]", primals_413: "f32[104]", primals_414: "f32[104]", primals_415: "f32[1024, 416, 1, 1]", primals_416: "f32[1024]", primals_417: "f32[1024]", primals_418: "f32[416, 1024, 1, 1]", primals_419: "f32[416]", primals_420: "f32[416]", primals_421: "f32[104, 104, 3, 3]", primals_422: "f32[104]", primals_423: "f32[104]", primals_424: "f32[104, 104, 3, 3]", primals_425: "f32[104]", primals_426: "f32[104]", primals_427: "f32[104, 104, 3, 3]", primals_428: "f32[104]", primals_429: "f32[104]", primals_430: "f32[1024, 416, 1, 1]", primals_431: "f32[1024]", primals_432: "f32[1024]", primals_433: "f32[416, 1024, 1, 1]", primals_434: "f32[416]", primals_435: "f32[416]", primals_436: "f32[104, 104, 3, 3]", primals_437: "f32[104]", primals_438: "f32[104]", primals_439: "f32[104, 104, 3, 3]", primals_440: "f32[104]", primals_441: "f32[104]", primals_442: "f32[104, 104, 3, 3]", primals_443: "f32[104]", primals_444: "f32[104]", primals_445: "f32[1024, 416, 1, 1]", primals_446: "f32[1024]", primals_447: "f32[1024]", primals_448: "f32[416, 1024, 1, 1]", primals_449: "f32[416]", primals_450: "f32[416]", primals_451: "f32[104, 104, 3, 3]", primals_452: "f32[104]", primals_453: "f32[104]", primals_454: "f32[104, 104, 3, 3]", primals_455: "f32[104]", primals_456: "f32[104]", primals_457: "f32[104, 104, 3, 3]", primals_458: "f32[104]", primals_459: "f32[104]", primals_460: "f32[1024, 416, 1, 1]", primals_461: "f32[1024]", primals_462: "f32[1024]", primals_463: "f32[832, 1024, 1, 1]", primals_464: "f32[832]", primals_465: "f32[832]", primals_466: "f32[208, 208, 3, 3]", primals_467: "f32[208]", primals_468: "f32[208]", primals_469: "f32[208, 208, 3, 3]", primals_470: "f32[208]", primals_471: "f32[208]", primals_472: "f32[208, 208, 3, 3]", primals_473: "f32[208]", primals_474: "f32[208]", primals_475: "f32[2048, 832, 1, 1]", primals_476: "f32[2048]", primals_477: "f32[2048]", primals_478: "f32[2048, 1024, 1, 1]", primals_479: "f32[2048]", primals_480: "f32[2048]", primals_481: "f32[832, 2048, 1, 1]", primals_482: "f32[832]", primals_483: "f32[832]", primals_484: "f32[208, 208, 3, 3]", primals_485: "f32[208]", primals_486: "f32[208]", primals_487: "f32[208, 208, 3, 3]", primals_488: "f32[208]", primals_489: "f32[208]", primals_490: "f32[208, 208, 3, 3]", primals_491: "f32[208]", primals_492: "f32[208]", primals_493: "f32[2048, 832, 1, 1]", primals_494: "f32[2048]", primals_495: "f32[2048]", primals_496: "f32[832, 2048, 1, 1]", primals_497: "f32[832]", primals_498: "f32[832]", primals_499: "f32[208, 208, 3, 3]", primals_500: "f32[208]", primals_501: "f32[208]", primals_502: "f32[208, 208, 3, 3]", primals_503: "f32[208]", primals_504: "f32[208]", primals_505: "f32[208, 208, 3, 3]", primals_506: "f32[208]", primals_507: "f32[208]", primals_508: "f32[2048, 832, 1, 1]", primals_509: "f32[2048]", primals_510: "f32[2048]", primals_511: "f32[1000, 2048]", primals_512: "f32[1000]", primals_513: "f32[64]", primals_514: "f32[64]", primals_515: "i64[]", primals_516: "f32[104]", primals_517: "f32[104]", primals_518: "i64[]", primals_519: "f32[26]", primals_520: "f32[26]", primals_521: "i64[]", primals_522: "f32[26]", primals_523: "f32[26]", primals_524: "i64[]", primals_525: "f32[26]", primals_526: "f32[26]", primals_527: "i64[]", primals_528: "f32[256]", primals_529: "f32[256]", primals_530: "i64[]", primals_531: "f32[256]", primals_532: "f32[256]", primals_533: "i64[]", primals_534: "f32[104]", primals_535: "f32[104]", primals_536: "i64[]", primals_537: "f32[26]", primals_538: "f32[26]", primals_539: "i64[]", primals_540: "f32[26]", primals_541: "f32[26]", primals_542: "i64[]", primals_543: "f32[26]", primals_544: "f32[26]", primals_545: "i64[]", primals_546: "f32[256]", primals_547: "f32[256]", primals_548: "i64[]", primals_549: "f32[104]", primals_550: "f32[104]", primals_551: "i64[]", primals_552: "f32[26]", primals_553: "f32[26]", primals_554: "i64[]", primals_555: "f32[26]", primals_556: "f32[26]", primals_557: "i64[]", primals_558: "f32[26]", primals_559: "f32[26]", primals_560: "i64[]", primals_561: "f32[256]", primals_562: "f32[256]", primals_563: "i64[]", primals_564: "f32[208]", primals_565: "f32[208]", primals_566: "i64[]", primals_567: "f32[52]", primals_568: "f32[52]", primals_569: "i64[]", primals_570: "f32[52]", primals_571: "f32[52]", primals_572: "i64[]", primals_573: "f32[52]", primals_574: "f32[52]", primals_575: "i64[]", primals_576: "f32[512]", primals_577: "f32[512]", primals_578: "i64[]", primals_579: "f32[512]", primals_580: "f32[512]", primals_581: "i64[]", primals_582: "f32[208]", primals_583: "f32[208]", primals_584: "i64[]", primals_585: "f32[52]", primals_586: "f32[52]", primals_587: "i64[]", primals_588: "f32[52]", primals_589: "f32[52]", primals_590: "i64[]", primals_591: "f32[52]", primals_592: "f32[52]", primals_593: "i64[]", primals_594: "f32[512]", primals_595: "f32[512]", primals_596: "i64[]", primals_597: "f32[208]", primals_598: "f32[208]", primals_599: "i64[]", primals_600: "f32[52]", primals_601: "f32[52]", primals_602: "i64[]", primals_603: "f32[52]", primals_604: "f32[52]", primals_605: "i64[]", primals_606: "f32[52]", primals_607: "f32[52]", primals_608: "i64[]", primals_609: "f32[512]", primals_610: "f32[512]", primals_611: "i64[]", primals_612: "f32[208]", primals_613: "f32[208]", primals_614: "i64[]", primals_615: "f32[52]", primals_616: "f32[52]", primals_617: "i64[]", primals_618: "f32[52]", primals_619: "f32[52]", primals_620: "i64[]", primals_621: "f32[52]", primals_622: "f32[52]", primals_623: "i64[]", primals_624: "f32[512]", primals_625: "f32[512]", primals_626: "i64[]", primals_627: "f32[416]", primals_628: "f32[416]", primals_629: "i64[]", primals_630: "f32[104]", primals_631: "f32[104]", primals_632: "i64[]", primals_633: "f32[104]", primals_634: "f32[104]", primals_635: "i64[]", primals_636: "f32[104]", primals_637: "f32[104]", primals_638: "i64[]", primals_639: "f32[1024]", primals_640: "f32[1024]", primals_641: "i64[]", primals_642: "f32[1024]", primals_643: "f32[1024]", primals_644: "i64[]", primals_645: "f32[416]", primals_646: "f32[416]", primals_647: "i64[]", primals_648: "f32[104]", primals_649: "f32[104]", primals_650: "i64[]", primals_651: "f32[104]", primals_652: "f32[104]", primals_653: "i64[]", primals_654: "f32[104]", primals_655: "f32[104]", primals_656: "i64[]", primals_657: "f32[1024]", primals_658: "f32[1024]", primals_659: "i64[]", primals_660: "f32[416]", primals_661: "f32[416]", primals_662: "i64[]", primals_663: "f32[104]", primals_664: "f32[104]", primals_665: "i64[]", primals_666: "f32[104]", primals_667: "f32[104]", primals_668: "i64[]", primals_669: "f32[104]", primals_670: "f32[104]", primals_671: "i64[]", primals_672: "f32[1024]", primals_673: "f32[1024]", primals_674: "i64[]", primals_675: "f32[416]", primals_676: "f32[416]", primals_677: "i64[]", primals_678: "f32[104]", primals_679: "f32[104]", primals_680: "i64[]", primals_681: "f32[104]", primals_682: "f32[104]", primals_683: "i64[]", primals_684: "f32[104]", primals_685: "f32[104]", primals_686: "i64[]", primals_687: "f32[1024]", primals_688: "f32[1024]", primals_689: "i64[]", primals_690: "f32[416]", primals_691: "f32[416]", primals_692: "i64[]", primals_693: "f32[104]", primals_694: "f32[104]", primals_695: "i64[]", primals_696: "f32[104]", primals_697: "f32[104]", primals_698: "i64[]", primals_699: "f32[104]", primals_700: "f32[104]", primals_701: "i64[]", primals_702: "f32[1024]", primals_703: "f32[1024]", primals_704: "i64[]", primals_705: "f32[416]", primals_706: "f32[416]", primals_707: "i64[]", primals_708: "f32[104]", primals_709: "f32[104]", primals_710: "i64[]", primals_711: "f32[104]", primals_712: "f32[104]", primals_713: "i64[]", primals_714: "f32[104]", primals_715: "f32[104]", primals_716: "i64[]", primals_717: "f32[1024]", primals_718: "f32[1024]", primals_719: "i64[]", primals_720: "f32[416]", primals_721: "f32[416]", primals_722: "i64[]", primals_723: "f32[104]", primals_724: "f32[104]", primals_725: "i64[]", primals_726: "f32[104]", primals_727: "f32[104]", primals_728: "i64[]", primals_729: "f32[104]", primals_730: "f32[104]", primals_731: "i64[]", primals_732: "f32[1024]", primals_733: "f32[1024]", primals_734: "i64[]", primals_735: "f32[416]", primals_736: "f32[416]", primals_737: "i64[]", primals_738: "f32[104]", primals_739: "f32[104]", primals_740: "i64[]", primals_741: "f32[104]", primals_742: "f32[104]", primals_743: "i64[]", primals_744: "f32[104]", primals_745: "f32[104]", primals_746: "i64[]", primals_747: "f32[1024]", primals_748: "f32[1024]", primals_749: "i64[]", primals_750: "f32[416]", primals_751: "f32[416]", primals_752: "i64[]", primals_753: "f32[104]", primals_754: "f32[104]", primals_755: "i64[]", primals_756: "f32[104]", primals_757: "f32[104]", primals_758: "i64[]", primals_759: "f32[104]", primals_760: "f32[104]", primals_761: "i64[]", primals_762: "f32[1024]", primals_763: "f32[1024]", primals_764: "i64[]", primals_765: "f32[416]", primals_766: "f32[416]", primals_767: "i64[]", primals_768: "f32[104]", primals_769: "f32[104]", primals_770: "i64[]", primals_771: "f32[104]", primals_772: "f32[104]", primals_773: "i64[]", primals_774: "f32[104]", primals_775: "f32[104]", primals_776: "i64[]", primals_777: "f32[1024]", primals_778: "f32[1024]", primals_779: "i64[]", primals_780: "f32[416]", primals_781: "f32[416]", primals_782: "i64[]", primals_783: "f32[104]", primals_784: "f32[104]", primals_785: "i64[]", primals_786: "f32[104]", primals_787: "f32[104]", primals_788: "i64[]", primals_789: "f32[104]", primals_790: "f32[104]", primals_791: "i64[]", primals_792: "f32[1024]", primals_793: "f32[1024]", primals_794: "i64[]", primals_795: "f32[416]", primals_796: "f32[416]", primals_797: "i64[]", primals_798: "f32[104]", primals_799: "f32[104]", primals_800: "i64[]", primals_801: "f32[104]", primals_802: "f32[104]", primals_803: "i64[]", primals_804: "f32[104]", primals_805: "f32[104]", primals_806: "i64[]", primals_807: "f32[1024]", primals_808: "f32[1024]", primals_809: "i64[]", primals_810: "f32[416]", primals_811: "f32[416]", primals_812: "i64[]", primals_813: "f32[104]", primals_814: "f32[104]", primals_815: "i64[]", primals_816: "f32[104]", primals_817: "f32[104]", primals_818: "i64[]", primals_819: "f32[104]", primals_820: "f32[104]", primals_821: "i64[]", primals_822: "f32[1024]", primals_823: "f32[1024]", primals_824: "i64[]", primals_825: "f32[416]", primals_826: "f32[416]", primals_827: "i64[]", primals_828: "f32[104]", primals_829: "f32[104]", primals_830: "i64[]", primals_831: "f32[104]", primals_832: "f32[104]", primals_833: "i64[]", primals_834: "f32[104]", primals_835: "f32[104]", primals_836: "i64[]", primals_837: "f32[1024]", primals_838: "f32[1024]", primals_839: "i64[]", primals_840: "f32[416]", primals_841: "f32[416]", primals_842: "i64[]", primals_843: "f32[104]", primals_844: "f32[104]", primals_845: "i64[]", primals_846: "f32[104]", primals_847: "f32[104]", primals_848: "i64[]", primals_849: "f32[104]", primals_850: "f32[104]", primals_851: "i64[]", primals_852: "f32[1024]", primals_853: "f32[1024]", primals_854: "i64[]", primals_855: "f32[416]", primals_856: "f32[416]", primals_857: "i64[]", primals_858: "f32[104]", primals_859: "f32[104]", primals_860: "i64[]", primals_861: "f32[104]", primals_862: "f32[104]", primals_863: "i64[]", primals_864: "f32[104]", primals_865: "f32[104]", primals_866: "i64[]", primals_867: "f32[1024]", primals_868: "f32[1024]", primals_869: "i64[]", primals_870: "f32[416]", primals_871: "f32[416]", primals_872: "i64[]", primals_873: "f32[104]", primals_874: "f32[104]", primals_875: "i64[]", primals_876: "f32[104]", primals_877: "f32[104]", primals_878: "i64[]", primals_879: "f32[104]", primals_880: "f32[104]", primals_881: "i64[]", primals_882: "f32[1024]", primals_883: "f32[1024]", primals_884: "i64[]", primals_885: "f32[416]", primals_886: "f32[416]", primals_887: "i64[]", primals_888: "f32[104]", primals_889: "f32[104]", primals_890: "i64[]", primals_891: "f32[104]", primals_892: "f32[104]", primals_893: "i64[]", primals_894: "f32[104]", primals_895: "f32[104]", primals_896: "i64[]", primals_897: "f32[1024]", primals_898: "f32[1024]", primals_899: "i64[]", primals_900: "f32[416]", primals_901: "f32[416]", primals_902: "i64[]", primals_903: "f32[104]", primals_904: "f32[104]", primals_905: "i64[]", primals_906: "f32[104]", primals_907: "f32[104]", primals_908: "i64[]", primals_909: "f32[104]", primals_910: "f32[104]", primals_911: "i64[]", primals_912: "f32[1024]", primals_913: "f32[1024]", primals_914: "i64[]", primals_915: "f32[416]", primals_916: "f32[416]", primals_917: "i64[]", primals_918: "f32[104]", primals_919: "f32[104]", primals_920: "i64[]", primals_921: "f32[104]", primals_922: "f32[104]", primals_923: "i64[]", primals_924: "f32[104]", primals_925: "f32[104]", primals_926: "i64[]", primals_927: "f32[1024]", primals_928: "f32[1024]", primals_929: "i64[]", primals_930: "f32[416]", primals_931: "f32[416]", primals_932: "i64[]", primals_933: "f32[104]", primals_934: "f32[104]", primals_935: "i64[]", primals_936: "f32[104]", primals_937: "f32[104]", primals_938: "i64[]", primals_939: "f32[104]", primals_940: "f32[104]", primals_941: "i64[]", primals_942: "f32[1024]", primals_943: "f32[1024]", primals_944: "i64[]", primals_945: "f32[416]", primals_946: "f32[416]", primals_947: "i64[]", primals_948: "f32[104]", primals_949: "f32[104]", primals_950: "i64[]", primals_951: "f32[104]", primals_952: "f32[104]", primals_953: "i64[]", primals_954: "f32[104]", primals_955: "f32[104]", primals_956: "i64[]", primals_957: "f32[1024]", primals_958: "f32[1024]", primals_959: "i64[]", primals_960: "f32[416]", primals_961: "f32[416]", primals_962: "i64[]", primals_963: "f32[104]", primals_964: "f32[104]", primals_965: "i64[]", primals_966: "f32[104]", primals_967: "f32[104]", primals_968: "i64[]", primals_969: "f32[104]", primals_970: "f32[104]", primals_971: "i64[]", primals_972: "f32[1024]", primals_973: "f32[1024]", primals_974: "i64[]", primals_975: "f32[832]", primals_976: "f32[832]", primals_977: "i64[]", primals_978: "f32[208]", primals_979: "f32[208]", primals_980: "i64[]", primals_981: "f32[208]", primals_982: "f32[208]", primals_983: "i64[]", primals_984: "f32[208]", primals_985: "f32[208]", primals_986: "i64[]", primals_987: "f32[2048]", primals_988: "f32[2048]", primals_989: "i64[]", primals_990: "f32[2048]", primals_991: "f32[2048]", primals_992: "i64[]", primals_993: "f32[832]", primals_994: "f32[832]", primals_995: "i64[]", primals_996: "f32[208]", primals_997: "f32[208]", primals_998: "i64[]", primals_999: "f32[208]", primals_1000: "f32[208]", primals_1001: "i64[]", primals_1002: "f32[208]", primals_1003: "f32[208]", primals_1004: "i64[]", primals_1005: "f32[2048]", primals_1006: "f32[2048]", primals_1007: "i64[]", primals_1008: "f32[832]", primals_1009: "f32[832]", primals_1010: "i64[]", primals_1011: "f32[208]", primals_1012: "f32[208]", primals_1013: "i64[]", primals_1014: "f32[208]", primals_1015: "f32[208]", primals_1016: "i64[]", primals_1017: "f32[208]", primals_1018: "f32[208]", primals_1019: "i64[]", primals_1020: "f32[2048]", primals_1021: "f32[2048]", primals_1022: "i64[]", primals_1023: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(primals_1023, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_515, 1)
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
    mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(primals_513, 0.9)
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(primals_514, 0.9)
    add_3: "f32[64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    relu: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1])
    getitem_2: "f32[8, 64, 56, 56]" = max_pool2d_with_indices[0]
    getitem_3: "i64[8, 64, 56, 56]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_1: "f32[8, 104, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, primals_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_518, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 104, 1, 1]" = var_mean_1[0]
    getitem_5: "f32[1, 104, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_1: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_5)
    mul_7: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_4: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[104]" = torch.ops.aten.mul.Tensor(primals_516, 0.9)
    add_7: "f32[104]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_10: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000398612827361);  squeeze_5 = None
    mul_11: "f32[104]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[104]" = torch.ops.aten.mul.Tensor(primals_517, 0.9)
    add_8: "f32[104]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_5: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_7: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 104, 56, 56]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_1: "f32[8, 104, 56, 56]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(relu_1, [26, 26, 26, 26], 1)
    getitem_10: "f32[8, 26, 56, 56]" = split_with_sizes_1[0]
    convolution_2: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_10, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_521, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 26, 1, 1]" = var_mean_2[0]
    getitem_15: "f32[1, 26, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 26, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_2: "f32[1, 26, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_15)
    mul_14: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_7: "f32[26]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[26]" = torch.ops.aten.mul.Tensor(primals_519, 0.9)
    add_12: "f32[26]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_17: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000398612827361);  squeeze_8 = None
    mul_18: "f32[26]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[26]" = torch.ops.aten.mul.Tensor(primals_520, 0.9)
    add_13: "f32[26]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_9: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_11: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_2: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    getitem_17: "f32[8, 26, 56, 56]" = split_with_sizes_1[1]
    convolution_3: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_17, primals_10, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_524, 1)
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 26, 1, 1]" = var_mean_3[0]
    getitem_21: "f32[1, 26, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 26, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_3: "f32[1, 26, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_21)
    mul_21: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_10: "f32[26]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[26]" = torch.ops.aten.mul.Tensor(primals_522, 0.9)
    add_17: "f32[26]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_24: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000398612827361);  squeeze_11 = None
    mul_25: "f32[26]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[26]" = torch.ops.aten.mul.Tensor(primals_523, 0.9)
    add_18: "f32[26]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_13: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_15: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_3: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    getitem_24: "f32[8, 26, 56, 56]" = split_with_sizes_1[2]
    convolution_4: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_24, primals_13, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_527, 1)
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 26, 1, 1]" = var_mean_4[0]
    getitem_27: "f32[1, 26, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 26, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_4: "f32[1, 26, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_27)
    mul_28: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_13: "f32[26]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[26]" = torch.ops.aten.mul.Tensor(primals_525, 0.9)
    add_22: "f32[26]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_31: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_32: "f32[26]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[26]" = torch.ops.aten.mul.Tensor(primals_526, 0.9)
    add_23: "f32[26]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_17: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_19: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_4: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    getitem_31: "f32[8, 26, 56, 56]" = split_with_sizes_1[3];  split_with_sizes_1 = None
    avg_pool2d: "f32[8, 26, 56, 56]" = torch.ops.aten.avg_pool2d.default(getitem_31, [3, 3], [1, 1], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat: "f32[8, 104, 56, 56]" = torch.ops.aten.cat.default([relu_2, relu_3, relu_4, avg_pool2d], 1);  avg_pool2d = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_5: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_530, 1)
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 256, 1, 1]" = var_mean_5[0]
    getitem_33: "f32[1, 256, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_5: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_33)
    mul_35: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_16: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[256]" = torch.ops.aten.mul.Tensor(primals_528, 0.9)
    add_27: "f32[256]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_38: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_39: "f32[256]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[256]" = torch.ops.aten.mul.Tensor(primals_529, 0.9)
    add_28: "f32[256]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_21: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_23: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_6: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, primals_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_533, 1)
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 256, 1, 1]" = var_mean_6[0]
    getitem_35: "f32[1, 256, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_6: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_35)
    mul_42: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_19: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[256]" = torch.ops.aten.mul.Tensor(primals_531, 0.9)
    add_32: "f32[256]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_45: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_46: "f32[256]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[256]" = torch.ops.aten.mul.Tensor(primals_532, 0.9)
    add_33: "f32[256]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
    unsqueeze_25: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
    unsqueeze_27: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_35: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_29, add_34);  add_29 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_5: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_7: "f32[8, 104, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_536, 1)
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 104, 1, 1]" = var_mean_7[0]
    getitem_37: "f32[1, 104, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_7: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_37)
    mul_49: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_22: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[104]" = torch.ops.aten.mul.Tensor(primals_534, 0.9)
    add_38: "f32[104]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_52: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_53: "f32[104]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[104]" = torch.ops.aten.mul.Tensor(primals_535, 0.9)
    add_39: "f32[104]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_29: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_31: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 104, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_6: "f32[8, 104, 56, 56]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(relu_6, [26, 26, 26, 26], 1)
    getitem_42: "f32[8, 26, 56, 56]" = split_with_sizes_6[0]
    convolution_8: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_42, primals_25, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_539, 1)
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 26, 1, 1]" = var_mean_8[0]
    getitem_47: "f32[1, 26, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 26, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_8: "f32[1, 26, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_47)
    mul_56: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_25: "f32[26]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[26]" = torch.ops.aten.mul.Tensor(primals_537, 0.9)
    add_43: "f32[26]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_59: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_60: "f32[26]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[26]" = torch.ops.aten.mul.Tensor(primals_538, 0.9)
    add_44: "f32[26]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_33: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_35: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_7: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_45);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_49: "f32[8, 26, 56, 56]" = split_with_sizes_6[1]
    add_46: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(relu_7, getitem_49);  getitem_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_9: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(add_46, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_542, 1)
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 26, 1, 1]" = var_mean_9[0]
    getitem_53: "f32[1, 26, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_48: "f32[1, 26, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_9: "f32[1, 26, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_9: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_53)
    mul_63: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_28: "f32[26]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[26]" = torch.ops.aten.mul.Tensor(primals_540, 0.9)
    add_49: "f32[26]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_66: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_67: "f32[26]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[26]" = torch.ops.aten.mul.Tensor(primals_541, 0.9)
    add_50: "f32[26]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_37: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_39: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_51: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_8: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_56: "f32[8, 26, 56, 56]" = split_with_sizes_6[2]
    add_52: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(relu_8, getitem_56);  getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_10: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(add_52, primals_31, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_53: "i64[]" = torch.ops.aten.add.Tensor(primals_545, 1)
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 26, 1, 1]" = var_mean_10[0]
    getitem_59: "f32[1, 26, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_54: "f32[1, 26, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_10: "f32[1, 26, 1, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_10: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_59)
    mul_70: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_31: "f32[26]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_71: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_72: "f32[26]" = torch.ops.aten.mul.Tensor(primals_543, 0.9)
    add_55: "f32[26]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_32: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_73: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000398612827361);  squeeze_32 = None
    mul_74: "f32[26]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[26]" = torch.ops.aten.mul.Tensor(primals_544, 0.9)
    add_56: "f32[26]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_40: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_41: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_76: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
    unsqueeze_42: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_43: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_57: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_9: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_63: "f32[8, 26, 56, 56]" = split_with_sizes_6[3];  split_with_sizes_6 = None
    cat_1: "f32[8, 104, 56, 56]" = torch.ops.aten.cat.default([relu_7, relu_8, relu_9, getitem_63], 1);  getitem_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_11: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_1, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_58: "i64[]" = torch.ops.aten.add.Tensor(primals_548, 1)
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 256, 1, 1]" = var_mean_11[0]
    getitem_65: "f32[1, 256, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_59: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_11: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_11: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_65)
    mul_77: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_34: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_78: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_79: "f32[256]" = torch.ops.aten.mul.Tensor(primals_546, 0.9)
    add_60: "f32[256]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_35: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_80: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000398612827361);  squeeze_35 = None
    mul_81: "f32[256]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[256]" = torch.ops.aten.mul.Tensor(primals_547, 0.9)
    add_61: "f32[256]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_83: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
    unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_62: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_63: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_62, relu_5);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_10: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_63);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_12: "f32[8, 104, 56, 56]" = torch.ops.aten.convolution.default(relu_10, primals_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_64: "i64[]" = torch.ops.aten.add.Tensor(primals_551, 1)
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 104, 1, 1]" = var_mean_12[0]
    getitem_67: "f32[1, 104, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_65: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_12: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_12: "f32[8, 104, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_67)
    mul_84: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_37: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_85: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_86: "f32[104]" = torch.ops.aten.mul.Tensor(primals_549, 0.9)
    add_66: "f32[104]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_38: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_87: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000398612827361);  squeeze_38 = None
    mul_88: "f32[104]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[104]" = torch.ops.aten.mul.Tensor(primals_550, 0.9)
    add_67: "f32[104]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_48: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1)
    unsqueeze_49: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_90: "f32[8, 104, 56, 56]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
    unsqueeze_50: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
    unsqueeze_51: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_68: "f32[8, 104, 56, 56]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_11: "f32[8, 104, 56, 56]" = torch.ops.aten.relu.default(add_68);  add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(relu_11, [26, 26, 26, 26], 1)
    getitem_72: "f32[8, 26, 56, 56]" = split_with_sizes_11[0]
    convolution_13: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(getitem_72, primals_40, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_69: "i64[]" = torch.ops.aten.add.Tensor(primals_554, 1)
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 26, 1, 1]" = var_mean_13[0]
    getitem_77: "f32[1, 26, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_70: "f32[1, 26, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_13: "f32[1, 26, 1, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_13: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_77)
    mul_91: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_40: "f32[26]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_92: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_93: "f32[26]" = torch.ops.aten.mul.Tensor(primals_552, 0.9)
    add_71: "f32[26]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    squeeze_41: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_94: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000398612827361);  squeeze_41 = None
    mul_95: "f32[26]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
    mul_96: "f32[26]" = torch.ops.aten.mul.Tensor(primals_553, 0.9)
    add_72: "f32[26]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    unsqueeze_52: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_53: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_97: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
    unsqueeze_54: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_55: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_73: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_12: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_73);  add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_79: "f32[8, 26, 56, 56]" = split_with_sizes_11[1]
    add_74: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(relu_12, getitem_79);  getitem_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_14: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(add_74, primals_43, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_75: "i64[]" = torch.ops.aten.add.Tensor(primals_557, 1)
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 26, 1, 1]" = var_mean_14[0]
    getitem_83: "f32[1, 26, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_76: "f32[1, 26, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_14: "f32[1, 26, 1, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_14: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_83)
    mul_98: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_43: "f32[26]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_99: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_100: "f32[26]" = torch.ops.aten.mul.Tensor(primals_555, 0.9)
    add_77: "f32[26]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_44: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_101: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000398612827361);  squeeze_44 = None
    mul_102: "f32[26]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[26]" = torch.ops.aten.mul.Tensor(primals_556, 0.9)
    add_78: "f32[26]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_56: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
    unsqueeze_57: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_104: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
    unsqueeze_58: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
    unsqueeze_59: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_79: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_13: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_79);  add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_86: "f32[8, 26, 56, 56]" = split_with_sizes_11[2]
    add_80: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(relu_13, getitem_86);  getitem_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_15: "f32[8, 26, 56, 56]" = torch.ops.aten.convolution.default(add_80, primals_46, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_81: "i64[]" = torch.ops.aten.add.Tensor(primals_560, 1)
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 26, 1, 1]" = var_mean_15[0]
    getitem_89: "f32[1, 26, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_82: "f32[1, 26, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_15: "f32[1, 26, 1, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_15: "f32[8, 26, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_89)
    mul_105: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_46: "f32[26]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_106: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_107: "f32[26]" = torch.ops.aten.mul.Tensor(primals_558, 0.9)
    add_83: "f32[26]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_47: "f32[26]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_108: "f32[26]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000398612827361);  squeeze_47 = None
    mul_109: "f32[26]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[26]" = torch.ops.aten.mul.Tensor(primals_559, 0.9)
    add_84: "f32[26]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_60: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_61: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_111: "f32[8, 26, 56, 56]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
    unsqueeze_62: "f32[26, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_63: "f32[26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_85: "f32[8, 26, 56, 56]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_14: "f32[8, 26, 56, 56]" = torch.ops.aten.relu.default(add_85);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_93: "f32[8, 26, 56, 56]" = split_with_sizes_11[3];  split_with_sizes_11 = None
    cat_2: "f32[8, 104, 56, 56]" = torch.ops.aten.cat.default([relu_12, relu_13, relu_14, getitem_93], 1);  getitem_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_16: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(cat_2, primals_49, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_86: "i64[]" = torch.ops.aten.add.Tensor(primals_563, 1)
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 256, 1, 1]" = var_mean_16[0]
    getitem_95: "f32[1, 256, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_87: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
    rsqrt_16: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_16: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_95)
    mul_112: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_49: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_113: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_114: "f32[256]" = torch.ops.aten.mul.Tensor(primals_561, 0.9)
    add_88: "f32[256]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_50: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_115: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000398612827361);  squeeze_50 = None
    mul_116: "f32[256]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[256]" = torch.ops.aten.mul.Tensor(primals_562, 0.9)
    add_89: "f32[256]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_64: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
    unsqueeze_65: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_118: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
    unsqueeze_66: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
    unsqueeze_67: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_90: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_91: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_90, relu_10);  add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_15: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_91);  add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_17: "f32[8, 208, 56, 56]" = torch.ops.aten.convolution.default(relu_15, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_92: "i64[]" = torch.ops.aten.add.Tensor(primals_566, 1)
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 208, 1, 1]" = var_mean_17[0]
    getitem_97: "f32[1, 208, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_93: "f32[1, 208, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_17: "f32[1, 208, 1, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_17: "f32[8, 208, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_97)
    mul_119: "f32[8, 208, 56, 56]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_52: "f32[208]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_120: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_121: "f32[208]" = torch.ops.aten.mul.Tensor(primals_564, 0.9)
    add_94: "f32[208]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_53: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_122: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000398612827361);  squeeze_53 = None
    mul_123: "f32[208]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[208]" = torch.ops.aten.mul.Tensor(primals_565, 0.9)
    add_95: "f32[208]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_68: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_69: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_125: "f32[8, 208, 56, 56]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
    unsqueeze_70: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_71: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_96: "f32[8, 208, 56, 56]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_16: "f32[8, 208, 56, 56]" = torch.ops.aten.relu.default(add_96);  add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_16 = torch.ops.aten.split_with_sizes.default(relu_16, [52, 52, 52, 52], 1)
    getitem_102: "f32[8, 52, 56, 56]" = split_with_sizes_16[0]
    convolution_18: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_102, primals_55, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_97: "i64[]" = torch.ops.aten.add.Tensor(primals_569, 1)
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 52, 1, 1]" = var_mean_18[0]
    getitem_107: "f32[1, 52, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_98: "f32[1, 52, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_18: "f32[1, 52, 1, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    sub_18: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_107)
    mul_126: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_55: "f32[52]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_127: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_128: "f32[52]" = torch.ops.aten.mul.Tensor(primals_567, 0.9)
    add_99: "f32[52]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_56: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_129: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
    mul_130: "f32[52]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[52]" = torch.ops.aten.mul.Tensor(primals_568, 0.9)
    add_100: "f32[52]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_72: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_73: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_132: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
    unsqueeze_74: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_75: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_101: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_17: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_101);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    getitem_109: "f32[8, 52, 56, 56]" = split_with_sizes_16[1]
    convolution_19: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_109, primals_58, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_102: "i64[]" = torch.ops.aten.add.Tensor(primals_572, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 52, 1, 1]" = var_mean_19[0]
    getitem_113: "f32[1, 52, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_103: "f32[1, 52, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_19: "f32[1, 52, 1, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    sub_19: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_113)
    mul_133: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_58: "f32[52]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_134: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_135: "f32[52]" = torch.ops.aten.mul.Tensor(primals_570, 0.9)
    add_104: "f32[52]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_59: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_136: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
    mul_137: "f32[52]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[52]" = torch.ops.aten.mul.Tensor(primals_571, 0.9)
    add_105: "f32[52]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_76: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_77: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_139: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
    unsqueeze_78: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_79: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_106: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_18: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_106);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    getitem_116: "f32[8, 52, 56, 56]" = split_with_sizes_16[2]
    convolution_20: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_116, primals_61, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_107: "i64[]" = torch.ops.aten.add.Tensor(primals_575, 1)
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 52, 1, 1]" = var_mean_20[0]
    getitem_119: "f32[1, 52, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_108: "f32[1, 52, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_20: "f32[1, 52, 1, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_20: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_119)
    mul_140: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_61: "f32[52]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_141: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_142: "f32[52]" = torch.ops.aten.mul.Tensor(primals_573, 0.9)
    add_109: "f32[52]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_62: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_143: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
    mul_144: "f32[52]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[52]" = torch.ops.aten.mul.Tensor(primals_574, 0.9)
    add_110: "f32[52]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    unsqueeze_80: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_81: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_146: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
    unsqueeze_82: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_83: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_111: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_19: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_111);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    getitem_123: "f32[8, 52, 56, 56]" = split_with_sizes_16[3];  split_with_sizes_16 = None
    avg_pool2d_1: "f32[8, 52, 28, 28]" = torch.ops.aten.avg_pool2d.default(getitem_123, [3, 3], [2, 2], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_3: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([relu_17, relu_18, relu_19, avg_pool2d_1], 1);  avg_pool2d_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_21: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_3, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_112: "i64[]" = torch.ops.aten.add.Tensor(primals_578, 1)
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 512, 1, 1]" = var_mean_21[0]
    getitem_125: "f32[1, 512, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_113: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
    rsqrt_21: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_21: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_125)
    mul_147: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_64: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_148: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_149: "f32[512]" = torch.ops.aten.mul.Tensor(primals_576, 0.9)
    add_114: "f32[512]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_65: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_150: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
    mul_151: "f32[512]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[512]" = torch.ops.aten.mul.Tensor(primals_577, 0.9)
    add_115: "f32[512]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_84: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_85: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_153: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
    unsqueeze_86: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_87: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_116: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_22: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_15, primals_67, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    add_117: "i64[]" = torch.ops.aten.add.Tensor(primals_581, 1)
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 512, 1, 1]" = var_mean_22[0]
    getitem_127: "f32[1, 512, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_118: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
    rsqrt_22: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_22: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_127)
    mul_154: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_67: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_155: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_156: "f32[512]" = torch.ops.aten.mul.Tensor(primals_579, 0.9)
    add_119: "f32[512]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    squeeze_68: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_157: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001594642002871);  squeeze_68 = None
    mul_158: "f32[512]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
    mul_159: "f32[512]" = torch.ops.aten.mul.Tensor(primals_580, 0.9)
    add_120: "f32[512]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    unsqueeze_88: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_89: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_160: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
    unsqueeze_90: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_91: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_121: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_122: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_116, add_121);  add_116 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_20: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_122);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_23: "f32[8, 208, 28, 28]" = torch.ops.aten.convolution.default(relu_20, primals_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_123: "i64[]" = torch.ops.aten.add.Tensor(primals_584, 1)
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[1, 208, 1, 1]" = var_mean_23[0]
    getitem_129: "f32[1, 208, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_124: "f32[1, 208, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
    rsqrt_23: "f32[1, 208, 1, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    sub_23: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_129)
    mul_161: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
    squeeze_70: "f32[208]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_162: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_163: "f32[208]" = torch.ops.aten.mul.Tensor(primals_582, 0.9)
    add_125: "f32[208]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    squeeze_71: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
    mul_164: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001594642002871);  squeeze_71 = None
    mul_165: "f32[208]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
    mul_166: "f32[208]" = torch.ops.aten.mul.Tensor(primals_583, 0.9)
    add_126: "f32[208]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    unsqueeze_92: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_93: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_167: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
    unsqueeze_94: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_95: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_127: "f32[8, 208, 28, 28]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_21: "f32[8, 208, 28, 28]" = torch.ops.aten.relu.default(add_127);  add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_21 = torch.ops.aten.split_with_sizes.default(relu_21, [52, 52, 52, 52], 1)
    getitem_134: "f32[8, 52, 28, 28]" = split_with_sizes_21[0]
    convolution_24: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_134, primals_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_128: "i64[]" = torch.ops.aten.add.Tensor(primals_587, 1)
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_138: "f32[1, 52, 1, 1]" = var_mean_24[0]
    getitem_139: "f32[1, 52, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_129: "f32[1, 52, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05)
    rsqrt_24: "f32[1, 52, 1, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_24: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_139)
    mul_168: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
    squeeze_73: "f32[52]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_169: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_170: "f32[52]" = torch.ops.aten.mul.Tensor(primals_585, 0.9)
    add_130: "f32[52]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_74: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
    mul_171: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001594642002871);  squeeze_74 = None
    mul_172: "f32[52]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[52]" = torch.ops.aten.mul.Tensor(primals_586, 0.9)
    add_131: "f32[52]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_96: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_97: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_174: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
    unsqueeze_98: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_99: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_132: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_22: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_132);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_141: "f32[8, 52, 28, 28]" = split_with_sizes_21[1]
    add_133: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_22, getitem_141);  getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_25: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_133, primals_76, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_134: "i64[]" = torch.ops.aten.add.Tensor(primals_590, 1)
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_144: "f32[1, 52, 1, 1]" = var_mean_25[0]
    getitem_145: "f32[1, 52, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_135: "f32[1, 52, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05)
    rsqrt_25: "f32[1, 52, 1, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    sub_25: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_145)
    mul_175: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
    squeeze_76: "f32[52]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_176: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_177: "f32[52]" = torch.ops.aten.mul.Tensor(primals_588, 0.9)
    add_136: "f32[52]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_77: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
    mul_178: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001594642002871);  squeeze_77 = None
    mul_179: "f32[52]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[52]" = torch.ops.aten.mul.Tensor(primals_589, 0.9)
    add_137: "f32[52]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_100: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_101: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_181: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
    unsqueeze_102: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_103: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_138: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_23: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_138);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_148: "f32[8, 52, 28, 28]" = split_with_sizes_21[2]
    add_139: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_23, getitem_148);  getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_26: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_139, primals_79, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_140: "i64[]" = torch.ops.aten.add.Tensor(primals_593, 1)
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_150: "f32[1, 52, 1, 1]" = var_mean_26[0]
    getitem_151: "f32[1, 52, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_141: "f32[1, 52, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
    rsqrt_26: "f32[1, 52, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_26: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_151)
    mul_182: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
    squeeze_79: "f32[52]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_183: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_184: "f32[52]" = torch.ops.aten.mul.Tensor(primals_591, 0.9)
    add_142: "f32[52]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_80: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
    mul_185: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001594642002871);  squeeze_80 = None
    mul_186: "f32[52]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[52]" = torch.ops.aten.mul.Tensor(primals_592, 0.9)
    add_143: "f32[52]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_104: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_105: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_188: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
    unsqueeze_106: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_107: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_144: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_24: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_144);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_155: "f32[8, 52, 28, 28]" = split_with_sizes_21[3];  split_with_sizes_21 = None
    cat_4: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([relu_22, relu_23, relu_24, getitem_155], 1);  getitem_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_27: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_4, primals_82, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_145: "i64[]" = torch.ops.aten.add.Tensor(primals_596, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_156: "f32[1, 512, 1, 1]" = var_mean_27[0]
    getitem_157: "f32[1, 512, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_146: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
    rsqrt_27: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_27: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_157)
    mul_189: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
    squeeze_82: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_190: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_191: "f32[512]" = torch.ops.aten.mul.Tensor(primals_594, 0.9)
    add_147: "f32[512]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    squeeze_83: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
    mul_192: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0001594642002871);  squeeze_83 = None
    mul_193: "f32[512]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
    mul_194: "f32[512]" = torch.ops.aten.mul.Tensor(primals_595, 0.9)
    add_148: "f32[512]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    unsqueeze_108: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_109: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_195: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
    unsqueeze_110: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_111: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_149: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_150: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_149, relu_20);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_25: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_150);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_28: "f32[8, 208, 28, 28]" = torch.ops.aten.convolution.default(relu_25, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_151: "i64[]" = torch.ops.aten.add.Tensor(primals_599, 1)
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_158: "f32[1, 208, 1, 1]" = var_mean_28[0]
    getitem_159: "f32[1, 208, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_152: "f32[1, 208, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
    rsqrt_28: "f32[1, 208, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_28: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_159)
    mul_196: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
    squeeze_85: "f32[208]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_197: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_198: "f32[208]" = torch.ops.aten.mul.Tensor(primals_597, 0.9)
    add_153: "f32[208]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_86: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
    mul_199: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0001594642002871);  squeeze_86 = None
    mul_200: "f32[208]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[208]" = torch.ops.aten.mul.Tensor(primals_598, 0.9)
    add_154: "f32[208]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_112: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_113: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_202: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
    unsqueeze_114: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_115: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_155: "f32[8, 208, 28, 28]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_26: "f32[8, 208, 28, 28]" = torch.ops.aten.relu.default(add_155);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_26 = torch.ops.aten.split_with_sizes.default(relu_26, [52, 52, 52, 52], 1)
    getitem_164: "f32[8, 52, 28, 28]" = split_with_sizes_26[0]
    convolution_29: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_164, primals_88, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_156: "i64[]" = torch.ops.aten.add.Tensor(primals_602, 1)
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_168: "f32[1, 52, 1, 1]" = var_mean_29[0]
    getitem_169: "f32[1, 52, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_157: "f32[1, 52, 1, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-05)
    rsqrt_29: "f32[1, 52, 1, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_29: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_169)
    mul_203: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
    squeeze_88: "f32[52]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_204: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_205: "f32[52]" = torch.ops.aten.mul.Tensor(primals_600, 0.9)
    add_158: "f32[52]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_89: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
    mul_206: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0001594642002871);  squeeze_89 = None
    mul_207: "f32[52]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[52]" = torch.ops.aten.mul.Tensor(primals_601, 0.9)
    add_159: "f32[52]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_116: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_117: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_209: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
    unsqueeze_118: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_119: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_160: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_27: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_160);  add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_171: "f32[8, 52, 28, 28]" = split_with_sizes_26[1]
    add_161: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_27, getitem_171);  getitem_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_30: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_161, primals_91, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_162: "i64[]" = torch.ops.aten.add.Tensor(primals_605, 1)
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_174: "f32[1, 52, 1, 1]" = var_mean_30[0]
    getitem_175: "f32[1, 52, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_163: "f32[1, 52, 1, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05)
    rsqrt_30: "f32[1, 52, 1, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_30: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_175)
    mul_210: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
    squeeze_91: "f32[52]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_211: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_212: "f32[52]" = torch.ops.aten.mul.Tensor(primals_603, 0.9)
    add_164: "f32[52]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    squeeze_92: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_174, [0, 2, 3]);  getitem_174 = None
    mul_213: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0001594642002871);  squeeze_92 = None
    mul_214: "f32[52]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
    mul_215: "f32[52]" = torch.ops.aten.mul.Tensor(primals_604, 0.9)
    add_165: "f32[52]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    unsqueeze_120: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_121: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_216: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
    unsqueeze_122: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_123: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_166: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_28: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_166);  add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_178: "f32[8, 52, 28, 28]" = split_with_sizes_26[2]
    add_167: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_28, getitem_178);  getitem_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_31: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_167, primals_94, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_168: "i64[]" = torch.ops.aten.add.Tensor(primals_608, 1)
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_180: "f32[1, 52, 1, 1]" = var_mean_31[0]
    getitem_181: "f32[1, 52, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_169: "f32[1, 52, 1, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-05)
    rsqrt_31: "f32[1, 52, 1, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    sub_31: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_181)
    mul_217: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_181, [0, 2, 3]);  getitem_181 = None
    squeeze_94: "f32[52]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_218: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_219: "f32[52]" = torch.ops.aten.mul.Tensor(primals_606, 0.9)
    add_170: "f32[52]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_95: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_180, [0, 2, 3]);  getitem_180 = None
    mul_220: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0001594642002871);  squeeze_95 = None
    mul_221: "f32[52]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[52]" = torch.ops.aten.mul.Tensor(primals_607, 0.9)
    add_171: "f32[52]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    unsqueeze_124: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_125: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_223: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
    unsqueeze_126: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_127: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_172: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_29: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_172);  add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_185: "f32[8, 52, 28, 28]" = split_with_sizes_26[3];  split_with_sizes_26 = None
    cat_5: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([relu_27, relu_28, relu_29, getitem_185], 1);  getitem_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_32: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_5, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_173: "i64[]" = torch.ops.aten.add.Tensor(primals_611, 1)
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_186: "f32[1, 512, 1, 1]" = var_mean_32[0]
    getitem_187: "f32[1, 512, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_174: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05)
    rsqrt_32: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    sub_32: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_187)
    mul_224: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
    squeeze_97: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_225: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_226: "f32[512]" = torch.ops.aten.mul.Tensor(primals_609, 0.9)
    add_175: "f32[512]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_98: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_186, [0, 2, 3]);  getitem_186 = None
    mul_227: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0001594642002871);  squeeze_98 = None
    mul_228: "f32[512]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[512]" = torch.ops.aten.mul.Tensor(primals_610, 0.9)
    add_176: "f32[512]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_128: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_129: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_230: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
    unsqueeze_130: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_131: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_177: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_178: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_177, relu_25);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_30: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_178);  add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_33: "f32[8, 208, 28, 28]" = torch.ops.aten.convolution.default(relu_30, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_179: "i64[]" = torch.ops.aten.add.Tensor(primals_614, 1)
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_188: "f32[1, 208, 1, 1]" = var_mean_33[0]
    getitem_189: "f32[1, 208, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_180: "f32[1, 208, 1, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-05)
    rsqrt_33: "f32[1, 208, 1, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
    sub_33: "f32[8, 208, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_189)
    mul_231: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
    squeeze_100: "f32[208]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_232: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_233: "f32[208]" = torch.ops.aten.mul.Tensor(primals_612, 0.9)
    add_181: "f32[208]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_101: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_188, [0, 2, 3]);  getitem_188 = None
    mul_234: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0001594642002871);  squeeze_101 = None
    mul_235: "f32[208]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[208]" = torch.ops.aten.mul.Tensor(primals_613, 0.9)
    add_182: "f32[208]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_132: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_133: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_237: "f32[8, 208, 28, 28]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
    unsqueeze_134: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_135: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_183: "f32[8, 208, 28, 28]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_31: "f32[8, 208, 28, 28]" = torch.ops.aten.relu.default(add_183);  add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_31 = torch.ops.aten.split_with_sizes.default(relu_31, [52, 52, 52, 52], 1)
    getitem_194: "f32[8, 52, 28, 28]" = split_with_sizes_31[0]
    convolution_34: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(getitem_194, primals_103, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_184: "i64[]" = torch.ops.aten.add.Tensor(primals_617, 1)
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_198: "f32[1, 52, 1, 1]" = var_mean_34[0]
    getitem_199: "f32[1, 52, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_185: "f32[1, 52, 1, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05)
    rsqrt_34: "f32[1, 52, 1, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_34: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_199)
    mul_238: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_199, [0, 2, 3]);  getitem_199 = None
    squeeze_103: "f32[52]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_239: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_240: "f32[52]" = torch.ops.aten.mul.Tensor(primals_615, 0.9)
    add_186: "f32[52]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_104: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_198, [0, 2, 3]);  getitem_198 = None
    mul_241: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0001594642002871);  squeeze_104 = None
    mul_242: "f32[52]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[52]" = torch.ops.aten.mul.Tensor(primals_616, 0.9)
    add_187: "f32[52]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_136: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
    unsqueeze_137: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_244: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
    unsqueeze_138: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
    unsqueeze_139: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_188: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_32: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_188);  add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_201: "f32[8, 52, 28, 28]" = split_with_sizes_31[1]
    add_189: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_32, getitem_201);  getitem_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_35: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_189, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_190: "i64[]" = torch.ops.aten.add.Tensor(primals_620, 1)
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_204: "f32[1, 52, 1, 1]" = var_mean_35[0]
    getitem_205: "f32[1, 52, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_191: "f32[1, 52, 1, 1]" = torch.ops.aten.add.Tensor(getitem_204, 1e-05)
    rsqrt_35: "f32[1, 52, 1, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_35: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_205)
    mul_245: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_205, [0, 2, 3]);  getitem_205 = None
    squeeze_106: "f32[52]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_246: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_247: "f32[52]" = torch.ops.aten.mul.Tensor(primals_618, 0.9)
    add_192: "f32[52]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_107: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_204, [0, 2, 3]);  getitem_204 = None
    mul_248: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0001594642002871);  squeeze_107 = None
    mul_249: "f32[52]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[52]" = torch.ops.aten.mul.Tensor(primals_619, 0.9)
    add_193: "f32[52]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_140: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_141: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_251: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
    unsqueeze_142: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_143: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_194: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_33: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_194);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_208: "f32[8, 52, 28, 28]" = split_with_sizes_31[2]
    add_195: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(relu_33, getitem_208);  getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_36: "f32[8, 52, 28, 28]" = torch.ops.aten.convolution.default(add_195, primals_109, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_196: "i64[]" = torch.ops.aten.add.Tensor(primals_623, 1)
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_210: "f32[1, 52, 1, 1]" = var_mean_36[0]
    getitem_211: "f32[1, 52, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_197: "f32[1, 52, 1, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-05)
    rsqrt_36: "f32[1, 52, 1, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    sub_36: "f32[8, 52, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_211)
    mul_252: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_211, [0, 2, 3]);  getitem_211 = None
    squeeze_109: "f32[52]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_253: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_254: "f32[52]" = torch.ops.aten.mul.Tensor(primals_621, 0.9)
    add_198: "f32[52]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    squeeze_110: "f32[52]" = torch.ops.aten.squeeze.dims(getitem_210, [0, 2, 3]);  getitem_210 = None
    mul_255: "f32[52]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0001594642002871);  squeeze_110 = None
    mul_256: "f32[52]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
    mul_257: "f32[52]" = torch.ops.aten.mul.Tensor(primals_622, 0.9)
    add_199: "f32[52]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    unsqueeze_144: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_145: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_258: "f32[8, 52, 28, 28]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
    unsqueeze_146: "f32[52, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_147: "f32[52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_200: "f32[8, 52, 28, 28]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_34: "f32[8, 52, 28, 28]" = torch.ops.aten.relu.default(add_200);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_215: "f32[8, 52, 28, 28]" = split_with_sizes_31[3];  split_with_sizes_31 = None
    cat_6: "f32[8, 208, 28, 28]" = torch.ops.aten.cat.default([relu_32, relu_33, relu_34, getitem_215], 1);  getitem_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_37: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(cat_6, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_201: "i64[]" = torch.ops.aten.add.Tensor(primals_626, 1)
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_216: "f32[1, 512, 1, 1]" = var_mean_37[0]
    getitem_217: "f32[1, 512, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_202: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_216, 1e-05)
    rsqrt_37: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    sub_37: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_217)
    mul_259: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_217, [0, 2, 3]);  getitem_217 = None
    squeeze_112: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_260: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_261: "f32[512]" = torch.ops.aten.mul.Tensor(primals_624, 0.9)
    add_203: "f32[512]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    squeeze_113: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_216, [0, 2, 3]);  getitem_216 = None
    mul_262: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0001594642002871);  squeeze_113 = None
    mul_263: "f32[512]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
    mul_264: "f32[512]" = torch.ops.aten.mul.Tensor(primals_625, 0.9)
    add_204: "f32[512]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    unsqueeze_148: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_149: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_265: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
    unsqueeze_150: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_151: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_205: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_206: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_205, relu_30);  add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_35: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_206);  add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_38: "f32[8, 416, 28, 28]" = torch.ops.aten.convolution.default(relu_35, primals_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_207: "i64[]" = torch.ops.aten.add.Tensor(primals_629, 1)
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_218: "f32[1, 416, 1, 1]" = var_mean_38[0]
    getitem_219: "f32[1, 416, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_208: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-05)
    rsqrt_38: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_38: "f32[8, 416, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_219)
    mul_266: "f32[8, 416, 28, 28]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_219, [0, 2, 3]);  getitem_219 = None
    squeeze_115: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_267: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_268: "f32[416]" = torch.ops.aten.mul.Tensor(primals_627, 0.9)
    add_209: "f32[416]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    squeeze_116: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_218, [0, 2, 3]);  getitem_218 = None
    mul_269: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0001594642002871);  squeeze_116 = None
    mul_270: "f32[416]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
    mul_271: "f32[416]" = torch.ops.aten.mul.Tensor(primals_628, 0.9)
    add_210: "f32[416]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    unsqueeze_152: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
    unsqueeze_153: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_272: "f32[8, 416, 28, 28]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
    unsqueeze_154: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
    unsqueeze_155: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_211: "f32[8, 416, 28, 28]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_36: "f32[8, 416, 28, 28]" = torch.ops.aten.relu.default(add_211);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_36 = torch.ops.aten.split_with_sizes.default(relu_36, [104, 104, 104, 104], 1)
    getitem_224: "f32[8, 104, 28, 28]" = split_with_sizes_36[0]
    convolution_39: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_224, primals_118, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_632, 1)
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_228: "f32[1, 104, 1, 1]" = var_mean_39[0]
    getitem_229: "f32[1, 104, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_213: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_228, 1e-05)
    rsqrt_39: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_39: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_229)
    mul_273: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_229, [0, 2, 3]);  getitem_229 = None
    squeeze_118: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_274: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_275: "f32[104]" = torch.ops.aten.mul.Tensor(primals_630, 0.9)
    add_214: "f32[104]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    squeeze_119: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_228, [0, 2, 3]);  getitem_228 = None
    mul_276: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0006381620931717);  squeeze_119 = None
    mul_277: "f32[104]" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
    mul_278: "f32[104]" = torch.ops.aten.mul.Tensor(primals_631, 0.9)
    add_215: "f32[104]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    unsqueeze_156: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_157: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_279: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
    unsqueeze_158: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_159: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_216: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_37: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_216);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    getitem_231: "f32[8, 104, 28, 28]" = split_with_sizes_36[1]
    convolution_40: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_231, primals_121, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_217: "i64[]" = torch.ops.aten.add.Tensor(primals_635, 1)
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_234: "f32[1, 104, 1, 1]" = var_mean_40[0]
    getitem_235: "f32[1, 104, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_218: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_234, 1e-05)
    rsqrt_40: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    sub_40: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_235)
    mul_280: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_235, [0, 2, 3]);  getitem_235 = None
    squeeze_121: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_281: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_282: "f32[104]" = torch.ops.aten.mul.Tensor(primals_633, 0.9)
    add_219: "f32[104]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    squeeze_122: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_234, [0, 2, 3]);  getitem_234 = None
    mul_283: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0006381620931717);  squeeze_122 = None
    mul_284: "f32[104]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
    mul_285: "f32[104]" = torch.ops.aten.mul.Tensor(primals_634, 0.9)
    add_220: "f32[104]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_160: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
    unsqueeze_161: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_286: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
    unsqueeze_162: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
    unsqueeze_163: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_221: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_38: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_221);  add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    getitem_238: "f32[8, 104, 28, 28]" = split_with_sizes_36[2]
    convolution_41: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_238, primals_124, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_222: "i64[]" = torch.ops.aten.add.Tensor(primals_638, 1)
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_240: "f32[1, 104, 1, 1]" = var_mean_41[0]
    getitem_241: "f32[1, 104, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_223: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-05)
    rsqrt_41: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
    sub_41: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_241)
    mul_287: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_241, [0, 2, 3]);  getitem_241 = None
    squeeze_124: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_288: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_289: "f32[104]" = torch.ops.aten.mul.Tensor(primals_636, 0.9)
    add_224: "f32[104]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    squeeze_125: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_240, [0, 2, 3]);  getitem_240 = None
    mul_290: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0006381620931717);  squeeze_125 = None
    mul_291: "f32[104]" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
    mul_292: "f32[104]" = torch.ops.aten.mul.Tensor(primals_637, 0.9)
    add_225: "f32[104]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    unsqueeze_164: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_165: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_293: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
    unsqueeze_166: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_167: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_226: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_39: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_226);  add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    getitem_245: "f32[8, 104, 28, 28]" = split_with_sizes_36[3];  split_with_sizes_36 = None
    avg_pool2d_2: "f32[8, 104, 14, 14]" = torch.ops.aten.avg_pool2d.default(getitem_245, [3, 3], [2, 2], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_7: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_37, relu_38, relu_39, avg_pool2d_2], 1);  avg_pool2d_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_42: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_7, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_227: "i64[]" = torch.ops.aten.add.Tensor(primals_641, 1)
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_246: "f32[1, 1024, 1, 1]" = var_mean_42[0]
    getitem_247: "f32[1, 1024, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_228: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-05)
    rsqrt_42: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
    sub_42: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_247)
    mul_294: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_247, [0, 2, 3]);  getitem_247 = None
    squeeze_127: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_295: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_296: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_639, 0.9)
    add_229: "f32[1024]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_128: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_246, [0, 2, 3]);  getitem_246 = None
    mul_297: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0006381620931717);  squeeze_128 = None
    mul_298: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_640, 0.9)
    add_230: "f32[1024]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_168: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
    unsqueeze_169: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_300: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
    unsqueeze_170: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
    unsqueeze_171: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_231: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_43: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_35, primals_130, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    add_232: "i64[]" = torch.ops.aten.add.Tensor(primals_644, 1)
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_248: "f32[1, 1024, 1, 1]" = var_mean_43[0]
    getitem_249: "f32[1, 1024, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_233: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_248, 1e-05)
    rsqrt_43: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_233);  add_233 = None
    sub_43: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_249)
    mul_301: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_249, [0, 2, 3]);  getitem_249 = None
    squeeze_130: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_302: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_303: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_642, 0.9)
    add_234: "f32[1024]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    squeeze_131: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_248, [0, 2, 3]);  getitem_248 = None
    mul_304: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0006381620931717);  squeeze_131 = None
    mul_305: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
    mul_306: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_643, 0.9)
    add_235: "f32[1024]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_172: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_173: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_307: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
    unsqueeze_174: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_175: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_236: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_237: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_231, add_236);  add_231 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_40: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_237);  add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_44: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_40, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_238: "i64[]" = torch.ops.aten.add.Tensor(primals_647, 1)
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_250: "f32[1, 416, 1, 1]" = var_mean_44[0]
    getitem_251: "f32[1, 416, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_239: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_250, 1e-05)
    rsqrt_44: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
    sub_44: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_251)
    mul_308: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_251, [0, 2, 3]);  getitem_251 = None
    squeeze_133: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_309: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_310: "f32[416]" = torch.ops.aten.mul.Tensor(primals_645, 0.9)
    add_240: "f32[416]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    squeeze_134: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_250, [0, 2, 3]);  getitem_250 = None
    mul_311: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0006381620931717);  squeeze_134 = None
    mul_312: "f32[416]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
    mul_313: "f32[416]" = torch.ops.aten.mul.Tensor(primals_646, 0.9)
    add_241: "f32[416]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_176: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
    unsqueeze_177: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_314: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
    unsqueeze_178: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
    unsqueeze_179: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_242: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_41: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_242);  add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_41 = torch.ops.aten.split_with_sizes.default(relu_41, [104, 104, 104, 104], 1)
    getitem_256: "f32[8, 104, 14, 14]" = split_with_sizes_41[0]
    convolution_45: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_256, primals_136, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_243: "i64[]" = torch.ops.aten.add.Tensor(primals_650, 1)
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_260: "f32[1, 104, 1, 1]" = var_mean_45[0]
    getitem_261: "f32[1, 104, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_244: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_260, 1e-05)
    rsqrt_45: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
    sub_45: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_261)
    mul_315: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_261, [0, 2, 3]);  getitem_261 = None
    squeeze_136: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_316: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_317: "f32[104]" = torch.ops.aten.mul.Tensor(primals_648, 0.9)
    add_245: "f32[104]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    squeeze_137: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_260, [0, 2, 3]);  getitem_260 = None
    mul_318: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0006381620931717);  squeeze_137 = None
    mul_319: "f32[104]" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
    mul_320: "f32[104]" = torch.ops.aten.mul.Tensor(primals_649, 0.9)
    add_246: "f32[104]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    unsqueeze_180: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
    unsqueeze_181: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_321: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
    unsqueeze_182: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
    unsqueeze_183: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_247: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_42: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_247);  add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_263: "f32[8, 104, 14, 14]" = split_with_sizes_41[1]
    add_248: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_42, getitem_263);  getitem_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_46: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_248, primals_139, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_249: "i64[]" = torch.ops.aten.add.Tensor(primals_653, 1)
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_266: "f32[1, 104, 1, 1]" = var_mean_46[0]
    getitem_267: "f32[1, 104, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_250: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-05)
    rsqrt_46: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
    sub_46: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_267)
    mul_322: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_267, [0, 2, 3]);  getitem_267 = None
    squeeze_139: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_323: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_324: "f32[104]" = torch.ops.aten.mul.Tensor(primals_651, 0.9)
    add_251: "f32[104]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    squeeze_140: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_266, [0, 2, 3]);  getitem_266 = None
    mul_325: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0006381620931717);  squeeze_140 = None
    mul_326: "f32[104]" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
    mul_327: "f32[104]" = torch.ops.aten.mul.Tensor(primals_652, 0.9)
    add_252: "f32[104]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    unsqueeze_184: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
    unsqueeze_185: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_328: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
    unsqueeze_186: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
    unsqueeze_187: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_253: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_43: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_253);  add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_270: "f32[8, 104, 14, 14]" = split_with_sizes_41[2]
    add_254: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_43, getitem_270);  getitem_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_47: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_254, primals_142, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_255: "i64[]" = torch.ops.aten.add.Tensor(primals_656, 1)
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_272: "f32[1, 104, 1, 1]" = var_mean_47[0]
    getitem_273: "f32[1, 104, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_256: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_272, 1e-05)
    rsqrt_47: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
    sub_47: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_273)
    mul_329: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_273, [0, 2, 3]);  getitem_273 = None
    squeeze_142: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_330: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_331: "f32[104]" = torch.ops.aten.mul.Tensor(primals_654, 0.9)
    add_257: "f32[104]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    squeeze_143: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_272, [0, 2, 3]);  getitem_272 = None
    mul_332: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0006381620931717);  squeeze_143 = None
    mul_333: "f32[104]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
    mul_334: "f32[104]" = torch.ops.aten.mul.Tensor(primals_655, 0.9)
    add_258: "f32[104]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    unsqueeze_188: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_189: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_335: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
    unsqueeze_190: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_191: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_259: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_44: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_259);  add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_277: "f32[8, 104, 14, 14]" = split_with_sizes_41[3];  split_with_sizes_41 = None
    cat_8: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_42, relu_43, relu_44, getitem_277], 1);  getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_48: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_8, primals_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_260: "i64[]" = torch.ops.aten.add.Tensor(primals_659, 1)
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_278: "f32[1, 1024, 1, 1]" = var_mean_48[0]
    getitem_279: "f32[1, 1024, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_261: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_278, 1e-05)
    rsqrt_48: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
    sub_48: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_279)
    mul_336: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_279, [0, 2, 3]);  getitem_279 = None
    squeeze_145: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_337: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_338: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_657, 0.9)
    add_262: "f32[1024]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    squeeze_146: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_278, [0, 2, 3]);  getitem_278 = None
    mul_339: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0006381620931717);  squeeze_146 = None
    mul_340: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
    mul_341: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_658, 0.9)
    add_263: "f32[1024]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    unsqueeze_192: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1)
    unsqueeze_193: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_342: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
    unsqueeze_194: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
    unsqueeze_195: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_264: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_265: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_264, relu_40);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_45: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_265);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_49: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_45, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_266: "i64[]" = torch.ops.aten.add.Tensor(primals_662, 1)
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_280: "f32[1, 416, 1, 1]" = var_mean_49[0]
    getitem_281: "f32[1, 416, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_267: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_280, 1e-05)
    rsqrt_49: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
    sub_49: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_281)
    mul_343: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_281, [0, 2, 3]);  getitem_281 = None
    squeeze_148: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_344: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_345: "f32[416]" = torch.ops.aten.mul.Tensor(primals_660, 0.9)
    add_268: "f32[416]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    squeeze_149: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_280, [0, 2, 3]);  getitem_280 = None
    mul_346: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0006381620931717);  squeeze_149 = None
    mul_347: "f32[416]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
    mul_348: "f32[416]" = torch.ops.aten.mul.Tensor(primals_661, 0.9)
    add_269: "f32[416]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    unsqueeze_196: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1)
    unsqueeze_197: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_349: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
    unsqueeze_198: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
    unsqueeze_199: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_270: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_46: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_270);  add_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_46 = torch.ops.aten.split_with_sizes.default(relu_46, [104, 104, 104, 104], 1)
    getitem_286: "f32[8, 104, 14, 14]" = split_with_sizes_46[0]
    convolution_50: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_286, primals_151, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_271: "i64[]" = torch.ops.aten.add.Tensor(primals_665, 1)
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_290: "f32[1, 104, 1, 1]" = var_mean_50[0]
    getitem_291: "f32[1, 104, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_272: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_290, 1e-05)
    rsqrt_50: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
    sub_50: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_291)
    mul_350: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_291, [0, 2, 3]);  getitem_291 = None
    squeeze_151: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_351: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_352: "f32[104]" = torch.ops.aten.mul.Tensor(primals_663, 0.9)
    add_273: "f32[104]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    squeeze_152: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_290, [0, 2, 3]);  getitem_290 = None
    mul_353: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0006381620931717);  squeeze_152 = None
    mul_354: "f32[104]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
    mul_355: "f32[104]" = torch.ops.aten.mul.Tensor(primals_664, 0.9)
    add_274: "f32[104]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_200: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1)
    unsqueeze_201: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_356: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
    unsqueeze_202: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
    unsqueeze_203: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_275: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_47: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_275);  add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_293: "f32[8, 104, 14, 14]" = split_with_sizes_46[1]
    add_276: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_47, getitem_293);  getitem_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_51: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_276, primals_154, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_277: "i64[]" = torch.ops.aten.add.Tensor(primals_668, 1)
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_296: "f32[1, 104, 1, 1]" = var_mean_51[0]
    getitem_297: "f32[1, 104, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_278: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_296, 1e-05)
    rsqrt_51: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
    sub_51: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_297)
    mul_357: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_297, [0, 2, 3]);  getitem_297 = None
    squeeze_154: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_358: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_359: "f32[104]" = torch.ops.aten.mul.Tensor(primals_666, 0.9)
    add_279: "f32[104]" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    squeeze_155: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_296, [0, 2, 3]);  getitem_296 = None
    mul_360: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0006381620931717);  squeeze_155 = None
    mul_361: "f32[104]" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
    mul_362: "f32[104]" = torch.ops.aten.mul.Tensor(primals_667, 0.9)
    add_280: "f32[104]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    unsqueeze_204: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
    unsqueeze_205: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_363: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
    unsqueeze_206: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
    unsqueeze_207: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_281: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_48: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_281);  add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_300: "f32[8, 104, 14, 14]" = split_with_sizes_46[2]
    add_282: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_48, getitem_300);  getitem_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_52: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_282, primals_157, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_283: "i64[]" = torch.ops.aten.add.Tensor(primals_671, 1)
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_302: "f32[1, 104, 1, 1]" = var_mean_52[0]
    getitem_303: "f32[1, 104, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_284: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_302, 1e-05)
    rsqrt_52: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_284);  add_284 = None
    sub_52: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_303)
    mul_364: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_303, [0, 2, 3]);  getitem_303 = None
    squeeze_157: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_365: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_366: "f32[104]" = torch.ops.aten.mul.Tensor(primals_669, 0.9)
    add_285: "f32[104]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    squeeze_158: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_302, [0, 2, 3]);  getitem_302 = None
    mul_367: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0006381620931717);  squeeze_158 = None
    mul_368: "f32[104]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
    mul_369: "f32[104]" = torch.ops.aten.mul.Tensor(primals_670, 0.9)
    add_286: "f32[104]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    unsqueeze_208: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_158, -1)
    unsqueeze_209: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_370: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
    unsqueeze_210: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1);  primals_159 = None
    unsqueeze_211: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_287: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_49: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_287);  add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_307: "f32[8, 104, 14, 14]" = split_with_sizes_46[3];  split_with_sizes_46 = None
    cat_9: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_47, relu_48, relu_49, getitem_307], 1);  getitem_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_53: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_9, primals_160, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_288: "i64[]" = torch.ops.aten.add.Tensor(primals_674, 1)
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_308: "f32[1, 1024, 1, 1]" = var_mean_53[0]
    getitem_309: "f32[1, 1024, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_289: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_308, 1e-05)
    rsqrt_53: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
    sub_53: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_309)
    mul_371: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_309, [0, 2, 3]);  getitem_309 = None
    squeeze_160: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_372: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_373: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_672, 0.9)
    add_290: "f32[1024]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    squeeze_161: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_308, [0, 2, 3]);  getitem_308 = None
    mul_374: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0006381620931717);  squeeze_161 = None
    mul_375: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
    mul_376: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_673, 0.9)
    add_291: "f32[1024]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_212: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_161, -1)
    unsqueeze_213: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_377: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
    unsqueeze_214: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1);  primals_162 = None
    unsqueeze_215: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_292: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_293: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_292, relu_45);  add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_50: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_293);  add_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_54: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_50, primals_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_294: "i64[]" = torch.ops.aten.add.Tensor(primals_677, 1)
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_310: "f32[1, 416, 1, 1]" = var_mean_54[0]
    getitem_311: "f32[1, 416, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_295: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_310, 1e-05)
    rsqrt_54: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_295);  add_295 = None
    sub_54: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_311)
    mul_378: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_311, [0, 2, 3]);  getitem_311 = None
    squeeze_163: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_379: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_380: "f32[416]" = torch.ops.aten.mul.Tensor(primals_675, 0.9)
    add_296: "f32[416]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    squeeze_164: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_310, [0, 2, 3]);  getitem_310 = None
    mul_381: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0006381620931717);  squeeze_164 = None
    mul_382: "f32[416]" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
    mul_383: "f32[416]" = torch.ops.aten.mul.Tensor(primals_676, 0.9)
    add_297: "f32[416]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    unsqueeze_216: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_164, -1)
    unsqueeze_217: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_384: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
    unsqueeze_218: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_165, -1);  primals_165 = None
    unsqueeze_219: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_298: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_51: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_298);  add_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_51 = torch.ops.aten.split_with_sizes.default(relu_51, [104, 104, 104, 104], 1)
    getitem_316: "f32[8, 104, 14, 14]" = split_with_sizes_51[0]
    convolution_55: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_316, primals_166, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_299: "i64[]" = torch.ops.aten.add.Tensor(primals_680, 1)
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_320: "f32[1, 104, 1, 1]" = var_mean_55[0]
    getitem_321: "f32[1, 104, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_300: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_320, 1e-05)
    rsqrt_55: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_300);  add_300 = None
    sub_55: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_321)
    mul_385: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_321, [0, 2, 3]);  getitem_321 = None
    squeeze_166: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_386: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_387: "f32[104]" = torch.ops.aten.mul.Tensor(primals_678, 0.9)
    add_301: "f32[104]" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
    squeeze_167: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_320, [0, 2, 3]);  getitem_320 = None
    mul_388: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0006381620931717);  squeeze_167 = None
    mul_389: "f32[104]" = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
    mul_390: "f32[104]" = torch.ops.aten.mul.Tensor(primals_679, 0.9)
    add_302: "f32[104]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    unsqueeze_220: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1)
    unsqueeze_221: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_391: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
    unsqueeze_222: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1);  primals_168 = None
    unsqueeze_223: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_303: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_52: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_303);  add_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_323: "f32[8, 104, 14, 14]" = split_with_sizes_51[1]
    add_304: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_52, getitem_323);  getitem_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_56: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_304, primals_169, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_305: "i64[]" = torch.ops.aten.add.Tensor(primals_683, 1)
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_326: "f32[1, 104, 1, 1]" = var_mean_56[0]
    getitem_327: "f32[1, 104, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_306: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_326, 1e-05)
    rsqrt_56: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
    sub_56: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_327)
    mul_392: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_327, [0, 2, 3]);  getitem_327 = None
    squeeze_169: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_393: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_394: "f32[104]" = torch.ops.aten.mul.Tensor(primals_681, 0.9)
    add_307: "f32[104]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    squeeze_170: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_326, [0, 2, 3]);  getitem_326 = None
    mul_395: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0006381620931717);  squeeze_170 = None
    mul_396: "f32[104]" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
    mul_397: "f32[104]" = torch.ops.aten.mul.Tensor(primals_682, 0.9)
    add_308: "f32[104]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    unsqueeze_224: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_170, -1)
    unsqueeze_225: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_398: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_225);  mul_392 = unsqueeze_225 = None
    unsqueeze_226: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_171, -1);  primals_171 = None
    unsqueeze_227: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_309: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_227);  mul_398 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_53: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_309);  add_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_330: "f32[8, 104, 14, 14]" = split_with_sizes_51[2]
    add_310: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_53, getitem_330);  getitem_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_57: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_310, primals_172, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_311: "i64[]" = torch.ops.aten.add.Tensor(primals_686, 1)
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
    getitem_332: "f32[1, 104, 1, 1]" = var_mean_57[0]
    getitem_333: "f32[1, 104, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_312: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_332, 1e-05)
    rsqrt_57: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_312);  add_312 = None
    sub_57: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_333)
    mul_399: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_333, [0, 2, 3]);  getitem_333 = None
    squeeze_172: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_400: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_401: "f32[104]" = torch.ops.aten.mul.Tensor(primals_684, 0.9)
    add_313: "f32[104]" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
    squeeze_173: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_332, [0, 2, 3]);  getitem_332 = None
    mul_402: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0006381620931717);  squeeze_173 = None
    mul_403: "f32[104]" = torch.ops.aten.mul.Tensor(mul_402, 0.1);  mul_402 = None
    mul_404: "f32[104]" = torch.ops.aten.mul.Tensor(primals_685, 0.9)
    add_314: "f32[104]" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
    unsqueeze_228: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_173, -1)
    unsqueeze_229: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_405: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_229);  mul_399 = unsqueeze_229 = None
    unsqueeze_230: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_174, -1);  primals_174 = None
    unsqueeze_231: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_315: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_405, unsqueeze_231);  mul_405 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_54: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_315);  add_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_337: "f32[8, 104, 14, 14]" = split_with_sizes_51[3];  split_with_sizes_51 = None
    cat_10: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_52, relu_53, relu_54, getitem_337], 1);  getitem_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_58: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_10, primals_175, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_316: "i64[]" = torch.ops.aten.add.Tensor(primals_689, 1)
    var_mean_58 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_338: "f32[1, 1024, 1, 1]" = var_mean_58[0]
    getitem_339: "f32[1, 1024, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_317: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_338, 1e-05)
    rsqrt_58: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_317);  add_317 = None
    sub_58: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_339)
    mul_406: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    squeeze_174: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_339, [0, 2, 3]);  getitem_339 = None
    squeeze_175: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_407: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_408: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_687, 0.9)
    add_318: "f32[1024]" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    squeeze_176: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_338, [0, 2, 3]);  getitem_338 = None
    mul_409: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0006381620931717);  squeeze_176 = None
    mul_410: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_409, 0.1);  mul_409 = None
    mul_411: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_688, 0.9)
    add_319: "f32[1024]" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    unsqueeze_232: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_176, -1)
    unsqueeze_233: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_412: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_233);  mul_406 = unsqueeze_233 = None
    unsqueeze_234: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_177, -1);  primals_177 = None
    unsqueeze_235: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_320: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_235);  mul_412 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_321: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_320, relu_50);  add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_55: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_321);  add_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_59: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_55, primals_178, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_322: "i64[]" = torch.ops.aten.add.Tensor(primals_692, 1)
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_340: "f32[1, 416, 1, 1]" = var_mean_59[0]
    getitem_341: "f32[1, 416, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_323: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_340, 1e-05)
    rsqrt_59: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_323);  add_323 = None
    sub_59: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_341)
    mul_413: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    squeeze_177: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_341, [0, 2, 3]);  getitem_341 = None
    squeeze_178: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    mul_414: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_415: "f32[416]" = torch.ops.aten.mul.Tensor(primals_690, 0.9)
    add_324: "f32[416]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    squeeze_179: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_340, [0, 2, 3]);  getitem_340 = None
    mul_416: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0006381620931717);  squeeze_179 = None
    mul_417: "f32[416]" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
    mul_418: "f32[416]" = torch.ops.aten.mul.Tensor(primals_691, 0.9)
    add_325: "f32[416]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    unsqueeze_236: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_179, -1)
    unsqueeze_237: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_419: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_237);  mul_413 = unsqueeze_237 = None
    unsqueeze_238: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_180, -1);  primals_180 = None
    unsqueeze_239: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_326: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_239);  mul_419 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_56: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_326);  add_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_56 = torch.ops.aten.split_with_sizes.default(relu_56, [104, 104, 104, 104], 1)
    getitem_346: "f32[8, 104, 14, 14]" = split_with_sizes_56[0]
    convolution_60: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_346, primals_181, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_327: "i64[]" = torch.ops.aten.add.Tensor(primals_695, 1)
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_350: "f32[1, 104, 1, 1]" = var_mean_60[0]
    getitem_351: "f32[1, 104, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_328: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_350, 1e-05)
    rsqrt_60: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_328);  add_328 = None
    sub_60: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_351)
    mul_420: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    squeeze_180: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_351, [0, 2, 3]);  getitem_351 = None
    squeeze_181: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    mul_421: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_422: "f32[104]" = torch.ops.aten.mul.Tensor(primals_693, 0.9)
    add_329: "f32[104]" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    squeeze_182: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_350, [0, 2, 3]);  getitem_350 = None
    mul_423: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0006381620931717);  squeeze_182 = None
    mul_424: "f32[104]" = torch.ops.aten.mul.Tensor(mul_423, 0.1);  mul_423 = None
    mul_425: "f32[104]" = torch.ops.aten.mul.Tensor(primals_694, 0.9)
    add_330: "f32[104]" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
    unsqueeze_240: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_182, -1)
    unsqueeze_241: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_426: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_241);  mul_420 = unsqueeze_241 = None
    unsqueeze_242: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_183, -1);  primals_183 = None
    unsqueeze_243: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_331: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_243);  mul_426 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_57: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_331);  add_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_353: "f32[8, 104, 14, 14]" = split_with_sizes_56[1]
    add_332: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_57, getitem_353);  getitem_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_61: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_332, primals_184, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_333: "i64[]" = torch.ops.aten.add.Tensor(primals_698, 1)
    var_mean_61 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_356: "f32[1, 104, 1, 1]" = var_mean_61[0]
    getitem_357: "f32[1, 104, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_334: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_356, 1e-05)
    rsqrt_61: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_334);  add_334 = None
    sub_61: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_357)
    mul_427: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    squeeze_183: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_357, [0, 2, 3]);  getitem_357 = None
    squeeze_184: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_428: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_429: "f32[104]" = torch.ops.aten.mul.Tensor(primals_696, 0.9)
    add_335: "f32[104]" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    squeeze_185: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_356, [0, 2, 3]);  getitem_356 = None
    mul_430: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0006381620931717);  squeeze_185 = None
    mul_431: "f32[104]" = torch.ops.aten.mul.Tensor(mul_430, 0.1);  mul_430 = None
    mul_432: "f32[104]" = torch.ops.aten.mul.Tensor(primals_697, 0.9)
    add_336: "f32[104]" = torch.ops.aten.add.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
    unsqueeze_244: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_185, -1)
    unsqueeze_245: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_433: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_245);  mul_427 = unsqueeze_245 = None
    unsqueeze_246: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_186, -1);  primals_186 = None
    unsqueeze_247: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_337: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_433, unsqueeze_247);  mul_433 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_58: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_337);  add_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_360: "f32[8, 104, 14, 14]" = split_with_sizes_56[2]
    add_338: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_58, getitem_360);  getitem_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_62: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_338, primals_187, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_339: "i64[]" = torch.ops.aten.add.Tensor(primals_701, 1)
    var_mean_62 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
    getitem_362: "f32[1, 104, 1, 1]" = var_mean_62[0]
    getitem_363: "f32[1, 104, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_340: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_362, 1e-05)
    rsqrt_62: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_340);  add_340 = None
    sub_62: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_363)
    mul_434: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
    squeeze_186: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_363, [0, 2, 3]);  getitem_363 = None
    squeeze_187: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
    mul_435: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_436: "f32[104]" = torch.ops.aten.mul.Tensor(primals_699, 0.9)
    add_341: "f32[104]" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
    squeeze_188: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_362, [0, 2, 3]);  getitem_362 = None
    mul_437: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0006381620931717);  squeeze_188 = None
    mul_438: "f32[104]" = torch.ops.aten.mul.Tensor(mul_437, 0.1);  mul_437 = None
    mul_439: "f32[104]" = torch.ops.aten.mul.Tensor(primals_700, 0.9)
    add_342: "f32[104]" = torch.ops.aten.add.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_248: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_188, -1)
    unsqueeze_249: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    mul_440: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_249);  mul_434 = unsqueeze_249 = None
    unsqueeze_250: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_189, -1);  primals_189 = None
    unsqueeze_251: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    add_343: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_251);  mul_440 = unsqueeze_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_59: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_343);  add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_367: "f32[8, 104, 14, 14]" = split_with_sizes_56[3];  split_with_sizes_56 = None
    cat_11: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_57, relu_58, relu_59, getitem_367], 1);  getitem_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_63: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_11, primals_190, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_344: "i64[]" = torch.ops.aten.add.Tensor(primals_704, 1)
    var_mean_63 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_368: "f32[1, 1024, 1, 1]" = var_mean_63[0]
    getitem_369: "f32[1, 1024, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_345: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_368, 1e-05)
    rsqrt_63: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_345);  add_345 = None
    sub_63: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_369)
    mul_441: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
    squeeze_189: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_369, [0, 2, 3]);  getitem_369 = None
    squeeze_190: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
    mul_442: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_443: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_702, 0.9)
    add_346: "f32[1024]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    squeeze_191: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_368, [0, 2, 3]);  getitem_368 = None
    mul_444: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0006381620931717);  squeeze_191 = None
    mul_445: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
    mul_446: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_703, 0.9)
    add_347: "f32[1024]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
    unsqueeze_252: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_191, -1)
    unsqueeze_253: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_447: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_253);  mul_441 = unsqueeze_253 = None
    unsqueeze_254: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_192, -1);  primals_192 = None
    unsqueeze_255: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_348: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_255);  mul_447 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_349: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_348, relu_55);  add_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_60: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_349);  add_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_64: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_60, primals_193, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_350: "i64[]" = torch.ops.aten.add.Tensor(primals_707, 1)
    var_mean_64 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_370: "f32[1, 416, 1, 1]" = var_mean_64[0]
    getitem_371: "f32[1, 416, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_351: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_370, 1e-05)
    rsqrt_64: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
    sub_64: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_371)
    mul_448: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
    squeeze_192: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_371, [0, 2, 3]);  getitem_371 = None
    squeeze_193: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
    mul_449: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
    mul_450: "f32[416]" = torch.ops.aten.mul.Tensor(primals_705, 0.9)
    add_352: "f32[416]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    squeeze_194: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_370, [0, 2, 3]);  getitem_370 = None
    mul_451: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0006381620931717);  squeeze_194 = None
    mul_452: "f32[416]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
    mul_453: "f32[416]" = torch.ops.aten.mul.Tensor(primals_706, 0.9)
    add_353: "f32[416]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    unsqueeze_256: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_194, -1)
    unsqueeze_257: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    mul_454: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_257);  mul_448 = unsqueeze_257 = None
    unsqueeze_258: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_195, -1);  primals_195 = None
    unsqueeze_259: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    add_354: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_259);  mul_454 = unsqueeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_61: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_354);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_61 = torch.ops.aten.split_with_sizes.default(relu_61, [104, 104, 104, 104], 1)
    getitem_376: "f32[8, 104, 14, 14]" = split_with_sizes_61[0]
    convolution_65: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_376, primals_196, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_355: "i64[]" = torch.ops.aten.add.Tensor(primals_710, 1)
    var_mean_65 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_380: "f32[1, 104, 1, 1]" = var_mean_65[0]
    getitem_381: "f32[1, 104, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_356: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_380, 1e-05)
    rsqrt_65: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_356);  add_356 = None
    sub_65: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_381)
    mul_455: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
    squeeze_195: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_381, [0, 2, 3]);  getitem_381 = None
    squeeze_196: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
    mul_456: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
    mul_457: "f32[104]" = torch.ops.aten.mul.Tensor(primals_708, 0.9)
    add_357: "f32[104]" = torch.ops.aten.add.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    squeeze_197: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_380, [0, 2, 3]);  getitem_380 = None
    mul_458: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0006381620931717);  squeeze_197 = None
    mul_459: "f32[104]" = torch.ops.aten.mul.Tensor(mul_458, 0.1);  mul_458 = None
    mul_460: "f32[104]" = torch.ops.aten.mul.Tensor(primals_709, 0.9)
    add_358: "f32[104]" = torch.ops.aten.add.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
    unsqueeze_260: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_197, -1)
    unsqueeze_261: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_461: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_455, unsqueeze_261);  mul_455 = unsqueeze_261 = None
    unsqueeze_262: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_198, -1);  primals_198 = None
    unsqueeze_263: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_359: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_263);  mul_461 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_62: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_359);  add_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_383: "f32[8, 104, 14, 14]" = split_with_sizes_61[1]
    add_360: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_62, getitem_383);  getitem_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_66: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_360, primals_199, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_361: "i64[]" = torch.ops.aten.add.Tensor(primals_713, 1)
    var_mean_66 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_386: "f32[1, 104, 1, 1]" = var_mean_66[0]
    getitem_387: "f32[1, 104, 1, 1]" = var_mean_66[1];  var_mean_66 = None
    add_362: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_386, 1e-05)
    rsqrt_66: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_362);  add_362 = None
    sub_66: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_387)
    mul_462: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
    squeeze_198: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_387, [0, 2, 3]);  getitem_387 = None
    squeeze_199: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
    mul_463: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
    mul_464: "f32[104]" = torch.ops.aten.mul.Tensor(primals_711, 0.9)
    add_363: "f32[104]" = torch.ops.aten.add.Tensor(mul_463, mul_464);  mul_463 = mul_464 = None
    squeeze_200: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_386, [0, 2, 3]);  getitem_386 = None
    mul_465: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0006381620931717);  squeeze_200 = None
    mul_466: "f32[104]" = torch.ops.aten.mul.Tensor(mul_465, 0.1);  mul_465 = None
    mul_467: "f32[104]" = torch.ops.aten.mul.Tensor(primals_712, 0.9)
    add_364: "f32[104]" = torch.ops.aten.add.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
    unsqueeze_264: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_200, -1)
    unsqueeze_265: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    mul_468: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_265);  mul_462 = unsqueeze_265 = None
    unsqueeze_266: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_201, -1);  primals_201 = None
    unsqueeze_267: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    add_365: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_468, unsqueeze_267);  mul_468 = unsqueeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_63: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_365);  add_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_390: "f32[8, 104, 14, 14]" = split_with_sizes_61[2]
    add_366: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_63, getitem_390);  getitem_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_67: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_366, primals_202, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_367: "i64[]" = torch.ops.aten.add.Tensor(primals_716, 1)
    var_mean_67 = torch.ops.aten.var_mean.correction(convolution_67, [0, 2, 3], correction = 0, keepdim = True)
    getitem_392: "f32[1, 104, 1, 1]" = var_mean_67[0]
    getitem_393: "f32[1, 104, 1, 1]" = var_mean_67[1];  var_mean_67 = None
    add_368: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_392, 1e-05)
    rsqrt_67: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_368);  add_368 = None
    sub_67: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, getitem_393)
    mul_469: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
    squeeze_201: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_393, [0, 2, 3]);  getitem_393 = None
    squeeze_202: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
    mul_470: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
    mul_471: "f32[104]" = torch.ops.aten.mul.Tensor(primals_714, 0.9)
    add_369: "f32[104]" = torch.ops.aten.add.Tensor(mul_470, mul_471);  mul_470 = mul_471 = None
    squeeze_203: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_392, [0, 2, 3]);  getitem_392 = None
    mul_472: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_203, 1.0006381620931717);  squeeze_203 = None
    mul_473: "f32[104]" = torch.ops.aten.mul.Tensor(mul_472, 0.1);  mul_472 = None
    mul_474: "f32[104]" = torch.ops.aten.mul.Tensor(primals_715, 0.9)
    add_370: "f32[104]" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    unsqueeze_268: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1)
    unsqueeze_269: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_475: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_269);  mul_469 = unsqueeze_269 = None
    unsqueeze_270: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
    unsqueeze_271: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_371: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_475, unsqueeze_271);  mul_475 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_64: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_371);  add_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_397: "f32[8, 104, 14, 14]" = split_with_sizes_61[3];  split_with_sizes_61 = None
    cat_12: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_62, relu_63, relu_64, getitem_397], 1);  getitem_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_68: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_12, primals_205, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_372: "i64[]" = torch.ops.aten.add.Tensor(primals_719, 1)
    var_mean_68 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True)
    getitem_398: "f32[1, 1024, 1, 1]" = var_mean_68[0]
    getitem_399: "f32[1, 1024, 1, 1]" = var_mean_68[1];  var_mean_68 = None
    add_373: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_398, 1e-05)
    rsqrt_68: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_373);  add_373 = None
    sub_68: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_399)
    mul_476: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
    squeeze_204: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_399, [0, 2, 3]);  getitem_399 = None
    squeeze_205: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
    mul_477: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1)
    mul_478: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_717, 0.9)
    add_374: "f32[1024]" = torch.ops.aten.add.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
    squeeze_206: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_398, [0, 2, 3]);  getitem_398 = None
    mul_479: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_206, 1.0006381620931717);  squeeze_206 = None
    mul_480: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_479, 0.1);  mul_479 = None
    mul_481: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_718, 0.9)
    add_375: "f32[1024]" = torch.ops.aten.add.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
    unsqueeze_272: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_206, -1)
    unsqueeze_273: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    mul_482: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_273);  mul_476 = unsqueeze_273 = None
    unsqueeze_274: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_207, -1);  primals_207 = None
    unsqueeze_275: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    add_376: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_275);  mul_482 = unsqueeze_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_377: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_376, relu_60);  add_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_65: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_377);  add_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_69: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_65, primals_208, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_378: "i64[]" = torch.ops.aten.add.Tensor(primals_722, 1)
    var_mean_69 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
    getitem_400: "f32[1, 416, 1, 1]" = var_mean_69[0]
    getitem_401: "f32[1, 416, 1, 1]" = var_mean_69[1];  var_mean_69 = None
    add_379: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_400, 1e-05)
    rsqrt_69: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_379);  add_379 = None
    sub_69: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_401)
    mul_483: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
    squeeze_207: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_401, [0, 2, 3]);  getitem_401 = None
    squeeze_208: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
    mul_484: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1)
    mul_485: "f32[416]" = torch.ops.aten.mul.Tensor(primals_720, 0.9)
    add_380: "f32[416]" = torch.ops.aten.add.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
    squeeze_209: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_400, [0, 2, 3]);  getitem_400 = None
    mul_486: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_209, 1.0006381620931717);  squeeze_209 = None
    mul_487: "f32[416]" = torch.ops.aten.mul.Tensor(mul_486, 0.1);  mul_486 = None
    mul_488: "f32[416]" = torch.ops.aten.mul.Tensor(primals_721, 0.9)
    add_381: "f32[416]" = torch.ops.aten.add.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
    unsqueeze_276: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_209, -1)
    unsqueeze_277: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_489: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_483, unsqueeze_277);  mul_483 = unsqueeze_277 = None
    unsqueeze_278: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_210, -1);  primals_210 = None
    unsqueeze_279: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_382: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_489, unsqueeze_279);  mul_489 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_66: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_382);  add_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_66 = torch.ops.aten.split_with_sizes.default(relu_66, [104, 104, 104, 104], 1)
    getitem_406: "f32[8, 104, 14, 14]" = split_with_sizes_66[0]
    convolution_70: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_406, primals_211, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_383: "i64[]" = torch.ops.aten.add.Tensor(primals_725, 1)
    var_mean_70 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
    getitem_410: "f32[1, 104, 1, 1]" = var_mean_70[0]
    getitem_411: "f32[1, 104, 1, 1]" = var_mean_70[1];  var_mean_70 = None
    add_384: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_410, 1e-05)
    rsqrt_70: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_384);  add_384 = None
    sub_70: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_411)
    mul_490: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
    squeeze_210: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_411, [0, 2, 3]);  getitem_411 = None
    squeeze_211: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
    mul_491: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
    mul_492: "f32[104]" = torch.ops.aten.mul.Tensor(primals_723, 0.9)
    add_385: "f32[104]" = torch.ops.aten.add.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
    squeeze_212: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_410, [0, 2, 3]);  getitem_410 = None
    mul_493: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_212, 1.0006381620931717);  squeeze_212 = None
    mul_494: "f32[104]" = torch.ops.aten.mul.Tensor(mul_493, 0.1);  mul_493 = None
    mul_495: "f32[104]" = torch.ops.aten.mul.Tensor(primals_724, 0.9)
    add_386: "f32[104]" = torch.ops.aten.add.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
    unsqueeze_280: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_212, -1)
    unsqueeze_281: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    mul_496: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_281);  mul_490 = unsqueeze_281 = None
    unsqueeze_282: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_213, -1);  primals_213 = None
    unsqueeze_283: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    add_387: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_496, unsqueeze_283);  mul_496 = unsqueeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_67: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_387);  add_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_413: "f32[8, 104, 14, 14]" = split_with_sizes_66[1]
    add_388: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_67, getitem_413);  getitem_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_71: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_388, primals_214, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_389: "i64[]" = torch.ops.aten.add.Tensor(primals_728, 1)
    var_mean_71 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
    getitem_416: "f32[1, 104, 1, 1]" = var_mean_71[0]
    getitem_417: "f32[1, 104, 1, 1]" = var_mean_71[1];  var_mean_71 = None
    add_390: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_416, 1e-05)
    rsqrt_71: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_390);  add_390 = None
    sub_71: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_417)
    mul_497: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
    squeeze_213: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_417, [0, 2, 3]);  getitem_417 = None
    squeeze_214: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
    mul_498: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1)
    mul_499: "f32[104]" = torch.ops.aten.mul.Tensor(primals_726, 0.9)
    add_391: "f32[104]" = torch.ops.aten.add.Tensor(mul_498, mul_499);  mul_498 = mul_499 = None
    squeeze_215: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_416, [0, 2, 3]);  getitem_416 = None
    mul_500: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_215, 1.0006381620931717);  squeeze_215 = None
    mul_501: "f32[104]" = torch.ops.aten.mul.Tensor(mul_500, 0.1);  mul_500 = None
    mul_502: "f32[104]" = torch.ops.aten.mul.Tensor(primals_727, 0.9)
    add_392: "f32[104]" = torch.ops.aten.add.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
    unsqueeze_284: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_215, -1)
    unsqueeze_285: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_503: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_497, unsqueeze_285);  mul_497 = unsqueeze_285 = None
    unsqueeze_286: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_216, -1);  primals_216 = None
    unsqueeze_287: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_393: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_287);  mul_503 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_68: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_393);  add_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_420: "f32[8, 104, 14, 14]" = split_with_sizes_66[2]
    add_394: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_68, getitem_420);  getitem_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_72: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_394, primals_217, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_395: "i64[]" = torch.ops.aten.add.Tensor(primals_731, 1)
    var_mean_72 = torch.ops.aten.var_mean.correction(convolution_72, [0, 2, 3], correction = 0, keepdim = True)
    getitem_422: "f32[1, 104, 1, 1]" = var_mean_72[0]
    getitem_423: "f32[1, 104, 1, 1]" = var_mean_72[1];  var_mean_72 = None
    add_396: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_422, 1e-05)
    rsqrt_72: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_396);  add_396 = None
    sub_72: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, getitem_423)
    mul_504: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
    squeeze_216: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_423, [0, 2, 3]);  getitem_423 = None
    squeeze_217: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
    mul_505: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1)
    mul_506: "f32[104]" = torch.ops.aten.mul.Tensor(primals_729, 0.9)
    add_397: "f32[104]" = torch.ops.aten.add.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    squeeze_218: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_422, [0, 2, 3]);  getitem_422 = None
    mul_507: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_218, 1.0006381620931717);  squeeze_218 = None
    mul_508: "f32[104]" = torch.ops.aten.mul.Tensor(mul_507, 0.1);  mul_507 = None
    mul_509: "f32[104]" = torch.ops.aten.mul.Tensor(primals_730, 0.9)
    add_398: "f32[104]" = torch.ops.aten.add.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
    unsqueeze_288: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_218, -1)
    unsqueeze_289: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    mul_510: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_504, unsqueeze_289);  mul_504 = unsqueeze_289 = None
    unsqueeze_290: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_219, -1);  primals_219 = None
    unsqueeze_291: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    add_399: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_510, unsqueeze_291);  mul_510 = unsqueeze_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_69: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_399);  add_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_427: "f32[8, 104, 14, 14]" = split_with_sizes_66[3];  split_with_sizes_66 = None
    cat_13: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_67, relu_68, relu_69, getitem_427], 1);  getitem_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_73: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_13, primals_220, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_400: "i64[]" = torch.ops.aten.add.Tensor(primals_734, 1)
    var_mean_73 = torch.ops.aten.var_mean.correction(convolution_73, [0, 2, 3], correction = 0, keepdim = True)
    getitem_428: "f32[1, 1024, 1, 1]" = var_mean_73[0]
    getitem_429: "f32[1, 1024, 1, 1]" = var_mean_73[1];  var_mean_73 = None
    add_401: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_428, 1e-05)
    rsqrt_73: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_401);  add_401 = None
    sub_73: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, getitem_429)
    mul_511: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
    squeeze_219: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_429, [0, 2, 3]);  getitem_429 = None
    squeeze_220: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
    mul_512: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
    mul_513: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_732, 0.9)
    add_402: "f32[1024]" = torch.ops.aten.add.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
    squeeze_221: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_428, [0, 2, 3]);  getitem_428 = None
    mul_514: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_221, 1.0006381620931717);  squeeze_221 = None
    mul_515: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_514, 0.1);  mul_514 = None
    mul_516: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_733, 0.9)
    add_403: "f32[1024]" = torch.ops.aten.add.Tensor(mul_515, mul_516);  mul_515 = mul_516 = None
    unsqueeze_292: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_221, -1)
    unsqueeze_293: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_517: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_511, unsqueeze_293);  mul_511 = unsqueeze_293 = None
    unsqueeze_294: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_222, -1);  primals_222 = None
    unsqueeze_295: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_404: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_517, unsqueeze_295);  mul_517 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_405: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_404, relu_65);  add_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_70: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_405);  add_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_74: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_70, primals_223, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_406: "i64[]" = torch.ops.aten.add.Tensor(primals_737, 1)
    var_mean_74 = torch.ops.aten.var_mean.correction(convolution_74, [0, 2, 3], correction = 0, keepdim = True)
    getitem_430: "f32[1, 416, 1, 1]" = var_mean_74[0]
    getitem_431: "f32[1, 416, 1, 1]" = var_mean_74[1];  var_mean_74 = None
    add_407: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_430, 1e-05)
    rsqrt_74: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_407);  add_407 = None
    sub_74: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, getitem_431)
    mul_518: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
    squeeze_222: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_431, [0, 2, 3]);  getitem_431 = None
    squeeze_223: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
    mul_519: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
    mul_520: "f32[416]" = torch.ops.aten.mul.Tensor(primals_735, 0.9)
    add_408: "f32[416]" = torch.ops.aten.add.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
    squeeze_224: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_430, [0, 2, 3]);  getitem_430 = None
    mul_521: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_224, 1.0006381620931717);  squeeze_224 = None
    mul_522: "f32[416]" = torch.ops.aten.mul.Tensor(mul_521, 0.1);  mul_521 = None
    mul_523: "f32[416]" = torch.ops.aten.mul.Tensor(primals_736, 0.9)
    add_409: "f32[416]" = torch.ops.aten.add.Tensor(mul_522, mul_523);  mul_522 = mul_523 = None
    unsqueeze_296: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_224, -1)
    unsqueeze_297: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    mul_524: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_518, unsqueeze_297);  mul_518 = unsqueeze_297 = None
    unsqueeze_298: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_225, -1);  primals_225 = None
    unsqueeze_299: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    add_410: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_524, unsqueeze_299);  mul_524 = unsqueeze_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_71: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_410);  add_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_71 = torch.ops.aten.split_with_sizes.default(relu_71, [104, 104, 104, 104], 1)
    getitem_436: "f32[8, 104, 14, 14]" = split_with_sizes_71[0]
    convolution_75: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_436, primals_226, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_411: "i64[]" = torch.ops.aten.add.Tensor(primals_740, 1)
    var_mean_75 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
    getitem_440: "f32[1, 104, 1, 1]" = var_mean_75[0]
    getitem_441: "f32[1, 104, 1, 1]" = var_mean_75[1];  var_mean_75 = None
    add_412: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_440, 1e-05)
    rsqrt_75: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_412);  add_412 = None
    sub_75: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_441)
    mul_525: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
    squeeze_225: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_441, [0, 2, 3]);  getitem_441 = None
    squeeze_226: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
    mul_526: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1)
    mul_527: "f32[104]" = torch.ops.aten.mul.Tensor(primals_738, 0.9)
    add_413: "f32[104]" = torch.ops.aten.add.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
    squeeze_227: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_440, [0, 2, 3]);  getitem_440 = None
    mul_528: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_227, 1.0006381620931717);  squeeze_227 = None
    mul_529: "f32[104]" = torch.ops.aten.mul.Tensor(mul_528, 0.1);  mul_528 = None
    mul_530: "f32[104]" = torch.ops.aten.mul.Tensor(primals_739, 0.9)
    add_414: "f32[104]" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    unsqueeze_300: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_227, -1)
    unsqueeze_301: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_531: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_301);  mul_525 = unsqueeze_301 = None
    unsqueeze_302: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1);  primals_228 = None
    unsqueeze_303: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_415: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_531, unsqueeze_303);  mul_531 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_72: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_415);  add_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_443: "f32[8, 104, 14, 14]" = split_with_sizes_71[1]
    add_416: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_72, getitem_443);  getitem_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_76: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_416, primals_229, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_417: "i64[]" = torch.ops.aten.add.Tensor(primals_743, 1)
    var_mean_76 = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
    getitem_446: "f32[1, 104, 1, 1]" = var_mean_76[0]
    getitem_447: "f32[1, 104, 1, 1]" = var_mean_76[1];  var_mean_76 = None
    add_418: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_446, 1e-05)
    rsqrt_76: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_418);  add_418 = None
    sub_76: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_447)
    mul_532: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
    squeeze_228: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_447, [0, 2, 3]);  getitem_447 = None
    squeeze_229: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
    mul_533: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1)
    mul_534: "f32[104]" = torch.ops.aten.mul.Tensor(primals_741, 0.9)
    add_419: "f32[104]" = torch.ops.aten.add.Tensor(mul_533, mul_534);  mul_533 = mul_534 = None
    squeeze_230: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_446, [0, 2, 3]);  getitem_446 = None
    mul_535: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_230, 1.0006381620931717);  squeeze_230 = None
    mul_536: "f32[104]" = torch.ops.aten.mul.Tensor(mul_535, 0.1);  mul_535 = None
    mul_537: "f32[104]" = torch.ops.aten.mul.Tensor(primals_742, 0.9)
    add_420: "f32[104]" = torch.ops.aten.add.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
    unsqueeze_304: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_230, -1)
    unsqueeze_305: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    mul_538: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_305);  mul_532 = unsqueeze_305 = None
    unsqueeze_306: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_231, -1);  primals_231 = None
    unsqueeze_307: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    add_421: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_538, unsqueeze_307);  mul_538 = unsqueeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_73: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_421);  add_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_450: "f32[8, 104, 14, 14]" = split_with_sizes_71[2]
    add_422: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_73, getitem_450);  getitem_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_77: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_422, primals_232, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_423: "i64[]" = torch.ops.aten.add.Tensor(primals_746, 1)
    var_mean_77 = torch.ops.aten.var_mean.correction(convolution_77, [0, 2, 3], correction = 0, keepdim = True)
    getitem_452: "f32[1, 104, 1, 1]" = var_mean_77[0]
    getitem_453: "f32[1, 104, 1, 1]" = var_mean_77[1];  var_mean_77 = None
    add_424: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_452, 1e-05)
    rsqrt_77: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_424);  add_424 = None
    sub_77: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, getitem_453)
    mul_539: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
    squeeze_231: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_453, [0, 2, 3]);  getitem_453 = None
    squeeze_232: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
    mul_540: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
    mul_541: "f32[104]" = torch.ops.aten.mul.Tensor(primals_744, 0.9)
    add_425: "f32[104]" = torch.ops.aten.add.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
    squeeze_233: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_452, [0, 2, 3]);  getitem_452 = None
    mul_542: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_233, 1.0006381620931717);  squeeze_233 = None
    mul_543: "f32[104]" = torch.ops.aten.mul.Tensor(mul_542, 0.1);  mul_542 = None
    mul_544: "f32[104]" = torch.ops.aten.mul.Tensor(primals_745, 0.9)
    add_426: "f32[104]" = torch.ops.aten.add.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
    unsqueeze_308: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_233, -1)
    unsqueeze_309: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_545: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_539, unsqueeze_309);  mul_539 = unsqueeze_309 = None
    unsqueeze_310: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_234, -1);  primals_234 = None
    unsqueeze_311: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_427: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_311);  mul_545 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_74: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_427);  add_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_457: "f32[8, 104, 14, 14]" = split_with_sizes_71[3];  split_with_sizes_71 = None
    cat_14: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_72, relu_73, relu_74, getitem_457], 1);  getitem_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_78: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_14, primals_235, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_428: "i64[]" = torch.ops.aten.add.Tensor(primals_749, 1)
    var_mean_78 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
    getitem_458: "f32[1, 1024, 1, 1]" = var_mean_78[0]
    getitem_459: "f32[1, 1024, 1, 1]" = var_mean_78[1];  var_mean_78 = None
    add_429: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_458, 1e-05)
    rsqrt_78: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_429);  add_429 = None
    sub_78: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_459)
    mul_546: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
    squeeze_234: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_459, [0, 2, 3]);  getitem_459 = None
    squeeze_235: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
    mul_547: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1)
    mul_548: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_747, 0.9)
    add_430: "f32[1024]" = torch.ops.aten.add.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
    squeeze_236: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_458, [0, 2, 3]);  getitem_458 = None
    mul_549: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_236, 1.0006381620931717);  squeeze_236 = None
    mul_550: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_549, 0.1);  mul_549 = None
    mul_551: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_748, 0.9)
    add_431: "f32[1024]" = torch.ops.aten.add.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
    unsqueeze_312: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_236, -1)
    unsqueeze_313: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    mul_552: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_546, unsqueeze_313);  mul_546 = unsqueeze_313 = None
    unsqueeze_314: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_237, -1);  primals_237 = None
    unsqueeze_315: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    add_432: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_552, unsqueeze_315);  mul_552 = unsqueeze_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_433: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_432, relu_70);  add_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_75: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_433);  add_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_79: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_75, primals_238, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_434: "i64[]" = torch.ops.aten.add.Tensor(primals_752, 1)
    var_mean_79 = torch.ops.aten.var_mean.correction(convolution_79, [0, 2, 3], correction = 0, keepdim = True)
    getitem_460: "f32[1, 416, 1, 1]" = var_mean_79[0]
    getitem_461: "f32[1, 416, 1, 1]" = var_mean_79[1];  var_mean_79 = None
    add_435: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_460, 1e-05)
    rsqrt_79: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_435);  add_435 = None
    sub_79: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, getitem_461)
    mul_553: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
    squeeze_237: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_461, [0, 2, 3]);  getitem_461 = None
    squeeze_238: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
    mul_554: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1)
    mul_555: "f32[416]" = torch.ops.aten.mul.Tensor(primals_750, 0.9)
    add_436: "f32[416]" = torch.ops.aten.add.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
    squeeze_239: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_460, [0, 2, 3]);  getitem_460 = None
    mul_556: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_239, 1.0006381620931717);  squeeze_239 = None
    mul_557: "f32[416]" = torch.ops.aten.mul.Tensor(mul_556, 0.1);  mul_556 = None
    mul_558: "f32[416]" = torch.ops.aten.mul.Tensor(primals_751, 0.9)
    add_437: "f32[416]" = torch.ops.aten.add.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
    unsqueeze_316: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_239, -1)
    unsqueeze_317: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_559: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_317);  mul_553 = unsqueeze_317 = None
    unsqueeze_318: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_240, -1);  primals_240 = None
    unsqueeze_319: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_438: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_559, unsqueeze_319);  mul_559 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_76: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_438);  add_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_76 = torch.ops.aten.split_with_sizes.default(relu_76, [104, 104, 104, 104], 1)
    getitem_466: "f32[8, 104, 14, 14]" = split_with_sizes_76[0]
    convolution_80: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_466, primals_241, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_439: "i64[]" = torch.ops.aten.add.Tensor(primals_755, 1)
    var_mean_80 = torch.ops.aten.var_mean.correction(convolution_80, [0, 2, 3], correction = 0, keepdim = True)
    getitem_470: "f32[1, 104, 1, 1]" = var_mean_80[0]
    getitem_471: "f32[1, 104, 1, 1]" = var_mean_80[1];  var_mean_80 = None
    add_440: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_470, 1e-05)
    rsqrt_80: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_440);  add_440 = None
    sub_80: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_471)
    mul_560: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = None
    squeeze_240: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_471, [0, 2, 3]);  getitem_471 = None
    squeeze_241: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
    mul_561: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_240, 0.1)
    mul_562: "f32[104]" = torch.ops.aten.mul.Tensor(primals_753, 0.9)
    add_441: "f32[104]" = torch.ops.aten.add.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    squeeze_242: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_470, [0, 2, 3]);  getitem_470 = None
    mul_563: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_242, 1.0006381620931717);  squeeze_242 = None
    mul_564: "f32[104]" = torch.ops.aten.mul.Tensor(mul_563, 0.1);  mul_563 = None
    mul_565: "f32[104]" = torch.ops.aten.mul.Tensor(primals_754, 0.9)
    add_442: "f32[104]" = torch.ops.aten.add.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_320: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_242, -1)
    unsqueeze_321: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    mul_566: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_321);  mul_560 = unsqueeze_321 = None
    unsqueeze_322: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_243, -1);  primals_243 = None
    unsqueeze_323: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    add_443: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_323);  mul_566 = unsqueeze_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_77: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_443);  add_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_473: "f32[8, 104, 14, 14]" = split_with_sizes_76[1]
    add_444: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_77, getitem_473);  getitem_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_81: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_444, primals_244, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_445: "i64[]" = torch.ops.aten.add.Tensor(primals_758, 1)
    var_mean_81 = torch.ops.aten.var_mean.correction(convolution_81, [0, 2, 3], correction = 0, keepdim = True)
    getitem_476: "f32[1, 104, 1, 1]" = var_mean_81[0]
    getitem_477: "f32[1, 104, 1, 1]" = var_mean_81[1];  var_mean_81 = None
    add_446: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_476, 1e-05)
    rsqrt_81: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_446);  add_446 = None
    sub_81: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, getitem_477)
    mul_567: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = None
    squeeze_243: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_477, [0, 2, 3]);  getitem_477 = None
    squeeze_244: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
    mul_568: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_243, 0.1)
    mul_569: "f32[104]" = torch.ops.aten.mul.Tensor(primals_756, 0.9)
    add_447: "f32[104]" = torch.ops.aten.add.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
    squeeze_245: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_476, [0, 2, 3]);  getitem_476 = None
    mul_570: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_245, 1.0006381620931717);  squeeze_245 = None
    mul_571: "f32[104]" = torch.ops.aten.mul.Tensor(mul_570, 0.1);  mul_570 = None
    mul_572: "f32[104]" = torch.ops.aten.mul.Tensor(primals_757, 0.9)
    add_448: "f32[104]" = torch.ops.aten.add.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
    unsqueeze_324: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_245, -1)
    unsqueeze_325: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_573: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_567, unsqueeze_325);  mul_567 = unsqueeze_325 = None
    unsqueeze_326: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_246, -1);  primals_246 = None
    unsqueeze_327: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_449: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_573, unsqueeze_327);  mul_573 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_78: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_449);  add_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_480: "f32[8, 104, 14, 14]" = split_with_sizes_76[2]
    add_450: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_78, getitem_480);  getitem_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_82: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_450, primals_247, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_451: "i64[]" = torch.ops.aten.add.Tensor(primals_761, 1)
    var_mean_82 = torch.ops.aten.var_mean.correction(convolution_82, [0, 2, 3], correction = 0, keepdim = True)
    getitem_482: "f32[1, 104, 1, 1]" = var_mean_82[0]
    getitem_483: "f32[1, 104, 1, 1]" = var_mean_82[1];  var_mean_82 = None
    add_452: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_482, 1e-05)
    rsqrt_82: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_452);  add_452 = None
    sub_82: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, getitem_483)
    mul_574: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
    squeeze_246: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_483, [0, 2, 3]);  getitem_483 = None
    squeeze_247: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
    mul_575: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_246, 0.1)
    mul_576: "f32[104]" = torch.ops.aten.mul.Tensor(primals_759, 0.9)
    add_453: "f32[104]" = torch.ops.aten.add.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
    squeeze_248: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_482, [0, 2, 3]);  getitem_482 = None
    mul_577: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_248, 1.0006381620931717);  squeeze_248 = None
    mul_578: "f32[104]" = torch.ops.aten.mul.Tensor(mul_577, 0.1);  mul_577 = None
    mul_579: "f32[104]" = torch.ops.aten.mul.Tensor(primals_760, 0.9)
    add_454: "f32[104]" = torch.ops.aten.add.Tensor(mul_578, mul_579);  mul_578 = mul_579 = None
    unsqueeze_328: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_248, -1)
    unsqueeze_329: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    mul_580: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_329);  mul_574 = unsqueeze_329 = None
    unsqueeze_330: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_249, -1);  primals_249 = None
    unsqueeze_331: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    add_455: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_580, unsqueeze_331);  mul_580 = unsqueeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_79: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_455);  add_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_487: "f32[8, 104, 14, 14]" = split_with_sizes_76[3];  split_with_sizes_76 = None
    cat_15: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_77, relu_78, relu_79, getitem_487], 1);  getitem_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_83: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_15, primals_250, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_456: "i64[]" = torch.ops.aten.add.Tensor(primals_764, 1)
    var_mean_83 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
    getitem_488: "f32[1, 1024, 1, 1]" = var_mean_83[0]
    getitem_489: "f32[1, 1024, 1, 1]" = var_mean_83[1];  var_mean_83 = None
    add_457: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_488, 1e-05)
    rsqrt_83: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_457);  add_457 = None
    sub_83: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_489)
    mul_581: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
    squeeze_249: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_489, [0, 2, 3]);  getitem_489 = None
    squeeze_250: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
    mul_582: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_249, 0.1)
    mul_583: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_762, 0.9)
    add_458: "f32[1024]" = torch.ops.aten.add.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    squeeze_251: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_488, [0, 2, 3]);  getitem_488 = None
    mul_584: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_251, 1.0006381620931717);  squeeze_251 = None
    mul_585: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_584, 0.1);  mul_584 = None
    mul_586: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_763, 0.9)
    add_459: "f32[1024]" = torch.ops.aten.add.Tensor(mul_585, mul_586);  mul_585 = mul_586 = None
    unsqueeze_332: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_251, -1)
    unsqueeze_333: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_587: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_581, unsqueeze_333);  mul_581 = unsqueeze_333 = None
    unsqueeze_334: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_252, -1);  primals_252 = None
    unsqueeze_335: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_460: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_335);  mul_587 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_461: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_460, relu_75);  add_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_80: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_461);  add_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_84: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_80, primals_253, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_462: "i64[]" = torch.ops.aten.add.Tensor(primals_767, 1)
    var_mean_84 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
    getitem_490: "f32[1, 416, 1, 1]" = var_mean_84[0]
    getitem_491: "f32[1, 416, 1, 1]" = var_mean_84[1];  var_mean_84 = None
    add_463: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_490, 1e-05)
    rsqrt_84: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_463);  add_463 = None
    sub_84: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_491)
    mul_588: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
    squeeze_252: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_491, [0, 2, 3]);  getitem_491 = None
    squeeze_253: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
    mul_589: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_252, 0.1)
    mul_590: "f32[416]" = torch.ops.aten.mul.Tensor(primals_765, 0.9)
    add_464: "f32[416]" = torch.ops.aten.add.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
    squeeze_254: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_490, [0, 2, 3]);  getitem_490 = None
    mul_591: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_254, 1.0006381620931717);  squeeze_254 = None
    mul_592: "f32[416]" = torch.ops.aten.mul.Tensor(mul_591, 0.1);  mul_591 = None
    mul_593: "f32[416]" = torch.ops.aten.mul.Tensor(primals_766, 0.9)
    add_465: "f32[416]" = torch.ops.aten.add.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
    unsqueeze_336: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_254, -1)
    unsqueeze_337: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    mul_594: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_588, unsqueeze_337);  mul_588 = unsqueeze_337 = None
    unsqueeze_338: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_255, -1);  primals_255 = None
    unsqueeze_339: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    add_466: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_594, unsqueeze_339);  mul_594 = unsqueeze_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_81: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_466);  add_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_81 = torch.ops.aten.split_with_sizes.default(relu_81, [104, 104, 104, 104], 1)
    getitem_496: "f32[8, 104, 14, 14]" = split_with_sizes_81[0]
    convolution_85: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_496, primals_256, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_467: "i64[]" = torch.ops.aten.add.Tensor(primals_770, 1)
    var_mean_85 = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True)
    getitem_500: "f32[1, 104, 1, 1]" = var_mean_85[0]
    getitem_501: "f32[1, 104, 1, 1]" = var_mean_85[1];  var_mean_85 = None
    add_468: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_500, 1e-05)
    rsqrt_85: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_468);  add_468 = None
    sub_85: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_501)
    mul_595: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = None
    squeeze_255: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_501, [0, 2, 3]);  getitem_501 = None
    squeeze_256: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
    mul_596: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_255, 0.1)
    mul_597: "f32[104]" = torch.ops.aten.mul.Tensor(primals_768, 0.9)
    add_469: "f32[104]" = torch.ops.aten.add.Tensor(mul_596, mul_597);  mul_596 = mul_597 = None
    squeeze_257: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_500, [0, 2, 3]);  getitem_500 = None
    mul_598: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_257, 1.0006381620931717);  squeeze_257 = None
    mul_599: "f32[104]" = torch.ops.aten.mul.Tensor(mul_598, 0.1);  mul_598 = None
    mul_600: "f32[104]" = torch.ops.aten.mul.Tensor(primals_769, 0.9)
    add_470: "f32[104]" = torch.ops.aten.add.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
    unsqueeze_340: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_257, -1)
    unsqueeze_341: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_601: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_341);  mul_595 = unsqueeze_341 = None
    unsqueeze_342: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_258, -1);  primals_258 = None
    unsqueeze_343: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_471: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_601, unsqueeze_343);  mul_601 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_82: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_471);  add_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_503: "f32[8, 104, 14, 14]" = split_with_sizes_81[1]
    add_472: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_82, getitem_503);  getitem_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_86: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_472, primals_259, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_473: "i64[]" = torch.ops.aten.add.Tensor(primals_773, 1)
    var_mean_86 = torch.ops.aten.var_mean.correction(convolution_86, [0, 2, 3], correction = 0, keepdim = True)
    getitem_506: "f32[1, 104, 1, 1]" = var_mean_86[0]
    getitem_507: "f32[1, 104, 1, 1]" = var_mean_86[1];  var_mean_86 = None
    add_474: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_506, 1e-05)
    rsqrt_86: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_474);  add_474 = None
    sub_86: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, getitem_507)
    mul_602: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = None
    squeeze_258: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_507, [0, 2, 3]);  getitem_507 = None
    squeeze_259: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_86, [0, 2, 3]);  rsqrt_86 = None
    mul_603: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_258, 0.1)
    mul_604: "f32[104]" = torch.ops.aten.mul.Tensor(primals_771, 0.9)
    add_475: "f32[104]" = torch.ops.aten.add.Tensor(mul_603, mul_604);  mul_603 = mul_604 = None
    squeeze_260: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_506, [0, 2, 3]);  getitem_506 = None
    mul_605: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_260, 1.0006381620931717);  squeeze_260 = None
    mul_606: "f32[104]" = torch.ops.aten.mul.Tensor(mul_605, 0.1);  mul_605 = None
    mul_607: "f32[104]" = torch.ops.aten.mul.Tensor(primals_772, 0.9)
    add_476: "f32[104]" = torch.ops.aten.add.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
    unsqueeze_344: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_260, -1)
    unsqueeze_345: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    mul_608: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_602, unsqueeze_345);  mul_602 = unsqueeze_345 = None
    unsqueeze_346: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_261, -1);  primals_261 = None
    unsqueeze_347: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    add_477: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_608, unsqueeze_347);  mul_608 = unsqueeze_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_83: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_477);  add_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_510: "f32[8, 104, 14, 14]" = split_with_sizes_81[2]
    add_478: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_83, getitem_510);  getitem_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_87: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_478, primals_262, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_479: "i64[]" = torch.ops.aten.add.Tensor(primals_776, 1)
    var_mean_87 = torch.ops.aten.var_mean.correction(convolution_87, [0, 2, 3], correction = 0, keepdim = True)
    getitem_512: "f32[1, 104, 1, 1]" = var_mean_87[0]
    getitem_513: "f32[1, 104, 1, 1]" = var_mean_87[1];  var_mean_87 = None
    add_480: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_512, 1e-05)
    rsqrt_87: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_480);  add_480 = None
    sub_87: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, getitem_513)
    mul_609: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = None
    squeeze_261: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_513, [0, 2, 3]);  getitem_513 = None
    squeeze_262: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
    mul_610: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_261, 0.1)
    mul_611: "f32[104]" = torch.ops.aten.mul.Tensor(primals_774, 0.9)
    add_481: "f32[104]" = torch.ops.aten.add.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
    squeeze_263: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_512, [0, 2, 3]);  getitem_512 = None
    mul_612: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_263, 1.0006381620931717);  squeeze_263 = None
    mul_613: "f32[104]" = torch.ops.aten.mul.Tensor(mul_612, 0.1);  mul_612 = None
    mul_614: "f32[104]" = torch.ops.aten.mul.Tensor(primals_775, 0.9)
    add_482: "f32[104]" = torch.ops.aten.add.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
    unsqueeze_348: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_263, -1)
    unsqueeze_349: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_615: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_609, unsqueeze_349);  mul_609 = unsqueeze_349 = None
    unsqueeze_350: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_264, -1);  primals_264 = None
    unsqueeze_351: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_483: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_615, unsqueeze_351);  mul_615 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_84: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_483);  add_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_517: "f32[8, 104, 14, 14]" = split_with_sizes_81[3];  split_with_sizes_81 = None
    cat_16: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_82, relu_83, relu_84, getitem_517], 1);  getitem_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_88: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_16, primals_265, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_484: "i64[]" = torch.ops.aten.add.Tensor(primals_779, 1)
    var_mean_88 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
    getitem_518: "f32[1, 1024, 1, 1]" = var_mean_88[0]
    getitem_519: "f32[1, 1024, 1, 1]" = var_mean_88[1];  var_mean_88 = None
    add_485: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_518, 1e-05)
    rsqrt_88: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_485);  add_485 = None
    sub_88: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_519)
    mul_616: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = None
    squeeze_264: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_519, [0, 2, 3]);  getitem_519 = None
    squeeze_265: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
    mul_617: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_264, 0.1)
    mul_618: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_777, 0.9)
    add_486: "f32[1024]" = torch.ops.aten.add.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
    squeeze_266: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_518, [0, 2, 3]);  getitem_518 = None
    mul_619: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_266, 1.0006381620931717);  squeeze_266 = None
    mul_620: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_619, 0.1);  mul_619 = None
    mul_621: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_778, 0.9)
    add_487: "f32[1024]" = torch.ops.aten.add.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
    unsqueeze_352: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_266, -1)
    unsqueeze_353: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    mul_622: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_353);  mul_616 = unsqueeze_353 = None
    unsqueeze_354: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_267, -1);  primals_267 = None
    unsqueeze_355: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    add_488: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_622, unsqueeze_355);  mul_622 = unsqueeze_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_489: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_488, relu_80);  add_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_85: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_489);  add_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_89: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_85, primals_268, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_490: "i64[]" = torch.ops.aten.add.Tensor(primals_782, 1)
    var_mean_89 = torch.ops.aten.var_mean.correction(convolution_89, [0, 2, 3], correction = 0, keepdim = True)
    getitem_520: "f32[1, 416, 1, 1]" = var_mean_89[0]
    getitem_521: "f32[1, 416, 1, 1]" = var_mean_89[1];  var_mean_89 = None
    add_491: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_520, 1e-05)
    rsqrt_89: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_491);  add_491 = None
    sub_89: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, getitem_521)
    mul_623: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = None
    squeeze_267: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_521, [0, 2, 3]);  getitem_521 = None
    squeeze_268: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_89, [0, 2, 3]);  rsqrt_89 = None
    mul_624: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_267, 0.1)
    mul_625: "f32[416]" = torch.ops.aten.mul.Tensor(primals_780, 0.9)
    add_492: "f32[416]" = torch.ops.aten.add.Tensor(mul_624, mul_625);  mul_624 = mul_625 = None
    squeeze_269: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_520, [0, 2, 3]);  getitem_520 = None
    mul_626: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_269, 1.0006381620931717);  squeeze_269 = None
    mul_627: "f32[416]" = torch.ops.aten.mul.Tensor(mul_626, 0.1);  mul_626 = None
    mul_628: "f32[416]" = torch.ops.aten.mul.Tensor(primals_781, 0.9)
    add_493: "f32[416]" = torch.ops.aten.add.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
    unsqueeze_356: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_269, -1)
    unsqueeze_357: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_629: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_623, unsqueeze_357);  mul_623 = unsqueeze_357 = None
    unsqueeze_358: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_270, -1);  primals_270 = None
    unsqueeze_359: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_494: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_629, unsqueeze_359);  mul_629 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_86: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_494);  add_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_86 = torch.ops.aten.split_with_sizes.default(relu_86, [104, 104, 104, 104], 1)
    getitem_526: "f32[8, 104, 14, 14]" = split_with_sizes_86[0]
    convolution_90: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_526, primals_271, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_495: "i64[]" = torch.ops.aten.add.Tensor(primals_785, 1)
    var_mean_90 = torch.ops.aten.var_mean.correction(convolution_90, [0, 2, 3], correction = 0, keepdim = True)
    getitem_530: "f32[1, 104, 1, 1]" = var_mean_90[0]
    getitem_531: "f32[1, 104, 1, 1]" = var_mean_90[1];  var_mean_90 = None
    add_496: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_530, 1e-05)
    rsqrt_90: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_496);  add_496 = None
    sub_90: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, getitem_531)
    mul_630: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = None
    squeeze_270: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_531, [0, 2, 3]);  getitem_531 = None
    squeeze_271: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_90, [0, 2, 3]);  rsqrt_90 = None
    mul_631: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_270, 0.1)
    mul_632: "f32[104]" = torch.ops.aten.mul.Tensor(primals_783, 0.9)
    add_497: "f32[104]" = torch.ops.aten.add.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
    squeeze_272: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_530, [0, 2, 3]);  getitem_530 = None
    mul_633: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_272, 1.0006381620931717);  squeeze_272 = None
    mul_634: "f32[104]" = torch.ops.aten.mul.Tensor(mul_633, 0.1);  mul_633 = None
    mul_635: "f32[104]" = torch.ops.aten.mul.Tensor(primals_784, 0.9)
    add_498: "f32[104]" = torch.ops.aten.add.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
    unsqueeze_360: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_272, -1)
    unsqueeze_361: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    mul_636: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_630, unsqueeze_361);  mul_630 = unsqueeze_361 = None
    unsqueeze_362: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_273, -1);  primals_273 = None
    unsqueeze_363: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    add_499: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_636, unsqueeze_363);  mul_636 = unsqueeze_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_87: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_499);  add_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_533: "f32[8, 104, 14, 14]" = split_with_sizes_86[1]
    add_500: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_87, getitem_533);  getitem_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_91: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_500, primals_274, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_501: "i64[]" = torch.ops.aten.add.Tensor(primals_788, 1)
    var_mean_91 = torch.ops.aten.var_mean.correction(convolution_91, [0, 2, 3], correction = 0, keepdim = True)
    getitem_536: "f32[1, 104, 1, 1]" = var_mean_91[0]
    getitem_537: "f32[1, 104, 1, 1]" = var_mean_91[1];  var_mean_91 = None
    add_502: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_536, 1e-05)
    rsqrt_91: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_502);  add_502 = None
    sub_91: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, getitem_537)
    mul_637: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = None
    squeeze_273: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_537, [0, 2, 3]);  getitem_537 = None
    squeeze_274: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_91, [0, 2, 3]);  rsqrt_91 = None
    mul_638: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_273, 0.1)
    mul_639: "f32[104]" = torch.ops.aten.mul.Tensor(primals_786, 0.9)
    add_503: "f32[104]" = torch.ops.aten.add.Tensor(mul_638, mul_639);  mul_638 = mul_639 = None
    squeeze_275: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_536, [0, 2, 3]);  getitem_536 = None
    mul_640: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_275, 1.0006381620931717);  squeeze_275 = None
    mul_641: "f32[104]" = torch.ops.aten.mul.Tensor(mul_640, 0.1);  mul_640 = None
    mul_642: "f32[104]" = torch.ops.aten.mul.Tensor(primals_787, 0.9)
    add_504: "f32[104]" = torch.ops.aten.add.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
    unsqueeze_364: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_275, -1)
    unsqueeze_365: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_643: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_365);  mul_637 = unsqueeze_365 = None
    unsqueeze_366: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_276, -1);  primals_276 = None
    unsqueeze_367: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_505: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_643, unsqueeze_367);  mul_643 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_88: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_505);  add_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_540: "f32[8, 104, 14, 14]" = split_with_sizes_86[2]
    add_506: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_88, getitem_540);  getitem_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_92: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_506, primals_277, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_507: "i64[]" = torch.ops.aten.add.Tensor(primals_791, 1)
    var_mean_92 = torch.ops.aten.var_mean.correction(convolution_92, [0, 2, 3], correction = 0, keepdim = True)
    getitem_542: "f32[1, 104, 1, 1]" = var_mean_92[0]
    getitem_543: "f32[1, 104, 1, 1]" = var_mean_92[1];  var_mean_92 = None
    add_508: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_542, 1e-05)
    rsqrt_92: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_508);  add_508 = None
    sub_92: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_543)
    mul_644: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = None
    squeeze_276: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_543, [0, 2, 3]);  getitem_543 = None
    squeeze_277: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_92, [0, 2, 3]);  rsqrt_92 = None
    mul_645: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_276, 0.1)
    mul_646: "f32[104]" = torch.ops.aten.mul.Tensor(primals_789, 0.9)
    add_509: "f32[104]" = torch.ops.aten.add.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
    squeeze_278: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_542, [0, 2, 3]);  getitem_542 = None
    mul_647: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_278, 1.0006381620931717);  squeeze_278 = None
    mul_648: "f32[104]" = torch.ops.aten.mul.Tensor(mul_647, 0.1);  mul_647 = None
    mul_649: "f32[104]" = torch.ops.aten.mul.Tensor(primals_790, 0.9)
    add_510: "f32[104]" = torch.ops.aten.add.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
    unsqueeze_368: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_278, -1)
    unsqueeze_369: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    mul_650: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_644, unsqueeze_369);  mul_644 = unsqueeze_369 = None
    unsqueeze_370: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_279, -1);  primals_279 = None
    unsqueeze_371: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    add_511: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_650, unsqueeze_371);  mul_650 = unsqueeze_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_89: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_511);  add_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_547: "f32[8, 104, 14, 14]" = split_with_sizes_86[3];  split_with_sizes_86 = None
    cat_17: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_87, relu_88, relu_89, getitem_547], 1);  getitem_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_93: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_17, primals_280, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_512: "i64[]" = torch.ops.aten.add.Tensor(primals_794, 1)
    var_mean_93 = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True)
    getitem_548: "f32[1, 1024, 1, 1]" = var_mean_93[0]
    getitem_549: "f32[1, 1024, 1, 1]" = var_mean_93[1];  var_mean_93 = None
    add_513: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_548, 1e-05)
    rsqrt_93: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_513);  add_513 = None
    sub_93: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_549)
    mul_651: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = None
    squeeze_279: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_549, [0, 2, 3]);  getitem_549 = None
    squeeze_280: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
    mul_652: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_279, 0.1)
    mul_653: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_792, 0.9)
    add_514: "f32[1024]" = torch.ops.aten.add.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
    squeeze_281: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_548, [0, 2, 3]);  getitem_548 = None
    mul_654: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_281, 1.0006381620931717);  squeeze_281 = None
    mul_655: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_654, 0.1);  mul_654 = None
    mul_656: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_793, 0.9)
    add_515: "f32[1024]" = torch.ops.aten.add.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
    unsqueeze_372: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_281, -1)
    unsqueeze_373: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_657: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_651, unsqueeze_373);  mul_651 = unsqueeze_373 = None
    unsqueeze_374: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_282, -1);  primals_282 = None
    unsqueeze_375: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_516: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_657, unsqueeze_375);  mul_657 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_517: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_516, relu_85);  add_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_90: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_517);  add_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_94: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_90, primals_283, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_518: "i64[]" = torch.ops.aten.add.Tensor(primals_797, 1)
    var_mean_94 = torch.ops.aten.var_mean.correction(convolution_94, [0, 2, 3], correction = 0, keepdim = True)
    getitem_550: "f32[1, 416, 1, 1]" = var_mean_94[0]
    getitem_551: "f32[1, 416, 1, 1]" = var_mean_94[1];  var_mean_94 = None
    add_519: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_550, 1e-05)
    rsqrt_94: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_519);  add_519 = None
    sub_94: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, getitem_551)
    mul_658: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = None
    squeeze_282: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_551, [0, 2, 3]);  getitem_551 = None
    squeeze_283: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_94, [0, 2, 3]);  rsqrt_94 = None
    mul_659: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_282, 0.1)
    mul_660: "f32[416]" = torch.ops.aten.mul.Tensor(primals_795, 0.9)
    add_520: "f32[416]" = torch.ops.aten.add.Tensor(mul_659, mul_660);  mul_659 = mul_660 = None
    squeeze_284: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_550, [0, 2, 3]);  getitem_550 = None
    mul_661: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_284, 1.0006381620931717);  squeeze_284 = None
    mul_662: "f32[416]" = torch.ops.aten.mul.Tensor(mul_661, 0.1);  mul_661 = None
    mul_663: "f32[416]" = torch.ops.aten.mul.Tensor(primals_796, 0.9)
    add_521: "f32[416]" = torch.ops.aten.add.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
    unsqueeze_376: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_284, -1)
    unsqueeze_377: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    mul_664: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_377);  mul_658 = unsqueeze_377 = None
    unsqueeze_378: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_285, -1);  primals_285 = None
    unsqueeze_379: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    add_522: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_664, unsqueeze_379);  mul_664 = unsqueeze_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_91: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_522);  add_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_91 = torch.ops.aten.split_with_sizes.default(relu_91, [104, 104, 104, 104], 1)
    getitem_556: "f32[8, 104, 14, 14]" = split_with_sizes_91[0]
    convolution_95: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_556, primals_286, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_523: "i64[]" = torch.ops.aten.add.Tensor(primals_800, 1)
    var_mean_95 = torch.ops.aten.var_mean.correction(convolution_95, [0, 2, 3], correction = 0, keepdim = True)
    getitem_560: "f32[1, 104, 1, 1]" = var_mean_95[0]
    getitem_561: "f32[1, 104, 1, 1]" = var_mean_95[1];  var_mean_95 = None
    add_524: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_560, 1e-05)
    rsqrt_95: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_524);  add_524 = None
    sub_95: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, getitem_561)
    mul_665: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = None
    squeeze_285: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_561, [0, 2, 3]);  getitem_561 = None
    squeeze_286: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_95, [0, 2, 3]);  rsqrt_95 = None
    mul_666: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_285, 0.1)
    mul_667: "f32[104]" = torch.ops.aten.mul.Tensor(primals_798, 0.9)
    add_525: "f32[104]" = torch.ops.aten.add.Tensor(mul_666, mul_667);  mul_666 = mul_667 = None
    squeeze_287: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_560, [0, 2, 3]);  getitem_560 = None
    mul_668: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_287, 1.0006381620931717);  squeeze_287 = None
    mul_669: "f32[104]" = torch.ops.aten.mul.Tensor(mul_668, 0.1);  mul_668 = None
    mul_670: "f32[104]" = torch.ops.aten.mul.Tensor(primals_799, 0.9)
    add_526: "f32[104]" = torch.ops.aten.add.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    unsqueeze_380: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_287, -1)
    unsqueeze_381: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_671: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_665, unsqueeze_381);  mul_665 = unsqueeze_381 = None
    unsqueeze_382: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_288, -1);  primals_288 = None
    unsqueeze_383: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_527: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_671, unsqueeze_383);  mul_671 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_92: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_527);  add_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_563: "f32[8, 104, 14, 14]" = split_with_sizes_91[1]
    add_528: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_92, getitem_563);  getitem_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_96: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_528, primals_289, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_529: "i64[]" = torch.ops.aten.add.Tensor(primals_803, 1)
    var_mean_96 = torch.ops.aten.var_mean.correction(convolution_96, [0, 2, 3], correction = 0, keepdim = True)
    getitem_566: "f32[1, 104, 1, 1]" = var_mean_96[0]
    getitem_567: "f32[1, 104, 1, 1]" = var_mean_96[1];  var_mean_96 = None
    add_530: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_566, 1e-05)
    rsqrt_96: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_530);  add_530 = None
    sub_96: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_96, getitem_567)
    mul_672: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = None
    squeeze_288: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_567, [0, 2, 3]);  getitem_567 = None
    squeeze_289: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_96, [0, 2, 3]);  rsqrt_96 = None
    mul_673: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_288, 0.1)
    mul_674: "f32[104]" = torch.ops.aten.mul.Tensor(primals_801, 0.9)
    add_531: "f32[104]" = torch.ops.aten.add.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
    squeeze_290: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_566, [0, 2, 3]);  getitem_566 = None
    mul_675: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_290, 1.0006381620931717);  squeeze_290 = None
    mul_676: "f32[104]" = torch.ops.aten.mul.Tensor(mul_675, 0.1);  mul_675 = None
    mul_677: "f32[104]" = torch.ops.aten.mul.Tensor(primals_802, 0.9)
    add_532: "f32[104]" = torch.ops.aten.add.Tensor(mul_676, mul_677);  mul_676 = mul_677 = None
    unsqueeze_384: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_290, -1)
    unsqueeze_385: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    mul_678: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_672, unsqueeze_385);  mul_672 = unsqueeze_385 = None
    unsqueeze_386: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_291, -1);  primals_291 = None
    unsqueeze_387: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    add_533: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_678, unsqueeze_387);  mul_678 = unsqueeze_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_93: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_533);  add_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_570: "f32[8, 104, 14, 14]" = split_with_sizes_91[2]
    add_534: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_93, getitem_570);  getitem_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_97: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_534, primals_292, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_535: "i64[]" = torch.ops.aten.add.Tensor(primals_806, 1)
    var_mean_97 = torch.ops.aten.var_mean.correction(convolution_97, [0, 2, 3], correction = 0, keepdim = True)
    getitem_572: "f32[1, 104, 1, 1]" = var_mean_97[0]
    getitem_573: "f32[1, 104, 1, 1]" = var_mean_97[1];  var_mean_97 = None
    add_536: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_572, 1e-05)
    rsqrt_97: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_536);  add_536 = None
    sub_97: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_97, getitem_573)
    mul_679: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = None
    squeeze_291: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_573, [0, 2, 3]);  getitem_573 = None
    squeeze_292: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_97, [0, 2, 3]);  rsqrt_97 = None
    mul_680: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_291, 0.1)
    mul_681: "f32[104]" = torch.ops.aten.mul.Tensor(primals_804, 0.9)
    add_537: "f32[104]" = torch.ops.aten.add.Tensor(mul_680, mul_681);  mul_680 = mul_681 = None
    squeeze_293: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_572, [0, 2, 3]);  getitem_572 = None
    mul_682: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_293, 1.0006381620931717);  squeeze_293 = None
    mul_683: "f32[104]" = torch.ops.aten.mul.Tensor(mul_682, 0.1);  mul_682 = None
    mul_684: "f32[104]" = torch.ops.aten.mul.Tensor(primals_805, 0.9)
    add_538: "f32[104]" = torch.ops.aten.add.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
    unsqueeze_388: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_293, -1)
    unsqueeze_389: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_685: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_679, unsqueeze_389);  mul_679 = unsqueeze_389 = None
    unsqueeze_390: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_294, -1);  primals_294 = None
    unsqueeze_391: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_539: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_685, unsqueeze_391);  mul_685 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_94: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_539);  add_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_577: "f32[8, 104, 14, 14]" = split_with_sizes_91[3];  split_with_sizes_91 = None
    cat_18: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_92, relu_93, relu_94, getitem_577], 1);  getitem_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_98: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_18, primals_295, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_540: "i64[]" = torch.ops.aten.add.Tensor(primals_809, 1)
    var_mean_98 = torch.ops.aten.var_mean.correction(convolution_98, [0, 2, 3], correction = 0, keepdim = True)
    getitem_578: "f32[1, 1024, 1, 1]" = var_mean_98[0]
    getitem_579: "f32[1, 1024, 1, 1]" = var_mean_98[1];  var_mean_98 = None
    add_541: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_578, 1e-05)
    rsqrt_98: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_541);  add_541 = None
    sub_98: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, getitem_579)
    mul_686: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = None
    squeeze_294: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_579, [0, 2, 3]);  getitem_579 = None
    squeeze_295: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_98, [0, 2, 3]);  rsqrt_98 = None
    mul_687: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_294, 0.1)
    mul_688: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_807, 0.9)
    add_542: "f32[1024]" = torch.ops.aten.add.Tensor(mul_687, mul_688);  mul_687 = mul_688 = None
    squeeze_296: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_578, [0, 2, 3]);  getitem_578 = None
    mul_689: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_296, 1.0006381620931717);  squeeze_296 = None
    mul_690: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_689, 0.1);  mul_689 = None
    mul_691: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_808, 0.9)
    add_543: "f32[1024]" = torch.ops.aten.add.Tensor(mul_690, mul_691);  mul_690 = mul_691 = None
    unsqueeze_392: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_296, -1)
    unsqueeze_393: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    mul_692: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_686, unsqueeze_393);  mul_686 = unsqueeze_393 = None
    unsqueeze_394: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_297, -1);  primals_297 = None
    unsqueeze_395: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    add_544: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_692, unsqueeze_395);  mul_692 = unsqueeze_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_545: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_544, relu_90);  add_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_95: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_545);  add_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_99: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_95, primals_298, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_546: "i64[]" = torch.ops.aten.add.Tensor(primals_812, 1)
    var_mean_99 = torch.ops.aten.var_mean.correction(convolution_99, [0, 2, 3], correction = 0, keepdim = True)
    getitem_580: "f32[1, 416, 1, 1]" = var_mean_99[0]
    getitem_581: "f32[1, 416, 1, 1]" = var_mean_99[1];  var_mean_99 = None
    add_547: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_580, 1e-05)
    rsqrt_99: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_547);  add_547 = None
    sub_99: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_99, getitem_581)
    mul_693: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = None
    squeeze_297: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_581, [0, 2, 3]);  getitem_581 = None
    squeeze_298: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_99, [0, 2, 3]);  rsqrt_99 = None
    mul_694: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_297, 0.1)
    mul_695: "f32[416]" = torch.ops.aten.mul.Tensor(primals_810, 0.9)
    add_548: "f32[416]" = torch.ops.aten.add.Tensor(mul_694, mul_695);  mul_694 = mul_695 = None
    squeeze_299: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_580, [0, 2, 3]);  getitem_580 = None
    mul_696: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_299, 1.0006381620931717);  squeeze_299 = None
    mul_697: "f32[416]" = torch.ops.aten.mul.Tensor(mul_696, 0.1);  mul_696 = None
    mul_698: "f32[416]" = torch.ops.aten.mul.Tensor(primals_811, 0.9)
    add_549: "f32[416]" = torch.ops.aten.add.Tensor(mul_697, mul_698);  mul_697 = mul_698 = None
    unsqueeze_396: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_299, -1)
    unsqueeze_397: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_699: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_693, unsqueeze_397);  mul_693 = unsqueeze_397 = None
    unsqueeze_398: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_300, -1);  primals_300 = None
    unsqueeze_399: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_550: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_699, unsqueeze_399);  mul_699 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_96: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_550);  add_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_96 = torch.ops.aten.split_with_sizes.default(relu_96, [104, 104, 104, 104], 1)
    getitem_586: "f32[8, 104, 14, 14]" = split_with_sizes_96[0]
    convolution_100: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_586, primals_301, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_551: "i64[]" = torch.ops.aten.add.Tensor(primals_815, 1)
    var_mean_100 = torch.ops.aten.var_mean.correction(convolution_100, [0, 2, 3], correction = 0, keepdim = True)
    getitem_590: "f32[1, 104, 1, 1]" = var_mean_100[0]
    getitem_591: "f32[1, 104, 1, 1]" = var_mean_100[1];  var_mean_100 = None
    add_552: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_590, 1e-05)
    rsqrt_100: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_552);  add_552 = None
    sub_100: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_100, getitem_591)
    mul_700: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = None
    squeeze_300: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_591, [0, 2, 3]);  getitem_591 = None
    squeeze_301: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_100, [0, 2, 3]);  rsqrt_100 = None
    mul_701: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_300, 0.1)
    mul_702: "f32[104]" = torch.ops.aten.mul.Tensor(primals_813, 0.9)
    add_553: "f32[104]" = torch.ops.aten.add.Tensor(mul_701, mul_702);  mul_701 = mul_702 = None
    squeeze_302: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_590, [0, 2, 3]);  getitem_590 = None
    mul_703: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_302, 1.0006381620931717);  squeeze_302 = None
    mul_704: "f32[104]" = torch.ops.aten.mul.Tensor(mul_703, 0.1);  mul_703 = None
    mul_705: "f32[104]" = torch.ops.aten.mul.Tensor(primals_814, 0.9)
    add_554: "f32[104]" = torch.ops.aten.add.Tensor(mul_704, mul_705);  mul_704 = mul_705 = None
    unsqueeze_400: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_302, -1)
    unsqueeze_401: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    mul_706: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_700, unsqueeze_401);  mul_700 = unsqueeze_401 = None
    unsqueeze_402: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_303, -1);  primals_303 = None
    unsqueeze_403: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    add_555: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_706, unsqueeze_403);  mul_706 = unsqueeze_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_97: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_555);  add_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_593: "f32[8, 104, 14, 14]" = split_with_sizes_96[1]
    add_556: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_97, getitem_593);  getitem_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_101: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_556, primals_304, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_557: "i64[]" = torch.ops.aten.add.Tensor(primals_818, 1)
    var_mean_101 = torch.ops.aten.var_mean.correction(convolution_101, [0, 2, 3], correction = 0, keepdim = True)
    getitem_596: "f32[1, 104, 1, 1]" = var_mean_101[0]
    getitem_597: "f32[1, 104, 1, 1]" = var_mean_101[1];  var_mean_101 = None
    add_558: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_596, 1e-05)
    rsqrt_101: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_558);  add_558 = None
    sub_101: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_101, getitem_597)
    mul_707: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = None
    squeeze_303: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_597, [0, 2, 3]);  getitem_597 = None
    squeeze_304: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_101, [0, 2, 3]);  rsqrt_101 = None
    mul_708: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_303, 0.1)
    mul_709: "f32[104]" = torch.ops.aten.mul.Tensor(primals_816, 0.9)
    add_559: "f32[104]" = torch.ops.aten.add.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
    squeeze_305: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_596, [0, 2, 3]);  getitem_596 = None
    mul_710: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_305, 1.0006381620931717);  squeeze_305 = None
    mul_711: "f32[104]" = torch.ops.aten.mul.Tensor(mul_710, 0.1);  mul_710 = None
    mul_712: "f32[104]" = torch.ops.aten.mul.Tensor(primals_817, 0.9)
    add_560: "f32[104]" = torch.ops.aten.add.Tensor(mul_711, mul_712);  mul_711 = mul_712 = None
    unsqueeze_404: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_305, -1)
    unsqueeze_405: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_713: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_707, unsqueeze_405);  mul_707 = unsqueeze_405 = None
    unsqueeze_406: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_306, -1);  primals_306 = None
    unsqueeze_407: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_561: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_713, unsqueeze_407);  mul_713 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_98: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_561);  add_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_600: "f32[8, 104, 14, 14]" = split_with_sizes_96[2]
    add_562: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_98, getitem_600);  getitem_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_102: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_562, primals_307, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_563: "i64[]" = torch.ops.aten.add.Tensor(primals_821, 1)
    var_mean_102 = torch.ops.aten.var_mean.correction(convolution_102, [0, 2, 3], correction = 0, keepdim = True)
    getitem_602: "f32[1, 104, 1, 1]" = var_mean_102[0]
    getitem_603: "f32[1, 104, 1, 1]" = var_mean_102[1];  var_mean_102 = None
    add_564: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_602, 1e-05)
    rsqrt_102: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_564);  add_564 = None
    sub_102: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_102, getitem_603)
    mul_714: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = None
    squeeze_306: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_603, [0, 2, 3]);  getitem_603 = None
    squeeze_307: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_102, [0, 2, 3]);  rsqrt_102 = None
    mul_715: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_306, 0.1)
    mul_716: "f32[104]" = torch.ops.aten.mul.Tensor(primals_819, 0.9)
    add_565: "f32[104]" = torch.ops.aten.add.Tensor(mul_715, mul_716);  mul_715 = mul_716 = None
    squeeze_308: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_602, [0, 2, 3]);  getitem_602 = None
    mul_717: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_308, 1.0006381620931717);  squeeze_308 = None
    mul_718: "f32[104]" = torch.ops.aten.mul.Tensor(mul_717, 0.1);  mul_717 = None
    mul_719: "f32[104]" = torch.ops.aten.mul.Tensor(primals_820, 0.9)
    add_566: "f32[104]" = torch.ops.aten.add.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
    unsqueeze_408: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_308, -1)
    unsqueeze_409: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    mul_720: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_714, unsqueeze_409);  mul_714 = unsqueeze_409 = None
    unsqueeze_410: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_309, -1);  primals_309 = None
    unsqueeze_411: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    add_567: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_720, unsqueeze_411);  mul_720 = unsqueeze_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_99: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_567);  add_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_607: "f32[8, 104, 14, 14]" = split_with_sizes_96[3];  split_with_sizes_96 = None
    cat_19: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_97, relu_98, relu_99, getitem_607], 1);  getitem_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_103: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_19, primals_310, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_568: "i64[]" = torch.ops.aten.add.Tensor(primals_824, 1)
    var_mean_103 = torch.ops.aten.var_mean.correction(convolution_103, [0, 2, 3], correction = 0, keepdim = True)
    getitem_608: "f32[1, 1024, 1, 1]" = var_mean_103[0]
    getitem_609: "f32[1, 1024, 1, 1]" = var_mean_103[1];  var_mean_103 = None
    add_569: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_608, 1e-05)
    rsqrt_103: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_569);  add_569 = None
    sub_103: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_103, getitem_609)
    mul_721: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = None
    squeeze_309: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_609, [0, 2, 3]);  getitem_609 = None
    squeeze_310: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_103, [0, 2, 3]);  rsqrt_103 = None
    mul_722: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_309, 0.1)
    mul_723: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_822, 0.9)
    add_570: "f32[1024]" = torch.ops.aten.add.Tensor(mul_722, mul_723);  mul_722 = mul_723 = None
    squeeze_311: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_608, [0, 2, 3]);  getitem_608 = None
    mul_724: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_311, 1.0006381620931717);  squeeze_311 = None
    mul_725: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_724, 0.1);  mul_724 = None
    mul_726: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_823, 0.9)
    add_571: "f32[1024]" = torch.ops.aten.add.Tensor(mul_725, mul_726);  mul_725 = mul_726 = None
    unsqueeze_412: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_311, -1)
    unsqueeze_413: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_727: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_721, unsqueeze_413);  mul_721 = unsqueeze_413 = None
    unsqueeze_414: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_312, -1);  primals_312 = None
    unsqueeze_415: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_572: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_727, unsqueeze_415);  mul_727 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_573: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_572, relu_95);  add_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_100: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_573);  add_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_104: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_100, primals_313, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_574: "i64[]" = torch.ops.aten.add.Tensor(primals_827, 1)
    var_mean_104 = torch.ops.aten.var_mean.correction(convolution_104, [0, 2, 3], correction = 0, keepdim = True)
    getitem_610: "f32[1, 416, 1, 1]" = var_mean_104[0]
    getitem_611: "f32[1, 416, 1, 1]" = var_mean_104[1];  var_mean_104 = None
    add_575: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_610, 1e-05)
    rsqrt_104: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_575);  add_575 = None
    sub_104: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_104, getitem_611)
    mul_728: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_104);  sub_104 = None
    squeeze_312: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_611, [0, 2, 3]);  getitem_611 = None
    squeeze_313: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_104, [0, 2, 3]);  rsqrt_104 = None
    mul_729: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_312, 0.1)
    mul_730: "f32[416]" = torch.ops.aten.mul.Tensor(primals_825, 0.9)
    add_576: "f32[416]" = torch.ops.aten.add.Tensor(mul_729, mul_730);  mul_729 = mul_730 = None
    squeeze_314: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_610, [0, 2, 3]);  getitem_610 = None
    mul_731: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_314, 1.0006381620931717);  squeeze_314 = None
    mul_732: "f32[416]" = torch.ops.aten.mul.Tensor(mul_731, 0.1);  mul_731 = None
    mul_733: "f32[416]" = torch.ops.aten.mul.Tensor(primals_826, 0.9)
    add_577: "f32[416]" = torch.ops.aten.add.Tensor(mul_732, mul_733);  mul_732 = mul_733 = None
    unsqueeze_416: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_314, -1)
    unsqueeze_417: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    mul_734: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_728, unsqueeze_417);  mul_728 = unsqueeze_417 = None
    unsqueeze_418: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_315, -1);  primals_315 = None
    unsqueeze_419: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    add_578: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_734, unsqueeze_419);  mul_734 = unsqueeze_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_101: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_578);  add_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_101 = torch.ops.aten.split_with_sizes.default(relu_101, [104, 104, 104, 104], 1)
    getitem_616: "f32[8, 104, 14, 14]" = split_with_sizes_101[0]
    convolution_105: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_616, primals_316, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_579: "i64[]" = torch.ops.aten.add.Tensor(primals_830, 1)
    var_mean_105 = torch.ops.aten.var_mean.correction(convolution_105, [0, 2, 3], correction = 0, keepdim = True)
    getitem_620: "f32[1, 104, 1, 1]" = var_mean_105[0]
    getitem_621: "f32[1, 104, 1, 1]" = var_mean_105[1];  var_mean_105 = None
    add_580: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_620, 1e-05)
    rsqrt_105: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_580);  add_580 = None
    sub_105: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_105, getitem_621)
    mul_735: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_105);  sub_105 = None
    squeeze_315: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_621, [0, 2, 3]);  getitem_621 = None
    squeeze_316: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_105, [0, 2, 3]);  rsqrt_105 = None
    mul_736: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_315, 0.1)
    mul_737: "f32[104]" = torch.ops.aten.mul.Tensor(primals_828, 0.9)
    add_581: "f32[104]" = torch.ops.aten.add.Tensor(mul_736, mul_737);  mul_736 = mul_737 = None
    squeeze_317: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_620, [0, 2, 3]);  getitem_620 = None
    mul_738: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_317, 1.0006381620931717);  squeeze_317 = None
    mul_739: "f32[104]" = torch.ops.aten.mul.Tensor(mul_738, 0.1);  mul_738 = None
    mul_740: "f32[104]" = torch.ops.aten.mul.Tensor(primals_829, 0.9)
    add_582: "f32[104]" = torch.ops.aten.add.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
    unsqueeze_420: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_317, -1)
    unsqueeze_421: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_741: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_735, unsqueeze_421);  mul_735 = unsqueeze_421 = None
    unsqueeze_422: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_318, -1);  primals_318 = None
    unsqueeze_423: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_583: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_741, unsqueeze_423);  mul_741 = unsqueeze_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_102: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_583);  add_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_623: "f32[8, 104, 14, 14]" = split_with_sizes_101[1]
    add_584: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_102, getitem_623);  getitem_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_106: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_584, primals_319, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_585: "i64[]" = torch.ops.aten.add.Tensor(primals_833, 1)
    var_mean_106 = torch.ops.aten.var_mean.correction(convolution_106, [0, 2, 3], correction = 0, keepdim = True)
    getitem_626: "f32[1, 104, 1, 1]" = var_mean_106[0]
    getitem_627: "f32[1, 104, 1, 1]" = var_mean_106[1];  var_mean_106 = None
    add_586: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_626, 1e-05)
    rsqrt_106: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_586);  add_586 = None
    sub_106: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_106, getitem_627)
    mul_742: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_106);  sub_106 = None
    squeeze_318: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_627, [0, 2, 3]);  getitem_627 = None
    squeeze_319: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_106, [0, 2, 3]);  rsqrt_106 = None
    mul_743: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_318, 0.1)
    mul_744: "f32[104]" = torch.ops.aten.mul.Tensor(primals_831, 0.9)
    add_587: "f32[104]" = torch.ops.aten.add.Tensor(mul_743, mul_744);  mul_743 = mul_744 = None
    squeeze_320: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_626, [0, 2, 3]);  getitem_626 = None
    mul_745: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_320, 1.0006381620931717);  squeeze_320 = None
    mul_746: "f32[104]" = torch.ops.aten.mul.Tensor(mul_745, 0.1);  mul_745 = None
    mul_747: "f32[104]" = torch.ops.aten.mul.Tensor(primals_832, 0.9)
    add_588: "f32[104]" = torch.ops.aten.add.Tensor(mul_746, mul_747);  mul_746 = mul_747 = None
    unsqueeze_424: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_320, -1)
    unsqueeze_425: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    mul_748: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_742, unsqueeze_425);  mul_742 = unsqueeze_425 = None
    unsqueeze_426: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_321, -1);  primals_321 = None
    unsqueeze_427: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    add_589: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_748, unsqueeze_427);  mul_748 = unsqueeze_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_103: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_589);  add_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_630: "f32[8, 104, 14, 14]" = split_with_sizes_101[2]
    add_590: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_103, getitem_630);  getitem_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_107: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_590, primals_322, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_591: "i64[]" = torch.ops.aten.add.Tensor(primals_836, 1)
    var_mean_107 = torch.ops.aten.var_mean.correction(convolution_107, [0, 2, 3], correction = 0, keepdim = True)
    getitem_632: "f32[1, 104, 1, 1]" = var_mean_107[0]
    getitem_633: "f32[1, 104, 1, 1]" = var_mean_107[1];  var_mean_107 = None
    add_592: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_632, 1e-05)
    rsqrt_107: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_592);  add_592 = None
    sub_107: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_107, getitem_633)
    mul_749: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_107);  sub_107 = None
    squeeze_321: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_633, [0, 2, 3]);  getitem_633 = None
    squeeze_322: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_107, [0, 2, 3]);  rsqrt_107 = None
    mul_750: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_321, 0.1)
    mul_751: "f32[104]" = torch.ops.aten.mul.Tensor(primals_834, 0.9)
    add_593: "f32[104]" = torch.ops.aten.add.Tensor(mul_750, mul_751);  mul_750 = mul_751 = None
    squeeze_323: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_632, [0, 2, 3]);  getitem_632 = None
    mul_752: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_323, 1.0006381620931717);  squeeze_323 = None
    mul_753: "f32[104]" = torch.ops.aten.mul.Tensor(mul_752, 0.1);  mul_752 = None
    mul_754: "f32[104]" = torch.ops.aten.mul.Tensor(primals_835, 0.9)
    add_594: "f32[104]" = torch.ops.aten.add.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
    unsqueeze_428: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_323, -1)
    unsqueeze_429: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_755: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_749, unsqueeze_429);  mul_749 = unsqueeze_429 = None
    unsqueeze_430: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_324, -1);  primals_324 = None
    unsqueeze_431: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_595: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_755, unsqueeze_431);  mul_755 = unsqueeze_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_104: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_595);  add_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_637: "f32[8, 104, 14, 14]" = split_with_sizes_101[3];  split_with_sizes_101 = None
    cat_20: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_102, relu_103, relu_104, getitem_637], 1);  getitem_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_108: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_20, primals_325, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_596: "i64[]" = torch.ops.aten.add.Tensor(primals_839, 1)
    var_mean_108 = torch.ops.aten.var_mean.correction(convolution_108, [0, 2, 3], correction = 0, keepdim = True)
    getitem_638: "f32[1, 1024, 1, 1]" = var_mean_108[0]
    getitem_639: "f32[1, 1024, 1, 1]" = var_mean_108[1];  var_mean_108 = None
    add_597: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_638, 1e-05)
    rsqrt_108: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_597);  add_597 = None
    sub_108: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_108, getitem_639)
    mul_756: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_108);  sub_108 = None
    squeeze_324: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_639, [0, 2, 3]);  getitem_639 = None
    squeeze_325: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_108, [0, 2, 3]);  rsqrt_108 = None
    mul_757: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_324, 0.1)
    mul_758: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_837, 0.9)
    add_598: "f32[1024]" = torch.ops.aten.add.Tensor(mul_757, mul_758);  mul_757 = mul_758 = None
    squeeze_326: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_638, [0, 2, 3]);  getitem_638 = None
    mul_759: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_326, 1.0006381620931717);  squeeze_326 = None
    mul_760: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_759, 0.1);  mul_759 = None
    mul_761: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_838, 0.9)
    add_599: "f32[1024]" = torch.ops.aten.add.Tensor(mul_760, mul_761);  mul_760 = mul_761 = None
    unsqueeze_432: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_326, -1)
    unsqueeze_433: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    mul_762: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_756, unsqueeze_433);  mul_756 = unsqueeze_433 = None
    unsqueeze_434: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_327, -1);  primals_327 = None
    unsqueeze_435: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    add_600: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_762, unsqueeze_435);  mul_762 = unsqueeze_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_601: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_600, relu_100);  add_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_105: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_601);  add_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_109: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_105, primals_328, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_602: "i64[]" = torch.ops.aten.add.Tensor(primals_842, 1)
    var_mean_109 = torch.ops.aten.var_mean.correction(convolution_109, [0, 2, 3], correction = 0, keepdim = True)
    getitem_640: "f32[1, 416, 1, 1]" = var_mean_109[0]
    getitem_641: "f32[1, 416, 1, 1]" = var_mean_109[1];  var_mean_109 = None
    add_603: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_640, 1e-05)
    rsqrt_109: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_603);  add_603 = None
    sub_109: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_109, getitem_641)
    mul_763: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_109);  sub_109 = None
    squeeze_327: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_641, [0, 2, 3]);  getitem_641 = None
    squeeze_328: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_109, [0, 2, 3]);  rsqrt_109 = None
    mul_764: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_327, 0.1)
    mul_765: "f32[416]" = torch.ops.aten.mul.Tensor(primals_840, 0.9)
    add_604: "f32[416]" = torch.ops.aten.add.Tensor(mul_764, mul_765);  mul_764 = mul_765 = None
    squeeze_329: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_640, [0, 2, 3]);  getitem_640 = None
    mul_766: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_329, 1.0006381620931717);  squeeze_329 = None
    mul_767: "f32[416]" = torch.ops.aten.mul.Tensor(mul_766, 0.1);  mul_766 = None
    mul_768: "f32[416]" = torch.ops.aten.mul.Tensor(primals_841, 0.9)
    add_605: "f32[416]" = torch.ops.aten.add.Tensor(mul_767, mul_768);  mul_767 = mul_768 = None
    unsqueeze_436: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_329, -1)
    unsqueeze_437: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_769: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_763, unsqueeze_437);  mul_763 = unsqueeze_437 = None
    unsqueeze_438: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_330, -1);  primals_330 = None
    unsqueeze_439: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_606: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_769, unsqueeze_439);  mul_769 = unsqueeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_106: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_606);  add_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_106 = torch.ops.aten.split_with_sizes.default(relu_106, [104, 104, 104, 104], 1)
    getitem_646: "f32[8, 104, 14, 14]" = split_with_sizes_106[0]
    convolution_110: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_646, primals_331, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_607: "i64[]" = torch.ops.aten.add.Tensor(primals_845, 1)
    var_mean_110 = torch.ops.aten.var_mean.correction(convolution_110, [0, 2, 3], correction = 0, keepdim = True)
    getitem_650: "f32[1, 104, 1, 1]" = var_mean_110[0]
    getitem_651: "f32[1, 104, 1, 1]" = var_mean_110[1];  var_mean_110 = None
    add_608: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_650, 1e-05)
    rsqrt_110: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_608);  add_608 = None
    sub_110: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_110, getitem_651)
    mul_770: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_110);  sub_110 = None
    squeeze_330: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_651, [0, 2, 3]);  getitem_651 = None
    squeeze_331: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_110, [0, 2, 3]);  rsqrt_110 = None
    mul_771: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_330, 0.1)
    mul_772: "f32[104]" = torch.ops.aten.mul.Tensor(primals_843, 0.9)
    add_609: "f32[104]" = torch.ops.aten.add.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
    squeeze_332: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_650, [0, 2, 3]);  getitem_650 = None
    mul_773: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_332, 1.0006381620931717);  squeeze_332 = None
    mul_774: "f32[104]" = torch.ops.aten.mul.Tensor(mul_773, 0.1);  mul_773 = None
    mul_775: "f32[104]" = torch.ops.aten.mul.Tensor(primals_844, 0.9)
    add_610: "f32[104]" = torch.ops.aten.add.Tensor(mul_774, mul_775);  mul_774 = mul_775 = None
    unsqueeze_440: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_332, -1)
    unsqueeze_441: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    mul_776: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_770, unsqueeze_441);  mul_770 = unsqueeze_441 = None
    unsqueeze_442: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_333, -1);  primals_333 = None
    unsqueeze_443: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    add_611: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_776, unsqueeze_443);  mul_776 = unsqueeze_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_107: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_611);  add_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_653: "f32[8, 104, 14, 14]" = split_with_sizes_106[1]
    add_612: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_107, getitem_653);  getitem_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_111: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_612, primals_334, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_613: "i64[]" = torch.ops.aten.add.Tensor(primals_848, 1)
    var_mean_111 = torch.ops.aten.var_mean.correction(convolution_111, [0, 2, 3], correction = 0, keepdim = True)
    getitem_656: "f32[1, 104, 1, 1]" = var_mean_111[0]
    getitem_657: "f32[1, 104, 1, 1]" = var_mean_111[1];  var_mean_111 = None
    add_614: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_656, 1e-05)
    rsqrt_111: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_614);  add_614 = None
    sub_111: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_111, getitem_657)
    mul_777: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_111);  sub_111 = None
    squeeze_333: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_657, [0, 2, 3]);  getitem_657 = None
    squeeze_334: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_111, [0, 2, 3]);  rsqrt_111 = None
    mul_778: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_333, 0.1)
    mul_779: "f32[104]" = torch.ops.aten.mul.Tensor(primals_846, 0.9)
    add_615: "f32[104]" = torch.ops.aten.add.Tensor(mul_778, mul_779);  mul_778 = mul_779 = None
    squeeze_335: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_656, [0, 2, 3]);  getitem_656 = None
    mul_780: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_335, 1.0006381620931717);  squeeze_335 = None
    mul_781: "f32[104]" = torch.ops.aten.mul.Tensor(mul_780, 0.1);  mul_780 = None
    mul_782: "f32[104]" = torch.ops.aten.mul.Tensor(primals_847, 0.9)
    add_616: "f32[104]" = torch.ops.aten.add.Tensor(mul_781, mul_782);  mul_781 = mul_782 = None
    unsqueeze_444: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_335, -1)
    unsqueeze_445: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_783: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_777, unsqueeze_445);  mul_777 = unsqueeze_445 = None
    unsqueeze_446: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_336, -1);  primals_336 = None
    unsqueeze_447: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_617: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_783, unsqueeze_447);  mul_783 = unsqueeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_108: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_617);  add_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_660: "f32[8, 104, 14, 14]" = split_with_sizes_106[2]
    add_618: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_108, getitem_660);  getitem_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_112: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_618, primals_337, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_619: "i64[]" = torch.ops.aten.add.Tensor(primals_851, 1)
    var_mean_112 = torch.ops.aten.var_mean.correction(convolution_112, [0, 2, 3], correction = 0, keepdim = True)
    getitem_662: "f32[1, 104, 1, 1]" = var_mean_112[0]
    getitem_663: "f32[1, 104, 1, 1]" = var_mean_112[1];  var_mean_112 = None
    add_620: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_662, 1e-05)
    rsqrt_112: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_620);  add_620 = None
    sub_112: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_112, getitem_663)
    mul_784: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_112);  sub_112 = None
    squeeze_336: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_663, [0, 2, 3]);  getitem_663 = None
    squeeze_337: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_112, [0, 2, 3]);  rsqrt_112 = None
    mul_785: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_336, 0.1)
    mul_786: "f32[104]" = torch.ops.aten.mul.Tensor(primals_849, 0.9)
    add_621: "f32[104]" = torch.ops.aten.add.Tensor(mul_785, mul_786);  mul_785 = mul_786 = None
    squeeze_338: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_662, [0, 2, 3]);  getitem_662 = None
    mul_787: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_338, 1.0006381620931717);  squeeze_338 = None
    mul_788: "f32[104]" = torch.ops.aten.mul.Tensor(mul_787, 0.1);  mul_787 = None
    mul_789: "f32[104]" = torch.ops.aten.mul.Tensor(primals_850, 0.9)
    add_622: "f32[104]" = torch.ops.aten.add.Tensor(mul_788, mul_789);  mul_788 = mul_789 = None
    unsqueeze_448: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_338, -1)
    unsqueeze_449: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    mul_790: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_784, unsqueeze_449);  mul_784 = unsqueeze_449 = None
    unsqueeze_450: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_339, -1);  primals_339 = None
    unsqueeze_451: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    add_623: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_790, unsqueeze_451);  mul_790 = unsqueeze_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_109: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_623);  add_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_667: "f32[8, 104, 14, 14]" = split_with_sizes_106[3];  split_with_sizes_106 = None
    cat_21: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_107, relu_108, relu_109, getitem_667], 1);  getitem_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_113: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_21, primals_340, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_624: "i64[]" = torch.ops.aten.add.Tensor(primals_854, 1)
    var_mean_113 = torch.ops.aten.var_mean.correction(convolution_113, [0, 2, 3], correction = 0, keepdim = True)
    getitem_668: "f32[1, 1024, 1, 1]" = var_mean_113[0]
    getitem_669: "f32[1, 1024, 1, 1]" = var_mean_113[1];  var_mean_113 = None
    add_625: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_668, 1e-05)
    rsqrt_113: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_625);  add_625 = None
    sub_113: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_113, getitem_669)
    mul_791: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_113);  sub_113 = None
    squeeze_339: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_669, [0, 2, 3]);  getitem_669 = None
    squeeze_340: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_113, [0, 2, 3]);  rsqrt_113 = None
    mul_792: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_339, 0.1)
    mul_793: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_852, 0.9)
    add_626: "f32[1024]" = torch.ops.aten.add.Tensor(mul_792, mul_793);  mul_792 = mul_793 = None
    squeeze_341: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_668, [0, 2, 3]);  getitem_668 = None
    mul_794: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_341, 1.0006381620931717);  squeeze_341 = None
    mul_795: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_794, 0.1);  mul_794 = None
    mul_796: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_853, 0.9)
    add_627: "f32[1024]" = torch.ops.aten.add.Tensor(mul_795, mul_796);  mul_795 = mul_796 = None
    unsqueeze_452: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_341, -1)
    unsqueeze_453: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_797: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_791, unsqueeze_453);  mul_791 = unsqueeze_453 = None
    unsqueeze_454: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_342, -1);  primals_342 = None
    unsqueeze_455: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_628: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_797, unsqueeze_455);  mul_797 = unsqueeze_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_629: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_628, relu_105);  add_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_110: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_629);  add_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_114: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_110, primals_343, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_630: "i64[]" = torch.ops.aten.add.Tensor(primals_857, 1)
    var_mean_114 = torch.ops.aten.var_mean.correction(convolution_114, [0, 2, 3], correction = 0, keepdim = True)
    getitem_670: "f32[1, 416, 1, 1]" = var_mean_114[0]
    getitem_671: "f32[1, 416, 1, 1]" = var_mean_114[1];  var_mean_114 = None
    add_631: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_670, 1e-05)
    rsqrt_114: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_631);  add_631 = None
    sub_114: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_114, getitem_671)
    mul_798: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_114);  sub_114 = None
    squeeze_342: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_671, [0, 2, 3]);  getitem_671 = None
    squeeze_343: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_114, [0, 2, 3]);  rsqrt_114 = None
    mul_799: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_342, 0.1)
    mul_800: "f32[416]" = torch.ops.aten.mul.Tensor(primals_855, 0.9)
    add_632: "f32[416]" = torch.ops.aten.add.Tensor(mul_799, mul_800);  mul_799 = mul_800 = None
    squeeze_344: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_670, [0, 2, 3]);  getitem_670 = None
    mul_801: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_344, 1.0006381620931717);  squeeze_344 = None
    mul_802: "f32[416]" = torch.ops.aten.mul.Tensor(mul_801, 0.1);  mul_801 = None
    mul_803: "f32[416]" = torch.ops.aten.mul.Tensor(primals_856, 0.9)
    add_633: "f32[416]" = torch.ops.aten.add.Tensor(mul_802, mul_803);  mul_802 = mul_803 = None
    unsqueeze_456: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_344, -1)
    unsqueeze_457: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    mul_804: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_798, unsqueeze_457);  mul_798 = unsqueeze_457 = None
    unsqueeze_458: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_345, -1);  primals_345 = None
    unsqueeze_459: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    add_634: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_804, unsqueeze_459);  mul_804 = unsqueeze_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_111: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_634);  add_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_111 = torch.ops.aten.split_with_sizes.default(relu_111, [104, 104, 104, 104], 1)
    getitem_676: "f32[8, 104, 14, 14]" = split_with_sizes_111[0]
    convolution_115: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_676, primals_346, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_635: "i64[]" = torch.ops.aten.add.Tensor(primals_860, 1)
    var_mean_115 = torch.ops.aten.var_mean.correction(convolution_115, [0, 2, 3], correction = 0, keepdim = True)
    getitem_680: "f32[1, 104, 1, 1]" = var_mean_115[0]
    getitem_681: "f32[1, 104, 1, 1]" = var_mean_115[1];  var_mean_115 = None
    add_636: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_680, 1e-05)
    rsqrt_115: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_636);  add_636 = None
    sub_115: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_115, getitem_681)
    mul_805: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_115);  sub_115 = None
    squeeze_345: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_681, [0, 2, 3]);  getitem_681 = None
    squeeze_346: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_115, [0, 2, 3]);  rsqrt_115 = None
    mul_806: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_345, 0.1)
    mul_807: "f32[104]" = torch.ops.aten.mul.Tensor(primals_858, 0.9)
    add_637: "f32[104]" = torch.ops.aten.add.Tensor(mul_806, mul_807);  mul_806 = mul_807 = None
    squeeze_347: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_680, [0, 2, 3]);  getitem_680 = None
    mul_808: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_347, 1.0006381620931717);  squeeze_347 = None
    mul_809: "f32[104]" = torch.ops.aten.mul.Tensor(mul_808, 0.1);  mul_808 = None
    mul_810: "f32[104]" = torch.ops.aten.mul.Tensor(primals_859, 0.9)
    add_638: "f32[104]" = torch.ops.aten.add.Tensor(mul_809, mul_810);  mul_809 = mul_810 = None
    unsqueeze_460: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_347, -1)
    unsqueeze_461: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_811: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_805, unsqueeze_461);  mul_805 = unsqueeze_461 = None
    unsqueeze_462: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_348, -1);  primals_348 = None
    unsqueeze_463: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_639: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_811, unsqueeze_463);  mul_811 = unsqueeze_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_112: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_639);  add_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_683: "f32[8, 104, 14, 14]" = split_with_sizes_111[1]
    add_640: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_112, getitem_683);  getitem_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_116: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_640, primals_349, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_641: "i64[]" = torch.ops.aten.add.Tensor(primals_863, 1)
    var_mean_116 = torch.ops.aten.var_mean.correction(convolution_116, [0, 2, 3], correction = 0, keepdim = True)
    getitem_686: "f32[1, 104, 1, 1]" = var_mean_116[0]
    getitem_687: "f32[1, 104, 1, 1]" = var_mean_116[1];  var_mean_116 = None
    add_642: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_686, 1e-05)
    rsqrt_116: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_642);  add_642 = None
    sub_116: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_116, getitem_687)
    mul_812: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_116);  sub_116 = None
    squeeze_348: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_687, [0, 2, 3]);  getitem_687 = None
    squeeze_349: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_116, [0, 2, 3]);  rsqrt_116 = None
    mul_813: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_348, 0.1)
    mul_814: "f32[104]" = torch.ops.aten.mul.Tensor(primals_861, 0.9)
    add_643: "f32[104]" = torch.ops.aten.add.Tensor(mul_813, mul_814);  mul_813 = mul_814 = None
    squeeze_350: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_686, [0, 2, 3]);  getitem_686 = None
    mul_815: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_350, 1.0006381620931717);  squeeze_350 = None
    mul_816: "f32[104]" = torch.ops.aten.mul.Tensor(mul_815, 0.1);  mul_815 = None
    mul_817: "f32[104]" = torch.ops.aten.mul.Tensor(primals_862, 0.9)
    add_644: "f32[104]" = torch.ops.aten.add.Tensor(mul_816, mul_817);  mul_816 = mul_817 = None
    unsqueeze_464: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_350, -1)
    unsqueeze_465: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
    mul_818: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_812, unsqueeze_465);  mul_812 = unsqueeze_465 = None
    unsqueeze_466: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_351, -1);  primals_351 = None
    unsqueeze_467: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    add_645: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_818, unsqueeze_467);  mul_818 = unsqueeze_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_113: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_645);  add_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_690: "f32[8, 104, 14, 14]" = split_with_sizes_111[2]
    add_646: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_113, getitem_690);  getitem_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_117: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_646, primals_352, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_647: "i64[]" = torch.ops.aten.add.Tensor(primals_866, 1)
    var_mean_117 = torch.ops.aten.var_mean.correction(convolution_117, [0, 2, 3], correction = 0, keepdim = True)
    getitem_692: "f32[1, 104, 1, 1]" = var_mean_117[0]
    getitem_693: "f32[1, 104, 1, 1]" = var_mean_117[1];  var_mean_117 = None
    add_648: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_692, 1e-05)
    rsqrt_117: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_648);  add_648 = None
    sub_117: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_117, getitem_693)
    mul_819: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_117);  sub_117 = None
    squeeze_351: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_693, [0, 2, 3]);  getitem_693 = None
    squeeze_352: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_117, [0, 2, 3]);  rsqrt_117 = None
    mul_820: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_351, 0.1)
    mul_821: "f32[104]" = torch.ops.aten.mul.Tensor(primals_864, 0.9)
    add_649: "f32[104]" = torch.ops.aten.add.Tensor(mul_820, mul_821);  mul_820 = mul_821 = None
    squeeze_353: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_692, [0, 2, 3]);  getitem_692 = None
    mul_822: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_353, 1.0006381620931717);  squeeze_353 = None
    mul_823: "f32[104]" = torch.ops.aten.mul.Tensor(mul_822, 0.1);  mul_822 = None
    mul_824: "f32[104]" = torch.ops.aten.mul.Tensor(primals_865, 0.9)
    add_650: "f32[104]" = torch.ops.aten.add.Tensor(mul_823, mul_824);  mul_823 = mul_824 = None
    unsqueeze_468: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_353, -1)
    unsqueeze_469: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    mul_825: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_819, unsqueeze_469);  mul_819 = unsqueeze_469 = None
    unsqueeze_470: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_354, -1);  primals_354 = None
    unsqueeze_471: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
    add_651: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_825, unsqueeze_471);  mul_825 = unsqueeze_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_114: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_651);  add_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_697: "f32[8, 104, 14, 14]" = split_with_sizes_111[3];  split_with_sizes_111 = None
    cat_22: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_112, relu_113, relu_114, getitem_697], 1);  getitem_697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_118: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_22, primals_355, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_652: "i64[]" = torch.ops.aten.add.Tensor(primals_869, 1)
    var_mean_118 = torch.ops.aten.var_mean.correction(convolution_118, [0, 2, 3], correction = 0, keepdim = True)
    getitem_698: "f32[1, 1024, 1, 1]" = var_mean_118[0]
    getitem_699: "f32[1, 1024, 1, 1]" = var_mean_118[1];  var_mean_118 = None
    add_653: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_698, 1e-05)
    rsqrt_118: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_653);  add_653 = None
    sub_118: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_118, getitem_699)
    mul_826: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_118);  sub_118 = None
    squeeze_354: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_699, [0, 2, 3]);  getitem_699 = None
    squeeze_355: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_118, [0, 2, 3]);  rsqrt_118 = None
    mul_827: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_354, 0.1)
    mul_828: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_867, 0.9)
    add_654: "f32[1024]" = torch.ops.aten.add.Tensor(mul_827, mul_828);  mul_827 = mul_828 = None
    squeeze_356: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_698, [0, 2, 3]);  getitem_698 = None
    mul_829: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_356, 1.0006381620931717);  squeeze_356 = None
    mul_830: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_829, 0.1);  mul_829 = None
    mul_831: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_868, 0.9)
    add_655: "f32[1024]" = torch.ops.aten.add.Tensor(mul_830, mul_831);  mul_830 = mul_831 = None
    unsqueeze_472: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_356, -1)
    unsqueeze_473: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
    mul_832: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_826, unsqueeze_473);  mul_826 = unsqueeze_473 = None
    unsqueeze_474: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_357, -1);  primals_357 = None
    unsqueeze_475: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
    add_656: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_832, unsqueeze_475);  mul_832 = unsqueeze_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_657: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_656, relu_110);  add_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_115: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_657);  add_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_119: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_115, primals_358, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_658: "i64[]" = torch.ops.aten.add.Tensor(primals_872, 1)
    var_mean_119 = torch.ops.aten.var_mean.correction(convolution_119, [0, 2, 3], correction = 0, keepdim = True)
    getitem_700: "f32[1, 416, 1, 1]" = var_mean_119[0]
    getitem_701: "f32[1, 416, 1, 1]" = var_mean_119[1];  var_mean_119 = None
    add_659: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_700, 1e-05)
    rsqrt_119: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_659);  add_659 = None
    sub_119: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, getitem_701)
    mul_833: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_119);  sub_119 = None
    squeeze_357: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_701, [0, 2, 3]);  getitem_701 = None
    squeeze_358: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_119, [0, 2, 3]);  rsqrt_119 = None
    mul_834: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_357, 0.1)
    mul_835: "f32[416]" = torch.ops.aten.mul.Tensor(primals_870, 0.9)
    add_660: "f32[416]" = torch.ops.aten.add.Tensor(mul_834, mul_835);  mul_834 = mul_835 = None
    squeeze_359: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_700, [0, 2, 3]);  getitem_700 = None
    mul_836: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_359, 1.0006381620931717);  squeeze_359 = None
    mul_837: "f32[416]" = torch.ops.aten.mul.Tensor(mul_836, 0.1);  mul_836 = None
    mul_838: "f32[416]" = torch.ops.aten.mul.Tensor(primals_871, 0.9)
    add_661: "f32[416]" = torch.ops.aten.add.Tensor(mul_837, mul_838);  mul_837 = mul_838 = None
    unsqueeze_476: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_359, -1)
    unsqueeze_477: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    mul_839: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_833, unsqueeze_477);  mul_833 = unsqueeze_477 = None
    unsqueeze_478: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_360, -1);  primals_360 = None
    unsqueeze_479: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    add_662: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_839, unsqueeze_479);  mul_839 = unsqueeze_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_116: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_662);  add_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_116 = torch.ops.aten.split_with_sizes.default(relu_116, [104, 104, 104, 104], 1)
    getitem_706: "f32[8, 104, 14, 14]" = split_with_sizes_116[0]
    convolution_120: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_706, primals_361, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_663: "i64[]" = torch.ops.aten.add.Tensor(primals_875, 1)
    var_mean_120 = torch.ops.aten.var_mean.correction(convolution_120, [0, 2, 3], correction = 0, keepdim = True)
    getitem_710: "f32[1, 104, 1, 1]" = var_mean_120[0]
    getitem_711: "f32[1, 104, 1, 1]" = var_mean_120[1];  var_mean_120 = None
    add_664: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_710, 1e-05)
    rsqrt_120: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_664);  add_664 = None
    sub_120: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_120, getitem_711)
    mul_840: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_120);  sub_120 = None
    squeeze_360: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_711, [0, 2, 3]);  getitem_711 = None
    squeeze_361: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_120, [0, 2, 3]);  rsqrt_120 = None
    mul_841: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_360, 0.1)
    mul_842: "f32[104]" = torch.ops.aten.mul.Tensor(primals_873, 0.9)
    add_665: "f32[104]" = torch.ops.aten.add.Tensor(mul_841, mul_842);  mul_841 = mul_842 = None
    squeeze_362: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_710, [0, 2, 3]);  getitem_710 = None
    mul_843: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_362, 1.0006381620931717);  squeeze_362 = None
    mul_844: "f32[104]" = torch.ops.aten.mul.Tensor(mul_843, 0.1);  mul_843 = None
    mul_845: "f32[104]" = torch.ops.aten.mul.Tensor(primals_874, 0.9)
    add_666: "f32[104]" = torch.ops.aten.add.Tensor(mul_844, mul_845);  mul_844 = mul_845 = None
    unsqueeze_480: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_362, -1)
    unsqueeze_481: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
    mul_846: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_840, unsqueeze_481);  mul_840 = unsqueeze_481 = None
    unsqueeze_482: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_363, -1);  primals_363 = None
    unsqueeze_483: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
    add_667: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_846, unsqueeze_483);  mul_846 = unsqueeze_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_117: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_667);  add_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_713: "f32[8, 104, 14, 14]" = split_with_sizes_116[1]
    add_668: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_117, getitem_713);  getitem_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_121: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_668, primals_364, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_669: "i64[]" = torch.ops.aten.add.Tensor(primals_878, 1)
    var_mean_121 = torch.ops.aten.var_mean.correction(convolution_121, [0, 2, 3], correction = 0, keepdim = True)
    getitem_716: "f32[1, 104, 1, 1]" = var_mean_121[0]
    getitem_717: "f32[1, 104, 1, 1]" = var_mean_121[1];  var_mean_121 = None
    add_670: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_716, 1e-05)
    rsqrt_121: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_670);  add_670 = None
    sub_121: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_121, getitem_717)
    mul_847: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_121);  sub_121 = None
    squeeze_363: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_717, [0, 2, 3]);  getitem_717 = None
    squeeze_364: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_121, [0, 2, 3]);  rsqrt_121 = None
    mul_848: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_363, 0.1)
    mul_849: "f32[104]" = torch.ops.aten.mul.Tensor(primals_876, 0.9)
    add_671: "f32[104]" = torch.ops.aten.add.Tensor(mul_848, mul_849);  mul_848 = mul_849 = None
    squeeze_365: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_716, [0, 2, 3]);  getitem_716 = None
    mul_850: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_365, 1.0006381620931717);  squeeze_365 = None
    mul_851: "f32[104]" = torch.ops.aten.mul.Tensor(mul_850, 0.1);  mul_850 = None
    mul_852: "f32[104]" = torch.ops.aten.mul.Tensor(primals_877, 0.9)
    add_672: "f32[104]" = torch.ops.aten.add.Tensor(mul_851, mul_852);  mul_851 = mul_852 = None
    unsqueeze_484: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_365, -1)
    unsqueeze_485: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
    mul_853: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_847, unsqueeze_485);  mul_847 = unsqueeze_485 = None
    unsqueeze_486: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_366, -1);  primals_366 = None
    unsqueeze_487: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    add_673: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_853, unsqueeze_487);  mul_853 = unsqueeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_118: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_673);  add_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_720: "f32[8, 104, 14, 14]" = split_with_sizes_116[2]
    add_674: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_118, getitem_720);  getitem_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_122: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_674, primals_367, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_675: "i64[]" = torch.ops.aten.add.Tensor(primals_881, 1)
    var_mean_122 = torch.ops.aten.var_mean.correction(convolution_122, [0, 2, 3], correction = 0, keepdim = True)
    getitem_722: "f32[1, 104, 1, 1]" = var_mean_122[0]
    getitem_723: "f32[1, 104, 1, 1]" = var_mean_122[1];  var_mean_122 = None
    add_676: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_722, 1e-05)
    rsqrt_122: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_676);  add_676 = None
    sub_122: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_122, getitem_723)
    mul_854: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_122);  sub_122 = None
    squeeze_366: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_723, [0, 2, 3]);  getitem_723 = None
    squeeze_367: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_122, [0, 2, 3]);  rsqrt_122 = None
    mul_855: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_366, 0.1)
    mul_856: "f32[104]" = torch.ops.aten.mul.Tensor(primals_879, 0.9)
    add_677: "f32[104]" = torch.ops.aten.add.Tensor(mul_855, mul_856);  mul_855 = mul_856 = None
    squeeze_368: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_722, [0, 2, 3]);  getitem_722 = None
    mul_857: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_368, 1.0006381620931717);  squeeze_368 = None
    mul_858: "f32[104]" = torch.ops.aten.mul.Tensor(mul_857, 0.1);  mul_857 = None
    mul_859: "f32[104]" = torch.ops.aten.mul.Tensor(primals_880, 0.9)
    add_678: "f32[104]" = torch.ops.aten.add.Tensor(mul_858, mul_859);  mul_858 = mul_859 = None
    unsqueeze_488: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_368, -1)
    unsqueeze_489: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    mul_860: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_854, unsqueeze_489);  mul_854 = unsqueeze_489 = None
    unsqueeze_490: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_369, -1);  primals_369 = None
    unsqueeze_491: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
    add_679: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_860, unsqueeze_491);  mul_860 = unsqueeze_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_119: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_679);  add_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_727: "f32[8, 104, 14, 14]" = split_with_sizes_116[3];  split_with_sizes_116 = None
    cat_23: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_117, relu_118, relu_119, getitem_727], 1);  getitem_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_123: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_23, primals_370, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_680: "i64[]" = torch.ops.aten.add.Tensor(primals_884, 1)
    var_mean_123 = torch.ops.aten.var_mean.correction(convolution_123, [0, 2, 3], correction = 0, keepdim = True)
    getitem_728: "f32[1, 1024, 1, 1]" = var_mean_123[0]
    getitem_729: "f32[1, 1024, 1, 1]" = var_mean_123[1];  var_mean_123 = None
    add_681: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_728, 1e-05)
    rsqrt_123: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_681);  add_681 = None
    sub_123: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_123, getitem_729)
    mul_861: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_123, rsqrt_123);  sub_123 = None
    squeeze_369: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_729, [0, 2, 3]);  getitem_729 = None
    squeeze_370: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_123, [0, 2, 3]);  rsqrt_123 = None
    mul_862: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_369, 0.1)
    mul_863: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_882, 0.9)
    add_682: "f32[1024]" = torch.ops.aten.add.Tensor(mul_862, mul_863);  mul_862 = mul_863 = None
    squeeze_371: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_728, [0, 2, 3]);  getitem_728 = None
    mul_864: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_371, 1.0006381620931717);  squeeze_371 = None
    mul_865: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_864, 0.1);  mul_864 = None
    mul_866: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_883, 0.9)
    add_683: "f32[1024]" = torch.ops.aten.add.Tensor(mul_865, mul_866);  mul_865 = mul_866 = None
    unsqueeze_492: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_371, -1)
    unsqueeze_493: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
    mul_867: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_861, unsqueeze_493);  mul_861 = unsqueeze_493 = None
    unsqueeze_494: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_372, -1);  primals_372 = None
    unsqueeze_495: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
    add_684: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_867, unsqueeze_495);  mul_867 = unsqueeze_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_685: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_684, relu_115);  add_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_120: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_685);  add_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_124: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_120, primals_373, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_686: "i64[]" = torch.ops.aten.add.Tensor(primals_887, 1)
    var_mean_124 = torch.ops.aten.var_mean.correction(convolution_124, [0, 2, 3], correction = 0, keepdim = True)
    getitem_730: "f32[1, 416, 1, 1]" = var_mean_124[0]
    getitem_731: "f32[1, 416, 1, 1]" = var_mean_124[1];  var_mean_124 = None
    add_687: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_730, 1e-05)
    rsqrt_124: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_687);  add_687 = None
    sub_124: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_124, getitem_731)
    mul_868: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_124);  sub_124 = None
    squeeze_372: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_731, [0, 2, 3]);  getitem_731 = None
    squeeze_373: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_124, [0, 2, 3]);  rsqrt_124 = None
    mul_869: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_372, 0.1)
    mul_870: "f32[416]" = torch.ops.aten.mul.Tensor(primals_885, 0.9)
    add_688: "f32[416]" = torch.ops.aten.add.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
    squeeze_374: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_730, [0, 2, 3]);  getitem_730 = None
    mul_871: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_374, 1.0006381620931717);  squeeze_374 = None
    mul_872: "f32[416]" = torch.ops.aten.mul.Tensor(mul_871, 0.1);  mul_871 = None
    mul_873: "f32[416]" = torch.ops.aten.mul.Tensor(primals_886, 0.9)
    add_689: "f32[416]" = torch.ops.aten.add.Tensor(mul_872, mul_873);  mul_872 = mul_873 = None
    unsqueeze_496: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_374, -1)
    unsqueeze_497: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
    mul_874: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_868, unsqueeze_497);  mul_868 = unsqueeze_497 = None
    unsqueeze_498: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_375, -1);  primals_375 = None
    unsqueeze_499: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
    add_690: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_874, unsqueeze_499);  mul_874 = unsqueeze_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_121: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_690);  add_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_121 = torch.ops.aten.split_with_sizes.default(relu_121, [104, 104, 104, 104], 1)
    getitem_736: "f32[8, 104, 14, 14]" = split_with_sizes_121[0]
    convolution_125: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_736, primals_376, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_691: "i64[]" = torch.ops.aten.add.Tensor(primals_890, 1)
    var_mean_125 = torch.ops.aten.var_mean.correction(convolution_125, [0, 2, 3], correction = 0, keepdim = True)
    getitem_740: "f32[1, 104, 1, 1]" = var_mean_125[0]
    getitem_741: "f32[1, 104, 1, 1]" = var_mean_125[1];  var_mean_125 = None
    add_692: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_740, 1e-05)
    rsqrt_125: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_692);  add_692 = None
    sub_125: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_125, getitem_741)
    mul_875: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_125);  sub_125 = None
    squeeze_375: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_741, [0, 2, 3]);  getitem_741 = None
    squeeze_376: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_125, [0, 2, 3]);  rsqrt_125 = None
    mul_876: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_375, 0.1)
    mul_877: "f32[104]" = torch.ops.aten.mul.Tensor(primals_888, 0.9)
    add_693: "f32[104]" = torch.ops.aten.add.Tensor(mul_876, mul_877);  mul_876 = mul_877 = None
    squeeze_377: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_740, [0, 2, 3]);  getitem_740 = None
    mul_878: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_377, 1.0006381620931717);  squeeze_377 = None
    mul_879: "f32[104]" = torch.ops.aten.mul.Tensor(mul_878, 0.1);  mul_878 = None
    mul_880: "f32[104]" = torch.ops.aten.mul.Tensor(primals_889, 0.9)
    add_694: "f32[104]" = torch.ops.aten.add.Tensor(mul_879, mul_880);  mul_879 = mul_880 = None
    unsqueeze_500: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_377, -1)
    unsqueeze_501: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
    mul_881: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_875, unsqueeze_501);  mul_875 = unsqueeze_501 = None
    unsqueeze_502: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_378, -1);  primals_378 = None
    unsqueeze_503: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
    add_695: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_881, unsqueeze_503);  mul_881 = unsqueeze_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_122: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_695);  add_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_743: "f32[8, 104, 14, 14]" = split_with_sizes_121[1]
    add_696: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_122, getitem_743);  getitem_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_126: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_696, primals_379, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_697: "i64[]" = torch.ops.aten.add.Tensor(primals_893, 1)
    var_mean_126 = torch.ops.aten.var_mean.correction(convolution_126, [0, 2, 3], correction = 0, keepdim = True)
    getitem_746: "f32[1, 104, 1, 1]" = var_mean_126[0]
    getitem_747: "f32[1, 104, 1, 1]" = var_mean_126[1];  var_mean_126 = None
    add_698: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_746, 1e-05)
    rsqrt_126: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_698);  add_698 = None
    sub_126: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_126, getitem_747)
    mul_882: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_126);  sub_126 = None
    squeeze_378: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_747, [0, 2, 3]);  getitem_747 = None
    squeeze_379: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_126, [0, 2, 3]);  rsqrt_126 = None
    mul_883: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_378, 0.1)
    mul_884: "f32[104]" = torch.ops.aten.mul.Tensor(primals_891, 0.9)
    add_699: "f32[104]" = torch.ops.aten.add.Tensor(mul_883, mul_884);  mul_883 = mul_884 = None
    squeeze_380: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_746, [0, 2, 3]);  getitem_746 = None
    mul_885: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_380, 1.0006381620931717);  squeeze_380 = None
    mul_886: "f32[104]" = torch.ops.aten.mul.Tensor(mul_885, 0.1);  mul_885 = None
    mul_887: "f32[104]" = torch.ops.aten.mul.Tensor(primals_892, 0.9)
    add_700: "f32[104]" = torch.ops.aten.add.Tensor(mul_886, mul_887);  mul_886 = mul_887 = None
    unsqueeze_504: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_380, -1)
    unsqueeze_505: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
    mul_888: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_882, unsqueeze_505);  mul_882 = unsqueeze_505 = None
    unsqueeze_506: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_381, -1);  primals_381 = None
    unsqueeze_507: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
    add_701: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_888, unsqueeze_507);  mul_888 = unsqueeze_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_123: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_701);  add_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_750: "f32[8, 104, 14, 14]" = split_with_sizes_121[2]
    add_702: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_123, getitem_750);  getitem_750 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_127: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_702, primals_382, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_703: "i64[]" = torch.ops.aten.add.Tensor(primals_896, 1)
    var_mean_127 = torch.ops.aten.var_mean.correction(convolution_127, [0, 2, 3], correction = 0, keepdim = True)
    getitem_752: "f32[1, 104, 1, 1]" = var_mean_127[0]
    getitem_753: "f32[1, 104, 1, 1]" = var_mean_127[1];  var_mean_127 = None
    add_704: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_752, 1e-05)
    rsqrt_127: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_704);  add_704 = None
    sub_127: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_127, getitem_753)
    mul_889: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_127);  sub_127 = None
    squeeze_381: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_753, [0, 2, 3]);  getitem_753 = None
    squeeze_382: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_127, [0, 2, 3]);  rsqrt_127 = None
    mul_890: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_381, 0.1)
    mul_891: "f32[104]" = torch.ops.aten.mul.Tensor(primals_894, 0.9)
    add_705: "f32[104]" = torch.ops.aten.add.Tensor(mul_890, mul_891);  mul_890 = mul_891 = None
    squeeze_383: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_752, [0, 2, 3]);  getitem_752 = None
    mul_892: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_383, 1.0006381620931717);  squeeze_383 = None
    mul_893: "f32[104]" = torch.ops.aten.mul.Tensor(mul_892, 0.1);  mul_892 = None
    mul_894: "f32[104]" = torch.ops.aten.mul.Tensor(primals_895, 0.9)
    add_706: "f32[104]" = torch.ops.aten.add.Tensor(mul_893, mul_894);  mul_893 = mul_894 = None
    unsqueeze_508: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_383, -1)
    unsqueeze_509: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
    mul_895: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_889, unsqueeze_509);  mul_889 = unsqueeze_509 = None
    unsqueeze_510: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_384, -1);  primals_384 = None
    unsqueeze_511: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
    add_707: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_895, unsqueeze_511);  mul_895 = unsqueeze_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_124: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_707);  add_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_757: "f32[8, 104, 14, 14]" = split_with_sizes_121[3];  split_with_sizes_121 = None
    cat_24: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_122, relu_123, relu_124, getitem_757], 1);  getitem_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_128: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_24, primals_385, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_708: "i64[]" = torch.ops.aten.add.Tensor(primals_899, 1)
    var_mean_128 = torch.ops.aten.var_mean.correction(convolution_128, [0, 2, 3], correction = 0, keepdim = True)
    getitem_758: "f32[1, 1024, 1, 1]" = var_mean_128[0]
    getitem_759: "f32[1, 1024, 1, 1]" = var_mean_128[1];  var_mean_128 = None
    add_709: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_758, 1e-05)
    rsqrt_128: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_709);  add_709 = None
    sub_128: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_128, getitem_759)
    mul_896: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_128);  sub_128 = None
    squeeze_384: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_759, [0, 2, 3]);  getitem_759 = None
    squeeze_385: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_128, [0, 2, 3]);  rsqrt_128 = None
    mul_897: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_384, 0.1)
    mul_898: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_897, 0.9)
    add_710: "f32[1024]" = torch.ops.aten.add.Tensor(mul_897, mul_898);  mul_897 = mul_898 = None
    squeeze_386: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_758, [0, 2, 3]);  getitem_758 = None
    mul_899: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_386, 1.0006381620931717);  squeeze_386 = None
    mul_900: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_899, 0.1);  mul_899 = None
    mul_901: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_898, 0.9)
    add_711: "f32[1024]" = torch.ops.aten.add.Tensor(mul_900, mul_901);  mul_900 = mul_901 = None
    unsqueeze_512: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_386, -1)
    unsqueeze_513: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
    mul_902: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_896, unsqueeze_513);  mul_896 = unsqueeze_513 = None
    unsqueeze_514: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_387, -1);  primals_387 = None
    unsqueeze_515: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
    add_712: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_902, unsqueeze_515);  mul_902 = unsqueeze_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_713: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_712, relu_120);  add_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_125: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_713);  add_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_129: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_125, primals_388, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_714: "i64[]" = torch.ops.aten.add.Tensor(primals_902, 1)
    var_mean_129 = torch.ops.aten.var_mean.correction(convolution_129, [0, 2, 3], correction = 0, keepdim = True)
    getitem_760: "f32[1, 416, 1, 1]" = var_mean_129[0]
    getitem_761: "f32[1, 416, 1, 1]" = var_mean_129[1];  var_mean_129 = None
    add_715: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_760, 1e-05)
    rsqrt_129: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_715);  add_715 = None
    sub_129: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_129, getitem_761)
    mul_903: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, rsqrt_129);  sub_129 = None
    squeeze_387: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_761, [0, 2, 3]);  getitem_761 = None
    squeeze_388: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_129, [0, 2, 3]);  rsqrt_129 = None
    mul_904: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_387, 0.1)
    mul_905: "f32[416]" = torch.ops.aten.mul.Tensor(primals_900, 0.9)
    add_716: "f32[416]" = torch.ops.aten.add.Tensor(mul_904, mul_905);  mul_904 = mul_905 = None
    squeeze_389: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_760, [0, 2, 3]);  getitem_760 = None
    mul_906: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_389, 1.0006381620931717);  squeeze_389 = None
    mul_907: "f32[416]" = torch.ops.aten.mul.Tensor(mul_906, 0.1);  mul_906 = None
    mul_908: "f32[416]" = torch.ops.aten.mul.Tensor(primals_901, 0.9)
    add_717: "f32[416]" = torch.ops.aten.add.Tensor(mul_907, mul_908);  mul_907 = mul_908 = None
    unsqueeze_516: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_389, -1)
    unsqueeze_517: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
    mul_909: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_903, unsqueeze_517);  mul_903 = unsqueeze_517 = None
    unsqueeze_518: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_390, -1);  primals_390 = None
    unsqueeze_519: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
    add_718: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_909, unsqueeze_519);  mul_909 = unsqueeze_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_126: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_718);  add_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_126 = torch.ops.aten.split_with_sizes.default(relu_126, [104, 104, 104, 104], 1)
    getitem_766: "f32[8, 104, 14, 14]" = split_with_sizes_126[0]
    convolution_130: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_766, primals_391, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_719: "i64[]" = torch.ops.aten.add.Tensor(primals_905, 1)
    var_mean_130 = torch.ops.aten.var_mean.correction(convolution_130, [0, 2, 3], correction = 0, keepdim = True)
    getitem_770: "f32[1, 104, 1, 1]" = var_mean_130[0]
    getitem_771: "f32[1, 104, 1, 1]" = var_mean_130[1];  var_mean_130 = None
    add_720: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_770, 1e-05)
    rsqrt_130: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_720);  add_720 = None
    sub_130: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_130, getitem_771)
    mul_910: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_130);  sub_130 = None
    squeeze_390: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_771, [0, 2, 3]);  getitem_771 = None
    squeeze_391: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_130, [0, 2, 3]);  rsqrt_130 = None
    mul_911: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_390, 0.1)
    mul_912: "f32[104]" = torch.ops.aten.mul.Tensor(primals_903, 0.9)
    add_721: "f32[104]" = torch.ops.aten.add.Tensor(mul_911, mul_912);  mul_911 = mul_912 = None
    squeeze_392: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_770, [0, 2, 3]);  getitem_770 = None
    mul_913: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_392, 1.0006381620931717);  squeeze_392 = None
    mul_914: "f32[104]" = torch.ops.aten.mul.Tensor(mul_913, 0.1);  mul_913 = None
    mul_915: "f32[104]" = torch.ops.aten.mul.Tensor(primals_904, 0.9)
    add_722: "f32[104]" = torch.ops.aten.add.Tensor(mul_914, mul_915);  mul_914 = mul_915 = None
    unsqueeze_520: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_392, -1)
    unsqueeze_521: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
    mul_916: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_910, unsqueeze_521);  mul_910 = unsqueeze_521 = None
    unsqueeze_522: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_393, -1);  primals_393 = None
    unsqueeze_523: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
    add_723: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_916, unsqueeze_523);  mul_916 = unsqueeze_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_127: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_723);  add_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_773: "f32[8, 104, 14, 14]" = split_with_sizes_126[1]
    add_724: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_127, getitem_773);  getitem_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_131: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_724, primals_394, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_725: "i64[]" = torch.ops.aten.add.Tensor(primals_908, 1)
    var_mean_131 = torch.ops.aten.var_mean.correction(convolution_131, [0, 2, 3], correction = 0, keepdim = True)
    getitem_776: "f32[1, 104, 1, 1]" = var_mean_131[0]
    getitem_777: "f32[1, 104, 1, 1]" = var_mean_131[1];  var_mean_131 = None
    add_726: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_776, 1e-05)
    rsqrt_131: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_726);  add_726 = None
    sub_131: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_131, getitem_777)
    mul_917: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_131);  sub_131 = None
    squeeze_393: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_777, [0, 2, 3]);  getitem_777 = None
    squeeze_394: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_131, [0, 2, 3]);  rsqrt_131 = None
    mul_918: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_393, 0.1)
    mul_919: "f32[104]" = torch.ops.aten.mul.Tensor(primals_906, 0.9)
    add_727: "f32[104]" = torch.ops.aten.add.Tensor(mul_918, mul_919);  mul_918 = mul_919 = None
    squeeze_395: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_776, [0, 2, 3]);  getitem_776 = None
    mul_920: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_395, 1.0006381620931717);  squeeze_395 = None
    mul_921: "f32[104]" = torch.ops.aten.mul.Tensor(mul_920, 0.1);  mul_920 = None
    mul_922: "f32[104]" = torch.ops.aten.mul.Tensor(primals_907, 0.9)
    add_728: "f32[104]" = torch.ops.aten.add.Tensor(mul_921, mul_922);  mul_921 = mul_922 = None
    unsqueeze_524: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_395, -1)
    unsqueeze_525: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
    mul_923: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_917, unsqueeze_525);  mul_917 = unsqueeze_525 = None
    unsqueeze_526: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_396, -1);  primals_396 = None
    unsqueeze_527: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
    add_729: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_923, unsqueeze_527);  mul_923 = unsqueeze_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_128: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_729);  add_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_780: "f32[8, 104, 14, 14]" = split_with_sizes_126[2]
    add_730: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_128, getitem_780);  getitem_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_132: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_730, primals_397, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_731: "i64[]" = torch.ops.aten.add.Tensor(primals_911, 1)
    var_mean_132 = torch.ops.aten.var_mean.correction(convolution_132, [0, 2, 3], correction = 0, keepdim = True)
    getitem_782: "f32[1, 104, 1, 1]" = var_mean_132[0]
    getitem_783: "f32[1, 104, 1, 1]" = var_mean_132[1];  var_mean_132 = None
    add_732: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_782, 1e-05)
    rsqrt_132: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_732);  add_732 = None
    sub_132: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_132, getitem_783)
    mul_924: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, rsqrt_132);  sub_132 = None
    squeeze_396: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_783, [0, 2, 3]);  getitem_783 = None
    squeeze_397: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_132, [0, 2, 3]);  rsqrt_132 = None
    mul_925: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_396, 0.1)
    mul_926: "f32[104]" = torch.ops.aten.mul.Tensor(primals_909, 0.9)
    add_733: "f32[104]" = torch.ops.aten.add.Tensor(mul_925, mul_926);  mul_925 = mul_926 = None
    squeeze_398: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_782, [0, 2, 3]);  getitem_782 = None
    mul_927: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_398, 1.0006381620931717);  squeeze_398 = None
    mul_928: "f32[104]" = torch.ops.aten.mul.Tensor(mul_927, 0.1);  mul_927 = None
    mul_929: "f32[104]" = torch.ops.aten.mul.Tensor(primals_910, 0.9)
    add_734: "f32[104]" = torch.ops.aten.add.Tensor(mul_928, mul_929);  mul_928 = mul_929 = None
    unsqueeze_528: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_398, -1)
    unsqueeze_529: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
    mul_930: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_924, unsqueeze_529);  mul_924 = unsqueeze_529 = None
    unsqueeze_530: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_399, -1);  primals_399 = None
    unsqueeze_531: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
    add_735: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_930, unsqueeze_531);  mul_930 = unsqueeze_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_129: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_735);  add_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_787: "f32[8, 104, 14, 14]" = split_with_sizes_126[3];  split_with_sizes_126 = None
    cat_25: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_127, relu_128, relu_129, getitem_787], 1);  getitem_787 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_133: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_25, primals_400, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_736: "i64[]" = torch.ops.aten.add.Tensor(primals_914, 1)
    var_mean_133 = torch.ops.aten.var_mean.correction(convolution_133, [0, 2, 3], correction = 0, keepdim = True)
    getitem_788: "f32[1, 1024, 1, 1]" = var_mean_133[0]
    getitem_789: "f32[1, 1024, 1, 1]" = var_mean_133[1];  var_mean_133 = None
    add_737: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_788, 1e-05)
    rsqrt_133: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_737);  add_737 = None
    sub_133: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_133, getitem_789)
    mul_931: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_133);  sub_133 = None
    squeeze_399: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_789, [0, 2, 3]);  getitem_789 = None
    squeeze_400: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_133, [0, 2, 3]);  rsqrt_133 = None
    mul_932: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_399, 0.1)
    mul_933: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_912, 0.9)
    add_738: "f32[1024]" = torch.ops.aten.add.Tensor(mul_932, mul_933);  mul_932 = mul_933 = None
    squeeze_401: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_788, [0, 2, 3]);  getitem_788 = None
    mul_934: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_401, 1.0006381620931717);  squeeze_401 = None
    mul_935: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_934, 0.1);  mul_934 = None
    mul_936: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_913, 0.9)
    add_739: "f32[1024]" = torch.ops.aten.add.Tensor(mul_935, mul_936);  mul_935 = mul_936 = None
    unsqueeze_532: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_401, -1)
    unsqueeze_533: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
    mul_937: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_931, unsqueeze_533);  mul_931 = unsqueeze_533 = None
    unsqueeze_534: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_402, -1);  primals_402 = None
    unsqueeze_535: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
    add_740: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_937, unsqueeze_535);  mul_937 = unsqueeze_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_741: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_740, relu_125);  add_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_130: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_741);  add_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_134: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_130, primals_403, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_742: "i64[]" = torch.ops.aten.add.Tensor(primals_917, 1)
    var_mean_134 = torch.ops.aten.var_mean.correction(convolution_134, [0, 2, 3], correction = 0, keepdim = True)
    getitem_790: "f32[1, 416, 1, 1]" = var_mean_134[0]
    getitem_791: "f32[1, 416, 1, 1]" = var_mean_134[1];  var_mean_134 = None
    add_743: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_790, 1e-05)
    rsqrt_134: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_743);  add_743 = None
    sub_134: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_134, getitem_791)
    mul_938: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_134);  sub_134 = None
    squeeze_402: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_791, [0, 2, 3]);  getitem_791 = None
    squeeze_403: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_134, [0, 2, 3]);  rsqrt_134 = None
    mul_939: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_402, 0.1)
    mul_940: "f32[416]" = torch.ops.aten.mul.Tensor(primals_915, 0.9)
    add_744: "f32[416]" = torch.ops.aten.add.Tensor(mul_939, mul_940);  mul_939 = mul_940 = None
    squeeze_404: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_790, [0, 2, 3]);  getitem_790 = None
    mul_941: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_404, 1.0006381620931717);  squeeze_404 = None
    mul_942: "f32[416]" = torch.ops.aten.mul.Tensor(mul_941, 0.1);  mul_941 = None
    mul_943: "f32[416]" = torch.ops.aten.mul.Tensor(primals_916, 0.9)
    add_745: "f32[416]" = torch.ops.aten.add.Tensor(mul_942, mul_943);  mul_942 = mul_943 = None
    unsqueeze_536: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_404, -1)
    unsqueeze_537: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
    mul_944: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_938, unsqueeze_537);  mul_938 = unsqueeze_537 = None
    unsqueeze_538: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_405, -1);  primals_405 = None
    unsqueeze_539: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
    add_746: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_944, unsqueeze_539);  mul_944 = unsqueeze_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_131: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_746);  add_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_131 = torch.ops.aten.split_with_sizes.default(relu_131, [104, 104, 104, 104], 1)
    getitem_796: "f32[8, 104, 14, 14]" = split_with_sizes_131[0]
    convolution_135: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_796, primals_406, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_747: "i64[]" = torch.ops.aten.add.Tensor(primals_920, 1)
    var_mean_135 = torch.ops.aten.var_mean.correction(convolution_135, [0, 2, 3], correction = 0, keepdim = True)
    getitem_800: "f32[1, 104, 1, 1]" = var_mean_135[0]
    getitem_801: "f32[1, 104, 1, 1]" = var_mean_135[1];  var_mean_135 = None
    add_748: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_800, 1e-05)
    rsqrt_135: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_748);  add_748 = None
    sub_135: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_135, getitem_801)
    mul_945: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_135, rsqrt_135);  sub_135 = None
    squeeze_405: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_801, [0, 2, 3]);  getitem_801 = None
    squeeze_406: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_135, [0, 2, 3]);  rsqrt_135 = None
    mul_946: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_405, 0.1)
    mul_947: "f32[104]" = torch.ops.aten.mul.Tensor(primals_918, 0.9)
    add_749: "f32[104]" = torch.ops.aten.add.Tensor(mul_946, mul_947);  mul_946 = mul_947 = None
    squeeze_407: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_800, [0, 2, 3]);  getitem_800 = None
    mul_948: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_407, 1.0006381620931717);  squeeze_407 = None
    mul_949: "f32[104]" = torch.ops.aten.mul.Tensor(mul_948, 0.1);  mul_948 = None
    mul_950: "f32[104]" = torch.ops.aten.mul.Tensor(primals_919, 0.9)
    add_750: "f32[104]" = torch.ops.aten.add.Tensor(mul_949, mul_950);  mul_949 = mul_950 = None
    unsqueeze_540: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_407, -1)
    unsqueeze_541: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
    mul_951: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_945, unsqueeze_541);  mul_945 = unsqueeze_541 = None
    unsqueeze_542: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_408, -1);  primals_408 = None
    unsqueeze_543: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
    add_751: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_951, unsqueeze_543);  mul_951 = unsqueeze_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_132: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_751);  add_751 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_803: "f32[8, 104, 14, 14]" = split_with_sizes_131[1]
    add_752: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_132, getitem_803);  getitem_803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_136: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_752, primals_409, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_753: "i64[]" = torch.ops.aten.add.Tensor(primals_923, 1)
    var_mean_136 = torch.ops.aten.var_mean.correction(convolution_136, [0, 2, 3], correction = 0, keepdim = True)
    getitem_806: "f32[1, 104, 1, 1]" = var_mean_136[0]
    getitem_807: "f32[1, 104, 1, 1]" = var_mean_136[1];  var_mean_136 = None
    add_754: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_806, 1e-05)
    rsqrt_136: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_754);  add_754 = None
    sub_136: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_136, getitem_807)
    mul_952: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, rsqrt_136);  sub_136 = None
    squeeze_408: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_807, [0, 2, 3]);  getitem_807 = None
    squeeze_409: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_136, [0, 2, 3]);  rsqrt_136 = None
    mul_953: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_408, 0.1)
    mul_954: "f32[104]" = torch.ops.aten.mul.Tensor(primals_921, 0.9)
    add_755: "f32[104]" = torch.ops.aten.add.Tensor(mul_953, mul_954);  mul_953 = mul_954 = None
    squeeze_410: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_806, [0, 2, 3]);  getitem_806 = None
    mul_955: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_410, 1.0006381620931717);  squeeze_410 = None
    mul_956: "f32[104]" = torch.ops.aten.mul.Tensor(mul_955, 0.1);  mul_955 = None
    mul_957: "f32[104]" = torch.ops.aten.mul.Tensor(primals_922, 0.9)
    add_756: "f32[104]" = torch.ops.aten.add.Tensor(mul_956, mul_957);  mul_956 = mul_957 = None
    unsqueeze_544: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_410, -1)
    unsqueeze_545: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
    mul_958: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_952, unsqueeze_545);  mul_952 = unsqueeze_545 = None
    unsqueeze_546: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_411, -1);  primals_411 = None
    unsqueeze_547: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
    add_757: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_958, unsqueeze_547);  mul_958 = unsqueeze_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_133: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_757);  add_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_810: "f32[8, 104, 14, 14]" = split_with_sizes_131[2]
    add_758: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_133, getitem_810);  getitem_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_137: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_758, primals_412, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_759: "i64[]" = torch.ops.aten.add.Tensor(primals_926, 1)
    var_mean_137 = torch.ops.aten.var_mean.correction(convolution_137, [0, 2, 3], correction = 0, keepdim = True)
    getitem_812: "f32[1, 104, 1, 1]" = var_mean_137[0]
    getitem_813: "f32[1, 104, 1, 1]" = var_mean_137[1];  var_mean_137 = None
    add_760: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_812, 1e-05)
    rsqrt_137: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_760);  add_760 = None
    sub_137: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_137, getitem_813)
    mul_959: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_137);  sub_137 = None
    squeeze_411: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_813, [0, 2, 3]);  getitem_813 = None
    squeeze_412: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_137, [0, 2, 3]);  rsqrt_137 = None
    mul_960: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_411, 0.1)
    mul_961: "f32[104]" = torch.ops.aten.mul.Tensor(primals_924, 0.9)
    add_761: "f32[104]" = torch.ops.aten.add.Tensor(mul_960, mul_961);  mul_960 = mul_961 = None
    squeeze_413: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_812, [0, 2, 3]);  getitem_812 = None
    mul_962: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_413, 1.0006381620931717);  squeeze_413 = None
    mul_963: "f32[104]" = torch.ops.aten.mul.Tensor(mul_962, 0.1);  mul_962 = None
    mul_964: "f32[104]" = torch.ops.aten.mul.Tensor(primals_925, 0.9)
    add_762: "f32[104]" = torch.ops.aten.add.Tensor(mul_963, mul_964);  mul_963 = mul_964 = None
    unsqueeze_548: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_413, -1)
    unsqueeze_549: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
    mul_965: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_959, unsqueeze_549);  mul_959 = unsqueeze_549 = None
    unsqueeze_550: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_414, -1);  primals_414 = None
    unsqueeze_551: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
    add_763: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_965, unsqueeze_551);  mul_965 = unsqueeze_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_134: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_763);  add_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_817: "f32[8, 104, 14, 14]" = split_with_sizes_131[3];  split_with_sizes_131 = None
    cat_26: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_132, relu_133, relu_134, getitem_817], 1);  getitem_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_138: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_26, primals_415, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_764: "i64[]" = torch.ops.aten.add.Tensor(primals_929, 1)
    var_mean_138 = torch.ops.aten.var_mean.correction(convolution_138, [0, 2, 3], correction = 0, keepdim = True)
    getitem_818: "f32[1, 1024, 1, 1]" = var_mean_138[0]
    getitem_819: "f32[1, 1024, 1, 1]" = var_mean_138[1];  var_mean_138 = None
    add_765: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_818, 1e-05)
    rsqrt_138: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_765);  add_765 = None
    sub_138: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_138, getitem_819)
    mul_966: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_138, rsqrt_138);  sub_138 = None
    squeeze_414: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_819, [0, 2, 3]);  getitem_819 = None
    squeeze_415: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_138, [0, 2, 3]);  rsqrt_138 = None
    mul_967: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_414, 0.1)
    mul_968: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_927, 0.9)
    add_766: "f32[1024]" = torch.ops.aten.add.Tensor(mul_967, mul_968);  mul_967 = mul_968 = None
    squeeze_416: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_818, [0, 2, 3]);  getitem_818 = None
    mul_969: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_416, 1.0006381620931717);  squeeze_416 = None
    mul_970: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_969, 0.1);  mul_969 = None
    mul_971: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_928, 0.9)
    add_767: "f32[1024]" = torch.ops.aten.add.Tensor(mul_970, mul_971);  mul_970 = mul_971 = None
    unsqueeze_552: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_416, -1)
    unsqueeze_553: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
    mul_972: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_966, unsqueeze_553);  mul_966 = unsqueeze_553 = None
    unsqueeze_554: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_417, -1);  primals_417 = None
    unsqueeze_555: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
    add_768: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_972, unsqueeze_555);  mul_972 = unsqueeze_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_769: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_768, relu_130);  add_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_135: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_769);  add_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_139: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_135, primals_418, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_770: "i64[]" = torch.ops.aten.add.Tensor(primals_932, 1)
    var_mean_139 = torch.ops.aten.var_mean.correction(convolution_139, [0, 2, 3], correction = 0, keepdim = True)
    getitem_820: "f32[1, 416, 1, 1]" = var_mean_139[0]
    getitem_821: "f32[1, 416, 1, 1]" = var_mean_139[1];  var_mean_139 = None
    add_771: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_820, 1e-05)
    rsqrt_139: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_771);  add_771 = None
    sub_139: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_139, getitem_821)
    mul_973: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_139, rsqrt_139);  sub_139 = None
    squeeze_417: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_821, [0, 2, 3]);  getitem_821 = None
    squeeze_418: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_139, [0, 2, 3]);  rsqrt_139 = None
    mul_974: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_417, 0.1)
    mul_975: "f32[416]" = torch.ops.aten.mul.Tensor(primals_930, 0.9)
    add_772: "f32[416]" = torch.ops.aten.add.Tensor(mul_974, mul_975);  mul_974 = mul_975 = None
    squeeze_419: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_820, [0, 2, 3]);  getitem_820 = None
    mul_976: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_419, 1.0006381620931717);  squeeze_419 = None
    mul_977: "f32[416]" = torch.ops.aten.mul.Tensor(mul_976, 0.1);  mul_976 = None
    mul_978: "f32[416]" = torch.ops.aten.mul.Tensor(primals_931, 0.9)
    add_773: "f32[416]" = torch.ops.aten.add.Tensor(mul_977, mul_978);  mul_977 = mul_978 = None
    unsqueeze_556: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_419, -1)
    unsqueeze_557: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
    mul_979: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_973, unsqueeze_557);  mul_973 = unsqueeze_557 = None
    unsqueeze_558: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_420, -1);  primals_420 = None
    unsqueeze_559: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
    add_774: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_979, unsqueeze_559);  mul_979 = unsqueeze_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_136: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_774);  add_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_136 = torch.ops.aten.split_with_sizes.default(relu_136, [104, 104, 104, 104], 1)
    getitem_826: "f32[8, 104, 14, 14]" = split_with_sizes_136[0]
    convolution_140: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_826, primals_421, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_775: "i64[]" = torch.ops.aten.add.Tensor(primals_935, 1)
    var_mean_140 = torch.ops.aten.var_mean.correction(convolution_140, [0, 2, 3], correction = 0, keepdim = True)
    getitem_830: "f32[1, 104, 1, 1]" = var_mean_140[0]
    getitem_831: "f32[1, 104, 1, 1]" = var_mean_140[1];  var_mean_140 = None
    add_776: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_830, 1e-05)
    rsqrt_140: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_776);  add_776 = None
    sub_140: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_140, getitem_831)
    mul_980: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_140);  sub_140 = None
    squeeze_420: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_831, [0, 2, 3]);  getitem_831 = None
    squeeze_421: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_140, [0, 2, 3]);  rsqrt_140 = None
    mul_981: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_420, 0.1)
    mul_982: "f32[104]" = torch.ops.aten.mul.Tensor(primals_933, 0.9)
    add_777: "f32[104]" = torch.ops.aten.add.Tensor(mul_981, mul_982);  mul_981 = mul_982 = None
    squeeze_422: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_830, [0, 2, 3]);  getitem_830 = None
    mul_983: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_422, 1.0006381620931717);  squeeze_422 = None
    mul_984: "f32[104]" = torch.ops.aten.mul.Tensor(mul_983, 0.1);  mul_983 = None
    mul_985: "f32[104]" = torch.ops.aten.mul.Tensor(primals_934, 0.9)
    add_778: "f32[104]" = torch.ops.aten.add.Tensor(mul_984, mul_985);  mul_984 = mul_985 = None
    unsqueeze_560: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_422, -1)
    unsqueeze_561: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
    mul_986: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_980, unsqueeze_561);  mul_980 = unsqueeze_561 = None
    unsqueeze_562: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_423, -1);  primals_423 = None
    unsqueeze_563: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
    add_779: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_986, unsqueeze_563);  mul_986 = unsqueeze_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_137: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_779);  add_779 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_833: "f32[8, 104, 14, 14]" = split_with_sizes_136[1]
    add_780: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_137, getitem_833);  getitem_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_141: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_780, primals_424, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_781: "i64[]" = torch.ops.aten.add.Tensor(primals_938, 1)
    var_mean_141 = torch.ops.aten.var_mean.correction(convolution_141, [0, 2, 3], correction = 0, keepdim = True)
    getitem_836: "f32[1, 104, 1, 1]" = var_mean_141[0]
    getitem_837: "f32[1, 104, 1, 1]" = var_mean_141[1];  var_mean_141 = None
    add_782: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_836, 1e-05)
    rsqrt_141: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_782);  add_782 = None
    sub_141: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_141, getitem_837)
    mul_987: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_141, rsqrt_141);  sub_141 = None
    squeeze_423: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_837, [0, 2, 3]);  getitem_837 = None
    squeeze_424: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_141, [0, 2, 3]);  rsqrt_141 = None
    mul_988: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_423, 0.1)
    mul_989: "f32[104]" = torch.ops.aten.mul.Tensor(primals_936, 0.9)
    add_783: "f32[104]" = torch.ops.aten.add.Tensor(mul_988, mul_989);  mul_988 = mul_989 = None
    squeeze_425: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_836, [0, 2, 3]);  getitem_836 = None
    mul_990: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_425, 1.0006381620931717);  squeeze_425 = None
    mul_991: "f32[104]" = torch.ops.aten.mul.Tensor(mul_990, 0.1);  mul_990 = None
    mul_992: "f32[104]" = torch.ops.aten.mul.Tensor(primals_937, 0.9)
    add_784: "f32[104]" = torch.ops.aten.add.Tensor(mul_991, mul_992);  mul_991 = mul_992 = None
    unsqueeze_564: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_425, -1)
    unsqueeze_565: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
    mul_993: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_987, unsqueeze_565);  mul_987 = unsqueeze_565 = None
    unsqueeze_566: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_426, -1);  primals_426 = None
    unsqueeze_567: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
    add_785: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_993, unsqueeze_567);  mul_993 = unsqueeze_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_138: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_785);  add_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_840: "f32[8, 104, 14, 14]" = split_with_sizes_136[2]
    add_786: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_138, getitem_840);  getitem_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_142: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_786, primals_427, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_787: "i64[]" = torch.ops.aten.add.Tensor(primals_941, 1)
    var_mean_142 = torch.ops.aten.var_mean.correction(convolution_142, [0, 2, 3], correction = 0, keepdim = True)
    getitem_842: "f32[1, 104, 1, 1]" = var_mean_142[0]
    getitem_843: "f32[1, 104, 1, 1]" = var_mean_142[1];  var_mean_142 = None
    add_788: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_842, 1e-05)
    rsqrt_142: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_788);  add_788 = None
    sub_142: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_142, getitem_843)
    mul_994: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_142, rsqrt_142);  sub_142 = None
    squeeze_426: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_843, [0, 2, 3]);  getitem_843 = None
    squeeze_427: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_142, [0, 2, 3]);  rsqrt_142 = None
    mul_995: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_426, 0.1)
    mul_996: "f32[104]" = torch.ops.aten.mul.Tensor(primals_939, 0.9)
    add_789: "f32[104]" = torch.ops.aten.add.Tensor(mul_995, mul_996);  mul_995 = mul_996 = None
    squeeze_428: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_842, [0, 2, 3]);  getitem_842 = None
    mul_997: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_428, 1.0006381620931717);  squeeze_428 = None
    mul_998: "f32[104]" = torch.ops.aten.mul.Tensor(mul_997, 0.1);  mul_997 = None
    mul_999: "f32[104]" = torch.ops.aten.mul.Tensor(primals_940, 0.9)
    add_790: "f32[104]" = torch.ops.aten.add.Tensor(mul_998, mul_999);  mul_998 = mul_999 = None
    unsqueeze_568: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_428, -1)
    unsqueeze_569: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
    mul_1000: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_994, unsqueeze_569);  mul_994 = unsqueeze_569 = None
    unsqueeze_570: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_429, -1);  primals_429 = None
    unsqueeze_571: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
    add_791: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_1000, unsqueeze_571);  mul_1000 = unsqueeze_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_139: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_791);  add_791 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_847: "f32[8, 104, 14, 14]" = split_with_sizes_136[3];  split_with_sizes_136 = None
    cat_27: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_137, relu_138, relu_139, getitem_847], 1);  getitem_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_143: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_27, primals_430, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_792: "i64[]" = torch.ops.aten.add.Tensor(primals_944, 1)
    var_mean_143 = torch.ops.aten.var_mean.correction(convolution_143, [0, 2, 3], correction = 0, keepdim = True)
    getitem_848: "f32[1, 1024, 1, 1]" = var_mean_143[0]
    getitem_849: "f32[1, 1024, 1, 1]" = var_mean_143[1];  var_mean_143 = None
    add_793: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_848, 1e-05)
    rsqrt_143: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_793);  add_793 = None
    sub_143: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_143, getitem_849)
    mul_1001: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, rsqrt_143);  sub_143 = None
    squeeze_429: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_849, [0, 2, 3]);  getitem_849 = None
    squeeze_430: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_143, [0, 2, 3]);  rsqrt_143 = None
    mul_1002: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_429, 0.1)
    mul_1003: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_942, 0.9)
    add_794: "f32[1024]" = torch.ops.aten.add.Tensor(mul_1002, mul_1003);  mul_1002 = mul_1003 = None
    squeeze_431: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_848, [0, 2, 3]);  getitem_848 = None
    mul_1004: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_431, 1.0006381620931717);  squeeze_431 = None
    mul_1005: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1004, 0.1);  mul_1004 = None
    mul_1006: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_943, 0.9)
    add_795: "f32[1024]" = torch.ops.aten.add.Tensor(mul_1005, mul_1006);  mul_1005 = mul_1006 = None
    unsqueeze_572: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_431, -1)
    unsqueeze_573: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
    mul_1007: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1001, unsqueeze_573);  mul_1001 = unsqueeze_573 = None
    unsqueeze_574: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_432, -1);  primals_432 = None
    unsqueeze_575: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
    add_796: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_1007, unsqueeze_575);  mul_1007 = unsqueeze_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_797: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_796, relu_135);  add_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_140: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_797);  add_797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_144: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_140, primals_433, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_798: "i64[]" = torch.ops.aten.add.Tensor(primals_947, 1)
    var_mean_144 = torch.ops.aten.var_mean.correction(convolution_144, [0, 2, 3], correction = 0, keepdim = True)
    getitem_850: "f32[1, 416, 1, 1]" = var_mean_144[0]
    getitem_851: "f32[1, 416, 1, 1]" = var_mean_144[1];  var_mean_144 = None
    add_799: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_850, 1e-05)
    rsqrt_144: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_799);  add_799 = None
    sub_144: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_144, getitem_851)
    mul_1008: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, rsqrt_144);  sub_144 = None
    squeeze_432: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_851, [0, 2, 3]);  getitem_851 = None
    squeeze_433: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_144, [0, 2, 3]);  rsqrt_144 = None
    mul_1009: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_432, 0.1)
    mul_1010: "f32[416]" = torch.ops.aten.mul.Tensor(primals_945, 0.9)
    add_800: "f32[416]" = torch.ops.aten.add.Tensor(mul_1009, mul_1010);  mul_1009 = mul_1010 = None
    squeeze_434: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_850, [0, 2, 3]);  getitem_850 = None
    mul_1011: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_434, 1.0006381620931717);  squeeze_434 = None
    mul_1012: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1011, 0.1);  mul_1011 = None
    mul_1013: "f32[416]" = torch.ops.aten.mul.Tensor(primals_946, 0.9)
    add_801: "f32[416]" = torch.ops.aten.add.Tensor(mul_1012, mul_1013);  mul_1012 = mul_1013 = None
    unsqueeze_576: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_434, -1)
    unsqueeze_577: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
    mul_1014: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1008, unsqueeze_577);  mul_1008 = unsqueeze_577 = None
    unsqueeze_578: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_435, -1);  primals_435 = None
    unsqueeze_579: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
    add_802: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_1014, unsqueeze_579);  mul_1014 = unsqueeze_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_141: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_802);  add_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_141 = torch.ops.aten.split_with_sizes.default(relu_141, [104, 104, 104, 104], 1)
    getitem_856: "f32[8, 104, 14, 14]" = split_with_sizes_141[0]
    convolution_145: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_856, primals_436, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_803: "i64[]" = torch.ops.aten.add.Tensor(primals_950, 1)
    var_mean_145 = torch.ops.aten.var_mean.correction(convolution_145, [0, 2, 3], correction = 0, keepdim = True)
    getitem_860: "f32[1, 104, 1, 1]" = var_mean_145[0]
    getitem_861: "f32[1, 104, 1, 1]" = var_mean_145[1];  var_mean_145 = None
    add_804: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_860, 1e-05)
    rsqrt_145: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_804);  add_804 = None
    sub_145: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_145, getitem_861)
    mul_1015: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_145);  sub_145 = None
    squeeze_435: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_861, [0, 2, 3]);  getitem_861 = None
    squeeze_436: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_145, [0, 2, 3]);  rsqrt_145 = None
    mul_1016: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_435, 0.1)
    mul_1017: "f32[104]" = torch.ops.aten.mul.Tensor(primals_948, 0.9)
    add_805: "f32[104]" = torch.ops.aten.add.Tensor(mul_1016, mul_1017);  mul_1016 = mul_1017 = None
    squeeze_437: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_860, [0, 2, 3]);  getitem_860 = None
    mul_1018: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_437, 1.0006381620931717);  squeeze_437 = None
    mul_1019: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1018, 0.1);  mul_1018 = None
    mul_1020: "f32[104]" = torch.ops.aten.mul.Tensor(primals_949, 0.9)
    add_806: "f32[104]" = torch.ops.aten.add.Tensor(mul_1019, mul_1020);  mul_1019 = mul_1020 = None
    unsqueeze_580: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_437, -1)
    unsqueeze_581: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
    mul_1021: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1015, unsqueeze_581);  mul_1015 = unsqueeze_581 = None
    unsqueeze_582: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_438, -1);  primals_438 = None
    unsqueeze_583: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
    add_807: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_1021, unsqueeze_583);  mul_1021 = unsqueeze_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_142: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_807);  add_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_863: "f32[8, 104, 14, 14]" = split_with_sizes_141[1]
    add_808: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_142, getitem_863);  getitem_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_146: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_808, primals_439, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_809: "i64[]" = torch.ops.aten.add.Tensor(primals_953, 1)
    var_mean_146 = torch.ops.aten.var_mean.correction(convolution_146, [0, 2, 3], correction = 0, keepdim = True)
    getitem_866: "f32[1, 104, 1, 1]" = var_mean_146[0]
    getitem_867: "f32[1, 104, 1, 1]" = var_mean_146[1];  var_mean_146 = None
    add_810: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_866, 1e-05)
    rsqrt_146: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_810);  add_810 = None
    sub_146: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_146, getitem_867)
    mul_1022: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_146, rsqrt_146);  sub_146 = None
    squeeze_438: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_867, [0, 2, 3]);  getitem_867 = None
    squeeze_439: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_146, [0, 2, 3]);  rsqrt_146 = None
    mul_1023: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_438, 0.1)
    mul_1024: "f32[104]" = torch.ops.aten.mul.Tensor(primals_951, 0.9)
    add_811: "f32[104]" = torch.ops.aten.add.Tensor(mul_1023, mul_1024);  mul_1023 = mul_1024 = None
    squeeze_440: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_866, [0, 2, 3]);  getitem_866 = None
    mul_1025: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_440, 1.0006381620931717);  squeeze_440 = None
    mul_1026: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1025, 0.1);  mul_1025 = None
    mul_1027: "f32[104]" = torch.ops.aten.mul.Tensor(primals_952, 0.9)
    add_812: "f32[104]" = torch.ops.aten.add.Tensor(mul_1026, mul_1027);  mul_1026 = mul_1027 = None
    unsqueeze_584: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_440, -1)
    unsqueeze_585: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
    mul_1028: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1022, unsqueeze_585);  mul_1022 = unsqueeze_585 = None
    unsqueeze_586: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_441, -1);  primals_441 = None
    unsqueeze_587: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
    add_813: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_1028, unsqueeze_587);  mul_1028 = unsqueeze_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_143: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_813);  add_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_870: "f32[8, 104, 14, 14]" = split_with_sizes_141[2]
    add_814: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_143, getitem_870);  getitem_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_147: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_814, primals_442, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_815: "i64[]" = torch.ops.aten.add.Tensor(primals_956, 1)
    var_mean_147 = torch.ops.aten.var_mean.correction(convolution_147, [0, 2, 3], correction = 0, keepdim = True)
    getitem_872: "f32[1, 104, 1, 1]" = var_mean_147[0]
    getitem_873: "f32[1, 104, 1, 1]" = var_mean_147[1];  var_mean_147 = None
    add_816: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_872, 1e-05)
    rsqrt_147: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_816);  add_816 = None
    sub_147: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_147, getitem_873)
    mul_1029: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, rsqrt_147);  sub_147 = None
    squeeze_441: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_873, [0, 2, 3]);  getitem_873 = None
    squeeze_442: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_147, [0, 2, 3]);  rsqrt_147 = None
    mul_1030: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_441, 0.1)
    mul_1031: "f32[104]" = torch.ops.aten.mul.Tensor(primals_954, 0.9)
    add_817: "f32[104]" = torch.ops.aten.add.Tensor(mul_1030, mul_1031);  mul_1030 = mul_1031 = None
    squeeze_443: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_872, [0, 2, 3]);  getitem_872 = None
    mul_1032: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_443, 1.0006381620931717);  squeeze_443 = None
    mul_1033: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1032, 0.1);  mul_1032 = None
    mul_1034: "f32[104]" = torch.ops.aten.mul.Tensor(primals_955, 0.9)
    add_818: "f32[104]" = torch.ops.aten.add.Tensor(mul_1033, mul_1034);  mul_1033 = mul_1034 = None
    unsqueeze_588: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_443, -1)
    unsqueeze_589: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
    mul_1035: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1029, unsqueeze_589);  mul_1029 = unsqueeze_589 = None
    unsqueeze_590: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_444, -1);  primals_444 = None
    unsqueeze_591: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
    add_819: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_1035, unsqueeze_591);  mul_1035 = unsqueeze_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_144: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_819);  add_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_877: "f32[8, 104, 14, 14]" = split_with_sizes_141[3];  split_with_sizes_141 = None
    cat_28: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_142, relu_143, relu_144, getitem_877], 1);  getitem_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_148: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_28, primals_445, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_820: "i64[]" = torch.ops.aten.add.Tensor(primals_959, 1)
    var_mean_148 = torch.ops.aten.var_mean.correction(convolution_148, [0, 2, 3], correction = 0, keepdim = True)
    getitem_878: "f32[1, 1024, 1, 1]" = var_mean_148[0]
    getitem_879: "f32[1, 1024, 1, 1]" = var_mean_148[1];  var_mean_148 = None
    add_821: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_878, 1e-05)
    rsqrt_148: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_821);  add_821 = None
    sub_148: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_148, getitem_879)
    mul_1036: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, rsqrt_148);  sub_148 = None
    squeeze_444: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_879, [0, 2, 3]);  getitem_879 = None
    squeeze_445: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_148, [0, 2, 3]);  rsqrt_148 = None
    mul_1037: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_444, 0.1)
    mul_1038: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_957, 0.9)
    add_822: "f32[1024]" = torch.ops.aten.add.Tensor(mul_1037, mul_1038);  mul_1037 = mul_1038 = None
    squeeze_446: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_878, [0, 2, 3]);  getitem_878 = None
    mul_1039: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_446, 1.0006381620931717);  squeeze_446 = None
    mul_1040: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1039, 0.1);  mul_1039 = None
    mul_1041: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_958, 0.9)
    add_823: "f32[1024]" = torch.ops.aten.add.Tensor(mul_1040, mul_1041);  mul_1040 = mul_1041 = None
    unsqueeze_592: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_446, -1)
    unsqueeze_593: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
    mul_1042: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1036, unsqueeze_593);  mul_1036 = unsqueeze_593 = None
    unsqueeze_594: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_447, -1);  primals_447 = None
    unsqueeze_595: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
    add_824: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_1042, unsqueeze_595);  mul_1042 = unsqueeze_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_825: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_824, relu_140);  add_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_145: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_825);  add_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_149: "f32[8, 416, 14, 14]" = torch.ops.aten.convolution.default(relu_145, primals_448, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_826: "i64[]" = torch.ops.aten.add.Tensor(primals_962, 1)
    var_mean_149 = torch.ops.aten.var_mean.correction(convolution_149, [0, 2, 3], correction = 0, keepdim = True)
    getitem_880: "f32[1, 416, 1, 1]" = var_mean_149[0]
    getitem_881: "f32[1, 416, 1, 1]" = var_mean_149[1];  var_mean_149 = None
    add_827: "f32[1, 416, 1, 1]" = torch.ops.aten.add.Tensor(getitem_880, 1e-05)
    rsqrt_149: "f32[1, 416, 1, 1]" = torch.ops.aten.rsqrt.default(add_827);  add_827 = None
    sub_149: "f32[8, 416, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_149, getitem_881)
    mul_1043: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, rsqrt_149);  sub_149 = None
    squeeze_447: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_881, [0, 2, 3]);  getitem_881 = None
    squeeze_448: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_149, [0, 2, 3]);  rsqrt_149 = None
    mul_1044: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_447, 0.1)
    mul_1045: "f32[416]" = torch.ops.aten.mul.Tensor(primals_960, 0.9)
    add_828: "f32[416]" = torch.ops.aten.add.Tensor(mul_1044, mul_1045);  mul_1044 = mul_1045 = None
    squeeze_449: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_880, [0, 2, 3]);  getitem_880 = None
    mul_1046: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_449, 1.0006381620931717);  squeeze_449 = None
    mul_1047: "f32[416]" = torch.ops.aten.mul.Tensor(mul_1046, 0.1);  mul_1046 = None
    mul_1048: "f32[416]" = torch.ops.aten.mul.Tensor(primals_961, 0.9)
    add_829: "f32[416]" = torch.ops.aten.add.Tensor(mul_1047, mul_1048);  mul_1047 = mul_1048 = None
    unsqueeze_596: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_449, -1)
    unsqueeze_597: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
    mul_1049: "f32[8, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1043, unsqueeze_597);  mul_1043 = unsqueeze_597 = None
    unsqueeze_598: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_450, -1);  primals_450 = None
    unsqueeze_599: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
    add_830: "f32[8, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_1049, unsqueeze_599);  mul_1049 = unsqueeze_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_146: "f32[8, 416, 14, 14]" = torch.ops.aten.relu.default(add_830);  add_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_146 = torch.ops.aten.split_with_sizes.default(relu_146, [104, 104, 104, 104], 1)
    getitem_886: "f32[8, 104, 14, 14]" = split_with_sizes_146[0]
    convolution_150: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(getitem_886, primals_451, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_831: "i64[]" = torch.ops.aten.add.Tensor(primals_965, 1)
    var_mean_150 = torch.ops.aten.var_mean.correction(convolution_150, [0, 2, 3], correction = 0, keepdim = True)
    getitem_890: "f32[1, 104, 1, 1]" = var_mean_150[0]
    getitem_891: "f32[1, 104, 1, 1]" = var_mean_150[1];  var_mean_150 = None
    add_832: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_890, 1e-05)
    rsqrt_150: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_832);  add_832 = None
    sub_150: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_150, getitem_891)
    mul_1050: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_150, rsqrt_150);  sub_150 = None
    squeeze_450: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_891, [0, 2, 3]);  getitem_891 = None
    squeeze_451: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_150, [0, 2, 3]);  rsqrt_150 = None
    mul_1051: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_450, 0.1)
    mul_1052: "f32[104]" = torch.ops.aten.mul.Tensor(primals_963, 0.9)
    add_833: "f32[104]" = torch.ops.aten.add.Tensor(mul_1051, mul_1052);  mul_1051 = mul_1052 = None
    squeeze_452: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_890, [0, 2, 3]);  getitem_890 = None
    mul_1053: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_452, 1.0006381620931717);  squeeze_452 = None
    mul_1054: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1053, 0.1);  mul_1053 = None
    mul_1055: "f32[104]" = torch.ops.aten.mul.Tensor(primals_964, 0.9)
    add_834: "f32[104]" = torch.ops.aten.add.Tensor(mul_1054, mul_1055);  mul_1054 = mul_1055 = None
    unsqueeze_600: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_452, -1)
    unsqueeze_601: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
    mul_1056: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1050, unsqueeze_601);  mul_1050 = unsqueeze_601 = None
    unsqueeze_602: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_453, -1);  primals_453 = None
    unsqueeze_603: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
    add_835: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_1056, unsqueeze_603);  mul_1056 = unsqueeze_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_147: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_835);  add_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_893: "f32[8, 104, 14, 14]" = split_with_sizes_146[1]
    add_836: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_147, getitem_893);  getitem_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_151: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_836, primals_454, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_837: "i64[]" = torch.ops.aten.add.Tensor(primals_968, 1)
    var_mean_151 = torch.ops.aten.var_mean.correction(convolution_151, [0, 2, 3], correction = 0, keepdim = True)
    getitem_896: "f32[1, 104, 1, 1]" = var_mean_151[0]
    getitem_897: "f32[1, 104, 1, 1]" = var_mean_151[1];  var_mean_151 = None
    add_838: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_896, 1e-05)
    rsqrt_151: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_838);  add_838 = None
    sub_151: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_151, getitem_897)
    mul_1057: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, rsqrt_151);  sub_151 = None
    squeeze_453: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_897, [0, 2, 3]);  getitem_897 = None
    squeeze_454: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_151, [0, 2, 3]);  rsqrt_151 = None
    mul_1058: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_453, 0.1)
    mul_1059: "f32[104]" = torch.ops.aten.mul.Tensor(primals_966, 0.9)
    add_839: "f32[104]" = torch.ops.aten.add.Tensor(mul_1058, mul_1059);  mul_1058 = mul_1059 = None
    squeeze_455: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_896, [0, 2, 3]);  getitem_896 = None
    mul_1060: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_455, 1.0006381620931717);  squeeze_455 = None
    mul_1061: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1060, 0.1);  mul_1060 = None
    mul_1062: "f32[104]" = torch.ops.aten.mul.Tensor(primals_967, 0.9)
    add_840: "f32[104]" = torch.ops.aten.add.Tensor(mul_1061, mul_1062);  mul_1061 = mul_1062 = None
    unsqueeze_604: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_455, -1)
    unsqueeze_605: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
    mul_1063: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1057, unsqueeze_605);  mul_1057 = unsqueeze_605 = None
    unsqueeze_606: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_456, -1);  primals_456 = None
    unsqueeze_607: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
    add_841: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_1063, unsqueeze_607);  mul_1063 = unsqueeze_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_148: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_841);  add_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_900: "f32[8, 104, 14, 14]" = split_with_sizes_146[2]
    add_842: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(relu_148, getitem_900);  getitem_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_152: "f32[8, 104, 14, 14]" = torch.ops.aten.convolution.default(add_842, primals_457, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_843: "i64[]" = torch.ops.aten.add.Tensor(primals_971, 1)
    var_mean_152 = torch.ops.aten.var_mean.correction(convolution_152, [0, 2, 3], correction = 0, keepdim = True)
    getitem_902: "f32[1, 104, 1, 1]" = var_mean_152[0]
    getitem_903: "f32[1, 104, 1, 1]" = var_mean_152[1];  var_mean_152 = None
    add_844: "f32[1, 104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_902, 1e-05)
    rsqrt_152: "f32[1, 104, 1, 1]" = torch.ops.aten.rsqrt.default(add_844);  add_844 = None
    sub_152: "f32[8, 104, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_152, getitem_903)
    mul_1064: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, rsqrt_152);  sub_152 = None
    squeeze_456: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_903, [0, 2, 3]);  getitem_903 = None
    squeeze_457: "f32[104]" = torch.ops.aten.squeeze.dims(rsqrt_152, [0, 2, 3]);  rsqrt_152 = None
    mul_1065: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_456, 0.1)
    mul_1066: "f32[104]" = torch.ops.aten.mul.Tensor(primals_969, 0.9)
    add_845: "f32[104]" = torch.ops.aten.add.Tensor(mul_1065, mul_1066);  mul_1065 = mul_1066 = None
    squeeze_458: "f32[104]" = torch.ops.aten.squeeze.dims(getitem_902, [0, 2, 3]);  getitem_902 = None
    mul_1067: "f32[104]" = torch.ops.aten.mul.Tensor(squeeze_458, 1.0006381620931717);  squeeze_458 = None
    mul_1068: "f32[104]" = torch.ops.aten.mul.Tensor(mul_1067, 0.1);  mul_1067 = None
    mul_1069: "f32[104]" = torch.ops.aten.mul.Tensor(primals_970, 0.9)
    add_846: "f32[104]" = torch.ops.aten.add.Tensor(mul_1068, mul_1069);  mul_1068 = mul_1069 = None
    unsqueeze_608: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_458, -1)
    unsqueeze_609: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
    mul_1070: "f32[8, 104, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1064, unsqueeze_609);  mul_1064 = unsqueeze_609 = None
    unsqueeze_610: "f32[104, 1]" = torch.ops.aten.unsqueeze.default(primals_459, -1);  primals_459 = None
    unsqueeze_611: "f32[104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
    add_847: "f32[8, 104, 14, 14]" = torch.ops.aten.add.Tensor(mul_1070, unsqueeze_611);  mul_1070 = unsqueeze_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_149: "f32[8, 104, 14, 14]" = torch.ops.aten.relu.default(add_847);  add_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_907: "f32[8, 104, 14, 14]" = split_with_sizes_146[3];  split_with_sizes_146 = None
    cat_29: "f32[8, 416, 14, 14]" = torch.ops.aten.cat.default([relu_147, relu_148, relu_149, getitem_907], 1);  getitem_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_153: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(cat_29, primals_460, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_848: "i64[]" = torch.ops.aten.add.Tensor(primals_974, 1)
    var_mean_153 = torch.ops.aten.var_mean.correction(convolution_153, [0, 2, 3], correction = 0, keepdim = True)
    getitem_908: "f32[1, 1024, 1, 1]" = var_mean_153[0]
    getitem_909: "f32[1, 1024, 1, 1]" = var_mean_153[1];  var_mean_153 = None
    add_849: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_908, 1e-05)
    rsqrt_153: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_849);  add_849 = None
    sub_153: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_153, getitem_909)
    mul_1071: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, rsqrt_153);  sub_153 = None
    squeeze_459: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_909, [0, 2, 3]);  getitem_909 = None
    squeeze_460: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_153, [0, 2, 3]);  rsqrt_153 = None
    mul_1072: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_459, 0.1)
    mul_1073: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_972, 0.9)
    add_850: "f32[1024]" = torch.ops.aten.add.Tensor(mul_1072, mul_1073);  mul_1072 = mul_1073 = None
    squeeze_461: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_908, [0, 2, 3]);  getitem_908 = None
    mul_1074: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_461, 1.0006381620931717);  squeeze_461 = None
    mul_1075: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1074, 0.1);  mul_1074 = None
    mul_1076: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_973, 0.9)
    add_851: "f32[1024]" = torch.ops.aten.add.Tensor(mul_1075, mul_1076);  mul_1075 = mul_1076 = None
    unsqueeze_612: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_461, -1)
    unsqueeze_613: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
    mul_1077: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1071, unsqueeze_613);  mul_1071 = unsqueeze_613 = None
    unsqueeze_614: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_462, -1);  primals_462 = None
    unsqueeze_615: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
    add_852: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_1077, unsqueeze_615);  mul_1077 = unsqueeze_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_853: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_852, relu_145);  add_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_150: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_853);  add_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_154: "f32[8, 832, 14, 14]" = torch.ops.aten.convolution.default(relu_150, primals_463, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_854: "i64[]" = torch.ops.aten.add.Tensor(primals_977, 1)
    var_mean_154 = torch.ops.aten.var_mean.correction(convolution_154, [0, 2, 3], correction = 0, keepdim = True)
    getitem_910: "f32[1, 832, 1, 1]" = var_mean_154[0]
    getitem_911: "f32[1, 832, 1, 1]" = var_mean_154[1];  var_mean_154 = None
    add_855: "f32[1, 832, 1, 1]" = torch.ops.aten.add.Tensor(getitem_910, 1e-05)
    rsqrt_154: "f32[1, 832, 1, 1]" = torch.ops.aten.rsqrt.default(add_855);  add_855 = None
    sub_154: "f32[8, 832, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_154, getitem_911)
    mul_1078: "f32[8, 832, 14, 14]" = torch.ops.aten.mul.Tensor(sub_154, rsqrt_154);  sub_154 = None
    squeeze_462: "f32[832]" = torch.ops.aten.squeeze.dims(getitem_911, [0, 2, 3]);  getitem_911 = None
    squeeze_463: "f32[832]" = torch.ops.aten.squeeze.dims(rsqrt_154, [0, 2, 3]);  rsqrt_154 = None
    mul_1079: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_462, 0.1)
    mul_1080: "f32[832]" = torch.ops.aten.mul.Tensor(primals_975, 0.9)
    add_856: "f32[832]" = torch.ops.aten.add.Tensor(mul_1079, mul_1080);  mul_1079 = mul_1080 = None
    squeeze_464: "f32[832]" = torch.ops.aten.squeeze.dims(getitem_910, [0, 2, 3]);  getitem_910 = None
    mul_1081: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_464, 1.0006381620931717);  squeeze_464 = None
    mul_1082: "f32[832]" = torch.ops.aten.mul.Tensor(mul_1081, 0.1);  mul_1081 = None
    mul_1083: "f32[832]" = torch.ops.aten.mul.Tensor(primals_976, 0.9)
    add_857: "f32[832]" = torch.ops.aten.add.Tensor(mul_1082, mul_1083);  mul_1082 = mul_1083 = None
    unsqueeze_616: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(primals_464, -1)
    unsqueeze_617: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
    mul_1084: "f32[8, 832, 14, 14]" = torch.ops.aten.mul.Tensor(mul_1078, unsqueeze_617);  mul_1078 = unsqueeze_617 = None
    unsqueeze_618: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(primals_465, -1);  primals_465 = None
    unsqueeze_619: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
    add_858: "f32[8, 832, 14, 14]" = torch.ops.aten.add.Tensor(mul_1084, unsqueeze_619);  mul_1084 = unsqueeze_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_151: "f32[8, 832, 14, 14]" = torch.ops.aten.relu.default(add_858);  add_858 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_151 = torch.ops.aten.split_with_sizes.default(relu_151, [208, 208, 208, 208], 1)
    getitem_916: "f32[8, 208, 14, 14]" = split_with_sizes_151[0]
    convolution_155: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_916, primals_466, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_859: "i64[]" = torch.ops.aten.add.Tensor(primals_980, 1)
    var_mean_155 = torch.ops.aten.var_mean.correction(convolution_155, [0, 2, 3], correction = 0, keepdim = True)
    getitem_920: "f32[1, 208, 1, 1]" = var_mean_155[0]
    getitem_921: "f32[1, 208, 1, 1]" = var_mean_155[1];  var_mean_155 = None
    add_860: "f32[1, 208, 1, 1]" = torch.ops.aten.add.Tensor(getitem_920, 1e-05)
    rsqrt_155: "f32[1, 208, 1, 1]" = torch.ops.aten.rsqrt.default(add_860);  add_860 = None
    sub_155: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_155, getitem_921)
    mul_1085: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_155, rsqrt_155);  sub_155 = None
    squeeze_465: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_921, [0, 2, 3]);  getitem_921 = None
    squeeze_466: "f32[208]" = torch.ops.aten.squeeze.dims(rsqrt_155, [0, 2, 3]);  rsqrt_155 = None
    mul_1086: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_465, 0.1)
    mul_1087: "f32[208]" = torch.ops.aten.mul.Tensor(primals_978, 0.9)
    add_861: "f32[208]" = torch.ops.aten.add.Tensor(mul_1086, mul_1087);  mul_1086 = mul_1087 = None
    squeeze_467: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_920, [0, 2, 3]);  getitem_920 = None
    mul_1088: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_467, 1.0025575447570332);  squeeze_467 = None
    mul_1089: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1088, 0.1);  mul_1088 = None
    mul_1090: "f32[208]" = torch.ops.aten.mul.Tensor(primals_979, 0.9)
    add_862: "f32[208]" = torch.ops.aten.add.Tensor(mul_1089, mul_1090);  mul_1089 = mul_1090 = None
    unsqueeze_620: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_467, -1)
    unsqueeze_621: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
    mul_1091: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1085, unsqueeze_621);  mul_1085 = unsqueeze_621 = None
    unsqueeze_622: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_468, -1);  primals_468 = None
    unsqueeze_623: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
    add_863: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_1091, unsqueeze_623);  mul_1091 = unsqueeze_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_152: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_863);  add_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    getitem_923: "f32[8, 208, 14, 14]" = split_with_sizes_151[1]
    convolution_156: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_923, primals_469, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_864: "i64[]" = torch.ops.aten.add.Tensor(primals_983, 1)
    var_mean_156 = torch.ops.aten.var_mean.correction(convolution_156, [0, 2, 3], correction = 0, keepdim = True)
    getitem_926: "f32[1, 208, 1, 1]" = var_mean_156[0]
    getitem_927: "f32[1, 208, 1, 1]" = var_mean_156[1];  var_mean_156 = None
    add_865: "f32[1, 208, 1, 1]" = torch.ops.aten.add.Tensor(getitem_926, 1e-05)
    rsqrt_156: "f32[1, 208, 1, 1]" = torch.ops.aten.rsqrt.default(add_865);  add_865 = None
    sub_156: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_156, getitem_927)
    mul_1092: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_156, rsqrt_156);  sub_156 = None
    squeeze_468: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_927, [0, 2, 3]);  getitem_927 = None
    squeeze_469: "f32[208]" = torch.ops.aten.squeeze.dims(rsqrt_156, [0, 2, 3]);  rsqrt_156 = None
    mul_1093: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_468, 0.1)
    mul_1094: "f32[208]" = torch.ops.aten.mul.Tensor(primals_981, 0.9)
    add_866: "f32[208]" = torch.ops.aten.add.Tensor(mul_1093, mul_1094);  mul_1093 = mul_1094 = None
    squeeze_470: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_926, [0, 2, 3]);  getitem_926 = None
    mul_1095: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_470, 1.0025575447570332);  squeeze_470 = None
    mul_1096: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1095, 0.1);  mul_1095 = None
    mul_1097: "f32[208]" = torch.ops.aten.mul.Tensor(primals_982, 0.9)
    add_867: "f32[208]" = torch.ops.aten.add.Tensor(mul_1096, mul_1097);  mul_1096 = mul_1097 = None
    unsqueeze_624: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_470, -1)
    unsqueeze_625: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
    mul_1098: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1092, unsqueeze_625);  mul_1092 = unsqueeze_625 = None
    unsqueeze_626: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_471, -1);  primals_471 = None
    unsqueeze_627: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
    add_868: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_1098, unsqueeze_627);  mul_1098 = unsqueeze_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_153: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_868);  add_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    getitem_930: "f32[8, 208, 14, 14]" = split_with_sizes_151[2]
    convolution_157: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_930, primals_472, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_869: "i64[]" = torch.ops.aten.add.Tensor(primals_986, 1)
    var_mean_157 = torch.ops.aten.var_mean.correction(convolution_157, [0, 2, 3], correction = 0, keepdim = True)
    getitem_932: "f32[1, 208, 1, 1]" = var_mean_157[0]
    getitem_933: "f32[1, 208, 1, 1]" = var_mean_157[1];  var_mean_157 = None
    add_870: "f32[1, 208, 1, 1]" = torch.ops.aten.add.Tensor(getitem_932, 1e-05)
    rsqrt_157: "f32[1, 208, 1, 1]" = torch.ops.aten.rsqrt.default(add_870);  add_870 = None
    sub_157: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_157, getitem_933)
    mul_1099: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_157, rsqrt_157);  sub_157 = None
    squeeze_471: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_933, [0, 2, 3]);  getitem_933 = None
    squeeze_472: "f32[208]" = torch.ops.aten.squeeze.dims(rsqrt_157, [0, 2, 3]);  rsqrt_157 = None
    mul_1100: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_471, 0.1)
    mul_1101: "f32[208]" = torch.ops.aten.mul.Tensor(primals_984, 0.9)
    add_871: "f32[208]" = torch.ops.aten.add.Tensor(mul_1100, mul_1101);  mul_1100 = mul_1101 = None
    squeeze_473: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_932, [0, 2, 3]);  getitem_932 = None
    mul_1102: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_473, 1.0025575447570332);  squeeze_473 = None
    mul_1103: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1102, 0.1);  mul_1102 = None
    mul_1104: "f32[208]" = torch.ops.aten.mul.Tensor(primals_985, 0.9)
    add_872: "f32[208]" = torch.ops.aten.add.Tensor(mul_1103, mul_1104);  mul_1103 = mul_1104 = None
    unsqueeze_628: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_473, -1)
    unsqueeze_629: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
    mul_1105: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1099, unsqueeze_629);  mul_1099 = unsqueeze_629 = None
    unsqueeze_630: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_474, -1);  primals_474 = None
    unsqueeze_631: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
    add_873: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_1105, unsqueeze_631);  mul_1105 = unsqueeze_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_154: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_873);  add_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:99, code: spo.append(self.pool(spx[-1]))
    getitem_937: "f32[8, 208, 14, 14]" = split_with_sizes_151[3];  split_with_sizes_151 = None
    avg_pool2d_3: "f32[8, 208, 7, 7]" = torch.ops.aten.avg_pool2d.default(getitem_937, [3, 3], [2, 2], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    cat_30: "f32[8, 832, 7, 7]" = torch.ops.aten.cat.default([relu_152, relu_153, relu_154, avg_pool2d_3], 1);  avg_pool2d_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_158: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_30, primals_475, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_874: "i64[]" = torch.ops.aten.add.Tensor(primals_989, 1)
    var_mean_158 = torch.ops.aten.var_mean.correction(convolution_158, [0, 2, 3], correction = 0, keepdim = True)
    getitem_938: "f32[1, 2048, 1, 1]" = var_mean_158[0]
    getitem_939: "f32[1, 2048, 1, 1]" = var_mean_158[1];  var_mean_158 = None
    add_875: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_938, 1e-05)
    rsqrt_158: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_875);  add_875 = None
    sub_158: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_158, getitem_939)
    mul_1106: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_158, rsqrt_158);  sub_158 = None
    squeeze_474: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_939, [0, 2, 3]);  getitem_939 = None
    squeeze_475: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_158, [0, 2, 3]);  rsqrt_158 = None
    mul_1107: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_474, 0.1)
    mul_1108: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_987, 0.9)
    add_876: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1107, mul_1108);  mul_1107 = mul_1108 = None
    squeeze_476: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_938, [0, 2, 3]);  getitem_938 = None
    mul_1109: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_476, 1.0025575447570332);  squeeze_476 = None
    mul_1110: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1109, 0.1);  mul_1109 = None
    mul_1111: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_988, 0.9)
    add_877: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1110, mul_1111);  mul_1110 = mul_1111 = None
    unsqueeze_632: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_476, -1)
    unsqueeze_633: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
    mul_1112: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1106, unsqueeze_633);  mul_1106 = unsqueeze_633 = None
    unsqueeze_634: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_477, -1);  primals_477 = None
    unsqueeze_635: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
    add_878: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_1112, unsqueeze_635);  mul_1112 = unsqueeze_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    convolution_159: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_150, primals_478, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    add_879: "i64[]" = torch.ops.aten.add.Tensor(primals_992, 1)
    var_mean_159 = torch.ops.aten.var_mean.correction(convolution_159, [0, 2, 3], correction = 0, keepdim = True)
    getitem_940: "f32[1, 2048, 1, 1]" = var_mean_159[0]
    getitem_941: "f32[1, 2048, 1, 1]" = var_mean_159[1];  var_mean_159 = None
    add_880: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_940, 1e-05)
    rsqrt_159: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_880);  add_880 = None
    sub_159: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_159, getitem_941)
    mul_1113: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_159, rsqrt_159);  sub_159 = None
    squeeze_477: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_941, [0, 2, 3]);  getitem_941 = None
    squeeze_478: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_159, [0, 2, 3]);  rsqrt_159 = None
    mul_1114: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_477, 0.1)
    mul_1115: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_990, 0.9)
    add_881: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1114, mul_1115);  mul_1114 = mul_1115 = None
    squeeze_479: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_940, [0, 2, 3]);  getitem_940 = None
    mul_1116: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_479, 1.0025575447570332);  squeeze_479 = None
    mul_1117: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1116, 0.1);  mul_1116 = None
    mul_1118: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_991, 0.9)
    add_882: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1117, mul_1118);  mul_1117 = mul_1118 = None
    unsqueeze_636: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_479, -1)
    unsqueeze_637: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
    mul_1119: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1113, unsqueeze_637);  mul_1113 = unsqueeze_637 = None
    unsqueeze_638: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_480, -1);  primals_480 = None
    unsqueeze_639: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
    add_883: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_1119, unsqueeze_639);  mul_1119 = unsqueeze_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_884: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_878, add_883);  add_878 = add_883 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_155: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_884);  add_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_160: "f32[8, 832, 7, 7]" = torch.ops.aten.convolution.default(relu_155, primals_481, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_885: "i64[]" = torch.ops.aten.add.Tensor(primals_995, 1)
    var_mean_160 = torch.ops.aten.var_mean.correction(convolution_160, [0, 2, 3], correction = 0, keepdim = True)
    getitem_942: "f32[1, 832, 1, 1]" = var_mean_160[0]
    getitem_943: "f32[1, 832, 1, 1]" = var_mean_160[1];  var_mean_160 = None
    add_886: "f32[1, 832, 1, 1]" = torch.ops.aten.add.Tensor(getitem_942, 1e-05)
    rsqrt_160: "f32[1, 832, 1, 1]" = torch.ops.aten.rsqrt.default(add_886);  add_886 = None
    sub_160: "f32[8, 832, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_160, getitem_943)
    mul_1120: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(sub_160, rsqrt_160);  sub_160 = None
    squeeze_480: "f32[832]" = torch.ops.aten.squeeze.dims(getitem_943, [0, 2, 3]);  getitem_943 = None
    squeeze_481: "f32[832]" = torch.ops.aten.squeeze.dims(rsqrt_160, [0, 2, 3]);  rsqrt_160 = None
    mul_1121: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_480, 0.1)
    mul_1122: "f32[832]" = torch.ops.aten.mul.Tensor(primals_993, 0.9)
    add_887: "f32[832]" = torch.ops.aten.add.Tensor(mul_1121, mul_1122);  mul_1121 = mul_1122 = None
    squeeze_482: "f32[832]" = torch.ops.aten.squeeze.dims(getitem_942, [0, 2, 3]);  getitem_942 = None
    mul_1123: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_482, 1.0025575447570332);  squeeze_482 = None
    mul_1124: "f32[832]" = torch.ops.aten.mul.Tensor(mul_1123, 0.1);  mul_1123 = None
    mul_1125: "f32[832]" = torch.ops.aten.mul.Tensor(primals_994, 0.9)
    add_888: "f32[832]" = torch.ops.aten.add.Tensor(mul_1124, mul_1125);  mul_1124 = mul_1125 = None
    unsqueeze_640: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(primals_482, -1)
    unsqueeze_641: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
    mul_1126: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1120, unsqueeze_641);  mul_1120 = unsqueeze_641 = None
    unsqueeze_642: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(primals_483, -1);  primals_483 = None
    unsqueeze_643: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
    add_889: "f32[8, 832, 7, 7]" = torch.ops.aten.add.Tensor(mul_1126, unsqueeze_643);  mul_1126 = unsqueeze_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_156: "f32[8, 832, 7, 7]" = torch.ops.aten.relu.default(add_889);  add_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_156 = torch.ops.aten.split_with_sizes.default(relu_156, [208, 208, 208, 208], 1)
    getitem_948: "f32[8, 208, 7, 7]" = split_with_sizes_156[0]
    convolution_161: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_948, primals_484, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_890: "i64[]" = torch.ops.aten.add.Tensor(primals_998, 1)
    var_mean_161 = torch.ops.aten.var_mean.correction(convolution_161, [0, 2, 3], correction = 0, keepdim = True)
    getitem_952: "f32[1, 208, 1, 1]" = var_mean_161[0]
    getitem_953: "f32[1, 208, 1, 1]" = var_mean_161[1];  var_mean_161 = None
    add_891: "f32[1, 208, 1, 1]" = torch.ops.aten.add.Tensor(getitem_952, 1e-05)
    rsqrt_161: "f32[1, 208, 1, 1]" = torch.ops.aten.rsqrt.default(add_891);  add_891 = None
    sub_161: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_161, getitem_953)
    mul_1127: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_161, rsqrt_161);  sub_161 = None
    squeeze_483: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_953, [0, 2, 3]);  getitem_953 = None
    squeeze_484: "f32[208]" = torch.ops.aten.squeeze.dims(rsqrt_161, [0, 2, 3]);  rsqrt_161 = None
    mul_1128: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_483, 0.1)
    mul_1129: "f32[208]" = torch.ops.aten.mul.Tensor(primals_996, 0.9)
    add_892: "f32[208]" = torch.ops.aten.add.Tensor(mul_1128, mul_1129);  mul_1128 = mul_1129 = None
    squeeze_485: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_952, [0, 2, 3]);  getitem_952 = None
    mul_1130: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_485, 1.0025575447570332);  squeeze_485 = None
    mul_1131: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1130, 0.1);  mul_1130 = None
    mul_1132: "f32[208]" = torch.ops.aten.mul.Tensor(primals_997, 0.9)
    add_893: "f32[208]" = torch.ops.aten.add.Tensor(mul_1131, mul_1132);  mul_1131 = mul_1132 = None
    unsqueeze_644: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_485, -1)
    unsqueeze_645: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
    mul_1133: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1127, unsqueeze_645);  mul_1127 = unsqueeze_645 = None
    unsqueeze_646: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_486, -1);  primals_486 = None
    unsqueeze_647: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
    add_894: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_1133, unsqueeze_647);  mul_1133 = unsqueeze_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_157: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_894);  add_894 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_955: "f32[8, 208, 7, 7]" = split_with_sizes_156[1]
    add_895: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(relu_157, getitem_955);  getitem_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_162: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(add_895, primals_487, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_896: "i64[]" = torch.ops.aten.add.Tensor(primals_1001, 1)
    var_mean_162 = torch.ops.aten.var_mean.correction(convolution_162, [0, 2, 3], correction = 0, keepdim = True)
    getitem_958: "f32[1, 208, 1, 1]" = var_mean_162[0]
    getitem_959: "f32[1, 208, 1, 1]" = var_mean_162[1];  var_mean_162 = None
    add_897: "f32[1, 208, 1, 1]" = torch.ops.aten.add.Tensor(getitem_958, 1e-05)
    rsqrt_162: "f32[1, 208, 1, 1]" = torch.ops.aten.rsqrt.default(add_897);  add_897 = None
    sub_162: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_162, getitem_959)
    mul_1134: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_162, rsqrt_162);  sub_162 = None
    squeeze_486: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_959, [0, 2, 3]);  getitem_959 = None
    squeeze_487: "f32[208]" = torch.ops.aten.squeeze.dims(rsqrt_162, [0, 2, 3]);  rsqrt_162 = None
    mul_1135: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_486, 0.1)
    mul_1136: "f32[208]" = torch.ops.aten.mul.Tensor(primals_999, 0.9)
    add_898: "f32[208]" = torch.ops.aten.add.Tensor(mul_1135, mul_1136);  mul_1135 = mul_1136 = None
    squeeze_488: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_958, [0, 2, 3]);  getitem_958 = None
    mul_1137: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_488, 1.0025575447570332);  squeeze_488 = None
    mul_1138: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1137, 0.1);  mul_1137 = None
    mul_1139: "f32[208]" = torch.ops.aten.mul.Tensor(primals_1000, 0.9)
    add_899: "f32[208]" = torch.ops.aten.add.Tensor(mul_1138, mul_1139);  mul_1138 = mul_1139 = None
    unsqueeze_648: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_488, -1)
    unsqueeze_649: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
    mul_1140: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1134, unsqueeze_649);  mul_1134 = unsqueeze_649 = None
    unsqueeze_650: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_489, -1);  primals_489 = None
    unsqueeze_651: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
    add_900: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_1140, unsqueeze_651);  mul_1140 = unsqueeze_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_158: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_900);  add_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_962: "f32[8, 208, 7, 7]" = split_with_sizes_156[2]
    add_901: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(relu_158, getitem_962);  getitem_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_163: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(add_901, primals_490, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_902: "i64[]" = torch.ops.aten.add.Tensor(primals_1004, 1)
    var_mean_163 = torch.ops.aten.var_mean.correction(convolution_163, [0, 2, 3], correction = 0, keepdim = True)
    getitem_964: "f32[1, 208, 1, 1]" = var_mean_163[0]
    getitem_965: "f32[1, 208, 1, 1]" = var_mean_163[1];  var_mean_163 = None
    add_903: "f32[1, 208, 1, 1]" = torch.ops.aten.add.Tensor(getitem_964, 1e-05)
    rsqrt_163: "f32[1, 208, 1, 1]" = torch.ops.aten.rsqrt.default(add_903);  add_903 = None
    sub_163: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_163, getitem_965)
    mul_1141: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_163, rsqrt_163);  sub_163 = None
    squeeze_489: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_965, [0, 2, 3]);  getitem_965 = None
    squeeze_490: "f32[208]" = torch.ops.aten.squeeze.dims(rsqrt_163, [0, 2, 3]);  rsqrt_163 = None
    mul_1142: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_489, 0.1)
    mul_1143: "f32[208]" = torch.ops.aten.mul.Tensor(primals_1002, 0.9)
    add_904: "f32[208]" = torch.ops.aten.add.Tensor(mul_1142, mul_1143);  mul_1142 = mul_1143 = None
    squeeze_491: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_964, [0, 2, 3]);  getitem_964 = None
    mul_1144: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_491, 1.0025575447570332);  squeeze_491 = None
    mul_1145: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1144, 0.1);  mul_1144 = None
    mul_1146: "f32[208]" = torch.ops.aten.mul.Tensor(primals_1003, 0.9)
    add_905: "f32[208]" = torch.ops.aten.add.Tensor(mul_1145, mul_1146);  mul_1145 = mul_1146 = None
    unsqueeze_652: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_491, -1)
    unsqueeze_653: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
    mul_1147: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1141, unsqueeze_653);  mul_1141 = unsqueeze_653 = None
    unsqueeze_654: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_492, -1);  primals_492 = None
    unsqueeze_655: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
    add_906: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_1147, unsqueeze_655);  mul_1147 = unsqueeze_655 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_159: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_906);  add_906 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_969: "f32[8, 208, 7, 7]" = split_with_sizes_156[3];  split_with_sizes_156 = None
    cat_31: "f32[8, 832, 7, 7]" = torch.ops.aten.cat.default([relu_157, relu_158, relu_159, getitem_969], 1);  getitem_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_164: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_31, primals_493, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_907: "i64[]" = torch.ops.aten.add.Tensor(primals_1007, 1)
    var_mean_164 = torch.ops.aten.var_mean.correction(convolution_164, [0, 2, 3], correction = 0, keepdim = True)
    getitem_970: "f32[1, 2048, 1, 1]" = var_mean_164[0]
    getitem_971: "f32[1, 2048, 1, 1]" = var_mean_164[1];  var_mean_164 = None
    add_908: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_970, 1e-05)
    rsqrt_164: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_908);  add_908 = None
    sub_164: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_164, getitem_971)
    mul_1148: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_164, rsqrt_164);  sub_164 = None
    squeeze_492: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_971, [0, 2, 3]);  getitem_971 = None
    squeeze_493: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_164, [0, 2, 3]);  rsqrt_164 = None
    mul_1149: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_492, 0.1)
    mul_1150: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_1005, 0.9)
    add_909: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1149, mul_1150);  mul_1149 = mul_1150 = None
    squeeze_494: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_970, [0, 2, 3]);  getitem_970 = None
    mul_1151: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_494, 1.0025575447570332);  squeeze_494 = None
    mul_1152: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1151, 0.1);  mul_1151 = None
    mul_1153: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_1006, 0.9)
    add_910: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1152, mul_1153);  mul_1152 = mul_1153 = None
    unsqueeze_656: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_494, -1)
    unsqueeze_657: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
    mul_1154: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1148, unsqueeze_657);  mul_1148 = unsqueeze_657 = None
    unsqueeze_658: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_495, -1);  primals_495 = None
    unsqueeze_659: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
    add_911: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_1154, unsqueeze_659);  mul_1154 = unsqueeze_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_912: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_911, relu_155);  add_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_160: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_912);  add_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:81, code: out = self.conv1(x)
    convolution_165: "f32[8, 832, 7, 7]" = torch.ops.aten.convolution.default(relu_160, primals_496, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    add_913: "i64[]" = torch.ops.aten.add.Tensor(primals_1010, 1)
    var_mean_165 = torch.ops.aten.var_mean.correction(convolution_165, [0, 2, 3], correction = 0, keepdim = True)
    getitem_972: "f32[1, 832, 1, 1]" = var_mean_165[0]
    getitem_973: "f32[1, 832, 1, 1]" = var_mean_165[1];  var_mean_165 = None
    add_914: "f32[1, 832, 1, 1]" = torch.ops.aten.add.Tensor(getitem_972, 1e-05)
    rsqrt_165: "f32[1, 832, 1, 1]" = torch.ops.aten.rsqrt.default(add_914);  add_914 = None
    sub_165: "f32[8, 832, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_165, getitem_973)
    mul_1155: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(sub_165, rsqrt_165);  sub_165 = None
    squeeze_495: "f32[832]" = torch.ops.aten.squeeze.dims(getitem_973, [0, 2, 3]);  getitem_973 = None
    squeeze_496: "f32[832]" = torch.ops.aten.squeeze.dims(rsqrt_165, [0, 2, 3]);  rsqrt_165 = None
    mul_1156: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_495, 0.1)
    mul_1157: "f32[832]" = torch.ops.aten.mul.Tensor(primals_1008, 0.9)
    add_915: "f32[832]" = torch.ops.aten.add.Tensor(mul_1156, mul_1157);  mul_1156 = mul_1157 = None
    squeeze_497: "f32[832]" = torch.ops.aten.squeeze.dims(getitem_972, [0, 2, 3]);  getitem_972 = None
    mul_1158: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_497, 1.0025575447570332);  squeeze_497 = None
    mul_1159: "f32[832]" = torch.ops.aten.mul.Tensor(mul_1158, 0.1);  mul_1158 = None
    mul_1160: "f32[832]" = torch.ops.aten.mul.Tensor(primals_1009, 0.9)
    add_916: "f32[832]" = torch.ops.aten.add.Tensor(mul_1159, mul_1160);  mul_1159 = mul_1160 = None
    unsqueeze_660: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(primals_497, -1)
    unsqueeze_661: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
    mul_1161: "f32[8, 832, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1155, unsqueeze_661);  mul_1155 = unsqueeze_661 = None
    unsqueeze_662: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(primals_498, -1);  primals_498 = None
    unsqueeze_663: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
    add_917: "f32[8, 832, 7, 7]" = torch.ops.aten.add.Tensor(mul_1161, unsqueeze_663);  mul_1161 = unsqueeze_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    relu_161: "f32[8, 832, 7, 7]" = torch.ops.aten.relu.default(add_917);  add_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    split_with_sizes_161 = torch.ops.aten.split_with_sizes.default(relu_161, [208, 208, 208, 208], 1)
    getitem_978: "f32[8, 208, 7, 7]" = split_with_sizes_161[0]
    convolution_166: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(getitem_978, primals_499, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_918: "i64[]" = torch.ops.aten.add.Tensor(primals_1013, 1)
    var_mean_166 = torch.ops.aten.var_mean.correction(convolution_166, [0, 2, 3], correction = 0, keepdim = True)
    getitem_982: "f32[1, 208, 1, 1]" = var_mean_166[0]
    getitem_983: "f32[1, 208, 1, 1]" = var_mean_166[1];  var_mean_166 = None
    add_919: "f32[1, 208, 1, 1]" = torch.ops.aten.add.Tensor(getitem_982, 1e-05)
    rsqrt_166: "f32[1, 208, 1, 1]" = torch.ops.aten.rsqrt.default(add_919);  add_919 = None
    sub_166: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_166, getitem_983)
    mul_1162: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_166, rsqrt_166);  sub_166 = None
    squeeze_498: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_983, [0, 2, 3]);  getitem_983 = None
    squeeze_499: "f32[208]" = torch.ops.aten.squeeze.dims(rsqrt_166, [0, 2, 3]);  rsqrt_166 = None
    mul_1163: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_498, 0.1)
    mul_1164: "f32[208]" = torch.ops.aten.mul.Tensor(primals_1011, 0.9)
    add_920: "f32[208]" = torch.ops.aten.add.Tensor(mul_1163, mul_1164);  mul_1163 = mul_1164 = None
    squeeze_500: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_982, [0, 2, 3]);  getitem_982 = None
    mul_1165: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_500, 1.0025575447570332);  squeeze_500 = None
    mul_1166: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1165, 0.1);  mul_1165 = None
    mul_1167: "f32[208]" = torch.ops.aten.mul.Tensor(primals_1012, 0.9)
    add_921: "f32[208]" = torch.ops.aten.add.Tensor(mul_1166, mul_1167);  mul_1166 = mul_1167 = None
    unsqueeze_664: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_500, -1)
    unsqueeze_665: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
    mul_1168: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1162, unsqueeze_665);  mul_1162 = unsqueeze_665 = None
    unsqueeze_666: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_501, -1);  primals_501 = None
    unsqueeze_667: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
    add_922: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_1168, unsqueeze_667);  mul_1168 = unsqueeze_667 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_162: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_922);  add_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_985: "f32[8, 208, 7, 7]" = split_with_sizes_161[1]
    add_923: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(relu_162, getitem_985);  getitem_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_167: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(add_923, primals_502, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_924: "i64[]" = torch.ops.aten.add.Tensor(primals_1016, 1)
    var_mean_167 = torch.ops.aten.var_mean.correction(convolution_167, [0, 2, 3], correction = 0, keepdim = True)
    getitem_988: "f32[1, 208, 1, 1]" = var_mean_167[0]
    getitem_989: "f32[1, 208, 1, 1]" = var_mean_167[1];  var_mean_167 = None
    add_925: "f32[1, 208, 1, 1]" = torch.ops.aten.add.Tensor(getitem_988, 1e-05)
    rsqrt_167: "f32[1, 208, 1, 1]" = torch.ops.aten.rsqrt.default(add_925);  add_925 = None
    sub_167: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_167, getitem_989)
    mul_1169: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_167, rsqrt_167);  sub_167 = None
    squeeze_501: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_989, [0, 2, 3]);  getitem_989 = None
    squeeze_502: "f32[208]" = torch.ops.aten.squeeze.dims(rsqrt_167, [0, 2, 3]);  rsqrt_167 = None
    mul_1170: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_501, 0.1)
    mul_1171: "f32[208]" = torch.ops.aten.mul.Tensor(primals_1014, 0.9)
    add_926: "f32[208]" = torch.ops.aten.add.Tensor(mul_1170, mul_1171);  mul_1170 = mul_1171 = None
    squeeze_503: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_988, [0, 2, 3]);  getitem_988 = None
    mul_1172: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_503, 1.0025575447570332);  squeeze_503 = None
    mul_1173: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1172, 0.1);  mul_1172 = None
    mul_1174: "f32[208]" = torch.ops.aten.mul.Tensor(primals_1015, 0.9)
    add_927: "f32[208]" = torch.ops.aten.add.Tensor(mul_1173, mul_1174);  mul_1173 = mul_1174 = None
    unsqueeze_668: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_503, -1)
    unsqueeze_669: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
    mul_1175: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1169, unsqueeze_669);  mul_1169 = unsqueeze_669 = None
    unsqueeze_670: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_504, -1);  primals_504 = None
    unsqueeze_671: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
    add_928: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_1175, unsqueeze_671);  mul_1175 = unsqueeze_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_163: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_928);  add_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:92, code: sp = sp + spx[i]
    getitem_992: "f32[8, 208, 7, 7]" = split_with_sizes_161[2]
    add_929: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(relu_163, getitem_992);  getitem_992 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:93, code: sp = conv(sp)
    convolution_168: "f32[8, 208, 7, 7]" = torch.ops.aten.convolution.default(add_929, primals_505, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    add_930: "i64[]" = torch.ops.aten.add.Tensor(primals_1019, 1)
    var_mean_168 = torch.ops.aten.var_mean.correction(convolution_168, [0, 2, 3], correction = 0, keepdim = True)
    getitem_994: "f32[1, 208, 1, 1]" = var_mean_168[0]
    getitem_995: "f32[1, 208, 1, 1]" = var_mean_168[1];  var_mean_168 = None
    add_931: "f32[1, 208, 1, 1]" = torch.ops.aten.add.Tensor(getitem_994, 1e-05)
    rsqrt_168: "f32[1, 208, 1, 1]" = torch.ops.aten.rsqrt.default(add_931);  add_931 = None
    sub_168: "f32[8, 208, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_168, getitem_995)
    mul_1176: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(sub_168, rsqrt_168);  sub_168 = None
    squeeze_504: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_995, [0, 2, 3]);  getitem_995 = None
    squeeze_505: "f32[208]" = torch.ops.aten.squeeze.dims(rsqrt_168, [0, 2, 3]);  rsqrt_168 = None
    mul_1177: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_504, 0.1)
    mul_1178: "f32[208]" = torch.ops.aten.mul.Tensor(primals_1017, 0.9)
    add_932: "f32[208]" = torch.ops.aten.add.Tensor(mul_1177, mul_1178);  mul_1177 = mul_1178 = None
    squeeze_506: "f32[208]" = torch.ops.aten.squeeze.dims(getitem_994, [0, 2, 3]);  getitem_994 = None
    mul_1179: "f32[208]" = torch.ops.aten.mul.Tensor(squeeze_506, 1.0025575447570332);  squeeze_506 = None
    mul_1180: "f32[208]" = torch.ops.aten.mul.Tensor(mul_1179, 0.1);  mul_1179 = None
    mul_1181: "f32[208]" = torch.ops.aten.mul.Tensor(primals_1018, 0.9)
    add_933: "f32[208]" = torch.ops.aten.add.Tensor(mul_1180, mul_1181);  mul_1180 = mul_1181 = None
    unsqueeze_672: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_506, -1)
    unsqueeze_673: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
    mul_1182: "f32[8, 208, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1176, unsqueeze_673);  mul_1176 = unsqueeze_673 = None
    unsqueeze_674: "f32[208, 1]" = torch.ops.aten.unsqueeze.default(primals_507, -1);  primals_507 = None
    unsqueeze_675: "f32[208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
    add_934: "f32[8, 208, 7, 7]" = torch.ops.aten.add.Tensor(mul_1182, unsqueeze_675);  mul_1182 = unsqueeze_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    relu_164: "f32[8, 208, 7, 7]" = torch.ops.aten.relu.default(add_934);  add_934 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:102, code: out = torch.cat(spo, 1)
    getitem_999: "f32[8, 208, 7, 7]" = split_with_sizes_161[3];  split_with_sizes_161 = None
    cat_32: "f32[8, 832, 7, 7]" = torch.ops.aten.cat.default([relu_162, relu_163, relu_164, getitem_999], 1);  getitem_999 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:104, code: out = self.conv3(out)
    convolution_169: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(cat_32, primals_508, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    add_935: "i64[]" = torch.ops.aten.add.Tensor(primals_1022, 1)
    var_mean_169 = torch.ops.aten.var_mean.correction(convolution_169, [0, 2, 3], correction = 0, keepdim = True)
    getitem_1000: "f32[1, 2048, 1, 1]" = var_mean_169[0]
    getitem_1001: "f32[1, 2048, 1, 1]" = var_mean_169[1];  var_mean_169 = None
    add_936: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_1000, 1e-05)
    rsqrt_169: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_936);  add_936 = None
    sub_169: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_169, getitem_1001)
    mul_1183: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_169, rsqrt_169);  sub_169 = None
    squeeze_507: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_1001, [0, 2, 3]);  getitem_1001 = None
    squeeze_508: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_169, [0, 2, 3]);  rsqrt_169 = None
    mul_1184: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_507, 0.1)
    mul_1185: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_1020, 0.9)
    add_937: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1184, mul_1185);  mul_1184 = mul_1185 = None
    squeeze_509: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_1000, [0, 2, 3]);  getitem_1000 = None
    mul_1186: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_509, 1.0025575447570332);  squeeze_509 = None
    mul_1187: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1186, 0.1);  mul_1186 = None
    mul_1188: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_1021, 0.9)
    add_938: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1187, mul_1188);  mul_1187 = mul_1188 = None
    unsqueeze_676: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_509, -1)
    unsqueeze_677: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
    mul_1189: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_1183, unsqueeze_677);  mul_1183 = unsqueeze_677 = None
    unsqueeze_678: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_510, -1);  primals_510 = None
    unsqueeze_679: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
    add_939: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_1189, unsqueeze_679);  mul_1189 = unsqueeze_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:113, code: out += shortcut
    add_940: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_939, relu_160);  add_939 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    relu_165: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_940);  add_940 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_165, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 2048]" = torch.ops.aten.view.default(mean, [8, 2048]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    permute: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_511, [1, 0]);  primals_511 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_512, view, permute);  primals_512 = None
    permute_1: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:114, code: out = self.relu(out)
    alias_167: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_165);  relu_165 = None
    alias_168: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_167);  alias_167 = None
    le: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_168, 0);  alias_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_680: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_507, 0);  squeeze_507 = None
    unsqueeze_681: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_170: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(relu_164);  relu_164 = None
    alias_171: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(alias_170);  alias_170 = None
    le_1: "b8[8, 208, 7, 7]" = torch.ops.aten.le.Scalar(alias_171, 0);  alias_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_692: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(squeeze_504, 0);  squeeze_504 = None
    unsqueeze_693: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_173: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(relu_163);  relu_163 = None
    alias_174: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(alias_173);  alias_173 = None
    le_2: "b8[8, 208, 7, 7]" = torch.ops.aten.le.Scalar(alias_174, 0);  alias_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_704: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(squeeze_501, 0);  squeeze_501 = None
    unsqueeze_705: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
    unsqueeze_706: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_176: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(relu_162);  relu_162 = None
    alias_177: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(alias_176);  alias_176 = None
    le_3: "b8[8, 208, 7, 7]" = torch.ops.aten.le.Scalar(alias_177, 0);  alias_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_716: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(squeeze_498, 0);  squeeze_498 = None
    unsqueeze_717: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
    unsqueeze_718: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_179: "f32[8, 832, 7, 7]" = torch.ops.aten.alias.default(relu_161);  relu_161 = None
    alias_180: "f32[8, 832, 7, 7]" = torch.ops.aten.alias.default(alias_179);  alias_179 = None
    le_4: "b8[8, 832, 7, 7]" = torch.ops.aten.le.Scalar(alias_180, 0);  alias_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_728: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(squeeze_495, 0);  squeeze_495 = None
    unsqueeze_729: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
    unsqueeze_730: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_740: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_492, 0);  squeeze_492 = None
    unsqueeze_741: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
    unsqueeze_742: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_185: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(relu_159);  relu_159 = None
    alias_186: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(alias_185);  alias_185 = None
    le_6: "b8[8, 208, 7, 7]" = torch.ops.aten.le.Scalar(alias_186, 0);  alias_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_752: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(squeeze_489, 0);  squeeze_489 = None
    unsqueeze_753: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
    unsqueeze_754: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_188: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(relu_158);  relu_158 = None
    alias_189: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(alias_188);  alias_188 = None
    le_7: "b8[8, 208, 7, 7]" = torch.ops.aten.le.Scalar(alias_189, 0);  alias_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_764: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(squeeze_486, 0);  squeeze_486 = None
    unsqueeze_765: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
    unsqueeze_766: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_191: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(relu_157);  relu_157 = None
    alias_192: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(alias_191);  alias_191 = None
    le_8: "b8[8, 208, 7, 7]" = torch.ops.aten.le.Scalar(alias_192, 0);  alias_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_776: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(squeeze_483, 0);  squeeze_483 = None
    unsqueeze_777: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
    unsqueeze_778: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_194: "f32[8, 832, 7, 7]" = torch.ops.aten.alias.default(relu_156);  relu_156 = None
    alias_195: "f32[8, 832, 7, 7]" = torch.ops.aten.alias.default(alias_194);  alias_194 = None
    le_9: "b8[8, 832, 7, 7]" = torch.ops.aten.le.Scalar(alias_195, 0);  alias_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_788: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(squeeze_480, 0);  squeeze_480 = None
    unsqueeze_789: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
    unsqueeze_790: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    unsqueeze_800: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_477, 0);  squeeze_477 = None
    unsqueeze_801: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
    unsqueeze_802: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_812: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_474, 0);  squeeze_474 = None
    unsqueeze_813: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
    unsqueeze_814: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_200: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(relu_154);  relu_154 = None
    alias_201: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(alias_200);  alias_200 = None
    le_11: "b8[8, 208, 7, 7]" = torch.ops.aten.le.Scalar(alias_201, 0);  alias_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_824: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(squeeze_471, 0);  squeeze_471 = None
    unsqueeze_825: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
    unsqueeze_826: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_203: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(relu_153);  relu_153 = None
    alias_204: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(alias_203);  alias_203 = None
    le_12: "b8[8, 208, 7, 7]" = torch.ops.aten.le.Scalar(alias_204, 0);  alias_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_836: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(squeeze_468, 0);  squeeze_468 = None
    unsqueeze_837: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
    unsqueeze_838: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_206: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(relu_152);  relu_152 = None
    alias_207: "f32[8, 208, 7, 7]" = torch.ops.aten.alias.default(alias_206);  alias_206 = None
    le_13: "b8[8, 208, 7, 7]" = torch.ops.aten.le.Scalar(alias_207, 0);  alias_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_848: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(squeeze_465, 0);  squeeze_465 = None
    unsqueeze_849: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
    unsqueeze_850: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_209: "f32[8, 832, 14, 14]" = torch.ops.aten.alias.default(relu_151);  relu_151 = None
    alias_210: "f32[8, 832, 14, 14]" = torch.ops.aten.alias.default(alias_209);  alias_209 = None
    le_14: "b8[8, 832, 14, 14]" = torch.ops.aten.le.Scalar(alias_210, 0);  alias_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_860: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(squeeze_462, 0);  squeeze_462 = None
    unsqueeze_861: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
    unsqueeze_862: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_872: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_459, 0);  squeeze_459 = None
    unsqueeze_873: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
    unsqueeze_874: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_215: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_149);  relu_149 = None
    alias_216: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_215);  alias_215 = None
    le_16: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_216, 0);  alias_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_884: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_456, 0);  squeeze_456 = None
    unsqueeze_885: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
    unsqueeze_886: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_218: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_148);  relu_148 = None
    alias_219: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_218);  alias_218 = None
    le_17: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_219, 0);  alias_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_896: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_453, 0);  squeeze_453 = None
    unsqueeze_897: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
    unsqueeze_898: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_221: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_147);  relu_147 = None
    alias_222: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_221);  alias_221 = None
    le_18: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_222, 0);  alias_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_908: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_450, 0);  squeeze_450 = None
    unsqueeze_909: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
    unsqueeze_910: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_224: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_146);  relu_146 = None
    alias_225: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_224);  alias_224 = None
    le_19: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_225, 0);  alias_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_920: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_447, 0);  squeeze_447 = None
    unsqueeze_921: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 2);  unsqueeze_920 = None
    unsqueeze_922: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 3);  unsqueeze_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_932: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_444, 0);  squeeze_444 = None
    unsqueeze_933: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 2);  unsqueeze_932 = None
    unsqueeze_934: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 3);  unsqueeze_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_230: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_144);  relu_144 = None
    alias_231: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_230);  alias_230 = None
    le_21: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_231, 0);  alias_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_944: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_441, 0);  squeeze_441 = None
    unsqueeze_945: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 2);  unsqueeze_944 = None
    unsqueeze_946: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 3);  unsqueeze_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_233: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_143);  relu_143 = None
    alias_234: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_233);  alias_233 = None
    le_22: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_234, 0);  alias_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_956: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_438, 0);  squeeze_438 = None
    unsqueeze_957: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 2);  unsqueeze_956 = None
    unsqueeze_958: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 3);  unsqueeze_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_236: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_142);  relu_142 = None
    alias_237: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_236);  alias_236 = None
    le_23: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_237, 0);  alias_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_968: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_435, 0);  squeeze_435 = None
    unsqueeze_969: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
    unsqueeze_970: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_239: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_141);  relu_141 = None
    alias_240: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_239);  alias_239 = None
    le_24: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_240, 0);  alias_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_980: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_432, 0);  squeeze_432 = None
    unsqueeze_981: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 2);  unsqueeze_980 = None
    unsqueeze_982: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 3);  unsqueeze_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_992: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_429, 0);  squeeze_429 = None
    unsqueeze_993: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 2);  unsqueeze_992 = None
    unsqueeze_994: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 3);  unsqueeze_993 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_245: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_139);  relu_139 = None
    alias_246: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_245);  alias_245 = None
    le_26: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_246, 0);  alias_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1004: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_426, 0);  squeeze_426 = None
    unsqueeze_1005: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 2);  unsqueeze_1004 = None
    unsqueeze_1006: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 3);  unsqueeze_1005 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_248: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_138);  relu_138 = None
    alias_249: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_248);  alias_248 = None
    le_27: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_249, 0);  alias_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1016: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_423, 0);  squeeze_423 = None
    unsqueeze_1017: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 2);  unsqueeze_1016 = None
    unsqueeze_1018: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 3);  unsqueeze_1017 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_251: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_137);  relu_137 = None
    alias_252: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_251);  alias_251 = None
    le_28: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_252, 0);  alias_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1028: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_420, 0);  squeeze_420 = None
    unsqueeze_1029: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 2);  unsqueeze_1028 = None
    unsqueeze_1030: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 3);  unsqueeze_1029 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_254: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_136);  relu_136 = None
    alias_255: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_254);  alias_254 = None
    le_29: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_255, 0);  alias_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1040: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_417, 0);  squeeze_417 = None
    unsqueeze_1041: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 2);  unsqueeze_1040 = None
    unsqueeze_1042: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 3);  unsqueeze_1041 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1052: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_414, 0);  squeeze_414 = None
    unsqueeze_1053: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 2);  unsqueeze_1052 = None
    unsqueeze_1054: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 3);  unsqueeze_1053 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_260: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_134);  relu_134 = None
    alias_261: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_260);  alias_260 = None
    le_31: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_261, 0);  alias_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1064: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_411, 0);  squeeze_411 = None
    unsqueeze_1065: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 2);  unsqueeze_1064 = None
    unsqueeze_1066: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 3);  unsqueeze_1065 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_263: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_133);  relu_133 = None
    alias_264: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_263);  alias_263 = None
    le_32: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_264, 0);  alias_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1076: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_408, 0);  squeeze_408 = None
    unsqueeze_1077: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, 2);  unsqueeze_1076 = None
    unsqueeze_1078: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 3);  unsqueeze_1077 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_266: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_132);  relu_132 = None
    alias_267: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_266);  alias_266 = None
    le_33: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_267, 0);  alias_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1088: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_405, 0);  squeeze_405 = None
    unsqueeze_1089: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, 2);  unsqueeze_1088 = None
    unsqueeze_1090: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 3);  unsqueeze_1089 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_269: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_131);  relu_131 = None
    alias_270: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_269);  alias_269 = None
    le_34: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_270, 0);  alias_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1100: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_402, 0);  squeeze_402 = None
    unsqueeze_1101: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, 2);  unsqueeze_1100 = None
    unsqueeze_1102: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 3);  unsqueeze_1101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1112: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_399, 0);  squeeze_399 = None
    unsqueeze_1113: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, 2);  unsqueeze_1112 = None
    unsqueeze_1114: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 3);  unsqueeze_1113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_275: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_129);  relu_129 = None
    alias_276: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_275);  alias_275 = None
    le_36: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_276, 0);  alias_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1124: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_396, 0);  squeeze_396 = None
    unsqueeze_1125: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, 2);  unsqueeze_1124 = None
    unsqueeze_1126: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 3);  unsqueeze_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_278: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_128);  relu_128 = None
    alias_279: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_278);  alias_278 = None
    le_37: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_279, 0);  alias_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1136: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_393, 0);  squeeze_393 = None
    unsqueeze_1137: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, 2);  unsqueeze_1136 = None
    unsqueeze_1138: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 3);  unsqueeze_1137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_281: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_127);  relu_127 = None
    alias_282: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_281);  alias_281 = None
    le_38: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_282, 0);  alias_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1148: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_390, 0);  squeeze_390 = None
    unsqueeze_1149: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, 2);  unsqueeze_1148 = None
    unsqueeze_1150: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 3);  unsqueeze_1149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_284: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_126);  relu_126 = None
    alias_285: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_284);  alias_284 = None
    le_39: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_285, 0);  alias_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1160: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_387, 0);  squeeze_387 = None
    unsqueeze_1161: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, 2);  unsqueeze_1160 = None
    unsqueeze_1162: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 3);  unsqueeze_1161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1172: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_384, 0);  squeeze_384 = None
    unsqueeze_1173: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, 2);  unsqueeze_1172 = None
    unsqueeze_1174: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 3);  unsqueeze_1173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_290: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_124);  relu_124 = None
    alias_291: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_290);  alias_290 = None
    le_41: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_291, 0);  alias_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1184: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_381, 0);  squeeze_381 = None
    unsqueeze_1185: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, 2);  unsqueeze_1184 = None
    unsqueeze_1186: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 3);  unsqueeze_1185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_293: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_123);  relu_123 = None
    alias_294: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_293);  alias_293 = None
    le_42: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_294, 0);  alias_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1196: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_378, 0);  squeeze_378 = None
    unsqueeze_1197: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, 2);  unsqueeze_1196 = None
    unsqueeze_1198: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 3);  unsqueeze_1197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_296: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_122);  relu_122 = None
    alias_297: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_296);  alias_296 = None
    le_43: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_297, 0);  alias_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1208: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_375, 0);  squeeze_375 = None
    unsqueeze_1209: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, 2);  unsqueeze_1208 = None
    unsqueeze_1210: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 3);  unsqueeze_1209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_299: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_121);  relu_121 = None
    alias_300: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_299);  alias_299 = None
    le_44: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_300, 0);  alias_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1220: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_372, 0);  squeeze_372 = None
    unsqueeze_1221: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, 2);  unsqueeze_1220 = None
    unsqueeze_1222: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 3);  unsqueeze_1221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1232: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_369, 0);  squeeze_369 = None
    unsqueeze_1233: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, 2);  unsqueeze_1232 = None
    unsqueeze_1234: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 3);  unsqueeze_1233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_305: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_119);  relu_119 = None
    alias_306: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_305);  alias_305 = None
    le_46: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_306, 0);  alias_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1244: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_366, 0);  squeeze_366 = None
    unsqueeze_1245: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, 2);  unsqueeze_1244 = None
    unsqueeze_1246: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 3);  unsqueeze_1245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_308: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_118);  relu_118 = None
    alias_309: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_308);  alias_308 = None
    le_47: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_309, 0);  alias_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1256: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_363, 0);  squeeze_363 = None
    unsqueeze_1257: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, 2);  unsqueeze_1256 = None
    unsqueeze_1258: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 3);  unsqueeze_1257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_311: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_117);  relu_117 = None
    alias_312: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_311);  alias_311 = None
    le_48: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_312, 0);  alias_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1268: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_360, 0);  squeeze_360 = None
    unsqueeze_1269: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, 2);  unsqueeze_1268 = None
    unsqueeze_1270: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 3);  unsqueeze_1269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_314: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_116);  relu_116 = None
    alias_315: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_314);  alias_314 = None
    le_49: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_315, 0);  alias_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1280: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_357, 0);  squeeze_357 = None
    unsqueeze_1281: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, 2);  unsqueeze_1280 = None
    unsqueeze_1282: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 3);  unsqueeze_1281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1292: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_354, 0);  squeeze_354 = None
    unsqueeze_1293: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, 2);  unsqueeze_1292 = None
    unsqueeze_1294: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1293, 3);  unsqueeze_1293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_320: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_114);  relu_114 = None
    alias_321: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_320);  alias_320 = None
    le_51: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_321, 0);  alias_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1304: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_351, 0);  squeeze_351 = None
    unsqueeze_1305: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, 2);  unsqueeze_1304 = None
    unsqueeze_1306: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1305, 3);  unsqueeze_1305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_323: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_113);  relu_113 = None
    alias_324: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_323);  alias_323 = None
    le_52: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_324, 0);  alias_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1316: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_348, 0);  squeeze_348 = None
    unsqueeze_1317: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, 2);  unsqueeze_1316 = None
    unsqueeze_1318: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1317, 3);  unsqueeze_1317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_326: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_112);  relu_112 = None
    alias_327: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_326);  alias_326 = None
    le_53: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_327, 0);  alias_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1328: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_345, 0);  squeeze_345 = None
    unsqueeze_1329: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, 2);  unsqueeze_1328 = None
    unsqueeze_1330: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1329, 3);  unsqueeze_1329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_329: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_111);  relu_111 = None
    alias_330: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_329);  alias_329 = None
    le_54: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_330, 0);  alias_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1340: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_342, 0);  squeeze_342 = None
    unsqueeze_1341: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, 2);  unsqueeze_1340 = None
    unsqueeze_1342: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 3);  unsqueeze_1341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1352: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_339, 0);  squeeze_339 = None
    unsqueeze_1353: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, 2);  unsqueeze_1352 = None
    unsqueeze_1354: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 3);  unsqueeze_1353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_335: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_109);  relu_109 = None
    alias_336: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_335);  alias_335 = None
    le_56: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_336, 0);  alias_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1364: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_336, 0);  squeeze_336 = None
    unsqueeze_1365: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, 2);  unsqueeze_1364 = None
    unsqueeze_1366: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 3);  unsqueeze_1365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_338: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_108);  relu_108 = None
    alias_339: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_338);  alias_338 = None
    le_57: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_339, 0);  alias_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1376: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_333, 0);  squeeze_333 = None
    unsqueeze_1377: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, 2);  unsqueeze_1376 = None
    unsqueeze_1378: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 3);  unsqueeze_1377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_341: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_107);  relu_107 = None
    alias_342: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_341);  alias_341 = None
    le_58: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_342, 0);  alias_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1388: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_330, 0);  squeeze_330 = None
    unsqueeze_1389: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, 2);  unsqueeze_1388 = None
    unsqueeze_1390: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1389, 3);  unsqueeze_1389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_344: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_106);  relu_106 = None
    alias_345: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_344);  alias_344 = None
    le_59: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_345, 0);  alias_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1400: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_327, 0);  squeeze_327 = None
    unsqueeze_1401: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, 2);  unsqueeze_1400 = None
    unsqueeze_1402: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1401, 3);  unsqueeze_1401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1412: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_324, 0);  squeeze_324 = None
    unsqueeze_1413: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, 2);  unsqueeze_1412 = None
    unsqueeze_1414: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1413, 3);  unsqueeze_1413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_350: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_104);  relu_104 = None
    alias_351: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_350);  alias_350 = None
    le_61: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_351, 0);  alias_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1424: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_321, 0);  squeeze_321 = None
    unsqueeze_1425: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, 2);  unsqueeze_1424 = None
    unsqueeze_1426: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1425, 3);  unsqueeze_1425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_353: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_103);  relu_103 = None
    alias_354: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_353);  alias_353 = None
    le_62: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_354, 0);  alias_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1436: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_318, 0);  squeeze_318 = None
    unsqueeze_1437: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1436, 2);  unsqueeze_1436 = None
    unsqueeze_1438: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1437, 3);  unsqueeze_1437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_356: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_102);  relu_102 = None
    alias_357: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_356);  alias_356 = None
    le_63: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_357, 0);  alias_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1448: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_315, 0);  squeeze_315 = None
    unsqueeze_1449: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, 2);  unsqueeze_1448 = None
    unsqueeze_1450: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1449, 3);  unsqueeze_1449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_359: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_101);  relu_101 = None
    alias_360: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_359);  alias_359 = None
    le_64: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_360, 0);  alias_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1460: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_312, 0);  squeeze_312 = None
    unsqueeze_1461: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, 2);  unsqueeze_1460 = None
    unsqueeze_1462: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1461, 3);  unsqueeze_1461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1472: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_309, 0);  squeeze_309 = None
    unsqueeze_1473: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, 2);  unsqueeze_1472 = None
    unsqueeze_1474: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1473, 3);  unsqueeze_1473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_365: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_99);  relu_99 = None
    alias_366: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_365);  alias_365 = None
    le_66: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_366, 0);  alias_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1484: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_306, 0);  squeeze_306 = None
    unsqueeze_1485: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, 2);  unsqueeze_1484 = None
    unsqueeze_1486: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1485, 3);  unsqueeze_1485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_368: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_98);  relu_98 = None
    alias_369: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_368);  alias_368 = None
    le_67: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_369, 0);  alias_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1496: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_303, 0);  squeeze_303 = None
    unsqueeze_1497: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1496, 2);  unsqueeze_1496 = None
    unsqueeze_1498: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1497, 3);  unsqueeze_1497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_371: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_97);  relu_97 = None
    alias_372: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_371);  alias_371 = None
    le_68: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_372, 0);  alias_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1508: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_300, 0);  squeeze_300 = None
    unsqueeze_1509: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1508, 2);  unsqueeze_1508 = None
    unsqueeze_1510: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1509, 3);  unsqueeze_1509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_374: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_96);  relu_96 = None
    alias_375: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_374);  alias_374 = None
    le_69: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_375, 0);  alias_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1520: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_297, 0);  squeeze_297 = None
    unsqueeze_1521: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1520, 2);  unsqueeze_1520 = None
    unsqueeze_1522: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1521, 3);  unsqueeze_1521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1532: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_294, 0);  squeeze_294 = None
    unsqueeze_1533: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1532, 2);  unsqueeze_1532 = None
    unsqueeze_1534: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1533, 3);  unsqueeze_1533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_380: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_94);  relu_94 = None
    alias_381: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_380);  alias_380 = None
    le_71: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_381, 0);  alias_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1544: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_291, 0);  squeeze_291 = None
    unsqueeze_1545: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1544, 2);  unsqueeze_1544 = None
    unsqueeze_1546: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1545, 3);  unsqueeze_1545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_383: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_93);  relu_93 = None
    alias_384: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_383);  alias_383 = None
    le_72: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_384, 0);  alias_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1556: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_288, 0);  squeeze_288 = None
    unsqueeze_1557: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1556, 2);  unsqueeze_1556 = None
    unsqueeze_1558: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1557, 3);  unsqueeze_1557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_386: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_92);  relu_92 = None
    alias_387: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_386);  alias_386 = None
    le_73: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_387, 0);  alias_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1568: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_285, 0);  squeeze_285 = None
    unsqueeze_1569: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1568, 2);  unsqueeze_1568 = None
    unsqueeze_1570: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1569, 3);  unsqueeze_1569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_389: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_91);  relu_91 = None
    alias_390: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_389);  alias_389 = None
    le_74: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_390, 0);  alias_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1580: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_282, 0);  squeeze_282 = None
    unsqueeze_1581: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1580, 2);  unsqueeze_1580 = None
    unsqueeze_1582: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1581, 3);  unsqueeze_1581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1592: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_279, 0);  squeeze_279 = None
    unsqueeze_1593: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1592, 2);  unsqueeze_1592 = None
    unsqueeze_1594: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1593, 3);  unsqueeze_1593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_395: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_89);  relu_89 = None
    alias_396: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_395);  alias_395 = None
    le_76: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_396, 0);  alias_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1604: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_276, 0);  squeeze_276 = None
    unsqueeze_1605: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1604, 2);  unsqueeze_1604 = None
    unsqueeze_1606: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1605, 3);  unsqueeze_1605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_398: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_88);  relu_88 = None
    alias_399: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_398);  alias_398 = None
    le_77: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_399, 0);  alias_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1616: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_273, 0);  squeeze_273 = None
    unsqueeze_1617: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1616, 2);  unsqueeze_1616 = None
    unsqueeze_1618: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1617, 3);  unsqueeze_1617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_401: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_87);  relu_87 = None
    alias_402: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_401);  alias_401 = None
    le_78: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_402, 0);  alias_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1628: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_270, 0);  squeeze_270 = None
    unsqueeze_1629: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1628, 2);  unsqueeze_1628 = None
    unsqueeze_1630: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1629, 3);  unsqueeze_1629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_404: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_86);  relu_86 = None
    alias_405: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_404);  alias_404 = None
    le_79: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_405, 0);  alias_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1640: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_267, 0);  squeeze_267 = None
    unsqueeze_1641: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1640, 2);  unsqueeze_1640 = None
    unsqueeze_1642: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1641, 3);  unsqueeze_1641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1652: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_264, 0);  squeeze_264 = None
    unsqueeze_1653: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1652, 2);  unsqueeze_1652 = None
    unsqueeze_1654: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1653, 3);  unsqueeze_1653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_410: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_84);  relu_84 = None
    alias_411: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_410);  alias_410 = None
    le_81: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_411, 0);  alias_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1664: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_261, 0);  squeeze_261 = None
    unsqueeze_1665: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1664, 2);  unsqueeze_1664 = None
    unsqueeze_1666: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1665, 3);  unsqueeze_1665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_413: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_83);  relu_83 = None
    alias_414: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_413);  alias_413 = None
    le_82: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_414, 0);  alias_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1676: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_258, 0);  squeeze_258 = None
    unsqueeze_1677: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1676, 2);  unsqueeze_1676 = None
    unsqueeze_1678: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1677, 3);  unsqueeze_1677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_416: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_82);  relu_82 = None
    alias_417: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_416);  alias_416 = None
    le_83: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_417, 0);  alias_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1688: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_255, 0);  squeeze_255 = None
    unsqueeze_1689: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1688, 2);  unsqueeze_1688 = None
    unsqueeze_1690: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1689, 3);  unsqueeze_1689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_419: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_81);  relu_81 = None
    alias_420: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_419);  alias_419 = None
    le_84: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_420, 0);  alias_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1700: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
    unsqueeze_1701: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1700, 2);  unsqueeze_1700 = None
    unsqueeze_1702: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1701, 3);  unsqueeze_1701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1712: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
    unsqueeze_1713: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1712, 2);  unsqueeze_1712 = None
    unsqueeze_1714: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1713, 3);  unsqueeze_1713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_425: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_79);  relu_79 = None
    alias_426: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_425);  alias_425 = None
    le_86: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_426, 0);  alias_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1724: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
    unsqueeze_1725: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1724, 2);  unsqueeze_1724 = None
    unsqueeze_1726: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1725, 3);  unsqueeze_1725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_428: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_78);  relu_78 = None
    alias_429: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_428);  alias_428 = None
    le_87: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_429, 0);  alias_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1736: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
    unsqueeze_1737: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1736, 2);  unsqueeze_1736 = None
    unsqueeze_1738: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1737, 3);  unsqueeze_1737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_431: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_77);  relu_77 = None
    alias_432: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_431);  alias_431 = None
    le_88: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_432, 0);  alias_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1748: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
    unsqueeze_1749: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1748, 2);  unsqueeze_1748 = None
    unsqueeze_1750: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1749, 3);  unsqueeze_1749 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_434: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_76);  relu_76 = None
    alias_435: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_434);  alias_434 = None
    le_89: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_435, 0);  alias_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1760: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
    unsqueeze_1761: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1760, 2);  unsqueeze_1760 = None
    unsqueeze_1762: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1761, 3);  unsqueeze_1761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1772: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
    unsqueeze_1773: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1772, 2);  unsqueeze_1772 = None
    unsqueeze_1774: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1773, 3);  unsqueeze_1773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_440: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_74);  relu_74 = None
    alias_441: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_440);  alias_440 = None
    le_91: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_441, 0);  alias_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1784: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
    unsqueeze_1785: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1784, 2);  unsqueeze_1784 = None
    unsqueeze_1786: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1785, 3);  unsqueeze_1785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_443: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_73);  relu_73 = None
    alias_444: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_443);  alias_443 = None
    le_92: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_444, 0);  alias_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1796: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
    unsqueeze_1797: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1796, 2);  unsqueeze_1796 = None
    unsqueeze_1798: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1797, 3);  unsqueeze_1797 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_446: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_72);  relu_72 = None
    alias_447: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_446);  alias_446 = None
    le_93: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_447, 0);  alias_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1808: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
    unsqueeze_1809: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1808, 2);  unsqueeze_1808 = None
    unsqueeze_1810: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1809, 3);  unsqueeze_1809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_449: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_71);  relu_71 = None
    alias_450: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_449);  alias_449 = None
    le_94: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_450, 0);  alias_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1820: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
    unsqueeze_1821: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1820, 2);  unsqueeze_1820 = None
    unsqueeze_1822: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1821, 3);  unsqueeze_1821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1832: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
    unsqueeze_1833: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1832, 2);  unsqueeze_1832 = None
    unsqueeze_1834: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1833, 3);  unsqueeze_1833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_455: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_69);  relu_69 = None
    alias_456: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_455);  alias_455 = None
    le_96: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_456, 0);  alias_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1844: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
    unsqueeze_1845: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1844, 2);  unsqueeze_1844 = None
    unsqueeze_1846: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1845, 3);  unsqueeze_1845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_458: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_68);  relu_68 = None
    alias_459: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_458);  alias_458 = None
    le_97: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_459, 0);  alias_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1856: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
    unsqueeze_1857: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1856, 2);  unsqueeze_1856 = None
    unsqueeze_1858: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1857, 3);  unsqueeze_1857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_461: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_67);  relu_67 = None
    alias_462: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_461);  alias_461 = None
    le_98: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_462, 0);  alias_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1868: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
    unsqueeze_1869: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1868, 2);  unsqueeze_1868 = None
    unsqueeze_1870: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1869, 3);  unsqueeze_1869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_464: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_66);  relu_66 = None
    alias_465: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_464);  alias_464 = None
    le_99: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_465, 0);  alias_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1880: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
    unsqueeze_1881: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1880, 2);  unsqueeze_1880 = None
    unsqueeze_1882: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1881, 3);  unsqueeze_1881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1892: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
    unsqueeze_1893: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1892, 2);  unsqueeze_1892 = None
    unsqueeze_1894: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1893, 3);  unsqueeze_1893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_470: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_64);  relu_64 = None
    alias_471: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_470);  alias_470 = None
    le_101: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_471, 0);  alias_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1904: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
    unsqueeze_1905: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1904, 2);  unsqueeze_1904 = None
    unsqueeze_1906: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1905, 3);  unsqueeze_1905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_473: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_63);  relu_63 = None
    alias_474: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_473);  alias_473 = None
    le_102: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_474, 0);  alias_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1916: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
    unsqueeze_1917: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1916, 2);  unsqueeze_1916 = None
    unsqueeze_1918: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1917, 3);  unsqueeze_1917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_476: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_62);  relu_62 = None
    alias_477: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_476);  alias_476 = None
    le_103: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_477, 0);  alias_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1928: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
    unsqueeze_1929: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1928, 2);  unsqueeze_1928 = None
    unsqueeze_1930: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1929, 3);  unsqueeze_1929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_479: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_61);  relu_61 = None
    alias_480: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_479);  alias_479 = None
    le_104: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_480, 0);  alias_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_1940: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_1941: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1940, 2);  unsqueeze_1940 = None
    unsqueeze_1942: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1941, 3);  unsqueeze_1941 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_1952: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_1953: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1952, 2);  unsqueeze_1952 = None
    unsqueeze_1954: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1953, 3);  unsqueeze_1953 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_485: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_59);  relu_59 = None
    alias_486: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_485);  alias_485 = None
    le_106: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_486, 0);  alias_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1964: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_1965: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1964, 2);  unsqueeze_1964 = None
    unsqueeze_1966: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1965, 3);  unsqueeze_1965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_488: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_58);  relu_58 = None
    alias_489: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_488);  alias_488 = None
    le_107: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_489, 0);  alias_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1976: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_1977: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1976, 2);  unsqueeze_1976 = None
    unsqueeze_1978: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1977, 3);  unsqueeze_1977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_491: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_57);  relu_57 = None
    alias_492: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_491);  alias_491 = None
    le_108: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_492, 0);  alias_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_1988: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_1989: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1988, 2);  unsqueeze_1988 = None
    unsqueeze_1990: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1989, 3);  unsqueeze_1989 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_494: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_56);  relu_56 = None
    alias_495: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_494);  alias_494 = None
    le_109: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_495, 0);  alias_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_2000: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_2001: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2000, 2);  unsqueeze_2000 = None
    unsqueeze_2002: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2001, 3);  unsqueeze_2001 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_2012: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_2013: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2012, 2);  unsqueeze_2012 = None
    unsqueeze_2014: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2013, 3);  unsqueeze_2013 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_500: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_54);  relu_54 = None
    alias_501: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_500);  alias_500 = None
    le_111: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_501, 0);  alias_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2024: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_2025: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2024, 2);  unsqueeze_2024 = None
    unsqueeze_2026: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2025, 3);  unsqueeze_2025 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_503: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_53);  relu_53 = None
    alias_504: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_503);  alias_503 = None
    le_112: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_504, 0);  alias_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2036: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_2037: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2036, 2);  unsqueeze_2036 = None
    unsqueeze_2038: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2037, 3);  unsqueeze_2037 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_506: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_52);  relu_52 = None
    alias_507: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_506);  alias_506 = None
    le_113: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_507, 0);  alias_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2048: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_2049: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2048, 2);  unsqueeze_2048 = None
    unsqueeze_2050: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2049, 3);  unsqueeze_2049 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_509: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_51);  relu_51 = None
    alias_510: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_509);  alias_509 = None
    le_114: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_510, 0);  alias_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_2060: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_2061: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2060, 2);  unsqueeze_2060 = None
    unsqueeze_2062: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2061, 3);  unsqueeze_2061 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_2072: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_2073: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2072, 2);  unsqueeze_2072 = None
    unsqueeze_2074: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2073, 3);  unsqueeze_2073 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_515: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_49);  relu_49 = None
    alias_516: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_515);  alias_515 = None
    le_116: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_516, 0);  alias_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2084: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_2085: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2084, 2);  unsqueeze_2084 = None
    unsqueeze_2086: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2085, 3);  unsqueeze_2085 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_518: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_48);  relu_48 = None
    alias_519: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_518);  alias_518 = None
    le_117: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_519, 0);  alias_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2096: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_2097: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2096, 2);  unsqueeze_2096 = None
    unsqueeze_2098: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2097, 3);  unsqueeze_2097 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_521: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_47);  relu_47 = None
    alias_522: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_521);  alias_521 = None
    le_118: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_522, 0);  alias_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2108: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_2109: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2108, 2);  unsqueeze_2108 = None
    unsqueeze_2110: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2109, 3);  unsqueeze_2109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_524: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_46);  relu_46 = None
    alias_525: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_524);  alias_524 = None
    le_119: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_525, 0);  alias_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_2120: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_2121: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2120, 2);  unsqueeze_2120 = None
    unsqueeze_2122: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2121, 3);  unsqueeze_2121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_2132: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_2133: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2132, 2);  unsqueeze_2132 = None
    unsqueeze_2134: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2133, 3);  unsqueeze_2133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_530: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_44);  relu_44 = None
    alias_531: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_530);  alias_530 = None
    le_121: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_531, 0);  alias_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2144: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_2145: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2144, 2);  unsqueeze_2144 = None
    unsqueeze_2146: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2145, 3);  unsqueeze_2145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_533: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_43);  relu_43 = None
    alias_534: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_533);  alias_533 = None
    le_122: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_534, 0);  alias_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2156: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_2157: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2156, 2);  unsqueeze_2156 = None
    unsqueeze_2158: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2157, 3);  unsqueeze_2157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_536: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
    alias_537: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_536);  alias_536 = None
    le_123: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_537, 0);  alias_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2168: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_2169: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2168, 2);  unsqueeze_2168 = None
    unsqueeze_2170: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2169, 3);  unsqueeze_2169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_539: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_540: "f32[8, 416, 14, 14]" = torch.ops.aten.alias.default(alias_539);  alias_539 = None
    le_124: "b8[8, 416, 14, 14]" = torch.ops.aten.le.Scalar(alias_540, 0);  alias_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_2180: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_2181: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2180, 2);  unsqueeze_2180 = None
    unsqueeze_2182: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2181, 3);  unsqueeze_2181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    unsqueeze_2192: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_2193: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2192, 2);  unsqueeze_2192 = None
    unsqueeze_2194: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2193, 3);  unsqueeze_2193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_2204: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_2205: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2204, 2);  unsqueeze_2204 = None
    unsqueeze_2206: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2205, 3);  unsqueeze_2205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_545: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_546: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_545);  alias_545 = None
    le_126: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_546, 0);  alias_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2216: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_2217: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2216, 2);  unsqueeze_2216 = None
    unsqueeze_2218: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2217, 3);  unsqueeze_2217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_548: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_549: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_548);  alias_548 = None
    le_127: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_549, 0);  alias_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2228: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_2229: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2228, 2);  unsqueeze_2228 = None
    unsqueeze_2230: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2229, 3);  unsqueeze_2229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_551: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_552: "f32[8, 104, 14, 14]" = torch.ops.aten.alias.default(alias_551);  alias_551 = None
    le_128: "b8[8, 104, 14, 14]" = torch.ops.aten.le.Scalar(alias_552, 0);  alias_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2240: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_2241: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2240, 2);  unsqueeze_2240 = None
    unsqueeze_2242: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2241, 3);  unsqueeze_2241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_554: "f32[8, 416, 28, 28]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_555: "f32[8, 416, 28, 28]" = torch.ops.aten.alias.default(alias_554);  alias_554 = None
    le_129: "b8[8, 416, 28, 28]" = torch.ops.aten.le.Scalar(alias_555, 0);  alias_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_2252: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_2253: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2252, 2);  unsqueeze_2252 = None
    unsqueeze_2254: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2253, 3);  unsqueeze_2253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_2264: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_2265: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2264, 2);  unsqueeze_2264 = None
    unsqueeze_2266: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2265, 3);  unsqueeze_2265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_560: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_561: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(alias_560);  alias_560 = None
    le_131: "b8[8, 52, 28, 28]" = torch.ops.aten.le.Scalar(alias_561, 0);  alias_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2276: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_2277: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2276, 2);  unsqueeze_2276 = None
    unsqueeze_2278: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2277, 3);  unsqueeze_2277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_563: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_564: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(alias_563);  alias_563 = None
    le_132: "b8[8, 52, 28, 28]" = torch.ops.aten.le.Scalar(alias_564, 0);  alias_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2288: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_2289: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2288, 2);  unsqueeze_2288 = None
    unsqueeze_2290: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2289, 3);  unsqueeze_2289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_566: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_567: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(alias_566);  alias_566 = None
    le_133: "b8[8, 52, 28, 28]" = torch.ops.aten.le.Scalar(alias_567, 0);  alias_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2300: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_2301: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2300, 2);  unsqueeze_2300 = None
    unsqueeze_2302: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2301, 3);  unsqueeze_2301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_569: "f32[8, 208, 28, 28]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_570: "f32[8, 208, 28, 28]" = torch.ops.aten.alias.default(alias_569);  alias_569 = None
    le_134: "b8[8, 208, 28, 28]" = torch.ops.aten.le.Scalar(alias_570, 0);  alias_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_2312: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_2313: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2312, 2);  unsqueeze_2312 = None
    unsqueeze_2314: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2313, 3);  unsqueeze_2313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_2324: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_2325: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2324, 2);  unsqueeze_2324 = None
    unsqueeze_2326: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2325, 3);  unsqueeze_2325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_575: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_576: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(alias_575);  alias_575 = None
    le_136: "b8[8, 52, 28, 28]" = torch.ops.aten.le.Scalar(alias_576, 0);  alias_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2336: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_2337: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2336, 2);  unsqueeze_2336 = None
    unsqueeze_2338: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2337, 3);  unsqueeze_2337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_578: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_579: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(alias_578);  alias_578 = None
    le_137: "b8[8, 52, 28, 28]" = torch.ops.aten.le.Scalar(alias_579, 0);  alias_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2348: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_2349: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2348, 2);  unsqueeze_2348 = None
    unsqueeze_2350: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2349, 3);  unsqueeze_2349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_581: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_582: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(alias_581);  alias_581 = None
    le_138: "b8[8, 52, 28, 28]" = torch.ops.aten.le.Scalar(alias_582, 0);  alias_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2360: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_2361: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2360, 2);  unsqueeze_2360 = None
    unsqueeze_2362: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2361, 3);  unsqueeze_2361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_584: "f32[8, 208, 28, 28]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_585: "f32[8, 208, 28, 28]" = torch.ops.aten.alias.default(alias_584);  alias_584 = None
    le_139: "b8[8, 208, 28, 28]" = torch.ops.aten.le.Scalar(alias_585, 0);  alias_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_2372: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_2373: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2372, 2);  unsqueeze_2372 = None
    unsqueeze_2374: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2373, 3);  unsqueeze_2373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_2384: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_2385: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2384, 2);  unsqueeze_2384 = None
    unsqueeze_2386: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2385, 3);  unsqueeze_2385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_590: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_591: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(alias_590);  alias_590 = None
    le_141: "b8[8, 52, 28, 28]" = torch.ops.aten.le.Scalar(alias_591, 0);  alias_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2396: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_2397: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2396, 2);  unsqueeze_2396 = None
    unsqueeze_2398: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2397, 3);  unsqueeze_2397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_593: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_594: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(alias_593);  alias_593 = None
    le_142: "b8[8, 52, 28, 28]" = torch.ops.aten.le.Scalar(alias_594, 0);  alias_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2408: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_2409: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2408, 2);  unsqueeze_2408 = None
    unsqueeze_2410: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2409, 3);  unsqueeze_2409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_596: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_597: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(alias_596);  alias_596 = None
    le_143: "b8[8, 52, 28, 28]" = torch.ops.aten.le.Scalar(alias_597, 0);  alias_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2420: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_2421: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2420, 2);  unsqueeze_2420 = None
    unsqueeze_2422: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2421, 3);  unsqueeze_2421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_599: "f32[8, 208, 28, 28]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_600: "f32[8, 208, 28, 28]" = torch.ops.aten.alias.default(alias_599);  alias_599 = None
    le_144: "b8[8, 208, 28, 28]" = torch.ops.aten.le.Scalar(alias_600, 0);  alias_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_2432: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_2433: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2432, 2);  unsqueeze_2432 = None
    unsqueeze_2434: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2433, 3);  unsqueeze_2433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    unsqueeze_2444: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_2445: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2444, 2);  unsqueeze_2444 = None
    unsqueeze_2446: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2445, 3);  unsqueeze_2445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_2456: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_2457: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2456, 2);  unsqueeze_2456 = None
    unsqueeze_2458: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2457, 3);  unsqueeze_2457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_605: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_606: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(alias_605);  alias_605 = None
    le_146: "b8[8, 52, 28, 28]" = torch.ops.aten.le.Scalar(alias_606, 0);  alias_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2468: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_2469: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2468, 2);  unsqueeze_2468 = None
    unsqueeze_2470: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2469, 3);  unsqueeze_2469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_608: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_609: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(alias_608);  alias_608 = None
    le_147: "b8[8, 52, 28, 28]" = torch.ops.aten.le.Scalar(alias_609, 0);  alias_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2480: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_2481: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2480, 2);  unsqueeze_2480 = None
    unsqueeze_2482: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2481, 3);  unsqueeze_2481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_611: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_612: "f32[8, 52, 28, 28]" = torch.ops.aten.alias.default(alias_611);  alias_611 = None
    le_148: "b8[8, 52, 28, 28]" = torch.ops.aten.le.Scalar(alias_612, 0);  alias_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2492: "f32[1, 52]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_2493: "f32[1, 52, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2492, 2);  unsqueeze_2492 = None
    unsqueeze_2494: "f32[1, 52, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2493, 3);  unsqueeze_2493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_614: "f32[8, 208, 56, 56]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_615: "f32[8, 208, 56, 56]" = torch.ops.aten.alias.default(alias_614);  alias_614 = None
    le_149: "b8[8, 208, 56, 56]" = torch.ops.aten.le.Scalar(alias_615, 0);  alias_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_2504: "f32[1, 208]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_2505: "f32[1, 208, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2504, 2);  unsqueeze_2504 = None
    unsqueeze_2506: "f32[1, 208, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2505, 3);  unsqueeze_2505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_2516: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_2517: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2516, 2);  unsqueeze_2516 = None
    unsqueeze_2518: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2517, 3);  unsqueeze_2517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_620: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_621: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(alias_620);  alias_620 = None
    le_151: "b8[8, 26, 56, 56]" = torch.ops.aten.le.Scalar(alias_621, 0);  alias_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2528: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_2529: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2528, 2);  unsqueeze_2528 = None
    unsqueeze_2530: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2529, 3);  unsqueeze_2529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_623: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_624: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(alias_623);  alias_623 = None
    le_152: "b8[8, 26, 56, 56]" = torch.ops.aten.le.Scalar(alias_624, 0);  alias_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2540: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_2541: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2540, 2);  unsqueeze_2540 = None
    unsqueeze_2542: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2541, 3);  unsqueeze_2541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_626: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_627: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(alias_626);  alias_626 = None
    le_153: "b8[8, 26, 56, 56]" = torch.ops.aten.le.Scalar(alias_627, 0);  alias_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2552: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_2553: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2552, 2);  unsqueeze_2552 = None
    unsqueeze_2554: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2553, 3);  unsqueeze_2553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_629: "f32[8, 104, 56, 56]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_630: "f32[8, 104, 56, 56]" = torch.ops.aten.alias.default(alias_629);  alias_629 = None
    le_154: "b8[8, 104, 56, 56]" = torch.ops.aten.le.Scalar(alias_630, 0);  alias_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_2564: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_2565: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2564, 2);  unsqueeze_2564 = None
    unsqueeze_2566: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2565, 3);  unsqueeze_2565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_2576: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_2577: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2576, 2);  unsqueeze_2576 = None
    unsqueeze_2578: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2577, 3);  unsqueeze_2577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_635: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_636: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(alias_635);  alias_635 = None
    le_156: "b8[8, 26, 56, 56]" = torch.ops.aten.le.Scalar(alias_636, 0);  alias_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2588: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_2589: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2588, 2);  unsqueeze_2588 = None
    unsqueeze_2590: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2589, 3);  unsqueeze_2589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_638: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_639: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(alias_638);  alias_638 = None
    le_157: "b8[8, 26, 56, 56]" = torch.ops.aten.le.Scalar(alias_639, 0);  alias_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2600: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_2601: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2600, 2);  unsqueeze_2600 = None
    unsqueeze_2602: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2601, 3);  unsqueeze_2601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_641: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_642: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(alias_641);  alias_641 = None
    le_158: "b8[8, 26, 56, 56]" = torch.ops.aten.le.Scalar(alias_642, 0);  alias_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2612: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_2613: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2612, 2);  unsqueeze_2612 = None
    unsqueeze_2614: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2613, 3);  unsqueeze_2613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_644: "f32[8, 104, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_645: "f32[8, 104, 56, 56]" = torch.ops.aten.alias.default(alias_644);  alias_644 = None
    le_159: "b8[8, 104, 56, 56]" = torch.ops.aten.le.Scalar(alias_645, 0);  alias_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_2624: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_2625: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2624, 2);  unsqueeze_2624 = None
    unsqueeze_2626: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2625, 3);  unsqueeze_2625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:111, code: shortcut = self.downsample(x)
    unsqueeze_2636: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_2637: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2636, 2);  unsqueeze_2636 = None
    unsqueeze_2638: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2637, 3);  unsqueeze_2637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:105, code: out = self.bn3(out)
    unsqueeze_2648: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_2649: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2648, 2);  unsqueeze_2648 = None
    unsqueeze_2650: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2649, 3);  unsqueeze_2649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_650: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_651: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(alias_650);  alias_650 = None
    le_161: "b8[8, 26, 56, 56]" = torch.ops.aten.le.Scalar(alias_651, 0);  alias_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2660: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_2661: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2660, 2);  unsqueeze_2660 = None
    unsqueeze_2662: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2661, 3);  unsqueeze_2661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_653: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_654: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(alias_653);  alias_653 = None
    le_162: "b8[8, 26, 56, 56]" = torch.ops.aten.le.Scalar(alias_654, 0);  alias_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2672: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_2673: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2672, 2);  unsqueeze_2672 = None
    unsqueeze_2674: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2673, 3);  unsqueeze_2673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:95, code: sp = self.relu(sp)
    alias_656: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_657: "f32[8, 26, 56, 56]" = torch.ops.aten.alias.default(alias_656);  alias_656 = None
    le_163: "b8[8, 26, 56, 56]" = torch.ops.aten.le.Scalar(alias_657, 0);  alias_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:94, code: sp = bn(sp)
    unsqueeze_2684: "f32[1, 26]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_2685: "f32[1, 26, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2684, 2);  unsqueeze_2684 = None
    unsqueeze_2686: "f32[1, 26, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2685, 3);  unsqueeze_2685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:83, code: out = self.relu(out)
    alias_659: "f32[8, 104, 56, 56]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_660: "f32[8, 104, 56, 56]" = torch.ops.aten.alias.default(alias_659);  alias_659 = None
    le_164: "b8[8, 104, 56, 56]" = torch.ops.aten.le.Scalar(alias_660, 0);  alias_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/res2net.py:82, code: out = self.bn1(out)
    unsqueeze_2696: "f32[1, 104]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_2697: "f32[1, 104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2696, 2);  unsqueeze_2696 = None
    unsqueeze_2698: "f32[1, 104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2697, 3);  unsqueeze_2697 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    unsqueeze_2708: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_2709: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2708, 2);  unsqueeze_2708 = None
    unsqueeze_2710: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2709, 3);  unsqueeze_2709 = None
    
    # No stacktrace found for following nodes
    copy_: "f32[64]" = torch.ops.aten.copy_.default(primals_513, add_2);  primals_513 = add_2 = None
    copy__1: "f32[64]" = torch.ops.aten.copy_.default(primals_514, add_3);  primals_514 = add_3 = None
    copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_515, add);  primals_515 = add = None
    copy__3: "f32[104]" = torch.ops.aten.copy_.default(primals_516, add_7);  primals_516 = add_7 = None
    copy__4: "f32[104]" = torch.ops.aten.copy_.default(primals_517, add_8);  primals_517 = add_8 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_518, add_5);  primals_518 = add_5 = None
    copy__6: "f32[26]" = torch.ops.aten.copy_.default(primals_519, add_12);  primals_519 = add_12 = None
    copy__7: "f32[26]" = torch.ops.aten.copy_.default(primals_520, add_13);  primals_520 = add_13 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_521, add_10);  primals_521 = add_10 = None
    copy__9: "f32[26]" = torch.ops.aten.copy_.default(primals_522, add_17);  primals_522 = add_17 = None
    copy__10: "f32[26]" = torch.ops.aten.copy_.default(primals_523, add_18);  primals_523 = add_18 = None
    copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_524, add_15);  primals_524 = add_15 = None
    copy__12: "f32[26]" = torch.ops.aten.copy_.default(primals_525, add_22);  primals_525 = add_22 = None
    copy__13: "f32[26]" = torch.ops.aten.copy_.default(primals_526, add_23);  primals_526 = add_23 = None
    copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_527, add_20);  primals_527 = add_20 = None
    copy__15: "f32[256]" = torch.ops.aten.copy_.default(primals_528, add_27);  primals_528 = add_27 = None
    copy__16: "f32[256]" = torch.ops.aten.copy_.default(primals_529, add_28);  primals_529 = add_28 = None
    copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_530, add_25);  primals_530 = add_25 = None
    copy__18: "f32[256]" = torch.ops.aten.copy_.default(primals_531, add_32);  primals_531 = add_32 = None
    copy__19: "f32[256]" = torch.ops.aten.copy_.default(primals_532, add_33);  primals_532 = add_33 = None
    copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_533, add_30);  primals_533 = add_30 = None
    copy__21: "f32[104]" = torch.ops.aten.copy_.default(primals_534, add_38);  primals_534 = add_38 = None
    copy__22: "f32[104]" = torch.ops.aten.copy_.default(primals_535, add_39);  primals_535 = add_39 = None
    copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_536, add_36);  primals_536 = add_36 = None
    copy__24: "f32[26]" = torch.ops.aten.copy_.default(primals_537, add_43);  primals_537 = add_43 = None
    copy__25: "f32[26]" = torch.ops.aten.copy_.default(primals_538, add_44);  primals_538 = add_44 = None
    copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_539, add_41);  primals_539 = add_41 = None
    copy__27: "f32[26]" = torch.ops.aten.copy_.default(primals_540, add_49);  primals_540 = add_49 = None
    copy__28: "f32[26]" = torch.ops.aten.copy_.default(primals_541, add_50);  primals_541 = add_50 = None
    copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_542, add_47);  primals_542 = add_47 = None
    copy__30: "f32[26]" = torch.ops.aten.copy_.default(primals_543, add_55);  primals_543 = add_55 = None
    copy__31: "f32[26]" = torch.ops.aten.copy_.default(primals_544, add_56);  primals_544 = add_56 = None
    copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_545, add_53);  primals_545 = add_53 = None
    copy__33: "f32[256]" = torch.ops.aten.copy_.default(primals_546, add_60);  primals_546 = add_60 = None
    copy__34: "f32[256]" = torch.ops.aten.copy_.default(primals_547, add_61);  primals_547 = add_61 = None
    copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_548, add_58);  primals_548 = add_58 = None
    copy__36: "f32[104]" = torch.ops.aten.copy_.default(primals_549, add_66);  primals_549 = add_66 = None
    copy__37: "f32[104]" = torch.ops.aten.copy_.default(primals_550, add_67);  primals_550 = add_67 = None
    copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_551, add_64);  primals_551 = add_64 = None
    copy__39: "f32[26]" = torch.ops.aten.copy_.default(primals_552, add_71);  primals_552 = add_71 = None
    copy__40: "f32[26]" = torch.ops.aten.copy_.default(primals_553, add_72);  primals_553 = add_72 = None
    copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_554, add_69);  primals_554 = add_69 = None
    copy__42: "f32[26]" = torch.ops.aten.copy_.default(primals_555, add_77);  primals_555 = add_77 = None
    copy__43: "f32[26]" = torch.ops.aten.copy_.default(primals_556, add_78);  primals_556 = add_78 = None
    copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_557, add_75);  primals_557 = add_75 = None
    copy__45: "f32[26]" = torch.ops.aten.copy_.default(primals_558, add_83);  primals_558 = add_83 = None
    copy__46: "f32[26]" = torch.ops.aten.copy_.default(primals_559, add_84);  primals_559 = add_84 = None
    copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_560, add_81);  primals_560 = add_81 = None
    copy__48: "f32[256]" = torch.ops.aten.copy_.default(primals_561, add_88);  primals_561 = add_88 = None
    copy__49: "f32[256]" = torch.ops.aten.copy_.default(primals_562, add_89);  primals_562 = add_89 = None
    copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_563, add_86);  primals_563 = add_86 = None
    copy__51: "f32[208]" = torch.ops.aten.copy_.default(primals_564, add_94);  primals_564 = add_94 = None
    copy__52: "f32[208]" = torch.ops.aten.copy_.default(primals_565, add_95);  primals_565 = add_95 = None
    copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_566, add_92);  primals_566 = add_92 = None
    copy__54: "f32[52]" = torch.ops.aten.copy_.default(primals_567, add_99);  primals_567 = add_99 = None
    copy__55: "f32[52]" = torch.ops.aten.copy_.default(primals_568, add_100);  primals_568 = add_100 = None
    copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_569, add_97);  primals_569 = add_97 = None
    copy__57: "f32[52]" = torch.ops.aten.copy_.default(primals_570, add_104);  primals_570 = add_104 = None
    copy__58: "f32[52]" = torch.ops.aten.copy_.default(primals_571, add_105);  primals_571 = add_105 = None
    copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_572, add_102);  primals_572 = add_102 = None
    copy__60: "f32[52]" = torch.ops.aten.copy_.default(primals_573, add_109);  primals_573 = add_109 = None
    copy__61: "f32[52]" = torch.ops.aten.copy_.default(primals_574, add_110);  primals_574 = add_110 = None
    copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_575, add_107);  primals_575 = add_107 = None
    copy__63: "f32[512]" = torch.ops.aten.copy_.default(primals_576, add_114);  primals_576 = add_114 = None
    copy__64: "f32[512]" = torch.ops.aten.copy_.default(primals_577, add_115);  primals_577 = add_115 = None
    copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_578, add_112);  primals_578 = add_112 = None
    copy__66: "f32[512]" = torch.ops.aten.copy_.default(primals_579, add_119);  primals_579 = add_119 = None
    copy__67: "f32[512]" = torch.ops.aten.copy_.default(primals_580, add_120);  primals_580 = add_120 = None
    copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_581, add_117);  primals_581 = add_117 = None
    copy__69: "f32[208]" = torch.ops.aten.copy_.default(primals_582, add_125);  primals_582 = add_125 = None
    copy__70: "f32[208]" = torch.ops.aten.copy_.default(primals_583, add_126);  primals_583 = add_126 = None
    copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_584, add_123);  primals_584 = add_123 = None
    copy__72: "f32[52]" = torch.ops.aten.copy_.default(primals_585, add_130);  primals_585 = add_130 = None
    copy__73: "f32[52]" = torch.ops.aten.copy_.default(primals_586, add_131);  primals_586 = add_131 = None
    copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_587, add_128);  primals_587 = add_128 = None
    copy__75: "f32[52]" = torch.ops.aten.copy_.default(primals_588, add_136);  primals_588 = add_136 = None
    copy__76: "f32[52]" = torch.ops.aten.copy_.default(primals_589, add_137);  primals_589 = add_137 = None
    copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_590, add_134);  primals_590 = add_134 = None
    copy__78: "f32[52]" = torch.ops.aten.copy_.default(primals_591, add_142);  primals_591 = add_142 = None
    copy__79: "f32[52]" = torch.ops.aten.copy_.default(primals_592, add_143);  primals_592 = add_143 = None
    copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_593, add_140);  primals_593 = add_140 = None
    copy__81: "f32[512]" = torch.ops.aten.copy_.default(primals_594, add_147);  primals_594 = add_147 = None
    copy__82: "f32[512]" = torch.ops.aten.copy_.default(primals_595, add_148);  primals_595 = add_148 = None
    copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_596, add_145);  primals_596 = add_145 = None
    copy__84: "f32[208]" = torch.ops.aten.copy_.default(primals_597, add_153);  primals_597 = add_153 = None
    copy__85: "f32[208]" = torch.ops.aten.copy_.default(primals_598, add_154);  primals_598 = add_154 = None
    copy__86: "i64[]" = torch.ops.aten.copy_.default(primals_599, add_151);  primals_599 = add_151 = None
    copy__87: "f32[52]" = torch.ops.aten.copy_.default(primals_600, add_158);  primals_600 = add_158 = None
    copy__88: "f32[52]" = torch.ops.aten.copy_.default(primals_601, add_159);  primals_601 = add_159 = None
    copy__89: "i64[]" = torch.ops.aten.copy_.default(primals_602, add_156);  primals_602 = add_156 = None
    copy__90: "f32[52]" = torch.ops.aten.copy_.default(primals_603, add_164);  primals_603 = add_164 = None
    copy__91: "f32[52]" = torch.ops.aten.copy_.default(primals_604, add_165);  primals_604 = add_165 = None
    copy__92: "i64[]" = torch.ops.aten.copy_.default(primals_605, add_162);  primals_605 = add_162 = None
    copy__93: "f32[52]" = torch.ops.aten.copy_.default(primals_606, add_170);  primals_606 = add_170 = None
    copy__94: "f32[52]" = torch.ops.aten.copy_.default(primals_607, add_171);  primals_607 = add_171 = None
    copy__95: "i64[]" = torch.ops.aten.copy_.default(primals_608, add_168);  primals_608 = add_168 = None
    copy__96: "f32[512]" = torch.ops.aten.copy_.default(primals_609, add_175);  primals_609 = add_175 = None
    copy__97: "f32[512]" = torch.ops.aten.copy_.default(primals_610, add_176);  primals_610 = add_176 = None
    copy__98: "i64[]" = torch.ops.aten.copy_.default(primals_611, add_173);  primals_611 = add_173 = None
    copy__99: "f32[208]" = torch.ops.aten.copy_.default(primals_612, add_181);  primals_612 = add_181 = None
    copy__100: "f32[208]" = torch.ops.aten.copy_.default(primals_613, add_182);  primals_613 = add_182 = None
    copy__101: "i64[]" = torch.ops.aten.copy_.default(primals_614, add_179);  primals_614 = add_179 = None
    copy__102: "f32[52]" = torch.ops.aten.copy_.default(primals_615, add_186);  primals_615 = add_186 = None
    copy__103: "f32[52]" = torch.ops.aten.copy_.default(primals_616, add_187);  primals_616 = add_187 = None
    copy__104: "i64[]" = torch.ops.aten.copy_.default(primals_617, add_184);  primals_617 = add_184 = None
    copy__105: "f32[52]" = torch.ops.aten.copy_.default(primals_618, add_192);  primals_618 = add_192 = None
    copy__106: "f32[52]" = torch.ops.aten.copy_.default(primals_619, add_193);  primals_619 = add_193 = None
    copy__107: "i64[]" = torch.ops.aten.copy_.default(primals_620, add_190);  primals_620 = add_190 = None
    copy__108: "f32[52]" = torch.ops.aten.copy_.default(primals_621, add_198);  primals_621 = add_198 = None
    copy__109: "f32[52]" = torch.ops.aten.copy_.default(primals_622, add_199);  primals_622 = add_199 = None
    copy__110: "i64[]" = torch.ops.aten.copy_.default(primals_623, add_196);  primals_623 = add_196 = None
    copy__111: "f32[512]" = torch.ops.aten.copy_.default(primals_624, add_203);  primals_624 = add_203 = None
    copy__112: "f32[512]" = torch.ops.aten.copy_.default(primals_625, add_204);  primals_625 = add_204 = None
    copy__113: "i64[]" = torch.ops.aten.copy_.default(primals_626, add_201);  primals_626 = add_201 = None
    copy__114: "f32[416]" = torch.ops.aten.copy_.default(primals_627, add_209);  primals_627 = add_209 = None
    copy__115: "f32[416]" = torch.ops.aten.copy_.default(primals_628, add_210);  primals_628 = add_210 = None
    copy__116: "i64[]" = torch.ops.aten.copy_.default(primals_629, add_207);  primals_629 = add_207 = None
    copy__117: "f32[104]" = torch.ops.aten.copy_.default(primals_630, add_214);  primals_630 = add_214 = None
    copy__118: "f32[104]" = torch.ops.aten.copy_.default(primals_631, add_215);  primals_631 = add_215 = None
    copy__119: "i64[]" = torch.ops.aten.copy_.default(primals_632, add_212);  primals_632 = add_212 = None
    copy__120: "f32[104]" = torch.ops.aten.copy_.default(primals_633, add_219);  primals_633 = add_219 = None
    copy__121: "f32[104]" = torch.ops.aten.copy_.default(primals_634, add_220);  primals_634 = add_220 = None
    copy__122: "i64[]" = torch.ops.aten.copy_.default(primals_635, add_217);  primals_635 = add_217 = None
    copy__123: "f32[104]" = torch.ops.aten.copy_.default(primals_636, add_224);  primals_636 = add_224 = None
    copy__124: "f32[104]" = torch.ops.aten.copy_.default(primals_637, add_225);  primals_637 = add_225 = None
    copy__125: "i64[]" = torch.ops.aten.copy_.default(primals_638, add_222);  primals_638 = add_222 = None
    copy__126: "f32[1024]" = torch.ops.aten.copy_.default(primals_639, add_229);  primals_639 = add_229 = None
    copy__127: "f32[1024]" = torch.ops.aten.copy_.default(primals_640, add_230);  primals_640 = add_230 = None
    copy__128: "i64[]" = torch.ops.aten.copy_.default(primals_641, add_227);  primals_641 = add_227 = None
    copy__129: "f32[1024]" = torch.ops.aten.copy_.default(primals_642, add_234);  primals_642 = add_234 = None
    copy__130: "f32[1024]" = torch.ops.aten.copy_.default(primals_643, add_235);  primals_643 = add_235 = None
    copy__131: "i64[]" = torch.ops.aten.copy_.default(primals_644, add_232);  primals_644 = add_232 = None
    copy__132: "f32[416]" = torch.ops.aten.copy_.default(primals_645, add_240);  primals_645 = add_240 = None
    copy__133: "f32[416]" = torch.ops.aten.copy_.default(primals_646, add_241);  primals_646 = add_241 = None
    copy__134: "i64[]" = torch.ops.aten.copy_.default(primals_647, add_238);  primals_647 = add_238 = None
    copy__135: "f32[104]" = torch.ops.aten.copy_.default(primals_648, add_245);  primals_648 = add_245 = None
    copy__136: "f32[104]" = torch.ops.aten.copy_.default(primals_649, add_246);  primals_649 = add_246 = None
    copy__137: "i64[]" = torch.ops.aten.copy_.default(primals_650, add_243);  primals_650 = add_243 = None
    copy__138: "f32[104]" = torch.ops.aten.copy_.default(primals_651, add_251);  primals_651 = add_251 = None
    copy__139: "f32[104]" = torch.ops.aten.copy_.default(primals_652, add_252);  primals_652 = add_252 = None
    copy__140: "i64[]" = torch.ops.aten.copy_.default(primals_653, add_249);  primals_653 = add_249 = None
    copy__141: "f32[104]" = torch.ops.aten.copy_.default(primals_654, add_257);  primals_654 = add_257 = None
    copy__142: "f32[104]" = torch.ops.aten.copy_.default(primals_655, add_258);  primals_655 = add_258 = None
    copy__143: "i64[]" = torch.ops.aten.copy_.default(primals_656, add_255);  primals_656 = add_255 = None
    copy__144: "f32[1024]" = torch.ops.aten.copy_.default(primals_657, add_262);  primals_657 = add_262 = None
    copy__145: "f32[1024]" = torch.ops.aten.copy_.default(primals_658, add_263);  primals_658 = add_263 = None
    copy__146: "i64[]" = torch.ops.aten.copy_.default(primals_659, add_260);  primals_659 = add_260 = None
    copy__147: "f32[416]" = torch.ops.aten.copy_.default(primals_660, add_268);  primals_660 = add_268 = None
    copy__148: "f32[416]" = torch.ops.aten.copy_.default(primals_661, add_269);  primals_661 = add_269 = None
    copy__149: "i64[]" = torch.ops.aten.copy_.default(primals_662, add_266);  primals_662 = add_266 = None
    copy__150: "f32[104]" = torch.ops.aten.copy_.default(primals_663, add_273);  primals_663 = add_273 = None
    copy__151: "f32[104]" = torch.ops.aten.copy_.default(primals_664, add_274);  primals_664 = add_274 = None
    copy__152: "i64[]" = torch.ops.aten.copy_.default(primals_665, add_271);  primals_665 = add_271 = None
    copy__153: "f32[104]" = torch.ops.aten.copy_.default(primals_666, add_279);  primals_666 = add_279 = None
    copy__154: "f32[104]" = torch.ops.aten.copy_.default(primals_667, add_280);  primals_667 = add_280 = None
    copy__155: "i64[]" = torch.ops.aten.copy_.default(primals_668, add_277);  primals_668 = add_277 = None
    copy__156: "f32[104]" = torch.ops.aten.copy_.default(primals_669, add_285);  primals_669 = add_285 = None
    copy__157: "f32[104]" = torch.ops.aten.copy_.default(primals_670, add_286);  primals_670 = add_286 = None
    copy__158: "i64[]" = torch.ops.aten.copy_.default(primals_671, add_283);  primals_671 = add_283 = None
    copy__159: "f32[1024]" = torch.ops.aten.copy_.default(primals_672, add_290);  primals_672 = add_290 = None
    copy__160: "f32[1024]" = torch.ops.aten.copy_.default(primals_673, add_291);  primals_673 = add_291 = None
    copy__161: "i64[]" = torch.ops.aten.copy_.default(primals_674, add_288);  primals_674 = add_288 = None
    copy__162: "f32[416]" = torch.ops.aten.copy_.default(primals_675, add_296);  primals_675 = add_296 = None
    copy__163: "f32[416]" = torch.ops.aten.copy_.default(primals_676, add_297);  primals_676 = add_297 = None
    copy__164: "i64[]" = torch.ops.aten.copy_.default(primals_677, add_294);  primals_677 = add_294 = None
    copy__165: "f32[104]" = torch.ops.aten.copy_.default(primals_678, add_301);  primals_678 = add_301 = None
    copy__166: "f32[104]" = torch.ops.aten.copy_.default(primals_679, add_302);  primals_679 = add_302 = None
    copy__167: "i64[]" = torch.ops.aten.copy_.default(primals_680, add_299);  primals_680 = add_299 = None
    copy__168: "f32[104]" = torch.ops.aten.copy_.default(primals_681, add_307);  primals_681 = add_307 = None
    copy__169: "f32[104]" = torch.ops.aten.copy_.default(primals_682, add_308);  primals_682 = add_308 = None
    copy__170: "i64[]" = torch.ops.aten.copy_.default(primals_683, add_305);  primals_683 = add_305 = None
    copy__171: "f32[104]" = torch.ops.aten.copy_.default(primals_684, add_313);  primals_684 = add_313 = None
    copy__172: "f32[104]" = torch.ops.aten.copy_.default(primals_685, add_314);  primals_685 = add_314 = None
    copy__173: "i64[]" = torch.ops.aten.copy_.default(primals_686, add_311);  primals_686 = add_311 = None
    copy__174: "f32[1024]" = torch.ops.aten.copy_.default(primals_687, add_318);  primals_687 = add_318 = None
    copy__175: "f32[1024]" = torch.ops.aten.copy_.default(primals_688, add_319);  primals_688 = add_319 = None
    copy__176: "i64[]" = torch.ops.aten.copy_.default(primals_689, add_316);  primals_689 = add_316 = None
    copy__177: "f32[416]" = torch.ops.aten.copy_.default(primals_690, add_324);  primals_690 = add_324 = None
    copy__178: "f32[416]" = torch.ops.aten.copy_.default(primals_691, add_325);  primals_691 = add_325 = None
    copy__179: "i64[]" = torch.ops.aten.copy_.default(primals_692, add_322);  primals_692 = add_322 = None
    copy__180: "f32[104]" = torch.ops.aten.copy_.default(primals_693, add_329);  primals_693 = add_329 = None
    copy__181: "f32[104]" = torch.ops.aten.copy_.default(primals_694, add_330);  primals_694 = add_330 = None
    copy__182: "i64[]" = torch.ops.aten.copy_.default(primals_695, add_327);  primals_695 = add_327 = None
    copy__183: "f32[104]" = torch.ops.aten.copy_.default(primals_696, add_335);  primals_696 = add_335 = None
    copy__184: "f32[104]" = torch.ops.aten.copy_.default(primals_697, add_336);  primals_697 = add_336 = None
    copy__185: "i64[]" = torch.ops.aten.copy_.default(primals_698, add_333);  primals_698 = add_333 = None
    copy__186: "f32[104]" = torch.ops.aten.copy_.default(primals_699, add_341);  primals_699 = add_341 = None
    copy__187: "f32[104]" = torch.ops.aten.copy_.default(primals_700, add_342);  primals_700 = add_342 = None
    copy__188: "i64[]" = torch.ops.aten.copy_.default(primals_701, add_339);  primals_701 = add_339 = None
    copy__189: "f32[1024]" = torch.ops.aten.copy_.default(primals_702, add_346);  primals_702 = add_346 = None
    copy__190: "f32[1024]" = torch.ops.aten.copy_.default(primals_703, add_347);  primals_703 = add_347 = None
    copy__191: "i64[]" = torch.ops.aten.copy_.default(primals_704, add_344);  primals_704 = add_344 = None
    copy__192: "f32[416]" = torch.ops.aten.copy_.default(primals_705, add_352);  primals_705 = add_352 = None
    copy__193: "f32[416]" = torch.ops.aten.copy_.default(primals_706, add_353);  primals_706 = add_353 = None
    copy__194: "i64[]" = torch.ops.aten.copy_.default(primals_707, add_350);  primals_707 = add_350 = None
    copy__195: "f32[104]" = torch.ops.aten.copy_.default(primals_708, add_357);  primals_708 = add_357 = None
    copy__196: "f32[104]" = torch.ops.aten.copy_.default(primals_709, add_358);  primals_709 = add_358 = None
    copy__197: "i64[]" = torch.ops.aten.copy_.default(primals_710, add_355);  primals_710 = add_355 = None
    copy__198: "f32[104]" = torch.ops.aten.copy_.default(primals_711, add_363);  primals_711 = add_363 = None
    copy__199: "f32[104]" = torch.ops.aten.copy_.default(primals_712, add_364);  primals_712 = add_364 = None
    copy__200: "i64[]" = torch.ops.aten.copy_.default(primals_713, add_361);  primals_713 = add_361 = None
    copy__201: "f32[104]" = torch.ops.aten.copy_.default(primals_714, add_369);  primals_714 = add_369 = None
    copy__202: "f32[104]" = torch.ops.aten.copy_.default(primals_715, add_370);  primals_715 = add_370 = None
    copy__203: "i64[]" = torch.ops.aten.copy_.default(primals_716, add_367);  primals_716 = add_367 = None
    copy__204: "f32[1024]" = torch.ops.aten.copy_.default(primals_717, add_374);  primals_717 = add_374 = None
    copy__205: "f32[1024]" = torch.ops.aten.copy_.default(primals_718, add_375);  primals_718 = add_375 = None
    copy__206: "i64[]" = torch.ops.aten.copy_.default(primals_719, add_372);  primals_719 = add_372 = None
    copy__207: "f32[416]" = torch.ops.aten.copy_.default(primals_720, add_380);  primals_720 = add_380 = None
    copy__208: "f32[416]" = torch.ops.aten.copy_.default(primals_721, add_381);  primals_721 = add_381 = None
    copy__209: "i64[]" = torch.ops.aten.copy_.default(primals_722, add_378);  primals_722 = add_378 = None
    copy__210: "f32[104]" = torch.ops.aten.copy_.default(primals_723, add_385);  primals_723 = add_385 = None
    copy__211: "f32[104]" = torch.ops.aten.copy_.default(primals_724, add_386);  primals_724 = add_386 = None
    copy__212: "i64[]" = torch.ops.aten.copy_.default(primals_725, add_383);  primals_725 = add_383 = None
    copy__213: "f32[104]" = torch.ops.aten.copy_.default(primals_726, add_391);  primals_726 = add_391 = None
    copy__214: "f32[104]" = torch.ops.aten.copy_.default(primals_727, add_392);  primals_727 = add_392 = None
    copy__215: "i64[]" = torch.ops.aten.copy_.default(primals_728, add_389);  primals_728 = add_389 = None
    copy__216: "f32[104]" = torch.ops.aten.copy_.default(primals_729, add_397);  primals_729 = add_397 = None
    copy__217: "f32[104]" = torch.ops.aten.copy_.default(primals_730, add_398);  primals_730 = add_398 = None
    copy__218: "i64[]" = torch.ops.aten.copy_.default(primals_731, add_395);  primals_731 = add_395 = None
    copy__219: "f32[1024]" = torch.ops.aten.copy_.default(primals_732, add_402);  primals_732 = add_402 = None
    copy__220: "f32[1024]" = torch.ops.aten.copy_.default(primals_733, add_403);  primals_733 = add_403 = None
    copy__221: "i64[]" = torch.ops.aten.copy_.default(primals_734, add_400);  primals_734 = add_400 = None
    copy__222: "f32[416]" = torch.ops.aten.copy_.default(primals_735, add_408);  primals_735 = add_408 = None
    copy__223: "f32[416]" = torch.ops.aten.copy_.default(primals_736, add_409);  primals_736 = add_409 = None
    copy__224: "i64[]" = torch.ops.aten.copy_.default(primals_737, add_406);  primals_737 = add_406 = None
    copy__225: "f32[104]" = torch.ops.aten.copy_.default(primals_738, add_413);  primals_738 = add_413 = None
    copy__226: "f32[104]" = torch.ops.aten.copy_.default(primals_739, add_414);  primals_739 = add_414 = None
    copy__227: "i64[]" = torch.ops.aten.copy_.default(primals_740, add_411);  primals_740 = add_411 = None
    copy__228: "f32[104]" = torch.ops.aten.copy_.default(primals_741, add_419);  primals_741 = add_419 = None
    copy__229: "f32[104]" = torch.ops.aten.copy_.default(primals_742, add_420);  primals_742 = add_420 = None
    copy__230: "i64[]" = torch.ops.aten.copy_.default(primals_743, add_417);  primals_743 = add_417 = None
    copy__231: "f32[104]" = torch.ops.aten.copy_.default(primals_744, add_425);  primals_744 = add_425 = None
    copy__232: "f32[104]" = torch.ops.aten.copy_.default(primals_745, add_426);  primals_745 = add_426 = None
    copy__233: "i64[]" = torch.ops.aten.copy_.default(primals_746, add_423);  primals_746 = add_423 = None
    copy__234: "f32[1024]" = torch.ops.aten.copy_.default(primals_747, add_430);  primals_747 = add_430 = None
    copy__235: "f32[1024]" = torch.ops.aten.copy_.default(primals_748, add_431);  primals_748 = add_431 = None
    copy__236: "i64[]" = torch.ops.aten.copy_.default(primals_749, add_428);  primals_749 = add_428 = None
    copy__237: "f32[416]" = torch.ops.aten.copy_.default(primals_750, add_436);  primals_750 = add_436 = None
    copy__238: "f32[416]" = torch.ops.aten.copy_.default(primals_751, add_437);  primals_751 = add_437 = None
    copy__239: "i64[]" = torch.ops.aten.copy_.default(primals_752, add_434);  primals_752 = add_434 = None
    copy__240: "f32[104]" = torch.ops.aten.copy_.default(primals_753, add_441);  primals_753 = add_441 = None
    copy__241: "f32[104]" = torch.ops.aten.copy_.default(primals_754, add_442);  primals_754 = add_442 = None
    copy__242: "i64[]" = torch.ops.aten.copy_.default(primals_755, add_439);  primals_755 = add_439 = None
    copy__243: "f32[104]" = torch.ops.aten.copy_.default(primals_756, add_447);  primals_756 = add_447 = None
    copy__244: "f32[104]" = torch.ops.aten.copy_.default(primals_757, add_448);  primals_757 = add_448 = None
    copy__245: "i64[]" = torch.ops.aten.copy_.default(primals_758, add_445);  primals_758 = add_445 = None
    copy__246: "f32[104]" = torch.ops.aten.copy_.default(primals_759, add_453);  primals_759 = add_453 = None
    copy__247: "f32[104]" = torch.ops.aten.copy_.default(primals_760, add_454);  primals_760 = add_454 = None
    copy__248: "i64[]" = torch.ops.aten.copy_.default(primals_761, add_451);  primals_761 = add_451 = None
    copy__249: "f32[1024]" = torch.ops.aten.copy_.default(primals_762, add_458);  primals_762 = add_458 = None
    copy__250: "f32[1024]" = torch.ops.aten.copy_.default(primals_763, add_459);  primals_763 = add_459 = None
    copy__251: "i64[]" = torch.ops.aten.copy_.default(primals_764, add_456);  primals_764 = add_456 = None
    copy__252: "f32[416]" = torch.ops.aten.copy_.default(primals_765, add_464);  primals_765 = add_464 = None
    copy__253: "f32[416]" = torch.ops.aten.copy_.default(primals_766, add_465);  primals_766 = add_465 = None
    copy__254: "i64[]" = torch.ops.aten.copy_.default(primals_767, add_462);  primals_767 = add_462 = None
    copy__255: "f32[104]" = torch.ops.aten.copy_.default(primals_768, add_469);  primals_768 = add_469 = None
    copy__256: "f32[104]" = torch.ops.aten.copy_.default(primals_769, add_470);  primals_769 = add_470 = None
    copy__257: "i64[]" = torch.ops.aten.copy_.default(primals_770, add_467);  primals_770 = add_467 = None
    copy__258: "f32[104]" = torch.ops.aten.copy_.default(primals_771, add_475);  primals_771 = add_475 = None
    copy__259: "f32[104]" = torch.ops.aten.copy_.default(primals_772, add_476);  primals_772 = add_476 = None
    copy__260: "i64[]" = torch.ops.aten.copy_.default(primals_773, add_473);  primals_773 = add_473 = None
    copy__261: "f32[104]" = torch.ops.aten.copy_.default(primals_774, add_481);  primals_774 = add_481 = None
    copy__262: "f32[104]" = torch.ops.aten.copy_.default(primals_775, add_482);  primals_775 = add_482 = None
    copy__263: "i64[]" = torch.ops.aten.copy_.default(primals_776, add_479);  primals_776 = add_479 = None
    copy__264: "f32[1024]" = torch.ops.aten.copy_.default(primals_777, add_486);  primals_777 = add_486 = None
    copy__265: "f32[1024]" = torch.ops.aten.copy_.default(primals_778, add_487);  primals_778 = add_487 = None
    copy__266: "i64[]" = torch.ops.aten.copy_.default(primals_779, add_484);  primals_779 = add_484 = None
    copy__267: "f32[416]" = torch.ops.aten.copy_.default(primals_780, add_492);  primals_780 = add_492 = None
    copy__268: "f32[416]" = torch.ops.aten.copy_.default(primals_781, add_493);  primals_781 = add_493 = None
    copy__269: "i64[]" = torch.ops.aten.copy_.default(primals_782, add_490);  primals_782 = add_490 = None
    copy__270: "f32[104]" = torch.ops.aten.copy_.default(primals_783, add_497);  primals_783 = add_497 = None
    copy__271: "f32[104]" = torch.ops.aten.copy_.default(primals_784, add_498);  primals_784 = add_498 = None
    copy__272: "i64[]" = torch.ops.aten.copy_.default(primals_785, add_495);  primals_785 = add_495 = None
    copy__273: "f32[104]" = torch.ops.aten.copy_.default(primals_786, add_503);  primals_786 = add_503 = None
    copy__274: "f32[104]" = torch.ops.aten.copy_.default(primals_787, add_504);  primals_787 = add_504 = None
    copy__275: "i64[]" = torch.ops.aten.copy_.default(primals_788, add_501);  primals_788 = add_501 = None
    copy__276: "f32[104]" = torch.ops.aten.copy_.default(primals_789, add_509);  primals_789 = add_509 = None
    copy__277: "f32[104]" = torch.ops.aten.copy_.default(primals_790, add_510);  primals_790 = add_510 = None
    copy__278: "i64[]" = torch.ops.aten.copy_.default(primals_791, add_507);  primals_791 = add_507 = None
    copy__279: "f32[1024]" = torch.ops.aten.copy_.default(primals_792, add_514);  primals_792 = add_514 = None
    copy__280: "f32[1024]" = torch.ops.aten.copy_.default(primals_793, add_515);  primals_793 = add_515 = None
    copy__281: "i64[]" = torch.ops.aten.copy_.default(primals_794, add_512);  primals_794 = add_512 = None
    copy__282: "f32[416]" = torch.ops.aten.copy_.default(primals_795, add_520);  primals_795 = add_520 = None
    copy__283: "f32[416]" = torch.ops.aten.copy_.default(primals_796, add_521);  primals_796 = add_521 = None
    copy__284: "i64[]" = torch.ops.aten.copy_.default(primals_797, add_518);  primals_797 = add_518 = None
    copy__285: "f32[104]" = torch.ops.aten.copy_.default(primals_798, add_525);  primals_798 = add_525 = None
    copy__286: "f32[104]" = torch.ops.aten.copy_.default(primals_799, add_526);  primals_799 = add_526 = None
    copy__287: "i64[]" = torch.ops.aten.copy_.default(primals_800, add_523);  primals_800 = add_523 = None
    copy__288: "f32[104]" = torch.ops.aten.copy_.default(primals_801, add_531);  primals_801 = add_531 = None
    copy__289: "f32[104]" = torch.ops.aten.copy_.default(primals_802, add_532);  primals_802 = add_532 = None
    copy__290: "i64[]" = torch.ops.aten.copy_.default(primals_803, add_529);  primals_803 = add_529 = None
    copy__291: "f32[104]" = torch.ops.aten.copy_.default(primals_804, add_537);  primals_804 = add_537 = None
    copy__292: "f32[104]" = torch.ops.aten.copy_.default(primals_805, add_538);  primals_805 = add_538 = None
    copy__293: "i64[]" = torch.ops.aten.copy_.default(primals_806, add_535);  primals_806 = add_535 = None
    copy__294: "f32[1024]" = torch.ops.aten.copy_.default(primals_807, add_542);  primals_807 = add_542 = None
    copy__295: "f32[1024]" = torch.ops.aten.copy_.default(primals_808, add_543);  primals_808 = add_543 = None
    copy__296: "i64[]" = torch.ops.aten.copy_.default(primals_809, add_540);  primals_809 = add_540 = None
    copy__297: "f32[416]" = torch.ops.aten.copy_.default(primals_810, add_548);  primals_810 = add_548 = None
    copy__298: "f32[416]" = torch.ops.aten.copy_.default(primals_811, add_549);  primals_811 = add_549 = None
    copy__299: "i64[]" = torch.ops.aten.copy_.default(primals_812, add_546);  primals_812 = add_546 = None
    copy__300: "f32[104]" = torch.ops.aten.copy_.default(primals_813, add_553);  primals_813 = add_553 = None
    copy__301: "f32[104]" = torch.ops.aten.copy_.default(primals_814, add_554);  primals_814 = add_554 = None
    copy__302: "i64[]" = torch.ops.aten.copy_.default(primals_815, add_551);  primals_815 = add_551 = None
    copy__303: "f32[104]" = torch.ops.aten.copy_.default(primals_816, add_559);  primals_816 = add_559 = None
    copy__304: "f32[104]" = torch.ops.aten.copy_.default(primals_817, add_560);  primals_817 = add_560 = None
    copy__305: "i64[]" = torch.ops.aten.copy_.default(primals_818, add_557);  primals_818 = add_557 = None
    copy__306: "f32[104]" = torch.ops.aten.copy_.default(primals_819, add_565);  primals_819 = add_565 = None
    copy__307: "f32[104]" = torch.ops.aten.copy_.default(primals_820, add_566);  primals_820 = add_566 = None
    copy__308: "i64[]" = torch.ops.aten.copy_.default(primals_821, add_563);  primals_821 = add_563 = None
    copy__309: "f32[1024]" = torch.ops.aten.copy_.default(primals_822, add_570);  primals_822 = add_570 = None
    copy__310: "f32[1024]" = torch.ops.aten.copy_.default(primals_823, add_571);  primals_823 = add_571 = None
    copy__311: "i64[]" = torch.ops.aten.copy_.default(primals_824, add_568);  primals_824 = add_568 = None
    copy__312: "f32[416]" = torch.ops.aten.copy_.default(primals_825, add_576);  primals_825 = add_576 = None
    copy__313: "f32[416]" = torch.ops.aten.copy_.default(primals_826, add_577);  primals_826 = add_577 = None
    copy__314: "i64[]" = torch.ops.aten.copy_.default(primals_827, add_574);  primals_827 = add_574 = None
    copy__315: "f32[104]" = torch.ops.aten.copy_.default(primals_828, add_581);  primals_828 = add_581 = None
    copy__316: "f32[104]" = torch.ops.aten.copy_.default(primals_829, add_582);  primals_829 = add_582 = None
    copy__317: "i64[]" = torch.ops.aten.copy_.default(primals_830, add_579);  primals_830 = add_579 = None
    copy__318: "f32[104]" = torch.ops.aten.copy_.default(primals_831, add_587);  primals_831 = add_587 = None
    copy__319: "f32[104]" = torch.ops.aten.copy_.default(primals_832, add_588);  primals_832 = add_588 = None
    copy__320: "i64[]" = torch.ops.aten.copy_.default(primals_833, add_585);  primals_833 = add_585 = None
    copy__321: "f32[104]" = torch.ops.aten.copy_.default(primals_834, add_593);  primals_834 = add_593 = None
    copy__322: "f32[104]" = torch.ops.aten.copy_.default(primals_835, add_594);  primals_835 = add_594 = None
    copy__323: "i64[]" = torch.ops.aten.copy_.default(primals_836, add_591);  primals_836 = add_591 = None
    copy__324: "f32[1024]" = torch.ops.aten.copy_.default(primals_837, add_598);  primals_837 = add_598 = None
    copy__325: "f32[1024]" = torch.ops.aten.copy_.default(primals_838, add_599);  primals_838 = add_599 = None
    copy__326: "i64[]" = torch.ops.aten.copy_.default(primals_839, add_596);  primals_839 = add_596 = None
    copy__327: "f32[416]" = torch.ops.aten.copy_.default(primals_840, add_604);  primals_840 = add_604 = None
    copy__328: "f32[416]" = torch.ops.aten.copy_.default(primals_841, add_605);  primals_841 = add_605 = None
    copy__329: "i64[]" = torch.ops.aten.copy_.default(primals_842, add_602);  primals_842 = add_602 = None
    copy__330: "f32[104]" = torch.ops.aten.copy_.default(primals_843, add_609);  primals_843 = add_609 = None
    copy__331: "f32[104]" = torch.ops.aten.copy_.default(primals_844, add_610);  primals_844 = add_610 = None
    copy__332: "i64[]" = torch.ops.aten.copy_.default(primals_845, add_607);  primals_845 = add_607 = None
    copy__333: "f32[104]" = torch.ops.aten.copy_.default(primals_846, add_615);  primals_846 = add_615 = None
    copy__334: "f32[104]" = torch.ops.aten.copy_.default(primals_847, add_616);  primals_847 = add_616 = None
    copy__335: "i64[]" = torch.ops.aten.copy_.default(primals_848, add_613);  primals_848 = add_613 = None
    copy__336: "f32[104]" = torch.ops.aten.copy_.default(primals_849, add_621);  primals_849 = add_621 = None
    copy__337: "f32[104]" = torch.ops.aten.copy_.default(primals_850, add_622);  primals_850 = add_622 = None
    copy__338: "i64[]" = torch.ops.aten.copy_.default(primals_851, add_619);  primals_851 = add_619 = None
    copy__339: "f32[1024]" = torch.ops.aten.copy_.default(primals_852, add_626);  primals_852 = add_626 = None
    copy__340: "f32[1024]" = torch.ops.aten.copy_.default(primals_853, add_627);  primals_853 = add_627 = None
    copy__341: "i64[]" = torch.ops.aten.copy_.default(primals_854, add_624);  primals_854 = add_624 = None
    copy__342: "f32[416]" = torch.ops.aten.copy_.default(primals_855, add_632);  primals_855 = add_632 = None
    copy__343: "f32[416]" = torch.ops.aten.copy_.default(primals_856, add_633);  primals_856 = add_633 = None
    copy__344: "i64[]" = torch.ops.aten.copy_.default(primals_857, add_630);  primals_857 = add_630 = None
    copy__345: "f32[104]" = torch.ops.aten.copy_.default(primals_858, add_637);  primals_858 = add_637 = None
    copy__346: "f32[104]" = torch.ops.aten.copy_.default(primals_859, add_638);  primals_859 = add_638 = None
    copy__347: "i64[]" = torch.ops.aten.copy_.default(primals_860, add_635);  primals_860 = add_635 = None
    copy__348: "f32[104]" = torch.ops.aten.copy_.default(primals_861, add_643);  primals_861 = add_643 = None
    copy__349: "f32[104]" = torch.ops.aten.copy_.default(primals_862, add_644);  primals_862 = add_644 = None
    copy__350: "i64[]" = torch.ops.aten.copy_.default(primals_863, add_641);  primals_863 = add_641 = None
    copy__351: "f32[104]" = torch.ops.aten.copy_.default(primals_864, add_649);  primals_864 = add_649 = None
    copy__352: "f32[104]" = torch.ops.aten.copy_.default(primals_865, add_650);  primals_865 = add_650 = None
    copy__353: "i64[]" = torch.ops.aten.copy_.default(primals_866, add_647);  primals_866 = add_647 = None
    copy__354: "f32[1024]" = torch.ops.aten.copy_.default(primals_867, add_654);  primals_867 = add_654 = None
    copy__355: "f32[1024]" = torch.ops.aten.copy_.default(primals_868, add_655);  primals_868 = add_655 = None
    copy__356: "i64[]" = torch.ops.aten.copy_.default(primals_869, add_652);  primals_869 = add_652 = None
    copy__357: "f32[416]" = torch.ops.aten.copy_.default(primals_870, add_660);  primals_870 = add_660 = None
    copy__358: "f32[416]" = torch.ops.aten.copy_.default(primals_871, add_661);  primals_871 = add_661 = None
    copy__359: "i64[]" = torch.ops.aten.copy_.default(primals_872, add_658);  primals_872 = add_658 = None
    copy__360: "f32[104]" = torch.ops.aten.copy_.default(primals_873, add_665);  primals_873 = add_665 = None
    copy__361: "f32[104]" = torch.ops.aten.copy_.default(primals_874, add_666);  primals_874 = add_666 = None
    copy__362: "i64[]" = torch.ops.aten.copy_.default(primals_875, add_663);  primals_875 = add_663 = None
    copy__363: "f32[104]" = torch.ops.aten.copy_.default(primals_876, add_671);  primals_876 = add_671 = None
    copy__364: "f32[104]" = torch.ops.aten.copy_.default(primals_877, add_672);  primals_877 = add_672 = None
    copy__365: "i64[]" = torch.ops.aten.copy_.default(primals_878, add_669);  primals_878 = add_669 = None
    copy__366: "f32[104]" = torch.ops.aten.copy_.default(primals_879, add_677);  primals_879 = add_677 = None
    copy__367: "f32[104]" = torch.ops.aten.copy_.default(primals_880, add_678);  primals_880 = add_678 = None
    copy__368: "i64[]" = torch.ops.aten.copy_.default(primals_881, add_675);  primals_881 = add_675 = None
    copy__369: "f32[1024]" = torch.ops.aten.copy_.default(primals_882, add_682);  primals_882 = add_682 = None
    copy__370: "f32[1024]" = torch.ops.aten.copy_.default(primals_883, add_683);  primals_883 = add_683 = None
    copy__371: "i64[]" = torch.ops.aten.copy_.default(primals_884, add_680);  primals_884 = add_680 = None
    copy__372: "f32[416]" = torch.ops.aten.copy_.default(primals_885, add_688);  primals_885 = add_688 = None
    copy__373: "f32[416]" = torch.ops.aten.copy_.default(primals_886, add_689);  primals_886 = add_689 = None
    copy__374: "i64[]" = torch.ops.aten.copy_.default(primals_887, add_686);  primals_887 = add_686 = None
    copy__375: "f32[104]" = torch.ops.aten.copy_.default(primals_888, add_693);  primals_888 = add_693 = None
    copy__376: "f32[104]" = torch.ops.aten.copy_.default(primals_889, add_694);  primals_889 = add_694 = None
    copy__377: "i64[]" = torch.ops.aten.copy_.default(primals_890, add_691);  primals_890 = add_691 = None
    copy__378: "f32[104]" = torch.ops.aten.copy_.default(primals_891, add_699);  primals_891 = add_699 = None
    copy__379: "f32[104]" = torch.ops.aten.copy_.default(primals_892, add_700);  primals_892 = add_700 = None
    copy__380: "i64[]" = torch.ops.aten.copy_.default(primals_893, add_697);  primals_893 = add_697 = None
    copy__381: "f32[104]" = torch.ops.aten.copy_.default(primals_894, add_705);  primals_894 = add_705 = None
    copy__382: "f32[104]" = torch.ops.aten.copy_.default(primals_895, add_706);  primals_895 = add_706 = None
    copy__383: "i64[]" = torch.ops.aten.copy_.default(primals_896, add_703);  primals_896 = add_703 = None
    copy__384: "f32[1024]" = torch.ops.aten.copy_.default(primals_897, add_710);  primals_897 = add_710 = None
    copy__385: "f32[1024]" = torch.ops.aten.copy_.default(primals_898, add_711);  primals_898 = add_711 = None
    copy__386: "i64[]" = torch.ops.aten.copy_.default(primals_899, add_708);  primals_899 = add_708 = None
    copy__387: "f32[416]" = torch.ops.aten.copy_.default(primals_900, add_716);  primals_900 = add_716 = None
    copy__388: "f32[416]" = torch.ops.aten.copy_.default(primals_901, add_717);  primals_901 = add_717 = None
    copy__389: "i64[]" = torch.ops.aten.copy_.default(primals_902, add_714);  primals_902 = add_714 = None
    copy__390: "f32[104]" = torch.ops.aten.copy_.default(primals_903, add_721);  primals_903 = add_721 = None
    copy__391: "f32[104]" = torch.ops.aten.copy_.default(primals_904, add_722);  primals_904 = add_722 = None
    copy__392: "i64[]" = torch.ops.aten.copy_.default(primals_905, add_719);  primals_905 = add_719 = None
    copy__393: "f32[104]" = torch.ops.aten.copy_.default(primals_906, add_727);  primals_906 = add_727 = None
    copy__394: "f32[104]" = torch.ops.aten.copy_.default(primals_907, add_728);  primals_907 = add_728 = None
    copy__395: "i64[]" = torch.ops.aten.copy_.default(primals_908, add_725);  primals_908 = add_725 = None
    copy__396: "f32[104]" = torch.ops.aten.copy_.default(primals_909, add_733);  primals_909 = add_733 = None
    copy__397: "f32[104]" = torch.ops.aten.copy_.default(primals_910, add_734);  primals_910 = add_734 = None
    copy__398: "i64[]" = torch.ops.aten.copy_.default(primals_911, add_731);  primals_911 = add_731 = None
    copy__399: "f32[1024]" = torch.ops.aten.copy_.default(primals_912, add_738);  primals_912 = add_738 = None
    copy__400: "f32[1024]" = torch.ops.aten.copy_.default(primals_913, add_739);  primals_913 = add_739 = None
    copy__401: "i64[]" = torch.ops.aten.copy_.default(primals_914, add_736);  primals_914 = add_736 = None
    copy__402: "f32[416]" = torch.ops.aten.copy_.default(primals_915, add_744);  primals_915 = add_744 = None
    copy__403: "f32[416]" = torch.ops.aten.copy_.default(primals_916, add_745);  primals_916 = add_745 = None
    copy__404: "i64[]" = torch.ops.aten.copy_.default(primals_917, add_742);  primals_917 = add_742 = None
    copy__405: "f32[104]" = torch.ops.aten.copy_.default(primals_918, add_749);  primals_918 = add_749 = None
    copy__406: "f32[104]" = torch.ops.aten.copy_.default(primals_919, add_750);  primals_919 = add_750 = None
    copy__407: "i64[]" = torch.ops.aten.copy_.default(primals_920, add_747);  primals_920 = add_747 = None
    copy__408: "f32[104]" = torch.ops.aten.copy_.default(primals_921, add_755);  primals_921 = add_755 = None
    copy__409: "f32[104]" = torch.ops.aten.copy_.default(primals_922, add_756);  primals_922 = add_756 = None
    copy__410: "i64[]" = torch.ops.aten.copy_.default(primals_923, add_753);  primals_923 = add_753 = None
    copy__411: "f32[104]" = torch.ops.aten.copy_.default(primals_924, add_761);  primals_924 = add_761 = None
    copy__412: "f32[104]" = torch.ops.aten.copy_.default(primals_925, add_762);  primals_925 = add_762 = None
    copy__413: "i64[]" = torch.ops.aten.copy_.default(primals_926, add_759);  primals_926 = add_759 = None
    copy__414: "f32[1024]" = torch.ops.aten.copy_.default(primals_927, add_766);  primals_927 = add_766 = None
    copy__415: "f32[1024]" = torch.ops.aten.copy_.default(primals_928, add_767);  primals_928 = add_767 = None
    copy__416: "i64[]" = torch.ops.aten.copy_.default(primals_929, add_764);  primals_929 = add_764 = None
    copy__417: "f32[416]" = torch.ops.aten.copy_.default(primals_930, add_772);  primals_930 = add_772 = None
    copy__418: "f32[416]" = torch.ops.aten.copy_.default(primals_931, add_773);  primals_931 = add_773 = None
    copy__419: "i64[]" = torch.ops.aten.copy_.default(primals_932, add_770);  primals_932 = add_770 = None
    copy__420: "f32[104]" = torch.ops.aten.copy_.default(primals_933, add_777);  primals_933 = add_777 = None
    copy__421: "f32[104]" = torch.ops.aten.copy_.default(primals_934, add_778);  primals_934 = add_778 = None
    copy__422: "i64[]" = torch.ops.aten.copy_.default(primals_935, add_775);  primals_935 = add_775 = None
    copy__423: "f32[104]" = torch.ops.aten.copy_.default(primals_936, add_783);  primals_936 = add_783 = None
    copy__424: "f32[104]" = torch.ops.aten.copy_.default(primals_937, add_784);  primals_937 = add_784 = None
    copy__425: "i64[]" = torch.ops.aten.copy_.default(primals_938, add_781);  primals_938 = add_781 = None
    copy__426: "f32[104]" = torch.ops.aten.copy_.default(primals_939, add_789);  primals_939 = add_789 = None
    copy__427: "f32[104]" = torch.ops.aten.copy_.default(primals_940, add_790);  primals_940 = add_790 = None
    copy__428: "i64[]" = torch.ops.aten.copy_.default(primals_941, add_787);  primals_941 = add_787 = None
    copy__429: "f32[1024]" = torch.ops.aten.copy_.default(primals_942, add_794);  primals_942 = add_794 = None
    copy__430: "f32[1024]" = torch.ops.aten.copy_.default(primals_943, add_795);  primals_943 = add_795 = None
    copy__431: "i64[]" = torch.ops.aten.copy_.default(primals_944, add_792);  primals_944 = add_792 = None
    copy__432: "f32[416]" = torch.ops.aten.copy_.default(primals_945, add_800);  primals_945 = add_800 = None
    copy__433: "f32[416]" = torch.ops.aten.copy_.default(primals_946, add_801);  primals_946 = add_801 = None
    copy__434: "i64[]" = torch.ops.aten.copy_.default(primals_947, add_798);  primals_947 = add_798 = None
    copy__435: "f32[104]" = torch.ops.aten.copy_.default(primals_948, add_805);  primals_948 = add_805 = None
    copy__436: "f32[104]" = torch.ops.aten.copy_.default(primals_949, add_806);  primals_949 = add_806 = None
    copy__437: "i64[]" = torch.ops.aten.copy_.default(primals_950, add_803);  primals_950 = add_803 = None
    copy__438: "f32[104]" = torch.ops.aten.copy_.default(primals_951, add_811);  primals_951 = add_811 = None
    copy__439: "f32[104]" = torch.ops.aten.copy_.default(primals_952, add_812);  primals_952 = add_812 = None
    copy__440: "i64[]" = torch.ops.aten.copy_.default(primals_953, add_809);  primals_953 = add_809 = None
    copy__441: "f32[104]" = torch.ops.aten.copy_.default(primals_954, add_817);  primals_954 = add_817 = None
    copy__442: "f32[104]" = torch.ops.aten.copy_.default(primals_955, add_818);  primals_955 = add_818 = None
    copy__443: "i64[]" = torch.ops.aten.copy_.default(primals_956, add_815);  primals_956 = add_815 = None
    copy__444: "f32[1024]" = torch.ops.aten.copy_.default(primals_957, add_822);  primals_957 = add_822 = None
    copy__445: "f32[1024]" = torch.ops.aten.copy_.default(primals_958, add_823);  primals_958 = add_823 = None
    copy__446: "i64[]" = torch.ops.aten.copy_.default(primals_959, add_820);  primals_959 = add_820 = None
    copy__447: "f32[416]" = torch.ops.aten.copy_.default(primals_960, add_828);  primals_960 = add_828 = None
    copy__448: "f32[416]" = torch.ops.aten.copy_.default(primals_961, add_829);  primals_961 = add_829 = None
    copy__449: "i64[]" = torch.ops.aten.copy_.default(primals_962, add_826);  primals_962 = add_826 = None
    copy__450: "f32[104]" = torch.ops.aten.copy_.default(primals_963, add_833);  primals_963 = add_833 = None
    copy__451: "f32[104]" = torch.ops.aten.copy_.default(primals_964, add_834);  primals_964 = add_834 = None
    copy__452: "i64[]" = torch.ops.aten.copy_.default(primals_965, add_831);  primals_965 = add_831 = None
    copy__453: "f32[104]" = torch.ops.aten.copy_.default(primals_966, add_839);  primals_966 = add_839 = None
    copy__454: "f32[104]" = torch.ops.aten.copy_.default(primals_967, add_840);  primals_967 = add_840 = None
    copy__455: "i64[]" = torch.ops.aten.copy_.default(primals_968, add_837);  primals_968 = add_837 = None
    copy__456: "f32[104]" = torch.ops.aten.copy_.default(primals_969, add_845);  primals_969 = add_845 = None
    copy__457: "f32[104]" = torch.ops.aten.copy_.default(primals_970, add_846);  primals_970 = add_846 = None
    copy__458: "i64[]" = torch.ops.aten.copy_.default(primals_971, add_843);  primals_971 = add_843 = None
    copy__459: "f32[1024]" = torch.ops.aten.copy_.default(primals_972, add_850);  primals_972 = add_850 = None
    copy__460: "f32[1024]" = torch.ops.aten.copy_.default(primals_973, add_851);  primals_973 = add_851 = None
    copy__461: "i64[]" = torch.ops.aten.copy_.default(primals_974, add_848);  primals_974 = add_848 = None
    copy__462: "f32[832]" = torch.ops.aten.copy_.default(primals_975, add_856);  primals_975 = add_856 = None
    copy__463: "f32[832]" = torch.ops.aten.copy_.default(primals_976, add_857);  primals_976 = add_857 = None
    copy__464: "i64[]" = torch.ops.aten.copy_.default(primals_977, add_854);  primals_977 = add_854 = None
    copy__465: "f32[208]" = torch.ops.aten.copy_.default(primals_978, add_861);  primals_978 = add_861 = None
    copy__466: "f32[208]" = torch.ops.aten.copy_.default(primals_979, add_862);  primals_979 = add_862 = None
    copy__467: "i64[]" = torch.ops.aten.copy_.default(primals_980, add_859);  primals_980 = add_859 = None
    copy__468: "f32[208]" = torch.ops.aten.copy_.default(primals_981, add_866);  primals_981 = add_866 = None
    copy__469: "f32[208]" = torch.ops.aten.copy_.default(primals_982, add_867);  primals_982 = add_867 = None
    copy__470: "i64[]" = torch.ops.aten.copy_.default(primals_983, add_864);  primals_983 = add_864 = None
    copy__471: "f32[208]" = torch.ops.aten.copy_.default(primals_984, add_871);  primals_984 = add_871 = None
    copy__472: "f32[208]" = torch.ops.aten.copy_.default(primals_985, add_872);  primals_985 = add_872 = None
    copy__473: "i64[]" = torch.ops.aten.copy_.default(primals_986, add_869);  primals_986 = add_869 = None
    copy__474: "f32[2048]" = torch.ops.aten.copy_.default(primals_987, add_876);  primals_987 = add_876 = None
    copy__475: "f32[2048]" = torch.ops.aten.copy_.default(primals_988, add_877);  primals_988 = add_877 = None
    copy__476: "i64[]" = torch.ops.aten.copy_.default(primals_989, add_874);  primals_989 = add_874 = None
    copy__477: "f32[2048]" = torch.ops.aten.copy_.default(primals_990, add_881);  primals_990 = add_881 = None
    copy__478: "f32[2048]" = torch.ops.aten.copy_.default(primals_991, add_882);  primals_991 = add_882 = None
    copy__479: "i64[]" = torch.ops.aten.copy_.default(primals_992, add_879);  primals_992 = add_879 = None
    copy__480: "f32[832]" = torch.ops.aten.copy_.default(primals_993, add_887);  primals_993 = add_887 = None
    copy__481: "f32[832]" = torch.ops.aten.copy_.default(primals_994, add_888);  primals_994 = add_888 = None
    copy__482: "i64[]" = torch.ops.aten.copy_.default(primals_995, add_885);  primals_995 = add_885 = None
    copy__483: "f32[208]" = torch.ops.aten.copy_.default(primals_996, add_892);  primals_996 = add_892 = None
    copy__484: "f32[208]" = torch.ops.aten.copy_.default(primals_997, add_893);  primals_997 = add_893 = None
    copy__485: "i64[]" = torch.ops.aten.copy_.default(primals_998, add_890);  primals_998 = add_890 = None
    copy__486: "f32[208]" = torch.ops.aten.copy_.default(primals_999, add_898);  primals_999 = add_898 = None
    copy__487: "f32[208]" = torch.ops.aten.copy_.default(primals_1000, add_899);  primals_1000 = add_899 = None
    copy__488: "i64[]" = torch.ops.aten.copy_.default(primals_1001, add_896);  primals_1001 = add_896 = None
    copy__489: "f32[208]" = torch.ops.aten.copy_.default(primals_1002, add_904);  primals_1002 = add_904 = None
    copy__490: "f32[208]" = torch.ops.aten.copy_.default(primals_1003, add_905);  primals_1003 = add_905 = None
    copy__491: "i64[]" = torch.ops.aten.copy_.default(primals_1004, add_902);  primals_1004 = add_902 = None
    copy__492: "f32[2048]" = torch.ops.aten.copy_.default(primals_1005, add_909);  primals_1005 = add_909 = None
    copy__493: "f32[2048]" = torch.ops.aten.copy_.default(primals_1006, add_910);  primals_1006 = add_910 = None
    copy__494: "i64[]" = torch.ops.aten.copy_.default(primals_1007, add_907);  primals_1007 = add_907 = None
    copy__495: "f32[832]" = torch.ops.aten.copy_.default(primals_1008, add_915);  primals_1008 = add_915 = None
    copy__496: "f32[832]" = torch.ops.aten.copy_.default(primals_1009, add_916);  primals_1009 = add_916 = None
    copy__497: "i64[]" = torch.ops.aten.copy_.default(primals_1010, add_913);  primals_1010 = add_913 = None
    copy__498: "f32[208]" = torch.ops.aten.copy_.default(primals_1011, add_920);  primals_1011 = add_920 = None
    copy__499: "f32[208]" = torch.ops.aten.copy_.default(primals_1012, add_921);  primals_1012 = add_921 = None
    copy__500: "i64[]" = torch.ops.aten.copy_.default(primals_1013, add_918);  primals_1013 = add_918 = None
    copy__501: "f32[208]" = torch.ops.aten.copy_.default(primals_1014, add_926);  primals_1014 = add_926 = None
    copy__502: "f32[208]" = torch.ops.aten.copy_.default(primals_1015, add_927);  primals_1015 = add_927 = None
    copy__503: "i64[]" = torch.ops.aten.copy_.default(primals_1016, add_924);  primals_1016 = add_924 = None
    copy__504: "f32[208]" = torch.ops.aten.copy_.default(primals_1017, add_932);  primals_1017 = add_932 = None
    copy__505: "f32[208]" = torch.ops.aten.copy_.default(primals_1018, add_933);  primals_1018 = add_933 = None
    copy__506: "i64[]" = torch.ops.aten.copy_.default(primals_1019, add_930);  primals_1019 = add_930 = None
    copy__507: "f32[2048]" = torch.ops.aten.copy_.default(primals_1020, add_937);  primals_1020 = add_937 = None
    copy__508: "f32[2048]" = torch.ops.aten.copy_.default(primals_1021, add_938);  primals_1021 = add_938 = None
    copy__509: "i64[]" = torch.ops.aten.copy_.default(primals_1022, add_935);  primals_1022 = add_935 = None
    return [addmm, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_313, primals_314, primals_316, primals_317, primals_319, primals_320, primals_322, primals_323, primals_325, primals_326, primals_328, primals_329, primals_331, primals_332, primals_334, primals_335, primals_337, primals_338, primals_340, primals_341, primals_343, primals_344, primals_346, primals_347, primals_349, primals_350, primals_352, primals_353, primals_355, primals_356, primals_358, primals_359, primals_361, primals_362, primals_364, primals_365, primals_367, primals_368, primals_370, primals_371, primals_373, primals_374, primals_376, primals_377, primals_379, primals_380, primals_382, primals_383, primals_385, primals_386, primals_388, primals_389, primals_391, primals_392, primals_394, primals_395, primals_397, primals_398, primals_400, primals_401, primals_403, primals_404, primals_406, primals_407, primals_409, primals_410, primals_412, primals_413, primals_415, primals_416, primals_418, primals_419, primals_421, primals_422, primals_424, primals_425, primals_427, primals_428, primals_430, primals_431, primals_433, primals_434, primals_436, primals_437, primals_439, primals_440, primals_442, primals_443, primals_445, primals_446, primals_448, primals_449, primals_451, primals_452, primals_454, primals_455, primals_457, primals_458, primals_460, primals_461, primals_463, primals_464, primals_466, primals_467, primals_469, primals_470, primals_472, primals_473, primals_475, primals_476, primals_478, primals_479, primals_481, primals_482, primals_484, primals_485, primals_487, primals_488, primals_490, primals_491, primals_493, primals_494, primals_496, primals_497, primals_499, primals_500, primals_502, primals_503, primals_505, primals_506, primals_508, primals_509, primals_1023, convolution, squeeze_1, relu, getitem_2, getitem_3, convolution_1, squeeze_4, getitem_10, convolution_2, squeeze_7, getitem_17, convolution_3, squeeze_10, getitem_24, convolution_4, squeeze_13, getitem_31, cat, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_5, convolution_7, squeeze_22, getitem_42, convolution_8, squeeze_25, add_46, convolution_9, squeeze_28, add_52, convolution_10, squeeze_31, cat_1, convolution_11, squeeze_34, relu_10, convolution_12, squeeze_37, getitem_72, convolution_13, squeeze_40, add_74, convolution_14, squeeze_43, add_80, convolution_15, squeeze_46, cat_2, convolution_16, squeeze_49, relu_15, convolution_17, squeeze_52, getitem_102, convolution_18, squeeze_55, getitem_109, convolution_19, squeeze_58, getitem_116, convolution_20, squeeze_61, getitem_123, cat_3, convolution_21, squeeze_64, convolution_22, squeeze_67, relu_20, convolution_23, squeeze_70, getitem_134, convolution_24, squeeze_73, add_133, convolution_25, squeeze_76, add_139, convolution_26, squeeze_79, cat_4, convolution_27, squeeze_82, relu_25, convolution_28, squeeze_85, getitem_164, convolution_29, squeeze_88, add_161, convolution_30, squeeze_91, add_167, convolution_31, squeeze_94, cat_5, convolution_32, squeeze_97, relu_30, convolution_33, squeeze_100, getitem_194, convolution_34, squeeze_103, add_189, convolution_35, squeeze_106, add_195, convolution_36, squeeze_109, cat_6, convolution_37, squeeze_112, relu_35, convolution_38, squeeze_115, getitem_224, convolution_39, squeeze_118, getitem_231, convolution_40, squeeze_121, getitem_238, convolution_41, squeeze_124, getitem_245, cat_7, convolution_42, squeeze_127, convolution_43, squeeze_130, relu_40, convolution_44, squeeze_133, getitem_256, convolution_45, squeeze_136, add_248, convolution_46, squeeze_139, add_254, convolution_47, squeeze_142, cat_8, convolution_48, squeeze_145, relu_45, convolution_49, squeeze_148, getitem_286, convolution_50, squeeze_151, add_276, convolution_51, squeeze_154, add_282, convolution_52, squeeze_157, cat_9, convolution_53, squeeze_160, relu_50, convolution_54, squeeze_163, getitem_316, convolution_55, squeeze_166, add_304, convolution_56, squeeze_169, add_310, convolution_57, squeeze_172, cat_10, convolution_58, squeeze_175, relu_55, convolution_59, squeeze_178, getitem_346, convolution_60, squeeze_181, add_332, convolution_61, squeeze_184, add_338, convolution_62, squeeze_187, cat_11, convolution_63, squeeze_190, relu_60, convolution_64, squeeze_193, getitem_376, convolution_65, squeeze_196, add_360, convolution_66, squeeze_199, add_366, convolution_67, squeeze_202, cat_12, convolution_68, squeeze_205, relu_65, convolution_69, squeeze_208, getitem_406, convolution_70, squeeze_211, add_388, convolution_71, squeeze_214, add_394, convolution_72, squeeze_217, cat_13, convolution_73, squeeze_220, relu_70, convolution_74, squeeze_223, getitem_436, convolution_75, squeeze_226, add_416, convolution_76, squeeze_229, add_422, convolution_77, squeeze_232, cat_14, convolution_78, squeeze_235, relu_75, convolution_79, squeeze_238, getitem_466, convolution_80, squeeze_241, add_444, convolution_81, squeeze_244, add_450, convolution_82, squeeze_247, cat_15, convolution_83, squeeze_250, relu_80, convolution_84, squeeze_253, getitem_496, convolution_85, squeeze_256, add_472, convolution_86, squeeze_259, add_478, convolution_87, squeeze_262, cat_16, convolution_88, squeeze_265, relu_85, convolution_89, squeeze_268, getitem_526, convolution_90, squeeze_271, add_500, convolution_91, squeeze_274, add_506, convolution_92, squeeze_277, cat_17, convolution_93, squeeze_280, relu_90, convolution_94, squeeze_283, getitem_556, convolution_95, squeeze_286, add_528, convolution_96, squeeze_289, add_534, convolution_97, squeeze_292, cat_18, convolution_98, squeeze_295, relu_95, convolution_99, squeeze_298, getitem_586, convolution_100, squeeze_301, add_556, convolution_101, squeeze_304, add_562, convolution_102, squeeze_307, cat_19, convolution_103, squeeze_310, relu_100, convolution_104, squeeze_313, getitem_616, convolution_105, squeeze_316, add_584, convolution_106, squeeze_319, add_590, convolution_107, squeeze_322, cat_20, convolution_108, squeeze_325, relu_105, convolution_109, squeeze_328, getitem_646, convolution_110, squeeze_331, add_612, convolution_111, squeeze_334, add_618, convolution_112, squeeze_337, cat_21, convolution_113, squeeze_340, relu_110, convolution_114, squeeze_343, getitem_676, convolution_115, squeeze_346, add_640, convolution_116, squeeze_349, add_646, convolution_117, squeeze_352, cat_22, convolution_118, squeeze_355, relu_115, convolution_119, squeeze_358, getitem_706, convolution_120, squeeze_361, add_668, convolution_121, squeeze_364, add_674, convolution_122, squeeze_367, cat_23, convolution_123, squeeze_370, relu_120, convolution_124, squeeze_373, getitem_736, convolution_125, squeeze_376, add_696, convolution_126, squeeze_379, add_702, convolution_127, squeeze_382, cat_24, convolution_128, squeeze_385, relu_125, convolution_129, squeeze_388, getitem_766, convolution_130, squeeze_391, add_724, convolution_131, squeeze_394, add_730, convolution_132, squeeze_397, cat_25, convolution_133, squeeze_400, relu_130, convolution_134, squeeze_403, getitem_796, convolution_135, squeeze_406, add_752, convolution_136, squeeze_409, add_758, convolution_137, squeeze_412, cat_26, convolution_138, squeeze_415, relu_135, convolution_139, squeeze_418, getitem_826, convolution_140, squeeze_421, add_780, convolution_141, squeeze_424, add_786, convolution_142, squeeze_427, cat_27, convolution_143, squeeze_430, relu_140, convolution_144, squeeze_433, getitem_856, convolution_145, squeeze_436, add_808, convolution_146, squeeze_439, add_814, convolution_147, squeeze_442, cat_28, convolution_148, squeeze_445, relu_145, convolution_149, squeeze_448, getitem_886, convolution_150, squeeze_451, add_836, convolution_151, squeeze_454, add_842, convolution_152, squeeze_457, cat_29, convolution_153, squeeze_460, relu_150, convolution_154, squeeze_463, getitem_916, convolution_155, squeeze_466, getitem_923, convolution_156, squeeze_469, getitem_930, convolution_157, squeeze_472, getitem_937, cat_30, convolution_158, squeeze_475, convolution_159, squeeze_478, relu_155, convolution_160, squeeze_481, getitem_948, convolution_161, squeeze_484, add_895, convolution_162, squeeze_487, add_901, convolution_163, squeeze_490, cat_31, convolution_164, squeeze_493, relu_160, convolution_165, squeeze_496, getitem_978, convolution_166, squeeze_499, add_923, convolution_167, squeeze_502, add_929, convolution_168, squeeze_505, cat_32, convolution_169, squeeze_508, view, permute_1, le, unsqueeze_682, le_1, unsqueeze_694, le_2, unsqueeze_706, le_3, unsqueeze_718, le_4, unsqueeze_730, unsqueeze_742, le_6, unsqueeze_754, le_7, unsqueeze_766, le_8, unsqueeze_778, le_9, unsqueeze_790, unsqueeze_802, unsqueeze_814, le_11, unsqueeze_826, le_12, unsqueeze_838, le_13, unsqueeze_850, le_14, unsqueeze_862, unsqueeze_874, le_16, unsqueeze_886, le_17, unsqueeze_898, le_18, unsqueeze_910, le_19, unsqueeze_922, unsqueeze_934, le_21, unsqueeze_946, le_22, unsqueeze_958, le_23, unsqueeze_970, le_24, unsqueeze_982, unsqueeze_994, le_26, unsqueeze_1006, le_27, unsqueeze_1018, le_28, unsqueeze_1030, le_29, unsqueeze_1042, unsqueeze_1054, le_31, unsqueeze_1066, le_32, unsqueeze_1078, le_33, unsqueeze_1090, le_34, unsqueeze_1102, unsqueeze_1114, le_36, unsqueeze_1126, le_37, unsqueeze_1138, le_38, unsqueeze_1150, le_39, unsqueeze_1162, unsqueeze_1174, le_41, unsqueeze_1186, le_42, unsqueeze_1198, le_43, unsqueeze_1210, le_44, unsqueeze_1222, unsqueeze_1234, le_46, unsqueeze_1246, le_47, unsqueeze_1258, le_48, unsqueeze_1270, le_49, unsqueeze_1282, unsqueeze_1294, le_51, unsqueeze_1306, le_52, unsqueeze_1318, le_53, unsqueeze_1330, le_54, unsqueeze_1342, unsqueeze_1354, le_56, unsqueeze_1366, le_57, unsqueeze_1378, le_58, unsqueeze_1390, le_59, unsqueeze_1402, unsqueeze_1414, le_61, unsqueeze_1426, le_62, unsqueeze_1438, le_63, unsqueeze_1450, le_64, unsqueeze_1462, unsqueeze_1474, le_66, unsqueeze_1486, le_67, unsqueeze_1498, le_68, unsqueeze_1510, le_69, unsqueeze_1522, unsqueeze_1534, le_71, unsqueeze_1546, le_72, unsqueeze_1558, le_73, unsqueeze_1570, le_74, unsqueeze_1582, unsqueeze_1594, le_76, unsqueeze_1606, le_77, unsqueeze_1618, le_78, unsqueeze_1630, le_79, unsqueeze_1642, unsqueeze_1654, le_81, unsqueeze_1666, le_82, unsqueeze_1678, le_83, unsqueeze_1690, le_84, unsqueeze_1702, unsqueeze_1714, le_86, unsqueeze_1726, le_87, unsqueeze_1738, le_88, unsqueeze_1750, le_89, unsqueeze_1762, unsqueeze_1774, le_91, unsqueeze_1786, le_92, unsqueeze_1798, le_93, unsqueeze_1810, le_94, unsqueeze_1822, unsqueeze_1834, le_96, unsqueeze_1846, le_97, unsqueeze_1858, le_98, unsqueeze_1870, le_99, unsqueeze_1882, unsqueeze_1894, le_101, unsqueeze_1906, le_102, unsqueeze_1918, le_103, unsqueeze_1930, le_104, unsqueeze_1942, unsqueeze_1954, le_106, unsqueeze_1966, le_107, unsqueeze_1978, le_108, unsqueeze_1990, le_109, unsqueeze_2002, unsqueeze_2014, le_111, unsqueeze_2026, le_112, unsqueeze_2038, le_113, unsqueeze_2050, le_114, unsqueeze_2062, unsqueeze_2074, le_116, unsqueeze_2086, le_117, unsqueeze_2098, le_118, unsqueeze_2110, le_119, unsqueeze_2122, unsqueeze_2134, le_121, unsqueeze_2146, le_122, unsqueeze_2158, le_123, unsqueeze_2170, le_124, unsqueeze_2182, unsqueeze_2194, unsqueeze_2206, le_126, unsqueeze_2218, le_127, unsqueeze_2230, le_128, unsqueeze_2242, le_129, unsqueeze_2254, unsqueeze_2266, le_131, unsqueeze_2278, le_132, unsqueeze_2290, le_133, unsqueeze_2302, le_134, unsqueeze_2314, unsqueeze_2326, le_136, unsqueeze_2338, le_137, unsqueeze_2350, le_138, unsqueeze_2362, le_139, unsqueeze_2374, unsqueeze_2386, le_141, unsqueeze_2398, le_142, unsqueeze_2410, le_143, unsqueeze_2422, le_144, unsqueeze_2434, unsqueeze_2446, unsqueeze_2458, le_146, unsqueeze_2470, le_147, unsqueeze_2482, le_148, unsqueeze_2494, le_149, unsqueeze_2506, unsqueeze_2518, le_151, unsqueeze_2530, le_152, unsqueeze_2542, le_153, unsqueeze_2554, le_154, unsqueeze_2566, unsqueeze_2578, le_156, unsqueeze_2590, le_157, unsqueeze_2602, le_158, unsqueeze_2614, le_159, unsqueeze_2626, unsqueeze_2638, unsqueeze_2650, le_161, unsqueeze_2662, le_162, unsqueeze_2674, le_163, unsqueeze_2686, le_164, unsqueeze_2698, unsqueeze_2710]
    