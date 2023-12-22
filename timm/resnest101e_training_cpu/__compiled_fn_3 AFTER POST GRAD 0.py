from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_2: "f32[64]", primals_3: "f32[64]", primals_4: "f32[64, 64, 3, 3]", primals_5: "f32[64]", primals_6: "f32[64]", primals_7: "f32[128, 64, 3, 3]", primals_8: "f32[128]", primals_9: "f32[128]", primals_10: "f32[64, 128, 1, 1]", primals_11: "f32[64]", primals_12: "f32[64]", primals_13: "f32[128, 32, 3, 3]", primals_14: "f32[128]", primals_15: "f32[128]", primals_16: "f32[32, 64, 1, 1]", primals_17: "f32[32]", primals_18: "f32[32]", primals_19: "f32[32]", primals_20: "f32[128, 32, 1, 1]", primals_21: "f32[128]", primals_22: "f32[256, 64, 1, 1]", primals_23: "f32[256]", primals_24: "f32[256]", primals_25: "f32[256, 128, 1, 1]", primals_26: "f32[256]", primals_27: "f32[256]", primals_28: "f32[64, 256, 1, 1]", primals_29: "f32[64]", primals_30: "f32[64]", primals_31: "f32[128, 32, 3, 3]", primals_32: "f32[128]", primals_33: "f32[128]", primals_34: "f32[32, 64, 1, 1]", primals_35: "f32[32]", primals_36: "f32[32]", primals_37: "f32[32]", primals_38: "f32[128, 32, 1, 1]", primals_39: "f32[128]", primals_40: "f32[256, 64, 1, 1]", primals_41: "f32[256]", primals_42: "f32[256]", primals_43: "f32[64, 256, 1, 1]", primals_44: "f32[64]", primals_45: "f32[64]", primals_46: "f32[128, 32, 3, 3]", primals_47: "f32[128]", primals_48: "f32[128]", primals_49: "f32[32, 64, 1, 1]", primals_50: "f32[32]", primals_51: "f32[32]", primals_52: "f32[32]", primals_53: "f32[128, 32, 1, 1]", primals_54: "f32[128]", primals_55: "f32[256, 64, 1, 1]", primals_56: "f32[256]", primals_57: "f32[256]", primals_58: "f32[128, 256, 1, 1]", primals_59: "f32[128]", primals_60: "f32[128]", primals_61: "f32[256, 64, 3, 3]", primals_62: "f32[256]", primals_63: "f32[256]", primals_64: "f32[64, 128, 1, 1]", primals_65: "f32[64]", primals_66: "f32[64]", primals_67: "f32[64]", primals_68: "f32[256, 64, 1, 1]", primals_69: "f32[256]", primals_70: "f32[512, 128, 1, 1]", primals_71: "f32[512]", primals_72: "f32[512]", primals_73: "f32[512, 256, 1, 1]", primals_74: "f32[512]", primals_75: "f32[512]", primals_76: "f32[128, 512, 1, 1]", primals_77: "f32[128]", primals_78: "f32[128]", primals_79: "f32[256, 64, 3, 3]", primals_80: "f32[256]", primals_81: "f32[256]", primals_82: "f32[64, 128, 1, 1]", primals_83: "f32[64]", primals_84: "f32[64]", primals_85: "f32[64]", primals_86: "f32[256, 64, 1, 1]", primals_87: "f32[256]", primals_88: "f32[512, 128, 1, 1]", primals_89: "f32[512]", primals_90: "f32[512]", primals_91: "f32[128, 512, 1, 1]", primals_92: "f32[128]", primals_93: "f32[128]", primals_94: "f32[256, 64, 3, 3]", primals_95: "f32[256]", primals_96: "f32[256]", primals_97: "f32[64, 128, 1, 1]", primals_98: "f32[64]", primals_99: "f32[64]", primals_100: "f32[64]", primals_101: "f32[256, 64, 1, 1]", primals_102: "f32[256]", primals_103: "f32[512, 128, 1, 1]", primals_104: "f32[512]", primals_105: "f32[512]", primals_106: "f32[128, 512, 1, 1]", primals_107: "f32[128]", primals_108: "f32[128]", primals_109: "f32[256, 64, 3, 3]", primals_110: "f32[256]", primals_111: "f32[256]", primals_112: "f32[64, 128, 1, 1]", primals_113: "f32[64]", primals_114: "f32[64]", primals_115: "f32[64]", primals_116: "f32[256, 64, 1, 1]", primals_117: "f32[256]", primals_118: "f32[512, 128, 1, 1]", primals_119: "f32[512]", primals_120: "f32[512]", primals_121: "f32[256, 512, 1, 1]", primals_122: "f32[256]", primals_123: "f32[256]", primals_124: "f32[512, 128, 3, 3]", primals_125: "f32[512]", primals_126: "f32[512]", primals_127: "f32[128, 256, 1, 1]", primals_128: "f32[128]", primals_129: "f32[128]", primals_130: "f32[128]", primals_131: "f32[512, 128, 1, 1]", primals_132: "f32[512]", primals_133: "f32[1024, 256, 1, 1]", primals_134: "f32[1024]", primals_135: "f32[1024]", primals_136: "f32[1024, 512, 1, 1]", primals_137: "f32[1024]", primals_138: "f32[1024]", primals_139: "f32[256, 1024, 1, 1]", primals_140: "f32[256]", primals_141: "f32[256]", primals_142: "f32[512, 128, 3, 3]", primals_143: "f32[512]", primals_144: "f32[512]", primals_145: "f32[128, 256, 1, 1]", primals_146: "f32[128]", primals_147: "f32[128]", primals_148: "f32[128]", primals_149: "f32[512, 128, 1, 1]", primals_150: "f32[512]", primals_151: "f32[1024, 256, 1, 1]", primals_152: "f32[1024]", primals_153: "f32[1024]", primals_154: "f32[256, 1024, 1, 1]", primals_155: "f32[256]", primals_156: "f32[256]", primals_157: "f32[512, 128, 3, 3]", primals_158: "f32[512]", primals_159: "f32[512]", primals_160: "f32[128, 256, 1, 1]", primals_161: "f32[128]", primals_162: "f32[128]", primals_163: "f32[128]", primals_164: "f32[512, 128, 1, 1]", primals_165: "f32[512]", primals_166: "f32[1024, 256, 1, 1]", primals_167: "f32[1024]", primals_168: "f32[1024]", primals_169: "f32[256, 1024, 1, 1]", primals_170: "f32[256]", primals_171: "f32[256]", primals_172: "f32[512, 128, 3, 3]", primals_173: "f32[512]", primals_174: "f32[512]", primals_175: "f32[128, 256, 1, 1]", primals_176: "f32[128]", primals_177: "f32[128]", primals_178: "f32[128]", primals_179: "f32[512, 128, 1, 1]", primals_180: "f32[512]", primals_181: "f32[1024, 256, 1, 1]", primals_182: "f32[1024]", primals_183: "f32[1024]", primals_184: "f32[256, 1024, 1, 1]", primals_185: "f32[256]", primals_186: "f32[256]", primals_187: "f32[512, 128, 3, 3]", primals_188: "f32[512]", primals_189: "f32[512]", primals_190: "f32[128, 256, 1, 1]", primals_191: "f32[128]", primals_192: "f32[128]", primals_193: "f32[128]", primals_194: "f32[512, 128, 1, 1]", primals_195: "f32[512]", primals_196: "f32[1024, 256, 1, 1]", primals_197: "f32[1024]", primals_198: "f32[1024]", primals_199: "f32[256, 1024, 1, 1]", primals_200: "f32[256]", primals_201: "f32[256]", primals_202: "f32[512, 128, 3, 3]", primals_203: "f32[512]", primals_204: "f32[512]", primals_205: "f32[128, 256, 1, 1]", primals_206: "f32[128]", primals_207: "f32[128]", primals_208: "f32[128]", primals_209: "f32[512, 128, 1, 1]", primals_210: "f32[512]", primals_211: "f32[1024, 256, 1, 1]", primals_212: "f32[1024]", primals_213: "f32[1024]", primals_214: "f32[256, 1024, 1, 1]", primals_215: "f32[256]", primals_216: "f32[256]", primals_217: "f32[512, 128, 3, 3]", primals_218: "f32[512]", primals_219: "f32[512]", primals_220: "f32[128, 256, 1, 1]", primals_221: "f32[128]", primals_222: "f32[128]", primals_223: "f32[128]", primals_224: "f32[512, 128, 1, 1]", primals_225: "f32[512]", primals_226: "f32[1024, 256, 1, 1]", primals_227: "f32[1024]", primals_228: "f32[1024]", primals_229: "f32[256, 1024, 1, 1]", primals_230: "f32[256]", primals_231: "f32[256]", primals_232: "f32[512, 128, 3, 3]", primals_233: "f32[512]", primals_234: "f32[512]", primals_235: "f32[128, 256, 1, 1]", primals_236: "f32[128]", primals_237: "f32[128]", primals_238: "f32[128]", primals_239: "f32[512, 128, 1, 1]", primals_240: "f32[512]", primals_241: "f32[1024, 256, 1, 1]", primals_242: "f32[1024]", primals_243: "f32[1024]", primals_244: "f32[256, 1024, 1, 1]", primals_245: "f32[256]", primals_246: "f32[256]", primals_247: "f32[512, 128, 3, 3]", primals_248: "f32[512]", primals_249: "f32[512]", primals_250: "f32[128, 256, 1, 1]", primals_251: "f32[128]", primals_252: "f32[128]", primals_253: "f32[128]", primals_254: "f32[512, 128, 1, 1]", primals_255: "f32[512]", primals_256: "f32[1024, 256, 1, 1]", primals_257: "f32[1024]", primals_258: "f32[1024]", primals_259: "f32[256, 1024, 1, 1]", primals_260: "f32[256]", primals_261: "f32[256]", primals_262: "f32[512, 128, 3, 3]", primals_263: "f32[512]", primals_264: "f32[512]", primals_265: "f32[128, 256, 1, 1]", primals_266: "f32[128]", primals_267: "f32[128]", primals_268: "f32[128]", primals_269: "f32[512, 128, 1, 1]", primals_270: "f32[512]", primals_271: "f32[1024, 256, 1, 1]", primals_272: "f32[1024]", primals_273: "f32[1024]", primals_274: "f32[256, 1024, 1, 1]", primals_275: "f32[256]", primals_276: "f32[256]", primals_277: "f32[512, 128, 3, 3]", primals_278: "f32[512]", primals_279: "f32[512]", primals_280: "f32[128, 256, 1, 1]", primals_281: "f32[128]", primals_282: "f32[128]", primals_283: "f32[128]", primals_284: "f32[512, 128, 1, 1]", primals_285: "f32[512]", primals_286: "f32[1024, 256, 1, 1]", primals_287: "f32[1024]", primals_288: "f32[1024]", primals_289: "f32[256, 1024, 1, 1]", primals_290: "f32[256]", primals_291: "f32[256]", primals_292: "f32[512, 128, 3, 3]", primals_293: "f32[512]", primals_294: "f32[512]", primals_295: "f32[128, 256, 1, 1]", primals_296: "f32[128]", primals_297: "f32[128]", primals_298: "f32[128]", primals_299: "f32[512, 128, 1, 1]", primals_300: "f32[512]", primals_301: "f32[1024, 256, 1, 1]", primals_302: "f32[1024]", primals_303: "f32[1024]", primals_304: "f32[256, 1024, 1, 1]", primals_305: "f32[256]", primals_306: "f32[256]", primals_307: "f32[512, 128, 3, 3]", primals_308: "f32[512]", primals_309: "f32[512]", primals_310: "f32[128, 256, 1, 1]", primals_311: "f32[128]", primals_312: "f32[128]", primals_313: "f32[128]", primals_314: "f32[512, 128, 1, 1]", primals_315: "f32[512]", primals_316: "f32[1024, 256, 1, 1]", primals_317: "f32[1024]", primals_318: "f32[1024]", primals_319: "f32[256, 1024, 1, 1]", primals_320: "f32[256]", primals_321: "f32[256]", primals_322: "f32[512, 128, 3, 3]", primals_323: "f32[512]", primals_324: "f32[512]", primals_325: "f32[128, 256, 1, 1]", primals_326: "f32[128]", primals_327: "f32[128]", primals_328: "f32[128]", primals_329: "f32[512, 128, 1, 1]", primals_330: "f32[512]", primals_331: "f32[1024, 256, 1, 1]", primals_332: "f32[1024]", primals_333: "f32[1024]", primals_334: "f32[256, 1024, 1, 1]", primals_335: "f32[256]", primals_336: "f32[256]", primals_337: "f32[512, 128, 3, 3]", primals_338: "f32[512]", primals_339: "f32[512]", primals_340: "f32[128, 256, 1, 1]", primals_341: "f32[128]", primals_342: "f32[128]", primals_343: "f32[128]", primals_344: "f32[512, 128, 1, 1]", primals_345: "f32[512]", primals_346: "f32[1024, 256, 1, 1]", primals_347: "f32[1024]", primals_348: "f32[1024]", primals_349: "f32[256, 1024, 1, 1]", primals_350: "f32[256]", primals_351: "f32[256]", primals_352: "f32[512, 128, 3, 3]", primals_353: "f32[512]", primals_354: "f32[512]", primals_355: "f32[128, 256, 1, 1]", primals_356: "f32[128]", primals_357: "f32[128]", primals_358: "f32[128]", primals_359: "f32[512, 128, 1, 1]", primals_360: "f32[512]", primals_361: "f32[1024, 256, 1, 1]", primals_362: "f32[1024]", primals_363: "f32[1024]", primals_364: "f32[256, 1024, 1, 1]", primals_365: "f32[256]", primals_366: "f32[256]", primals_367: "f32[512, 128, 3, 3]", primals_368: "f32[512]", primals_369: "f32[512]", primals_370: "f32[128, 256, 1, 1]", primals_371: "f32[128]", primals_372: "f32[128]", primals_373: "f32[128]", primals_374: "f32[512, 128, 1, 1]", primals_375: "f32[512]", primals_376: "f32[1024, 256, 1, 1]", primals_377: "f32[1024]", primals_378: "f32[1024]", primals_379: "f32[256, 1024, 1, 1]", primals_380: "f32[256]", primals_381: "f32[256]", primals_382: "f32[512, 128, 3, 3]", primals_383: "f32[512]", primals_384: "f32[512]", primals_385: "f32[128, 256, 1, 1]", primals_386: "f32[128]", primals_387: "f32[128]", primals_388: "f32[128]", primals_389: "f32[512, 128, 1, 1]", primals_390: "f32[512]", primals_391: "f32[1024, 256, 1, 1]", primals_392: "f32[1024]", primals_393: "f32[1024]", primals_394: "f32[256, 1024, 1, 1]", primals_395: "f32[256]", primals_396: "f32[256]", primals_397: "f32[512, 128, 3, 3]", primals_398: "f32[512]", primals_399: "f32[512]", primals_400: "f32[128, 256, 1, 1]", primals_401: "f32[128]", primals_402: "f32[128]", primals_403: "f32[128]", primals_404: "f32[512, 128, 1, 1]", primals_405: "f32[512]", primals_406: "f32[1024, 256, 1, 1]", primals_407: "f32[1024]", primals_408: "f32[1024]", primals_409: "f32[256, 1024, 1, 1]", primals_410: "f32[256]", primals_411: "f32[256]", primals_412: "f32[512, 128, 3, 3]", primals_413: "f32[512]", primals_414: "f32[512]", primals_415: "f32[128, 256, 1, 1]", primals_416: "f32[128]", primals_417: "f32[128]", primals_418: "f32[128]", primals_419: "f32[512, 128, 1, 1]", primals_420: "f32[512]", primals_421: "f32[1024, 256, 1, 1]", primals_422: "f32[1024]", primals_423: "f32[1024]", primals_424: "f32[256, 1024, 1, 1]", primals_425: "f32[256]", primals_426: "f32[256]", primals_427: "f32[512, 128, 3, 3]", primals_428: "f32[512]", primals_429: "f32[512]", primals_430: "f32[128, 256, 1, 1]", primals_431: "f32[128]", primals_432: "f32[128]", primals_433: "f32[128]", primals_434: "f32[512, 128, 1, 1]", primals_435: "f32[512]", primals_436: "f32[1024, 256, 1, 1]", primals_437: "f32[1024]", primals_438: "f32[1024]", primals_439: "f32[256, 1024, 1, 1]", primals_440: "f32[256]", primals_441: "f32[256]", primals_442: "f32[512, 128, 3, 3]", primals_443: "f32[512]", primals_444: "f32[512]", primals_445: "f32[128, 256, 1, 1]", primals_446: "f32[128]", primals_447: "f32[128]", primals_448: "f32[128]", primals_449: "f32[512, 128, 1, 1]", primals_450: "f32[512]", primals_451: "f32[1024, 256, 1, 1]", primals_452: "f32[1024]", primals_453: "f32[1024]", primals_454: "f32[256, 1024, 1, 1]", primals_455: "f32[256]", primals_456: "f32[256]", primals_457: "f32[512, 128, 3, 3]", primals_458: "f32[512]", primals_459: "f32[512]", primals_460: "f32[128, 256, 1, 1]", primals_461: "f32[128]", primals_462: "f32[128]", primals_463: "f32[128]", primals_464: "f32[512, 128, 1, 1]", primals_465: "f32[512]", primals_466: "f32[1024, 256, 1, 1]", primals_467: "f32[1024]", primals_468: "f32[1024]", primals_469: "f32[512, 1024, 1, 1]", primals_470: "f32[512]", primals_471: "f32[512]", primals_472: "f32[1024, 256, 3, 3]", primals_473: "f32[1024]", primals_474: "f32[1024]", primals_475: "f32[256, 512, 1, 1]", primals_476: "f32[256]", primals_477: "f32[256]", primals_478: "f32[256]", primals_479: "f32[1024, 256, 1, 1]", primals_480: "f32[1024]", primals_481: "f32[2048, 512, 1, 1]", primals_482: "f32[2048]", primals_483: "f32[2048]", primals_484: "f32[2048, 1024, 1, 1]", primals_485: "f32[2048]", primals_486: "f32[2048]", primals_487: "f32[512, 2048, 1, 1]", primals_488: "f32[512]", primals_489: "f32[512]", primals_490: "f32[1024, 256, 3, 3]", primals_491: "f32[1024]", primals_492: "f32[1024]", primals_493: "f32[256, 512, 1, 1]", primals_494: "f32[256]", primals_495: "f32[256]", primals_496: "f32[256]", primals_497: "f32[1024, 256, 1, 1]", primals_498: "f32[1024]", primals_499: "f32[2048, 512, 1, 1]", primals_500: "f32[2048]", primals_501: "f32[2048]", primals_502: "f32[512, 2048, 1, 1]", primals_503: "f32[512]", primals_504: "f32[512]", primals_505: "f32[1024, 256, 3, 3]", primals_506: "f32[1024]", primals_507: "f32[1024]", primals_508: "f32[256, 512, 1, 1]", primals_509: "f32[256]", primals_510: "f32[256]", primals_511: "f32[256]", primals_512: "f32[1024, 256, 1, 1]", primals_513: "f32[1024]", primals_514: "f32[2048, 512, 1, 1]", primals_515: "f32[2048]", primals_516: "f32[2048]", primals_517: "f32[1000, 2048]", primals_518: "f32[1000]", primals_519: "f32[64]", primals_520: "f32[64]", primals_521: "i64[]", primals_522: "f32[64]", primals_523: "f32[64]", primals_524: "i64[]", primals_525: "f32[128]", primals_526: "f32[128]", primals_527: "i64[]", primals_528: "f32[64]", primals_529: "f32[64]", primals_530: "i64[]", primals_531: "f32[128]", primals_532: "f32[128]", primals_533: "i64[]", primals_534: "f32[32]", primals_535: "f32[32]", primals_536: "i64[]", primals_537: "f32[256]", primals_538: "f32[256]", primals_539: "i64[]", primals_540: "f32[256]", primals_541: "f32[256]", primals_542: "i64[]", primals_543: "f32[64]", primals_544: "f32[64]", primals_545: "i64[]", primals_546: "f32[128]", primals_547: "f32[128]", primals_548: "i64[]", primals_549: "f32[32]", primals_550: "f32[32]", primals_551: "i64[]", primals_552: "f32[256]", primals_553: "f32[256]", primals_554: "i64[]", primals_555: "f32[64]", primals_556: "f32[64]", primals_557: "i64[]", primals_558: "f32[128]", primals_559: "f32[128]", primals_560: "i64[]", primals_561: "f32[32]", primals_562: "f32[32]", primals_563: "i64[]", primals_564: "f32[256]", primals_565: "f32[256]", primals_566: "i64[]", primals_567: "f32[128]", primals_568: "f32[128]", primals_569: "i64[]", primals_570: "f32[256]", primals_571: "f32[256]", primals_572: "i64[]", primals_573: "f32[64]", primals_574: "f32[64]", primals_575: "i64[]", primals_576: "f32[512]", primals_577: "f32[512]", primals_578: "i64[]", primals_579: "f32[512]", primals_580: "f32[512]", primals_581: "i64[]", primals_582: "f32[128]", primals_583: "f32[128]", primals_584: "i64[]", primals_585: "f32[256]", primals_586: "f32[256]", primals_587: "i64[]", primals_588: "f32[64]", primals_589: "f32[64]", primals_590: "i64[]", primals_591: "f32[512]", primals_592: "f32[512]", primals_593: "i64[]", primals_594: "f32[128]", primals_595: "f32[128]", primals_596: "i64[]", primals_597: "f32[256]", primals_598: "f32[256]", primals_599: "i64[]", primals_600: "f32[64]", primals_601: "f32[64]", primals_602: "i64[]", primals_603: "f32[512]", primals_604: "f32[512]", primals_605: "i64[]", primals_606: "f32[128]", primals_607: "f32[128]", primals_608: "i64[]", primals_609: "f32[256]", primals_610: "f32[256]", primals_611: "i64[]", primals_612: "f32[64]", primals_613: "f32[64]", primals_614: "i64[]", primals_615: "f32[512]", primals_616: "f32[512]", primals_617: "i64[]", primals_618: "f32[256]", primals_619: "f32[256]", primals_620: "i64[]", primals_621: "f32[512]", primals_622: "f32[512]", primals_623: "i64[]", primals_624: "f32[128]", primals_625: "f32[128]", primals_626: "i64[]", primals_627: "f32[1024]", primals_628: "f32[1024]", primals_629: "i64[]", primals_630: "f32[1024]", primals_631: "f32[1024]", primals_632: "i64[]", primals_633: "f32[256]", primals_634: "f32[256]", primals_635: "i64[]", primals_636: "f32[512]", primals_637: "f32[512]", primals_638: "i64[]", primals_639: "f32[128]", primals_640: "f32[128]", primals_641: "i64[]", primals_642: "f32[1024]", primals_643: "f32[1024]", primals_644: "i64[]", primals_645: "f32[256]", primals_646: "f32[256]", primals_647: "i64[]", primals_648: "f32[512]", primals_649: "f32[512]", primals_650: "i64[]", primals_651: "f32[128]", primals_652: "f32[128]", primals_653: "i64[]", primals_654: "f32[1024]", primals_655: "f32[1024]", primals_656: "i64[]", primals_657: "f32[256]", primals_658: "f32[256]", primals_659: "i64[]", primals_660: "f32[512]", primals_661: "f32[512]", primals_662: "i64[]", primals_663: "f32[128]", primals_664: "f32[128]", primals_665: "i64[]", primals_666: "f32[1024]", primals_667: "f32[1024]", primals_668: "i64[]", primals_669: "f32[256]", primals_670: "f32[256]", primals_671: "i64[]", primals_672: "f32[512]", primals_673: "f32[512]", primals_674: "i64[]", primals_675: "f32[128]", primals_676: "f32[128]", primals_677: "i64[]", primals_678: "f32[1024]", primals_679: "f32[1024]", primals_680: "i64[]", primals_681: "f32[256]", primals_682: "f32[256]", primals_683: "i64[]", primals_684: "f32[512]", primals_685: "f32[512]", primals_686: "i64[]", primals_687: "f32[128]", primals_688: "f32[128]", primals_689: "i64[]", primals_690: "f32[1024]", primals_691: "f32[1024]", primals_692: "i64[]", primals_693: "f32[256]", primals_694: "f32[256]", primals_695: "i64[]", primals_696: "f32[512]", primals_697: "f32[512]", primals_698: "i64[]", primals_699: "f32[128]", primals_700: "f32[128]", primals_701: "i64[]", primals_702: "f32[1024]", primals_703: "f32[1024]", primals_704: "i64[]", primals_705: "f32[256]", primals_706: "f32[256]", primals_707: "i64[]", primals_708: "f32[512]", primals_709: "f32[512]", primals_710: "i64[]", primals_711: "f32[128]", primals_712: "f32[128]", primals_713: "i64[]", primals_714: "f32[1024]", primals_715: "f32[1024]", primals_716: "i64[]", primals_717: "f32[256]", primals_718: "f32[256]", primals_719: "i64[]", primals_720: "f32[512]", primals_721: "f32[512]", primals_722: "i64[]", primals_723: "f32[128]", primals_724: "f32[128]", primals_725: "i64[]", primals_726: "f32[1024]", primals_727: "f32[1024]", primals_728: "i64[]", primals_729: "f32[256]", primals_730: "f32[256]", primals_731: "i64[]", primals_732: "f32[512]", primals_733: "f32[512]", primals_734: "i64[]", primals_735: "f32[128]", primals_736: "f32[128]", primals_737: "i64[]", primals_738: "f32[1024]", primals_739: "f32[1024]", primals_740: "i64[]", primals_741: "f32[256]", primals_742: "f32[256]", primals_743: "i64[]", primals_744: "f32[512]", primals_745: "f32[512]", primals_746: "i64[]", primals_747: "f32[128]", primals_748: "f32[128]", primals_749: "i64[]", primals_750: "f32[1024]", primals_751: "f32[1024]", primals_752: "i64[]", primals_753: "f32[256]", primals_754: "f32[256]", primals_755: "i64[]", primals_756: "f32[512]", primals_757: "f32[512]", primals_758: "i64[]", primals_759: "f32[128]", primals_760: "f32[128]", primals_761: "i64[]", primals_762: "f32[1024]", primals_763: "f32[1024]", primals_764: "i64[]", primals_765: "f32[256]", primals_766: "f32[256]", primals_767: "i64[]", primals_768: "f32[512]", primals_769: "f32[512]", primals_770: "i64[]", primals_771: "f32[128]", primals_772: "f32[128]", primals_773: "i64[]", primals_774: "f32[1024]", primals_775: "f32[1024]", primals_776: "i64[]", primals_777: "f32[256]", primals_778: "f32[256]", primals_779: "i64[]", primals_780: "f32[512]", primals_781: "f32[512]", primals_782: "i64[]", primals_783: "f32[128]", primals_784: "f32[128]", primals_785: "i64[]", primals_786: "f32[1024]", primals_787: "f32[1024]", primals_788: "i64[]", primals_789: "f32[256]", primals_790: "f32[256]", primals_791: "i64[]", primals_792: "f32[512]", primals_793: "f32[512]", primals_794: "i64[]", primals_795: "f32[128]", primals_796: "f32[128]", primals_797: "i64[]", primals_798: "f32[1024]", primals_799: "f32[1024]", primals_800: "i64[]", primals_801: "f32[256]", primals_802: "f32[256]", primals_803: "i64[]", primals_804: "f32[512]", primals_805: "f32[512]", primals_806: "i64[]", primals_807: "f32[128]", primals_808: "f32[128]", primals_809: "i64[]", primals_810: "f32[1024]", primals_811: "f32[1024]", primals_812: "i64[]", primals_813: "f32[256]", primals_814: "f32[256]", primals_815: "i64[]", primals_816: "f32[512]", primals_817: "f32[512]", primals_818: "i64[]", primals_819: "f32[128]", primals_820: "f32[128]", primals_821: "i64[]", primals_822: "f32[1024]", primals_823: "f32[1024]", primals_824: "i64[]", primals_825: "f32[256]", primals_826: "f32[256]", primals_827: "i64[]", primals_828: "f32[512]", primals_829: "f32[512]", primals_830: "i64[]", primals_831: "f32[128]", primals_832: "f32[128]", primals_833: "i64[]", primals_834: "f32[1024]", primals_835: "f32[1024]", primals_836: "i64[]", primals_837: "f32[256]", primals_838: "f32[256]", primals_839: "i64[]", primals_840: "f32[512]", primals_841: "f32[512]", primals_842: "i64[]", primals_843: "f32[128]", primals_844: "f32[128]", primals_845: "i64[]", primals_846: "f32[1024]", primals_847: "f32[1024]", primals_848: "i64[]", primals_849: "f32[256]", primals_850: "f32[256]", primals_851: "i64[]", primals_852: "f32[512]", primals_853: "f32[512]", primals_854: "i64[]", primals_855: "f32[128]", primals_856: "f32[128]", primals_857: "i64[]", primals_858: "f32[1024]", primals_859: "f32[1024]", primals_860: "i64[]", primals_861: "f32[256]", primals_862: "f32[256]", primals_863: "i64[]", primals_864: "f32[512]", primals_865: "f32[512]", primals_866: "i64[]", primals_867: "f32[128]", primals_868: "f32[128]", primals_869: "i64[]", primals_870: "f32[1024]", primals_871: "f32[1024]", primals_872: "i64[]", primals_873: "f32[256]", primals_874: "f32[256]", primals_875: "i64[]", primals_876: "f32[512]", primals_877: "f32[512]", primals_878: "i64[]", primals_879: "f32[128]", primals_880: "f32[128]", primals_881: "i64[]", primals_882: "f32[1024]", primals_883: "f32[1024]", primals_884: "i64[]", primals_885: "f32[256]", primals_886: "f32[256]", primals_887: "i64[]", primals_888: "f32[512]", primals_889: "f32[512]", primals_890: "i64[]", primals_891: "f32[128]", primals_892: "f32[128]", primals_893: "i64[]", primals_894: "f32[1024]", primals_895: "f32[1024]", primals_896: "i64[]", primals_897: "f32[512]", primals_898: "f32[512]", primals_899: "i64[]", primals_900: "f32[1024]", primals_901: "f32[1024]", primals_902: "i64[]", primals_903: "f32[256]", primals_904: "f32[256]", primals_905: "i64[]", primals_906: "f32[2048]", primals_907: "f32[2048]", primals_908: "i64[]", primals_909: "f32[2048]", primals_910: "f32[2048]", primals_911: "i64[]", primals_912: "f32[512]", primals_913: "f32[512]", primals_914: "i64[]", primals_915: "f32[1024]", primals_916: "f32[1024]", primals_917: "i64[]", primals_918: "f32[256]", primals_919: "f32[256]", primals_920: "i64[]", primals_921: "f32[2048]", primals_922: "f32[2048]", primals_923: "i64[]", primals_924: "f32[512]", primals_925: "f32[512]", primals_926: "i64[]", primals_927: "f32[1024]", primals_928: "f32[1024]", primals_929: "i64[]", primals_930: "f32[256]", primals_931: "f32[256]", primals_932: "i64[]", primals_933: "f32[2048]", primals_934: "f32[2048]", primals_935: "i64[]", primals_936: "f32[8, 3, 256, 256]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(primals_936, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_521, 1)
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 64, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 64, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(primals_519, 0.9)
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000076294527394);  squeeze_2 = None
    mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(primals_520, 0.9)
    add_3: "f32[64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    relu: "f32[8, 64, 128, 128]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    convolution_1: "f32[8, 64, 128, 128]" = torch.ops.aten.convolution.default(relu, primals_4, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_524, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 64, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 64, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[64]" = torch.ops.aten.mul.Tensor(primals_522, 0.9)
    add_7: "f32[64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000076294527394);  squeeze_5 = None
    mul_11: "f32[64]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[64]" = torch.ops.aten.mul.Tensor(primals_523, 0.9)
    add_8: "f32[64]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    relu_1: "f32[8, 64, 128, 128]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    convolution_2: "f32[8, 128, 128, 128]" = torch.ops.aten.convolution.default(relu_1, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_527, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[128]" = torch.ops.aten.mul.Tensor(primals_525, 0.9)
    add_12: "f32[128]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000076294527394);  squeeze_8 = None
    mul_18: "f32[128]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[128]" = torch.ops.aten.mul.Tensor(primals_526, 0.9)
    add_13: "f32[128]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_9: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_11: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 128, 128, 128]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    relu_2: "f32[8, 128, 128, 128]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu_2, [3, 3], [2, 2], [1, 1])
    getitem_6: "f32[8, 128, 64, 64]" = max_pool2d_with_indices[0]
    getitem_7: "i64[8, 128, 64, 64]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_3: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(getitem_6, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_530, 1)
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 64, 1, 1]" = var_mean_3[0]
    getitem_9: "f32[1, 64, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_3: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
    mul_21: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_10: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[64]" = torch.ops.aten.mul.Tensor(primals_528, 0.9)
    add_17: "f32[64]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_24: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.000030518509476);  squeeze_11 = None
    mul_25: "f32[64]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[64]" = torch.ops.aten.mul.Tensor(primals_529, 0.9)
    add_18: "f32[64]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_3: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_4: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(relu_3, primals_13, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_533, 1)
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1, 1]" = var_mean_4[0]
    getitem_11: "f32[1, 128, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_4: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
    mul_28: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_13: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[128]" = torch.ops.aten.mul.Tensor(primals_531, 0.9)
    add_22: "f32[128]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_31: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.000030518509476);  squeeze_14 = None
    mul_32: "f32[128]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[128]" = torch.ops.aten.mul.Tensor(primals_532, 0.9)
    add_23: "f32[128]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_4: "f32[8, 128, 64, 64]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_1: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.reshape.default(relu_4, [8, 2, 64, 64, 64])
    sum_1: "f32[8, 64, 64, 64]" = torch.ops.aten.sum.dim_IntList(view_1, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean: "f32[8, 64, 1, 1]" = torch.ops.aten.mean.dim(sum_1, [2, 3], True);  sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_5: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_16, primals_17, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_536, 1)
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 32, 1, 1]" = var_mean_5[0]
    getitem_13: "f32[1, 32, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_5: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_13)
    mul_35: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
    squeeze_15: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    mul_36: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1);  squeeze_15 = None
    mul_37: "f32[32]" = torch.ops.aten.mul.Tensor(primals_534, 0.9)
    add_27: "f32[32]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_38: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.1428571428571428);  squeeze_17 = None
    mul_39: "f32[32]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[32]" = torch.ops.aten.mul.Tensor(primals_535, 0.9)
    add_28: "f32[32]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
    unsqueeze_21: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
    unsqueeze_23: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_5: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_6: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_5, primals_20, primals_21, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_2: "f32[8, 1, 2, 64]" = torch.ops.aten.reshape.default(convolution_6, [8, 1, 2, -1]);  convolution_6 = None
    permute: "f32[8, 2, 1, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax: "f32[8, 1, 1, 64]" = torch.ops.aten.amax.default(permute, [1], True)
    sub_6: "f32[8, 2, 1, 64]" = torch.ops.aten.sub.Tensor(permute, amax);  permute = amax = None
    exp: "f32[8, 2, 1, 64]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
    sum_2: "f32[8, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp, [1], True)
    div: "f32[8, 2, 1, 64]" = torch.ops.aten.div.Tensor(exp, sum_2);  exp = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_3: "f32[8, 128]" = torch.ops.aten.reshape.default(div, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_4: "f32[8, 128, 1, 1]" = torch.ops.aten.reshape.default(view_3, [8, -1, 1, 1]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_5: "f32[8, 2, 64, 1, 1]" = torch.ops.aten.reshape.default(view_4, [8, 2, 64, 1, 1]);  view_4 = None
    mul_42: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.mul.Tensor(view_1, view_5);  view_1 = view_5 = None
    sum_3: "f32[8, 64, 64, 64]" = torch.ops.aten.sum.dim_IntList(mul_42, [1]);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_7: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(sum_3, primals_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_539, 1)
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 256, 1, 1]" = var_mean_6[0]
    getitem_15: "f32[1, 256, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_6: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_7: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
    mul_43: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = None
    squeeze_18: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_19: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_44: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_45: "f32[256]" = torch.ops.aten.mul.Tensor(primals_537, 0.9)
    add_32: "f32[256]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    squeeze_20: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_46: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.000030518509476);  squeeze_20 = None
    mul_47: "f32[256]" = torch.ops.aten.mul.Tensor(mul_46, 0.1);  mul_46 = None
    mul_48: "f32[256]" = torch.ops.aten.mul.Tensor(primals_538, 0.9)
    add_33: "f32[256]" = torch.ops.aten.add.Tensor(mul_47, mul_48);  mul_47 = mul_48 = None
    unsqueeze_24: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_25: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_49: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_25);  mul_43 = unsqueeze_25 = None
    unsqueeze_26: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_27: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_27);  mul_49 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    convolution_8: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(getitem_6, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_35: "i64[]" = torch.ops.aten.add.Tensor(primals_542, 1)
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 256, 1, 1]" = var_mean_7[0]
    getitem_17: "f32[1, 256, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_36: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_7: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_8: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
    mul_50: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = None
    squeeze_21: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_22: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_51: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(primals_540, 0.9)
    add_37: "f32[256]" = torch.ops.aten.add.Tensor(mul_51, mul_52);  mul_51 = mul_52 = None
    squeeze_23: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_53: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.000030518509476);  squeeze_23 = None
    mul_54: "f32[256]" = torch.ops.aten.mul.Tensor(mul_53, 0.1);  mul_53 = None
    mul_55: "f32[256]" = torch.ops.aten.mul.Tensor(primals_541, 0.9)
    add_38: "f32[256]" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
    unsqueeze_28: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_29: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_56: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_50, unsqueeze_29);  mul_50 = unsqueeze_29 = None
    unsqueeze_30: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_31: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_39: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_31);  mul_56 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_40: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_34, add_39);  add_34 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_6: "f32[8, 256, 64, 64]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_9: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(relu_6, primals_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_545, 1)
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 64, 1, 1]" = var_mean_8[0]
    getitem_19: "f32[1, 64, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_8: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_9: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
    mul_57: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = None
    squeeze_24: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_25: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_58: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_59: "f32[64]" = torch.ops.aten.mul.Tensor(primals_543, 0.9)
    add_43: "f32[64]" = torch.ops.aten.add.Tensor(mul_58, mul_59);  mul_58 = mul_59 = None
    squeeze_26: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.000030518509476);  squeeze_26 = None
    mul_61: "f32[64]" = torch.ops.aten.mul.Tensor(mul_60, 0.1);  mul_60 = None
    mul_62: "f32[64]" = torch.ops.aten.mul.Tensor(primals_544, 0.9)
    add_44: "f32[64]" = torch.ops.aten.add.Tensor(mul_61, mul_62);  mul_61 = mul_62 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_63: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_57, unsqueeze_33);  mul_57 = unsqueeze_33 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_63, unsqueeze_35);  mul_63 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_7: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_45);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_10: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(relu_7, primals_31, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_548, 1)
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1, 1]" = var_mean_9[0]
    getitem_21: "f32[1, 128, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_9: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_10: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
    mul_64: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = None
    squeeze_27: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_28: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_65: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_66: "f32[128]" = torch.ops.aten.mul.Tensor(primals_546, 0.9)
    add_48: "f32[128]" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
    squeeze_29: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_67: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.000030518509476);  squeeze_29 = None
    mul_68: "f32[128]" = torch.ops.aten.mul.Tensor(mul_67, 0.1);  mul_67 = None
    mul_69: "f32[128]" = torch.ops.aten.mul.Tensor(primals_547, 0.9)
    add_49: "f32[128]" = torch.ops.aten.add.Tensor(mul_68, mul_69);  mul_68 = mul_69 = None
    unsqueeze_36: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_37: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_70: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_37);  mul_64 = unsqueeze_37 = None
    unsqueeze_38: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_39: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_70, unsqueeze_39);  mul_70 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_8: "f32[8, 128, 64, 64]" = torch.ops.aten.relu.default(add_50);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_7: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.reshape.default(relu_8, [8, 2, 64, 64, 64])
    sum_4: "f32[8, 64, 64, 64]" = torch.ops.aten.sum.dim_IntList(view_7, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 64, 1, 1]" = torch.ops.aten.mean.dim(sum_4, [2, 3], True);  sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_11: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_34, primals_35, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_51: "i64[]" = torch.ops.aten.add.Tensor(primals_551, 1)
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 32, 1, 1]" = var_mean_10[0]
    getitem_23: "f32[1, 32, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_52: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_10: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_11: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
    mul_71: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = rsqrt_10 = None
    squeeze_30: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    mul_72: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1);  squeeze_30 = None
    mul_73: "f32[32]" = torch.ops.aten.mul.Tensor(primals_549, 0.9)
    add_53: "f32[32]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
    squeeze_32: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_74: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.1428571428571428);  squeeze_32 = None
    mul_75: "f32[32]" = torch.ops.aten.mul.Tensor(mul_74, 0.1);  mul_74 = None
    mul_76: "f32[32]" = torch.ops.aten.mul.Tensor(primals_550, 0.9)
    add_54: "f32[32]" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
    unsqueeze_40: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1)
    unsqueeze_41: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_77: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_41);  mul_71 = unsqueeze_41 = None
    unsqueeze_42: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
    unsqueeze_43: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_55: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_43);  mul_77 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_9: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(add_55);  add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_12: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_9, primals_38, primals_39, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_8: "f32[8, 1, 2, 64]" = torch.ops.aten.reshape.default(convolution_12, [8, 1, 2, -1]);  convolution_12 = None
    permute_1: "f32[8, 2, 1, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_1: "f32[8, 1, 1, 64]" = torch.ops.aten.amax.default(permute_1, [1], True)
    sub_12: "f32[8, 2, 1, 64]" = torch.ops.aten.sub.Tensor(permute_1, amax_1);  permute_1 = amax_1 = None
    exp_1: "f32[8, 2, 1, 64]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
    sum_5: "f32[8, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp_1, [1], True)
    div_1: "f32[8, 2, 1, 64]" = torch.ops.aten.div.Tensor(exp_1, sum_5);  exp_1 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_9: "f32[8, 128]" = torch.ops.aten.reshape.default(div_1, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_10: "f32[8, 128, 1, 1]" = torch.ops.aten.reshape.default(view_9, [8, -1, 1, 1]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_11: "f32[8, 2, 64, 1, 1]" = torch.ops.aten.reshape.default(view_10, [8, 2, 64, 1, 1]);  view_10 = None
    mul_78: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.mul.Tensor(view_7, view_11);  view_7 = view_11 = None
    sum_6: "f32[8, 64, 64, 64]" = torch.ops.aten.sum.dim_IntList(mul_78, [1]);  mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_13: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(sum_6, primals_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_56: "i64[]" = torch.ops.aten.add.Tensor(primals_554, 1)
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 256, 1, 1]" = var_mean_11[0]
    getitem_25: "f32[1, 256, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_11: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_13: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_25)
    mul_79: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = None
    squeeze_33: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_34: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_80: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_81: "f32[256]" = torch.ops.aten.mul.Tensor(primals_552, 0.9)
    add_58: "f32[256]" = torch.ops.aten.add.Tensor(mul_80, mul_81);  mul_80 = mul_81 = None
    squeeze_35: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_82: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.000030518509476);  squeeze_35 = None
    mul_83: "f32[256]" = torch.ops.aten.mul.Tensor(mul_82, 0.1);  mul_82 = None
    mul_84: "f32[256]" = torch.ops.aten.mul.Tensor(primals_553, 0.9)
    add_59: "f32[256]" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
    unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_85: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_45);  mul_79 = unsqueeze_45 = None
    unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_60: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_85, unsqueeze_47);  mul_85 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_61: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_60, relu_6);  add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_10: "f32[8, 256, 64, 64]" = torch.ops.aten.relu.default(add_61);  add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_14: "f32[8, 64, 64, 64]" = torch.ops.aten.convolution.default(relu_10, primals_43, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_62: "i64[]" = torch.ops.aten.add.Tensor(primals_557, 1)
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 64, 1, 1]" = var_mean_12[0]
    getitem_27: "f32[1, 64, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_63: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_12: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_14: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_27)
    mul_86: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_12);  sub_14 = None
    squeeze_36: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_37: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_87: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_88: "f32[64]" = torch.ops.aten.mul.Tensor(primals_555, 0.9)
    add_64: "f32[64]" = torch.ops.aten.add.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
    squeeze_38: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_89: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.000030518509476);  squeeze_38 = None
    mul_90: "f32[64]" = torch.ops.aten.mul.Tensor(mul_89, 0.1);  mul_89 = None
    mul_91: "f32[64]" = torch.ops.aten.mul.Tensor(primals_556, 0.9)
    add_65: "f32[64]" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
    unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
    unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_92: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(mul_86, unsqueeze_49);  mul_86 = unsqueeze_49 = None
    unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
    unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_66: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_51);  mul_92 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_11: "f32[8, 64, 64, 64]" = torch.ops.aten.relu.default(add_66);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_15: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(relu_11, primals_46, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_67: "i64[]" = torch.ops.aten.add.Tensor(primals_560, 1)
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1, 1]" = var_mean_13[0]
    getitem_29: "f32[1, 128, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_68: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_13: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_15: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_29)
    mul_93: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_13);  sub_15 = None
    squeeze_39: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_40: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_94: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_95: "f32[128]" = torch.ops.aten.mul.Tensor(primals_558, 0.9)
    add_69: "f32[128]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
    squeeze_41: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_96: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.000030518509476);  squeeze_41 = None
    mul_97: "f32[128]" = torch.ops.aten.mul.Tensor(mul_96, 0.1);  mul_96 = None
    mul_98: "f32[128]" = torch.ops.aten.mul.Tensor(primals_559, 0.9)
    add_70: "f32[128]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
    unsqueeze_52: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_53: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_99: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_53);  mul_93 = unsqueeze_53 = None
    unsqueeze_54: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_55: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_71: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_99, unsqueeze_55);  mul_99 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_12: "f32[8, 128, 64, 64]" = torch.ops.aten.relu.default(add_71);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_13: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.reshape.default(relu_12, [8, 2, 64, 64, 64])
    sum_7: "f32[8, 64, 64, 64]" = torch.ops.aten.sum.dim_IntList(view_13, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 64, 1, 1]" = torch.ops.aten.mean.dim(sum_7, [2, 3], True);  sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_16: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_49, primals_50, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_72: "i64[]" = torch.ops.aten.add.Tensor(primals_563, 1)
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 32, 1, 1]" = var_mean_14[0]
    getitem_31: "f32[1, 32, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_73: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_14: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_16: "f32[8, 32, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_31)
    mul_100: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_14);  sub_16 = rsqrt_14 = None
    squeeze_42: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    mul_101: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1);  squeeze_42 = None
    mul_102: "f32[32]" = torch.ops.aten.mul.Tensor(primals_561, 0.9)
    add_74: "f32[32]" = torch.ops.aten.add.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
    squeeze_44: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_103: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.1428571428571428);  squeeze_44 = None
    mul_104: "f32[32]" = torch.ops.aten.mul.Tensor(mul_103, 0.1);  mul_103 = None
    mul_105: "f32[32]" = torch.ops.aten.mul.Tensor(primals_562, 0.9)
    add_75: "f32[32]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
    unsqueeze_56: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_57: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_106: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_57);  mul_100 = unsqueeze_57 = None
    unsqueeze_58: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_59: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_76: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_59);  mul_106 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_13: "f32[8, 32, 1, 1]" = torch.ops.aten.relu.default(add_76);  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_17: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(relu_13, primals_53, primals_54, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_14: "f32[8, 1, 2, 64]" = torch.ops.aten.reshape.default(convolution_17, [8, 1, 2, -1]);  convolution_17 = None
    permute_2: "f32[8, 2, 1, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_2: "f32[8, 1, 1, 64]" = torch.ops.aten.amax.default(permute_2, [1], True)
    sub_17: "f32[8, 2, 1, 64]" = torch.ops.aten.sub.Tensor(permute_2, amax_2);  permute_2 = amax_2 = None
    exp_2: "f32[8, 2, 1, 64]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_8: "f32[8, 1, 1, 64]" = torch.ops.aten.sum.dim_IntList(exp_2, [1], True)
    div_2: "f32[8, 2, 1, 64]" = torch.ops.aten.div.Tensor(exp_2, sum_8);  exp_2 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_15: "f32[8, 128]" = torch.ops.aten.reshape.default(div_2, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_16: "f32[8, 128, 1, 1]" = torch.ops.aten.reshape.default(view_15, [8, -1, 1, 1]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_17: "f32[8, 2, 64, 1, 1]" = torch.ops.aten.reshape.default(view_16, [8, 2, 64, 1, 1]);  view_16 = None
    mul_107: "f32[8, 2, 64, 64, 64]" = torch.ops.aten.mul.Tensor(view_13, view_17);  view_13 = view_17 = None
    sum_9: "f32[8, 64, 64, 64]" = torch.ops.aten.sum.dim_IntList(mul_107, [1]);  mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_18: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(sum_9, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_77: "i64[]" = torch.ops.aten.add.Tensor(primals_566, 1)
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 256, 1, 1]" = var_mean_15[0]
    getitem_33: "f32[1, 256, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_78: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_15: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_18: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_33)
    mul_108: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_15);  sub_18 = None
    squeeze_45: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_46: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_109: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_110: "f32[256]" = torch.ops.aten.mul.Tensor(primals_564, 0.9)
    add_79: "f32[256]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    squeeze_47: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_111: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.000030518509476);  squeeze_47 = None
    mul_112: "f32[256]" = torch.ops.aten.mul.Tensor(mul_111, 0.1);  mul_111 = None
    mul_113: "f32[256]" = torch.ops.aten.mul.Tensor(primals_565, 0.9)
    add_80: "f32[256]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    unsqueeze_60: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_61: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_114: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_108, unsqueeze_61);  mul_108 = unsqueeze_61 = None
    unsqueeze_62: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_63: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_81: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_114, unsqueeze_63);  mul_114 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_82: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_81, relu_10);  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_14: "f32[8, 256, 64, 64]" = torch.ops.aten.relu.default(add_82);  add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_19: "f32[8, 128, 64, 64]" = torch.ops.aten.convolution.default(relu_14, primals_58, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_569, 1)
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1, 1]" = var_mean_16[0]
    getitem_35: "f32[1, 128, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_84: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_16: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_19: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_35)
    mul_115: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_16);  sub_19 = None
    squeeze_48: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_49: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_117: "f32[128]" = torch.ops.aten.mul.Tensor(primals_567, 0.9)
    add_85: "f32[128]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    squeeze_50: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_118: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.000030518509476);  squeeze_50 = None
    mul_119: "f32[128]" = torch.ops.aten.mul.Tensor(mul_118, 0.1);  mul_118 = None
    mul_120: "f32[128]" = torch.ops.aten.mul.Tensor(primals_568, 0.9)
    add_86: "f32[128]" = torch.ops.aten.add.Tensor(mul_119, mul_120);  mul_119 = mul_120 = None
    unsqueeze_64: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_65: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_121: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_65);  mul_115 = unsqueeze_65 = None
    unsqueeze_66: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_67: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_87: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_67);  mul_121 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_15: "f32[8, 128, 64, 64]" = torch.ops.aten.relu.default(add_87);  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_20: "f32[8, 256, 64, 64]" = torch.ops.aten.convolution.default(relu_15, primals_61, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_88: "i64[]" = torch.ops.aten.add.Tensor(primals_572, 1)
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 256, 1, 1]" = var_mean_17[0]
    getitem_37: "f32[1, 256, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_89: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_17: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_20: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_37)
    mul_122: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_17);  sub_20 = None
    squeeze_51: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_52: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_123: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_124: "f32[256]" = torch.ops.aten.mul.Tensor(primals_570, 0.9)
    add_90: "f32[256]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    squeeze_53: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_125: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.000030518509476);  squeeze_53 = None
    mul_126: "f32[256]" = torch.ops.aten.mul.Tensor(mul_125, 0.1);  mul_125 = None
    mul_127: "f32[256]" = torch.ops.aten.mul.Tensor(primals_571, 0.9)
    add_91: "f32[256]" = torch.ops.aten.add.Tensor(mul_126, mul_127);  mul_126 = mul_127 = None
    unsqueeze_68: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_69: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_128: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_122, unsqueeze_69);  mul_122 = unsqueeze_69 = None
    unsqueeze_70: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_71: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_92: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_71);  mul_128 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_16: "f32[8, 256, 64, 64]" = torch.ops.aten.relu.default(add_92);  add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_19: "f32[8, 2, 128, 64, 64]" = torch.ops.aten.reshape.default(relu_16, [8, 2, 128, 64, 64])
    sum_10: "f32[8, 128, 64, 64]" = torch.ops.aten.sum.dim_IntList(view_19, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(sum_10, [2, 3], True);  sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_21: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_64, primals_65, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_93: "i64[]" = torch.ops.aten.add.Tensor(primals_575, 1)
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 64, 1, 1]" = var_mean_18[0]
    getitem_39: "f32[1, 64, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_94: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_18: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_21: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_39)
    mul_129: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_18);  sub_21 = rsqrt_18 = None
    squeeze_54: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    mul_130: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1);  squeeze_54 = None
    mul_131: "f32[64]" = torch.ops.aten.mul.Tensor(primals_573, 0.9)
    add_95: "f32[64]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    squeeze_56: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_132: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.1428571428571428);  squeeze_56 = None
    mul_133: "f32[64]" = torch.ops.aten.mul.Tensor(mul_132, 0.1);  mul_132 = None
    mul_134: "f32[64]" = torch.ops.aten.mul.Tensor(primals_574, 0.9)
    add_96: "f32[64]" = torch.ops.aten.add.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None
    unsqueeze_72: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1)
    unsqueeze_73: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_135: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_73);  mul_129 = unsqueeze_73 = None
    unsqueeze_74: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
    unsqueeze_75: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_97: "f32[8, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_135, unsqueeze_75);  mul_135 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_17: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(add_97);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_22: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_17, primals_68, primals_69, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_20: "f32[8, 1, 2, 128]" = torch.ops.aten.reshape.default(convolution_22, [8, 1, 2, -1]);  convolution_22 = None
    permute_3: "f32[8, 2, 1, 128]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_3: "f32[8, 1, 1, 128]" = torch.ops.aten.amax.default(permute_3, [1], True)
    sub_22: "f32[8, 2, 1, 128]" = torch.ops.aten.sub.Tensor(permute_3, amax_3);  permute_3 = amax_3 = None
    exp_3: "f32[8, 2, 1, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_11: "f32[8, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp_3, [1], True)
    div_3: "f32[8, 2, 1, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_11);  exp_3 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_21: "f32[8, 256]" = torch.ops.aten.reshape.default(div_3, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_22: "f32[8, 256, 1, 1]" = torch.ops.aten.reshape.default(view_21, [8, -1, 1, 1]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_23: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.reshape.default(view_22, [8, 2, 128, 1, 1]);  view_22 = None
    mul_136: "f32[8, 2, 128, 64, 64]" = torch.ops.aten.mul.Tensor(view_19, view_23);  view_19 = view_23 = None
    sum_12: "f32[8, 128, 64, 64]" = torch.ops.aten.sum.dim_IntList(mul_136, [1]);  mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    avg_pool2d: "f32[8, 128, 32, 32]" = torch.ops.aten.avg_pool2d.default(sum_12, [3, 3], [2, 2], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_23: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(avg_pool2d, primals_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_98: "i64[]" = torch.ops.aten.add.Tensor(primals_578, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 512, 1, 1]" = var_mean_19[0]
    getitem_41: "f32[1, 512, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_99: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_19: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_23: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_41)
    mul_137: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_19);  sub_23 = None
    squeeze_57: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_58: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_138: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_139: "f32[512]" = torch.ops.aten.mul.Tensor(primals_576, 0.9)
    add_100: "f32[512]" = torch.ops.aten.add.Tensor(mul_138, mul_139);  mul_138 = mul_139 = None
    squeeze_59: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_140: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001220852154804);  squeeze_59 = None
    mul_141: "f32[512]" = torch.ops.aten.mul.Tensor(mul_140, 0.1);  mul_140 = None
    mul_142: "f32[512]" = torch.ops.aten.mul.Tensor(primals_577, 0.9)
    add_101: "f32[512]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    unsqueeze_76: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_77: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_143: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_137, unsqueeze_77);  mul_137 = unsqueeze_77 = None
    unsqueeze_78: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_79: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_102: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_79);  mul_143 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    avg_pool2d_1: "f32[8, 256, 32, 32]" = torch.ops.aten.avg_pool2d.default(relu_14, [2, 2], [2, 2], [0, 0], True, False)
    convolution_24: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(avg_pool2d_1, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_103: "i64[]" = torch.ops.aten.add.Tensor(primals_581, 1)
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1, 1]" = var_mean_20[0]
    getitem_43: "f32[1, 512, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_104: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_20: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_24: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_43)
    mul_144: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_20);  sub_24 = None
    squeeze_60: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_61: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_145: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_146: "f32[512]" = torch.ops.aten.mul.Tensor(primals_579, 0.9)
    add_105: "f32[512]" = torch.ops.aten.add.Tensor(mul_145, mul_146);  mul_145 = mul_146 = None
    squeeze_62: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_147: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001220852154804);  squeeze_62 = None
    mul_148: "f32[512]" = torch.ops.aten.mul.Tensor(mul_147, 0.1);  mul_147 = None
    mul_149: "f32[512]" = torch.ops.aten.mul.Tensor(primals_580, 0.9)
    add_106: "f32[512]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    unsqueeze_80: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_81: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_150: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_144, unsqueeze_81);  mul_144 = unsqueeze_81 = None
    unsqueeze_82: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_83: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_107: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_150, unsqueeze_83);  mul_150 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_108: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_102, add_107);  add_102 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_18: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_108);  add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_25: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(relu_18, primals_76, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_109: "i64[]" = torch.ops.aten.add.Tensor(primals_584, 1)
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 128, 1, 1]" = var_mean_21[0]
    getitem_45: "f32[1, 128, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_110: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_21: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_25: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_45)
    mul_151: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_21);  sub_25 = None
    squeeze_63: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_64: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_152: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_153: "f32[128]" = torch.ops.aten.mul.Tensor(primals_582, 0.9)
    add_111: "f32[128]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
    squeeze_65: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_154: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001220852154804);  squeeze_65 = None
    mul_155: "f32[128]" = torch.ops.aten.mul.Tensor(mul_154, 0.1);  mul_154 = None
    mul_156: "f32[128]" = torch.ops.aten.mul.Tensor(primals_583, 0.9)
    add_112: "f32[128]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    unsqueeze_84: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_85: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_157: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_85);  mul_151 = unsqueeze_85 = None
    unsqueeze_86: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_87: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_113: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_157, unsqueeze_87);  mul_157 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_19: "f32[8, 128, 32, 32]" = torch.ops.aten.relu.default(add_113);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_26: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(relu_19, primals_79, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_114: "i64[]" = torch.ops.aten.add.Tensor(primals_587, 1)
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 256, 1, 1]" = var_mean_22[0]
    getitem_47: "f32[1, 256, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_115: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_22: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_26: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_47)
    mul_158: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_22);  sub_26 = None
    squeeze_66: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_67: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_159: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_160: "f32[256]" = torch.ops.aten.mul.Tensor(primals_585, 0.9)
    add_116: "f32[256]" = torch.ops.aten.add.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    squeeze_68: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_161: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001220852154804);  squeeze_68 = None
    mul_162: "f32[256]" = torch.ops.aten.mul.Tensor(mul_161, 0.1);  mul_161 = None
    mul_163: "f32[256]" = torch.ops.aten.mul.Tensor(primals_586, 0.9)
    add_117: "f32[256]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    unsqueeze_88: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_89: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_164: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_158, unsqueeze_89);  mul_158 = unsqueeze_89 = None
    unsqueeze_90: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_91: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_118: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_91);  mul_164 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_20: "f32[8, 256, 32, 32]" = torch.ops.aten.relu.default(add_118);  add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_25: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.reshape.default(relu_20, [8, 2, 128, 32, 32])
    sum_13: "f32[8, 128, 32, 32]" = torch.ops.aten.sum.dim_IntList(view_25, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(sum_13, [2, 3], True);  sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_27: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_82, primals_83, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_119: "i64[]" = torch.ops.aten.add.Tensor(primals_590, 1)
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 64, 1, 1]" = var_mean_23[0]
    getitem_49: "f32[1, 64, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_120: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_23: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    sub_27: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_49)
    mul_165: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_23);  sub_27 = rsqrt_23 = None
    squeeze_69: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    mul_166: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1);  squeeze_69 = None
    mul_167: "f32[64]" = torch.ops.aten.mul.Tensor(primals_588, 0.9)
    add_121: "f32[64]" = torch.ops.aten.add.Tensor(mul_166, mul_167);  mul_166 = mul_167 = None
    squeeze_71: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_168: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.1428571428571428);  squeeze_71 = None
    mul_169: "f32[64]" = torch.ops.aten.mul.Tensor(mul_168, 0.1);  mul_168 = None
    mul_170: "f32[64]" = torch.ops.aten.mul.Tensor(primals_589, 0.9)
    add_122: "f32[64]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    unsqueeze_92: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
    unsqueeze_93: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_171: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_165, unsqueeze_93);  mul_165 = unsqueeze_93 = None
    unsqueeze_94: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
    unsqueeze_95: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_123: "f32[8, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_171, unsqueeze_95);  mul_171 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_21: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(add_123);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_28: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_21, primals_86, primals_87, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_26: "f32[8, 1, 2, 128]" = torch.ops.aten.reshape.default(convolution_28, [8, 1, 2, -1]);  convolution_28 = None
    permute_4: "f32[8, 2, 1, 128]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_4: "f32[8, 1, 1, 128]" = torch.ops.aten.amax.default(permute_4, [1], True)
    sub_28: "f32[8, 2, 1, 128]" = torch.ops.aten.sub.Tensor(permute_4, amax_4);  permute_4 = amax_4 = None
    exp_4: "f32[8, 2, 1, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_14: "f32[8, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp_4, [1], True)
    div_4: "f32[8, 2, 1, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_14);  exp_4 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_27: "f32[8, 256]" = torch.ops.aten.reshape.default(div_4, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_28: "f32[8, 256, 1, 1]" = torch.ops.aten.reshape.default(view_27, [8, -1, 1, 1]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_29: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.reshape.default(view_28, [8, 2, 128, 1, 1]);  view_28 = None
    mul_172: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.mul.Tensor(view_25, view_29);  view_25 = view_29 = None
    sum_15: "f32[8, 128, 32, 32]" = torch.ops.aten.sum.dim_IntList(mul_172, [1]);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_29: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(sum_15, primals_88, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_124: "i64[]" = torch.ops.aten.add.Tensor(primals_593, 1)
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 512, 1, 1]" = var_mean_24[0]
    getitem_51: "f32[1, 512, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_125: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_24: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
    sub_29: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_51)
    mul_173: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_24);  sub_29 = None
    squeeze_72: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_73: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_174: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_175: "f32[512]" = torch.ops.aten.mul.Tensor(primals_591, 0.9)
    add_126: "f32[512]" = torch.ops.aten.add.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None
    squeeze_74: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_176: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001220852154804);  squeeze_74 = None
    mul_177: "f32[512]" = torch.ops.aten.mul.Tensor(mul_176, 0.1);  mul_176 = None
    mul_178: "f32[512]" = torch.ops.aten.mul.Tensor(primals_592, 0.9)
    add_127: "f32[512]" = torch.ops.aten.add.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None
    unsqueeze_96: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_97: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_179: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_173, unsqueeze_97);  mul_173 = unsqueeze_97 = None
    unsqueeze_98: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_99: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_128: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_179, unsqueeze_99);  mul_179 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_129: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_128, relu_18);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_22: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_129);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_30: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(relu_22, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_130: "i64[]" = torch.ops.aten.add.Tensor(primals_596, 1)
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1, 1]" = var_mean_25[0]
    getitem_53: "f32[1, 128, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_131: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_25: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    sub_30: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_53)
    mul_180: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_25);  sub_30 = None
    squeeze_75: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_76: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_181: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_182: "f32[128]" = torch.ops.aten.mul.Tensor(primals_594, 0.9)
    add_132: "f32[128]" = torch.ops.aten.add.Tensor(mul_181, mul_182);  mul_181 = mul_182 = None
    squeeze_77: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_183: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001220852154804);  squeeze_77 = None
    mul_184: "f32[128]" = torch.ops.aten.mul.Tensor(mul_183, 0.1);  mul_183 = None
    mul_185: "f32[128]" = torch.ops.aten.mul.Tensor(primals_595, 0.9)
    add_133: "f32[128]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
    unsqueeze_100: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_101: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_186: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_101);  mul_180 = unsqueeze_101 = None
    unsqueeze_102: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_103: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_134: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_186, unsqueeze_103);  mul_186 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_23: "f32[8, 128, 32, 32]" = torch.ops.aten.relu.default(add_134);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_31: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(relu_23, primals_94, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_135: "i64[]" = torch.ops.aten.add.Tensor(primals_599, 1)
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 256, 1, 1]" = var_mean_26[0]
    getitem_55: "f32[1, 256, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_136: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_26: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_31: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_55)
    mul_187: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_26);  sub_31 = None
    squeeze_78: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_79: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_188: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_189: "f32[256]" = torch.ops.aten.mul.Tensor(primals_597, 0.9)
    add_137: "f32[256]" = torch.ops.aten.add.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    squeeze_80: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_190: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001220852154804);  squeeze_80 = None
    mul_191: "f32[256]" = torch.ops.aten.mul.Tensor(mul_190, 0.1);  mul_190 = None
    mul_192: "f32[256]" = torch.ops.aten.mul.Tensor(primals_598, 0.9)
    add_138: "f32[256]" = torch.ops.aten.add.Tensor(mul_191, mul_192);  mul_191 = mul_192 = None
    unsqueeze_104: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_105: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_193: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_105);  mul_187 = unsqueeze_105 = None
    unsqueeze_106: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_107: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_139: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_193, unsqueeze_107);  mul_193 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_24: "f32[8, 256, 32, 32]" = torch.ops.aten.relu.default(add_139);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_31: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.reshape.default(relu_24, [8, 2, 128, 32, 32])
    sum_16: "f32[8, 128, 32, 32]" = torch.ops.aten.sum.dim_IntList(view_31, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(sum_16, [2, 3], True);  sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_32: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_97, primals_98, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_140: "i64[]" = torch.ops.aten.add.Tensor(primals_602, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 64, 1, 1]" = var_mean_27[0]
    getitem_57: "f32[1, 64, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_141: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_27: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_32: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_57)
    mul_194: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_27);  sub_32 = rsqrt_27 = None
    squeeze_81: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    mul_195: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1);  squeeze_81 = None
    mul_196: "f32[64]" = torch.ops.aten.mul.Tensor(primals_600, 0.9)
    add_142: "f32[64]" = torch.ops.aten.add.Tensor(mul_195, mul_196);  mul_195 = mul_196 = None
    squeeze_83: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_197: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.1428571428571428);  squeeze_83 = None
    mul_198: "f32[64]" = torch.ops.aten.mul.Tensor(mul_197, 0.1);  mul_197 = None
    mul_199: "f32[64]" = torch.ops.aten.mul.Tensor(primals_601, 0.9)
    add_143: "f32[64]" = torch.ops.aten.add.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None
    unsqueeze_108: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1)
    unsqueeze_109: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_200: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_194, unsqueeze_109);  mul_194 = unsqueeze_109 = None
    unsqueeze_110: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1);  primals_100 = None
    unsqueeze_111: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_144: "f32[8, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_200, unsqueeze_111);  mul_200 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_25: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(add_144);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_33: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_25, primals_101, primals_102, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_32: "f32[8, 1, 2, 128]" = torch.ops.aten.reshape.default(convolution_33, [8, 1, 2, -1]);  convolution_33 = None
    permute_5: "f32[8, 2, 1, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_5: "f32[8, 1, 1, 128]" = torch.ops.aten.amax.default(permute_5, [1], True)
    sub_33: "f32[8, 2, 1, 128]" = torch.ops.aten.sub.Tensor(permute_5, amax_5);  permute_5 = amax_5 = None
    exp_5: "f32[8, 2, 1, 128]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_17: "f32[8, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp_5, [1], True)
    div_5: "f32[8, 2, 1, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_17);  exp_5 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_33: "f32[8, 256]" = torch.ops.aten.reshape.default(div_5, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_34: "f32[8, 256, 1, 1]" = torch.ops.aten.reshape.default(view_33, [8, -1, 1, 1]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_35: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.reshape.default(view_34, [8, 2, 128, 1, 1]);  view_34 = None
    mul_201: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.mul.Tensor(view_31, view_35);  view_31 = view_35 = None
    sum_18: "f32[8, 128, 32, 32]" = torch.ops.aten.sum.dim_IntList(mul_201, [1]);  mul_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_34: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(sum_18, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_145: "i64[]" = torch.ops.aten.add.Tensor(primals_605, 1)
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 512, 1, 1]" = var_mean_28[0]
    getitem_59: "f32[1, 512, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_146: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_28: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_34: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_59)
    mul_202: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_28);  sub_34 = None
    squeeze_84: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_85: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_203: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_204: "f32[512]" = torch.ops.aten.mul.Tensor(primals_603, 0.9)
    add_147: "f32[512]" = torch.ops.aten.add.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
    squeeze_86: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_205: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0001220852154804);  squeeze_86 = None
    mul_206: "f32[512]" = torch.ops.aten.mul.Tensor(mul_205, 0.1);  mul_205 = None
    mul_207: "f32[512]" = torch.ops.aten.mul.Tensor(primals_604, 0.9)
    add_148: "f32[512]" = torch.ops.aten.add.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
    unsqueeze_112: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
    unsqueeze_113: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_208: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_113);  mul_202 = unsqueeze_113 = None
    unsqueeze_114: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
    unsqueeze_115: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_149: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_208, unsqueeze_115);  mul_208 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_150: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_149, relu_22);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_26: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_150);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_35: "f32[8, 128, 32, 32]" = torch.ops.aten.convolution.default(relu_26, primals_106, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_151: "i64[]" = torch.ops.aten.add.Tensor(primals_608, 1)
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1, 1]" = var_mean_29[0]
    getitem_61: "f32[1, 128, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_152: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_29: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    sub_35: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_61)
    mul_209: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_29);  sub_35 = None
    squeeze_87: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_88: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_210: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_211: "f32[128]" = torch.ops.aten.mul.Tensor(primals_606, 0.9)
    add_153: "f32[128]" = torch.ops.aten.add.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
    squeeze_89: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_212: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0001220852154804);  squeeze_89 = None
    mul_213: "f32[128]" = torch.ops.aten.mul.Tensor(mul_212, 0.1);  mul_212 = None
    mul_214: "f32[128]" = torch.ops.aten.mul.Tensor(primals_607, 0.9)
    add_154: "f32[128]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
    unsqueeze_116: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_117: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_215: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_117);  mul_209 = unsqueeze_117 = None
    unsqueeze_118: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_119: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_155: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_119);  mul_215 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_27: "f32[8, 128, 32, 32]" = torch.ops.aten.relu.default(add_155);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_36: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(relu_27, primals_109, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_156: "i64[]" = torch.ops.aten.add.Tensor(primals_611, 1)
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 256, 1, 1]" = var_mean_30[0]
    getitem_63: "f32[1, 256, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_157: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_30: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    sub_36: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_63)
    mul_216: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_30);  sub_36 = None
    squeeze_90: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_91: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_217: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_218: "f32[256]" = torch.ops.aten.mul.Tensor(primals_609, 0.9)
    add_158: "f32[256]" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
    squeeze_92: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_219: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0001220852154804);  squeeze_92 = None
    mul_220: "f32[256]" = torch.ops.aten.mul.Tensor(mul_219, 0.1);  mul_219 = None
    mul_221: "f32[256]" = torch.ops.aten.mul.Tensor(primals_610, 0.9)
    add_159: "f32[256]" = torch.ops.aten.add.Tensor(mul_220, mul_221);  mul_220 = mul_221 = None
    unsqueeze_120: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_121: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_222: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_216, unsqueeze_121);  mul_216 = unsqueeze_121 = None
    unsqueeze_122: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_123: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_160: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_222, unsqueeze_123);  mul_222 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_28: "f32[8, 256, 32, 32]" = torch.ops.aten.relu.default(add_160);  add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_37: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.reshape.default(relu_28, [8, 2, 128, 32, 32])
    sum_19: "f32[8, 128, 32, 32]" = torch.ops.aten.sum.dim_IntList(view_37, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 128, 1, 1]" = torch.ops.aten.mean.dim(sum_19, [2, 3], True);  sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_37: "f32[8, 64, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_112, primals_113, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_161: "i64[]" = torch.ops.aten.add.Tensor(primals_614, 1)
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 64, 1, 1]" = var_mean_31[0]
    getitem_65: "f32[1, 64, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_162: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_31: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
    sub_37: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_65)
    mul_223: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_31);  sub_37 = rsqrt_31 = None
    squeeze_93: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    mul_224: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1);  squeeze_93 = None
    mul_225: "f32[64]" = torch.ops.aten.mul.Tensor(primals_612, 0.9)
    add_163: "f32[64]" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
    squeeze_95: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_226: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.1428571428571428);  squeeze_95 = None
    mul_227: "f32[64]" = torch.ops.aten.mul.Tensor(mul_226, 0.1);  mul_226 = None
    mul_228: "f32[64]" = torch.ops.aten.mul.Tensor(primals_613, 0.9)
    add_164: "f32[64]" = torch.ops.aten.add.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
    unsqueeze_124: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1)
    unsqueeze_125: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_229: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_125);  mul_223 = unsqueeze_125 = None
    unsqueeze_126: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
    unsqueeze_127: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_165: "f32[8, 64, 1, 1]" = torch.ops.aten.add.Tensor(mul_229, unsqueeze_127);  mul_229 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_29: "f32[8, 64, 1, 1]" = torch.ops.aten.relu.default(add_165);  add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_38: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(relu_29, primals_116, primals_117, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_38: "f32[8, 1, 2, 128]" = torch.ops.aten.reshape.default(convolution_38, [8, 1, 2, -1]);  convolution_38 = None
    permute_6: "f32[8, 2, 1, 128]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_6: "f32[8, 1, 1, 128]" = torch.ops.aten.amax.default(permute_6, [1], True)
    sub_38: "f32[8, 2, 1, 128]" = torch.ops.aten.sub.Tensor(permute_6, amax_6);  permute_6 = amax_6 = None
    exp_6: "f32[8, 2, 1, 128]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_20: "f32[8, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp_6, [1], True)
    div_6: "f32[8, 2, 1, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_20);  exp_6 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_39: "f32[8, 256]" = torch.ops.aten.reshape.default(div_6, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_40: "f32[8, 256, 1, 1]" = torch.ops.aten.reshape.default(view_39, [8, -1, 1, 1]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_41: "f32[8, 2, 128, 1, 1]" = torch.ops.aten.reshape.default(view_40, [8, 2, 128, 1, 1]);  view_40 = None
    mul_230: "f32[8, 2, 128, 32, 32]" = torch.ops.aten.mul.Tensor(view_37, view_41);  view_37 = view_41 = None
    sum_21: "f32[8, 128, 32, 32]" = torch.ops.aten.sum.dim_IntList(mul_230, [1]);  mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_39: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(sum_21, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_166: "i64[]" = torch.ops.aten.add.Tensor(primals_617, 1)
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 512, 1, 1]" = var_mean_32[0]
    getitem_67: "f32[1, 512, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_167: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_32: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    sub_39: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_67)
    mul_231: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_32);  sub_39 = None
    squeeze_96: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_97: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_232: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_233: "f32[512]" = torch.ops.aten.mul.Tensor(primals_615, 0.9)
    add_168: "f32[512]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_98: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_234: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0001220852154804);  squeeze_98 = None
    mul_235: "f32[512]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[512]" = torch.ops.aten.mul.Tensor(primals_616, 0.9)
    add_169: "f32[512]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_128: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_129: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_237: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_129);  mul_231 = unsqueeze_129 = None
    unsqueeze_130: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_131: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_170: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_131);  mul_237 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_171: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(add_170, relu_26);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_30: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_171);  add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_40: "f32[8, 256, 32, 32]" = torch.ops.aten.convolution.default(relu_30, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_172: "i64[]" = torch.ops.aten.add.Tensor(primals_620, 1)
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 256, 1, 1]" = var_mean_33[0]
    getitem_69: "f32[1, 256, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_173: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_33: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    sub_40: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_69)
    mul_238: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_33);  sub_40 = None
    squeeze_99: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_100: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_239: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_240: "f32[256]" = torch.ops.aten.mul.Tensor(primals_618, 0.9)
    add_174: "f32[256]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_101: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_241: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0001220852154804);  squeeze_101 = None
    mul_242: "f32[256]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[256]" = torch.ops.aten.mul.Tensor(primals_619, 0.9)
    add_175: "f32[256]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_132: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
    unsqueeze_133: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_244: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_133);  mul_238 = unsqueeze_133 = None
    unsqueeze_134: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
    unsqueeze_135: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_176: "f32[8, 256, 32, 32]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_135);  mul_244 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_31: "f32[8, 256, 32, 32]" = torch.ops.aten.relu.default(add_176);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_41: "f32[8, 512, 32, 32]" = torch.ops.aten.convolution.default(relu_31, primals_124, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_177: "i64[]" = torch.ops.aten.add.Tensor(primals_623, 1)
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 512, 1, 1]" = var_mean_34[0]
    getitem_71: "f32[1, 512, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_178: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_34: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_41: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_71)
    mul_245: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_34);  sub_41 = None
    squeeze_102: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_103: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_246: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_247: "f32[512]" = torch.ops.aten.mul.Tensor(primals_621, 0.9)
    add_179: "f32[512]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_104: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_248: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0001220852154804);  squeeze_104 = None
    mul_249: "f32[512]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[512]" = torch.ops.aten.mul.Tensor(primals_622, 0.9)
    add_180: "f32[512]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_136: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_137: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_251: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_137);  mul_245 = unsqueeze_137 = None
    unsqueeze_138: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_139: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_181: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_139);  mul_251 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_32: "f32[8, 512, 32, 32]" = torch.ops.aten.relu.default(add_181);  add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_43: "f32[8, 2, 256, 32, 32]" = torch.ops.aten.reshape.default(relu_32, [8, 2, 256, 32, 32])
    sum_22: "f32[8, 256, 32, 32]" = torch.ops.aten.sum.dim_IntList(view_43, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_22, [2, 3], True);  sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_42: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_127, primals_128, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_182: "i64[]" = torch.ops.aten.add.Tensor(primals_626, 1)
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1, 1]" = var_mean_35[0]
    getitem_73: "f32[1, 128, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_183: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_35: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    sub_42: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_73)
    mul_252: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_35);  sub_42 = rsqrt_35 = None
    squeeze_105: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    mul_253: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1);  squeeze_105 = None
    mul_254: "f32[128]" = torch.ops.aten.mul.Tensor(primals_624, 0.9)
    add_184: "f32[128]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    squeeze_107: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_255: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.1428571428571428);  squeeze_107 = None
    mul_256: "f32[128]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
    mul_257: "f32[128]" = torch.ops.aten.mul.Tensor(primals_625, 0.9)
    add_185: "f32[128]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    unsqueeze_140: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1)
    unsqueeze_141: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_258: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_141);  mul_252 = unsqueeze_141 = None
    unsqueeze_142: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1);  primals_130 = None
    unsqueeze_143: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_186: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_143);  mul_258 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_33: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_186);  add_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_43: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_33, primals_131, primals_132, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_44: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_43, [8, 1, 2, -1]);  convolution_43 = None
    permute_7: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_7: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_7, [1], True)
    sub_43: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_7, amax_7);  permute_7 = amax_7 = None
    exp_7: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_23: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_7, [1], True)
    div_7: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_7, sum_23);  exp_7 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_45: "f32[8, 512]" = torch.ops.aten.reshape.default(div_7, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_46: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_45, [8, -1, 1, 1]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_47: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_46, [8, 2, 256, 1, 1]);  view_46 = None
    mul_259: "f32[8, 2, 256, 32, 32]" = torch.ops.aten.mul.Tensor(view_43, view_47);  view_43 = view_47 = None
    sum_24: "f32[8, 256, 32, 32]" = torch.ops.aten.sum.dim_IntList(mul_259, [1]);  mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    avg_pool2d_2: "f32[8, 256, 16, 16]" = torch.ops.aten.avg_pool2d.default(sum_24, [3, 3], [2, 2], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_44: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(avg_pool2d_2, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_187: "i64[]" = torch.ops.aten.add.Tensor(primals_629, 1)
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 1024, 1, 1]" = var_mean_36[0]
    getitem_75: "f32[1, 1024, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_188: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_36: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    sub_44: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_75)
    mul_260: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_36);  sub_44 = None
    squeeze_108: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_109: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_261: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_262: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_627, 0.9)
    add_189: "f32[1024]" = torch.ops.aten.add.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
    squeeze_110: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_263: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0004885197850513);  squeeze_110 = None
    mul_264: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_263, 0.1);  mul_263 = None
    mul_265: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_628, 0.9)
    add_190: "f32[1024]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
    unsqueeze_144: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
    unsqueeze_145: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_266: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_260, unsqueeze_145);  mul_260 = unsqueeze_145 = None
    unsqueeze_146: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
    unsqueeze_147: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_191: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_266, unsqueeze_147);  mul_266 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    avg_pool2d_3: "f32[8, 512, 16, 16]" = torch.ops.aten.avg_pool2d.default(relu_30, [2, 2], [2, 2], [0, 0], True, False)
    convolution_45: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(avg_pool2d_3, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_192: "i64[]" = torch.ops.aten.add.Tensor(primals_632, 1)
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 1024, 1, 1]" = var_mean_37[0]
    getitem_77: "f32[1, 1024, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_193: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_37: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    sub_45: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_77)
    mul_267: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_37);  sub_45 = None
    squeeze_111: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_112: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_268: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_269: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_630, 0.9)
    add_194: "f32[1024]" = torch.ops.aten.add.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
    squeeze_113: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_270: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0004885197850513);  squeeze_113 = None
    mul_271: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_270, 0.1);  mul_270 = None
    mul_272: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_631, 0.9)
    add_195: "f32[1024]" = torch.ops.aten.add.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
    unsqueeze_148: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
    unsqueeze_149: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_273: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_267, unsqueeze_149);  mul_267 = unsqueeze_149 = None
    unsqueeze_150: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
    unsqueeze_151: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_196: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_273, unsqueeze_151);  mul_273 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_197: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_191, add_196);  add_191 = add_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_34: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_197);  add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_46: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_34, primals_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_198: "i64[]" = torch.ops.aten.add.Tensor(primals_635, 1)
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 256, 1, 1]" = var_mean_38[0]
    getitem_79: "f32[1, 256, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_199: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_38: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_46: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_79)
    mul_274: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_38);  sub_46 = None
    squeeze_114: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_115: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_275: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_276: "f32[256]" = torch.ops.aten.mul.Tensor(primals_633, 0.9)
    add_200: "f32[256]" = torch.ops.aten.add.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    squeeze_116: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_277: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0004885197850513);  squeeze_116 = None
    mul_278: "f32[256]" = torch.ops.aten.mul.Tensor(mul_277, 0.1);  mul_277 = None
    mul_279: "f32[256]" = torch.ops.aten.mul.Tensor(primals_634, 0.9)
    add_201: "f32[256]" = torch.ops.aten.add.Tensor(mul_278, mul_279);  mul_278 = mul_279 = None
    unsqueeze_152: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
    unsqueeze_153: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_280: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_153);  mul_274 = unsqueeze_153 = None
    unsqueeze_154: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
    unsqueeze_155: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_202: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_280, unsqueeze_155);  mul_280 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_35: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_202);  add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_47: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_35, primals_142, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_203: "i64[]" = torch.ops.aten.add.Tensor(primals_638, 1)
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 512, 1, 1]" = var_mean_39[0]
    getitem_81: "f32[1, 512, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_204: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_39: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
    sub_47: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_81)
    mul_281: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_39);  sub_47 = None
    squeeze_117: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_118: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_282: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_283: "f32[512]" = torch.ops.aten.mul.Tensor(primals_636, 0.9)
    add_205: "f32[512]" = torch.ops.aten.add.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
    squeeze_119: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_284: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0004885197850513);  squeeze_119 = None
    mul_285: "f32[512]" = torch.ops.aten.mul.Tensor(mul_284, 0.1);  mul_284 = None
    mul_286: "f32[512]" = torch.ops.aten.mul.Tensor(primals_637, 0.9)
    add_206: "f32[512]" = torch.ops.aten.add.Tensor(mul_285, mul_286);  mul_285 = mul_286 = None
    unsqueeze_156: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_157: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_287: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_281, unsqueeze_157);  mul_281 = unsqueeze_157 = None
    unsqueeze_158: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_159: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_207: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_287, unsqueeze_159);  mul_287 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_36: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_207);  add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_49: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_36, [8, 2, 256, 16, 16])
    sum_25: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_49, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_8: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_25, [2, 3], True);  sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_48: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_145, primals_146, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_208: "i64[]" = torch.ops.aten.add.Tensor(primals_641, 1)
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1, 1]" = var_mean_40[0]
    getitem_83: "f32[1, 128, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_209: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_40: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
    sub_48: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_83)
    mul_288: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_40);  sub_48 = rsqrt_40 = None
    squeeze_120: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    mul_289: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1);  squeeze_120 = None
    mul_290: "f32[128]" = torch.ops.aten.mul.Tensor(primals_639, 0.9)
    add_210: "f32[128]" = torch.ops.aten.add.Tensor(mul_289, mul_290);  mul_289 = mul_290 = None
    squeeze_122: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_291: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.1428571428571428);  squeeze_122 = None
    mul_292: "f32[128]" = torch.ops.aten.mul.Tensor(mul_291, 0.1);  mul_291 = None
    mul_293: "f32[128]" = torch.ops.aten.mul.Tensor(primals_640, 0.9)
    add_211: "f32[128]" = torch.ops.aten.add.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
    unsqueeze_160: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1)
    unsqueeze_161: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_294: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_288, unsqueeze_161);  mul_288 = unsqueeze_161 = None
    unsqueeze_162: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_148, -1);  primals_148 = None
    unsqueeze_163: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_212: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_294, unsqueeze_163);  mul_294 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_37: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_212);  add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_49: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_37, primals_149, primals_150, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_50: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_49, [8, 1, 2, -1]);  convolution_49 = None
    permute_8: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_8: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_8, [1], True)
    sub_49: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_8, amax_8);  permute_8 = amax_8 = None
    exp_8: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_26: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_8, [1], True)
    div_8: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_8, sum_26);  exp_8 = sum_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_51: "f32[8, 512]" = torch.ops.aten.reshape.default(div_8, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_52: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_51, [8, -1, 1, 1]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_53: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_52, [8, 2, 256, 1, 1]);  view_52 = None
    mul_295: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_49, view_53);  view_49 = view_53 = None
    sum_27: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_295, [1]);  mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_50: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_27, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_213: "i64[]" = torch.ops.aten.add.Tensor(primals_644, 1)
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 1024, 1, 1]" = var_mean_41[0]
    getitem_85: "f32[1, 1024, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_214: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_41: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
    sub_50: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_85)
    mul_296: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_41);  sub_50 = None
    squeeze_123: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_124: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_297: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_298: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_642, 0.9)
    add_215: "f32[1024]" = torch.ops.aten.add.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
    squeeze_125: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_299: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0004885197850513);  squeeze_125 = None
    mul_300: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_299, 0.1);  mul_299 = None
    mul_301: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_643, 0.9)
    add_216: "f32[1024]" = torch.ops.aten.add.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    unsqueeze_164: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1)
    unsqueeze_165: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_302: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_165);  mul_296 = unsqueeze_165 = None
    unsqueeze_166: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
    unsqueeze_167: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_217: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_167);  mul_302 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_218: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_217, relu_34);  add_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_38: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_218);  add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_51: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_38, primals_154, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_219: "i64[]" = torch.ops.aten.add.Tensor(primals_647, 1)
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 256, 1, 1]" = var_mean_42[0]
    getitem_87: "f32[1, 256, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_220: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_42: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
    sub_51: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_87)
    mul_303: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_42);  sub_51 = None
    squeeze_126: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_127: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_304: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_305: "f32[256]" = torch.ops.aten.mul.Tensor(primals_645, 0.9)
    add_221: "f32[256]" = torch.ops.aten.add.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
    squeeze_128: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_306: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0004885197850513);  squeeze_128 = None
    mul_307: "f32[256]" = torch.ops.aten.mul.Tensor(mul_306, 0.1);  mul_306 = None
    mul_308: "f32[256]" = torch.ops.aten.mul.Tensor(primals_646, 0.9)
    add_222: "f32[256]" = torch.ops.aten.add.Tensor(mul_307, mul_308);  mul_307 = mul_308 = None
    unsqueeze_168: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
    unsqueeze_169: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_309: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_303, unsqueeze_169);  mul_303 = unsqueeze_169 = None
    unsqueeze_170: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
    unsqueeze_171: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_223: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_309, unsqueeze_171);  mul_309 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_39: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_223);  add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_52: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_39, primals_157, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_224: "i64[]" = torch.ops.aten.add.Tensor(primals_650, 1)
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 512, 1, 1]" = var_mean_43[0]
    getitem_89: "f32[1, 512, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_225: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_43: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
    sub_52: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_89)
    mul_310: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_43);  sub_52 = None
    squeeze_129: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_130: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_311: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_312: "f32[512]" = torch.ops.aten.mul.Tensor(primals_648, 0.9)
    add_226: "f32[512]" = torch.ops.aten.add.Tensor(mul_311, mul_312);  mul_311 = mul_312 = None
    squeeze_131: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_313: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0004885197850513);  squeeze_131 = None
    mul_314: "f32[512]" = torch.ops.aten.mul.Tensor(mul_313, 0.1);  mul_313 = None
    mul_315: "f32[512]" = torch.ops.aten.mul.Tensor(primals_649, 0.9)
    add_227: "f32[512]" = torch.ops.aten.add.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
    unsqueeze_172: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_158, -1)
    unsqueeze_173: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_316: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_173);  mul_310 = unsqueeze_173 = None
    unsqueeze_174: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1);  primals_159 = None
    unsqueeze_175: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_228: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_316, unsqueeze_175);  mul_316 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_40: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_228);  add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_55: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_40, [8, 2, 256, 16, 16])
    sum_28: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_55, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_9: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_28, [2, 3], True);  sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_53: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_160, primals_161, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_229: "i64[]" = torch.ops.aten.add.Tensor(primals_653, 1)
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 128, 1, 1]" = var_mean_44[0]
    getitem_91: "f32[1, 128, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_230: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
    rsqrt_44: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_230);  add_230 = None
    sub_53: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_91)
    mul_317: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_44);  sub_53 = rsqrt_44 = None
    squeeze_132: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    mul_318: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1);  squeeze_132 = None
    mul_319: "f32[128]" = torch.ops.aten.mul.Tensor(primals_651, 0.9)
    add_231: "f32[128]" = torch.ops.aten.add.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
    squeeze_134: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_320: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.1428571428571428);  squeeze_134 = None
    mul_321: "f32[128]" = torch.ops.aten.mul.Tensor(mul_320, 0.1);  mul_320 = None
    mul_322: "f32[128]" = torch.ops.aten.mul.Tensor(primals_652, 0.9)
    add_232: "f32[128]" = torch.ops.aten.add.Tensor(mul_321, mul_322);  mul_321 = mul_322 = None
    unsqueeze_176: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1)
    unsqueeze_177: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_323: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_317, unsqueeze_177);  mul_317 = unsqueeze_177 = None
    unsqueeze_178: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1);  primals_163 = None
    unsqueeze_179: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_233: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_323, unsqueeze_179);  mul_323 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_41: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_233);  add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_54: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_41, primals_164, primals_165, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_56: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_54, [8, 1, 2, -1]);  convolution_54 = None
    permute_9: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_9: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_9, [1], True)
    sub_54: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_9, amax_9);  permute_9 = amax_9 = None
    exp_9: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_54);  sub_54 = None
    sum_29: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_9, [1], True)
    div_9: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_9, sum_29);  exp_9 = sum_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_57: "f32[8, 512]" = torch.ops.aten.reshape.default(div_9, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_58: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_57, [8, -1, 1, 1]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_59: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_58, [8, 2, 256, 1, 1]);  view_58 = None
    mul_324: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_55, view_59);  view_55 = view_59 = None
    sum_30: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_324, [1]);  mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_55: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_30, primals_166, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_234: "i64[]" = torch.ops.aten.add.Tensor(primals_656, 1)
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[1, 1024, 1, 1]" = var_mean_45[0]
    getitem_93: "f32[1, 1024, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_235: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
    rsqrt_45: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    sub_55: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_93)
    mul_325: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_45);  sub_55 = None
    squeeze_135: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
    squeeze_136: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_326: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_327: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_654, 0.9)
    add_236: "f32[1024]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    squeeze_137: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
    mul_328: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0004885197850513);  squeeze_137 = None
    mul_329: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_328, 0.1);  mul_328 = None
    mul_330: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_655, 0.9)
    add_237: "f32[1024]" = torch.ops.aten.add.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
    unsqueeze_180: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1)
    unsqueeze_181: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_331: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_181);  mul_325 = unsqueeze_181 = None
    unsqueeze_182: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1);  primals_168 = None
    unsqueeze_183: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_238: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_331, unsqueeze_183);  mul_331 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_239: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_238, relu_38);  add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_42: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_239);  add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_56: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_42, primals_169, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_240: "i64[]" = torch.ops.aten.add.Tensor(primals_659, 1)
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 256, 1, 1]" = var_mean_46[0]
    getitem_95: "f32[1, 256, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_241: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
    rsqrt_46: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
    sub_56: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_95)
    mul_332: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_46);  sub_56 = None
    squeeze_138: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_139: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_333: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_334: "f32[256]" = torch.ops.aten.mul.Tensor(primals_657, 0.9)
    add_242: "f32[256]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    squeeze_140: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_335: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0004885197850513);  squeeze_140 = None
    mul_336: "f32[256]" = torch.ops.aten.mul.Tensor(mul_335, 0.1);  mul_335 = None
    mul_337: "f32[256]" = torch.ops.aten.mul.Tensor(primals_658, 0.9)
    add_243: "f32[256]" = torch.ops.aten.add.Tensor(mul_336, mul_337);  mul_336 = mul_337 = None
    unsqueeze_184: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_170, -1)
    unsqueeze_185: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_338: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_332, unsqueeze_185);  mul_332 = unsqueeze_185 = None
    unsqueeze_186: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_171, -1);  primals_171 = None
    unsqueeze_187: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_244: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_338, unsqueeze_187);  mul_338 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_43: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_244);  add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_57: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_43, primals_172, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_245: "i64[]" = torch.ops.aten.add.Tensor(primals_662, 1)
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 512, 1, 1]" = var_mean_47[0]
    getitem_97: "f32[1, 512, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_246: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_47: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
    sub_57: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_97)
    mul_339: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_47);  sub_57 = None
    squeeze_141: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_142: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_340: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_341: "f32[512]" = torch.ops.aten.mul.Tensor(primals_660, 0.9)
    add_247: "f32[512]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    squeeze_143: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_342: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0004885197850513);  squeeze_143 = None
    mul_343: "f32[512]" = torch.ops.aten.mul.Tensor(mul_342, 0.1);  mul_342 = None
    mul_344: "f32[512]" = torch.ops.aten.mul.Tensor(primals_661, 0.9)
    add_248: "f32[512]" = torch.ops.aten.add.Tensor(mul_343, mul_344);  mul_343 = mul_344 = None
    unsqueeze_188: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_173, -1)
    unsqueeze_189: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_345: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_339, unsqueeze_189);  mul_339 = unsqueeze_189 = None
    unsqueeze_190: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_174, -1);  primals_174 = None
    unsqueeze_191: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_249: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_345, unsqueeze_191);  mul_345 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_44: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_249);  add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_61: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_44, [8, 2, 256, 16, 16])
    sum_31: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_61, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_10: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_31, [2, 3], True);  sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_58: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_175, primals_176, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_250: "i64[]" = torch.ops.aten.add.Tensor(primals_665, 1)
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 128, 1, 1]" = var_mean_48[0]
    getitem_99: "f32[1, 128, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_251: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
    rsqrt_48: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
    sub_58: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_99)
    mul_346: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_48);  sub_58 = rsqrt_48 = None
    squeeze_144: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    mul_347: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1);  squeeze_144 = None
    mul_348: "f32[128]" = torch.ops.aten.mul.Tensor(primals_663, 0.9)
    add_252: "f32[128]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    squeeze_146: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_349: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.1428571428571428);  squeeze_146 = None
    mul_350: "f32[128]" = torch.ops.aten.mul.Tensor(mul_349, 0.1);  mul_349 = None
    mul_351: "f32[128]" = torch.ops.aten.mul.Tensor(primals_664, 0.9)
    add_253: "f32[128]" = torch.ops.aten.add.Tensor(mul_350, mul_351);  mul_350 = mul_351 = None
    unsqueeze_192: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_177, -1)
    unsqueeze_193: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_352: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_193);  mul_346 = unsqueeze_193 = None
    unsqueeze_194: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_178, -1);  primals_178 = None
    unsqueeze_195: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_254: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_352, unsqueeze_195);  mul_352 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_45: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_254);  add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_59: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_45, primals_179, primals_180, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_62: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_59, [8, 1, 2, -1]);  convolution_59 = None
    permute_10: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_10: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_10, [1], True)
    sub_59: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_10, amax_10);  permute_10 = amax_10 = None
    exp_10: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_59);  sub_59 = None
    sum_32: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_10, [1], True)
    div_10: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_10, sum_32);  exp_10 = sum_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_63: "f32[8, 512]" = torch.ops.aten.reshape.default(div_10, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_64: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_63, [8, -1, 1, 1]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_65: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_64, [8, 2, 256, 1, 1]);  view_64 = None
    mul_353: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_61, view_65);  view_61 = view_65 = None
    sum_33: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_353, [1]);  mul_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_60: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_33, primals_181, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_255: "i64[]" = torch.ops.aten.add.Tensor(primals_668, 1)
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 1024, 1, 1]" = var_mean_49[0]
    getitem_101: "f32[1, 1024, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_256: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
    rsqrt_49: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
    sub_60: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_101)
    mul_354: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_49);  sub_60 = None
    squeeze_147: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_148: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_355: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_356: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_666, 0.9)
    add_257: "f32[1024]" = torch.ops.aten.add.Tensor(mul_355, mul_356);  mul_355 = mul_356 = None
    squeeze_149: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_357: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0004885197850513);  squeeze_149 = None
    mul_358: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_357, 0.1);  mul_357 = None
    mul_359: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_667, 0.9)
    add_258: "f32[1024]" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    unsqueeze_196: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_182, -1)
    unsqueeze_197: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_360: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_354, unsqueeze_197);  mul_354 = unsqueeze_197 = None
    unsqueeze_198: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_183, -1);  primals_183 = None
    unsqueeze_199: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_259: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_360, unsqueeze_199);  mul_360 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_260: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_259, relu_42);  add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_46: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_260);  add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_61: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_46, primals_184, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_261: "i64[]" = torch.ops.aten.add.Tensor(primals_671, 1)
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 256, 1, 1]" = var_mean_50[0]
    getitem_103: "f32[1, 256, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_262: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_50: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
    sub_61: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_103)
    mul_361: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_50);  sub_61 = None
    squeeze_150: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_151: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_362: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_363: "f32[256]" = torch.ops.aten.mul.Tensor(primals_669, 0.9)
    add_263: "f32[256]" = torch.ops.aten.add.Tensor(mul_362, mul_363);  mul_362 = mul_363 = None
    squeeze_152: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_364: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0004885197850513);  squeeze_152 = None
    mul_365: "f32[256]" = torch.ops.aten.mul.Tensor(mul_364, 0.1);  mul_364 = None
    mul_366: "f32[256]" = torch.ops.aten.mul.Tensor(primals_670, 0.9)
    add_264: "f32[256]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    unsqueeze_200: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_185, -1)
    unsqueeze_201: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_367: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_201);  mul_361 = unsqueeze_201 = None
    unsqueeze_202: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_186, -1);  primals_186 = None
    unsqueeze_203: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_265: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_367, unsqueeze_203);  mul_367 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_47: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_265);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_62: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_47, primals_187, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_266: "i64[]" = torch.ops.aten.add.Tensor(primals_674, 1)
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 512, 1, 1]" = var_mean_51[0]
    getitem_105: "f32[1, 512, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_267: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
    rsqrt_51: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
    sub_62: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_105)
    mul_368: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_51);  sub_62 = None
    squeeze_153: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_154: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_369: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_370: "f32[512]" = torch.ops.aten.mul.Tensor(primals_672, 0.9)
    add_268: "f32[512]" = torch.ops.aten.add.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
    squeeze_155: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_371: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0004885197850513);  squeeze_155 = None
    mul_372: "f32[512]" = torch.ops.aten.mul.Tensor(mul_371, 0.1);  mul_371 = None
    mul_373: "f32[512]" = torch.ops.aten.mul.Tensor(primals_673, 0.9)
    add_269: "f32[512]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    unsqueeze_204: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_188, -1)
    unsqueeze_205: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_374: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_368, unsqueeze_205);  mul_368 = unsqueeze_205 = None
    unsqueeze_206: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_189, -1);  primals_189 = None
    unsqueeze_207: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_270: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_207);  mul_374 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_48: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_270);  add_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_67: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_48, [8, 2, 256, 16, 16])
    sum_34: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_67, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_11: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_34, [2, 3], True);  sum_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_63: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_190, primals_191, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_271: "i64[]" = torch.ops.aten.add.Tensor(primals_677, 1)
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 128, 1, 1]" = var_mean_52[0]
    getitem_107: "f32[1, 128, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_272: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_52: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
    sub_63: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_107)
    mul_375: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_52);  sub_63 = rsqrt_52 = None
    squeeze_156: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    mul_376: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1);  squeeze_156 = None
    mul_377: "f32[128]" = torch.ops.aten.mul.Tensor(primals_675, 0.9)
    add_273: "f32[128]" = torch.ops.aten.add.Tensor(mul_376, mul_377);  mul_376 = mul_377 = None
    squeeze_158: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_378: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.1428571428571428);  squeeze_158 = None
    mul_379: "f32[128]" = torch.ops.aten.mul.Tensor(mul_378, 0.1);  mul_378 = None
    mul_380: "f32[128]" = torch.ops.aten.mul.Tensor(primals_676, 0.9)
    add_274: "f32[128]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    unsqueeze_208: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_192, -1)
    unsqueeze_209: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_381: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_375, unsqueeze_209);  mul_375 = unsqueeze_209 = None
    unsqueeze_210: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_193, -1);  primals_193 = None
    unsqueeze_211: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_275: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_381, unsqueeze_211);  mul_381 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_49: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_275);  add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_64: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_49, primals_194, primals_195, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_68: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_64, [8, 1, 2, -1]);  convolution_64 = None
    permute_11: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_11: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_11, [1], True)
    sub_64: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_11, amax_11);  permute_11 = amax_11 = None
    exp_11: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_35: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_11, [1], True)
    div_11: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_11, sum_35);  exp_11 = sum_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_69: "f32[8, 512]" = torch.ops.aten.reshape.default(div_11, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_70: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_69, [8, -1, 1, 1]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_71: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_70, [8, 2, 256, 1, 1]);  view_70 = None
    mul_382: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_67, view_71);  view_67 = view_71 = None
    sum_36: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_382, [1]);  mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_65: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_36, primals_196, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_276: "i64[]" = torch.ops.aten.add.Tensor(primals_680, 1)
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 1024, 1, 1]" = var_mean_53[0]
    getitem_109: "f32[1, 1024, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_277: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
    rsqrt_53: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_277);  add_277 = None
    sub_65: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_109)
    mul_383: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_53);  sub_65 = None
    squeeze_159: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_160: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_384: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_385: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_678, 0.9)
    add_278: "f32[1024]" = torch.ops.aten.add.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
    squeeze_161: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_386: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0004885197850513);  squeeze_161 = None
    mul_387: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_386, 0.1);  mul_386 = None
    mul_388: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_679, 0.9)
    add_279: "f32[1024]" = torch.ops.aten.add.Tensor(mul_387, mul_388);  mul_387 = mul_388 = None
    unsqueeze_212: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_197, -1)
    unsqueeze_213: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_389: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_383, unsqueeze_213);  mul_383 = unsqueeze_213 = None
    unsqueeze_214: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_198, -1);  primals_198 = None
    unsqueeze_215: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_280: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_389, unsqueeze_215);  mul_389 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_281: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_280, relu_46);  add_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_50: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_281);  add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_66: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_50, primals_199, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_282: "i64[]" = torch.ops.aten.add.Tensor(primals_683, 1)
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 256, 1, 1]" = var_mean_54[0]
    getitem_111: "f32[1, 256, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_283: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
    rsqrt_54: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
    sub_66: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_111)
    mul_390: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_54);  sub_66 = None
    squeeze_162: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_163: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_391: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_392: "f32[256]" = torch.ops.aten.mul.Tensor(primals_681, 0.9)
    add_284: "f32[256]" = torch.ops.aten.add.Tensor(mul_391, mul_392);  mul_391 = mul_392 = None
    squeeze_164: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_393: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0004885197850513);  squeeze_164 = None
    mul_394: "f32[256]" = torch.ops.aten.mul.Tensor(mul_393, 0.1);  mul_393 = None
    mul_395: "f32[256]" = torch.ops.aten.mul.Tensor(primals_682, 0.9)
    add_285: "f32[256]" = torch.ops.aten.add.Tensor(mul_394, mul_395);  mul_394 = mul_395 = None
    unsqueeze_216: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_200, -1)
    unsqueeze_217: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_396: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_390, unsqueeze_217);  mul_390 = unsqueeze_217 = None
    unsqueeze_218: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_201, -1);  primals_201 = None
    unsqueeze_219: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_286: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_396, unsqueeze_219);  mul_396 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_51: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_286);  add_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_67: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_51, primals_202, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_287: "i64[]" = torch.ops.aten.add.Tensor(primals_686, 1)
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_67, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 512, 1, 1]" = var_mean_55[0]
    getitem_113: "f32[1, 512, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_288: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_55: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_288);  add_288 = None
    sub_67: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_67, getitem_113)
    mul_397: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_55);  sub_67 = None
    squeeze_165: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_166: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_398: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_399: "f32[512]" = torch.ops.aten.mul.Tensor(primals_684, 0.9)
    add_289: "f32[512]" = torch.ops.aten.add.Tensor(mul_398, mul_399);  mul_398 = mul_399 = None
    squeeze_167: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_400: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0004885197850513);  squeeze_167 = None
    mul_401: "f32[512]" = torch.ops.aten.mul.Tensor(mul_400, 0.1);  mul_400 = None
    mul_402: "f32[512]" = torch.ops.aten.mul.Tensor(primals_685, 0.9)
    add_290: "f32[512]" = torch.ops.aten.add.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
    unsqueeze_220: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1)
    unsqueeze_221: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_403: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_397, unsqueeze_221);  mul_397 = unsqueeze_221 = None
    unsqueeze_222: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
    unsqueeze_223: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_291: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_403, unsqueeze_223);  mul_403 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_52: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_291);  add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_73: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_52, [8, 2, 256, 16, 16])
    sum_37: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_73, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_12: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_37, [2, 3], True);  sum_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_68: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_205, primals_206, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_292: "i64[]" = torch.ops.aten.add.Tensor(primals_689, 1)
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 128, 1, 1]" = var_mean_56[0]
    getitem_115: "f32[1, 128, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_293: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
    rsqrt_56: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
    sub_68: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_115)
    mul_404: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_56);  sub_68 = rsqrt_56 = None
    squeeze_168: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    mul_405: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1);  squeeze_168 = None
    mul_406: "f32[128]" = torch.ops.aten.mul.Tensor(primals_687, 0.9)
    add_294: "f32[128]" = torch.ops.aten.add.Tensor(mul_405, mul_406);  mul_405 = mul_406 = None
    squeeze_170: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_407: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.1428571428571428);  squeeze_170 = None
    mul_408: "f32[128]" = torch.ops.aten.mul.Tensor(mul_407, 0.1);  mul_407 = None
    mul_409: "f32[128]" = torch.ops.aten.mul.Tensor(primals_688, 0.9)
    add_295: "f32[128]" = torch.ops.aten.add.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    unsqueeze_224: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_207, -1)
    unsqueeze_225: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_410: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_404, unsqueeze_225);  mul_404 = unsqueeze_225 = None
    unsqueeze_226: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_208, -1);  primals_208 = None
    unsqueeze_227: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_296: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_410, unsqueeze_227);  mul_410 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_53: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_296);  add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_69: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_53, primals_209, primals_210, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_74: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_69, [8, 1, 2, -1]);  convolution_69 = None
    permute_12: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_12: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_12, [1], True)
    sub_69: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_12, amax_12);  permute_12 = amax_12 = None
    exp_12: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_69);  sub_69 = None
    sum_38: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_12, [1], True)
    div_12: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_12, sum_38);  exp_12 = sum_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_75: "f32[8, 512]" = torch.ops.aten.reshape.default(div_12, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_76: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_75, [8, -1, 1, 1]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_77: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_76, [8, 2, 256, 1, 1]);  view_76 = None
    mul_411: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_73, view_77);  view_73 = view_77 = None
    sum_39: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_411, [1]);  mul_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_70: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_39, primals_211, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_297: "i64[]" = torch.ops.aten.add.Tensor(primals_692, 1)
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[1, 1024, 1, 1]" = var_mean_57[0]
    getitem_117: "f32[1, 1024, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_298: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
    rsqrt_57: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_298);  add_298 = None
    sub_70: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_117)
    mul_412: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_57);  sub_70 = None
    squeeze_171: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
    squeeze_172: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_413: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_414: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_690, 0.9)
    add_299: "f32[1024]" = torch.ops.aten.add.Tensor(mul_413, mul_414);  mul_413 = mul_414 = None
    squeeze_173: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
    mul_415: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0004885197850513);  squeeze_173 = None
    mul_416: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_415, 0.1);  mul_415 = None
    mul_417: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_691, 0.9)
    add_300: "f32[1024]" = torch.ops.aten.add.Tensor(mul_416, mul_417);  mul_416 = mul_417 = None
    unsqueeze_228: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_212, -1)
    unsqueeze_229: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_418: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_229);  mul_412 = unsqueeze_229 = None
    unsqueeze_230: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_213, -1);  primals_213 = None
    unsqueeze_231: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_301: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_418, unsqueeze_231);  mul_418 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_302: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_301, relu_50);  add_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_54: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_302);  add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_71: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_54, primals_214, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_303: "i64[]" = torch.ops.aten.add.Tensor(primals_695, 1)
    var_mean_58 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 256, 1, 1]" = var_mean_58[0]
    getitem_119: "f32[1, 256, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_304: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_58: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_304);  add_304 = None
    sub_71: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_119)
    mul_419: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_58);  sub_71 = None
    squeeze_174: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_175: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_420: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_421: "f32[256]" = torch.ops.aten.mul.Tensor(primals_693, 0.9)
    add_305: "f32[256]" = torch.ops.aten.add.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    squeeze_176: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_422: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0004885197850513);  squeeze_176 = None
    mul_423: "f32[256]" = torch.ops.aten.mul.Tensor(mul_422, 0.1);  mul_422 = None
    mul_424: "f32[256]" = torch.ops.aten.mul.Tensor(primals_694, 0.9)
    add_306: "f32[256]" = torch.ops.aten.add.Tensor(mul_423, mul_424);  mul_423 = mul_424 = None
    unsqueeze_232: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_215, -1)
    unsqueeze_233: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_425: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_419, unsqueeze_233);  mul_419 = unsqueeze_233 = None
    unsqueeze_234: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_216, -1);  primals_216 = None
    unsqueeze_235: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_307: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_425, unsqueeze_235);  mul_425 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_55: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_307);  add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_72: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_55, primals_217, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_308: "i64[]" = torch.ops.aten.add.Tensor(primals_698, 1)
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_72, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 512, 1, 1]" = var_mean_59[0]
    getitem_121: "f32[1, 512, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_309: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
    rsqrt_59: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_309);  add_309 = None
    sub_72: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_72, getitem_121)
    mul_426: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_59);  sub_72 = None
    squeeze_177: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_178: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    mul_427: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_428: "f32[512]" = torch.ops.aten.mul.Tensor(primals_696, 0.9)
    add_310: "f32[512]" = torch.ops.aten.add.Tensor(mul_427, mul_428);  mul_427 = mul_428 = None
    squeeze_179: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_429: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0004885197850513);  squeeze_179 = None
    mul_430: "f32[512]" = torch.ops.aten.mul.Tensor(mul_429, 0.1);  mul_429 = None
    mul_431: "f32[512]" = torch.ops.aten.mul.Tensor(primals_697, 0.9)
    add_311: "f32[512]" = torch.ops.aten.add.Tensor(mul_430, mul_431);  mul_430 = mul_431 = None
    unsqueeze_236: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_218, -1)
    unsqueeze_237: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_432: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_426, unsqueeze_237);  mul_426 = unsqueeze_237 = None
    unsqueeze_238: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_219, -1);  primals_219 = None
    unsqueeze_239: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_312: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_432, unsqueeze_239);  mul_432 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_56: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_312);  add_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_79: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_56, [8, 2, 256, 16, 16])
    sum_40: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_79, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_13: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_40, [2, 3], True);  sum_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_73: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_13, primals_220, primals_221, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_313: "i64[]" = torch.ops.aten.add.Tensor(primals_701, 1)
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_73, [0, 2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[1, 128, 1, 1]" = var_mean_60[0]
    getitem_123: "f32[1, 128, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_314: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
    rsqrt_60: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
    sub_73: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_73, getitem_123)
    mul_433: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_60);  sub_73 = rsqrt_60 = None
    squeeze_180: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    mul_434: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1);  squeeze_180 = None
    mul_435: "f32[128]" = torch.ops.aten.mul.Tensor(primals_699, 0.9)
    add_315: "f32[128]" = torch.ops.aten.add.Tensor(mul_434, mul_435);  mul_434 = mul_435 = None
    squeeze_182: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
    mul_436: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.1428571428571428);  squeeze_182 = None
    mul_437: "f32[128]" = torch.ops.aten.mul.Tensor(mul_436, 0.1);  mul_436 = None
    mul_438: "f32[128]" = torch.ops.aten.mul.Tensor(primals_700, 0.9)
    add_316: "f32[128]" = torch.ops.aten.add.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
    unsqueeze_240: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_222, -1)
    unsqueeze_241: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_439: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_241);  mul_433 = unsqueeze_241 = None
    unsqueeze_242: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_223, -1);  primals_223 = None
    unsqueeze_243: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_317: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_439, unsqueeze_243);  mul_439 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_57: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_317);  add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_74: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_57, primals_224, primals_225, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_80: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_74, [8, 1, 2, -1]);  convolution_74 = None
    permute_13: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_13: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_13, [1], True)
    sub_74: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_13, amax_13);  permute_13 = amax_13 = None
    exp_13: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_74);  sub_74 = None
    sum_41: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_13, [1], True)
    div_13: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_13, sum_41);  exp_13 = sum_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_81: "f32[8, 512]" = torch.ops.aten.reshape.default(div_13, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_82: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_81, [8, -1, 1, 1]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_83: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_82, [8, 2, 256, 1, 1]);  view_82 = None
    mul_440: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_79, view_83);  view_79 = view_83 = None
    sum_42: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_440, [1]);  mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_75: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_42, primals_226, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_318: "i64[]" = torch.ops.aten.add.Tensor(primals_704, 1)
    var_mean_61 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 1024, 1, 1]" = var_mean_61[0]
    getitem_125: "f32[1, 1024, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_319: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
    rsqrt_61: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_319);  add_319 = None
    sub_75: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_125)
    mul_441: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_61);  sub_75 = None
    squeeze_183: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_184: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_442: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_443: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_702, 0.9)
    add_320: "f32[1024]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    squeeze_185: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_444: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0004885197850513);  squeeze_185 = None
    mul_445: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
    mul_446: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_703, 0.9)
    add_321: "f32[1024]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
    unsqueeze_244: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_227, -1)
    unsqueeze_245: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_447: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_245);  mul_441 = unsqueeze_245 = None
    unsqueeze_246: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1);  primals_228 = None
    unsqueeze_247: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_322: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_247);  mul_447 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_323: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_322, relu_54);  add_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_58: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_323);  add_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_76: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_58, primals_229, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_324: "i64[]" = torch.ops.aten.add.Tensor(primals_707, 1)
    var_mean_62 = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 256, 1, 1]" = var_mean_62[0]
    getitem_127: "f32[1, 256, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_325: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
    rsqrt_62: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_325);  add_325 = None
    sub_76: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_127)
    mul_448: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_62);  sub_76 = None
    squeeze_186: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_187: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
    mul_449: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_450: "f32[256]" = torch.ops.aten.mul.Tensor(primals_705, 0.9)
    add_326: "f32[256]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    squeeze_188: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_451: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0004885197850513);  squeeze_188 = None
    mul_452: "f32[256]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
    mul_453: "f32[256]" = torch.ops.aten.mul.Tensor(primals_706, 0.9)
    add_327: "f32[256]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    unsqueeze_248: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_230, -1)
    unsqueeze_249: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    mul_454: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_249);  mul_448 = unsqueeze_249 = None
    unsqueeze_250: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_231, -1);  primals_231 = None
    unsqueeze_251: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    add_328: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_251);  mul_454 = unsqueeze_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_59: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_328);  add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_77: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_59, primals_232, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_329: "i64[]" = torch.ops.aten.add.Tensor(primals_710, 1)
    var_mean_63 = torch.ops.aten.var_mean.correction(convolution_77, [0, 2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[1, 512, 1, 1]" = var_mean_63[0]
    getitem_129: "f32[1, 512, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_330: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
    rsqrt_63: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_330);  add_330 = None
    sub_77: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_77, getitem_129)
    mul_455: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_63);  sub_77 = None
    squeeze_189: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
    squeeze_190: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
    mul_456: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_457: "f32[512]" = torch.ops.aten.mul.Tensor(primals_708, 0.9)
    add_331: "f32[512]" = torch.ops.aten.add.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    squeeze_191: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
    mul_458: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0004885197850513);  squeeze_191 = None
    mul_459: "f32[512]" = torch.ops.aten.mul.Tensor(mul_458, 0.1);  mul_458 = None
    mul_460: "f32[512]" = torch.ops.aten.mul.Tensor(primals_709, 0.9)
    add_332: "f32[512]" = torch.ops.aten.add.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
    unsqueeze_252: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_233, -1)
    unsqueeze_253: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_461: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_455, unsqueeze_253);  mul_455 = unsqueeze_253 = None
    unsqueeze_254: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_234, -1);  primals_234 = None
    unsqueeze_255: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_333: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_255);  mul_461 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_60: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_333);  add_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_85: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_60, [8, 2, 256, 16, 16])
    sum_43: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_85, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_14: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_43, [2, 3], True);  sum_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_78: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_14, primals_235, primals_236, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_334: "i64[]" = torch.ops.aten.add.Tensor(primals_713, 1)
    var_mean_64 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
    getitem_130: "f32[1, 128, 1, 1]" = var_mean_64[0]
    getitem_131: "f32[1, 128, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_335: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05)
    rsqrt_64: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_335);  add_335 = None
    sub_78: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_131)
    mul_462: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_64);  sub_78 = rsqrt_64 = None
    squeeze_192: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
    mul_463: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1);  squeeze_192 = None
    mul_464: "f32[128]" = torch.ops.aten.mul.Tensor(primals_711, 0.9)
    add_336: "f32[128]" = torch.ops.aten.add.Tensor(mul_463, mul_464);  mul_463 = mul_464 = None
    squeeze_194: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
    mul_465: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.1428571428571428);  squeeze_194 = None
    mul_466: "f32[128]" = torch.ops.aten.mul.Tensor(mul_465, 0.1);  mul_465 = None
    mul_467: "f32[128]" = torch.ops.aten.mul.Tensor(primals_712, 0.9)
    add_337: "f32[128]" = torch.ops.aten.add.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
    unsqueeze_256: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_237, -1)
    unsqueeze_257: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    mul_468: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_257);  mul_462 = unsqueeze_257 = None
    unsqueeze_258: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_238, -1);  primals_238 = None
    unsqueeze_259: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    add_338: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_468, unsqueeze_259);  mul_468 = unsqueeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_61: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_338);  add_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_79: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_61, primals_239, primals_240, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_86: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_79, [8, 1, 2, -1]);  convolution_79 = None
    permute_14: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_14: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_14, [1], True)
    sub_79: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_14, amax_14);  permute_14 = amax_14 = None
    exp_14: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_79);  sub_79 = None
    sum_44: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_14, [1], True)
    div_14: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_14, sum_44);  exp_14 = sum_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_87: "f32[8, 512]" = torch.ops.aten.reshape.default(div_14, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_88: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_87, [8, -1, 1, 1]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_89: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_88, [8, 2, 256, 1, 1]);  view_88 = None
    mul_469: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_85, view_89);  view_85 = view_89 = None
    sum_45: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_469, [1]);  mul_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_80: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_45, primals_241, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_339: "i64[]" = torch.ops.aten.add.Tensor(primals_716, 1)
    var_mean_65 = torch.ops.aten.var_mean.correction(convolution_80, [0, 2, 3], correction = 0, keepdim = True)
    getitem_132: "f32[1, 1024, 1, 1]" = var_mean_65[0]
    getitem_133: "f32[1, 1024, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_340: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05)
    rsqrt_65: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_340);  add_340 = None
    sub_80: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_133)
    mul_470: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_65);  sub_80 = None
    squeeze_195: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
    squeeze_196: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
    mul_471: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
    mul_472: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_714, 0.9)
    add_341: "f32[1024]" = torch.ops.aten.add.Tensor(mul_471, mul_472);  mul_471 = mul_472 = None
    squeeze_197: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
    mul_473: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0004885197850513);  squeeze_197 = None
    mul_474: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_473, 0.1);  mul_473 = None
    mul_475: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_715, 0.9)
    add_342: "f32[1024]" = torch.ops.aten.add.Tensor(mul_474, mul_475);  mul_474 = mul_475 = None
    unsqueeze_260: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_242, -1)
    unsqueeze_261: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_476: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_470, unsqueeze_261);  mul_470 = unsqueeze_261 = None
    unsqueeze_262: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_243, -1);  primals_243 = None
    unsqueeze_263: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_343: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_476, unsqueeze_263);  mul_476 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_344: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_343, relu_58);  add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_62: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_344);  add_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_81: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_62, primals_244, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_345: "i64[]" = torch.ops.aten.add.Tensor(primals_719, 1)
    var_mean_66 = torch.ops.aten.var_mean.correction(convolution_81, [0, 2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[1, 256, 1, 1]" = var_mean_66[0]
    getitem_135: "f32[1, 256, 1, 1]" = var_mean_66[1];  var_mean_66 = None
    add_346: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05)
    rsqrt_66: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_346);  add_346 = None
    sub_81: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_81, getitem_135)
    mul_477: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_66);  sub_81 = None
    squeeze_198: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
    squeeze_199: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
    mul_478: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
    mul_479: "f32[256]" = torch.ops.aten.mul.Tensor(primals_717, 0.9)
    add_347: "f32[256]" = torch.ops.aten.add.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
    squeeze_200: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
    mul_480: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0004885197850513);  squeeze_200 = None
    mul_481: "f32[256]" = torch.ops.aten.mul.Tensor(mul_480, 0.1);  mul_480 = None
    mul_482: "f32[256]" = torch.ops.aten.mul.Tensor(primals_718, 0.9)
    add_348: "f32[256]" = torch.ops.aten.add.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
    unsqueeze_264: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_245, -1)
    unsqueeze_265: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    mul_483: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_477, unsqueeze_265);  mul_477 = unsqueeze_265 = None
    unsqueeze_266: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_246, -1);  primals_246 = None
    unsqueeze_267: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    add_349: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_483, unsqueeze_267);  mul_483 = unsqueeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_63: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_349);  add_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_82: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_63, primals_247, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_350: "i64[]" = torch.ops.aten.add.Tensor(primals_722, 1)
    var_mean_67 = torch.ops.aten.var_mean.correction(convolution_82, [0, 2, 3], correction = 0, keepdim = True)
    getitem_136: "f32[1, 512, 1, 1]" = var_mean_67[0]
    getitem_137: "f32[1, 512, 1, 1]" = var_mean_67[1];  var_mean_67 = None
    add_351: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05)
    rsqrt_67: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
    sub_82: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_82, getitem_137)
    mul_484: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_67);  sub_82 = None
    squeeze_201: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
    squeeze_202: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
    mul_485: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
    mul_486: "f32[512]" = torch.ops.aten.mul.Tensor(primals_720, 0.9)
    add_352: "f32[512]" = torch.ops.aten.add.Tensor(mul_485, mul_486);  mul_485 = mul_486 = None
    squeeze_203: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
    mul_487: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_203, 1.0004885197850513);  squeeze_203 = None
    mul_488: "f32[512]" = torch.ops.aten.mul.Tensor(mul_487, 0.1);  mul_487 = None
    mul_489: "f32[512]" = torch.ops.aten.mul.Tensor(primals_721, 0.9)
    add_353: "f32[512]" = torch.ops.aten.add.Tensor(mul_488, mul_489);  mul_488 = mul_489 = None
    unsqueeze_268: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_248, -1)
    unsqueeze_269: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_490: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_484, unsqueeze_269);  mul_484 = unsqueeze_269 = None
    unsqueeze_270: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_249, -1);  primals_249 = None
    unsqueeze_271: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_354: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_490, unsqueeze_271);  mul_490 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_64: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_354);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_91: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_64, [8, 2, 256, 16, 16])
    sum_46: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_91, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_15: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_46, [2, 3], True);  sum_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_83: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_15, primals_250, primals_251, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_355: "i64[]" = torch.ops.aten.add.Tensor(primals_725, 1)
    var_mean_68 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
    getitem_138: "f32[1, 128, 1, 1]" = var_mean_68[0]
    getitem_139: "f32[1, 128, 1, 1]" = var_mean_68[1];  var_mean_68 = None
    add_356: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05)
    rsqrt_68: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_356);  add_356 = None
    sub_83: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_139)
    mul_491: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_68);  sub_83 = rsqrt_68 = None
    squeeze_204: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
    mul_492: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1);  squeeze_204 = None
    mul_493: "f32[128]" = torch.ops.aten.mul.Tensor(primals_723, 0.9)
    add_357: "f32[128]" = torch.ops.aten.add.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
    squeeze_206: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
    mul_494: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_206, 1.1428571428571428);  squeeze_206 = None
    mul_495: "f32[128]" = torch.ops.aten.mul.Tensor(mul_494, 0.1);  mul_494 = None
    mul_496: "f32[128]" = torch.ops.aten.mul.Tensor(primals_724, 0.9)
    add_358: "f32[128]" = torch.ops.aten.add.Tensor(mul_495, mul_496);  mul_495 = mul_496 = None
    unsqueeze_272: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_252, -1)
    unsqueeze_273: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    mul_497: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_491, unsqueeze_273);  mul_491 = unsqueeze_273 = None
    unsqueeze_274: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_253, -1);  primals_253 = None
    unsqueeze_275: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    add_359: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_497, unsqueeze_275);  mul_497 = unsqueeze_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_65: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_359);  add_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_84: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_65, primals_254, primals_255, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_92: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_84, [8, 1, 2, -1]);  convolution_84 = None
    permute_15: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_15: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_15, [1], True)
    sub_84: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_15, amax_15);  permute_15 = amax_15 = None
    exp_15: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_84);  sub_84 = None
    sum_47: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_15, [1], True)
    div_15: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_15, sum_47);  exp_15 = sum_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_93: "f32[8, 512]" = torch.ops.aten.reshape.default(div_15, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_94: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_93, [8, -1, 1, 1]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_95: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_94, [8, 2, 256, 1, 1]);  view_94 = None
    mul_498: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_91, view_95);  view_91 = view_95 = None
    sum_48: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_498, [1]);  mul_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_85: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_48, primals_256, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_360: "i64[]" = torch.ops.aten.add.Tensor(primals_728, 1)
    var_mean_69 = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True)
    getitem_140: "f32[1, 1024, 1, 1]" = var_mean_69[0]
    getitem_141: "f32[1, 1024, 1, 1]" = var_mean_69[1];  var_mean_69 = None
    add_361: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05)
    rsqrt_69: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_361);  add_361 = None
    sub_85: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_141)
    mul_499: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_69);  sub_85 = None
    squeeze_207: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
    squeeze_208: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
    mul_500: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1)
    mul_501: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_726, 0.9)
    add_362: "f32[1024]" = torch.ops.aten.add.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
    squeeze_209: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
    mul_502: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_209, 1.0004885197850513);  squeeze_209 = None
    mul_503: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_502, 0.1);  mul_502 = None
    mul_504: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_727, 0.9)
    add_363: "f32[1024]" = torch.ops.aten.add.Tensor(mul_503, mul_504);  mul_503 = mul_504 = None
    unsqueeze_276: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_257, -1)
    unsqueeze_277: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_505: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_499, unsqueeze_277);  mul_499 = unsqueeze_277 = None
    unsqueeze_278: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_258, -1);  primals_258 = None
    unsqueeze_279: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_364: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_505, unsqueeze_279);  mul_505 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_365: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_364, relu_62);  add_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_66: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_365);  add_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_86: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_66, primals_259, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_366: "i64[]" = torch.ops.aten.add.Tensor(primals_731, 1)
    var_mean_70 = torch.ops.aten.var_mean.correction(convolution_86, [0, 2, 3], correction = 0, keepdim = True)
    getitem_142: "f32[1, 256, 1, 1]" = var_mean_70[0]
    getitem_143: "f32[1, 256, 1, 1]" = var_mean_70[1];  var_mean_70 = None
    add_367: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05)
    rsqrt_70: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_367);  add_367 = None
    sub_86: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_86, getitem_143)
    mul_506: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_70);  sub_86 = None
    squeeze_210: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
    squeeze_211: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
    mul_507: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
    mul_508: "f32[256]" = torch.ops.aten.mul.Tensor(primals_729, 0.9)
    add_368: "f32[256]" = torch.ops.aten.add.Tensor(mul_507, mul_508);  mul_507 = mul_508 = None
    squeeze_212: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
    mul_509: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_212, 1.0004885197850513);  squeeze_212 = None
    mul_510: "f32[256]" = torch.ops.aten.mul.Tensor(mul_509, 0.1);  mul_509 = None
    mul_511: "f32[256]" = torch.ops.aten.mul.Tensor(primals_730, 0.9)
    add_369: "f32[256]" = torch.ops.aten.add.Tensor(mul_510, mul_511);  mul_510 = mul_511 = None
    unsqueeze_280: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_260, -1)
    unsqueeze_281: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    mul_512: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_506, unsqueeze_281);  mul_506 = unsqueeze_281 = None
    unsqueeze_282: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_261, -1);  primals_261 = None
    unsqueeze_283: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    add_370: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_512, unsqueeze_283);  mul_512 = unsqueeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_67: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_370);  add_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_87: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_67, primals_262, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_371: "i64[]" = torch.ops.aten.add.Tensor(primals_734, 1)
    var_mean_71 = torch.ops.aten.var_mean.correction(convolution_87, [0, 2, 3], correction = 0, keepdim = True)
    getitem_144: "f32[1, 512, 1, 1]" = var_mean_71[0]
    getitem_145: "f32[1, 512, 1, 1]" = var_mean_71[1];  var_mean_71 = None
    add_372: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05)
    rsqrt_71: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_372);  add_372 = None
    sub_87: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_87, getitem_145)
    mul_513: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_71);  sub_87 = None
    squeeze_213: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
    squeeze_214: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
    mul_514: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1)
    mul_515: "f32[512]" = torch.ops.aten.mul.Tensor(primals_732, 0.9)
    add_373: "f32[512]" = torch.ops.aten.add.Tensor(mul_514, mul_515);  mul_514 = mul_515 = None
    squeeze_215: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
    mul_516: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_215, 1.0004885197850513);  squeeze_215 = None
    mul_517: "f32[512]" = torch.ops.aten.mul.Tensor(mul_516, 0.1);  mul_516 = None
    mul_518: "f32[512]" = torch.ops.aten.mul.Tensor(primals_733, 0.9)
    add_374: "f32[512]" = torch.ops.aten.add.Tensor(mul_517, mul_518);  mul_517 = mul_518 = None
    unsqueeze_284: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_263, -1)
    unsqueeze_285: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_519: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_513, unsqueeze_285);  mul_513 = unsqueeze_285 = None
    unsqueeze_286: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_264, -1);  primals_264 = None
    unsqueeze_287: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_375: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_519, unsqueeze_287);  mul_519 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_68: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_375);  add_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_97: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_68, [8, 2, 256, 16, 16])
    sum_49: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_97, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_16: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_49, [2, 3], True);  sum_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_88: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_16, primals_265, primals_266, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_376: "i64[]" = torch.ops.aten.add.Tensor(primals_737, 1)
    var_mean_72 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
    getitem_146: "f32[1, 128, 1, 1]" = var_mean_72[0]
    getitem_147: "f32[1, 128, 1, 1]" = var_mean_72[1];  var_mean_72 = None
    add_377: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05)
    rsqrt_72: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_377);  add_377 = None
    sub_88: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_147)
    mul_520: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_72);  sub_88 = rsqrt_72 = None
    squeeze_216: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
    mul_521: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1);  squeeze_216 = None
    mul_522: "f32[128]" = torch.ops.aten.mul.Tensor(primals_735, 0.9)
    add_378: "f32[128]" = torch.ops.aten.add.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
    squeeze_218: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
    mul_523: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_218, 1.1428571428571428);  squeeze_218 = None
    mul_524: "f32[128]" = torch.ops.aten.mul.Tensor(mul_523, 0.1);  mul_523 = None
    mul_525: "f32[128]" = torch.ops.aten.mul.Tensor(primals_736, 0.9)
    add_379: "f32[128]" = torch.ops.aten.add.Tensor(mul_524, mul_525);  mul_524 = mul_525 = None
    unsqueeze_288: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_267, -1)
    unsqueeze_289: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    mul_526: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_520, unsqueeze_289);  mul_520 = unsqueeze_289 = None
    unsqueeze_290: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_268, -1);  primals_268 = None
    unsqueeze_291: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    add_380: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_526, unsqueeze_291);  mul_526 = unsqueeze_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_69: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_380);  add_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_89: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_69, primals_269, primals_270, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_98: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_89, [8, 1, 2, -1]);  convolution_89 = None
    permute_16: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_16: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_16, [1], True)
    sub_89: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_16, amax_16);  permute_16 = amax_16 = None
    exp_16: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_89);  sub_89 = None
    sum_50: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_16, [1], True)
    div_16: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_16, sum_50);  exp_16 = sum_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_99: "f32[8, 512]" = torch.ops.aten.reshape.default(div_16, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_100: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_99, [8, -1, 1, 1]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_101: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_100, [8, 2, 256, 1, 1]);  view_100 = None
    mul_527: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_97, view_101);  view_97 = view_101 = None
    sum_51: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_527, [1]);  mul_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_90: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_51, primals_271, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_381: "i64[]" = torch.ops.aten.add.Tensor(primals_740, 1)
    var_mean_73 = torch.ops.aten.var_mean.correction(convolution_90, [0, 2, 3], correction = 0, keepdim = True)
    getitem_148: "f32[1, 1024, 1, 1]" = var_mean_73[0]
    getitem_149: "f32[1, 1024, 1, 1]" = var_mean_73[1];  var_mean_73 = None
    add_382: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05)
    rsqrt_73: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_382);  add_382 = None
    sub_90: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_90, getitem_149)
    mul_528: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_73);  sub_90 = None
    squeeze_219: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
    squeeze_220: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
    mul_529: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
    mul_530: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_738, 0.9)
    add_383: "f32[1024]" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    squeeze_221: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
    mul_531: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_221, 1.0004885197850513);  squeeze_221 = None
    mul_532: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_531, 0.1);  mul_531 = None
    mul_533: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_739, 0.9)
    add_384: "f32[1024]" = torch.ops.aten.add.Tensor(mul_532, mul_533);  mul_532 = mul_533 = None
    unsqueeze_292: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_272, -1)
    unsqueeze_293: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_534: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_528, unsqueeze_293);  mul_528 = unsqueeze_293 = None
    unsqueeze_294: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_273, -1);  primals_273 = None
    unsqueeze_295: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_385: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_534, unsqueeze_295);  mul_534 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_386: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_385, relu_66);  add_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_70: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_386);  add_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_91: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_70, primals_274, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_387: "i64[]" = torch.ops.aten.add.Tensor(primals_743, 1)
    var_mean_74 = torch.ops.aten.var_mean.correction(convolution_91, [0, 2, 3], correction = 0, keepdim = True)
    getitem_150: "f32[1, 256, 1, 1]" = var_mean_74[0]
    getitem_151: "f32[1, 256, 1, 1]" = var_mean_74[1];  var_mean_74 = None
    add_388: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
    rsqrt_74: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_388);  add_388 = None
    sub_91: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_91, getitem_151)
    mul_535: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_74);  sub_91 = None
    squeeze_222: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
    squeeze_223: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
    mul_536: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
    mul_537: "f32[256]" = torch.ops.aten.mul.Tensor(primals_741, 0.9)
    add_389: "f32[256]" = torch.ops.aten.add.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
    squeeze_224: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
    mul_538: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_224, 1.0004885197850513);  squeeze_224 = None
    mul_539: "f32[256]" = torch.ops.aten.mul.Tensor(mul_538, 0.1);  mul_538 = None
    mul_540: "f32[256]" = torch.ops.aten.mul.Tensor(primals_742, 0.9)
    add_390: "f32[256]" = torch.ops.aten.add.Tensor(mul_539, mul_540);  mul_539 = mul_540 = None
    unsqueeze_296: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_275, -1)
    unsqueeze_297: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    mul_541: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_535, unsqueeze_297);  mul_535 = unsqueeze_297 = None
    unsqueeze_298: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_276, -1);  primals_276 = None
    unsqueeze_299: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    add_391: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_541, unsqueeze_299);  mul_541 = unsqueeze_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_71: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_391);  add_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_92: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_71, primals_277, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_392: "i64[]" = torch.ops.aten.add.Tensor(primals_746, 1)
    var_mean_75 = torch.ops.aten.var_mean.correction(convolution_92, [0, 2, 3], correction = 0, keepdim = True)
    getitem_152: "f32[1, 512, 1, 1]" = var_mean_75[0]
    getitem_153: "f32[1, 512, 1, 1]" = var_mean_75[1];  var_mean_75 = None
    add_393: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05)
    rsqrt_75: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_393);  add_393 = None
    sub_92: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_153)
    mul_542: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_75);  sub_92 = None
    squeeze_225: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
    squeeze_226: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
    mul_543: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1)
    mul_544: "f32[512]" = torch.ops.aten.mul.Tensor(primals_744, 0.9)
    add_394: "f32[512]" = torch.ops.aten.add.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
    squeeze_227: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
    mul_545: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_227, 1.0004885197850513);  squeeze_227 = None
    mul_546: "f32[512]" = torch.ops.aten.mul.Tensor(mul_545, 0.1);  mul_545 = None
    mul_547: "f32[512]" = torch.ops.aten.mul.Tensor(primals_745, 0.9)
    add_395: "f32[512]" = torch.ops.aten.add.Tensor(mul_546, mul_547);  mul_546 = mul_547 = None
    unsqueeze_300: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_278, -1)
    unsqueeze_301: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_548: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_542, unsqueeze_301);  mul_542 = unsqueeze_301 = None
    unsqueeze_302: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_279, -1);  primals_279 = None
    unsqueeze_303: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_396: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_548, unsqueeze_303);  mul_548 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_72: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_396);  add_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_103: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_72, [8, 2, 256, 16, 16])
    sum_52: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_103, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_17: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_52, [2, 3], True);  sum_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_93: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_17, primals_280, primals_281, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_397: "i64[]" = torch.ops.aten.add.Tensor(primals_749, 1)
    var_mean_76 = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True)
    getitem_154: "f32[1, 128, 1, 1]" = var_mean_76[0]
    getitem_155: "f32[1, 128, 1, 1]" = var_mean_76[1];  var_mean_76 = None
    add_398: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05)
    rsqrt_76: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_398);  add_398 = None
    sub_93: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_155)
    mul_549: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_76);  sub_93 = rsqrt_76 = None
    squeeze_228: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
    mul_550: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1);  squeeze_228 = None
    mul_551: "f32[128]" = torch.ops.aten.mul.Tensor(primals_747, 0.9)
    add_399: "f32[128]" = torch.ops.aten.add.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
    squeeze_230: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
    mul_552: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_230, 1.1428571428571428);  squeeze_230 = None
    mul_553: "f32[128]" = torch.ops.aten.mul.Tensor(mul_552, 0.1);  mul_552 = None
    mul_554: "f32[128]" = torch.ops.aten.mul.Tensor(primals_748, 0.9)
    add_400: "f32[128]" = torch.ops.aten.add.Tensor(mul_553, mul_554);  mul_553 = mul_554 = None
    unsqueeze_304: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_282, -1)
    unsqueeze_305: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    mul_555: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_549, unsqueeze_305);  mul_549 = unsqueeze_305 = None
    unsqueeze_306: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_283, -1);  primals_283 = None
    unsqueeze_307: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    add_401: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_555, unsqueeze_307);  mul_555 = unsqueeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_73: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_401);  add_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_94: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_73, primals_284, primals_285, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_104: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_94, [8, 1, 2, -1]);  convolution_94 = None
    permute_17: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_17: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_17, [1], True)
    sub_94: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_17, amax_17);  permute_17 = amax_17 = None
    exp_17: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_94);  sub_94 = None
    sum_53: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_17, [1], True)
    div_17: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_17, sum_53);  exp_17 = sum_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_105: "f32[8, 512]" = torch.ops.aten.reshape.default(div_17, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_106: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_105, [8, -1, 1, 1]);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_107: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_106, [8, 2, 256, 1, 1]);  view_106 = None
    mul_556: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_103, view_107);  view_103 = view_107 = None
    sum_54: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_556, [1]);  mul_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_95: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_54, primals_286, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_402: "i64[]" = torch.ops.aten.add.Tensor(primals_752, 1)
    var_mean_77 = torch.ops.aten.var_mean.correction(convolution_95, [0, 2, 3], correction = 0, keepdim = True)
    getitem_156: "f32[1, 1024, 1, 1]" = var_mean_77[0]
    getitem_157: "f32[1, 1024, 1, 1]" = var_mean_77[1];  var_mean_77 = None
    add_403: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
    rsqrt_77: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_403);  add_403 = None
    sub_95: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_95, getitem_157)
    mul_557: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_77);  sub_95 = None
    squeeze_231: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
    squeeze_232: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
    mul_558: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
    mul_559: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_750, 0.9)
    add_404: "f32[1024]" = torch.ops.aten.add.Tensor(mul_558, mul_559);  mul_558 = mul_559 = None
    squeeze_233: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
    mul_560: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_233, 1.0004885197850513);  squeeze_233 = None
    mul_561: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_560, 0.1);  mul_560 = None
    mul_562: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_751, 0.9)
    add_405: "f32[1024]" = torch.ops.aten.add.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    unsqueeze_308: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_287, -1)
    unsqueeze_309: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_563: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_557, unsqueeze_309);  mul_557 = unsqueeze_309 = None
    unsqueeze_310: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_288, -1);  primals_288 = None
    unsqueeze_311: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_406: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_563, unsqueeze_311);  mul_563 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_407: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_406, relu_70);  add_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_74: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_407);  add_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_96: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_74, primals_289, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_408: "i64[]" = torch.ops.aten.add.Tensor(primals_755, 1)
    var_mean_78 = torch.ops.aten.var_mean.correction(convolution_96, [0, 2, 3], correction = 0, keepdim = True)
    getitem_158: "f32[1, 256, 1, 1]" = var_mean_78[0]
    getitem_159: "f32[1, 256, 1, 1]" = var_mean_78[1];  var_mean_78 = None
    add_409: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
    rsqrt_78: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_409);  add_409 = None
    sub_96: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_96, getitem_159)
    mul_564: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_78);  sub_96 = None
    squeeze_234: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
    squeeze_235: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
    mul_565: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1)
    mul_566: "f32[256]" = torch.ops.aten.mul.Tensor(primals_753, 0.9)
    add_410: "f32[256]" = torch.ops.aten.add.Tensor(mul_565, mul_566);  mul_565 = mul_566 = None
    squeeze_236: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
    mul_567: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_236, 1.0004885197850513);  squeeze_236 = None
    mul_568: "f32[256]" = torch.ops.aten.mul.Tensor(mul_567, 0.1);  mul_567 = None
    mul_569: "f32[256]" = torch.ops.aten.mul.Tensor(primals_754, 0.9)
    add_411: "f32[256]" = torch.ops.aten.add.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
    unsqueeze_312: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_290, -1)
    unsqueeze_313: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    mul_570: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_564, unsqueeze_313);  mul_564 = unsqueeze_313 = None
    unsqueeze_314: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_291, -1);  primals_291 = None
    unsqueeze_315: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    add_412: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_570, unsqueeze_315);  mul_570 = unsqueeze_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_75: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_412);  add_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_97: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_75, primals_292, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_413: "i64[]" = torch.ops.aten.add.Tensor(primals_758, 1)
    var_mean_79 = torch.ops.aten.var_mean.correction(convolution_97, [0, 2, 3], correction = 0, keepdim = True)
    getitem_160: "f32[1, 512, 1, 1]" = var_mean_79[0]
    getitem_161: "f32[1, 512, 1, 1]" = var_mean_79[1];  var_mean_79 = None
    add_414: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05)
    rsqrt_79: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_414);  add_414 = None
    sub_97: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_97, getitem_161)
    mul_571: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_79);  sub_97 = None
    squeeze_237: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3]);  getitem_161 = None
    squeeze_238: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
    mul_572: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1)
    mul_573: "f32[512]" = torch.ops.aten.mul.Tensor(primals_756, 0.9)
    add_415: "f32[512]" = torch.ops.aten.add.Tensor(mul_572, mul_573);  mul_572 = mul_573 = None
    squeeze_239: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_160, [0, 2, 3]);  getitem_160 = None
    mul_574: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_239, 1.0004885197850513);  squeeze_239 = None
    mul_575: "f32[512]" = torch.ops.aten.mul.Tensor(mul_574, 0.1);  mul_574 = None
    mul_576: "f32[512]" = torch.ops.aten.mul.Tensor(primals_757, 0.9)
    add_416: "f32[512]" = torch.ops.aten.add.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
    unsqueeze_316: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_293, -1)
    unsqueeze_317: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_577: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_571, unsqueeze_317);  mul_571 = unsqueeze_317 = None
    unsqueeze_318: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_294, -1);  primals_294 = None
    unsqueeze_319: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_417: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_577, unsqueeze_319);  mul_577 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_76: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_417);  add_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_109: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_76, [8, 2, 256, 16, 16])
    sum_55: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_109, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_18: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_55, [2, 3], True);  sum_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_98: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_18, primals_295, primals_296, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_418: "i64[]" = torch.ops.aten.add.Tensor(primals_761, 1)
    var_mean_80 = torch.ops.aten.var_mean.correction(convolution_98, [0, 2, 3], correction = 0, keepdim = True)
    getitem_162: "f32[1, 128, 1, 1]" = var_mean_80[0]
    getitem_163: "f32[1, 128, 1, 1]" = var_mean_80[1];  var_mean_80 = None
    add_419: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05)
    rsqrt_80: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_419);  add_419 = None
    sub_98: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_98, getitem_163)
    mul_578: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_80);  sub_98 = rsqrt_80 = None
    squeeze_240: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
    mul_579: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_240, 0.1);  squeeze_240 = None
    mul_580: "f32[128]" = torch.ops.aten.mul.Tensor(primals_759, 0.9)
    add_420: "f32[128]" = torch.ops.aten.add.Tensor(mul_579, mul_580);  mul_579 = mul_580 = None
    squeeze_242: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
    mul_581: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_242, 1.1428571428571428);  squeeze_242 = None
    mul_582: "f32[128]" = torch.ops.aten.mul.Tensor(mul_581, 0.1);  mul_581 = None
    mul_583: "f32[128]" = torch.ops.aten.mul.Tensor(primals_760, 0.9)
    add_421: "f32[128]" = torch.ops.aten.add.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    unsqueeze_320: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_297, -1)
    unsqueeze_321: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    mul_584: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_578, unsqueeze_321);  mul_578 = unsqueeze_321 = None
    unsqueeze_322: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_298, -1);  primals_298 = None
    unsqueeze_323: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    add_422: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_584, unsqueeze_323);  mul_584 = unsqueeze_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_77: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_422);  add_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_99: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_77, primals_299, primals_300, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_110: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_99, [8, 1, 2, -1]);  convolution_99 = None
    permute_18: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_18: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_18, [1], True)
    sub_99: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_18, amax_18);  permute_18 = amax_18 = None
    exp_18: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_99);  sub_99 = None
    sum_56: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_18, [1], True)
    div_18: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_18, sum_56);  exp_18 = sum_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_111: "f32[8, 512]" = torch.ops.aten.reshape.default(div_18, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_112: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_111, [8, -1, 1, 1]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_113: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_112, [8, 2, 256, 1, 1]);  view_112 = None
    mul_585: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_109, view_113);  view_109 = view_113 = None
    sum_57: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_585, [1]);  mul_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_100: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_57, primals_301, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_423: "i64[]" = torch.ops.aten.add.Tensor(primals_764, 1)
    var_mean_81 = torch.ops.aten.var_mean.correction(convolution_100, [0, 2, 3], correction = 0, keepdim = True)
    getitem_164: "f32[1, 1024, 1, 1]" = var_mean_81[0]
    getitem_165: "f32[1, 1024, 1, 1]" = var_mean_81[1];  var_mean_81 = None
    add_424: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-05)
    rsqrt_81: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_424);  add_424 = None
    sub_100: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_100, getitem_165)
    mul_586: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_81);  sub_100 = None
    squeeze_243: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
    squeeze_244: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
    mul_587: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_243, 0.1)
    mul_588: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_762, 0.9)
    add_425: "f32[1024]" = torch.ops.aten.add.Tensor(mul_587, mul_588);  mul_587 = mul_588 = None
    squeeze_245: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
    mul_589: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_245, 1.0004885197850513);  squeeze_245 = None
    mul_590: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_589, 0.1);  mul_589 = None
    mul_591: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_763, 0.9)
    add_426: "f32[1024]" = torch.ops.aten.add.Tensor(mul_590, mul_591);  mul_590 = mul_591 = None
    unsqueeze_324: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_302, -1)
    unsqueeze_325: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_592: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_586, unsqueeze_325);  mul_586 = unsqueeze_325 = None
    unsqueeze_326: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_303, -1);  primals_303 = None
    unsqueeze_327: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_427: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_592, unsqueeze_327);  mul_592 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_428: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_427, relu_74);  add_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_78: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_428);  add_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_101: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_78, primals_304, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_429: "i64[]" = torch.ops.aten.add.Tensor(primals_767, 1)
    var_mean_82 = torch.ops.aten.var_mean.correction(convolution_101, [0, 2, 3], correction = 0, keepdim = True)
    getitem_166: "f32[1, 256, 1, 1]" = var_mean_82[0]
    getitem_167: "f32[1, 256, 1, 1]" = var_mean_82[1];  var_mean_82 = None
    add_430: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05)
    rsqrt_82: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_430);  add_430 = None
    sub_101: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_101, getitem_167)
    mul_593: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_82);  sub_101 = None
    squeeze_246: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
    squeeze_247: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
    mul_594: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_246, 0.1)
    mul_595: "f32[256]" = torch.ops.aten.mul.Tensor(primals_765, 0.9)
    add_431: "f32[256]" = torch.ops.aten.add.Tensor(mul_594, mul_595);  mul_594 = mul_595 = None
    squeeze_248: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
    mul_596: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_248, 1.0004885197850513);  squeeze_248 = None
    mul_597: "f32[256]" = torch.ops.aten.mul.Tensor(mul_596, 0.1);  mul_596 = None
    mul_598: "f32[256]" = torch.ops.aten.mul.Tensor(primals_766, 0.9)
    add_432: "f32[256]" = torch.ops.aten.add.Tensor(mul_597, mul_598);  mul_597 = mul_598 = None
    unsqueeze_328: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_305, -1)
    unsqueeze_329: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    mul_599: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_593, unsqueeze_329);  mul_593 = unsqueeze_329 = None
    unsqueeze_330: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_306, -1);  primals_306 = None
    unsqueeze_331: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    add_433: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_599, unsqueeze_331);  mul_599 = unsqueeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_79: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_433);  add_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_102: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_79, primals_307, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_434: "i64[]" = torch.ops.aten.add.Tensor(primals_770, 1)
    var_mean_83 = torch.ops.aten.var_mean.correction(convolution_102, [0, 2, 3], correction = 0, keepdim = True)
    getitem_168: "f32[1, 512, 1, 1]" = var_mean_83[0]
    getitem_169: "f32[1, 512, 1, 1]" = var_mean_83[1];  var_mean_83 = None
    add_435: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-05)
    rsqrt_83: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_435);  add_435 = None
    sub_102: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_102, getitem_169)
    mul_600: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_83);  sub_102 = None
    squeeze_249: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
    squeeze_250: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
    mul_601: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_249, 0.1)
    mul_602: "f32[512]" = torch.ops.aten.mul.Tensor(primals_768, 0.9)
    add_436: "f32[512]" = torch.ops.aten.add.Tensor(mul_601, mul_602);  mul_601 = mul_602 = None
    squeeze_251: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
    mul_603: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_251, 1.0004885197850513);  squeeze_251 = None
    mul_604: "f32[512]" = torch.ops.aten.mul.Tensor(mul_603, 0.1);  mul_603 = None
    mul_605: "f32[512]" = torch.ops.aten.mul.Tensor(primals_769, 0.9)
    add_437: "f32[512]" = torch.ops.aten.add.Tensor(mul_604, mul_605);  mul_604 = mul_605 = None
    unsqueeze_332: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_308, -1)
    unsqueeze_333: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_606: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_600, unsqueeze_333);  mul_600 = unsqueeze_333 = None
    unsqueeze_334: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_309, -1);  primals_309 = None
    unsqueeze_335: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_438: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_606, unsqueeze_335);  mul_606 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_80: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_438);  add_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_115: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_80, [8, 2, 256, 16, 16])
    sum_58: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_115, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_19: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_58, [2, 3], True);  sum_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_103: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_19, primals_310, primals_311, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_439: "i64[]" = torch.ops.aten.add.Tensor(primals_773, 1)
    var_mean_84 = torch.ops.aten.var_mean.correction(convolution_103, [0, 2, 3], correction = 0, keepdim = True)
    getitem_170: "f32[1, 128, 1, 1]" = var_mean_84[0]
    getitem_171: "f32[1, 128, 1, 1]" = var_mean_84[1];  var_mean_84 = None
    add_440: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-05)
    rsqrt_84: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_440);  add_440 = None
    sub_103: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_103, getitem_171)
    mul_607: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_84);  sub_103 = rsqrt_84 = None
    squeeze_252: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
    mul_608: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_252, 0.1);  squeeze_252 = None
    mul_609: "f32[128]" = torch.ops.aten.mul.Tensor(primals_771, 0.9)
    add_441: "f32[128]" = torch.ops.aten.add.Tensor(mul_608, mul_609);  mul_608 = mul_609 = None
    squeeze_254: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_170, [0, 2, 3]);  getitem_170 = None
    mul_610: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_254, 1.1428571428571428);  squeeze_254 = None
    mul_611: "f32[128]" = torch.ops.aten.mul.Tensor(mul_610, 0.1);  mul_610 = None
    mul_612: "f32[128]" = torch.ops.aten.mul.Tensor(primals_772, 0.9)
    add_442: "f32[128]" = torch.ops.aten.add.Tensor(mul_611, mul_612);  mul_611 = mul_612 = None
    unsqueeze_336: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_312, -1)
    unsqueeze_337: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    mul_613: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_607, unsqueeze_337);  mul_607 = unsqueeze_337 = None
    unsqueeze_338: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_313, -1);  primals_313 = None
    unsqueeze_339: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    add_443: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_613, unsqueeze_339);  mul_613 = unsqueeze_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_81: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_443);  add_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_104: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_81, primals_314, primals_315, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_116: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_104, [8, 1, 2, -1]);  convolution_104 = None
    permute_19: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_19: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_19, [1], True)
    sub_104: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_19, amax_19);  permute_19 = amax_19 = None
    exp_19: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_104);  sub_104 = None
    sum_59: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_19, [1], True)
    div_19: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_19, sum_59);  exp_19 = sum_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_117: "f32[8, 512]" = torch.ops.aten.reshape.default(div_19, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_118: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_117, [8, -1, 1, 1]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_119: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_118, [8, 2, 256, 1, 1]);  view_118 = None
    mul_614: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_115, view_119);  view_115 = view_119 = None
    sum_60: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_614, [1]);  mul_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_105: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_60, primals_316, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_444: "i64[]" = torch.ops.aten.add.Tensor(primals_776, 1)
    var_mean_85 = torch.ops.aten.var_mean.correction(convolution_105, [0, 2, 3], correction = 0, keepdim = True)
    getitem_172: "f32[1, 1024, 1, 1]" = var_mean_85[0]
    getitem_173: "f32[1, 1024, 1, 1]" = var_mean_85[1];  var_mean_85 = None
    add_445: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05)
    rsqrt_85: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_445);  add_445 = None
    sub_105: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_105, getitem_173)
    mul_615: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_85);  sub_105 = None
    squeeze_255: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3]);  getitem_173 = None
    squeeze_256: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
    mul_616: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_255, 0.1)
    mul_617: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_774, 0.9)
    add_446: "f32[1024]" = torch.ops.aten.add.Tensor(mul_616, mul_617);  mul_616 = mul_617 = None
    squeeze_257: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_172, [0, 2, 3]);  getitem_172 = None
    mul_618: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_257, 1.0004885197850513);  squeeze_257 = None
    mul_619: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_618, 0.1);  mul_618 = None
    mul_620: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_775, 0.9)
    add_447: "f32[1024]" = torch.ops.aten.add.Tensor(mul_619, mul_620);  mul_619 = mul_620 = None
    unsqueeze_340: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_317, -1)
    unsqueeze_341: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_621: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_615, unsqueeze_341);  mul_615 = unsqueeze_341 = None
    unsqueeze_342: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_318, -1);  primals_318 = None
    unsqueeze_343: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_448: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_621, unsqueeze_343);  mul_621 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_449: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_448, relu_78);  add_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_82: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_449);  add_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_106: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_82, primals_319, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_450: "i64[]" = torch.ops.aten.add.Tensor(primals_779, 1)
    var_mean_86 = torch.ops.aten.var_mean.correction(convolution_106, [0, 2, 3], correction = 0, keepdim = True)
    getitem_174: "f32[1, 256, 1, 1]" = var_mean_86[0]
    getitem_175: "f32[1, 256, 1, 1]" = var_mean_86[1];  var_mean_86 = None
    add_451: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05)
    rsqrt_86: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_451);  add_451 = None
    sub_106: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_106, getitem_175)
    mul_622: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_86);  sub_106 = None
    squeeze_258: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
    squeeze_259: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_86, [0, 2, 3]);  rsqrt_86 = None
    mul_623: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_258, 0.1)
    mul_624: "f32[256]" = torch.ops.aten.mul.Tensor(primals_777, 0.9)
    add_452: "f32[256]" = torch.ops.aten.add.Tensor(mul_623, mul_624);  mul_623 = mul_624 = None
    squeeze_260: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_174, [0, 2, 3]);  getitem_174 = None
    mul_625: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_260, 1.0004885197850513);  squeeze_260 = None
    mul_626: "f32[256]" = torch.ops.aten.mul.Tensor(mul_625, 0.1);  mul_625 = None
    mul_627: "f32[256]" = torch.ops.aten.mul.Tensor(primals_778, 0.9)
    add_453: "f32[256]" = torch.ops.aten.add.Tensor(mul_626, mul_627);  mul_626 = mul_627 = None
    unsqueeze_344: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_320, -1)
    unsqueeze_345: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    mul_628: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_622, unsqueeze_345);  mul_622 = unsqueeze_345 = None
    unsqueeze_346: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_321, -1);  primals_321 = None
    unsqueeze_347: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    add_454: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_628, unsqueeze_347);  mul_628 = unsqueeze_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_83: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_454);  add_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_107: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_83, primals_322, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_455: "i64[]" = torch.ops.aten.add.Tensor(primals_782, 1)
    var_mean_87 = torch.ops.aten.var_mean.correction(convolution_107, [0, 2, 3], correction = 0, keepdim = True)
    getitem_176: "f32[1, 512, 1, 1]" = var_mean_87[0]
    getitem_177: "f32[1, 512, 1, 1]" = var_mean_87[1];  var_mean_87 = None
    add_456: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05)
    rsqrt_87: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_456);  add_456 = None
    sub_107: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_107, getitem_177)
    mul_629: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_87);  sub_107 = None
    squeeze_261: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_177, [0, 2, 3]);  getitem_177 = None
    squeeze_262: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
    mul_630: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_261, 0.1)
    mul_631: "f32[512]" = torch.ops.aten.mul.Tensor(primals_780, 0.9)
    add_457: "f32[512]" = torch.ops.aten.add.Tensor(mul_630, mul_631);  mul_630 = mul_631 = None
    squeeze_263: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_176, [0, 2, 3]);  getitem_176 = None
    mul_632: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_263, 1.0004885197850513);  squeeze_263 = None
    mul_633: "f32[512]" = torch.ops.aten.mul.Tensor(mul_632, 0.1);  mul_632 = None
    mul_634: "f32[512]" = torch.ops.aten.mul.Tensor(primals_781, 0.9)
    add_458: "f32[512]" = torch.ops.aten.add.Tensor(mul_633, mul_634);  mul_633 = mul_634 = None
    unsqueeze_348: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_323, -1)
    unsqueeze_349: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_635: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_629, unsqueeze_349);  mul_629 = unsqueeze_349 = None
    unsqueeze_350: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_324, -1);  primals_324 = None
    unsqueeze_351: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_459: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_635, unsqueeze_351);  mul_635 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_84: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_459);  add_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_121: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_84, [8, 2, 256, 16, 16])
    sum_61: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_121, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_20: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_61, [2, 3], True);  sum_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_108: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_20, primals_325, primals_326, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_460: "i64[]" = torch.ops.aten.add.Tensor(primals_785, 1)
    var_mean_88 = torch.ops.aten.var_mean.correction(convolution_108, [0, 2, 3], correction = 0, keepdim = True)
    getitem_178: "f32[1, 128, 1, 1]" = var_mean_88[0]
    getitem_179: "f32[1, 128, 1, 1]" = var_mean_88[1];  var_mean_88 = None
    add_461: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05)
    rsqrt_88: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_461);  add_461 = None
    sub_108: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_108, getitem_179)
    mul_636: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_88);  sub_108 = rsqrt_88 = None
    squeeze_264: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
    mul_637: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_264, 0.1);  squeeze_264 = None
    mul_638: "f32[128]" = torch.ops.aten.mul.Tensor(primals_783, 0.9)
    add_462: "f32[128]" = torch.ops.aten.add.Tensor(mul_637, mul_638);  mul_637 = mul_638 = None
    squeeze_266: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_178, [0, 2, 3]);  getitem_178 = None
    mul_639: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_266, 1.1428571428571428);  squeeze_266 = None
    mul_640: "f32[128]" = torch.ops.aten.mul.Tensor(mul_639, 0.1);  mul_639 = None
    mul_641: "f32[128]" = torch.ops.aten.mul.Tensor(primals_784, 0.9)
    add_463: "f32[128]" = torch.ops.aten.add.Tensor(mul_640, mul_641);  mul_640 = mul_641 = None
    unsqueeze_352: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_327, -1)
    unsqueeze_353: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    mul_642: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_636, unsqueeze_353);  mul_636 = unsqueeze_353 = None
    unsqueeze_354: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_328, -1);  primals_328 = None
    unsqueeze_355: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    add_464: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_642, unsqueeze_355);  mul_642 = unsqueeze_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_85: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_464);  add_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_109: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_85, primals_329, primals_330, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_122: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_109, [8, 1, 2, -1]);  convolution_109 = None
    permute_20: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_20: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_20, [1], True)
    sub_109: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_20, amax_20);  permute_20 = amax_20 = None
    exp_20: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_109);  sub_109 = None
    sum_62: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_20, [1], True)
    div_20: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_20, sum_62);  exp_20 = sum_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_123: "f32[8, 512]" = torch.ops.aten.reshape.default(div_20, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_124: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_123, [8, -1, 1, 1]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_125: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_124, [8, 2, 256, 1, 1]);  view_124 = None
    mul_643: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_121, view_125);  view_121 = view_125 = None
    sum_63: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_643, [1]);  mul_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_110: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_63, primals_331, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_465: "i64[]" = torch.ops.aten.add.Tensor(primals_788, 1)
    var_mean_89 = torch.ops.aten.var_mean.correction(convolution_110, [0, 2, 3], correction = 0, keepdim = True)
    getitem_180: "f32[1, 1024, 1, 1]" = var_mean_89[0]
    getitem_181: "f32[1, 1024, 1, 1]" = var_mean_89[1];  var_mean_89 = None
    add_466: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-05)
    rsqrt_89: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_466);  add_466 = None
    sub_110: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_110, getitem_181)
    mul_644: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_89);  sub_110 = None
    squeeze_267: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_181, [0, 2, 3]);  getitem_181 = None
    squeeze_268: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_89, [0, 2, 3]);  rsqrt_89 = None
    mul_645: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_267, 0.1)
    mul_646: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_786, 0.9)
    add_467: "f32[1024]" = torch.ops.aten.add.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
    squeeze_269: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_180, [0, 2, 3]);  getitem_180 = None
    mul_647: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_269, 1.0004885197850513);  squeeze_269 = None
    mul_648: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_647, 0.1);  mul_647 = None
    mul_649: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_787, 0.9)
    add_468: "f32[1024]" = torch.ops.aten.add.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
    unsqueeze_356: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_332, -1)
    unsqueeze_357: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_650: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_644, unsqueeze_357);  mul_644 = unsqueeze_357 = None
    unsqueeze_358: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_333, -1);  primals_333 = None
    unsqueeze_359: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_469: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_650, unsqueeze_359);  mul_650 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_470: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_469, relu_82);  add_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_86: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_470);  add_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_111: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_86, primals_334, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_471: "i64[]" = torch.ops.aten.add.Tensor(primals_791, 1)
    var_mean_90 = torch.ops.aten.var_mean.correction(convolution_111, [0, 2, 3], correction = 0, keepdim = True)
    getitem_182: "f32[1, 256, 1, 1]" = var_mean_90[0]
    getitem_183: "f32[1, 256, 1, 1]" = var_mean_90[1];  var_mean_90 = None
    add_472: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-05)
    rsqrt_90: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_472);  add_472 = None
    sub_111: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_111, getitem_183)
    mul_651: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_90);  sub_111 = None
    squeeze_270: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3]);  getitem_183 = None
    squeeze_271: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_90, [0, 2, 3]);  rsqrt_90 = None
    mul_652: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_270, 0.1)
    mul_653: "f32[256]" = torch.ops.aten.mul.Tensor(primals_789, 0.9)
    add_473: "f32[256]" = torch.ops.aten.add.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
    squeeze_272: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_182, [0, 2, 3]);  getitem_182 = None
    mul_654: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_272, 1.0004885197850513);  squeeze_272 = None
    mul_655: "f32[256]" = torch.ops.aten.mul.Tensor(mul_654, 0.1);  mul_654 = None
    mul_656: "f32[256]" = torch.ops.aten.mul.Tensor(primals_790, 0.9)
    add_474: "f32[256]" = torch.ops.aten.add.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
    unsqueeze_360: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_335, -1)
    unsqueeze_361: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    mul_657: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_651, unsqueeze_361);  mul_651 = unsqueeze_361 = None
    unsqueeze_362: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_336, -1);  primals_336 = None
    unsqueeze_363: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    add_475: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_657, unsqueeze_363);  mul_657 = unsqueeze_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_87: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_475);  add_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_112: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_87, primals_337, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_476: "i64[]" = torch.ops.aten.add.Tensor(primals_794, 1)
    var_mean_91 = torch.ops.aten.var_mean.correction(convolution_112, [0, 2, 3], correction = 0, keepdim = True)
    getitem_184: "f32[1, 512, 1, 1]" = var_mean_91[0]
    getitem_185: "f32[1, 512, 1, 1]" = var_mean_91[1];  var_mean_91 = None
    add_477: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-05)
    rsqrt_91: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_477);  add_477 = None
    sub_112: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_112, getitem_185)
    mul_658: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_91);  sub_112 = None
    squeeze_273: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_185, [0, 2, 3]);  getitem_185 = None
    squeeze_274: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_91, [0, 2, 3]);  rsqrt_91 = None
    mul_659: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_273, 0.1)
    mul_660: "f32[512]" = torch.ops.aten.mul.Tensor(primals_792, 0.9)
    add_478: "f32[512]" = torch.ops.aten.add.Tensor(mul_659, mul_660);  mul_659 = mul_660 = None
    squeeze_275: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_184, [0, 2, 3]);  getitem_184 = None
    mul_661: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_275, 1.0004885197850513);  squeeze_275 = None
    mul_662: "f32[512]" = torch.ops.aten.mul.Tensor(mul_661, 0.1);  mul_661 = None
    mul_663: "f32[512]" = torch.ops.aten.mul.Tensor(primals_793, 0.9)
    add_479: "f32[512]" = torch.ops.aten.add.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
    unsqueeze_364: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_338, -1)
    unsqueeze_365: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_664: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_365);  mul_658 = unsqueeze_365 = None
    unsqueeze_366: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_339, -1);  primals_339 = None
    unsqueeze_367: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_480: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_664, unsqueeze_367);  mul_664 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_88: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_480);  add_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_127: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_88, [8, 2, 256, 16, 16])
    sum_64: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_127, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_21: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_64, [2, 3], True);  sum_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_113: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_21, primals_340, primals_341, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_481: "i64[]" = torch.ops.aten.add.Tensor(primals_797, 1)
    var_mean_92 = torch.ops.aten.var_mean.correction(convolution_113, [0, 2, 3], correction = 0, keepdim = True)
    getitem_186: "f32[1, 128, 1, 1]" = var_mean_92[0]
    getitem_187: "f32[1, 128, 1, 1]" = var_mean_92[1];  var_mean_92 = None
    add_482: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05)
    rsqrt_92: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_482);  add_482 = None
    sub_113: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_113, getitem_187)
    mul_665: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_92);  sub_113 = rsqrt_92 = None
    squeeze_276: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
    mul_666: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_276, 0.1);  squeeze_276 = None
    mul_667: "f32[128]" = torch.ops.aten.mul.Tensor(primals_795, 0.9)
    add_483: "f32[128]" = torch.ops.aten.add.Tensor(mul_666, mul_667);  mul_666 = mul_667 = None
    squeeze_278: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_186, [0, 2, 3]);  getitem_186 = None
    mul_668: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_278, 1.1428571428571428);  squeeze_278 = None
    mul_669: "f32[128]" = torch.ops.aten.mul.Tensor(mul_668, 0.1);  mul_668 = None
    mul_670: "f32[128]" = torch.ops.aten.mul.Tensor(primals_796, 0.9)
    add_484: "f32[128]" = torch.ops.aten.add.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    unsqueeze_368: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_342, -1)
    unsqueeze_369: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    mul_671: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_665, unsqueeze_369);  mul_665 = unsqueeze_369 = None
    unsqueeze_370: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_343, -1);  primals_343 = None
    unsqueeze_371: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    add_485: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_671, unsqueeze_371);  mul_671 = unsqueeze_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_89: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_485);  add_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_114: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_89, primals_344, primals_345, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_128: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_114, [8, 1, 2, -1]);  convolution_114 = None
    permute_21: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_21: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_21, [1], True)
    sub_114: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_21, amax_21);  permute_21 = amax_21 = None
    exp_21: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_114);  sub_114 = None
    sum_65: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_21, [1], True)
    div_21: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_21, sum_65);  exp_21 = sum_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_129: "f32[8, 512]" = torch.ops.aten.reshape.default(div_21, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_130: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_129, [8, -1, 1, 1]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_131: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_130, [8, 2, 256, 1, 1]);  view_130 = None
    mul_672: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_127, view_131);  view_127 = view_131 = None
    sum_66: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_672, [1]);  mul_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_115: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_66, primals_346, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_486: "i64[]" = torch.ops.aten.add.Tensor(primals_800, 1)
    var_mean_93 = torch.ops.aten.var_mean.correction(convolution_115, [0, 2, 3], correction = 0, keepdim = True)
    getitem_188: "f32[1, 1024, 1, 1]" = var_mean_93[0]
    getitem_189: "f32[1, 1024, 1, 1]" = var_mean_93[1];  var_mean_93 = None
    add_487: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-05)
    rsqrt_93: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_487);  add_487 = None
    sub_115: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_115, getitem_189)
    mul_673: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_93);  sub_115 = None
    squeeze_279: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
    squeeze_280: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
    mul_674: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_279, 0.1)
    mul_675: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_798, 0.9)
    add_488: "f32[1024]" = torch.ops.aten.add.Tensor(mul_674, mul_675);  mul_674 = mul_675 = None
    squeeze_281: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_188, [0, 2, 3]);  getitem_188 = None
    mul_676: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_281, 1.0004885197850513);  squeeze_281 = None
    mul_677: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_676, 0.1);  mul_676 = None
    mul_678: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_799, 0.9)
    add_489: "f32[1024]" = torch.ops.aten.add.Tensor(mul_677, mul_678);  mul_677 = mul_678 = None
    unsqueeze_372: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_347, -1)
    unsqueeze_373: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_679: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_673, unsqueeze_373);  mul_673 = unsqueeze_373 = None
    unsqueeze_374: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_348, -1);  primals_348 = None
    unsqueeze_375: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_490: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_679, unsqueeze_375);  mul_679 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_491: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_490, relu_86);  add_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_90: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_491);  add_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_116: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_90, primals_349, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_492: "i64[]" = torch.ops.aten.add.Tensor(primals_803, 1)
    var_mean_94 = torch.ops.aten.var_mean.correction(convolution_116, [0, 2, 3], correction = 0, keepdim = True)
    getitem_190: "f32[1, 256, 1, 1]" = var_mean_94[0]
    getitem_191: "f32[1, 256, 1, 1]" = var_mean_94[1];  var_mean_94 = None
    add_493: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-05)
    rsqrt_94: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_493);  add_493 = None
    sub_116: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_116, getitem_191)
    mul_680: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_94);  sub_116 = None
    squeeze_282: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_191, [0, 2, 3]);  getitem_191 = None
    squeeze_283: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_94, [0, 2, 3]);  rsqrt_94 = None
    mul_681: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_282, 0.1)
    mul_682: "f32[256]" = torch.ops.aten.mul.Tensor(primals_801, 0.9)
    add_494: "f32[256]" = torch.ops.aten.add.Tensor(mul_681, mul_682);  mul_681 = mul_682 = None
    squeeze_284: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_190, [0, 2, 3]);  getitem_190 = None
    mul_683: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_284, 1.0004885197850513);  squeeze_284 = None
    mul_684: "f32[256]" = torch.ops.aten.mul.Tensor(mul_683, 0.1);  mul_683 = None
    mul_685: "f32[256]" = torch.ops.aten.mul.Tensor(primals_802, 0.9)
    add_495: "f32[256]" = torch.ops.aten.add.Tensor(mul_684, mul_685);  mul_684 = mul_685 = None
    unsqueeze_376: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_350, -1)
    unsqueeze_377: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    mul_686: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_680, unsqueeze_377);  mul_680 = unsqueeze_377 = None
    unsqueeze_378: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_351, -1);  primals_351 = None
    unsqueeze_379: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    add_496: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_686, unsqueeze_379);  mul_686 = unsqueeze_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_91: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_496);  add_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_117: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_91, primals_352, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_497: "i64[]" = torch.ops.aten.add.Tensor(primals_806, 1)
    var_mean_95 = torch.ops.aten.var_mean.correction(convolution_117, [0, 2, 3], correction = 0, keepdim = True)
    getitem_192: "f32[1, 512, 1, 1]" = var_mean_95[0]
    getitem_193: "f32[1, 512, 1, 1]" = var_mean_95[1];  var_mean_95 = None
    add_498: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05)
    rsqrt_95: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_498);  add_498 = None
    sub_117: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_117, getitem_193)
    mul_687: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_95);  sub_117 = None
    squeeze_285: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
    squeeze_286: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_95, [0, 2, 3]);  rsqrt_95 = None
    mul_688: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_285, 0.1)
    mul_689: "f32[512]" = torch.ops.aten.mul.Tensor(primals_804, 0.9)
    add_499: "f32[512]" = torch.ops.aten.add.Tensor(mul_688, mul_689);  mul_688 = mul_689 = None
    squeeze_287: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_192, [0, 2, 3]);  getitem_192 = None
    mul_690: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_287, 1.0004885197850513);  squeeze_287 = None
    mul_691: "f32[512]" = torch.ops.aten.mul.Tensor(mul_690, 0.1);  mul_690 = None
    mul_692: "f32[512]" = torch.ops.aten.mul.Tensor(primals_805, 0.9)
    add_500: "f32[512]" = torch.ops.aten.add.Tensor(mul_691, mul_692);  mul_691 = mul_692 = None
    unsqueeze_380: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_353, -1)
    unsqueeze_381: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_693: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_687, unsqueeze_381);  mul_687 = unsqueeze_381 = None
    unsqueeze_382: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_354, -1);  primals_354 = None
    unsqueeze_383: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_501: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_693, unsqueeze_383);  mul_693 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_92: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_501);  add_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_133: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_92, [8, 2, 256, 16, 16])
    sum_67: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_133, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_22: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_67, [2, 3], True);  sum_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_118: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_22, primals_355, primals_356, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_502: "i64[]" = torch.ops.aten.add.Tensor(primals_809, 1)
    var_mean_96 = torch.ops.aten.var_mean.correction(convolution_118, [0, 2, 3], correction = 0, keepdim = True)
    getitem_194: "f32[1, 128, 1, 1]" = var_mean_96[0]
    getitem_195: "f32[1, 128, 1, 1]" = var_mean_96[1];  var_mean_96 = None
    add_503: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-05)
    rsqrt_96: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_503);  add_503 = None
    sub_118: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_118, getitem_195)
    mul_694: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_96);  sub_118 = rsqrt_96 = None
    squeeze_288: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3]);  getitem_195 = None
    mul_695: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_288, 0.1);  squeeze_288 = None
    mul_696: "f32[128]" = torch.ops.aten.mul.Tensor(primals_807, 0.9)
    add_504: "f32[128]" = torch.ops.aten.add.Tensor(mul_695, mul_696);  mul_695 = mul_696 = None
    squeeze_290: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_194, [0, 2, 3]);  getitem_194 = None
    mul_697: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_290, 1.1428571428571428);  squeeze_290 = None
    mul_698: "f32[128]" = torch.ops.aten.mul.Tensor(mul_697, 0.1);  mul_697 = None
    mul_699: "f32[128]" = torch.ops.aten.mul.Tensor(primals_808, 0.9)
    add_505: "f32[128]" = torch.ops.aten.add.Tensor(mul_698, mul_699);  mul_698 = mul_699 = None
    unsqueeze_384: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_357, -1)
    unsqueeze_385: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    mul_700: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_694, unsqueeze_385);  mul_694 = unsqueeze_385 = None
    unsqueeze_386: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_358, -1);  primals_358 = None
    unsqueeze_387: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    add_506: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_700, unsqueeze_387);  mul_700 = unsqueeze_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_93: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_506);  add_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_119: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_93, primals_359, primals_360, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_134: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_119, [8, 1, 2, -1]);  convolution_119 = None
    permute_22: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_22: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_22, [1], True)
    sub_119: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_22, amax_22);  permute_22 = amax_22 = None
    exp_22: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_119);  sub_119 = None
    sum_68: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_22, [1], True)
    div_22: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_22, sum_68);  exp_22 = sum_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_135: "f32[8, 512]" = torch.ops.aten.reshape.default(div_22, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_136: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_135, [8, -1, 1, 1]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_137: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_136, [8, 2, 256, 1, 1]);  view_136 = None
    mul_701: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_133, view_137);  view_133 = view_137 = None
    sum_69: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_701, [1]);  mul_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_120: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_69, primals_361, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_507: "i64[]" = torch.ops.aten.add.Tensor(primals_812, 1)
    var_mean_97 = torch.ops.aten.var_mean.correction(convolution_120, [0, 2, 3], correction = 0, keepdim = True)
    getitem_196: "f32[1, 1024, 1, 1]" = var_mean_97[0]
    getitem_197: "f32[1, 1024, 1, 1]" = var_mean_97[1];  var_mean_97 = None
    add_508: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-05)
    rsqrt_97: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_508);  add_508 = None
    sub_120: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_120, getitem_197)
    mul_702: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_97);  sub_120 = None
    squeeze_291: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_197, [0, 2, 3]);  getitem_197 = None
    squeeze_292: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_97, [0, 2, 3]);  rsqrt_97 = None
    mul_703: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_291, 0.1)
    mul_704: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_810, 0.9)
    add_509: "f32[1024]" = torch.ops.aten.add.Tensor(mul_703, mul_704);  mul_703 = mul_704 = None
    squeeze_293: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_196, [0, 2, 3]);  getitem_196 = None
    mul_705: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_293, 1.0004885197850513);  squeeze_293 = None
    mul_706: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_705, 0.1);  mul_705 = None
    mul_707: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_811, 0.9)
    add_510: "f32[1024]" = torch.ops.aten.add.Tensor(mul_706, mul_707);  mul_706 = mul_707 = None
    unsqueeze_388: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_362, -1)
    unsqueeze_389: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_708: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_702, unsqueeze_389);  mul_702 = unsqueeze_389 = None
    unsqueeze_390: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_363, -1);  primals_363 = None
    unsqueeze_391: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_511: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_708, unsqueeze_391);  mul_708 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_512: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_511, relu_90);  add_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_94: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_512);  add_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_121: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_94, primals_364, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_513: "i64[]" = torch.ops.aten.add.Tensor(primals_815, 1)
    var_mean_98 = torch.ops.aten.var_mean.correction(convolution_121, [0, 2, 3], correction = 0, keepdim = True)
    getitem_198: "f32[1, 256, 1, 1]" = var_mean_98[0]
    getitem_199: "f32[1, 256, 1, 1]" = var_mean_98[1];  var_mean_98 = None
    add_514: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05)
    rsqrt_98: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_514);  add_514 = None
    sub_121: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_121, getitem_199)
    mul_709: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_98);  sub_121 = None
    squeeze_294: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_199, [0, 2, 3]);  getitem_199 = None
    squeeze_295: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_98, [0, 2, 3]);  rsqrt_98 = None
    mul_710: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_294, 0.1)
    mul_711: "f32[256]" = torch.ops.aten.mul.Tensor(primals_813, 0.9)
    add_515: "f32[256]" = torch.ops.aten.add.Tensor(mul_710, mul_711);  mul_710 = mul_711 = None
    squeeze_296: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_198, [0, 2, 3]);  getitem_198 = None
    mul_712: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_296, 1.0004885197850513);  squeeze_296 = None
    mul_713: "f32[256]" = torch.ops.aten.mul.Tensor(mul_712, 0.1);  mul_712 = None
    mul_714: "f32[256]" = torch.ops.aten.mul.Tensor(primals_814, 0.9)
    add_516: "f32[256]" = torch.ops.aten.add.Tensor(mul_713, mul_714);  mul_713 = mul_714 = None
    unsqueeze_392: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_365, -1)
    unsqueeze_393: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    mul_715: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_709, unsqueeze_393);  mul_709 = unsqueeze_393 = None
    unsqueeze_394: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_366, -1);  primals_366 = None
    unsqueeze_395: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    add_517: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_715, unsqueeze_395);  mul_715 = unsqueeze_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_95: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_517);  add_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_122: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_95, primals_367, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_518: "i64[]" = torch.ops.aten.add.Tensor(primals_818, 1)
    var_mean_99 = torch.ops.aten.var_mean.correction(convolution_122, [0, 2, 3], correction = 0, keepdim = True)
    getitem_200: "f32[1, 512, 1, 1]" = var_mean_99[0]
    getitem_201: "f32[1, 512, 1, 1]" = var_mean_99[1];  var_mean_99 = None
    add_519: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-05)
    rsqrt_99: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_519);  add_519 = None
    sub_122: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_122, getitem_201)
    mul_716: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_99);  sub_122 = None
    squeeze_297: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_201, [0, 2, 3]);  getitem_201 = None
    squeeze_298: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_99, [0, 2, 3]);  rsqrt_99 = None
    mul_717: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_297, 0.1)
    mul_718: "f32[512]" = torch.ops.aten.mul.Tensor(primals_816, 0.9)
    add_520: "f32[512]" = torch.ops.aten.add.Tensor(mul_717, mul_718);  mul_717 = mul_718 = None
    squeeze_299: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_200, [0, 2, 3]);  getitem_200 = None
    mul_719: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_299, 1.0004885197850513);  squeeze_299 = None
    mul_720: "f32[512]" = torch.ops.aten.mul.Tensor(mul_719, 0.1);  mul_719 = None
    mul_721: "f32[512]" = torch.ops.aten.mul.Tensor(primals_817, 0.9)
    add_521: "f32[512]" = torch.ops.aten.add.Tensor(mul_720, mul_721);  mul_720 = mul_721 = None
    unsqueeze_396: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_368, -1)
    unsqueeze_397: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_722: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_716, unsqueeze_397);  mul_716 = unsqueeze_397 = None
    unsqueeze_398: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_369, -1);  primals_369 = None
    unsqueeze_399: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_522: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_722, unsqueeze_399);  mul_722 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_96: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_522);  add_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_139: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_96, [8, 2, 256, 16, 16])
    sum_70: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_139, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_23: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_70, [2, 3], True);  sum_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_123: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_23, primals_370, primals_371, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_523: "i64[]" = torch.ops.aten.add.Tensor(primals_821, 1)
    var_mean_100 = torch.ops.aten.var_mean.correction(convolution_123, [0, 2, 3], correction = 0, keepdim = True)
    getitem_202: "f32[1, 128, 1, 1]" = var_mean_100[0]
    getitem_203: "f32[1, 128, 1, 1]" = var_mean_100[1];  var_mean_100 = None
    add_524: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_202, 1e-05)
    rsqrt_100: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_524);  add_524 = None
    sub_123: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_123, getitem_203)
    mul_723: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_123, rsqrt_100);  sub_123 = rsqrt_100 = None
    squeeze_300: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_203, [0, 2, 3]);  getitem_203 = None
    mul_724: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_300, 0.1);  squeeze_300 = None
    mul_725: "f32[128]" = torch.ops.aten.mul.Tensor(primals_819, 0.9)
    add_525: "f32[128]" = torch.ops.aten.add.Tensor(mul_724, mul_725);  mul_724 = mul_725 = None
    squeeze_302: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_202, [0, 2, 3]);  getitem_202 = None
    mul_726: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_302, 1.1428571428571428);  squeeze_302 = None
    mul_727: "f32[128]" = torch.ops.aten.mul.Tensor(mul_726, 0.1);  mul_726 = None
    mul_728: "f32[128]" = torch.ops.aten.mul.Tensor(primals_820, 0.9)
    add_526: "f32[128]" = torch.ops.aten.add.Tensor(mul_727, mul_728);  mul_727 = mul_728 = None
    unsqueeze_400: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_372, -1)
    unsqueeze_401: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    mul_729: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_723, unsqueeze_401);  mul_723 = unsqueeze_401 = None
    unsqueeze_402: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_373, -1);  primals_373 = None
    unsqueeze_403: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    add_527: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_729, unsqueeze_403);  mul_729 = unsqueeze_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_97: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_527);  add_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_124: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_97, primals_374, primals_375, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_140: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_124, [8, 1, 2, -1]);  convolution_124 = None
    permute_23: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_23: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_23, [1], True)
    sub_124: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_23, amax_23);  permute_23 = amax_23 = None
    exp_23: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_124);  sub_124 = None
    sum_71: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_23, [1], True)
    div_23: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_23, sum_71);  exp_23 = sum_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_141: "f32[8, 512]" = torch.ops.aten.reshape.default(div_23, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_142: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_141, [8, -1, 1, 1]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_143: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_142, [8, 2, 256, 1, 1]);  view_142 = None
    mul_730: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_139, view_143);  view_139 = view_143 = None
    sum_72: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_730, [1]);  mul_730 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_125: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_72, primals_376, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_528: "i64[]" = torch.ops.aten.add.Tensor(primals_824, 1)
    var_mean_101 = torch.ops.aten.var_mean.correction(convolution_125, [0, 2, 3], correction = 0, keepdim = True)
    getitem_204: "f32[1, 1024, 1, 1]" = var_mean_101[0]
    getitem_205: "f32[1, 1024, 1, 1]" = var_mean_101[1];  var_mean_101 = None
    add_529: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_204, 1e-05)
    rsqrt_101: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_529);  add_529 = None
    sub_125: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_125, getitem_205)
    mul_731: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_101);  sub_125 = None
    squeeze_303: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_205, [0, 2, 3]);  getitem_205 = None
    squeeze_304: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_101, [0, 2, 3]);  rsqrt_101 = None
    mul_732: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_303, 0.1)
    mul_733: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_822, 0.9)
    add_530: "f32[1024]" = torch.ops.aten.add.Tensor(mul_732, mul_733);  mul_732 = mul_733 = None
    squeeze_305: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_204, [0, 2, 3]);  getitem_204 = None
    mul_734: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_305, 1.0004885197850513);  squeeze_305 = None
    mul_735: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_734, 0.1);  mul_734 = None
    mul_736: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_823, 0.9)
    add_531: "f32[1024]" = torch.ops.aten.add.Tensor(mul_735, mul_736);  mul_735 = mul_736 = None
    unsqueeze_404: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_377, -1)
    unsqueeze_405: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_737: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_731, unsqueeze_405);  mul_731 = unsqueeze_405 = None
    unsqueeze_406: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_378, -1);  primals_378 = None
    unsqueeze_407: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_532: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_737, unsqueeze_407);  mul_737 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_533: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_532, relu_94);  add_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_98: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_533);  add_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_126: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_98, primals_379, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_534: "i64[]" = torch.ops.aten.add.Tensor(primals_827, 1)
    var_mean_102 = torch.ops.aten.var_mean.correction(convolution_126, [0, 2, 3], correction = 0, keepdim = True)
    getitem_206: "f32[1, 256, 1, 1]" = var_mean_102[0]
    getitem_207: "f32[1, 256, 1, 1]" = var_mean_102[1];  var_mean_102 = None
    add_535: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-05)
    rsqrt_102: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_535);  add_535 = None
    sub_126: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_126, getitem_207)
    mul_738: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_102);  sub_126 = None
    squeeze_306: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_207, [0, 2, 3]);  getitem_207 = None
    squeeze_307: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_102, [0, 2, 3]);  rsqrt_102 = None
    mul_739: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_306, 0.1)
    mul_740: "f32[256]" = torch.ops.aten.mul.Tensor(primals_825, 0.9)
    add_536: "f32[256]" = torch.ops.aten.add.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
    squeeze_308: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_206, [0, 2, 3]);  getitem_206 = None
    mul_741: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_308, 1.0004885197850513);  squeeze_308 = None
    mul_742: "f32[256]" = torch.ops.aten.mul.Tensor(mul_741, 0.1);  mul_741 = None
    mul_743: "f32[256]" = torch.ops.aten.mul.Tensor(primals_826, 0.9)
    add_537: "f32[256]" = torch.ops.aten.add.Tensor(mul_742, mul_743);  mul_742 = mul_743 = None
    unsqueeze_408: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_380, -1)
    unsqueeze_409: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    mul_744: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_738, unsqueeze_409);  mul_738 = unsqueeze_409 = None
    unsqueeze_410: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_381, -1);  primals_381 = None
    unsqueeze_411: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    add_538: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_744, unsqueeze_411);  mul_744 = unsqueeze_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_99: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_538);  add_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_127: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_99, primals_382, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_539: "i64[]" = torch.ops.aten.add.Tensor(primals_830, 1)
    var_mean_103 = torch.ops.aten.var_mean.correction(convolution_127, [0, 2, 3], correction = 0, keepdim = True)
    getitem_208: "f32[1, 512, 1, 1]" = var_mean_103[0]
    getitem_209: "f32[1, 512, 1, 1]" = var_mean_103[1];  var_mean_103 = None
    add_540: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-05)
    rsqrt_103: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_540);  add_540 = None
    sub_127: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_127, getitem_209)
    mul_745: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_103);  sub_127 = None
    squeeze_309: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_209, [0, 2, 3]);  getitem_209 = None
    squeeze_310: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_103, [0, 2, 3]);  rsqrt_103 = None
    mul_746: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_309, 0.1)
    mul_747: "f32[512]" = torch.ops.aten.mul.Tensor(primals_828, 0.9)
    add_541: "f32[512]" = torch.ops.aten.add.Tensor(mul_746, mul_747);  mul_746 = mul_747 = None
    squeeze_311: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_208, [0, 2, 3]);  getitem_208 = None
    mul_748: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_311, 1.0004885197850513);  squeeze_311 = None
    mul_749: "f32[512]" = torch.ops.aten.mul.Tensor(mul_748, 0.1);  mul_748 = None
    mul_750: "f32[512]" = torch.ops.aten.mul.Tensor(primals_829, 0.9)
    add_542: "f32[512]" = torch.ops.aten.add.Tensor(mul_749, mul_750);  mul_749 = mul_750 = None
    unsqueeze_412: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_383, -1)
    unsqueeze_413: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_751: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_745, unsqueeze_413);  mul_745 = unsqueeze_413 = None
    unsqueeze_414: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_384, -1);  primals_384 = None
    unsqueeze_415: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_543: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_751, unsqueeze_415);  mul_751 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_100: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_543);  add_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_145: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_100, [8, 2, 256, 16, 16])
    sum_73: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_145, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_24: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_73, [2, 3], True);  sum_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_128: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_24, primals_385, primals_386, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_544: "i64[]" = torch.ops.aten.add.Tensor(primals_833, 1)
    var_mean_104 = torch.ops.aten.var_mean.correction(convolution_128, [0, 2, 3], correction = 0, keepdim = True)
    getitem_210: "f32[1, 128, 1, 1]" = var_mean_104[0]
    getitem_211: "f32[1, 128, 1, 1]" = var_mean_104[1];  var_mean_104 = None
    add_545: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-05)
    rsqrt_104: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_545);  add_545 = None
    sub_128: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_128, getitem_211)
    mul_752: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_104);  sub_128 = rsqrt_104 = None
    squeeze_312: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_211, [0, 2, 3]);  getitem_211 = None
    mul_753: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_312, 0.1);  squeeze_312 = None
    mul_754: "f32[128]" = torch.ops.aten.mul.Tensor(primals_831, 0.9)
    add_546: "f32[128]" = torch.ops.aten.add.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
    squeeze_314: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_210, [0, 2, 3]);  getitem_210 = None
    mul_755: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_314, 1.1428571428571428);  squeeze_314 = None
    mul_756: "f32[128]" = torch.ops.aten.mul.Tensor(mul_755, 0.1);  mul_755 = None
    mul_757: "f32[128]" = torch.ops.aten.mul.Tensor(primals_832, 0.9)
    add_547: "f32[128]" = torch.ops.aten.add.Tensor(mul_756, mul_757);  mul_756 = mul_757 = None
    unsqueeze_416: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_387, -1)
    unsqueeze_417: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    mul_758: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_752, unsqueeze_417);  mul_752 = unsqueeze_417 = None
    unsqueeze_418: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_388, -1);  primals_388 = None
    unsqueeze_419: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    add_548: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_758, unsqueeze_419);  mul_758 = unsqueeze_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_101: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_548);  add_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_129: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_101, primals_389, primals_390, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_146: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_129, [8, 1, 2, -1]);  convolution_129 = None
    permute_24: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_24: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_24, [1], True)
    sub_129: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_24, amax_24);  permute_24 = amax_24 = None
    exp_24: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_129);  sub_129 = None
    sum_74: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True)
    div_24: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_24, sum_74);  exp_24 = sum_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_147: "f32[8, 512]" = torch.ops.aten.reshape.default(div_24, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_148: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_147, [8, -1, 1, 1]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_149: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_148, [8, 2, 256, 1, 1]);  view_148 = None
    mul_759: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_145, view_149);  view_145 = view_149 = None
    sum_75: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_759, [1]);  mul_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_130: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_75, primals_391, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_549: "i64[]" = torch.ops.aten.add.Tensor(primals_836, 1)
    var_mean_105 = torch.ops.aten.var_mean.correction(convolution_130, [0, 2, 3], correction = 0, keepdim = True)
    getitem_212: "f32[1, 1024, 1, 1]" = var_mean_105[0]
    getitem_213: "f32[1, 1024, 1, 1]" = var_mean_105[1];  var_mean_105 = None
    add_550: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-05)
    rsqrt_105: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_550);  add_550 = None
    sub_130: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_130, getitem_213)
    mul_760: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_130, rsqrt_105);  sub_130 = None
    squeeze_315: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_213, [0, 2, 3]);  getitem_213 = None
    squeeze_316: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_105, [0, 2, 3]);  rsqrt_105 = None
    mul_761: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_315, 0.1)
    mul_762: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_834, 0.9)
    add_551: "f32[1024]" = torch.ops.aten.add.Tensor(mul_761, mul_762);  mul_761 = mul_762 = None
    squeeze_317: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_212, [0, 2, 3]);  getitem_212 = None
    mul_763: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_317, 1.0004885197850513);  squeeze_317 = None
    mul_764: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_763, 0.1);  mul_763 = None
    mul_765: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_835, 0.9)
    add_552: "f32[1024]" = torch.ops.aten.add.Tensor(mul_764, mul_765);  mul_764 = mul_765 = None
    unsqueeze_420: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_392, -1)
    unsqueeze_421: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_766: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_760, unsqueeze_421);  mul_760 = unsqueeze_421 = None
    unsqueeze_422: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_393, -1);  primals_393 = None
    unsqueeze_423: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_553: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_766, unsqueeze_423);  mul_766 = unsqueeze_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_554: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_553, relu_98);  add_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_102: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_554);  add_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_131: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_102, primals_394, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_555: "i64[]" = torch.ops.aten.add.Tensor(primals_839, 1)
    var_mean_106 = torch.ops.aten.var_mean.correction(convolution_131, [0, 2, 3], correction = 0, keepdim = True)
    getitem_214: "f32[1, 256, 1, 1]" = var_mean_106[0]
    getitem_215: "f32[1, 256, 1, 1]" = var_mean_106[1];  var_mean_106 = None
    add_556: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_214, 1e-05)
    rsqrt_106: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_556);  add_556 = None
    sub_131: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_131, getitem_215)
    mul_767: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_106);  sub_131 = None
    squeeze_318: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_215, [0, 2, 3]);  getitem_215 = None
    squeeze_319: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_106, [0, 2, 3]);  rsqrt_106 = None
    mul_768: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_318, 0.1)
    mul_769: "f32[256]" = torch.ops.aten.mul.Tensor(primals_837, 0.9)
    add_557: "f32[256]" = torch.ops.aten.add.Tensor(mul_768, mul_769);  mul_768 = mul_769 = None
    squeeze_320: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_214, [0, 2, 3]);  getitem_214 = None
    mul_770: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_320, 1.0004885197850513);  squeeze_320 = None
    mul_771: "f32[256]" = torch.ops.aten.mul.Tensor(mul_770, 0.1);  mul_770 = None
    mul_772: "f32[256]" = torch.ops.aten.mul.Tensor(primals_838, 0.9)
    add_558: "f32[256]" = torch.ops.aten.add.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
    unsqueeze_424: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_395, -1)
    unsqueeze_425: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    mul_773: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_767, unsqueeze_425);  mul_767 = unsqueeze_425 = None
    unsqueeze_426: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_396, -1);  primals_396 = None
    unsqueeze_427: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    add_559: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_773, unsqueeze_427);  mul_773 = unsqueeze_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_103: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_559);  add_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_132: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_103, primals_397, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_560: "i64[]" = torch.ops.aten.add.Tensor(primals_842, 1)
    var_mean_107 = torch.ops.aten.var_mean.correction(convolution_132, [0, 2, 3], correction = 0, keepdim = True)
    getitem_216: "f32[1, 512, 1, 1]" = var_mean_107[0]
    getitem_217: "f32[1, 512, 1, 1]" = var_mean_107[1];  var_mean_107 = None
    add_561: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_216, 1e-05)
    rsqrt_107: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_561);  add_561 = None
    sub_132: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_132, getitem_217)
    mul_774: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_132, rsqrt_107);  sub_132 = None
    squeeze_321: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_217, [0, 2, 3]);  getitem_217 = None
    squeeze_322: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_107, [0, 2, 3]);  rsqrt_107 = None
    mul_775: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_321, 0.1)
    mul_776: "f32[512]" = torch.ops.aten.mul.Tensor(primals_840, 0.9)
    add_562: "f32[512]" = torch.ops.aten.add.Tensor(mul_775, mul_776);  mul_775 = mul_776 = None
    squeeze_323: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_216, [0, 2, 3]);  getitem_216 = None
    mul_777: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_323, 1.0004885197850513);  squeeze_323 = None
    mul_778: "f32[512]" = torch.ops.aten.mul.Tensor(mul_777, 0.1);  mul_777 = None
    mul_779: "f32[512]" = torch.ops.aten.mul.Tensor(primals_841, 0.9)
    add_563: "f32[512]" = torch.ops.aten.add.Tensor(mul_778, mul_779);  mul_778 = mul_779 = None
    unsqueeze_428: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_398, -1)
    unsqueeze_429: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_780: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_774, unsqueeze_429);  mul_774 = unsqueeze_429 = None
    unsqueeze_430: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_399, -1);  primals_399 = None
    unsqueeze_431: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_564: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_780, unsqueeze_431);  mul_780 = unsqueeze_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_104: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_564);  add_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_151: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_104, [8, 2, 256, 16, 16])
    sum_76: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_151, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_25: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_76, [2, 3], True);  sum_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_133: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_25, primals_400, primals_401, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_565: "i64[]" = torch.ops.aten.add.Tensor(primals_845, 1)
    var_mean_108 = torch.ops.aten.var_mean.correction(convolution_133, [0, 2, 3], correction = 0, keepdim = True)
    getitem_218: "f32[1, 128, 1, 1]" = var_mean_108[0]
    getitem_219: "f32[1, 128, 1, 1]" = var_mean_108[1];  var_mean_108 = None
    add_566: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-05)
    rsqrt_108: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_566);  add_566 = None
    sub_133: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_133, getitem_219)
    mul_781: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_133, rsqrt_108);  sub_133 = rsqrt_108 = None
    squeeze_324: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_219, [0, 2, 3]);  getitem_219 = None
    mul_782: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_324, 0.1);  squeeze_324 = None
    mul_783: "f32[128]" = torch.ops.aten.mul.Tensor(primals_843, 0.9)
    add_567: "f32[128]" = torch.ops.aten.add.Tensor(mul_782, mul_783);  mul_782 = mul_783 = None
    squeeze_326: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_218, [0, 2, 3]);  getitem_218 = None
    mul_784: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_326, 1.1428571428571428);  squeeze_326 = None
    mul_785: "f32[128]" = torch.ops.aten.mul.Tensor(mul_784, 0.1);  mul_784 = None
    mul_786: "f32[128]" = torch.ops.aten.mul.Tensor(primals_844, 0.9)
    add_568: "f32[128]" = torch.ops.aten.add.Tensor(mul_785, mul_786);  mul_785 = mul_786 = None
    unsqueeze_432: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_402, -1)
    unsqueeze_433: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    mul_787: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_781, unsqueeze_433);  mul_781 = unsqueeze_433 = None
    unsqueeze_434: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_403, -1);  primals_403 = None
    unsqueeze_435: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    add_569: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_787, unsqueeze_435);  mul_787 = unsqueeze_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_105: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_569);  add_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_134: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_105, primals_404, primals_405, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_152: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_134, [8, 1, 2, -1]);  convolution_134 = None
    permute_25: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_25: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_25, [1], True)
    sub_134: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_25, amax_25);  permute_25 = amax_25 = None
    exp_25: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_134);  sub_134 = None
    sum_77: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_25, [1], True)
    div_25: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_25, sum_77);  exp_25 = sum_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_153: "f32[8, 512]" = torch.ops.aten.reshape.default(div_25, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_154: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_153, [8, -1, 1, 1]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_155: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_154, [8, 2, 256, 1, 1]);  view_154 = None
    mul_788: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_151, view_155);  view_151 = view_155 = None
    sum_78: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_788, [1]);  mul_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_135: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_78, primals_406, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_570: "i64[]" = torch.ops.aten.add.Tensor(primals_848, 1)
    var_mean_109 = torch.ops.aten.var_mean.correction(convolution_135, [0, 2, 3], correction = 0, keepdim = True)
    getitem_220: "f32[1, 1024, 1, 1]" = var_mean_109[0]
    getitem_221: "f32[1, 1024, 1, 1]" = var_mean_109[1];  var_mean_109 = None
    add_571: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-05)
    rsqrt_109: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_571);  add_571 = None
    sub_135: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_135, getitem_221)
    mul_789: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_135, rsqrt_109);  sub_135 = None
    squeeze_327: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_221, [0, 2, 3]);  getitem_221 = None
    squeeze_328: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_109, [0, 2, 3]);  rsqrt_109 = None
    mul_790: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_327, 0.1)
    mul_791: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_846, 0.9)
    add_572: "f32[1024]" = torch.ops.aten.add.Tensor(mul_790, mul_791);  mul_790 = mul_791 = None
    squeeze_329: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_220, [0, 2, 3]);  getitem_220 = None
    mul_792: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_329, 1.0004885197850513);  squeeze_329 = None
    mul_793: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_792, 0.1);  mul_792 = None
    mul_794: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_847, 0.9)
    add_573: "f32[1024]" = torch.ops.aten.add.Tensor(mul_793, mul_794);  mul_793 = mul_794 = None
    unsqueeze_436: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_407, -1)
    unsqueeze_437: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_795: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_789, unsqueeze_437);  mul_789 = unsqueeze_437 = None
    unsqueeze_438: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_408, -1);  primals_408 = None
    unsqueeze_439: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_574: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_795, unsqueeze_439);  mul_795 = unsqueeze_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_575: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_574, relu_102);  add_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_106: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_575);  add_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_136: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_106, primals_409, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_576: "i64[]" = torch.ops.aten.add.Tensor(primals_851, 1)
    var_mean_110 = torch.ops.aten.var_mean.correction(convolution_136, [0, 2, 3], correction = 0, keepdim = True)
    getitem_222: "f32[1, 256, 1, 1]" = var_mean_110[0]
    getitem_223: "f32[1, 256, 1, 1]" = var_mean_110[1];  var_mean_110 = None
    add_577: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-05)
    rsqrt_110: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_577);  add_577 = None
    sub_136: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_136, getitem_223)
    mul_796: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_136, rsqrt_110);  sub_136 = None
    squeeze_330: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_223, [0, 2, 3]);  getitem_223 = None
    squeeze_331: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_110, [0, 2, 3]);  rsqrt_110 = None
    mul_797: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_330, 0.1)
    mul_798: "f32[256]" = torch.ops.aten.mul.Tensor(primals_849, 0.9)
    add_578: "f32[256]" = torch.ops.aten.add.Tensor(mul_797, mul_798);  mul_797 = mul_798 = None
    squeeze_332: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_222, [0, 2, 3]);  getitem_222 = None
    mul_799: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_332, 1.0004885197850513);  squeeze_332 = None
    mul_800: "f32[256]" = torch.ops.aten.mul.Tensor(mul_799, 0.1);  mul_799 = None
    mul_801: "f32[256]" = torch.ops.aten.mul.Tensor(primals_850, 0.9)
    add_579: "f32[256]" = torch.ops.aten.add.Tensor(mul_800, mul_801);  mul_800 = mul_801 = None
    unsqueeze_440: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_410, -1)
    unsqueeze_441: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    mul_802: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_796, unsqueeze_441);  mul_796 = unsqueeze_441 = None
    unsqueeze_442: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_411, -1);  primals_411 = None
    unsqueeze_443: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    add_580: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_802, unsqueeze_443);  mul_802 = unsqueeze_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_107: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_580);  add_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_137: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_107, primals_412, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_581: "i64[]" = torch.ops.aten.add.Tensor(primals_854, 1)
    var_mean_111 = torch.ops.aten.var_mean.correction(convolution_137, [0, 2, 3], correction = 0, keepdim = True)
    getitem_224: "f32[1, 512, 1, 1]" = var_mean_111[0]
    getitem_225: "f32[1, 512, 1, 1]" = var_mean_111[1];  var_mean_111 = None
    add_582: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-05)
    rsqrt_111: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_582);  add_582 = None
    sub_137: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_137, getitem_225)
    mul_803: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_111);  sub_137 = None
    squeeze_333: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_225, [0, 2, 3]);  getitem_225 = None
    squeeze_334: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_111, [0, 2, 3]);  rsqrt_111 = None
    mul_804: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_333, 0.1)
    mul_805: "f32[512]" = torch.ops.aten.mul.Tensor(primals_852, 0.9)
    add_583: "f32[512]" = torch.ops.aten.add.Tensor(mul_804, mul_805);  mul_804 = mul_805 = None
    squeeze_335: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_224, [0, 2, 3]);  getitem_224 = None
    mul_806: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_335, 1.0004885197850513);  squeeze_335 = None
    mul_807: "f32[512]" = torch.ops.aten.mul.Tensor(mul_806, 0.1);  mul_806 = None
    mul_808: "f32[512]" = torch.ops.aten.mul.Tensor(primals_853, 0.9)
    add_584: "f32[512]" = torch.ops.aten.add.Tensor(mul_807, mul_808);  mul_807 = mul_808 = None
    unsqueeze_444: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_413, -1)
    unsqueeze_445: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_809: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_803, unsqueeze_445);  mul_803 = unsqueeze_445 = None
    unsqueeze_446: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_414, -1);  primals_414 = None
    unsqueeze_447: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_585: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_809, unsqueeze_447);  mul_809 = unsqueeze_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_108: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_585);  add_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_157: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_108, [8, 2, 256, 16, 16])
    sum_79: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_157, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_26: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_79, [2, 3], True);  sum_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_138: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_26, primals_415, primals_416, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_586: "i64[]" = torch.ops.aten.add.Tensor(primals_857, 1)
    var_mean_112 = torch.ops.aten.var_mean.correction(convolution_138, [0, 2, 3], correction = 0, keepdim = True)
    getitem_226: "f32[1, 128, 1, 1]" = var_mean_112[0]
    getitem_227: "f32[1, 128, 1, 1]" = var_mean_112[1];  var_mean_112 = None
    add_587: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-05)
    rsqrt_112: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_587);  add_587 = None
    sub_138: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_138, getitem_227)
    mul_810: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_138, rsqrt_112);  sub_138 = rsqrt_112 = None
    squeeze_336: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_227, [0, 2, 3]);  getitem_227 = None
    mul_811: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_336, 0.1);  squeeze_336 = None
    mul_812: "f32[128]" = torch.ops.aten.mul.Tensor(primals_855, 0.9)
    add_588: "f32[128]" = torch.ops.aten.add.Tensor(mul_811, mul_812);  mul_811 = mul_812 = None
    squeeze_338: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_226, [0, 2, 3]);  getitem_226 = None
    mul_813: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_338, 1.1428571428571428);  squeeze_338 = None
    mul_814: "f32[128]" = torch.ops.aten.mul.Tensor(mul_813, 0.1);  mul_813 = None
    mul_815: "f32[128]" = torch.ops.aten.mul.Tensor(primals_856, 0.9)
    add_589: "f32[128]" = torch.ops.aten.add.Tensor(mul_814, mul_815);  mul_814 = mul_815 = None
    unsqueeze_448: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_417, -1)
    unsqueeze_449: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    mul_816: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_810, unsqueeze_449);  mul_810 = unsqueeze_449 = None
    unsqueeze_450: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_418, -1);  primals_418 = None
    unsqueeze_451: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    add_590: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_816, unsqueeze_451);  mul_816 = unsqueeze_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_109: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_590);  add_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_139: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_109, primals_419, primals_420, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_158: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_139, [8, 1, 2, -1]);  convolution_139 = None
    permute_26: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_26: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_26, [1], True)
    sub_139: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_26, amax_26);  permute_26 = amax_26 = None
    exp_26: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_139);  sub_139 = None
    sum_80: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_26, [1], True)
    div_26: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_26, sum_80);  exp_26 = sum_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_159: "f32[8, 512]" = torch.ops.aten.reshape.default(div_26, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_160: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_159, [8, -1, 1, 1]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_161: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_160, [8, 2, 256, 1, 1]);  view_160 = None
    mul_817: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_157, view_161);  view_157 = view_161 = None
    sum_81: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_817, [1]);  mul_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_140: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_81, primals_421, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_591: "i64[]" = torch.ops.aten.add.Tensor(primals_860, 1)
    var_mean_113 = torch.ops.aten.var_mean.correction(convolution_140, [0, 2, 3], correction = 0, keepdim = True)
    getitem_228: "f32[1, 1024, 1, 1]" = var_mean_113[0]
    getitem_229: "f32[1, 1024, 1, 1]" = var_mean_113[1];  var_mean_113 = None
    add_592: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_228, 1e-05)
    rsqrt_113: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_592);  add_592 = None
    sub_140: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_140, getitem_229)
    mul_818: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_113);  sub_140 = None
    squeeze_339: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_229, [0, 2, 3]);  getitem_229 = None
    squeeze_340: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_113, [0, 2, 3]);  rsqrt_113 = None
    mul_819: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_339, 0.1)
    mul_820: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_858, 0.9)
    add_593: "f32[1024]" = torch.ops.aten.add.Tensor(mul_819, mul_820);  mul_819 = mul_820 = None
    squeeze_341: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_228, [0, 2, 3]);  getitem_228 = None
    mul_821: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_341, 1.0004885197850513);  squeeze_341 = None
    mul_822: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_821, 0.1);  mul_821 = None
    mul_823: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_859, 0.9)
    add_594: "f32[1024]" = torch.ops.aten.add.Tensor(mul_822, mul_823);  mul_822 = mul_823 = None
    unsqueeze_452: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_422, -1)
    unsqueeze_453: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_824: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_818, unsqueeze_453);  mul_818 = unsqueeze_453 = None
    unsqueeze_454: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_423, -1);  primals_423 = None
    unsqueeze_455: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_595: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_824, unsqueeze_455);  mul_824 = unsqueeze_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_596: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_595, relu_106);  add_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_110: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_596);  add_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_141: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_110, primals_424, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_597: "i64[]" = torch.ops.aten.add.Tensor(primals_863, 1)
    var_mean_114 = torch.ops.aten.var_mean.correction(convolution_141, [0, 2, 3], correction = 0, keepdim = True)
    getitem_230: "f32[1, 256, 1, 1]" = var_mean_114[0]
    getitem_231: "f32[1, 256, 1, 1]" = var_mean_114[1];  var_mean_114 = None
    add_598: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_230, 1e-05)
    rsqrt_114: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_598);  add_598 = None
    sub_141: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_141, getitem_231)
    mul_825: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_141, rsqrt_114);  sub_141 = None
    squeeze_342: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_231, [0, 2, 3]);  getitem_231 = None
    squeeze_343: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_114, [0, 2, 3]);  rsqrt_114 = None
    mul_826: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_342, 0.1)
    mul_827: "f32[256]" = torch.ops.aten.mul.Tensor(primals_861, 0.9)
    add_599: "f32[256]" = torch.ops.aten.add.Tensor(mul_826, mul_827);  mul_826 = mul_827 = None
    squeeze_344: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_230, [0, 2, 3]);  getitem_230 = None
    mul_828: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_344, 1.0004885197850513);  squeeze_344 = None
    mul_829: "f32[256]" = torch.ops.aten.mul.Tensor(mul_828, 0.1);  mul_828 = None
    mul_830: "f32[256]" = torch.ops.aten.mul.Tensor(primals_862, 0.9)
    add_600: "f32[256]" = torch.ops.aten.add.Tensor(mul_829, mul_830);  mul_829 = mul_830 = None
    unsqueeze_456: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_425, -1)
    unsqueeze_457: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    mul_831: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_825, unsqueeze_457);  mul_825 = unsqueeze_457 = None
    unsqueeze_458: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_426, -1);  primals_426 = None
    unsqueeze_459: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    add_601: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_831, unsqueeze_459);  mul_831 = unsqueeze_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_111: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_601);  add_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_142: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_111, primals_427, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_602: "i64[]" = torch.ops.aten.add.Tensor(primals_866, 1)
    var_mean_115 = torch.ops.aten.var_mean.correction(convolution_142, [0, 2, 3], correction = 0, keepdim = True)
    getitem_232: "f32[1, 512, 1, 1]" = var_mean_115[0]
    getitem_233: "f32[1, 512, 1, 1]" = var_mean_115[1];  var_mean_115 = None
    add_603: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_232, 1e-05)
    rsqrt_115: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_603);  add_603 = None
    sub_142: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_142, getitem_233)
    mul_832: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_142, rsqrt_115);  sub_142 = None
    squeeze_345: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_233, [0, 2, 3]);  getitem_233 = None
    squeeze_346: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_115, [0, 2, 3]);  rsqrt_115 = None
    mul_833: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_345, 0.1)
    mul_834: "f32[512]" = torch.ops.aten.mul.Tensor(primals_864, 0.9)
    add_604: "f32[512]" = torch.ops.aten.add.Tensor(mul_833, mul_834);  mul_833 = mul_834 = None
    squeeze_347: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_232, [0, 2, 3]);  getitem_232 = None
    mul_835: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_347, 1.0004885197850513);  squeeze_347 = None
    mul_836: "f32[512]" = torch.ops.aten.mul.Tensor(mul_835, 0.1);  mul_835 = None
    mul_837: "f32[512]" = torch.ops.aten.mul.Tensor(primals_865, 0.9)
    add_605: "f32[512]" = torch.ops.aten.add.Tensor(mul_836, mul_837);  mul_836 = mul_837 = None
    unsqueeze_460: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_428, -1)
    unsqueeze_461: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_838: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_832, unsqueeze_461);  mul_832 = unsqueeze_461 = None
    unsqueeze_462: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_429, -1);  primals_429 = None
    unsqueeze_463: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_606: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_838, unsqueeze_463);  mul_838 = unsqueeze_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_112: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_606);  add_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_163: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_112, [8, 2, 256, 16, 16])
    sum_82: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_163, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_27: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_82, [2, 3], True);  sum_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_143: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_27, primals_430, primals_431, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_607: "i64[]" = torch.ops.aten.add.Tensor(primals_869, 1)
    var_mean_116 = torch.ops.aten.var_mean.correction(convolution_143, [0, 2, 3], correction = 0, keepdim = True)
    getitem_234: "f32[1, 128, 1, 1]" = var_mean_116[0]
    getitem_235: "f32[1, 128, 1, 1]" = var_mean_116[1];  var_mean_116 = None
    add_608: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_234, 1e-05)
    rsqrt_116: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_608);  add_608 = None
    sub_143: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_143, getitem_235)
    mul_839: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_143, rsqrt_116);  sub_143 = rsqrt_116 = None
    squeeze_348: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_235, [0, 2, 3]);  getitem_235 = None
    mul_840: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_348, 0.1);  squeeze_348 = None
    mul_841: "f32[128]" = torch.ops.aten.mul.Tensor(primals_867, 0.9)
    add_609: "f32[128]" = torch.ops.aten.add.Tensor(mul_840, mul_841);  mul_840 = mul_841 = None
    squeeze_350: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_234, [0, 2, 3]);  getitem_234 = None
    mul_842: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_350, 1.1428571428571428);  squeeze_350 = None
    mul_843: "f32[128]" = torch.ops.aten.mul.Tensor(mul_842, 0.1);  mul_842 = None
    mul_844: "f32[128]" = torch.ops.aten.mul.Tensor(primals_868, 0.9)
    add_610: "f32[128]" = torch.ops.aten.add.Tensor(mul_843, mul_844);  mul_843 = mul_844 = None
    unsqueeze_464: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_432, -1)
    unsqueeze_465: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
    mul_845: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_839, unsqueeze_465);  mul_839 = unsqueeze_465 = None
    unsqueeze_466: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_433, -1);  primals_433 = None
    unsqueeze_467: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    add_611: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_845, unsqueeze_467);  mul_845 = unsqueeze_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_113: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_611);  add_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_144: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_113, primals_434, primals_435, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_164: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_144, [8, 1, 2, -1]);  convolution_144 = None
    permute_27: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_27: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_27, [1], True)
    sub_144: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_27, amax_27);  permute_27 = amax_27 = None
    exp_27: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_144);  sub_144 = None
    sum_83: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_27, [1], True)
    div_27: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_27, sum_83);  exp_27 = sum_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_165: "f32[8, 512]" = torch.ops.aten.reshape.default(div_27, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_166: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_165, [8, -1, 1, 1]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_167: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_166, [8, 2, 256, 1, 1]);  view_166 = None
    mul_846: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_163, view_167);  view_163 = view_167 = None
    sum_84: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_846, [1]);  mul_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_145: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_84, primals_436, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_612: "i64[]" = torch.ops.aten.add.Tensor(primals_872, 1)
    var_mean_117 = torch.ops.aten.var_mean.correction(convolution_145, [0, 2, 3], correction = 0, keepdim = True)
    getitem_236: "f32[1, 1024, 1, 1]" = var_mean_117[0]
    getitem_237: "f32[1, 1024, 1, 1]" = var_mean_117[1];  var_mean_117 = None
    add_613: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_236, 1e-05)
    rsqrt_117: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_613);  add_613 = None
    sub_145: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_145, getitem_237)
    mul_847: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_145, rsqrt_117);  sub_145 = None
    squeeze_351: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_237, [0, 2, 3]);  getitem_237 = None
    squeeze_352: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_117, [0, 2, 3]);  rsqrt_117 = None
    mul_848: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_351, 0.1)
    mul_849: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_870, 0.9)
    add_614: "f32[1024]" = torch.ops.aten.add.Tensor(mul_848, mul_849);  mul_848 = mul_849 = None
    squeeze_353: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_236, [0, 2, 3]);  getitem_236 = None
    mul_850: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_353, 1.0004885197850513);  squeeze_353 = None
    mul_851: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_850, 0.1);  mul_850 = None
    mul_852: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_871, 0.9)
    add_615: "f32[1024]" = torch.ops.aten.add.Tensor(mul_851, mul_852);  mul_851 = mul_852 = None
    unsqueeze_468: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_437, -1)
    unsqueeze_469: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    mul_853: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_847, unsqueeze_469);  mul_847 = unsqueeze_469 = None
    unsqueeze_470: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_438, -1);  primals_438 = None
    unsqueeze_471: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
    add_616: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_853, unsqueeze_471);  mul_853 = unsqueeze_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_617: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_616, relu_110);  add_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_114: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_617);  add_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_146: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_114, primals_439, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_618: "i64[]" = torch.ops.aten.add.Tensor(primals_875, 1)
    var_mean_118 = torch.ops.aten.var_mean.correction(convolution_146, [0, 2, 3], correction = 0, keepdim = True)
    getitem_238: "f32[1, 256, 1, 1]" = var_mean_118[0]
    getitem_239: "f32[1, 256, 1, 1]" = var_mean_118[1];  var_mean_118 = None
    add_619: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_238, 1e-05)
    rsqrt_118: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_619);  add_619 = None
    sub_146: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_146, getitem_239)
    mul_854: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_146, rsqrt_118);  sub_146 = None
    squeeze_354: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_239, [0, 2, 3]);  getitem_239 = None
    squeeze_355: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_118, [0, 2, 3]);  rsqrt_118 = None
    mul_855: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_354, 0.1)
    mul_856: "f32[256]" = torch.ops.aten.mul.Tensor(primals_873, 0.9)
    add_620: "f32[256]" = torch.ops.aten.add.Tensor(mul_855, mul_856);  mul_855 = mul_856 = None
    squeeze_356: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_238, [0, 2, 3]);  getitem_238 = None
    mul_857: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_356, 1.0004885197850513);  squeeze_356 = None
    mul_858: "f32[256]" = torch.ops.aten.mul.Tensor(mul_857, 0.1);  mul_857 = None
    mul_859: "f32[256]" = torch.ops.aten.mul.Tensor(primals_874, 0.9)
    add_621: "f32[256]" = torch.ops.aten.add.Tensor(mul_858, mul_859);  mul_858 = mul_859 = None
    unsqueeze_472: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_440, -1)
    unsqueeze_473: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
    mul_860: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_854, unsqueeze_473);  mul_854 = unsqueeze_473 = None
    unsqueeze_474: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_441, -1);  primals_441 = None
    unsqueeze_475: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
    add_622: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_860, unsqueeze_475);  mul_860 = unsqueeze_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_115: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_622);  add_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_147: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_115, primals_442, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_623: "i64[]" = torch.ops.aten.add.Tensor(primals_878, 1)
    var_mean_119 = torch.ops.aten.var_mean.correction(convolution_147, [0, 2, 3], correction = 0, keepdim = True)
    getitem_240: "f32[1, 512, 1, 1]" = var_mean_119[0]
    getitem_241: "f32[1, 512, 1, 1]" = var_mean_119[1];  var_mean_119 = None
    add_624: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-05)
    rsqrt_119: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_624);  add_624 = None
    sub_147: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_147, getitem_241)
    mul_861: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_147, rsqrt_119);  sub_147 = None
    squeeze_357: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_241, [0, 2, 3]);  getitem_241 = None
    squeeze_358: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_119, [0, 2, 3]);  rsqrt_119 = None
    mul_862: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_357, 0.1)
    mul_863: "f32[512]" = torch.ops.aten.mul.Tensor(primals_876, 0.9)
    add_625: "f32[512]" = torch.ops.aten.add.Tensor(mul_862, mul_863);  mul_862 = mul_863 = None
    squeeze_359: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_240, [0, 2, 3]);  getitem_240 = None
    mul_864: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_359, 1.0004885197850513);  squeeze_359 = None
    mul_865: "f32[512]" = torch.ops.aten.mul.Tensor(mul_864, 0.1);  mul_864 = None
    mul_866: "f32[512]" = torch.ops.aten.mul.Tensor(primals_877, 0.9)
    add_626: "f32[512]" = torch.ops.aten.add.Tensor(mul_865, mul_866);  mul_865 = mul_866 = None
    unsqueeze_476: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_443, -1)
    unsqueeze_477: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    mul_867: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_861, unsqueeze_477);  mul_861 = unsqueeze_477 = None
    unsqueeze_478: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_444, -1);  primals_444 = None
    unsqueeze_479: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    add_627: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_867, unsqueeze_479);  mul_867 = unsqueeze_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_116: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_627);  add_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_169: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_116, [8, 2, 256, 16, 16])
    sum_85: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_169, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_28: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_85, [2, 3], True);  sum_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_148: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_28, primals_445, primals_446, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_628: "i64[]" = torch.ops.aten.add.Tensor(primals_881, 1)
    var_mean_120 = torch.ops.aten.var_mean.correction(convolution_148, [0, 2, 3], correction = 0, keepdim = True)
    getitem_242: "f32[1, 128, 1, 1]" = var_mean_120[0]
    getitem_243: "f32[1, 128, 1, 1]" = var_mean_120[1];  var_mean_120 = None
    add_629: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-05)
    rsqrt_120: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_629);  add_629 = None
    sub_148: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_148, getitem_243)
    mul_868: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_148, rsqrt_120);  sub_148 = rsqrt_120 = None
    squeeze_360: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_243, [0, 2, 3]);  getitem_243 = None
    mul_869: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_360, 0.1);  squeeze_360 = None
    mul_870: "f32[128]" = torch.ops.aten.mul.Tensor(primals_879, 0.9)
    add_630: "f32[128]" = torch.ops.aten.add.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
    squeeze_362: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_242, [0, 2, 3]);  getitem_242 = None
    mul_871: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_362, 1.1428571428571428);  squeeze_362 = None
    mul_872: "f32[128]" = torch.ops.aten.mul.Tensor(mul_871, 0.1);  mul_871 = None
    mul_873: "f32[128]" = torch.ops.aten.mul.Tensor(primals_880, 0.9)
    add_631: "f32[128]" = torch.ops.aten.add.Tensor(mul_872, mul_873);  mul_872 = mul_873 = None
    unsqueeze_480: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_447, -1)
    unsqueeze_481: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
    mul_874: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_868, unsqueeze_481);  mul_868 = unsqueeze_481 = None
    unsqueeze_482: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_448, -1);  primals_448 = None
    unsqueeze_483: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
    add_632: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_874, unsqueeze_483);  mul_874 = unsqueeze_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_117: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_632);  add_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_149: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_117, primals_449, primals_450, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_170: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_149, [8, 1, 2, -1]);  convolution_149 = None
    permute_28: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_28: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_28, [1], True)
    sub_149: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_28, amax_28);  permute_28 = amax_28 = None
    exp_28: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_149);  sub_149 = None
    sum_86: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_28, [1], True)
    div_28: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_28, sum_86);  exp_28 = sum_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_171: "f32[8, 512]" = torch.ops.aten.reshape.default(div_28, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_172: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_171, [8, -1, 1, 1]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_173: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_172, [8, 2, 256, 1, 1]);  view_172 = None
    mul_875: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_169, view_173);  view_169 = view_173 = None
    sum_87: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_875, [1]);  mul_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_150: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_87, primals_451, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_633: "i64[]" = torch.ops.aten.add.Tensor(primals_884, 1)
    var_mean_121 = torch.ops.aten.var_mean.correction(convolution_150, [0, 2, 3], correction = 0, keepdim = True)
    getitem_244: "f32[1, 1024, 1, 1]" = var_mean_121[0]
    getitem_245: "f32[1, 1024, 1, 1]" = var_mean_121[1];  var_mean_121 = None
    add_634: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_244, 1e-05)
    rsqrt_121: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_634);  add_634 = None
    sub_150: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_150, getitem_245)
    mul_876: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_150, rsqrt_121);  sub_150 = None
    squeeze_363: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_245, [0, 2, 3]);  getitem_245 = None
    squeeze_364: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_121, [0, 2, 3]);  rsqrt_121 = None
    mul_877: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_363, 0.1)
    mul_878: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_882, 0.9)
    add_635: "f32[1024]" = torch.ops.aten.add.Tensor(mul_877, mul_878);  mul_877 = mul_878 = None
    squeeze_365: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_244, [0, 2, 3]);  getitem_244 = None
    mul_879: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_365, 1.0004885197850513);  squeeze_365 = None
    mul_880: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_879, 0.1);  mul_879 = None
    mul_881: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_883, 0.9)
    add_636: "f32[1024]" = torch.ops.aten.add.Tensor(mul_880, mul_881);  mul_880 = mul_881 = None
    unsqueeze_484: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_452, -1)
    unsqueeze_485: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
    mul_882: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_876, unsqueeze_485);  mul_876 = unsqueeze_485 = None
    unsqueeze_486: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_453, -1);  primals_453 = None
    unsqueeze_487: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    add_637: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_882, unsqueeze_487);  mul_882 = unsqueeze_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_638: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_637, relu_114);  add_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_118: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_638);  add_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_151: "f32[8, 256, 16, 16]" = torch.ops.aten.convolution.default(relu_118, primals_454, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_639: "i64[]" = torch.ops.aten.add.Tensor(primals_887, 1)
    var_mean_122 = torch.ops.aten.var_mean.correction(convolution_151, [0, 2, 3], correction = 0, keepdim = True)
    getitem_246: "f32[1, 256, 1, 1]" = var_mean_122[0]
    getitem_247: "f32[1, 256, 1, 1]" = var_mean_122[1];  var_mean_122 = None
    add_640: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-05)
    rsqrt_122: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_640);  add_640 = None
    sub_151: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_151, getitem_247)
    mul_883: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_151, rsqrt_122);  sub_151 = None
    squeeze_366: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_247, [0, 2, 3]);  getitem_247 = None
    squeeze_367: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_122, [0, 2, 3]);  rsqrt_122 = None
    mul_884: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_366, 0.1)
    mul_885: "f32[256]" = torch.ops.aten.mul.Tensor(primals_885, 0.9)
    add_641: "f32[256]" = torch.ops.aten.add.Tensor(mul_884, mul_885);  mul_884 = mul_885 = None
    squeeze_368: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_246, [0, 2, 3]);  getitem_246 = None
    mul_886: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_368, 1.0004885197850513);  squeeze_368 = None
    mul_887: "f32[256]" = torch.ops.aten.mul.Tensor(mul_886, 0.1);  mul_886 = None
    mul_888: "f32[256]" = torch.ops.aten.mul.Tensor(primals_886, 0.9)
    add_642: "f32[256]" = torch.ops.aten.add.Tensor(mul_887, mul_888);  mul_887 = mul_888 = None
    unsqueeze_488: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_455, -1)
    unsqueeze_489: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    mul_889: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(mul_883, unsqueeze_489);  mul_883 = unsqueeze_489 = None
    unsqueeze_490: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_456, -1);  primals_456 = None
    unsqueeze_491: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
    add_643: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(mul_889, unsqueeze_491);  mul_889 = unsqueeze_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_119: "f32[8, 256, 16, 16]" = torch.ops.aten.relu.default(add_643);  add_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_152: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_119, primals_457, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_644: "i64[]" = torch.ops.aten.add.Tensor(primals_890, 1)
    var_mean_123 = torch.ops.aten.var_mean.correction(convolution_152, [0, 2, 3], correction = 0, keepdim = True)
    getitem_248: "f32[1, 512, 1, 1]" = var_mean_123[0]
    getitem_249: "f32[1, 512, 1, 1]" = var_mean_123[1];  var_mean_123 = None
    add_645: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_248, 1e-05)
    rsqrt_123: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_645);  add_645 = None
    sub_152: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_152, getitem_249)
    mul_890: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_152, rsqrt_123);  sub_152 = None
    squeeze_369: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_249, [0, 2, 3]);  getitem_249 = None
    squeeze_370: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_123, [0, 2, 3]);  rsqrt_123 = None
    mul_891: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_369, 0.1)
    mul_892: "f32[512]" = torch.ops.aten.mul.Tensor(primals_888, 0.9)
    add_646: "f32[512]" = torch.ops.aten.add.Tensor(mul_891, mul_892);  mul_891 = mul_892 = None
    squeeze_371: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_248, [0, 2, 3]);  getitem_248 = None
    mul_893: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_371, 1.0004885197850513);  squeeze_371 = None
    mul_894: "f32[512]" = torch.ops.aten.mul.Tensor(mul_893, 0.1);  mul_893 = None
    mul_895: "f32[512]" = torch.ops.aten.mul.Tensor(primals_889, 0.9)
    add_647: "f32[512]" = torch.ops.aten.add.Tensor(mul_894, mul_895);  mul_894 = mul_895 = None
    unsqueeze_492: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_458, -1)
    unsqueeze_493: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
    mul_896: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_890, unsqueeze_493);  mul_890 = unsqueeze_493 = None
    unsqueeze_494: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_459, -1);  primals_459 = None
    unsqueeze_495: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
    add_648: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_896, unsqueeze_495);  mul_896 = unsqueeze_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_120: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_648);  add_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_175: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.reshape.default(relu_120, [8, 2, 256, 16, 16])
    sum_88: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_175, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_29: "f32[8, 256, 1, 1]" = torch.ops.aten.mean.dim(sum_88, [2, 3], True);  sum_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_153: "f32[8, 128, 1, 1]" = torch.ops.aten.convolution.default(mean_29, primals_460, primals_461, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_649: "i64[]" = torch.ops.aten.add.Tensor(primals_893, 1)
    var_mean_124 = torch.ops.aten.var_mean.correction(convolution_153, [0, 2, 3], correction = 0, keepdim = True)
    getitem_250: "f32[1, 128, 1, 1]" = var_mean_124[0]
    getitem_251: "f32[1, 128, 1, 1]" = var_mean_124[1];  var_mean_124 = None
    add_650: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_250, 1e-05)
    rsqrt_124: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_650);  add_650 = None
    sub_153: "f32[8, 128, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_153, getitem_251)
    mul_897: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(sub_153, rsqrt_124);  sub_153 = rsqrt_124 = None
    squeeze_372: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_251, [0, 2, 3]);  getitem_251 = None
    mul_898: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_372, 0.1);  squeeze_372 = None
    mul_899: "f32[128]" = torch.ops.aten.mul.Tensor(primals_891, 0.9)
    add_651: "f32[128]" = torch.ops.aten.add.Tensor(mul_898, mul_899);  mul_898 = mul_899 = None
    squeeze_374: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_250, [0, 2, 3]);  getitem_250 = None
    mul_900: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_374, 1.1428571428571428);  squeeze_374 = None
    mul_901: "f32[128]" = torch.ops.aten.mul.Tensor(mul_900, 0.1);  mul_900 = None
    mul_902: "f32[128]" = torch.ops.aten.mul.Tensor(primals_892, 0.9)
    add_652: "f32[128]" = torch.ops.aten.add.Tensor(mul_901, mul_902);  mul_901 = mul_902 = None
    unsqueeze_496: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_462, -1)
    unsqueeze_497: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
    mul_903: "f32[8, 128, 1, 1]" = torch.ops.aten.mul.Tensor(mul_897, unsqueeze_497);  mul_897 = unsqueeze_497 = None
    unsqueeze_498: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_463, -1);  primals_463 = None
    unsqueeze_499: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
    add_653: "f32[8, 128, 1, 1]" = torch.ops.aten.add.Tensor(mul_903, unsqueeze_499);  mul_903 = unsqueeze_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_121: "f32[8, 128, 1, 1]" = torch.ops.aten.relu.default(add_653);  add_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_154: "f32[8, 512, 1, 1]" = torch.ops.aten.convolution.default(relu_121, primals_464, primals_465, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_176: "f32[8, 1, 2, 256]" = torch.ops.aten.reshape.default(convolution_154, [8, 1, 2, -1]);  convolution_154 = None
    permute_29: "f32[8, 2, 1, 256]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_29: "f32[8, 1, 1, 256]" = torch.ops.aten.amax.default(permute_29, [1], True)
    sub_154: "f32[8, 2, 1, 256]" = torch.ops.aten.sub.Tensor(permute_29, amax_29);  permute_29 = amax_29 = None
    exp_29: "f32[8, 2, 1, 256]" = torch.ops.aten.exp.default(sub_154);  sub_154 = None
    sum_89: "f32[8, 1, 1, 256]" = torch.ops.aten.sum.dim_IntList(exp_29, [1], True)
    div_29: "f32[8, 2, 1, 256]" = torch.ops.aten.div.Tensor(exp_29, sum_89);  exp_29 = sum_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_177: "f32[8, 512]" = torch.ops.aten.reshape.default(div_29, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_178: "f32[8, 512, 1, 1]" = torch.ops.aten.reshape.default(view_177, [8, -1, 1, 1]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_179: "f32[8, 2, 256, 1, 1]" = torch.ops.aten.reshape.default(view_178, [8, 2, 256, 1, 1]);  view_178 = None
    mul_904: "f32[8, 2, 256, 16, 16]" = torch.ops.aten.mul.Tensor(view_175, view_179);  view_175 = view_179 = None
    sum_90: "f32[8, 256, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_904, [1]);  mul_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_155: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(sum_90, primals_466, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_654: "i64[]" = torch.ops.aten.add.Tensor(primals_896, 1)
    var_mean_125 = torch.ops.aten.var_mean.correction(convolution_155, [0, 2, 3], correction = 0, keepdim = True)
    getitem_252: "f32[1, 1024, 1, 1]" = var_mean_125[0]
    getitem_253: "f32[1, 1024, 1, 1]" = var_mean_125[1];  var_mean_125 = None
    add_655: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_252, 1e-05)
    rsqrt_125: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_655);  add_655 = None
    sub_155: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_155, getitem_253)
    mul_905: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_155, rsqrt_125);  sub_155 = None
    squeeze_375: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_253, [0, 2, 3]);  getitem_253 = None
    squeeze_376: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_125, [0, 2, 3]);  rsqrt_125 = None
    mul_906: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_375, 0.1)
    mul_907: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_894, 0.9)
    add_656: "f32[1024]" = torch.ops.aten.add.Tensor(mul_906, mul_907);  mul_906 = mul_907 = None
    squeeze_377: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_252, [0, 2, 3]);  getitem_252 = None
    mul_908: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_377, 1.0004885197850513);  squeeze_377 = None
    mul_909: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_908, 0.1);  mul_908 = None
    mul_910: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_895, 0.9)
    add_657: "f32[1024]" = torch.ops.aten.add.Tensor(mul_909, mul_910);  mul_909 = mul_910 = None
    unsqueeze_500: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_467, -1)
    unsqueeze_501: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
    mul_911: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_905, unsqueeze_501);  mul_905 = unsqueeze_501 = None
    unsqueeze_502: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_468, -1);  primals_468 = None
    unsqueeze_503: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
    add_658: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_911, unsqueeze_503);  mul_911 = unsqueeze_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_659: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(add_658, relu_118);  add_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_122: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_659);  add_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_156: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(relu_122, primals_469, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_660: "i64[]" = torch.ops.aten.add.Tensor(primals_899, 1)
    var_mean_126 = torch.ops.aten.var_mean.correction(convolution_156, [0, 2, 3], correction = 0, keepdim = True)
    getitem_254: "f32[1, 512, 1, 1]" = var_mean_126[0]
    getitem_255: "f32[1, 512, 1, 1]" = var_mean_126[1];  var_mean_126 = None
    add_661: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_254, 1e-05)
    rsqrt_126: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_661);  add_661 = None
    sub_156: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_156, getitem_255)
    mul_912: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_156, rsqrt_126);  sub_156 = None
    squeeze_378: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_255, [0, 2, 3]);  getitem_255 = None
    squeeze_379: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_126, [0, 2, 3]);  rsqrt_126 = None
    mul_913: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_378, 0.1)
    mul_914: "f32[512]" = torch.ops.aten.mul.Tensor(primals_897, 0.9)
    add_662: "f32[512]" = torch.ops.aten.add.Tensor(mul_913, mul_914);  mul_913 = mul_914 = None
    squeeze_380: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_254, [0, 2, 3]);  getitem_254 = None
    mul_915: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_380, 1.0004885197850513);  squeeze_380 = None
    mul_916: "f32[512]" = torch.ops.aten.mul.Tensor(mul_915, 0.1);  mul_915 = None
    mul_917: "f32[512]" = torch.ops.aten.mul.Tensor(primals_898, 0.9)
    add_663: "f32[512]" = torch.ops.aten.add.Tensor(mul_916, mul_917);  mul_916 = mul_917 = None
    unsqueeze_504: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_470, -1)
    unsqueeze_505: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
    mul_918: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_912, unsqueeze_505);  mul_912 = unsqueeze_505 = None
    unsqueeze_506: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_471, -1);  primals_471 = None
    unsqueeze_507: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
    add_664: "f32[8, 512, 16, 16]" = torch.ops.aten.add.Tensor(mul_918, unsqueeze_507);  mul_918 = unsqueeze_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_123: "f32[8, 512, 16, 16]" = torch.ops.aten.relu.default(add_664);  add_664 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_157: "f32[8, 1024, 16, 16]" = torch.ops.aten.convolution.default(relu_123, primals_472, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_665: "i64[]" = torch.ops.aten.add.Tensor(primals_902, 1)
    var_mean_127 = torch.ops.aten.var_mean.correction(convolution_157, [0, 2, 3], correction = 0, keepdim = True)
    getitem_256: "f32[1, 1024, 1, 1]" = var_mean_127[0]
    getitem_257: "f32[1, 1024, 1, 1]" = var_mean_127[1];  var_mean_127 = None
    add_666: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_256, 1e-05)
    rsqrt_127: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_666);  add_666 = None
    sub_157: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_157, getitem_257)
    mul_919: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_157, rsqrt_127);  sub_157 = None
    squeeze_381: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_257, [0, 2, 3]);  getitem_257 = None
    squeeze_382: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_127, [0, 2, 3]);  rsqrt_127 = None
    mul_920: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_381, 0.1)
    mul_921: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_900, 0.9)
    add_667: "f32[1024]" = torch.ops.aten.add.Tensor(mul_920, mul_921);  mul_920 = mul_921 = None
    squeeze_383: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_256, [0, 2, 3]);  getitem_256 = None
    mul_922: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_383, 1.0004885197850513);  squeeze_383 = None
    mul_923: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_922, 0.1);  mul_922 = None
    mul_924: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_901, 0.9)
    add_668: "f32[1024]" = torch.ops.aten.add.Tensor(mul_923, mul_924);  mul_923 = mul_924 = None
    unsqueeze_508: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_473, -1)
    unsqueeze_509: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
    mul_925: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(mul_919, unsqueeze_509);  mul_919 = unsqueeze_509 = None
    unsqueeze_510: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_474, -1);  primals_474 = None
    unsqueeze_511: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
    add_669: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(mul_925, unsqueeze_511);  mul_925 = unsqueeze_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_124: "f32[8, 1024, 16, 16]" = torch.ops.aten.relu.default(add_669);  add_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_181: "f32[8, 2, 512, 16, 16]" = torch.ops.aten.reshape.default(relu_124, [8, 2, 512, 16, 16])
    sum_91: "f32[8, 512, 16, 16]" = torch.ops.aten.sum.dim_IntList(view_181, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_30: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(sum_91, [2, 3], True);  sum_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_158: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_30, primals_475, primals_476, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_670: "i64[]" = torch.ops.aten.add.Tensor(primals_905, 1)
    var_mean_128 = torch.ops.aten.var_mean.correction(convolution_158, [0, 2, 3], correction = 0, keepdim = True)
    getitem_258: "f32[1, 256, 1, 1]" = var_mean_128[0]
    getitem_259: "f32[1, 256, 1, 1]" = var_mean_128[1];  var_mean_128 = None
    add_671: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_258, 1e-05)
    rsqrt_128: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_671);  add_671 = None
    sub_158: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_158, getitem_259)
    mul_926: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_158, rsqrt_128);  sub_158 = rsqrt_128 = None
    squeeze_384: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_259, [0, 2, 3]);  getitem_259 = None
    mul_927: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_384, 0.1);  squeeze_384 = None
    mul_928: "f32[256]" = torch.ops.aten.mul.Tensor(primals_903, 0.9)
    add_672: "f32[256]" = torch.ops.aten.add.Tensor(mul_927, mul_928);  mul_927 = mul_928 = None
    squeeze_386: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_258, [0, 2, 3]);  getitem_258 = None
    mul_929: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_386, 1.1428571428571428);  squeeze_386 = None
    mul_930: "f32[256]" = torch.ops.aten.mul.Tensor(mul_929, 0.1);  mul_929 = None
    mul_931: "f32[256]" = torch.ops.aten.mul.Tensor(primals_904, 0.9)
    add_673: "f32[256]" = torch.ops.aten.add.Tensor(mul_930, mul_931);  mul_930 = mul_931 = None
    unsqueeze_512: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_477, -1)
    unsqueeze_513: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
    mul_932: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(mul_926, unsqueeze_513);  mul_926 = unsqueeze_513 = None
    unsqueeze_514: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_478, -1);  primals_478 = None
    unsqueeze_515: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
    add_674: "f32[8, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_932, unsqueeze_515);  mul_932 = unsqueeze_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_125: "f32[8, 256, 1, 1]" = torch.ops.aten.relu.default(add_674);  add_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_159: "f32[8, 1024, 1, 1]" = torch.ops.aten.convolution.default(relu_125, primals_479, primals_480, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_182: "f32[8, 1, 2, 512]" = torch.ops.aten.reshape.default(convolution_159, [8, 1, 2, -1]);  convolution_159 = None
    permute_30: "f32[8, 2, 1, 512]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_30: "f32[8, 1, 1, 512]" = torch.ops.aten.amax.default(permute_30, [1], True)
    sub_159: "f32[8, 2, 1, 512]" = torch.ops.aten.sub.Tensor(permute_30, amax_30);  permute_30 = amax_30 = None
    exp_30: "f32[8, 2, 1, 512]" = torch.ops.aten.exp.default(sub_159);  sub_159 = None
    sum_92: "f32[8, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(exp_30, [1], True)
    div_30: "f32[8, 2, 1, 512]" = torch.ops.aten.div.Tensor(exp_30, sum_92);  exp_30 = sum_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_183: "f32[8, 1024]" = torch.ops.aten.reshape.default(div_30, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_184: "f32[8, 1024, 1, 1]" = torch.ops.aten.reshape.default(view_183, [8, -1, 1, 1]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_185: "f32[8, 2, 512, 1, 1]" = torch.ops.aten.reshape.default(view_184, [8, 2, 512, 1, 1]);  view_184 = None
    mul_933: "f32[8, 2, 512, 16, 16]" = torch.ops.aten.mul.Tensor(view_181, view_185);  view_181 = view_185 = None
    sum_93: "f32[8, 512, 16, 16]" = torch.ops.aten.sum.dim_IntList(mul_933, [1]);  mul_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:107, code: out = self.avd_last(out)
    avg_pool2d_4: "f32[8, 512, 8, 8]" = torch.ops.aten.avg_pool2d.default(sum_93, [3, 3], [2, 2], [1, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_160: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(avg_pool2d_4, primals_481, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_675: "i64[]" = torch.ops.aten.add.Tensor(primals_908, 1)
    var_mean_129 = torch.ops.aten.var_mean.correction(convolution_160, [0, 2, 3], correction = 0, keepdim = True)
    getitem_260: "f32[1, 2048, 1, 1]" = var_mean_129[0]
    getitem_261: "f32[1, 2048, 1, 1]" = var_mean_129[1];  var_mean_129 = None
    add_676: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_260, 1e-05)
    rsqrt_129: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_676);  add_676 = None
    sub_160: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_160, getitem_261)
    mul_934: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_160, rsqrt_129);  sub_160 = None
    squeeze_387: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_261, [0, 2, 3]);  getitem_261 = None
    squeeze_388: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_129, [0, 2, 3]);  rsqrt_129 = None
    mul_935: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_387, 0.1)
    mul_936: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_906, 0.9)
    add_677: "f32[2048]" = torch.ops.aten.add.Tensor(mul_935, mul_936);  mul_935 = mul_936 = None
    squeeze_389: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_260, [0, 2, 3]);  getitem_260 = None
    mul_937: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_389, 1.0019569471624266);  squeeze_389 = None
    mul_938: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_937, 0.1);  mul_937 = None
    mul_939: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_907, 0.9)
    add_678: "f32[2048]" = torch.ops.aten.add.Tensor(mul_938, mul_939);  mul_938 = mul_939 = None
    unsqueeze_516: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_482, -1)
    unsqueeze_517: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
    mul_940: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_934, unsqueeze_517);  mul_934 = unsqueeze_517 = None
    unsqueeze_518: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_483, -1);  primals_483 = None
    unsqueeze_519: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
    add_679: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_940, unsqueeze_519);  mul_940 = unsqueeze_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    avg_pool2d_5: "f32[8, 1024, 8, 8]" = torch.ops.aten.avg_pool2d.default(relu_122, [2, 2], [2, 2], [0, 0], True, False)
    convolution_161: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(avg_pool2d_5, primals_484, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_680: "i64[]" = torch.ops.aten.add.Tensor(primals_911, 1)
    var_mean_130 = torch.ops.aten.var_mean.correction(convolution_161, [0, 2, 3], correction = 0, keepdim = True)
    getitem_262: "f32[1, 2048, 1, 1]" = var_mean_130[0]
    getitem_263: "f32[1, 2048, 1, 1]" = var_mean_130[1];  var_mean_130 = None
    add_681: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_262, 1e-05)
    rsqrt_130: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_681);  add_681 = None
    sub_161: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_161, getitem_263)
    mul_941: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_161, rsqrt_130);  sub_161 = None
    squeeze_390: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_263, [0, 2, 3]);  getitem_263 = None
    squeeze_391: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_130, [0, 2, 3]);  rsqrt_130 = None
    mul_942: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_390, 0.1)
    mul_943: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_909, 0.9)
    add_682: "f32[2048]" = torch.ops.aten.add.Tensor(mul_942, mul_943);  mul_942 = mul_943 = None
    squeeze_392: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_262, [0, 2, 3]);  getitem_262 = None
    mul_944: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_392, 1.0019569471624266);  squeeze_392 = None
    mul_945: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_944, 0.1);  mul_944 = None
    mul_946: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_910, 0.9)
    add_683: "f32[2048]" = torch.ops.aten.add.Tensor(mul_945, mul_946);  mul_945 = mul_946 = None
    unsqueeze_520: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_485, -1)
    unsqueeze_521: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
    mul_947: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_941, unsqueeze_521);  mul_941 = unsqueeze_521 = None
    unsqueeze_522: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_486, -1);  primals_486 = None
    unsqueeze_523: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
    add_684: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_947, unsqueeze_523);  mul_947 = unsqueeze_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_685: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_679, add_684);  add_679 = add_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_126: "f32[8, 2048, 8, 8]" = torch.ops.aten.relu.default(add_685);  add_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_162: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(relu_126, primals_487, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_686: "i64[]" = torch.ops.aten.add.Tensor(primals_914, 1)
    var_mean_131 = torch.ops.aten.var_mean.correction(convolution_162, [0, 2, 3], correction = 0, keepdim = True)
    getitem_264: "f32[1, 512, 1, 1]" = var_mean_131[0]
    getitem_265: "f32[1, 512, 1, 1]" = var_mean_131[1];  var_mean_131 = None
    add_687: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_264, 1e-05)
    rsqrt_131: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_687);  add_687 = None
    sub_162: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_162, getitem_265)
    mul_948: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_162, rsqrt_131);  sub_162 = None
    squeeze_393: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_265, [0, 2, 3]);  getitem_265 = None
    squeeze_394: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_131, [0, 2, 3]);  rsqrt_131 = None
    mul_949: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_393, 0.1)
    mul_950: "f32[512]" = torch.ops.aten.mul.Tensor(primals_912, 0.9)
    add_688: "f32[512]" = torch.ops.aten.add.Tensor(mul_949, mul_950);  mul_949 = mul_950 = None
    squeeze_395: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_264, [0, 2, 3]);  getitem_264 = None
    mul_951: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_395, 1.0019569471624266);  squeeze_395 = None
    mul_952: "f32[512]" = torch.ops.aten.mul.Tensor(mul_951, 0.1);  mul_951 = None
    mul_953: "f32[512]" = torch.ops.aten.mul.Tensor(primals_913, 0.9)
    add_689: "f32[512]" = torch.ops.aten.add.Tensor(mul_952, mul_953);  mul_952 = mul_953 = None
    unsqueeze_524: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_488, -1)
    unsqueeze_525: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
    mul_954: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_948, unsqueeze_525);  mul_948 = unsqueeze_525 = None
    unsqueeze_526: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_489, -1);  primals_489 = None
    unsqueeze_527: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
    add_690: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_954, unsqueeze_527);  mul_954 = unsqueeze_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_127: "f32[8, 512, 8, 8]" = torch.ops.aten.relu.default(add_690);  add_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_163: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(relu_127, primals_490, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_691: "i64[]" = torch.ops.aten.add.Tensor(primals_917, 1)
    var_mean_132 = torch.ops.aten.var_mean.correction(convolution_163, [0, 2, 3], correction = 0, keepdim = True)
    getitem_266: "f32[1, 1024, 1, 1]" = var_mean_132[0]
    getitem_267: "f32[1, 1024, 1, 1]" = var_mean_132[1];  var_mean_132 = None
    add_692: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-05)
    rsqrt_132: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_692);  add_692 = None
    sub_163: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_163, getitem_267)
    mul_955: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_163, rsqrt_132);  sub_163 = None
    squeeze_396: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_267, [0, 2, 3]);  getitem_267 = None
    squeeze_397: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_132, [0, 2, 3]);  rsqrt_132 = None
    mul_956: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_396, 0.1)
    mul_957: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_915, 0.9)
    add_693: "f32[1024]" = torch.ops.aten.add.Tensor(mul_956, mul_957);  mul_956 = mul_957 = None
    squeeze_398: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_266, [0, 2, 3]);  getitem_266 = None
    mul_958: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_398, 1.0019569471624266);  squeeze_398 = None
    mul_959: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_958, 0.1);  mul_958 = None
    mul_960: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_916, 0.9)
    add_694: "f32[1024]" = torch.ops.aten.add.Tensor(mul_959, mul_960);  mul_959 = mul_960 = None
    unsqueeze_528: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_491, -1)
    unsqueeze_529: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
    mul_961: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_955, unsqueeze_529);  mul_955 = unsqueeze_529 = None
    unsqueeze_530: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_492, -1);  primals_492 = None
    unsqueeze_531: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
    add_695: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_961, unsqueeze_531);  mul_961 = unsqueeze_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_128: "f32[8, 1024, 8, 8]" = torch.ops.aten.relu.default(add_695);  add_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_187: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.reshape.default(relu_128, [8, 2, 512, 8, 8])
    sum_94: "f32[8, 512, 8, 8]" = torch.ops.aten.sum.dim_IntList(view_187, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_31: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(sum_94, [2, 3], True);  sum_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_164: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_31, primals_493, primals_494, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_696: "i64[]" = torch.ops.aten.add.Tensor(primals_920, 1)
    var_mean_133 = torch.ops.aten.var_mean.correction(convolution_164, [0, 2, 3], correction = 0, keepdim = True)
    getitem_268: "f32[1, 256, 1, 1]" = var_mean_133[0]
    getitem_269: "f32[1, 256, 1, 1]" = var_mean_133[1];  var_mean_133 = None
    add_697: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_268, 1e-05)
    rsqrt_133: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_697);  add_697 = None
    sub_164: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_164, getitem_269)
    mul_962: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_164, rsqrt_133);  sub_164 = rsqrt_133 = None
    squeeze_399: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_269, [0, 2, 3]);  getitem_269 = None
    mul_963: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_399, 0.1);  squeeze_399 = None
    mul_964: "f32[256]" = torch.ops.aten.mul.Tensor(primals_918, 0.9)
    add_698: "f32[256]" = torch.ops.aten.add.Tensor(mul_963, mul_964);  mul_963 = mul_964 = None
    squeeze_401: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_268, [0, 2, 3]);  getitem_268 = None
    mul_965: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_401, 1.1428571428571428);  squeeze_401 = None
    mul_966: "f32[256]" = torch.ops.aten.mul.Tensor(mul_965, 0.1);  mul_965 = None
    mul_967: "f32[256]" = torch.ops.aten.mul.Tensor(primals_919, 0.9)
    add_699: "f32[256]" = torch.ops.aten.add.Tensor(mul_966, mul_967);  mul_966 = mul_967 = None
    unsqueeze_532: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_495, -1)
    unsqueeze_533: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
    mul_968: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(mul_962, unsqueeze_533);  mul_962 = unsqueeze_533 = None
    unsqueeze_534: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_496, -1);  primals_496 = None
    unsqueeze_535: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
    add_700: "f32[8, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_968, unsqueeze_535);  mul_968 = unsqueeze_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_129: "f32[8, 256, 1, 1]" = torch.ops.aten.relu.default(add_700);  add_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_165: "f32[8, 1024, 1, 1]" = torch.ops.aten.convolution.default(relu_129, primals_497, primals_498, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_188: "f32[8, 1, 2, 512]" = torch.ops.aten.reshape.default(convolution_165, [8, 1, 2, -1]);  convolution_165 = None
    permute_31: "f32[8, 2, 1, 512]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_31: "f32[8, 1, 1, 512]" = torch.ops.aten.amax.default(permute_31, [1], True)
    sub_165: "f32[8, 2, 1, 512]" = torch.ops.aten.sub.Tensor(permute_31, amax_31);  permute_31 = amax_31 = None
    exp_31: "f32[8, 2, 1, 512]" = torch.ops.aten.exp.default(sub_165);  sub_165 = None
    sum_95: "f32[8, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(exp_31, [1], True)
    div_31: "f32[8, 2, 1, 512]" = torch.ops.aten.div.Tensor(exp_31, sum_95);  exp_31 = sum_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_189: "f32[8, 1024]" = torch.ops.aten.reshape.default(div_31, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_190: "f32[8, 1024, 1, 1]" = torch.ops.aten.reshape.default(view_189, [8, -1, 1, 1]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_191: "f32[8, 2, 512, 1, 1]" = torch.ops.aten.reshape.default(view_190, [8, 2, 512, 1, 1]);  view_190 = None
    mul_969: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.mul.Tensor(view_187, view_191);  view_187 = view_191 = None
    sum_96: "f32[8, 512, 8, 8]" = torch.ops.aten.sum.dim_IntList(mul_969, [1]);  mul_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_166: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(sum_96, primals_499, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_701: "i64[]" = torch.ops.aten.add.Tensor(primals_923, 1)
    var_mean_134 = torch.ops.aten.var_mean.correction(convolution_166, [0, 2, 3], correction = 0, keepdim = True)
    getitem_270: "f32[1, 2048, 1, 1]" = var_mean_134[0]
    getitem_271: "f32[1, 2048, 1, 1]" = var_mean_134[1];  var_mean_134 = None
    add_702: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_270, 1e-05)
    rsqrt_134: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_702);  add_702 = None
    sub_166: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_166, getitem_271)
    mul_970: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_166, rsqrt_134);  sub_166 = None
    squeeze_402: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_271, [0, 2, 3]);  getitem_271 = None
    squeeze_403: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_134, [0, 2, 3]);  rsqrt_134 = None
    mul_971: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_402, 0.1)
    mul_972: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_921, 0.9)
    add_703: "f32[2048]" = torch.ops.aten.add.Tensor(mul_971, mul_972);  mul_971 = mul_972 = None
    squeeze_404: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_270, [0, 2, 3]);  getitem_270 = None
    mul_973: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_404, 1.0019569471624266);  squeeze_404 = None
    mul_974: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_973, 0.1);  mul_973 = None
    mul_975: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_922, 0.9)
    add_704: "f32[2048]" = torch.ops.aten.add.Tensor(mul_974, mul_975);  mul_974 = mul_975 = None
    unsqueeze_536: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_500, -1)
    unsqueeze_537: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
    mul_976: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_970, unsqueeze_537);  mul_970 = unsqueeze_537 = None
    unsqueeze_538: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_501, -1);  primals_501 = None
    unsqueeze_539: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
    add_705: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_976, unsqueeze_539);  mul_976 = unsqueeze_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_706: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_705, relu_126);  add_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_130: "f32[8, 2048, 8, 8]" = torch.ops.aten.relu.default(add_706);  add_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:94, code: out = self.conv1(x)
    convolution_167: "f32[8, 512, 8, 8]" = torch.ops.aten.convolution.default(relu_130, primals_502, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    add_707: "i64[]" = torch.ops.aten.add.Tensor(primals_926, 1)
    var_mean_135 = torch.ops.aten.var_mean.correction(convolution_167, [0, 2, 3], correction = 0, keepdim = True)
    getitem_272: "f32[1, 512, 1, 1]" = var_mean_135[0]
    getitem_273: "f32[1, 512, 1, 1]" = var_mean_135[1];  var_mean_135 = None
    add_708: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_272, 1e-05)
    rsqrt_135: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_708);  add_708 = None
    sub_167: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_167, getitem_273)
    mul_977: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_167, rsqrt_135);  sub_167 = None
    squeeze_405: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_273, [0, 2, 3]);  getitem_273 = None
    squeeze_406: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_135, [0, 2, 3]);  rsqrt_135 = None
    mul_978: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_405, 0.1)
    mul_979: "f32[512]" = torch.ops.aten.mul.Tensor(primals_924, 0.9)
    add_709: "f32[512]" = torch.ops.aten.add.Tensor(mul_978, mul_979);  mul_978 = mul_979 = None
    squeeze_407: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_272, [0, 2, 3]);  getitem_272 = None
    mul_980: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_407, 1.0019569471624266);  squeeze_407 = None
    mul_981: "f32[512]" = torch.ops.aten.mul.Tensor(mul_980, 0.1);  mul_980 = None
    mul_982: "f32[512]" = torch.ops.aten.mul.Tensor(primals_925, 0.9)
    add_710: "f32[512]" = torch.ops.aten.add.Tensor(mul_981, mul_982);  mul_981 = mul_982 = None
    unsqueeze_540: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_503, -1)
    unsqueeze_541: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
    mul_983: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_977, unsqueeze_541);  mul_977 = unsqueeze_541 = None
    unsqueeze_542: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_504, -1);  primals_504 = None
    unsqueeze_543: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
    add_711: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(mul_983, unsqueeze_543);  mul_983 = unsqueeze_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:96, code: out = self.act1(out)
    relu_131: "f32[8, 512, 8, 8]" = torch.ops.aten.relu.default(add_711);  add_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:62, code: x = self.conv(x)
    convolution_168: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(relu_131, primals_505, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    add_712: "i64[]" = torch.ops.aten.add.Tensor(primals_929, 1)
    var_mean_136 = torch.ops.aten.var_mean.correction(convolution_168, [0, 2, 3], correction = 0, keepdim = True)
    getitem_274: "f32[1, 1024, 1, 1]" = var_mean_136[0]
    getitem_275: "f32[1, 1024, 1, 1]" = var_mean_136[1];  var_mean_136 = None
    add_713: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_274, 1e-05)
    rsqrt_136: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_713);  add_713 = None
    sub_168: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_168, getitem_275)
    mul_984: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_168, rsqrt_136);  sub_168 = None
    squeeze_408: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_275, [0, 2, 3]);  getitem_275 = None
    squeeze_409: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_136, [0, 2, 3]);  rsqrt_136 = None
    mul_985: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_408, 0.1)
    mul_986: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_927, 0.9)
    add_714: "f32[1024]" = torch.ops.aten.add.Tensor(mul_985, mul_986);  mul_985 = mul_986 = None
    squeeze_410: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_274, [0, 2, 3]);  getitem_274 = None
    mul_987: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_410, 1.0019569471624266);  squeeze_410 = None
    mul_988: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_987, 0.1);  mul_987 = None
    mul_989: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_928, 0.9)
    add_715: "f32[1024]" = torch.ops.aten.add.Tensor(mul_988, mul_989);  mul_988 = mul_989 = None
    unsqueeze_544: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_506, -1)
    unsqueeze_545: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
    mul_990: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(mul_984, unsqueeze_545);  mul_984 = unsqueeze_545 = None
    unsqueeze_546: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_507, -1);  primals_507 = None
    unsqueeze_547: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
    add_716: "f32[8, 1024, 8, 8]" = torch.ops.aten.add.Tensor(mul_990, unsqueeze_547);  mul_990 = unsqueeze_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:65, code: x = self.act0(x)
    relu_132: "f32[8, 1024, 8, 8]" = torch.ops.aten.relu.default(add_716);  add_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:70, code: x_gap = x.sum(dim=1)
    view_193: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.reshape.default(relu_132, [8, 2, 512, 8, 8])
    sum_97: "f32[8, 512, 8, 8]" = torch.ops.aten.sum.dim_IntList(view_193, [1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:73, code: x_gap = x_gap.mean((2, 3), keepdim=True)
    mean_32: "f32[8, 512, 1, 1]" = torch.ops.aten.mean.dim(sum_97, [2, 3], True);  sum_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:74, code: x_gap = self.fc1(x_gap)
    convolution_169: "f32[8, 256, 1, 1]" = torch.ops.aten.convolution.default(mean_32, primals_508, primals_509, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:75, code: x_gap = self.bn1(x_gap)
    add_717: "i64[]" = torch.ops.aten.add.Tensor(primals_932, 1)
    var_mean_137 = torch.ops.aten.var_mean.correction(convolution_169, [0, 2, 3], correction = 0, keepdim = True)
    getitem_276: "f32[1, 256, 1, 1]" = var_mean_137[0]
    getitem_277: "f32[1, 256, 1, 1]" = var_mean_137[1];  var_mean_137 = None
    add_718: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_276, 1e-05)
    rsqrt_137: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_718);  add_718 = None
    sub_169: "f32[8, 256, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_169, getitem_277)
    mul_991: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(sub_169, rsqrt_137);  sub_169 = rsqrt_137 = None
    squeeze_411: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_277, [0, 2, 3]);  getitem_277 = None
    mul_992: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_411, 0.1);  squeeze_411 = None
    mul_993: "f32[256]" = torch.ops.aten.mul.Tensor(primals_930, 0.9)
    add_719: "f32[256]" = torch.ops.aten.add.Tensor(mul_992, mul_993);  mul_992 = mul_993 = None
    squeeze_413: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_276, [0, 2, 3]);  getitem_276 = None
    mul_994: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_413, 1.1428571428571428);  squeeze_413 = None
    mul_995: "f32[256]" = torch.ops.aten.mul.Tensor(mul_994, 0.1);  mul_994 = None
    mul_996: "f32[256]" = torch.ops.aten.mul.Tensor(primals_931, 0.9)
    add_720: "f32[256]" = torch.ops.aten.add.Tensor(mul_995, mul_996);  mul_995 = mul_996 = None
    unsqueeze_548: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_510, -1)
    unsqueeze_549: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
    mul_997: "f32[8, 256, 1, 1]" = torch.ops.aten.mul.Tensor(mul_991, unsqueeze_549);  mul_991 = unsqueeze_549 = None
    unsqueeze_550: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_511, -1);  primals_511 = None
    unsqueeze_551: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
    add_721: "f32[8, 256, 1, 1]" = torch.ops.aten.add.Tensor(mul_997, unsqueeze_551);  mul_997 = unsqueeze_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:76, code: x_gap = self.act1(x_gap)
    relu_133: "f32[8, 256, 1, 1]" = torch.ops.aten.relu.default(add_721);  add_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:77, code: x_attn = self.fc2(x_gap)
    convolution_170: "f32[8, 1024, 1, 1]" = torch.ops.aten.convolution.default(relu_133, primals_512, primals_513, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:25, code: x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
    view_194: "f32[8, 1, 2, 512]" = torch.ops.aten.reshape.default(convolution_170, [8, 1, 2, -1]);  convolution_170 = None
    permute_32: "f32[8, 2, 1, 512]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:26, code: x = F.softmax(x, dim=1)
    amax_32: "f32[8, 1, 1, 512]" = torch.ops.aten.amax.default(permute_32, [1], True)
    sub_170: "f32[8, 2, 1, 512]" = torch.ops.aten.sub.Tensor(permute_32, amax_32);  permute_32 = amax_32 = None
    exp_32: "f32[8, 2, 1, 512]" = torch.ops.aten.exp.default(sub_170);  sub_170 = None
    sum_98: "f32[8, 1, 1, 512]" = torch.ops.aten.sum.dim_IntList(exp_32, [1], True)
    div_32: "f32[8, 2, 1, 512]" = torch.ops.aten.div.Tensor(exp_32, sum_98);  exp_32 = sum_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:27, code: x = x.reshape(batch, -1)
    view_195: "f32[8, 1024]" = torch.ops.aten.reshape.default(div_32, [8, -1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:79, code: x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
    view_196: "f32[8, 1024, 1, 1]" = torch.ops.aten.reshape.default(view_195, [8, -1, 1, 1]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:81, code: out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
    view_197: "f32[8, 2, 512, 1, 1]" = torch.ops.aten.reshape.default(view_196, [8, 2, 512, 1, 1]);  view_196 = None
    mul_998: "f32[8, 2, 512, 8, 8]" = torch.ops.aten.mul.Tensor(view_193, view_197);  view_193 = view_197 = None
    sum_99: "f32[8, 512, 8, 8]" = torch.ops.aten.sum.dim_IntList(mul_998, [1]);  mul_998 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:109, code: out = self.conv3(out)
    convolution_171: "f32[8, 2048, 8, 8]" = torch.ops.aten.convolution.default(sum_99, primals_514, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    add_722: "i64[]" = torch.ops.aten.add.Tensor(primals_935, 1)
    var_mean_138 = torch.ops.aten.var_mean.correction(convolution_171, [0, 2, 3], correction = 0, keepdim = True)
    getitem_278: "f32[1, 2048, 1, 1]" = var_mean_138[0]
    getitem_279: "f32[1, 2048, 1, 1]" = var_mean_138[1];  var_mean_138 = None
    add_723: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_278, 1e-05)
    rsqrt_138: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_723);  add_723 = None
    sub_171: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_171, getitem_279)
    mul_999: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_171, rsqrt_138);  sub_171 = None
    squeeze_414: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_279, [0, 2, 3]);  getitem_279 = None
    squeeze_415: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_138, [0, 2, 3]);  rsqrt_138 = None
    mul_1000: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_414, 0.1)
    mul_1001: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_933, 0.9)
    add_724: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1000, mul_1001);  mul_1000 = mul_1001 = None
    squeeze_416: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_278, [0, 2, 3]);  getitem_278 = None
    mul_1002: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_416, 1.0019569471624266);  squeeze_416 = None
    mul_1003: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1002, 0.1);  mul_1002 = None
    mul_1004: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_934, 0.9)
    add_725: "f32[2048]" = torch.ops.aten.add.Tensor(mul_1003, mul_1004);  mul_1003 = mul_1004 = None
    unsqueeze_552: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_515, -1)
    unsqueeze_553: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
    mul_1005: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(mul_999, unsqueeze_553);  mul_999 = unsqueeze_553 = None
    unsqueeze_554: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_516, -1);  primals_516 = None
    unsqueeze_555: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
    add_726: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(mul_1005, unsqueeze_555);  mul_1005 = unsqueeze_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:115, code: out += shortcut
    add_727: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(add_726, relu_130);  add_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    relu_134: "f32[8, 2048, 8, 8]" = torch.ops.aten.relu.default(add_727);  add_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_33: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_134, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_198: "f32[8, 2048]" = torch.ops.aten.reshape.default(mean_33, [8, 2048]);  mean_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    permute_33: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_517, [1, 0]);  primals_517 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_518, view_198, permute_33);  primals_518 = None
    permute_34: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:116, code: out = self.act3(out)
    le: "b8[8, 2048, 8, 8]" = torch.ops.aten.le.Scalar(relu_134, 0);  relu_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_556: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_414, 0);  squeeze_414 = None
    unsqueeze_557: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
    unsqueeze_558: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_582: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_408, 0);  squeeze_408 = None
    unsqueeze_583: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 2);  unsqueeze_582 = None
    unsqueeze_584: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 3);  unsqueeze_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_594: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_405, 0);  squeeze_405 = None
    unsqueeze_595: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 2);  unsqueeze_594 = None
    unsqueeze_596: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 3);  unsqueeze_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_606: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_402, 0);  squeeze_402 = None
    unsqueeze_607: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 2);  unsqueeze_606 = None
    unsqueeze_608: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 3);  unsqueeze_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_632: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_396, 0);  squeeze_396 = None
    unsqueeze_633: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_644: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_393, 0);  squeeze_393 = None
    unsqueeze_645: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    unsqueeze_656: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_390, 0);  squeeze_390 = None
    unsqueeze_657: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_668: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_387, 0);  squeeze_387 = None
    unsqueeze_669: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_694: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_381, 0);  squeeze_381 = None
    unsqueeze_695: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_706: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_378, 0);  squeeze_378 = None
    unsqueeze_707: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_718: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_375, 0);  squeeze_375 = None
    unsqueeze_719: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_744: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_369, 0);  squeeze_369 = None
    unsqueeze_745: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 2);  unsqueeze_744 = None
    unsqueeze_746: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 3);  unsqueeze_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_756: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_366, 0);  squeeze_366 = None
    unsqueeze_757: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 2);  unsqueeze_756 = None
    unsqueeze_758: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 3);  unsqueeze_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_768: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_363, 0);  squeeze_363 = None
    unsqueeze_769: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 2);  unsqueeze_768 = None
    unsqueeze_770: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 3);  unsqueeze_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_794: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_357, 0);  squeeze_357 = None
    unsqueeze_795: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
    unsqueeze_796: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_806: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_354, 0);  squeeze_354 = None
    unsqueeze_807: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
    unsqueeze_808: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_818: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_351, 0);  squeeze_351 = None
    unsqueeze_819: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
    unsqueeze_820: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_844: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_345, 0);  squeeze_345 = None
    unsqueeze_845: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
    unsqueeze_846: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_856: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_342, 0);  squeeze_342 = None
    unsqueeze_857: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
    unsqueeze_858: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_868: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_339, 0);  squeeze_339 = None
    unsqueeze_869: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
    unsqueeze_870: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_894: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_333, 0);  squeeze_333 = None
    unsqueeze_895: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 2);  unsqueeze_894 = None
    unsqueeze_896: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 3);  unsqueeze_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_906: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_330, 0);  squeeze_330 = None
    unsqueeze_907: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 2);  unsqueeze_906 = None
    unsqueeze_908: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 3);  unsqueeze_907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_918: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_327, 0);  squeeze_327 = None
    unsqueeze_919: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 2);  unsqueeze_918 = None
    unsqueeze_920: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 3);  unsqueeze_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_944: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_321, 0);  squeeze_321 = None
    unsqueeze_945: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 2);  unsqueeze_944 = None
    unsqueeze_946: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 3);  unsqueeze_945 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_956: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_318, 0);  squeeze_318 = None
    unsqueeze_957: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 2);  unsqueeze_956 = None
    unsqueeze_958: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 3);  unsqueeze_957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_968: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_315, 0);  squeeze_315 = None
    unsqueeze_969: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
    unsqueeze_970: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_994: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_309, 0);  squeeze_309 = None
    unsqueeze_995: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 2);  unsqueeze_994 = None
    unsqueeze_996: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 3);  unsqueeze_995 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1006: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_306, 0);  squeeze_306 = None
    unsqueeze_1007: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 2);  unsqueeze_1006 = None
    unsqueeze_1008: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 3);  unsqueeze_1007 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1018: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_303, 0);  squeeze_303 = None
    unsqueeze_1019: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 2);  unsqueeze_1018 = None
    unsqueeze_1020: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 3);  unsqueeze_1019 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1044: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_297, 0);  squeeze_297 = None
    unsqueeze_1045: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, 2);  unsqueeze_1044 = None
    unsqueeze_1046: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 3);  unsqueeze_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1056: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_294, 0);  squeeze_294 = None
    unsqueeze_1057: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, 2);  unsqueeze_1056 = None
    unsqueeze_1058: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 3);  unsqueeze_1057 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1068: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_291, 0);  squeeze_291 = None
    unsqueeze_1069: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, 2);  unsqueeze_1068 = None
    unsqueeze_1070: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 3);  unsqueeze_1069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1094: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_285, 0);  squeeze_285 = None
    unsqueeze_1095: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 2);  unsqueeze_1094 = None
    unsqueeze_1096: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 3);  unsqueeze_1095 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1106: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_282, 0);  squeeze_282 = None
    unsqueeze_1107: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 2);  unsqueeze_1106 = None
    unsqueeze_1108: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 3);  unsqueeze_1107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1118: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_279, 0);  squeeze_279 = None
    unsqueeze_1119: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 2);  unsqueeze_1118 = None
    unsqueeze_1120: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 3);  unsqueeze_1119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1144: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_273, 0);  squeeze_273 = None
    unsqueeze_1145: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 2);  unsqueeze_1144 = None
    unsqueeze_1146: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 3);  unsqueeze_1145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1156: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_270, 0);  squeeze_270 = None
    unsqueeze_1157: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 2);  unsqueeze_1156 = None
    unsqueeze_1158: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 3);  unsqueeze_1157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1168: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_267, 0);  squeeze_267 = None
    unsqueeze_1169: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 2);  unsqueeze_1168 = None
    unsqueeze_1170: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 3);  unsqueeze_1169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1194: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_261, 0);  squeeze_261 = None
    unsqueeze_1195: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 2);  unsqueeze_1194 = None
    unsqueeze_1196: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 3);  unsqueeze_1195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1206: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_258, 0);  squeeze_258 = None
    unsqueeze_1207: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 2);  unsqueeze_1206 = None
    unsqueeze_1208: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 3);  unsqueeze_1207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1218: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_255, 0);  squeeze_255 = None
    unsqueeze_1219: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 2);  unsqueeze_1218 = None
    unsqueeze_1220: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 3);  unsqueeze_1219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1244: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
    unsqueeze_1245: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, 2);  unsqueeze_1244 = None
    unsqueeze_1246: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 3);  unsqueeze_1245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1256: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
    unsqueeze_1257: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, 2);  unsqueeze_1256 = None
    unsqueeze_1258: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 3);  unsqueeze_1257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1268: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
    unsqueeze_1269: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, 2);  unsqueeze_1268 = None
    unsqueeze_1270: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 3);  unsqueeze_1269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1294: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
    unsqueeze_1295: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 2);  unsqueeze_1294 = None
    unsqueeze_1296: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1295, 3);  unsqueeze_1295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1306: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
    unsqueeze_1307: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 2);  unsqueeze_1306 = None
    unsqueeze_1308: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1307, 3);  unsqueeze_1307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1318: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
    unsqueeze_1319: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 2);  unsqueeze_1318 = None
    unsqueeze_1320: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1319, 3);  unsqueeze_1319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1344: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
    unsqueeze_1345: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, 2);  unsqueeze_1344 = None
    unsqueeze_1346: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1345, 3);  unsqueeze_1345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1356: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
    unsqueeze_1357: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, 2);  unsqueeze_1356 = None
    unsqueeze_1358: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1357, 3);  unsqueeze_1357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1368: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
    unsqueeze_1369: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, 2);  unsqueeze_1368 = None
    unsqueeze_1370: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1369, 3);  unsqueeze_1369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1394: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
    unsqueeze_1395: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, 2);  unsqueeze_1394 = None
    unsqueeze_1396: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1395, 3);  unsqueeze_1395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1406: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
    unsqueeze_1407: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, 2);  unsqueeze_1406 = None
    unsqueeze_1408: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1407, 3);  unsqueeze_1407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1418: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
    unsqueeze_1419: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, 2);  unsqueeze_1418 = None
    unsqueeze_1420: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1419, 3);  unsqueeze_1419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1444: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
    unsqueeze_1445: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, 2);  unsqueeze_1444 = None
    unsqueeze_1446: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1445, 3);  unsqueeze_1445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1456: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
    unsqueeze_1457: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, 2);  unsqueeze_1456 = None
    unsqueeze_1458: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1457, 3);  unsqueeze_1457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1468: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
    unsqueeze_1469: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, 2);  unsqueeze_1468 = None
    unsqueeze_1470: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1469, 3);  unsqueeze_1469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1494: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_1495: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, 2);  unsqueeze_1494 = None
    unsqueeze_1496: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1495, 3);  unsqueeze_1495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1506: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_1507: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, 2);  unsqueeze_1506 = None
    unsqueeze_1508: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1507, 3);  unsqueeze_1507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1518: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_1519: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, 2);  unsqueeze_1518 = None
    unsqueeze_1520: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1519, 3);  unsqueeze_1519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1544: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_1545: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1544, 2);  unsqueeze_1544 = None
    unsqueeze_1546: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1545, 3);  unsqueeze_1545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1556: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_1557: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1556, 2);  unsqueeze_1556 = None
    unsqueeze_1558: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1557, 3);  unsqueeze_1557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1568: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_1569: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1568, 2);  unsqueeze_1568 = None
    unsqueeze_1570: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1569, 3);  unsqueeze_1569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1594: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_1595: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1594, 2);  unsqueeze_1594 = None
    unsqueeze_1596: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1595, 3);  unsqueeze_1595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1606: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_1607: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1606, 2);  unsqueeze_1606 = None
    unsqueeze_1608: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1607, 3);  unsqueeze_1607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1618: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_1619: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1618, 2);  unsqueeze_1618 = None
    unsqueeze_1620: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1619, 3);  unsqueeze_1619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1644: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_1645: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1644, 2);  unsqueeze_1644 = None
    unsqueeze_1646: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1645, 3);  unsqueeze_1645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1656: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_1657: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1656, 2);  unsqueeze_1656 = None
    unsqueeze_1658: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1657, 3);  unsqueeze_1657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1668: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_1669: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1668, 2);  unsqueeze_1668 = None
    unsqueeze_1670: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1669, 3);  unsqueeze_1669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1694: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_1695: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1694, 2);  unsqueeze_1694 = None
    unsqueeze_1696: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1695, 3);  unsqueeze_1695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1706: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_1707: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1706, 2);  unsqueeze_1706 = None
    unsqueeze_1708: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1707, 3);  unsqueeze_1707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1718: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_1719: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1718, 2);  unsqueeze_1718 = None
    unsqueeze_1720: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1719, 3);  unsqueeze_1719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1744: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_1745: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1744, 2);  unsqueeze_1744 = None
    unsqueeze_1746: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1745, 3);  unsqueeze_1745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1756: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_1757: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1756, 2);  unsqueeze_1756 = None
    unsqueeze_1758: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1757, 3);  unsqueeze_1757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1768: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_1769: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1768, 2);  unsqueeze_1768 = None
    unsqueeze_1770: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1769, 3);  unsqueeze_1769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1794: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_1795: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1794, 2);  unsqueeze_1794 = None
    unsqueeze_1796: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1795, 3);  unsqueeze_1795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1806: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_1807: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1806, 2);  unsqueeze_1806 = None
    unsqueeze_1808: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1807, 3);  unsqueeze_1807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    unsqueeze_1818: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_1819: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1818, 2);  unsqueeze_1818 = None
    unsqueeze_1820: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1819, 3);  unsqueeze_1819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1830: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_1831: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1830, 2);  unsqueeze_1830 = None
    unsqueeze_1832: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1831, 3);  unsqueeze_1831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1856: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_1857: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1856, 2);  unsqueeze_1856 = None
    unsqueeze_1858: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1857, 3);  unsqueeze_1857 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1868: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_1869: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1868, 2);  unsqueeze_1868 = None
    unsqueeze_1870: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1869, 3);  unsqueeze_1869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1880: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_1881: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1880, 2);  unsqueeze_1880 = None
    unsqueeze_1882: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1881, 3);  unsqueeze_1881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1906: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_1907: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1906, 2);  unsqueeze_1906 = None
    unsqueeze_1908: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1907, 3);  unsqueeze_1907 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1918: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_1919: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1918, 2);  unsqueeze_1918 = None
    unsqueeze_1920: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1919, 3);  unsqueeze_1919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1930: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_1931: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1930, 2);  unsqueeze_1930 = None
    unsqueeze_1932: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1931, 3);  unsqueeze_1931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_1956: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_1957: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1956, 2);  unsqueeze_1956 = None
    unsqueeze_1958: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1957, 3);  unsqueeze_1957 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_1968: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_1969: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1968, 2);  unsqueeze_1968 = None
    unsqueeze_1970: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1969, 3);  unsqueeze_1969 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_1980: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_1981: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1980, 2);  unsqueeze_1980 = None
    unsqueeze_1982: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1981, 3);  unsqueeze_1981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_2006: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_2007: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2006, 2);  unsqueeze_2006 = None
    unsqueeze_2008: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2007, 3);  unsqueeze_2007 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_2018: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_2019: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2018, 2);  unsqueeze_2018 = None
    unsqueeze_2020: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2019, 3);  unsqueeze_2019 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    unsqueeze_2030: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_2031: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2030, 2);  unsqueeze_2030 = None
    unsqueeze_2032: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2031, 3);  unsqueeze_2031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_2042: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_2043: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2042, 2);  unsqueeze_2042 = None
    unsqueeze_2044: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2043, 3);  unsqueeze_2043 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_2068: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_2069: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2068, 2);  unsqueeze_2068 = None
    unsqueeze_2070: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2069, 3);  unsqueeze_2069 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_2080: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_2081: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2080, 2);  unsqueeze_2080 = None
    unsqueeze_2082: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2081, 3);  unsqueeze_2081 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_2092: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_2093: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2092, 2);  unsqueeze_2092 = None
    unsqueeze_2094: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2093, 3);  unsqueeze_2093 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_2118: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_2119: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2118, 2);  unsqueeze_2118 = None
    unsqueeze_2120: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2119, 3);  unsqueeze_2119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_2130: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_2131: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2130, 2);  unsqueeze_2130 = None
    unsqueeze_2132: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2131, 3);  unsqueeze_2131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_2142: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_2143: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2142, 2);  unsqueeze_2142 = None
    unsqueeze_2144: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2143, 3);  unsqueeze_2143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_2168: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_2169: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2168, 2);  unsqueeze_2168 = None
    unsqueeze_2170: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2169, 3);  unsqueeze_2169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_2180: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_2181: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2180, 2);  unsqueeze_2180 = None
    unsqueeze_2182: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2181, 3);  unsqueeze_2181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:113, code: shortcut = self.downsample(x)
    unsqueeze_2192: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_2193: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2192, 2);  unsqueeze_2192 = None
    unsqueeze_2194: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2193, 3);  unsqueeze_2193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:110, code: out = self.bn3(out)
    unsqueeze_2204: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_2205: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2204, 2);  unsqueeze_2204 = None
    unsqueeze_2206: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2205, 3);  unsqueeze_2205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/split_attn.py:63, code: x = self.bn0(x)
    unsqueeze_2230: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_2231: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2230, 2);  unsqueeze_2230 = None
    unsqueeze_2232: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2231, 3);  unsqueeze_2231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnest.py:95, code: out = self.bn1(out)
    unsqueeze_2242: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_2243: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2242, 2);  unsqueeze_2242 = None
    unsqueeze_2244: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2243, 3);  unsqueeze_2243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    unsqueeze_2254: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_2255: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2254, 2);  unsqueeze_2254 = None
    unsqueeze_2256: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2255, 3);  unsqueeze_2255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    unsqueeze_2266: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_2267: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2266, 2);  unsqueeze_2266 = None
    unsqueeze_2268: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2267, 3);  unsqueeze_2267 = None
    unsqueeze_2278: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_2279: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2278, 2);  unsqueeze_2278 = None
    unsqueeze_2280: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2279, 3);  unsqueeze_2279 = None
    
    # No stacktrace found for following nodes
    copy_: "f32[64]" = torch.ops.aten.copy_.default(primals_519, add_2);  primals_519 = add_2 = None
    copy__1: "f32[64]" = torch.ops.aten.copy_.default(primals_520, add_3);  primals_520 = add_3 = None
    copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_521, add);  primals_521 = add = None
    copy__3: "f32[64]" = torch.ops.aten.copy_.default(primals_522, add_7);  primals_522 = add_7 = None
    copy__4: "f32[64]" = torch.ops.aten.copy_.default(primals_523, add_8);  primals_523 = add_8 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_524, add_5);  primals_524 = add_5 = None
    copy__6: "f32[128]" = torch.ops.aten.copy_.default(primals_525, add_12);  primals_525 = add_12 = None
    copy__7: "f32[128]" = torch.ops.aten.copy_.default(primals_526, add_13);  primals_526 = add_13 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_527, add_10);  primals_527 = add_10 = None
    copy__9: "f32[64]" = torch.ops.aten.copy_.default(primals_528, add_17);  primals_528 = add_17 = None
    copy__10: "f32[64]" = torch.ops.aten.copy_.default(primals_529, add_18);  primals_529 = add_18 = None
    copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_530, add_15);  primals_530 = add_15 = None
    copy__12: "f32[128]" = torch.ops.aten.copy_.default(primals_531, add_22);  primals_531 = add_22 = None
    copy__13: "f32[128]" = torch.ops.aten.copy_.default(primals_532, add_23);  primals_532 = add_23 = None
    copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_533, add_20);  primals_533 = add_20 = None
    copy__15: "f32[32]" = torch.ops.aten.copy_.default(primals_534, add_27);  primals_534 = add_27 = None
    copy__16: "f32[32]" = torch.ops.aten.copy_.default(primals_535, add_28);  primals_535 = add_28 = None
    copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_536, add_25);  primals_536 = add_25 = None
    copy__18: "f32[256]" = torch.ops.aten.copy_.default(primals_537, add_32);  primals_537 = add_32 = None
    copy__19: "f32[256]" = torch.ops.aten.copy_.default(primals_538, add_33);  primals_538 = add_33 = None
    copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_539, add_30);  primals_539 = add_30 = None
    copy__21: "f32[256]" = torch.ops.aten.copy_.default(primals_540, add_37);  primals_540 = add_37 = None
    copy__22: "f32[256]" = torch.ops.aten.copy_.default(primals_541, add_38);  primals_541 = add_38 = None
    copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_542, add_35);  primals_542 = add_35 = None
    copy__24: "f32[64]" = torch.ops.aten.copy_.default(primals_543, add_43);  primals_543 = add_43 = None
    copy__25: "f32[64]" = torch.ops.aten.copy_.default(primals_544, add_44);  primals_544 = add_44 = None
    copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_545, add_41);  primals_545 = add_41 = None
    copy__27: "f32[128]" = torch.ops.aten.copy_.default(primals_546, add_48);  primals_546 = add_48 = None
    copy__28: "f32[128]" = torch.ops.aten.copy_.default(primals_547, add_49);  primals_547 = add_49 = None
    copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_548, add_46);  primals_548 = add_46 = None
    copy__30: "f32[32]" = torch.ops.aten.copy_.default(primals_549, add_53);  primals_549 = add_53 = None
    copy__31: "f32[32]" = torch.ops.aten.copy_.default(primals_550, add_54);  primals_550 = add_54 = None
    copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_551, add_51);  primals_551 = add_51 = None
    copy__33: "f32[256]" = torch.ops.aten.copy_.default(primals_552, add_58);  primals_552 = add_58 = None
    copy__34: "f32[256]" = torch.ops.aten.copy_.default(primals_553, add_59);  primals_553 = add_59 = None
    copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_554, add_56);  primals_554 = add_56 = None
    copy__36: "f32[64]" = torch.ops.aten.copy_.default(primals_555, add_64);  primals_555 = add_64 = None
    copy__37: "f32[64]" = torch.ops.aten.copy_.default(primals_556, add_65);  primals_556 = add_65 = None
    copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_557, add_62);  primals_557 = add_62 = None
    copy__39: "f32[128]" = torch.ops.aten.copy_.default(primals_558, add_69);  primals_558 = add_69 = None
    copy__40: "f32[128]" = torch.ops.aten.copy_.default(primals_559, add_70);  primals_559 = add_70 = None
    copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_560, add_67);  primals_560 = add_67 = None
    copy__42: "f32[32]" = torch.ops.aten.copy_.default(primals_561, add_74);  primals_561 = add_74 = None
    copy__43: "f32[32]" = torch.ops.aten.copy_.default(primals_562, add_75);  primals_562 = add_75 = None
    copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_563, add_72);  primals_563 = add_72 = None
    copy__45: "f32[256]" = torch.ops.aten.copy_.default(primals_564, add_79);  primals_564 = add_79 = None
    copy__46: "f32[256]" = torch.ops.aten.copy_.default(primals_565, add_80);  primals_565 = add_80 = None
    copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_566, add_77);  primals_566 = add_77 = None
    copy__48: "f32[128]" = torch.ops.aten.copy_.default(primals_567, add_85);  primals_567 = add_85 = None
    copy__49: "f32[128]" = torch.ops.aten.copy_.default(primals_568, add_86);  primals_568 = add_86 = None
    copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_569, add_83);  primals_569 = add_83 = None
    copy__51: "f32[256]" = torch.ops.aten.copy_.default(primals_570, add_90);  primals_570 = add_90 = None
    copy__52: "f32[256]" = torch.ops.aten.copy_.default(primals_571, add_91);  primals_571 = add_91 = None
    copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_572, add_88);  primals_572 = add_88 = None
    copy__54: "f32[64]" = torch.ops.aten.copy_.default(primals_573, add_95);  primals_573 = add_95 = None
    copy__55: "f32[64]" = torch.ops.aten.copy_.default(primals_574, add_96);  primals_574 = add_96 = None
    copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_575, add_93);  primals_575 = add_93 = None
    copy__57: "f32[512]" = torch.ops.aten.copy_.default(primals_576, add_100);  primals_576 = add_100 = None
    copy__58: "f32[512]" = torch.ops.aten.copy_.default(primals_577, add_101);  primals_577 = add_101 = None
    copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_578, add_98);  primals_578 = add_98 = None
    copy__60: "f32[512]" = torch.ops.aten.copy_.default(primals_579, add_105);  primals_579 = add_105 = None
    copy__61: "f32[512]" = torch.ops.aten.copy_.default(primals_580, add_106);  primals_580 = add_106 = None
    copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_581, add_103);  primals_581 = add_103 = None
    copy__63: "f32[128]" = torch.ops.aten.copy_.default(primals_582, add_111);  primals_582 = add_111 = None
    copy__64: "f32[128]" = torch.ops.aten.copy_.default(primals_583, add_112);  primals_583 = add_112 = None
    copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_584, add_109);  primals_584 = add_109 = None
    copy__66: "f32[256]" = torch.ops.aten.copy_.default(primals_585, add_116);  primals_585 = add_116 = None
    copy__67: "f32[256]" = torch.ops.aten.copy_.default(primals_586, add_117);  primals_586 = add_117 = None
    copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_587, add_114);  primals_587 = add_114 = None
    copy__69: "f32[64]" = torch.ops.aten.copy_.default(primals_588, add_121);  primals_588 = add_121 = None
    copy__70: "f32[64]" = torch.ops.aten.copy_.default(primals_589, add_122);  primals_589 = add_122 = None
    copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_590, add_119);  primals_590 = add_119 = None
    copy__72: "f32[512]" = torch.ops.aten.copy_.default(primals_591, add_126);  primals_591 = add_126 = None
    copy__73: "f32[512]" = torch.ops.aten.copy_.default(primals_592, add_127);  primals_592 = add_127 = None
    copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_593, add_124);  primals_593 = add_124 = None
    copy__75: "f32[128]" = torch.ops.aten.copy_.default(primals_594, add_132);  primals_594 = add_132 = None
    copy__76: "f32[128]" = torch.ops.aten.copy_.default(primals_595, add_133);  primals_595 = add_133 = None
    copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_596, add_130);  primals_596 = add_130 = None
    copy__78: "f32[256]" = torch.ops.aten.copy_.default(primals_597, add_137);  primals_597 = add_137 = None
    copy__79: "f32[256]" = torch.ops.aten.copy_.default(primals_598, add_138);  primals_598 = add_138 = None
    copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_599, add_135);  primals_599 = add_135 = None
    copy__81: "f32[64]" = torch.ops.aten.copy_.default(primals_600, add_142);  primals_600 = add_142 = None
    copy__82: "f32[64]" = torch.ops.aten.copy_.default(primals_601, add_143);  primals_601 = add_143 = None
    copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_602, add_140);  primals_602 = add_140 = None
    copy__84: "f32[512]" = torch.ops.aten.copy_.default(primals_603, add_147);  primals_603 = add_147 = None
    copy__85: "f32[512]" = torch.ops.aten.copy_.default(primals_604, add_148);  primals_604 = add_148 = None
    copy__86: "i64[]" = torch.ops.aten.copy_.default(primals_605, add_145);  primals_605 = add_145 = None
    copy__87: "f32[128]" = torch.ops.aten.copy_.default(primals_606, add_153);  primals_606 = add_153 = None
    copy__88: "f32[128]" = torch.ops.aten.copy_.default(primals_607, add_154);  primals_607 = add_154 = None
    copy__89: "i64[]" = torch.ops.aten.copy_.default(primals_608, add_151);  primals_608 = add_151 = None
    copy__90: "f32[256]" = torch.ops.aten.copy_.default(primals_609, add_158);  primals_609 = add_158 = None
    copy__91: "f32[256]" = torch.ops.aten.copy_.default(primals_610, add_159);  primals_610 = add_159 = None
    copy__92: "i64[]" = torch.ops.aten.copy_.default(primals_611, add_156);  primals_611 = add_156 = None
    copy__93: "f32[64]" = torch.ops.aten.copy_.default(primals_612, add_163);  primals_612 = add_163 = None
    copy__94: "f32[64]" = torch.ops.aten.copy_.default(primals_613, add_164);  primals_613 = add_164 = None
    copy__95: "i64[]" = torch.ops.aten.copy_.default(primals_614, add_161);  primals_614 = add_161 = None
    copy__96: "f32[512]" = torch.ops.aten.copy_.default(primals_615, add_168);  primals_615 = add_168 = None
    copy__97: "f32[512]" = torch.ops.aten.copy_.default(primals_616, add_169);  primals_616 = add_169 = None
    copy__98: "i64[]" = torch.ops.aten.copy_.default(primals_617, add_166);  primals_617 = add_166 = None
    copy__99: "f32[256]" = torch.ops.aten.copy_.default(primals_618, add_174);  primals_618 = add_174 = None
    copy__100: "f32[256]" = torch.ops.aten.copy_.default(primals_619, add_175);  primals_619 = add_175 = None
    copy__101: "i64[]" = torch.ops.aten.copy_.default(primals_620, add_172);  primals_620 = add_172 = None
    copy__102: "f32[512]" = torch.ops.aten.copy_.default(primals_621, add_179);  primals_621 = add_179 = None
    copy__103: "f32[512]" = torch.ops.aten.copy_.default(primals_622, add_180);  primals_622 = add_180 = None
    copy__104: "i64[]" = torch.ops.aten.copy_.default(primals_623, add_177);  primals_623 = add_177 = None
    copy__105: "f32[128]" = torch.ops.aten.copy_.default(primals_624, add_184);  primals_624 = add_184 = None
    copy__106: "f32[128]" = torch.ops.aten.copy_.default(primals_625, add_185);  primals_625 = add_185 = None
    copy__107: "i64[]" = torch.ops.aten.copy_.default(primals_626, add_182);  primals_626 = add_182 = None
    copy__108: "f32[1024]" = torch.ops.aten.copy_.default(primals_627, add_189);  primals_627 = add_189 = None
    copy__109: "f32[1024]" = torch.ops.aten.copy_.default(primals_628, add_190);  primals_628 = add_190 = None
    copy__110: "i64[]" = torch.ops.aten.copy_.default(primals_629, add_187);  primals_629 = add_187 = None
    copy__111: "f32[1024]" = torch.ops.aten.copy_.default(primals_630, add_194);  primals_630 = add_194 = None
    copy__112: "f32[1024]" = torch.ops.aten.copy_.default(primals_631, add_195);  primals_631 = add_195 = None
    copy__113: "i64[]" = torch.ops.aten.copy_.default(primals_632, add_192);  primals_632 = add_192 = None
    copy__114: "f32[256]" = torch.ops.aten.copy_.default(primals_633, add_200);  primals_633 = add_200 = None
    copy__115: "f32[256]" = torch.ops.aten.copy_.default(primals_634, add_201);  primals_634 = add_201 = None
    copy__116: "i64[]" = torch.ops.aten.copy_.default(primals_635, add_198);  primals_635 = add_198 = None
    copy__117: "f32[512]" = torch.ops.aten.copy_.default(primals_636, add_205);  primals_636 = add_205 = None
    copy__118: "f32[512]" = torch.ops.aten.copy_.default(primals_637, add_206);  primals_637 = add_206 = None
    copy__119: "i64[]" = torch.ops.aten.copy_.default(primals_638, add_203);  primals_638 = add_203 = None
    copy__120: "f32[128]" = torch.ops.aten.copy_.default(primals_639, add_210);  primals_639 = add_210 = None
    copy__121: "f32[128]" = torch.ops.aten.copy_.default(primals_640, add_211);  primals_640 = add_211 = None
    copy__122: "i64[]" = torch.ops.aten.copy_.default(primals_641, add_208);  primals_641 = add_208 = None
    copy__123: "f32[1024]" = torch.ops.aten.copy_.default(primals_642, add_215);  primals_642 = add_215 = None
    copy__124: "f32[1024]" = torch.ops.aten.copy_.default(primals_643, add_216);  primals_643 = add_216 = None
    copy__125: "i64[]" = torch.ops.aten.copy_.default(primals_644, add_213);  primals_644 = add_213 = None
    copy__126: "f32[256]" = torch.ops.aten.copy_.default(primals_645, add_221);  primals_645 = add_221 = None
    copy__127: "f32[256]" = torch.ops.aten.copy_.default(primals_646, add_222);  primals_646 = add_222 = None
    copy__128: "i64[]" = torch.ops.aten.copy_.default(primals_647, add_219);  primals_647 = add_219 = None
    copy__129: "f32[512]" = torch.ops.aten.copy_.default(primals_648, add_226);  primals_648 = add_226 = None
    copy__130: "f32[512]" = torch.ops.aten.copy_.default(primals_649, add_227);  primals_649 = add_227 = None
    copy__131: "i64[]" = torch.ops.aten.copy_.default(primals_650, add_224);  primals_650 = add_224 = None
    copy__132: "f32[128]" = torch.ops.aten.copy_.default(primals_651, add_231);  primals_651 = add_231 = None
    copy__133: "f32[128]" = torch.ops.aten.copy_.default(primals_652, add_232);  primals_652 = add_232 = None
    copy__134: "i64[]" = torch.ops.aten.copy_.default(primals_653, add_229);  primals_653 = add_229 = None
    copy__135: "f32[1024]" = torch.ops.aten.copy_.default(primals_654, add_236);  primals_654 = add_236 = None
    copy__136: "f32[1024]" = torch.ops.aten.copy_.default(primals_655, add_237);  primals_655 = add_237 = None
    copy__137: "i64[]" = torch.ops.aten.copy_.default(primals_656, add_234);  primals_656 = add_234 = None
    copy__138: "f32[256]" = torch.ops.aten.copy_.default(primals_657, add_242);  primals_657 = add_242 = None
    copy__139: "f32[256]" = torch.ops.aten.copy_.default(primals_658, add_243);  primals_658 = add_243 = None
    copy__140: "i64[]" = torch.ops.aten.copy_.default(primals_659, add_240);  primals_659 = add_240 = None
    copy__141: "f32[512]" = torch.ops.aten.copy_.default(primals_660, add_247);  primals_660 = add_247 = None
    copy__142: "f32[512]" = torch.ops.aten.copy_.default(primals_661, add_248);  primals_661 = add_248 = None
    copy__143: "i64[]" = torch.ops.aten.copy_.default(primals_662, add_245);  primals_662 = add_245 = None
    copy__144: "f32[128]" = torch.ops.aten.copy_.default(primals_663, add_252);  primals_663 = add_252 = None
    copy__145: "f32[128]" = torch.ops.aten.copy_.default(primals_664, add_253);  primals_664 = add_253 = None
    copy__146: "i64[]" = torch.ops.aten.copy_.default(primals_665, add_250);  primals_665 = add_250 = None
    copy__147: "f32[1024]" = torch.ops.aten.copy_.default(primals_666, add_257);  primals_666 = add_257 = None
    copy__148: "f32[1024]" = torch.ops.aten.copy_.default(primals_667, add_258);  primals_667 = add_258 = None
    copy__149: "i64[]" = torch.ops.aten.copy_.default(primals_668, add_255);  primals_668 = add_255 = None
    copy__150: "f32[256]" = torch.ops.aten.copy_.default(primals_669, add_263);  primals_669 = add_263 = None
    copy__151: "f32[256]" = torch.ops.aten.copy_.default(primals_670, add_264);  primals_670 = add_264 = None
    copy__152: "i64[]" = torch.ops.aten.copy_.default(primals_671, add_261);  primals_671 = add_261 = None
    copy__153: "f32[512]" = torch.ops.aten.copy_.default(primals_672, add_268);  primals_672 = add_268 = None
    copy__154: "f32[512]" = torch.ops.aten.copy_.default(primals_673, add_269);  primals_673 = add_269 = None
    copy__155: "i64[]" = torch.ops.aten.copy_.default(primals_674, add_266);  primals_674 = add_266 = None
    copy__156: "f32[128]" = torch.ops.aten.copy_.default(primals_675, add_273);  primals_675 = add_273 = None
    copy__157: "f32[128]" = torch.ops.aten.copy_.default(primals_676, add_274);  primals_676 = add_274 = None
    copy__158: "i64[]" = torch.ops.aten.copy_.default(primals_677, add_271);  primals_677 = add_271 = None
    copy__159: "f32[1024]" = torch.ops.aten.copy_.default(primals_678, add_278);  primals_678 = add_278 = None
    copy__160: "f32[1024]" = torch.ops.aten.copy_.default(primals_679, add_279);  primals_679 = add_279 = None
    copy__161: "i64[]" = torch.ops.aten.copy_.default(primals_680, add_276);  primals_680 = add_276 = None
    copy__162: "f32[256]" = torch.ops.aten.copy_.default(primals_681, add_284);  primals_681 = add_284 = None
    copy__163: "f32[256]" = torch.ops.aten.copy_.default(primals_682, add_285);  primals_682 = add_285 = None
    copy__164: "i64[]" = torch.ops.aten.copy_.default(primals_683, add_282);  primals_683 = add_282 = None
    copy__165: "f32[512]" = torch.ops.aten.copy_.default(primals_684, add_289);  primals_684 = add_289 = None
    copy__166: "f32[512]" = torch.ops.aten.copy_.default(primals_685, add_290);  primals_685 = add_290 = None
    copy__167: "i64[]" = torch.ops.aten.copy_.default(primals_686, add_287);  primals_686 = add_287 = None
    copy__168: "f32[128]" = torch.ops.aten.copy_.default(primals_687, add_294);  primals_687 = add_294 = None
    copy__169: "f32[128]" = torch.ops.aten.copy_.default(primals_688, add_295);  primals_688 = add_295 = None
    copy__170: "i64[]" = torch.ops.aten.copy_.default(primals_689, add_292);  primals_689 = add_292 = None
    copy__171: "f32[1024]" = torch.ops.aten.copy_.default(primals_690, add_299);  primals_690 = add_299 = None
    copy__172: "f32[1024]" = torch.ops.aten.copy_.default(primals_691, add_300);  primals_691 = add_300 = None
    copy__173: "i64[]" = torch.ops.aten.copy_.default(primals_692, add_297);  primals_692 = add_297 = None
    copy__174: "f32[256]" = torch.ops.aten.copy_.default(primals_693, add_305);  primals_693 = add_305 = None
    copy__175: "f32[256]" = torch.ops.aten.copy_.default(primals_694, add_306);  primals_694 = add_306 = None
    copy__176: "i64[]" = torch.ops.aten.copy_.default(primals_695, add_303);  primals_695 = add_303 = None
    copy__177: "f32[512]" = torch.ops.aten.copy_.default(primals_696, add_310);  primals_696 = add_310 = None
    copy__178: "f32[512]" = torch.ops.aten.copy_.default(primals_697, add_311);  primals_697 = add_311 = None
    copy__179: "i64[]" = torch.ops.aten.copy_.default(primals_698, add_308);  primals_698 = add_308 = None
    copy__180: "f32[128]" = torch.ops.aten.copy_.default(primals_699, add_315);  primals_699 = add_315 = None
    copy__181: "f32[128]" = torch.ops.aten.copy_.default(primals_700, add_316);  primals_700 = add_316 = None
    copy__182: "i64[]" = torch.ops.aten.copy_.default(primals_701, add_313);  primals_701 = add_313 = None
    copy__183: "f32[1024]" = torch.ops.aten.copy_.default(primals_702, add_320);  primals_702 = add_320 = None
    copy__184: "f32[1024]" = torch.ops.aten.copy_.default(primals_703, add_321);  primals_703 = add_321 = None
    copy__185: "i64[]" = torch.ops.aten.copy_.default(primals_704, add_318);  primals_704 = add_318 = None
    copy__186: "f32[256]" = torch.ops.aten.copy_.default(primals_705, add_326);  primals_705 = add_326 = None
    copy__187: "f32[256]" = torch.ops.aten.copy_.default(primals_706, add_327);  primals_706 = add_327 = None
    copy__188: "i64[]" = torch.ops.aten.copy_.default(primals_707, add_324);  primals_707 = add_324 = None
    copy__189: "f32[512]" = torch.ops.aten.copy_.default(primals_708, add_331);  primals_708 = add_331 = None
    copy__190: "f32[512]" = torch.ops.aten.copy_.default(primals_709, add_332);  primals_709 = add_332 = None
    copy__191: "i64[]" = torch.ops.aten.copy_.default(primals_710, add_329);  primals_710 = add_329 = None
    copy__192: "f32[128]" = torch.ops.aten.copy_.default(primals_711, add_336);  primals_711 = add_336 = None
    copy__193: "f32[128]" = torch.ops.aten.copy_.default(primals_712, add_337);  primals_712 = add_337 = None
    copy__194: "i64[]" = torch.ops.aten.copy_.default(primals_713, add_334);  primals_713 = add_334 = None
    copy__195: "f32[1024]" = torch.ops.aten.copy_.default(primals_714, add_341);  primals_714 = add_341 = None
    copy__196: "f32[1024]" = torch.ops.aten.copy_.default(primals_715, add_342);  primals_715 = add_342 = None
    copy__197: "i64[]" = torch.ops.aten.copy_.default(primals_716, add_339);  primals_716 = add_339 = None
    copy__198: "f32[256]" = torch.ops.aten.copy_.default(primals_717, add_347);  primals_717 = add_347 = None
    copy__199: "f32[256]" = torch.ops.aten.copy_.default(primals_718, add_348);  primals_718 = add_348 = None
    copy__200: "i64[]" = torch.ops.aten.copy_.default(primals_719, add_345);  primals_719 = add_345 = None
    copy__201: "f32[512]" = torch.ops.aten.copy_.default(primals_720, add_352);  primals_720 = add_352 = None
    copy__202: "f32[512]" = torch.ops.aten.copy_.default(primals_721, add_353);  primals_721 = add_353 = None
    copy__203: "i64[]" = torch.ops.aten.copy_.default(primals_722, add_350);  primals_722 = add_350 = None
    copy__204: "f32[128]" = torch.ops.aten.copy_.default(primals_723, add_357);  primals_723 = add_357 = None
    copy__205: "f32[128]" = torch.ops.aten.copy_.default(primals_724, add_358);  primals_724 = add_358 = None
    copy__206: "i64[]" = torch.ops.aten.copy_.default(primals_725, add_355);  primals_725 = add_355 = None
    copy__207: "f32[1024]" = torch.ops.aten.copy_.default(primals_726, add_362);  primals_726 = add_362 = None
    copy__208: "f32[1024]" = torch.ops.aten.copy_.default(primals_727, add_363);  primals_727 = add_363 = None
    copy__209: "i64[]" = torch.ops.aten.copy_.default(primals_728, add_360);  primals_728 = add_360 = None
    copy__210: "f32[256]" = torch.ops.aten.copy_.default(primals_729, add_368);  primals_729 = add_368 = None
    copy__211: "f32[256]" = torch.ops.aten.copy_.default(primals_730, add_369);  primals_730 = add_369 = None
    copy__212: "i64[]" = torch.ops.aten.copy_.default(primals_731, add_366);  primals_731 = add_366 = None
    copy__213: "f32[512]" = torch.ops.aten.copy_.default(primals_732, add_373);  primals_732 = add_373 = None
    copy__214: "f32[512]" = torch.ops.aten.copy_.default(primals_733, add_374);  primals_733 = add_374 = None
    copy__215: "i64[]" = torch.ops.aten.copy_.default(primals_734, add_371);  primals_734 = add_371 = None
    copy__216: "f32[128]" = torch.ops.aten.copy_.default(primals_735, add_378);  primals_735 = add_378 = None
    copy__217: "f32[128]" = torch.ops.aten.copy_.default(primals_736, add_379);  primals_736 = add_379 = None
    copy__218: "i64[]" = torch.ops.aten.copy_.default(primals_737, add_376);  primals_737 = add_376 = None
    copy__219: "f32[1024]" = torch.ops.aten.copy_.default(primals_738, add_383);  primals_738 = add_383 = None
    copy__220: "f32[1024]" = torch.ops.aten.copy_.default(primals_739, add_384);  primals_739 = add_384 = None
    copy__221: "i64[]" = torch.ops.aten.copy_.default(primals_740, add_381);  primals_740 = add_381 = None
    copy__222: "f32[256]" = torch.ops.aten.copy_.default(primals_741, add_389);  primals_741 = add_389 = None
    copy__223: "f32[256]" = torch.ops.aten.copy_.default(primals_742, add_390);  primals_742 = add_390 = None
    copy__224: "i64[]" = torch.ops.aten.copy_.default(primals_743, add_387);  primals_743 = add_387 = None
    copy__225: "f32[512]" = torch.ops.aten.copy_.default(primals_744, add_394);  primals_744 = add_394 = None
    copy__226: "f32[512]" = torch.ops.aten.copy_.default(primals_745, add_395);  primals_745 = add_395 = None
    copy__227: "i64[]" = torch.ops.aten.copy_.default(primals_746, add_392);  primals_746 = add_392 = None
    copy__228: "f32[128]" = torch.ops.aten.copy_.default(primals_747, add_399);  primals_747 = add_399 = None
    copy__229: "f32[128]" = torch.ops.aten.copy_.default(primals_748, add_400);  primals_748 = add_400 = None
    copy__230: "i64[]" = torch.ops.aten.copy_.default(primals_749, add_397);  primals_749 = add_397 = None
    copy__231: "f32[1024]" = torch.ops.aten.copy_.default(primals_750, add_404);  primals_750 = add_404 = None
    copy__232: "f32[1024]" = torch.ops.aten.copy_.default(primals_751, add_405);  primals_751 = add_405 = None
    copy__233: "i64[]" = torch.ops.aten.copy_.default(primals_752, add_402);  primals_752 = add_402 = None
    copy__234: "f32[256]" = torch.ops.aten.copy_.default(primals_753, add_410);  primals_753 = add_410 = None
    copy__235: "f32[256]" = torch.ops.aten.copy_.default(primals_754, add_411);  primals_754 = add_411 = None
    copy__236: "i64[]" = torch.ops.aten.copy_.default(primals_755, add_408);  primals_755 = add_408 = None
    copy__237: "f32[512]" = torch.ops.aten.copy_.default(primals_756, add_415);  primals_756 = add_415 = None
    copy__238: "f32[512]" = torch.ops.aten.copy_.default(primals_757, add_416);  primals_757 = add_416 = None
    copy__239: "i64[]" = torch.ops.aten.copy_.default(primals_758, add_413);  primals_758 = add_413 = None
    copy__240: "f32[128]" = torch.ops.aten.copy_.default(primals_759, add_420);  primals_759 = add_420 = None
    copy__241: "f32[128]" = torch.ops.aten.copy_.default(primals_760, add_421);  primals_760 = add_421 = None
    copy__242: "i64[]" = torch.ops.aten.copy_.default(primals_761, add_418);  primals_761 = add_418 = None
    copy__243: "f32[1024]" = torch.ops.aten.copy_.default(primals_762, add_425);  primals_762 = add_425 = None
    copy__244: "f32[1024]" = torch.ops.aten.copy_.default(primals_763, add_426);  primals_763 = add_426 = None
    copy__245: "i64[]" = torch.ops.aten.copy_.default(primals_764, add_423);  primals_764 = add_423 = None
    copy__246: "f32[256]" = torch.ops.aten.copy_.default(primals_765, add_431);  primals_765 = add_431 = None
    copy__247: "f32[256]" = torch.ops.aten.copy_.default(primals_766, add_432);  primals_766 = add_432 = None
    copy__248: "i64[]" = torch.ops.aten.copy_.default(primals_767, add_429);  primals_767 = add_429 = None
    copy__249: "f32[512]" = torch.ops.aten.copy_.default(primals_768, add_436);  primals_768 = add_436 = None
    copy__250: "f32[512]" = torch.ops.aten.copy_.default(primals_769, add_437);  primals_769 = add_437 = None
    copy__251: "i64[]" = torch.ops.aten.copy_.default(primals_770, add_434);  primals_770 = add_434 = None
    copy__252: "f32[128]" = torch.ops.aten.copy_.default(primals_771, add_441);  primals_771 = add_441 = None
    copy__253: "f32[128]" = torch.ops.aten.copy_.default(primals_772, add_442);  primals_772 = add_442 = None
    copy__254: "i64[]" = torch.ops.aten.copy_.default(primals_773, add_439);  primals_773 = add_439 = None
    copy__255: "f32[1024]" = torch.ops.aten.copy_.default(primals_774, add_446);  primals_774 = add_446 = None
    copy__256: "f32[1024]" = torch.ops.aten.copy_.default(primals_775, add_447);  primals_775 = add_447 = None
    copy__257: "i64[]" = torch.ops.aten.copy_.default(primals_776, add_444);  primals_776 = add_444 = None
    copy__258: "f32[256]" = torch.ops.aten.copy_.default(primals_777, add_452);  primals_777 = add_452 = None
    copy__259: "f32[256]" = torch.ops.aten.copy_.default(primals_778, add_453);  primals_778 = add_453 = None
    copy__260: "i64[]" = torch.ops.aten.copy_.default(primals_779, add_450);  primals_779 = add_450 = None
    copy__261: "f32[512]" = torch.ops.aten.copy_.default(primals_780, add_457);  primals_780 = add_457 = None
    copy__262: "f32[512]" = torch.ops.aten.copy_.default(primals_781, add_458);  primals_781 = add_458 = None
    copy__263: "i64[]" = torch.ops.aten.copy_.default(primals_782, add_455);  primals_782 = add_455 = None
    copy__264: "f32[128]" = torch.ops.aten.copy_.default(primals_783, add_462);  primals_783 = add_462 = None
    copy__265: "f32[128]" = torch.ops.aten.copy_.default(primals_784, add_463);  primals_784 = add_463 = None
    copy__266: "i64[]" = torch.ops.aten.copy_.default(primals_785, add_460);  primals_785 = add_460 = None
    copy__267: "f32[1024]" = torch.ops.aten.copy_.default(primals_786, add_467);  primals_786 = add_467 = None
    copy__268: "f32[1024]" = torch.ops.aten.copy_.default(primals_787, add_468);  primals_787 = add_468 = None
    copy__269: "i64[]" = torch.ops.aten.copy_.default(primals_788, add_465);  primals_788 = add_465 = None
    copy__270: "f32[256]" = torch.ops.aten.copy_.default(primals_789, add_473);  primals_789 = add_473 = None
    copy__271: "f32[256]" = torch.ops.aten.copy_.default(primals_790, add_474);  primals_790 = add_474 = None
    copy__272: "i64[]" = torch.ops.aten.copy_.default(primals_791, add_471);  primals_791 = add_471 = None
    copy__273: "f32[512]" = torch.ops.aten.copy_.default(primals_792, add_478);  primals_792 = add_478 = None
    copy__274: "f32[512]" = torch.ops.aten.copy_.default(primals_793, add_479);  primals_793 = add_479 = None
    copy__275: "i64[]" = torch.ops.aten.copy_.default(primals_794, add_476);  primals_794 = add_476 = None
    copy__276: "f32[128]" = torch.ops.aten.copy_.default(primals_795, add_483);  primals_795 = add_483 = None
    copy__277: "f32[128]" = torch.ops.aten.copy_.default(primals_796, add_484);  primals_796 = add_484 = None
    copy__278: "i64[]" = torch.ops.aten.copy_.default(primals_797, add_481);  primals_797 = add_481 = None
    copy__279: "f32[1024]" = torch.ops.aten.copy_.default(primals_798, add_488);  primals_798 = add_488 = None
    copy__280: "f32[1024]" = torch.ops.aten.copy_.default(primals_799, add_489);  primals_799 = add_489 = None
    copy__281: "i64[]" = torch.ops.aten.copy_.default(primals_800, add_486);  primals_800 = add_486 = None
    copy__282: "f32[256]" = torch.ops.aten.copy_.default(primals_801, add_494);  primals_801 = add_494 = None
    copy__283: "f32[256]" = torch.ops.aten.copy_.default(primals_802, add_495);  primals_802 = add_495 = None
    copy__284: "i64[]" = torch.ops.aten.copy_.default(primals_803, add_492);  primals_803 = add_492 = None
    copy__285: "f32[512]" = torch.ops.aten.copy_.default(primals_804, add_499);  primals_804 = add_499 = None
    copy__286: "f32[512]" = torch.ops.aten.copy_.default(primals_805, add_500);  primals_805 = add_500 = None
    copy__287: "i64[]" = torch.ops.aten.copy_.default(primals_806, add_497);  primals_806 = add_497 = None
    copy__288: "f32[128]" = torch.ops.aten.copy_.default(primals_807, add_504);  primals_807 = add_504 = None
    copy__289: "f32[128]" = torch.ops.aten.copy_.default(primals_808, add_505);  primals_808 = add_505 = None
    copy__290: "i64[]" = torch.ops.aten.copy_.default(primals_809, add_502);  primals_809 = add_502 = None
    copy__291: "f32[1024]" = torch.ops.aten.copy_.default(primals_810, add_509);  primals_810 = add_509 = None
    copy__292: "f32[1024]" = torch.ops.aten.copy_.default(primals_811, add_510);  primals_811 = add_510 = None
    copy__293: "i64[]" = torch.ops.aten.copy_.default(primals_812, add_507);  primals_812 = add_507 = None
    copy__294: "f32[256]" = torch.ops.aten.copy_.default(primals_813, add_515);  primals_813 = add_515 = None
    copy__295: "f32[256]" = torch.ops.aten.copy_.default(primals_814, add_516);  primals_814 = add_516 = None
    copy__296: "i64[]" = torch.ops.aten.copy_.default(primals_815, add_513);  primals_815 = add_513 = None
    copy__297: "f32[512]" = torch.ops.aten.copy_.default(primals_816, add_520);  primals_816 = add_520 = None
    copy__298: "f32[512]" = torch.ops.aten.copy_.default(primals_817, add_521);  primals_817 = add_521 = None
    copy__299: "i64[]" = torch.ops.aten.copy_.default(primals_818, add_518);  primals_818 = add_518 = None
    copy__300: "f32[128]" = torch.ops.aten.copy_.default(primals_819, add_525);  primals_819 = add_525 = None
    copy__301: "f32[128]" = torch.ops.aten.copy_.default(primals_820, add_526);  primals_820 = add_526 = None
    copy__302: "i64[]" = torch.ops.aten.copy_.default(primals_821, add_523);  primals_821 = add_523 = None
    copy__303: "f32[1024]" = torch.ops.aten.copy_.default(primals_822, add_530);  primals_822 = add_530 = None
    copy__304: "f32[1024]" = torch.ops.aten.copy_.default(primals_823, add_531);  primals_823 = add_531 = None
    copy__305: "i64[]" = torch.ops.aten.copy_.default(primals_824, add_528);  primals_824 = add_528 = None
    copy__306: "f32[256]" = torch.ops.aten.copy_.default(primals_825, add_536);  primals_825 = add_536 = None
    copy__307: "f32[256]" = torch.ops.aten.copy_.default(primals_826, add_537);  primals_826 = add_537 = None
    copy__308: "i64[]" = torch.ops.aten.copy_.default(primals_827, add_534);  primals_827 = add_534 = None
    copy__309: "f32[512]" = torch.ops.aten.copy_.default(primals_828, add_541);  primals_828 = add_541 = None
    copy__310: "f32[512]" = torch.ops.aten.copy_.default(primals_829, add_542);  primals_829 = add_542 = None
    copy__311: "i64[]" = torch.ops.aten.copy_.default(primals_830, add_539);  primals_830 = add_539 = None
    copy__312: "f32[128]" = torch.ops.aten.copy_.default(primals_831, add_546);  primals_831 = add_546 = None
    copy__313: "f32[128]" = torch.ops.aten.copy_.default(primals_832, add_547);  primals_832 = add_547 = None
    copy__314: "i64[]" = torch.ops.aten.copy_.default(primals_833, add_544);  primals_833 = add_544 = None
    copy__315: "f32[1024]" = torch.ops.aten.copy_.default(primals_834, add_551);  primals_834 = add_551 = None
    copy__316: "f32[1024]" = torch.ops.aten.copy_.default(primals_835, add_552);  primals_835 = add_552 = None
    copy__317: "i64[]" = torch.ops.aten.copy_.default(primals_836, add_549);  primals_836 = add_549 = None
    copy__318: "f32[256]" = torch.ops.aten.copy_.default(primals_837, add_557);  primals_837 = add_557 = None
    copy__319: "f32[256]" = torch.ops.aten.copy_.default(primals_838, add_558);  primals_838 = add_558 = None
    copy__320: "i64[]" = torch.ops.aten.copy_.default(primals_839, add_555);  primals_839 = add_555 = None
    copy__321: "f32[512]" = torch.ops.aten.copy_.default(primals_840, add_562);  primals_840 = add_562 = None
    copy__322: "f32[512]" = torch.ops.aten.copy_.default(primals_841, add_563);  primals_841 = add_563 = None
    copy__323: "i64[]" = torch.ops.aten.copy_.default(primals_842, add_560);  primals_842 = add_560 = None
    copy__324: "f32[128]" = torch.ops.aten.copy_.default(primals_843, add_567);  primals_843 = add_567 = None
    copy__325: "f32[128]" = torch.ops.aten.copy_.default(primals_844, add_568);  primals_844 = add_568 = None
    copy__326: "i64[]" = torch.ops.aten.copy_.default(primals_845, add_565);  primals_845 = add_565 = None
    copy__327: "f32[1024]" = torch.ops.aten.copy_.default(primals_846, add_572);  primals_846 = add_572 = None
    copy__328: "f32[1024]" = torch.ops.aten.copy_.default(primals_847, add_573);  primals_847 = add_573 = None
    copy__329: "i64[]" = torch.ops.aten.copy_.default(primals_848, add_570);  primals_848 = add_570 = None
    copy__330: "f32[256]" = torch.ops.aten.copy_.default(primals_849, add_578);  primals_849 = add_578 = None
    copy__331: "f32[256]" = torch.ops.aten.copy_.default(primals_850, add_579);  primals_850 = add_579 = None
    copy__332: "i64[]" = torch.ops.aten.copy_.default(primals_851, add_576);  primals_851 = add_576 = None
    copy__333: "f32[512]" = torch.ops.aten.copy_.default(primals_852, add_583);  primals_852 = add_583 = None
    copy__334: "f32[512]" = torch.ops.aten.copy_.default(primals_853, add_584);  primals_853 = add_584 = None
    copy__335: "i64[]" = torch.ops.aten.copy_.default(primals_854, add_581);  primals_854 = add_581 = None
    copy__336: "f32[128]" = torch.ops.aten.copy_.default(primals_855, add_588);  primals_855 = add_588 = None
    copy__337: "f32[128]" = torch.ops.aten.copy_.default(primals_856, add_589);  primals_856 = add_589 = None
    copy__338: "i64[]" = torch.ops.aten.copy_.default(primals_857, add_586);  primals_857 = add_586 = None
    copy__339: "f32[1024]" = torch.ops.aten.copy_.default(primals_858, add_593);  primals_858 = add_593 = None
    copy__340: "f32[1024]" = torch.ops.aten.copy_.default(primals_859, add_594);  primals_859 = add_594 = None
    copy__341: "i64[]" = torch.ops.aten.copy_.default(primals_860, add_591);  primals_860 = add_591 = None
    copy__342: "f32[256]" = torch.ops.aten.copy_.default(primals_861, add_599);  primals_861 = add_599 = None
    copy__343: "f32[256]" = torch.ops.aten.copy_.default(primals_862, add_600);  primals_862 = add_600 = None
    copy__344: "i64[]" = torch.ops.aten.copy_.default(primals_863, add_597);  primals_863 = add_597 = None
    copy__345: "f32[512]" = torch.ops.aten.copy_.default(primals_864, add_604);  primals_864 = add_604 = None
    copy__346: "f32[512]" = torch.ops.aten.copy_.default(primals_865, add_605);  primals_865 = add_605 = None
    copy__347: "i64[]" = torch.ops.aten.copy_.default(primals_866, add_602);  primals_866 = add_602 = None
    copy__348: "f32[128]" = torch.ops.aten.copy_.default(primals_867, add_609);  primals_867 = add_609 = None
    copy__349: "f32[128]" = torch.ops.aten.copy_.default(primals_868, add_610);  primals_868 = add_610 = None
    copy__350: "i64[]" = torch.ops.aten.copy_.default(primals_869, add_607);  primals_869 = add_607 = None
    copy__351: "f32[1024]" = torch.ops.aten.copy_.default(primals_870, add_614);  primals_870 = add_614 = None
    copy__352: "f32[1024]" = torch.ops.aten.copy_.default(primals_871, add_615);  primals_871 = add_615 = None
    copy__353: "i64[]" = torch.ops.aten.copy_.default(primals_872, add_612);  primals_872 = add_612 = None
    copy__354: "f32[256]" = torch.ops.aten.copy_.default(primals_873, add_620);  primals_873 = add_620 = None
    copy__355: "f32[256]" = torch.ops.aten.copy_.default(primals_874, add_621);  primals_874 = add_621 = None
    copy__356: "i64[]" = torch.ops.aten.copy_.default(primals_875, add_618);  primals_875 = add_618 = None
    copy__357: "f32[512]" = torch.ops.aten.copy_.default(primals_876, add_625);  primals_876 = add_625 = None
    copy__358: "f32[512]" = torch.ops.aten.copy_.default(primals_877, add_626);  primals_877 = add_626 = None
    copy__359: "i64[]" = torch.ops.aten.copy_.default(primals_878, add_623);  primals_878 = add_623 = None
    copy__360: "f32[128]" = torch.ops.aten.copy_.default(primals_879, add_630);  primals_879 = add_630 = None
    copy__361: "f32[128]" = torch.ops.aten.copy_.default(primals_880, add_631);  primals_880 = add_631 = None
    copy__362: "i64[]" = torch.ops.aten.copy_.default(primals_881, add_628);  primals_881 = add_628 = None
    copy__363: "f32[1024]" = torch.ops.aten.copy_.default(primals_882, add_635);  primals_882 = add_635 = None
    copy__364: "f32[1024]" = torch.ops.aten.copy_.default(primals_883, add_636);  primals_883 = add_636 = None
    copy__365: "i64[]" = torch.ops.aten.copy_.default(primals_884, add_633);  primals_884 = add_633 = None
    copy__366: "f32[256]" = torch.ops.aten.copy_.default(primals_885, add_641);  primals_885 = add_641 = None
    copy__367: "f32[256]" = torch.ops.aten.copy_.default(primals_886, add_642);  primals_886 = add_642 = None
    copy__368: "i64[]" = torch.ops.aten.copy_.default(primals_887, add_639);  primals_887 = add_639 = None
    copy__369: "f32[512]" = torch.ops.aten.copy_.default(primals_888, add_646);  primals_888 = add_646 = None
    copy__370: "f32[512]" = torch.ops.aten.copy_.default(primals_889, add_647);  primals_889 = add_647 = None
    copy__371: "i64[]" = torch.ops.aten.copy_.default(primals_890, add_644);  primals_890 = add_644 = None
    copy__372: "f32[128]" = torch.ops.aten.copy_.default(primals_891, add_651);  primals_891 = add_651 = None
    copy__373: "f32[128]" = torch.ops.aten.copy_.default(primals_892, add_652);  primals_892 = add_652 = None
    copy__374: "i64[]" = torch.ops.aten.copy_.default(primals_893, add_649);  primals_893 = add_649 = None
    copy__375: "f32[1024]" = torch.ops.aten.copy_.default(primals_894, add_656);  primals_894 = add_656 = None
    copy__376: "f32[1024]" = torch.ops.aten.copy_.default(primals_895, add_657);  primals_895 = add_657 = None
    copy__377: "i64[]" = torch.ops.aten.copy_.default(primals_896, add_654);  primals_896 = add_654 = None
    copy__378: "f32[512]" = torch.ops.aten.copy_.default(primals_897, add_662);  primals_897 = add_662 = None
    copy__379: "f32[512]" = torch.ops.aten.copy_.default(primals_898, add_663);  primals_898 = add_663 = None
    copy__380: "i64[]" = torch.ops.aten.copy_.default(primals_899, add_660);  primals_899 = add_660 = None
    copy__381: "f32[1024]" = torch.ops.aten.copy_.default(primals_900, add_667);  primals_900 = add_667 = None
    copy__382: "f32[1024]" = torch.ops.aten.copy_.default(primals_901, add_668);  primals_901 = add_668 = None
    copy__383: "i64[]" = torch.ops.aten.copy_.default(primals_902, add_665);  primals_902 = add_665 = None
    copy__384: "f32[256]" = torch.ops.aten.copy_.default(primals_903, add_672);  primals_903 = add_672 = None
    copy__385: "f32[256]" = torch.ops.aten.copy_.default(primals_904, add_673);  primals_904 = add_673 = None
    copy__386: "i64[]" = torch.ops.aten.copy_.default(primals_905, add_670);  primals_905 = add_670 = None
    copy__387: "f32[2048]" = torch.ops.aten.copy_.default(primals_906, add_677);  primals_906 = add_677 = None
    copy__388: "f32[2048]" = torch.ops.aten.copy_.default(primals_907, add_678);  primals_907 = add_678 = None
    copy__389: "i64[]" = torch.ops.aten.copy_.default(primals_908, add_675);  primals_908 = add_675 = None
    copy__390: "f32[2048]" = torch.ops.aten.copy_.default(primals_909, add_682);  primals_909 = add_682 = None
    copy__391: "f32[2048]" = torch.ops.aten.copy_.default(primals_910, add_683);  primals_910 = add_683 = None
    copy__392: "i64[]" = torch.ops.aten.copy_.default(primals_911, add_680);  primals_911 = add_680 = None
    copy__393: "f32[512]" = torch.ops.aten.copy_.default(primals_912, add_688);  primals_912 = add_688 = None
    copy__394: "f32[512]" = torch.ops.aten.copy_.default(primals_913, add_689);  primals_913 = add_689 = None
    copy__395: "i64[]" = torch.ops.aten.copy_.default(primals_914, add_686);  primals_914 = add_686 = None
    copy__396: "f32[1024]" = torch.ops.aten.copy_.default(primals_915, add_693);  primals_915 = add_693 = None
    copy__397: "f32[1024]" = torch.ops.aten.copy_.default(primals_916, add_694);  primals_916 = add_694 = None
    copy__398: "i64[]" = torch.ops.aten.copy_.default(primals_917, add_691);  primals_917 = add_691 = None
    copy__399: "f32[256]" = torch.ops.aten.copy_.default(primals_918, add_698);  primals_918 = add_698 = None
    copy__400: "f32[256]" = torch.ops.aten.copy_.default(primals_919, add_699);  primals_919 = add_699 = None
    copy__401: "i64[]" = torch.ops.aten.copy_.default(primals_920, add_696);  primals_920 = add_696 = None
    copy__402: "f32[2048]" = torch.ops.aten.copy_.default(primals_921, add_703);  primals_921 = add_703 = None
    copy__403: "f32[2048]" = torch.ops.aten.copy_.default(primals_922, add_704);  primals_922 = add_704 = None
    copy__404: "i64[]" = torch.ops.aten.copy_.default(primals_923, add_701);  primals_923 = add_701 = None
    copy__405: "f32[512]" = torch.ops.aten.copy_.default(primals_924, add_709);  primals_924 = add_709 = None
    copy__406: "f32[512]" = torch.ops.aten.copy_.default(primals_925, add_710);  primals_925 = add_710 = None
    copy__407: "i64[]" = torch.ops.aten.copy_.default(primals_926, add_707);  primals_926 = add_707 = None
    copy__408: "f32[1024]" = torch.ops.aten.copy_.default(primals_927, add_714);  primals_927 = add_714 = None
    copy__409: "f32[1024]" = torch.ops.aten.copy_.default(primals_928, add_715);  primals_928 = add_715 = None
    copy__410: "i64[]" = torch.ops.aten.copy_.default(primals_929, add_712);  primals_929 = add_712 = None
    copy__411: "f32[256]" = torch.ops.aten.copy_.default(primals_930, add_719);  primals_930 = add_719 = None
    copy__412: "f32[256]" = torch.ops.aten.copy_.default(primals_931, add_720);  primals_931 = add_720 = None
    copy__413: "i64[]" = torch.ops.aten.copy_.default(primals_932, add_717);  primals_932 = add_717 = None
    copy__414: "f32[2048]" = torch.ops.aten.copy_.default(primals_933, add_724);  primals_933 = add_724 = None
    copy__415: "f32[2048]" = torch.ops.aten.copy_.default(primals_934, add_725);  primals_934 = add_725 = None
    copy__416: "i64[]" = torch.ops.aten.copy_.default(primals_935, add_722);  primals_935 = add_722 = None
    return [addmm, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_18, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_51, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_66, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_84, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_99, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_114, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_129, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_147, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_162, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_177, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_192, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_207, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_222, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_237, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_252, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_267, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_282, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_297, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_312, primals_314, primals_316, primals_317, primals_319, primals_320, primals_322, primals_323, primals_325, primals_327, primals_329, primals_331, primals_332, primals_334, primals_335, primals_337, primals_338, primals_340, primals_342, primals_344, primals_346, primals_347, primals_349, primals_350, primals_352, primals_353, primals_355, primals_357, primals_359, primals_361, primals_362, primals_364, primals_365, primals_367, primals_368, primals_370, primals_372, primals_374, primals_376, primals_377, primals_379, primals_380, primals_382, primals_383, primals_385, primals_387, primals_389, primals_391, primals_392, primals_394, primals_395, primals_397, primals_398, primals_400, primals_402, primals_404, primals_406, primals_407, primals_409, primals_410, primals_412, primals_413, primals_415, primals_417, primals_419, primals_421, primals_422, primals_424, primals_425, primals_427, primals_428, primals_430, primals_432, primals_434, primals_436, primals_437, primals_439, primals_440, primals_442, primals_443, primals_445, primals_447, primals_449, primals_451, primals_452, primals_454, primals_455, primals_457, primals_458, primals_460, primals_462, primals_464, primals_466, primals_467, primals_469, primals_470, primals_472, primals_473, primals_475, primals_477, primals_479, primals_481, primals_482, primals_484, primals_485, primals_487, primals_488, primals_490, primals_491, primals_493, primals_495, primals_497, primals_499, primals_500, primals_502, primals_503, primals_505, primals_506, primals_508, primals_510, primals_512, primals_514, primals_515, primals_936, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, getitem_6, getitem_7, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, mean, convolution_5, relu_5, div, sum_3, convolution_7, squeeze_19, convolution_8, squeeze_22, relu_6, convolution_9, squeeze_25, relu_7, convolution_10, squeeze_28, relu_8, mean_1, convolution_11, relu_9, div_1, sum_6, convolution_13, squeeze_34, relu_10, convolution_14, squeeze_37, relu_11, convolution_15, squeeze_40, relu_12, mean_2, convolution_16, relu_13, div_2, sum_9, convolution_18, squeeze_46, relu_14, convolution_19, squeeze_49, relu_15, convolution_20, squeeze_52, relu_16, mean_3, convolution_21, relu_17, div_3, sum_12, avg_pool2d, convolution_23, squeeze_58, avg_pool2d_1, convolution_24, squeeze_61, relu_18, convolution_25, squeeze_64, relu_19, convolution_26, squeeze_67, relu_20, mean_4, convolution_27, relu_21, div_4, sum_15, convolution_29, squeeze_73, relu_22, convolution_30, squeeze_76, relu_23, convolution_31, squeeze_79, relu_24, mean_5, convolution_32, relu_25, div_5, sum_18, convolution_34, squeeze_85, relu_26, convolution_35, squeeze_88, relu_27, convolution_36, squeeze_91, relu_28, mean_6, convolution_37, relu_29, div_6, sum_21, convolution_39, squeeze_97, relu_30, convolution_40, squeeze_100, relu_31, convolution_41, squeeze_103, relu_32, mean_7, convolution_42, relu_33, div_7, sum_24, avg_pool2d_2, convolution_44, squeeze_109, avg_pool2d_3, convolution_45, squeeze_112, relu_34, convolution_46, squeeze_115, relu_35, convolution_47, squeeze_118, relu_36, mean_8, convolution_48, relu_37, div_8, sum_27, convolution_50, squeeze_124, relu_38, convolution_51, squeeze_127, relu_39, convolution_52, squeeze_130, relu_40, mean_9, convolution_53, relu_41, div_9, sum_30, convolution_55, squeeze_136, relu_42, convolution_56, squeeze_139, relu_43, convolution_57, squeeze_142, relu_44, mean_10, convolution_58, relu_45, div_10, sum_33, convolution_60, squeeze_148, relu_46, convolution_61, squeeze_151, relu_47, convolution_62, squeeze_154, relu_48, mean_11, convolution_63, relu_49, div_11, sum_36, convolution_65, squeeze_160, relu_50, convolution_66, squeeze_163, relu_51, convolution_67, squeeze_166, relu_52, mean_12, convolution_68, relu_53, div_12, sum_39, convolution_70, squeeze_172, relu_54, convolution_71, squeeze_175, relu_55, convolution_72, squeeze_178, relu_56, mean_13, convolution_73, relu_57, div_13, sum_42, convolution_75, squeeze_184, relu_58, convolution_76, squeeze_187, relu_59, convolution_77, squeeze_190, relu_60, mean_14, convolution_78, relu_61, div_14, sum_45, convolution_80, squeeze_196, relu_62, convolution_81, squeeze_199, relu_63, convolution_82, squeeze_202, relu_64, mean_15, convolution_83, relu_65, div_15, sum_48, convolution_85, squeeze_208, relu_66, convolution_86, squeeze_211, relu_67, convolution_87, squeeze_214, relu_68, mean_16, convolution_88, relu_69, div_16, sum_51, convolution_90, squeeze_220, relu_70, convolution_91, squeeze_223, relu_71, convolution_92, squeeze_226, relu_72, mean_17, convolution_93, relu_73, div_17, sum_54, convolution_95, squeeze_232, relu_74, convolution_96, squeeze_235, relu_75, convolution_97, squeeze_238, relu_76, mean_18, convolution_98, relu_77, div_18, sum_57, convolution_100, squeeze_244, relu_78, convolution_101, squeeze_247, relu_79, convolution_102, squeeze_250, relu_80, mean_19, convolution_103, relu_81, div_19, sum_60, convolution_105, squeeze_256, relu_82, convolution_106, squeeze_259, relu_83, convolution_107, squeeze_262, relu_84, mean_20, convolution_108, relu_85, div_20, sum_63, convolution_110, squeeze_268, relu_86, convolution_111, squeeze_271, relu_87, convolution_112, squeeze_274, relu_88, mean_21, convolution_113, relu_89, div_21, sum_66, convolution_115, squeeze_280, relu_90, convolution_116, squeeze_283, relu_91, convolution_117, squeeze_286, relu_92, mean_22, convolution_118, relu_93, div_22, sum_69, convolution_120, squeeze_292, relu_94, convolution_121, squeeze_295, relu_95, convolution_122, squeeze_298, relu_96, mean_23, convolution_123, relu_97, div_23, sum_72, convolution_125, squeeze_304, relu_98, convolution_126, squeeze_307, relu_99, convolution_127, squeeze_310, relu_100, mean_24, convolution_128, relu_101, div_24, sum_75, convolution_130, squeeze_316, relu_102, convolution_131, squeeze_319, relu_103, convolution_132, squeeze_322, relu_104, mean_25, convolution_133, relu_105, div_25, sum_78, convolution_135, squeeze_328, relu_106, convolution_136, squeeze_331, relu_107, convolution_137, squeeze_334, relu_108, mean_26, convolution_138, relu_109, div_26, sum_81, convolution_140, squeeze_340, relu_110, convolution_141, squeeze_343, relu_111, convolution_142, squeeze_346, relu_112, mean_27, convolution_143, relu_113, div_27, sum_84, convolution_145, squeeze_352, relu_114, convolution_146, squeeze_355, relu_115, convolution_147, squeeze_358, relu_116, mean_28, convolution_148, relu_117, div_28, sum_87, convolution_150, squeeze_364, relu_118, convolution_151, squeeze_367, relu_119, convolution_152, squeeze_370, relu_120, mean_29, convolution_153, relu_121, div_29, sum_90, convolution_155, squeeze_376, relu_122, convolution_156, squeeze_379, relu_123, convolution_157, squeeze_382, relu_124, mean_30, convolution_158, relu_125, div_30, sum_93, avg_pool2d_4, convolution_160, squeeze_388, avg_pool2d_5, convolution_161, squeeze_391, relu_126, convolution_162, squeeze_394, relu_127, convolution_163, squeeze_397, relu_128, mean_31, convolution_164, relu_129, div_31, sum_96, convolution_166, squeeze_403, relu_130, convolution_167, squeeze_406, relu_131, convolution_168, squeeze_409, relu_132, mean_32, convolution_169, relu_133, div_32, sum_99, convolution_171, squeeze_415, view_198, permute_34, le, unsqueeze_558, unsqueeze_584, unsqueeze_596, unsqueeze_608, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_696, unsqueeze_708, unsqueeze_720, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_796, unsqueeze_808, unsqueeze_820, unsqueeze_846, unsqueeze_858, unsqueeze_870, unsqueeze_896, unsqueeze_908, unsqueeze_920, unsqueeze_946, unsqueeze_958, unsqueeze_970, unsqueeze_996, unsqueeze_1008, unsqueeze_1020, unsqueeze_1046, unsqueeze_1058, unsqueeze_1070, unsqueeze_1096, unsqueeze_1108, unsqueeze_1120, unsqueeze_1146, unsqueeze_1158, unsqueeze_1170, unsqueeze_1196, unsqueeze_1208, unsqueeze_1220, unsqueeze_1246, unsqueeze_1258, unsqueeze_1270, unsqueeze_1296, unsqueeze_1308, unsqueeze_1320, unsqueeze_1346, unsqueeze_1358, unsqueeze_1370, unsqueeze_1396, unsqueeze_1408, unsqueeze_1420, unsqueeze_1446, unsqueeze_1458, unsqueeze_1470, unsqueeze_1496, unsqueeze_1508, unsqueeze_1520, unsqueeze_1546, unsqueeze_1558, unsqueeze_1570, unsqueeze_1596, unsqueeze_1608, unsqueeze_1620, unsqueeze_1646, unsqueeze_1658, unsqueeze_1670, unsqueeze_1696, unsqueeze_1708, unsqueeze_1720, unsqueeze_1746, unsqueeze_1758, unsqueeze_1770, unsqueeze_1796, unsqueeze_1808, unsqueeze_1820, unsqueeze_1832, unsqueeze_1858, unsqueeze_1870, unsqueeze_1882, unsqueeze_1908, unsqueeze_1920, unsqueeze_1932, unsqueeze_1958, unsqueeze_1970, unsqueeze_1982, unsqueeze_2008, unsqueeze_2020, unsqueeze_2032, unsqueeze_2044, unsqueeze_2070, unsqueeze_2082, unsqueeze_2094, unsqueeze_2120, unsqueeze_2132, unsqueeze_2144, unsqueeze_2170, unsqueeze_2182, unsqueeze_2194, unsqueeze_2206, unsqueeze_2232, unsqueeze_2244, unsqueeze_2256, unsqueeze_2268, unsqueeze_2280]
    