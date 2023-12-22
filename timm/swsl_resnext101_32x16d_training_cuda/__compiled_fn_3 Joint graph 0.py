from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[64, 3, 7, 7]"; primals_2: "f32[64]"; primals_3: "f32[64]"; primals_4: "f32[512, 64, 1, 1]"; primals_5: "f32[512]"; primals_6: "f32[512]"; primals_7: "f32[512, 16, 3, 3]"; primals_8: "f32[512]"; primals_9: "f32[512]"; primals_10: "f32[256, 512, 1, 1]"; primals_11: "f32[256]"; primals_12: "f32[256]"; primals_13: "f32[256, 64, 1, 1]"; primals_14: "f32[256]"; primals_15: "f32[256]"; primals_16: "f32[512, 256, 1, 1]"; primals_17: "f32[512]"; primals_18: "f32[512]"; primals_19: "f32[512, 16, 3, 3]"; primals_20: "f32[512]"; primals_21: "f32[512]"; primals_22: "f32[256, 512, 1, 1]"; primals_23: "f32[256]"; primals_24: "f32[256]"; primals_25: "f32[512, 256, 1, 1]"; primals_26: "f32[512]"; primals_27: "f32[512]"; primals_28: "f32[512, 16, 3, 3]"; primals_29: "f32[512]"; primals_30: "f32[512]"; primals_31: "f32[256, 512, 1, 1]"; primals_32: "f32[256]"; primals_33: "f32[256]"; primals_34: "f32[1024, 256, 1, 1]"; primals_35: "f32[1024]"; primals_36: "f32[1024]"; primals_37: "f32[1024, 32, 3, 3]"; primals_38: "f32[1024]"; primals_39: "f32[1024]"; primals_40: "f32[512, 1024, 1, 1]"; primals_41: "f32[512]"; primals_42: "f32[512]"; primals_43: "f32[512, 256, 1, 1]"; primals_44: "f32[512]"; primals_45: "f32[512]"; primals_46: "f32[1024, 512, 1, 1]"; primals_47: "f32[1024]"; primals_48: "f32[1024]"; primals_49: "f32[1024, 32, 3, 3]"; primals_50: "f32[1024]"; primals_51: "f32[1024]"; primals_52: "f32[512, 1024, 1, 1]"; primals_53: "f32[512]"; primals_54: "f32[512]"; primals_55: "f32[1024, 512, 1, 1]"; primals_56: "f32[1024]"; primals_57: "f32[1024]"; primals_58: "f32[1024, 32, 3, 3]"; primals_59: "f32[1024]"; primals_60: "f32[1024]"; primals_61: "f32[512, 1024, 1, 1]"; primals_62: "f32[512]"; primals_63: "f32[512]"; primals_64: "f32[1024, 512, 1, 1]"; primals_65: "f32[1024]"; primals_66: "f32[1024]"; primals_67: "f32[1024, 32, 3, 3]"; primals_68: "f32[1024]"; primals_69: "f32[1024]"; primals_70: "f32[512, 1024, 1, 1]"; primals_71: "f32[512]"; primals_72: "f32[512]"; primals_73: "f32[2048, 512, 1, 1]"; primals_74: "f32[2048]"; primals_75: "f32[2048]"; primals_76: "f32[2048, 64, 3, 3]"; primals_77: "f32[2048]"; primals_78: "f32[2048]"; primals_79: "f32[1024, 2048, 1, 1]"; primals_80: "f32[1024]"; primals_81: "f32[1024]"; primals_82: "f32[1024, 512, 1, 1]"; primals_83: "f32[1024]"; primals_84: "f32[1024]"; primals_85: "f32[2048, 1024, 1, 1]"; primals_86: "f32[2048]"; primals_87: "f32[2048]"; primals_88: "f32[2048, 64, 3, 3]"; primals_89: "f32[2048]"; primals_90: "f32[2048]"; primals_91: "f32[1024, 2048, 1, 1]"; primals_92: "f32[1024]"; primals_93: "f32[1024]"; primals_94: "f32[2048, 1024, 1, 1]"; primals_95: "f32[2048]"; primals_96: "f32[2048]"; primals_97: "f32[2048, 64, 3, 3]"; primals_98: "f32[2048]"; primals_99: "f32[2048]"; primals_100: "f32[1024, 2048, 1, 1]"; primals_101: "f32[1024]"; primals_102: "f32[1024]"; primals_103: "f32[2048, 1024, 1, 1]"; primals_104: "f32[2048]"; primals_105: "f32[2048]"; primals_106: "f32[2048, 64, 3, 3]"; primals_107: "f32[2048]"; primals_108: "f32[2048]"; primals_109: "f32[1024, 2048, 1, 1]"; primals_110: "f32[1024]"; primals_111: "f32[1024]"; primals_112: "f32[2048, 1024, 1, 1]"; primals_113: "f32[2048]"; primals_114: "f32[2048]"; primals_115: "f32[2048, 64, 3, 3]"; primals_116: "f32[2048]"; primals_117: "f32[2048]"; primals_118: "f32[1024, 2048, 1, 1]"; primals_119: "f32[1024]"; primals_120: "f32[1024]"; primals_121: "f32[2048, 1024, 1, 1]"; primals_122: "f32[2048]"; primals_123: "f32[2048]"; primals_124: "f32[2048, 64, 3, 3]"; primals_125: "f32[2048]"; primals_126: "f32[2048]"; primals_127: "f32[1024, 2048, 1, 1]"; primals_128: "f32[1024]"; primals_129: "f32[1024]"; primals_130: "f32[2048, 1024, 1, 1]"; primals_131: "f32[2048]"; primals_132: "f32[2048]"; primals_133: "f32[2048, 64, 3, 3]"; primals_134: "f32[2048]"; primals_135: "f32[2048]"; primals_136: "f32[1024, 2048, 1, 1]"; primals_137: "f32[1024]"; primals_138: "f32[1024]"; primals_139: "f32[2048, 1024, 1, 1]"; primals_140: "f32[2048]"; primals_141: "f32[2048]"; primals_142: "f32[2048, 64, 3, 3]"; primals_143: "f32[2048]"; primals_144: "f32[2048]"; primals_145: "f32[1024, 2048, 1, 1]"; primals_146: "f32[1024]"; primals_147: "f32[1024]"; primals_148: "f32[2048, 1024, 1, 1]"; primals_149: "f32[2048]"; primals_150: "f32[2048]"; primals_151: "f32[2048, 64, 3, 3]"; primals_152: "f32[2048]"; primals_153: "f32[2048]"; primals_154: "f32[1024, 2048, 1, 1]"; primals_155: "f32[1024]"; primals_156: "f32[1024]"; primals_157: "f32[2048, 1024, 1, 1]"; primals_158: "f32[2048]"; primals_159: "f32[2048]"; primals_160: "f32[2048, 64, 3, 3]"; primals_161: "f32[2048]"; primals_162: "f32[2048]"; primals_163: "f32[1024, 2048, 1, 1]"; primals_164: "f32[1024]"; primals_165: "f32[1024]"; primals_166: "f32[2048, 1024, 1, 1]"; primals_167: "f32[2048]"; primals_168: "f32[2048]"; primals_169: "f32[2048, 64, 3, 3]"; primals_170: "f32[2048]"; primals_171: "f32[2048]"; primals_172: "f32[1024, 2048, 1, 1]"; primals_173: "f32[1024]"; primals_174: "f32[1024]"; primals_175: "f32[2048, 1024, 1, 1]"; primals_176: "f32[2048]"; primals_177: "f32[2048]"; primals_178: "f32[2048, 64, 3, 3]"; primals_179: "f32[2048]"; primals_180: "f32[2048]"; primals_181: "f32[1024, 2048, 1, 1]"; primals_182: "f32[1024]"; primals_183: "f32[1024]"; primals_184: "f32[2048, 1024, 1, 1]"; primals_185: "f32[2048]"; primals_186: "f32[2048]"; primals_187: "f32[2048, 64, 3, 3]"; primals_188: "f32[2048]"; primals_189: "f32[2048]"; primals_190: "f32[1024, 2048, 1, 1]"; primals_191: "f32[1024]"; primals_192: "f32[1024]"; primals_193: "f32[2048, 1024, 1, 1]"; primals_194: "f32[2048]"; primals_195: "f32[2048]"; primals_196: "f32[2048, 64, 3, 3]"; primals_197: "f32[2048]"; primals_198: "f32[2048]"; primals_199: "f32[1024, 2048, 1, 1]"; primals_200: "f32[1024]"; primals_201: "f32[1024]"; primals_202: "f32[2048, 1024, 1, 1]"; primals_203: "f32[2048]"; primals_204: "f32[2048]"; primals_205: "f32[2048, 64, 3, 3]"; primals_206: "f32[2048]"; primals_207: "f32[2048]"; primals_208: "f32[1024, 2048, 1, 1]"; primals_209: "f32[1024]"; primals_210: "f32[1024]"; primals_211: "f32[2048, 1024, 1, 1]"; primals_212: "f32[2048]"; primals_213: "f32[2048]"; primals_214: "f32[2048, 64, 3, 3]"; primals_215: "f32[2048]"; primals_216: "f32[2048]"; primals_217: "f32[1024, 2048, 1, 1]"; primals_218: "f32[1024]"; primals_219: "f32[1024]"; primals_220: "f32[2048, 1024, 1, 1]"; primals_221: "f32[2048]"; primals_222: "f32[2048]"; primals_223: "f32[2048, 64, 3, 3]"; primals_224: "f32[2048]"; primals_225: "f32[2048]"; primals_226: "f32[1024, 2048, 1, 1]"; primals_227: "f32[1024]"; primals_228: "f32[1024]"; primals_229: "f32[2048, 1024, 1, 1]"; primals_230: "f32[2048]"; primals_231: "f32[2048]"; primals_232: "f32[2048, 64, 3, 3]"; primals_233: "f32[2048]"; primals_234: "f32[2048]"; primals_235: "f32[1024, 2048, 1, 1]"; primals_236: "f32[1024]"; primals_237: "f32[1024]"; primals_238: "f32[2048, 1024, 1, 1]"; primals_239: "f32[2048]"; primals_240: "f32[2048]"; primals_241: "f32[2048, 64, 3, 3]"; primals_242: "f32[2048]"; primals_243: "f32[2048]"; primals_244: "f32[1024, 2048, 1, 1]"; primals_245: "f32[1024]"; primals_246: "f32[1024]"; primals_247: "f32[2048, 1024, 1, 1]"; primals_248: "f32[2048]"; primals_249: "f32[2048]"; primals_250: "f32[2048, 64, 3, 3]"; primals_251: "f32[2048]"; primals_252: "f32[2048]"; primals_253: "f32[1024, 2048, 1, 1]"; primals_254: "f32[1024]"; primals_255: "f32[1024]"; primals_256: "f32[2048, 1024, 1, 1]"; primals_257: "f32[2048]"; primals_258: "f32[2048]"; primals_259: "f32[2048, 64, 3, 3]"; primals_260: "f32[2048]"; primals_261: "f32[2048]"; primals_262: "f32[1024, 2048, 1, 1]"; primals_263: "f32[1024]"; primals_264: "f32[1024]"; primals_265: "f32[2048, 1024, 1, 1]"; primals_266: "f32[2048]"; primals_267: "f32[2048]"; primals_268: "f32[2048, 64, 3, 3]"; primals_269: "f32[2048]"; primals_270: "f32[2048]"; primals_271: "f32[1024, 2048, 1, 1]"; primals_272: "f32[1024]"; primals_273: "f32[1024]"; primals_274: "f32[2048, 1024, 1, 1]"; primals_275: "f32[2048]"; primals_276: "f32[2048]"; primals_277: "f32[2048, 64, 3, 3]"; primals_278: "f32[2048]"; primals_279: "f32[2048]"; primals_280: "f32[1024, 2048, 1, 1]"; primals_281: "f32[1024]"; primals_282: "f32[1024]"; primals_283: "f32[4096, 1024, 1, 1]"; primals_284: "f32[4096]"; primals_285: "f32[4096]"; primals_286: "f32[4096, 128, 3, 3]"; primals_287: "f32[4096]"; primals_288: "f32[4096]"; primals_289: "f32[2048, 4096, 1, 1]"; primals_290: "f32[2048]"; primals_291: "f32[2048]"; primals_292: "f32[2048, 1024, 1, 1]"; primals_293: "f32[2048]"; primals_294: "f32[2048]"; primals_295: "f32[4096, 2048, 1, 1]"; primals_296: "f32[4096]"; primals_297: "f32[4096]"; primals_298: "f32[4096, 128, 3, 3]"; primals_299: "f32[4096]"; primals_300: "f32[4096]"; primals_301: "f32[2048, 4096, 1, 1]"; primals_302: "f32[2048]"; primals_303: "f32[2048]"; primals_304: "f32[4096, 2048, 1, 1]"; primals_305: "f32[4096]"; primals_306: "f32[4096]"; primals_307: "f32[4096, 128, 3, 3]"; primals_308: "f32[4096]"; primals_309: "f32[4096]"; primals_310: "f32[2048, 4096, 1, 1]"; primals_311: "f32[2048]"; primals_312: "f32[2048]"; primals_313: "f32[1000, 2048]"; primals_314: "f32[1000]"; primals_315: "f32[64]"; primals_316: "f32[64]"; primals_317: "i64[]"; primals_318: "f32[512]"; primals_319: "f32[512]"; primals_320: "i64[]"; primals_321: "f32[512]"; primals_322: "f32[512]"; primals_323: "i64[]"; primals_324: "f32[256]"; primals_325: "f32[256]"; primals_326: "i64[]"; primals_327: "f32[256]"; primals_328: "f32[256]"; primals_329: "i64[]"; primals_330: "f32[512]"; primals_331: "f32[512]"; primals_332: "i64[]"; primals_333: "f32[512]"; primals_334: "f32[512]"; primals_335: "i64[]"; primals_336: "f32[256]"; primals_337: "f32[256]"; primals_338: "i64[]"; primals_339: "f32[512]"; primals_340: "f32[512]"; primals_341: "i64[]"; primals_342: "f32[512]"; primals_343: "f32[512]"; primals_344: "i64[]"; primals_345: "f32[256]"; primals_346: "f32[256]"; primals_347: "i64[]"; primals_348: "f32[1024]"; primals_349: "f32[1024]"; primals_350: "i64[]"; primals_351: "f32[1024]"; primals_352: "f32[1024]"; primals_353: "i64[]"; primals_354: "f32[512]"; primals_355: "f32[512]"; primals_356: "i64[]"; primals_357: "f32[512]"; primals_358: "f32[512]"; primals_359: "i64[]"; primals_360: "f32[1024]"; primals_361: "f32[1024]"; primals_362: "i64[]"; primals_363: "f32[1024]"; primals_364: "f32[1024]"; primals_365: "i64[]"; primals_366: "f32[512]"; primals_367: "f32[512]"; primals_368: "i64[]"; primals_369: "f32[1024]"; primals_370: "f32[1024]"; primals_371: "i64[]"; primals_372: "f32[1024]"; primals_373: "f32[1024]"; primals_374: "i64[]"; primals_375: "f32[512]"; primals_376: "f32[512]"; primals_377: "i64[]"; primals_378: "f32[1024]"; primals_379: "f32[1024]"; primals_380: "i64[]"; primals_381: "f32[1024]"; primals_382: "f32[1024]"; primals_383: "i64[]"; primals_384: "f32[512]"; primals_385: "f32[512]"; primals_386: "i64[]"; primals_387: "f32[2048]"; primals_388: "f32[2048]"; primals_389: "i64[]"; primals_390: "f32[2048]"; primals_391: "f32[2048]"; primals_392: "i64[]"; primals_393: "f32[1024]"; primals_394: "f32[1024]"; primals_395: "i64[]"; primals_396: "f32[1024]"; primals_397: "f32[1024]"; primals_398: "i64[]"; primals_399: "f32[2048]"; primals_400: "f32[2048]"; primals_401: "i64[]"; primals_402: "f32[2048]"; primals_403: "f32[2048]"; primals_404: "i64[]"; primals_405: "f32[1024]"; primals_406: "f32[1024]"; primals_407: "i64[]"; primals_408: "f32[2048]"; primals_409: "f32[2048]"; primals_410: "i64[]"; primals_411: "f32[2048]"; primals_412: "f32[2048]"; primals_413: "i64[]"; primals_414: "f32[1024]"; primals_415: "f32[1024]"; primals_416: "i64[]"; primals_417: "f32[2048]"; primals_418: "f32[2048]"; primals_419: "i64[]"; primals_420: "f32[2048]"; primals_421: "f32[2048]"; primals_422: "i64[]"; primals_423: "f32[1024]"; primals_424: "f32[1024]"; primals_425: "i64[]"; primals_426: "f32[2048]"; primals_427: "f32[2048]"; primals_428: "i64[]"; primals_429: "f32[2048]"; primals_430: "f32[2048]"; primals_431: "i64[]"; primals_432: "f32[1024]"; primals_433: "f32[1024]"; primals_434: "i64[]"; primals_435: "f32[2048]"; primals_436: "f32[2048]"; primals_437: "i64[]"; primals_438: "f32[2048]"; primals_439: "f32[2048]"; primals_440: "i64[]"; primals_441: "f32[1024]"; primals_442: "f32[1024]"; primals_443: "i64[]"; primals_444: "f32[2048]"; primals_445: "f32[2048]"; primals_446: "i64[]"; primals_447: "f32[2048]"; primals_448: "f32[2048]"; primals_449: "i64[]"; primals_450: "f32[1024]"; primals_451: "f32[1024]"; primals_452: "i64[]"; primals_453: "f32[2048]"; primals_454: "f32[2048]"; primals_455: "i64[]"; primals_456: "f32[2048]"; primals_457: "f32[2048]"; primals_458: "i64[]"; primals_459: "f32[1024]"; primals_460: "f32[1024]"; primals_461: "i64[]"; primals_462: "f32[2048]"; primals_463: "f32[2048]"; primals_464: "i64[]"; primals_465: "f32[2048]"; primals_466: "f32[2048]"; primals_467: "i64[]"; primals_468: "f32[1024]"; primals_469: "f32[1024]"; primals_470: "i64[]"; primals_471: "f32[2048]"; primals_472: "f32[2048]"; primals_473: "i64[]"; primals_474: "f32[2048]"; primals_475: "f32[2048]"; primals_476: "i64[]"; primals_477: "f32[1024]"; primals_478: "f32[1024]"; primals_479: "i64[]"; primals_480: "f32[2048]"; primals_481: "f32[2048]"; primals_482: "i64[]"; primals_483: "f32[2048]"; primals_484: "f32[2048]"; primals_485: "i64[]"; primals_486: "f32[1024]"; primals_487: "f32[1024]"; primals_488: "i64[]"; primals_489: "f32[2048]"; primals_490: "f32[2048]"; primals_491: "i64[]"; primals_492: "f32[2048]"; primals_493: "f32[2048]"; primals_494: "i64[]"; primals_495: "f32[1024]"; primals_496: "f32[1024]"; primals_497: "i64[]"; primals_498: "f32[2048]"; primals_499: "f32[2048]"; primals_500: "i64[]"; primals_501: "f32[2048]"; primals_502: "f32[2048]"; primals_503: "i64[]"; primals_504: "f32[1024]"; primals_505: "f32[1024]"; primals_506: "i64[]"; primals_507: "f32[2048]"; primals_508: "f32[2048]"; primals_509: "i64[]"; primals_510: "f32[2048]"; primals_511: "f32[2048]"; primals_512: "i64[]"; primals_513: "f32[1024]"; primals_514: "f32[1024]"; primals_515: "i64[]"; primals_516: "f32[2048]"; primals_517: "f32[2048]"; primals_518: "i64[]"; primals_519: "f32[2048]"; primals_520: "f32[2048]"; primals_521: "i64[]"; primals_522: "f32[1024]"; primals_523: "f32[1024]"; primals_524: "i64[]"; primals_525: "f32[2048]"; primals_526: "f32[2048]"; primals_527: "i64[]"; primals_528: "f32[2048]"; primals_529: "f32[2048]"; primals_530: "i64[]"; primals_531: "f32[1024]"; primals_532: "f32[1024]"; primals_533: "i64[]"; primals_534: "f32[2048]"; primals_535: "f32[2048]"; primals_536: "i64[]"; primals_537: "f32[2048]"; primals_538: "f32[2048]"; primals_539: "i64[]"; primals_540: "f32[1024]"; primals_541: "f32[1024]"; primals_542: "i64[]"; primals_543: "f32[2048]"; primals_544: "f32[2048]"; primals_545: "i64[]"; primals_546: "f32[2048]"; primals_547: "f32[2048]"; primals_548: "i64[]"; primals_549: "f32[1024]"; primals_550: "f32[1024]"; primals_551: "i64[]"; primals_552: "f32[2048]"; primals_553: "f32[2048]"; primals_554: "i64[]"; primals_555: "f32[2048]"; primals_556: "f32[2048]"; primals_557: "i64[]"; primals_558: "f32[1024]"; primals_559: "f32[1024]"; primals_560: "i64[]"; primals_561: "f32[2048]"; primals_562: "f32[2048]"; primals_563: "i64[]"; primals_564: "f32[2048]"; primals_565: "f32[2048]"; primals_566: "i64[]"; primals_567: "f32[1024]"; primals_568: "f32[1024]"; primals_569: "i64[]"; primals_570: "f32[2048]"; primals_571: "f32[2048]"; primals_572: "i64[]"; primals_573: "f32[2048]"; primals_574: "f32[2048]"; primals_575: "i64[]"; primals_576: "f32[1024]"; primals_577: "f32[1024]"; primals_578: "i64[]"; primals_579: "f32[2048]"; primals_580: "f32[2048]"; primals_581: "i64[]"; primals_582: "f32[2048]"; primals_583: "f32[2048]"; primals_584: "i64[]"; primals_585: "f32[1024]"; primals_586: "f32[1024]"; primals_587: "i64[]"; primals_588: "f32[2048]"; primals_589: "f32[2048]"; primals_590: "i64[]"; primals_591: "f32[2048]"; primals_592: "f32[2048]"; primals_593: "i64[]"; primals_594: "f32[1024]"; primals_595: "f32[1024]"; primals_596: "i64[]"; primals_597: "f32[4096]"; primals_598: "f32[4096]"; primals_599: "i64[]"; primals_600: "f32[4096]"; primals_601: "f32[4096]"; primals_602: "i64[]"; primals_603: "f32[2048]"; primals_604: "f32[2048]"; primals_605: "i64[]"; primals_606: "f32[2048]"; primals_607: "f32[2048]"; primals_608: "i64[]"; primals_609: "f32[4096]"; primals_610: "f32[4096]"; primals_611: "i64[]"; primals_612: "f32[4096]"; primals_613: "f32[4096]"; primals_614: "i64[]"; primals_615: "f32[2048]"; primals_616: "f32[2048]"; primals_617: "i64[]"; primals_618: "f32[4096]"; primals_619: "f32[4096]"; primals_620: "i64[]"; primals_621: "f32[4096]"; primals_622: "f32[4096]"; primals_623: "i64[]"; primals_624: "f32[2048]"; primals_625: "f32[2048]"; primals_626: "i64[]"; primals_627: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(primals_627, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_317, 1)
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
    mul_2: "f32[64]" = torch.ops.aten.mul.Tensor(primals_315, 0.9)
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[64]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_1: "f32[8, 512, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, primals_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_320, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 512, 1, 1]" = var_mean_1[0]
    getitem_5: "f32[1, 512, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_1: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_5)
    mul_7: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_4: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[512]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
    add_7: "f32[512]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_10: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000398612827361);  squeeze_5 = None
    mul_11: "f32[512]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[512]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_8: "f32[512]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_5: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_7: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 512, 56, 56]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_1: "f32[8, 512, 56, 56]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_2: "f32[8, 512, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_323, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 512, 1, 1]" = var_mean_2[0]
    getitem_7: "f32[1, 512, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_2: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
    mul_14: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_7: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[512]" = torch.ops.aten.mul.Tensor(primals_321, 0.9)
    add_12: "f32[512]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_17: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000398612827361);  squeeze_8 = None
    mul_18: "f32[512]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[512]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_13: "f32[512]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_9: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_11: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 512, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_2: "f32[8, 512, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_3: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_326, 1)
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 256, 1, 1]" = var_mean_3[0]
    getitem_9: "f32[1, 256, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_3: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
    mul_21: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_10: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[256]" = torch.ops.aten.mul.Tensor(primals_324, 0.9)
    add_17: "f32[256]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_24: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000398612827361);  squeeze_11 = None
    mul_25: "f32[256]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[256]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_18: "f32[256]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_13: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_15: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    convolution_4: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem_2, primals_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_329, 1)
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 256, 1, 1]" = var_mean_4[0]
    getitem_11: "f32[1, 256, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_4: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
    mul_28: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_13: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(primals_327, 0.9)
    add_22: "f32[256]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_31: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000398612827361);  squeeze_14 = None
    mul_32: "f32[256]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[256]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_23: "f32[256]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_17: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_19: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_25: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_19, add_24);  add_19 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_3: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_5: "f32[8, 512, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_332, 1)
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 512, 1, 1]" = var_mean_5[0]
    getitem_13: "f32[1, 512, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_27: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_5: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_5: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_13)
    mul_35: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_16: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[512]" = torch.ops.aten.mul.Tensor(primals_330, 0.9)
    add_28: "f32[512]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_38: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_39: "f32[512]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[512]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_29: "f32[512]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_21: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_23: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_30: "f32[8, 512, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_4: "f32[8, 512, 56, 56]" = torch.ops.aten.relu.default(add_30);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_6: "f32[8, 512, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_19, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_335, 1)
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 512, 1, 1]" = var_mean_6[0]
    getitem_15: "f32[1, 512, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_32: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_6: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_6: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_15)
    mul_42: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_19: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[512]" = torch.ops.aten.mul.Tensor(primals_333, 0.9)
    add_33: "f32[512]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_45: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_46: "f32[512]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[512]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_34: "f32[512]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
    unsqueeze_25: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
    unsqueeze_27: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_35: "f32[8, 512, 56, 56]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_5: "f32[8, 512, 56, 56]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_7: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_338, 1)
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 256, 1, 1]" = var_mean_7[0]
    getitem_17: "f32[1, 256, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_7: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_17)
    mul_49: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_22: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[256]" = torch.ops.aten.mul.Tensor(primals_336, 0.9)
    add_38: "f32[256]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_52: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_53: "f32[256]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[256]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_39: "f32[256]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_29: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_31: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_41: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_40, relu_3);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_6: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_41);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_8: "f32[8, 512, 56, 56]" = torch.ops.aten.convolution.default(relu_6, primals_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_42: "i64[]" = torch.ops.aten.add.Tensor(primals_341, 1)
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 512, 1, 1]" = var_mean_8[0]
    getitem_19: "f32[1, 512, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_43: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_8: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_8: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_19)
    mul_56: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_25: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[512]" = torch.ops.aten.mul.Tensor(primals_339, 0.9)
    add_44: "f32[512]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_59: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_60: "f32[512]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[512]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_45: "f32[512]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_33: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_35: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_46: "f32[8, 512, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_7: "f32[8, 512, 56, 56]" = torch.ops.aten.relu.default(add_46);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_9: "f32[8, 512, 56, 56]" = torch.ops.aten.convolution.default(relu_7, primals_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_344, 1)
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 512, 1, 1]" = var_mean_9[0]
    getitem_21: "f32[1, 512, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_48: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_9: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_9: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_21)
    mul_63: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_28: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[512]" = torch.ops.aten.mul.Tensor(primals_342, 0.9)
    add_49: "f32[512]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_66: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_67: "f32[512]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[512]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_50: "f32[512]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_37: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_39: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_51: "f32[8, 512, 56, 56]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_8: "f32[8, 512, 56, 56]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_10: "f32[8, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_8, primals_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_347, 1)
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 256, 1, 1]" = var_mean_10[0]
    getitem_23: "f32[1, 256, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_53: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_10: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_10: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_23)
    mul_70: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_31: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_71: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_72: "f32[256]" = torch.ops.aten.mul.Tensor(primals_345, 0.9)
    add_54: "f32[256]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_32: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_73: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000398612827361);  squeeze_32 = None
    mul_74: "f32[256]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[256]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
    add_55: "f32[256]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_40: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_41: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_76: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
    unsqueeze_42: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_43: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_56: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_57: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(add_56, relu_6);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_9: "f32[8, 256, 56, 56]" = torch.ops.aten.relu.default(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_11: "f32[8, 1024, 56, 56]" = torch.ops.aten.convolution.default(relu_9, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_58: "i64[]" = torch.ops.aten.add.Tensor(primals_350, 1)
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 1024, 1, 1]" = var_mean_11[0]
    getitem_25: "f32[1, 1024, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_59: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_11: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_11: "f32[8, 1024, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_25)
    mul_77: "f32[8, 1024, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_34: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_78: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_79: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_348, 0.9)
    add_60: "f32[1024]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_35: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_80: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000398612827361);  squeeze_35 = None
    mul_81: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_349, 0.9)
    add_61: "f32[1024]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_44: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_45: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_83: "f32[8, 1024, 56, 56]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
    unsqueeze_46: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_47: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_62: "f32[8, 1024, 56, 56]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_10: "f32[8, 1024, 56, 56]" = torch.ops.aten.relu.default(add_62);  add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_12: "f32[8, 1024, 28, 28]" = torch.ops.aten.convolution.default(relu_10, primals_37, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_63: "i64[]" = torch.ops.aten.add.Tensor(primals_353, 1)
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 1024, 1, 1]" = var_mean_12[0]
    getitem_27: "f32[1, 1024, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_64: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_12: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_12: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_27)
    mul_84: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_37: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_85: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_86: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_351, 0.9)
    add_65: "f32[1024]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_38: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_87: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001594642002871);  squeeze_38 = None
    mul_88: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
    add_66: "f32[1024]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_48: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1)
    unsqueeze_49: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_90: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
    unsqueeze_50: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
    unsqueeze_51: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_67: "f32[8, 1024, 28, 28]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_11: "f32[8, 1024, 28, 28]" = torch.ops.aten.relu.default(add_67);  add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_13: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_356, 1)
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 512, 1, 1]" = var_mean_13[0]
    getitem_29: "f32[1, 512, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_69: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_13: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_13: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_29)
    mul_91: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_40: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_92: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_93: "f32[512]" = torch.ops.aten.mul.Tensor(primals_354, 0.9)
    add_70: "f32[512]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    squeeze_41: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_94: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001594642002871);  squeeze_41 = None
    mul_95: "f32[512]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
    mul_96: "f32[512]" = torch.ops.aten.mul.Tensor(primals_355, 0.9)
    add_71: "f32[512]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    unsqueeze_52: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_53: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_97: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
    unsqueeze_54: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_55: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_72: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    convolution_14: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_9, primals_43, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_359, 1)
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 512, 1, 1]" = var_mean_14[0]
    getitem_31: "f32[1, 512, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_74: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_14: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_14: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_31)
    mul_98: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_43: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_99: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_100: "f32[512]" = torch.ops.aten.mul.Tensor(primals_357, 0.9)
    add_75: "f32[512]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_44: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_101: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001594642002871);  squeeze_44 = None
    mul_102: "f32[512]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[512]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
    add_76: "f32[512]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_56: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
    unsqueeze_57: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_104: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
    unsqueeze_58: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
    unsqueeze_59: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_77: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_78: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_72, add_77);  add_72 = add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_12: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_78);  add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_15: "f32[8, 1024, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_79: "i64[]" = torch.ops.aten.add.Tensor(primals_362, 1)
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 1024, 1, 1]" = var_mean_15[0]
    getitem_33: "f32[1, 1024, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_80: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_15: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_15: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_33)
    mul_105: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_46: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_106: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_107: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_360, 0.9)
    add_81: "f32[1024]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_47: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_108: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001594642002871);  squeeze_47 = None
    mul_109: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_361, 0.9)
    add_82: "f32[1024]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_60: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_61: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_111: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
    unsqueeze_62: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_63: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_83: "f32[8, 1024, 28, 28]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_13: "f32[8, 1024, 28, 28]" = torch.ops.aten.relu.default(add_83);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_16: "f32[8, 1024, 28, 28]" = torch.ops.aten.convolution.default(relu_13, primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_84: "i64[]" = torch.ops.aten.add.Tensor(primals_365, 1)
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 1024, 1, 1]" = var_mean_16[0]
    getitem_35: "f32[1, 1024, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_85: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_16: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_16: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_35)
    mul_112: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_49: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_113: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_114: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_363, 0.9)
    add_86: "f32[1024]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_50: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_115: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
    mul_116: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
    add_87: "f32[1024]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_64: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
    unsqueeze_65: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_118: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
    unsqueeze_66: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
    unsqueeze_67: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_88: "f32[8, 1024, 28, 28]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_14: "f32[8, 1024, 28, 28]" = torch.ops.aten.relu.default(add_88);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_17: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_14, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_89: "i64[]" = torch.ops.aten.add.Tensor(primals_368, 1)
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 512, 1, 1]" = var_mean_17[0]
    getitem_37: "f32[1, 512, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_90: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_17: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_17: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_37)
    mul_119: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_52: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_120: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_121: "f32[512]" = torch.ops.aten.mul.Tensor(primals_366, 0.9)
    add_91: "f32[512]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_53: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_122: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001594642002871);  squeeze_53 = None
    mul_123: "f32[512]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[512]" = torch.ops.aten.mul.Tensor(primals_367, 0.9)
    add_92: "f32[512]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_68: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_69: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_125: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
    unsqueeze_70: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_71: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_93: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_94: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_93, relu_12);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_15: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_94);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_18: "f32[8, 1024, 28, 28]" = torch.ops.aten.convolution.default(relu_15, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_95: "i64[]" = torch.ops.aten.add.Tensor(primals_371, 1)
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 1024, 1, 1]" = var_mean_18[0]
    getitem_39: "f32[1, 1024, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_96: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_18: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_18: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_39)
    mul_126: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_55: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_127: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_128: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_369, 0.9)
    add_97: "f32[1024]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_56: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_129: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
    mul_130: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
    add_98: "f32[1024]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_72: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_73: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_132: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
    unsqueeze_74: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_75: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_99: "f32[8, 1024, 28, 28]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_16: "f32[8, 1024, 28, 28]" = torch.ops.aten.relu.default(add_99);  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_19: "f32[8, 1024, 28, 28]" = torch.ops.aten.convolution.default(relu_16, primals_58, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_100: "i64[]" = torch.ops.aten.add.Tensor(primals_374, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 1024, 1, 1]" = var_mean_19[0]
    getitem_41: "f32[1, 1024, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_101: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_19: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    sub_19: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_41)
    mul_133: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_58: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_134: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_135: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_372, 0.9)
    add_102: "f32[1024]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_59: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_136: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
    mul_137: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_373, 0.9)
    add_103: "f32[1024]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_76: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_77: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_139: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
    unsqueeze_78: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_79: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_104: "f32[8, 1024, 28, 28]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_17: "f32[8, 1024, 28, 28]" = torch.ops.aten.relu.default(add_104);  add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_20: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_17, primals_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_105: "i64[]" = torch.ops.aten.add.Tensor(primals_377, 1)
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 512, 1, 1]" = var_mean_20[0]
    getitem_43: "f32[1, 512, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_106: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_20: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_20: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_43)
    mul_140: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_61: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_141: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_142: "f32[512]" = torch.ops.aten.mul.Tensor(primals_375, 0.9)
    add_107: "f32[512]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_62: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_143: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
    mul_144: "f32[512]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[512]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
    add_108: "f32[512]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    unsqueeze_80: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_81: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_146: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
    unsqueeze_82: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_83: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_109: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_110: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_109, relu_15);  add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_18: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_110);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_21: "f32[8, 1024, 28, 28]" = torch.ops.aten.convolution.default(relu_18, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_111: "i64[]" = torch.ops.aten.add.Tensor(primals_380, 1)
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 1024, 1, 1]" = var_mean_21[0]
    getitem_45: "f32[1, 1024, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_112: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_21: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_21: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_45)
    mul_147: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_64: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_148: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_149: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_378, 0.9)
    add_113: "f32[1024]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_65: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_150: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
    mul_151: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
    add_114: "f32[1024]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_84: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_85: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_153: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
    unsqueeze_86: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_87: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_115: "f32[8, 1024, 28, 28]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_19: "f32[8, 1024, 28, 28]" = torch.ops.aten.relu.default(add_115);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_22: "f32[8, 1024, 28, 28]" = torch.ops.aten.convolution.default(relu_19, primals_67, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_116: "i64[]" = torch.ops.aten.add.Tensor(primals_383, 1)
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 1024, 1, 1]" = var_mean_22[0]
    getitem_47: "f32[1, 1024, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_117: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_22: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_22: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_47)
    mul_154: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_67: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_155: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_156: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_381, 0.9)
    add_118: "f32[1024]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    squeeze_68: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_157: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001594642002871);  squeeze_68 = None
    mul_158: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
    mul_159: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
    add_119: "f32[1024]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    unsqueeze_88: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_89: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_160: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
    unsqueeze_90: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_91: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_120: "f32[8, 1024, 28, 28]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_20: "f32[8, 1024, 28, 28]" = torch.ops.aten.relu.default(add_120);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_23: "f32[8, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_20, primals_70, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_121: "i64[]" = torch.ops.aten.add.Tensor(primals_386, 1)
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 512, 1, 1]" = var_mean_23[0]
    getitem_49: "f32[1, 512, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_122: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_23: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
    sub_23: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_49)
    mul_161: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_70: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_162: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_163: "f32[512]" = torch.ops.aten.mul.Tensor(primals_384, 0.9)
    add_123: "f32[512]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    squeeze_71: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_164: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001594642002871);  squeeze_71 = None
    mul_165: "f32[512]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
    mul_166: "f32[512]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
    add_124: "f32[512]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    unsqueeze_92: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_93: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_167: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
    unsqueeze_94: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_95: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_125: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_126: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(add_125, relu_18);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_21: "f32[8, 512, 28, 28]" = torch.ops.aten.relu.default(add_126);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_24: "f32[8, 2048, 28, 28]" = torch.ops.aten.convolution.default(relu_21, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_127: "i64[]" = torch.ops.aten.add.Tensor(primals_389, 1)
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 2048, 1, 1]" = var_mean_24[0]
    getitem_51: "f32[1, 2048, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_128: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_24: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    sub_24: "f32[8, 2048, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_51)
    mul_168: "f32[8, 2048, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_73: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_169: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_170: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_387, 0.9)
    add_129: "f32[2048]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_74: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_171: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001594642002871);  squeeze_74 = None
    mul_172: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
    add_130: "f32[2048]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_96: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_97: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_174: "f32[8, 2048, 28, 28]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
    unsqueeze_98: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_99: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_131: "f32[8, 2048, 28, 28]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_22: "f32[8, 2048, 28, 28]" = torch.ops.aten.relu.default(add_131);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_25: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_22, primals_76, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_132: "i64[]" = torch.ops.aten.add.Tensor(primals_392, 1)
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 2048, 1, 1]" = var_mean_25[0]
    getitem_53: "f32[1, 2048, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_133: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_25: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_25: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_53)
    mul_175: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_76: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_176: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_177: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_390, 0.9)
    add_134: "f32[2048]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_77: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_178: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0006381620931717);  squeeze_77 = None
    mul_179: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
    add_135: "f32[2048]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_100: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_101: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_181: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
    unsqueeze_102: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_103: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_136: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_23: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_136);  add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_26: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_23, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_137: "i64[]" = torch.ops.aten.add.Tensor(primals_395, 1)
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 1024, 1, 1]" = var_mean_26[0]
    getitem_55: "f32[1, 1024, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_138: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_26: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_26: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_55)
    mul_182: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_79: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_183: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_184: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_393, 0.9)
    add_139: "f32[1024]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_80: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_185: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0006381620931717);  squeeze_80 = None
    mul_186: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
    add_140: "f32[1024]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_104: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_105: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_188: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
    unsqueeze_106: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_107: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_141: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    convolution_27: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_21, primals_82, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    add_142: "i64[]" = torch.ops.aten.add.Tensor(primals_398, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 1024, 1, 1]" = var_mean_27[0]
    getitem_57: "f32[1, 1024, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_143: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_27: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_27: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_57)
    mul_189: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_82: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_190: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_191: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_396, 0.9)
    add_144: "f32[1024]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    squeeze_83: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_192: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0006381620931717);  squeeze_83 = None
    mul_193: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
    mul_194: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
    add_145: "f32[1024]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    unsqueeze_108: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_109: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_195: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
    unsqueeze_110: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_111: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_146: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_147: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_141, add_146);  add_141 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_24: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_147);  add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_28: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_24, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_148: "i64[]" = torch.ops.aten.add.Tensor(primals_401, 1)
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 2048, 1, 1]" = var_mean_28[0]
    getitem_59: "f32[1, 2048, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_149: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_28: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    sub_28: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_59)
    mul_196: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_85: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_197: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_198: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_399, 0.9)
    add_150: "f32[2048]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_86: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_199: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0006381620931717);  squeeze_86 = None
    mul_200: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
    add_151: "f32[2048]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_112: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_113: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_202: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
    unsqueeze_114: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_115: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_152: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_25: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_152);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_29: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_25, primals_88, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_153: "i64[]" = torch.ops.aten.add.Tensor(primals_404, 1)
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 2048, 1, 1]" = var_mean_29[0]
    getitem_61: "f32[1, 2048, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_154: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_29: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_29: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_61)
    mul_203: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_88: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_204: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_205: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_402, 0.9)
    add_155: "f32[2048]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_89: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_206: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0006381620931717);  squeeze_89 = None
    mul_207: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
    add_156: "f32[2048]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_116: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_117: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_209: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
    unsqueeze_118: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_119: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_157: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_26: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_157);  add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_30: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_26, primals_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_158: "i64[]" = torch.ops.aten.add.Tensor(primals_407, 1)
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 1024, 1, 1]" = var_mean_30[0]
    getitem_63: "f32[1, 1024, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_159: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_30: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    sub_30: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_63)
    mul_210: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_91: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_211: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_212: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_405, 0.9)
    add_160: "f32[1024]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    squeeze_92: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_213: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0006381620931717);  squeeze_92 = None
    mul_214: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
    mul_215: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
    add_161: "f32[1024]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    unsqueeze_120: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_121: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_216: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
    unsqueeze_122: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_123: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_162: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_163: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_162, relu_24);  add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_27: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_163);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_31: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_27, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_164: "i64[]" = torch.ops.aten.add.Tensor(primals_410, 1)
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 2048, 1, 1]" = var_mean_31[0]
    getitem_65: "f32[1, 2048, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_165: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_31: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_31: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_65)
    mul_217: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_94: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_218: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_219: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_408, 0.9)
    add_166: "f32[2048]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_95: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_220: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0006381620931717);  squeeze_95 = None
    mul_221: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_409, 0.9)
    add_167: "f32[2048]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    unsqueeze_124: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_125: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_223: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
    unsqueeze_126: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_127: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_168: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_28: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_168);  add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_32: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_28, primals_97, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_169: "i64[]" = torch.ops.aten.add.Tensor(primals_413, 1)
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 2048, 1, 1]" = var_mean_32[0]
    getitem_67: "f32[1, 2048, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_170: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_32: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
    sub_32: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_67)
    mul_224: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_97: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_225: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_226: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_411, 0.9)
    add_171: "f32[2048]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_98: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_227: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0006381620931717);  squeeze_98 = None
    mul_228: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
    add_172: "f32[2048]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_128: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_129: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_230: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
    unsqueeze_130: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_131: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_173: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_29: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_173);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_33: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_29, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_174: "i64[]" = torch.ops.aten.add.Tensor(primals_416, 1)
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 1024, 1, 1]" = var_mean_33[0]
    getitem_69: "f32[1, 1024, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_175: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_33: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_33: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_69)
    mul_231: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_100: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_232: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_233: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_414, 0.9)
    add_176: "f32[1024]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_101: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_234: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0006381620931717);  squeeze_101 = None
    mul_235: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_415, 0.9)
    add_177: "f32[1024]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_132: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_133: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_237: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
    unsqueeze_134: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_135: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_178: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_179: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_178, relu_27);  add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_30: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_179);  add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_34: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_30, primals_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_419, 1)
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 2048, 1, 1]" = var_mean_34[0]
    getitem_71: "f32[1, 2048, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_181: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_34: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    sub_34: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_71)
    mul_238: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_103: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_239: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_240: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_417, 0.9)
    add_182: "f32[2048]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_104: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_241: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0006381620931717);  squeeze_104 = None
    mul_242: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_418, 0.9)
    add_183: "f32[2048]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_136: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
    unsqueeze_137: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_244: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
    unsqueeze_138: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
    unsqueeze_139: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_184: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_31: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_184);  add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_35: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_31, primals_106, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_422, 1)
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 2048, 1, 1]" = var_mean_35[0]
    getitem_73: "f32[1, 2048, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_186: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_35: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    sub_35: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_73)
    mul_245: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_106: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_246: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_247: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_420, 0.9)
    add_187: "f32[2048]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_107: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_248: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0006381620931717);  squeeze_107 = None
    mul_249: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_421, 0.9)
    add_188: "f32[2048]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_140: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_141: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_251: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
    unsqueeze_142: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_143: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_189: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_32: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_189);  add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_36: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_32, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_190: "i64[]" = torch.ops.aten.add.Tensor(primals_425, 1)
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 1024, 1, 1]" = var_mean_36[0]
    getitem_75: "f32[1, 1024, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_191: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_36: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_36: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_75)
    mul_252: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_109: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_253: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_254: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_423, 0.9)
    add_192: "f32[1024]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    squeeze_110: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_255: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0006381620931717);  squeeze_110 = None
    mul_256: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
    mul_257: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_424, 0.9)
    add_193: "f32[1024]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    unsqueeze_144: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_145: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_258: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
    unsqueeze_146: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_147: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_194: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_195: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_194, relu_30);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_33: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_195);  add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_37: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_33, primals_112, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_196: "i64[]" = torch.ops.aten.add.Tensor(primals_428, 1)
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 2048, 1, 1]" = var_mean_37[0]
    getitem_77: "f32[1, 2048, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_197: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_37: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    sub_37: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_77)
    mul_259: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_112: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_260: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_261: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_426, 0.9)
    add_198: "f32[2048]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    squeeze_113: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_262: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0006381620931717);  squeeze_113 = None
    mul_263: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
    mul_264: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_427, 0.9)
    add_199: "f32[2048]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    unsqueeze_148: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_149: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_265: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
    unsqueeze_150: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_151: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_200: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_34: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_200);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_38: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_34, primals_115, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_201: "i64[]" = torch.ops.aten.add.Tensor(primals_431, 1)
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 2048, 1, 1]" = var_mean_38[0]
    getitem_79: "f32[1, 2048, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_202: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_38: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    sub_38: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_79)
    mul_266: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_115: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_267: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_268: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_429, 0.9)
    add_203: "f32[2048]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    squeeze_116: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_269: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0006381620931717);  squeeze_116 = None
    mul_270: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
    mul_271: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_430, 0.9)
    add_204: "f32[2048]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    unsqueeze_152: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
    unsqueeze_153: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_272: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
    unsqueeze_154: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
    unsqueeze_155: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_205: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_35: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_205);  add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_39: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_35, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_206: "i64[]" = torch.ops.aten.add.Tensor(primals_434, 1)
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 1024, 1, 1]" = var_mean_39[0]
    getitem_81: "f32[1, 1024, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_207: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_39: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
    sub_39: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_81)
    mul_273: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_118: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_274: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_275: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_432, 0.9)
    add_208: "f32[1024]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    squeeze_119: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_276: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0006381620931717);  squeeze_119 = None
    mul_277: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
    mul_278: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_433, 0.9)
    add_209: "f32[1024]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    unsqueeze_156: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_157: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_279: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
    unsqueeze_158: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_159: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_210: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_211: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_210, relu_33);  add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_36: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_211);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_40: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_36, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_437, 1)
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 2048, 1, 1]" = var_mean_40[0]
    getitem_83: "f32[1, 2048, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_213: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_40: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_40: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_83)
    mul_280: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_121: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_281: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_282: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_435, 0.9)
    add_214: "f32[2048]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    squeeze_122: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_283: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0006381620931717);  squeeze_122 = None
    mul_284: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
    mul_285: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_436, 0.9)
    add_215: "f32[2048]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_160: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
    unsqueeze_161: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_286: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
    unsqueeze_162: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
    unsqueeze_163: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_216: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_37: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_216);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_41: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_37, primals_124, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_217: "i64[]" = torch.ops.aten.add.Tensor(primals_440, 1)
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 2048, 1, 1]" = var_mean_41[0]
    getitem_85: "f32[1, 2048, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_218: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_41: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    sub_41: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_85)
    mul_287: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_124: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_288: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_289: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_438, 0.9)
    add_219: "f32[2048]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    squeeze_125: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_290: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0006381620931717);  squeeze_125 = None
    mul_291: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
    mul_292: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_439, 0.9)
    add_220: "f32[2048]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    unsqueeze_164: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_165: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_293: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
    unsqueeze_166: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_167: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_221: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_38: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_221);  add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_42: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_38, primals_127, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_222: "i64[]" = torch.ops.aten.add.Tensor(primals_443, 1)
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 1024, 1, 1]" = var_mean_42[0]
    getitem_87: "f32[1, 1024, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_223: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_42: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
    sub_42: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_87)
    mul_294: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_127: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_295: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_296: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_441, 0.9)
    add_224: "f32[1024]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_128: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_297: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0006381620931717);  squeeze_128 = None
    mul_298: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_442, 0.9)
    add_225: "f32[1024]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_168: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
    unsqueeze_169: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_300: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
    unsqueeze_170: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
    unsqueeze_171: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_226: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_227: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_226, relu_36);  add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_39: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_227);  add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_43: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_39, primals_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_228: "i64[]" = torch.ops.aten.add.Tensor(primals_446, 1)
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 2048, 1, 1]" = var_mean_43[0]
    getitem_89: "f32[1, 2048, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_229: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_43: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
    sub_43: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_89)
    mul_301: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_130: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_302: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_303: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_444, 0.9)
    add_230: "f32[2048]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    squeeze_131: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_304: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0006381620931717);  squeeze_131 = None
    mul_305: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
    mul_306: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_445, 0.9)
    add_231: "f32[2048]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_172: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_173: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_307: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
    unsqueeze_174: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_175: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_232: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_40: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_232);  add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_44: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_40, primals_133, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_233: "i64[]" = torch.ops.aten.add.Tensor(primals_449, 1)
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 2048, 1, 1]" = var_mean_44[0]
    getitem_91: "f32[1, 2048, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_234: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
    rsqrt_44: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
    sub_44: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_91)
    mul_308: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_133: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_309: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_310: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_447, 0.9)
    add_235: "f32[2048]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    squeeze_134: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_311: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0006381620931717);  squeeze_134 = None
    mul_312: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
    mul_313: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_448, 0.9)
    add_236: "f32[2048]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_176: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
    unsqueeze_177: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_314: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
    unsqueeze_178: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
    unsqueeze_179: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_237: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_41: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_237);  add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_45: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_41, primals_136, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_238: "i64[]" = torch.ops.aten.add.Tensor(primals_452, 1)
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[1, 1024, 1, 1]" = var_mean_45[0]
    getitem_93: "f32[1, 1024, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_239: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
    rsqrt_45: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
    sub_45: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_93)
    mul_315: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
    squeeze_136: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_316: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_317: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_450, 0.9)
    add_240: "f32[1024]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    squeeze_137: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
    mul_318: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0006381620931717);  squeeze_137 = None
    mul_319: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
    mul_320: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_451, 0.9)
    add_241: "f32[1024]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    unsqueeze_180: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
    unsqueeze_181: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_321: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
    unsqueeze_182: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
    unsqueeze_183: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_242: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_243: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_242, relu_39);  add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_42: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_243);  add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_46: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_42, primals_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_244: "i64[]" = torch.ops.aten.add.Tensor(primals_455, 1)
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 2048, 1, 1]" = var_mean_46[0]
    getitem_95: "f32[1, 2048, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_245: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
    rsqrt_46: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
    sub_46: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_95)
    mul_322: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_139: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_323: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_324: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_453, 0.9)
    add_246: "f32[2048]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    squeeze_140: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_325: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0006381620931717);  squeeze_140 = None
    mul_326: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
    mul_327: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_454, 0.9)
    add_247: "f32[2048]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    unsqueeze_184: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
    unsqueeze_185: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_328: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
    unsqueeze_186: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
    unsqueeze_187: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_248: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_43: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_248);  add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_47: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_43, primals_142, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_249: "i64[]" = torch.ops.aten.add.Tensor(primals_458, 1)
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 2048, 1, 1]" = var_mean_47[0]
    getitem_97: "f32[1, 2048, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_250: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_47: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
    sub_47: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_97)
    mul_329: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_142: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_330: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_331: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_456, 0.9)
    add_251: "f32[2048]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    squeeze_143: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_332: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0006381620931717);  squeeze_143 = None
    mul_333: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
    mul_334: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_457, 0.9)
    add_252: "f32[2048]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    unsqueeze_188: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_189: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_335: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
    unsqueeze_190: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_191: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_253: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_44: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_253);  add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_48: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_44, primals_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_254: "i64[]" = torch.ops.aten.add.Tensor(primals_461, 1)
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 1024, 1, 1]" = var_mean_48[0]
    getitem_99: "f32[1, 1024, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_255: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
    rsqrt_48: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
    sub_48: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_99)
    mul_336: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_145: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_337: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_338: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_459, 0.9)
    add_256: "f32[1024]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    squeeze_146: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_339: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0006381620931717);  squeeze_146 = None
    mul_340: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
    mul_341: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_460, 0.9)
    add_257: "f32[1024]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    unsqueeze_192: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1)
    unsqueeze_193: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_342: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
    unsqueeze_194: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
    unsqueeze_195: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_258: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_259: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_258, relu_42);  add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_45: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_259);  add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_49: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_45, primals_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_260: "i64[]" = torch.ops.aten.add.Tensor(primals_464, 1)
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 2048, 1, 1]" = var_mean_49[0]
    getitem_101: "f32[1, 2048, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_261: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
    rsqrt_49: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
    sub_49: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_101)
    mul_343: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_148: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_344: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_345: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_462, 0.9)
    add_262: "f32[2048]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    squeeze_149: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_346: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0006381620931717);  squeeze_149 = None
    mul_347: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
    mul_348: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_463, 0.9)
    add_263: "f32[2048]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    unsqueeze_196: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1)
    unsqueeze_197: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_349: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
    unsqueeze_198: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
    unsqueeze_199: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_264: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_46: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_264);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_50: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_46, primals_151, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_265: "i64[]" = torch.ops.aten.add.Tensor(primals_467, 1)
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 2048, 1, 1]" = var_mean_50[0]
    getitem_103: "f32[1, 2048, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_266: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_50: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
    sub_50: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_103)
    mul_350: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_151: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_351: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_352: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_465, 0.9)
    add_267: "f32[2048]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    squeeze_152: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_353: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0006381620931717);  squeeze_152 = None
    mul_354: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
    mul_355: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_466, 0.9)
    add_268: "f32[2048]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_200: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1)
    unsqueeze_201: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_356: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
    unsqueeze_202: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
    unsqueeze_203: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_269: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_47: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_269);  add_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_51: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_47, primals_154, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_270: "i64[]" = torch.ops.aten.add.Tensor(primals_470, 1)
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 1024, 1, 1]" = var_mean_51[0]
    getitem_105: "f32[1, 1024, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_271: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
    rsqrt_51: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
    sub_51: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_105)
    mul_357: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_154: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_358: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_359: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_468, 0.9)
    add_272: "f32[1024]" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    squeeze_155: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_360: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0006381620931717);  squeeze_155 = None
    mul_361: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
    mul_362: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_469, 0.9)
    add_273: "f32[1024]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    unsqueeze_204: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
    unsqueeze_205: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_363: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
    unsqueeze_206: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
    unsqueeze_207: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_274: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_275: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_274, relu_45);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_48: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_275);  add_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_52: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_48, primals_157, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_276: "i64[]" = torch.ops.aten.add.Tensor(primals_473, 1)
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 2048, 1, 1]" = var_mean_52[0]
    getitem_107: "f32[1, 2048, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_277: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_52: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_277);  add_277 = None
    sub_52: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_107)
    mul_364: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_157: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_365: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_366: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_471, 0.9)
    add_278: "f32[2048]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    squeeze_158: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_367: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0006381620931717);  squeeze_158 = None
    mul_368: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
    mul_369: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_472, 0.9)
    add_279: "f32[2048]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    unsqueeze_208: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_158, -1)
    unsqueeze_209: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_370: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
    unsqueeze_210: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1);  primals_159 = None
    unsqueeze_211: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_280: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_49: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_280);  add_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_53: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_49, primals_160, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_281: "i64[]" = torch.ops.aten.add.Tensor(primals_476, 1)
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 2048, 1, 1]" = var_mean_53[0]
    getitem_109: "f32[1, 2048, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_282: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
    rsqrt_53: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
    sub_53: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_109)
    mul_371: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_160: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_372: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_373: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_474, 0.9)
    add_283: "f32[2048]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    squeeze_161: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_374: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0006381620931717);  squeeze_161 = None
    mul_375: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
    mul_376: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_475, 0.9)
    add_284: "f32[2048]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_212: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_161, -1)
    unsqueeze_213: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_377: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
    unsqueeze_214: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1);  primals_162 = None
    unsqueeze_215: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_285: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_50: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_285);  add_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_54: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_50, primals_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_286: "i64[]" = torch.ops.aten.add.Tensor(primals_479, 1)
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 1024, 1, 1]" = var_mean_54[0]
    getitem_111: "f32[1, 1024, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_287: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
    rsqrt_54: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_287);  add_287 = None
    sub_54: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_111)
    mul_378: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_163: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_379: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_380: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_477, 0.9)
    add_288: "f32[1024]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    squeeze_164: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_381: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0006381620931717);  squeeze_164 = None
    mul_382: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
    mul_383: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_478, 0.9)
    add_289: "f32[1024]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    unsqueeze_216: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_164, -1)
    unsqueeze_217: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_384: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
    unsqueeze_218: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_165, -1);  primals_165 = None
    unsqueeze_219: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_290: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_291: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_290, relu_48);  add_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_51: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_291);  add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_55: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_51, primals_166, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_292: "i64[]" = torch.ops.aten.add.Tensor(primals_482, 1)
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 2048, 1, 1]" = var_mean_55[0]
    getitem_113: "f32[1, 2048, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_293: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_55: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
    sub_55: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_113)
    mul_385: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_166: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_386: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_387: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_480, 0.9)
    add_294: "f32[2048]" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
    squeeze_167: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_388: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0006381620931717);  squeeze_167 = None
    mul_389: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
    mul_390: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_481, 0.9)
    add_295: "f32[2048]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    unsqueeze_220: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1)
    unsqueeze_221: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_391: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
    unsqueeze_222: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1);  primals_168 = None
    unsqueeze_223: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_296: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_52: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_296);  add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_56: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_52, primals_169, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_297: "i64[]" = torch.ops.aten.add.Tensor(primals_485, 1)
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 2048, 1, 1]" = var_mean_56[0]
    getitem_115: "f32[1, 2048, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_298: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
    rsqrt_56: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_298);  add_298 = None
    sub_56: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_115)
    mul_392: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_169: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_393: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_394: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_483, 0.9)
    add_299: "f32[2048]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    squeeze_170: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_395: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0006381620931717);  squeeze_170 = None
    mul_396: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
    mul_397: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_484, 0.9)
    add_300: "f32[2048]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    unsqueeze_224: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_170, -1)
    unsqueeze_225: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_398: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_225);  mul_392 = unsqueeze_225 = None
    unsqueeze_226: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_171, -1);  primals_171 = None
    unsqueeze_227: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_301: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_227);  mul_398 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_53: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_301);  add_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_57: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_53, primals_172, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_302: "i64[]" = torch.ops.aten.add.Tensor(primals_488, 1)
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[1, 1024, 1, 1]" = var_mean_57[0]
    getitem_117: "f32[1, 1024, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_303: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
    rsqrt_57: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
    sub_57: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_117)
    mul_399: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
    squeeze_172: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_400: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_401: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_486, 0.9)
    add_304: "f32[1024]" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
    squeeze_173: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
    mul_402: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0006381620931717);  squeeze_173 = None
    mul_403: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_402, 0.1);  mul_402 = None
    mul_404: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_487, 0.9)
    add_305: "f32[1024]" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
    unsqueeze_228: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_173, -1)
    unsqueeze_229: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_405: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_229);  mul_399 = unsqueeze_229 = None
    unsqueeze_230: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_174, -1);  primals_174 = None
    unsqueeze_231: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_306: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_405, unsqueeze_231);  mul_405 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_307: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_306, relu_51);  add_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_54: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_307);  add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_58: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_54, primals_175, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_308: "i64[]" = torch.ops.aten.add.Tensor(primals_491, 1)
    var_mean_58 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 2048, 1, 1]" = var_mean_58[0]
    getitem_119: "f32[1, 2048, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_309: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_58: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_309);  add_309 = None
    sub_58: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_119)
    mul_406: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    squeeze_174: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_175: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_407: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_408: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_489, 0.9)
    add_310: "f32[2048]" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    squeeze_176: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_409: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0006381620931717);  squeeze_176 = None
    mul_410: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_409, 0.1);  mul_409 = None
    mul_411: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_490, 0.9)
    add_311: "f32[2048]" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    unsqueeze_232: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_176, -1)
    unsqueeze_233: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_412: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_233);  mul_406 = unsqueeze_233 = None
    unsqueeze_234: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_177, -1);  primals_177 = None
    unsqueeze_235: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_312: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_235);  mul_412 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_55: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_312);  add_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_59: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_55, primals_178, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_313: "i64[]" = torch.ops.aten.add.Tensor(primals_494, 1)
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 2048, 1, 1]" = var_mean_59[0]
    getitem_121: "f32[1, 2048, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_314: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
    rsqrt_59: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
    sub_59: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_121)
    mul_413: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    squeeze_177: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_178: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    mul_414: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_415: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_492, 0.9)
    add_315: "f32[2048]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    squeeze_179: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_416: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0006381620931717);  squeeze_179 = None
    mul_417: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
    mul_418: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_493, 0.9)
    add_316: "f32[2048]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    unsqueeze_236: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_179, -1)
    unsqueeze_237: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_419: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_237);  mul_413 = unsqueeze_237 = None
    unsqueeze_238: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_180, -1);  primals_180 = None
    unsqueeze_239: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_317: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_239);  mul_419 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_56: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_317);  add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_60: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_56, primals_181, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_318: "i64[]" = torch.ops.aten.add.Tensor(primals_497, 1)
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[1, 1024, 1, 1]" = var_mean_60[0]
    getitem_123: "f32[1, 1024, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_319: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
    rsqrt_60: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_319);  add_319 = None
    sub_60: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_123)
    mul_420: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    squeeze_180: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    squeeze_181: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    mul_421: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_422: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_495, 0.9)
    add_320: "f32[1024]" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    squeeze_182: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
    mul_423: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0006381620931717);  squeeze_182 = None
    mul_424: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_423, 0.1);  mul_423 = None
    mul_425: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_496, 0.9)
    add_321: "f32[1024]" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
    unsqueeze_240: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_182, -1)
    unsqueeze_241: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_426: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_241);  mul_420 = unsqueeze_241 = None
    unsqueeze_242: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_183, -1);  primals_183 = None
    unsqueeze_243: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_322: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_243);  mul_426 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_323: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_322, relu_54);  add_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_57: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_323);  add_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_61: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_57, primals_184, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_324: "i64[]" = torch.ops.aten.add.Tensor(primals_500, 1)
    var_mean_61 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 2048, 1, 1]" = var_mean_61[0]
    getitem_125: "f32[1, 2048, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_325: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
    rsqrt_61: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_325);  add_325 = None
    sub_61: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_125)
    mul_427: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    squeeze_183: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_184: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_428: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_429: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_498, 0.9)
    add_326: "f32[2048]" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    squeeze_185: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_430: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0006381620931717);  squeeze_185 = None
    mul_431: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_430, 0.1);  mul_430 = None
    mul_432: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_499, 0.9)
    add_327: "f32[2048]" = torch.ops.aten.add.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
    unsqueeze_244: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_185, -1)
    unsqueeze_245: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_433: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_245);  mul_427 = unsqueeze_245 = None
    unsqueeze_246: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_186, -1);  primals_186 = None
    unsqueeze_247: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_328: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_433, unsqueeze_247);  mul_433 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_58: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_328);  add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_62: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_58, primals_187, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_329: "i64[]" = torch.ops.aten.add.Tensor(primals_503, 1)
    var_mean_62 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 2048, 1, 1]" = var_mean_62[0]
    getitem_127: "f32[1, 2048, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_330: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
    rsqrt_62: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_330);  add_330 = None
    sub_62: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_127)
    mul_434: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
    squeeze_186: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_187: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
    mul_435: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_436: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_501, 0.9)
    add_331: "f32[2048]" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
    squeeze_188: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_437: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0006381620931717);  squeeze_188 = None
    mul_438: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_437, 0.1);  mul_437 = None
    mul_439: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_502, 0.9)
    add_332: "f32[2048]" = torch.ops.aten.add.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_248: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_188, -1)
    unsqueeze_249: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    mul_440: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_249);  mul_434 = unsqueeze_249 = None
    unsqueeze_250: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_189, -1);  primals_189 = None
    unsqueeze_251: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    add_333: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_251);  mul_440 = unsqueeze_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_59: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_333);  add_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_63: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_59, primals_190, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_334: "i64[]" = torch.ops.aten.add.Tensor(primals_506, 1)
    var_mean_63 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[1, 1024, 1, 1]" = var_mean_63[0]
    getitem_129: "f32[1, 1024, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_335: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
    rsqrt_63: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_335);  add_335 = None
    sub_63: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_129)
    mul_441: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
    squeeze_189: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
    squeeze_190: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
    mul_442: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_443: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_504, 0.9)
    add_336: "f32[1024]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    squeeze_191: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
    mul_444: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0006381620931717);  squeeze_191 = None
    mul_445: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
    mul_446: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_505, 0.9)
    add_337: "f32[1024]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
    unsqueeze_252: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_191, -1)
    unsqueeze_253: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_447: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_253);  mul_441 = unsqueeze_253 = None
    unsqueeze_254: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_192, -1);  primals_192 = None
    unsqueeze_255: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_338: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_255);  mul_447 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_339: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_338, relu_57);  add_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_60: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_339);  add_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_64: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_60, primals_193, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_340: "i64[]" = torch.ops.aten.add.Tensor(primals_509, 1)
    var_mean_64 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_130: "f32[1, 2048, 1, 1]" = var_mean_64[0]
    getitem_131: "f32[1, 2048, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_341: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05)
    rsqrt_64: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
    sub_64: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_131)
    mul_448: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
    squeeze_192: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
    squeeze_193: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
    mul_449: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
    mul_450: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_507, 0.9)
    add_342: "f32[2048]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    squeeze_194: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
    mul_451: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0006381620931717);  squeeze_194 = None
    mul_452: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
    mul_453: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_508, 0.9)
    add_343: "f32[2048]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    unsqueeze_256: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_194, -1)
    unsqueeze_257: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    mul_454: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_257);  mul_448 = unsqueeze_257 = None
    unsqueeze_258: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_195, -1);  primals_195 = None
    unsqueeze_259: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    add_344: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_259);  mul_454 = unsqueeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_61: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_344);  add_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_65: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_61, primals_196, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_345: "i64[]" = torch.ops.aten.add.Tensor(primals_512, 1)
    var_mean_65 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_132: "f32[1, 2048, 1, 1]" = var_mean_65[0]
    getitem_133: "f32[1, 2048, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_346: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05)
    rsqrt_65: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_346);  add_346 = None
    sub_65: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_133)
    mul_455: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
    squeeze_195: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
    squeeze_196: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
    mul_456: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
    mul_457: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_510, 0.9)
    add_347: "f32[2048]" = torch.ops.aten.add.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    squeeze_197: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
    mul_458: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0006381620931717);  squeeze_197 = None
    mul_459: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_458, 0.1);  mul_458 = None
    mul_460: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_511, 0.9)
    add_348: "f32[2048]" = torch.ops.aten.add.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
    unsqueeze_260: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_197, -1)
    unsqueeze_261: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_461: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_455, unsqueeze_261);  mul_455 = unsqueeze_261 = None
    unsqueeze_262: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_198, -1);  primals_198 = None
    unsqueeze_263: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_349: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_263);  mul_461 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_62: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_349);  add_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_66: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_62, primals_199, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_350: "i64[]" = torch.ops.aten.add.Tensor(primals_515, 1)
    var_mean_66 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[1, 1024, 1, 1]" = var_mean_66[0]
    getitem_135: "f32[1, 1024, 1, 1]" = var_mean_66[1];  var_mean_66 = None
    add_351: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05)
    rsqrt_66: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
    sub_66: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_135)
    mul_462: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
    squeeze_198: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
    squeeze_199: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
    mul_463: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
    mul_464: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_513, 0.9)
    add_352: "f32[1024]" = torch.ops.aten.add.Tensor(mul_463, mul_464);  mul_463 = mul_464 = None
    squeeze_200: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
    mul_465: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0006381620931717);  squeeze_200 = None
    mul_466: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_465, 0.1);  mul_465 = None
    mul_467: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_514, 0.9)
    add_353: "f32[1024]" = torch.ops.aten.add.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
    unsqueeze_264: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_200, -1)
    unsqueeze_265: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    mul_468: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_265);  mul_462 = unsqueeze_265 = None
    unsqueeze_266: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_201, -1);  primals_201 = None
    unsqueeze_267: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    add_354: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_468, unsqueeze_267);  mul_468 = unsqueeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_355: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_354, relu_60);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_63: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_355);  add_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_67: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_63, primals_202, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_356: "i64[]" = torch.ops.aten.add.Tensor(primals_518, 1)
    var_mean_67 = torch.ops.aten.var_mean.correction(convolution_67, [0, 2, 3], correction = 0, keepdim = True)
    getitem_136: "f32[1, 2048, 1, 1]" = var_mean_67[0]
    getitem_137: "f32[1, 2048, 1, 1]" = var_mean_67[1];  var_mean_67 = None
    add_357: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05)
    rsqrt_67: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_357);  add_357 = None
    sub_67: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, getitem_137)
    mul_469: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
    squeeze_201: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
    squeeze_202: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
    mul_470: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
    mul_471: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_516, 0.9)
    add_358: "f32[2048]" = torch.ops.aten.add.Tensor(mul_470, mul_471);  mul_470 = mul_471 = None
    squeeze_203: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
    mul_472: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_203, 1.0006381620931717);  squeeze_203 = None
    mul_473: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_472, 0.1);  mul_472 = None
    mul_474: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_517, 0.9)
    add_359: "f32[2048]" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    unsqueeze_268: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1)
    unsqueeze_269: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_475: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_269);  mul_469 = unsqueeze_269 = None
    unsqueeze_270: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
    unsqueeze_271: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_360: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_475, unsqueeze_271);  mul_475 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_64: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_360);  add_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_68: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_64, primals_205, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_361: "i64[]" = torch.ops.aten.add.Tensor(primals_521, 1)
    var_mean_68 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True)
    getitem_138: "f32[1, 2048, 1, 1]" = var_mean_68[0]
    getitem_139: "f32[1, 2048, 1, 1]" = var_mean_68[1];  var_mean_68 = None
    add_362: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05)
    rsqrt_68: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_362);  add_362 = None
    sub_68: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_139)
    mul_476: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
    squeeze_204: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
    squeeze_205: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
    mul_477: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1)
    mul_478: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_519, 0.9)
    add_363: "f32[2048]" = torch.ops.aten.add.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
    squeeze_206: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
    mul_479: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_206, 1.0006381620931717);  squeeze_206 = None
    mul_480: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_479, 0.1);  mul_479 = None
    mul_481: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_520, 0.9)
    add_364: "f32[2048]" = torch.ops.aten.add.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
    unsqueeze_272: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_206, -1)
    unsqueeze_273: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    mul_482: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_273);  mul_476 = unsqueeze_273 = None
    unsqueeze_274: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_207, -1);  primals_207 = None
    unsqueeze_275: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    add_365: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_275);  mul_482 = unsqueeze_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_65: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_365);  add_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_69: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_65, primals_208, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_366: "i64[]" = torch.ops.aten.add.Tensor(primals_524, 1)
    var_mean_69 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
    getitem_140: "f32[1, 1024, 1, 1]" = var_mean_69[0]
    getitem_141: "f32[1, 1024, 1, 1]" = var_mean_69[1];  var_mean_69 = None
    add_367: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05)
    rsqrt_69: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_367);  add_367 = None
    sub_69: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_141)
    mul_483: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
    squeeze_207: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
    squeeze_208: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
    mul_484: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1)
    mul_485: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_522, 0.9)
    add_368: "f32[1024]" = torch.ops.aten.add.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
    squeeze_209: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
    mul_486: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_209, 1.0006381620931717);  squeeze_209 = None
    mul_487: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_486, 0.1);  mul_486 = None
    mul_488: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_523, 0.9)
    add_369: "f32[1024]" = torch.ops.aten.add.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
    unsqueeze_276: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_209, -1)
    unsqueeze_277: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_489: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_483, unsqueeze_277);  mul_483 = unsqueeze_277 = None
    unsqueeze_278: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_210, -1);  primals_210 = None
    unsqueeze_279: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_370: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_489, unsqueeze_279);  mul_489 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_371: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_370, relu_63);  add_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_66: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_371);  add_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_70: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_66, primals_211, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_372: "i64[]" = torch.ops.aten.add.Tensor(primals_527, 1)
    var_mean_70 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
    getitem_142: "f32[1, 2048, 1, 1]" = var_mean_70[0]
    getitem_143: "f32[1, 2048, 1, 1]" = var_mean_70[1];  var_mean_70 = None
    add_373: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05)
    rsqrt_70: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_373);  add_373 = None
    sub_70: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_143)
    mul_490: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
    squeeze_210: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
    squeeze_211: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
    mul_491: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
    mul_492: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_525, 0.9)
    add_374: "f32[2048]" = torch.ops.aten.add.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
    squeeze_212: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
    mul_493: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_212, 1.0006381620931717);  squeeze_212 = None
    mul_494: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_493, 0.1);  mul_493 = None
    mul_495: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_526, 0.9)
    add_375: "f32[2048]" = torch.ops.aten.add.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
    unsqueeze_280: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_212, -1)
    unsqueeze_281: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    mul_496: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_281);  mul_490 = unsqueeze_281 = None
    unsqueeze_282: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_213, -1);  primals_213 = None
    unsqueeze_283: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    add_376: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_496, unsqueeze_283);  mul_496 = unsqueeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_67: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_376);  add_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_71: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_67, primals_214, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_377: "i64[]" = torch.ops.aten.add.Tensor(primals_530, 1)
    var_mean_71 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
    getitem_144: "f32[1, 2048, 1, 1]" = var_mean_71[0]
    getitem_145: "f32[1, 2048, 1, 1]" = var_mean_71[1];  var_mean_71 = None
    add_378: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05)
    rsqrt_71: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_378);  add_378 = None
    sub_71: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_145)
    mul_497: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
    squeeze_213: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
    squeeze_214: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
    mul_498: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1)
    mul_499: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_528, 0.9)
    add_379: "f32[2048]" = torch.ops.aten.add.Tensor(mul_498, mul_499);  mul_498 = mul_499 = None
    squeeze_215: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
    mul_500: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_215, 1.0006381620931717);  squeeze_215 = None
    mul_501: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_500, 0.1);  mul_500 = None
    mul_502: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_529, 0.9)
    add_380: "f32[2048]" = torch.ops.aten.add.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
    unsqueeze_284: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_215, -1)
    unsqueeze_285: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_503: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_497, unsqueeze_285);  mul_497 = unsqueeze_285 = None
    unsqueeze_286: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_216, -1);  primals_216 = None
    unsqueeze_287: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_381: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_287);  mul_503 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_68: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_381);  add_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_72: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_68, primals_217, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_382: "i64[]" = torch.ops.aten.add.Tensor(primals_533, 1)
    var_mean_72 = torch.ops.aten.var_mean.correction(convolution_72, [0, 2, 3], correction = 0, keepdim = True)
    getitem_146: "f32[1, 1024, 1, 1]" = var_mean_72[0]
    getitem_147: "f32[1, 1024, 1, 1]" = var_mean_72[1];  var_mean_72 = None
    add_383: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05)
    rsqrt_72: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_383);  add_383 = None
    sub_72: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, getitem_147)
    mul_504: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
    squeeze_216: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
    squeeze_217: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
    mul_505: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1)
    mul_506: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_531, 0.9)
    add_384: "f32[1024]" = torch.ops.aten.add.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    squeeze_218: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
    mul_507: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_218, 1.0006381620931717);  squeeze_218 = None
    mul_508: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_507, 0.1);  mul_507 = None
    mul_509: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_532, 0.9)
    add_385: "f32[1024]" = torch.ops.aten.add.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
    unsqueeze_288: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_218, -1)
    unsqueeze_289: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    mul_510: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_504, unsqueeze_289);  mul_504 = unsqueeze_289 = None
    unsqueeze_290: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_219, -1);  primals_219 = None
    unsqueeze_291: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    add_386: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_510, unsqueeze_291);  mul_510 = unsqueeze_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_387: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_386, relu_66);  add_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_69: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_387);  add_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_73: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_69, primals_220, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_388: "i64[]" = torch.ops.aten.add.Tensor(primals_536, 1)
    var_mean_73 = torch.ops.aten.var_mean.correction(convolution_73, [0, 2, 3], correction = 0, keepdim = True)
    getitem_148: "f32[1, 2048, 1, 1]" = var_mean_73[0]
    getitem_149: "f32[1, 2048, 1, 1]" = var_mean_73[1];  var_mean_73 = None
    add_389: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05)
    rsqrt_73: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_389);  add_389 = None
    sub_73: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, getitem_149)
    mul_511: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
    squeeze_219: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
    squeeze_220: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
    mul_512: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
    mul_513: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_534, 0.9)
    add_390: "f32[2048]" = torch.ops.aten.add.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
    squeeze_221: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
    mul_514: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_221, 1.0006381620931717);  squeeze_221 = None
    mul_515: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_514, 0.1);  mul_514 = None
    mul_516: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_535, 0.9)
    add_391: "f32[2048]" = torch.ops.aten.add.Tensor(mul_515, mul_516);  mul_515 = mul_516 = None
    unsqueeze_292: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_221, -1)
    unsqueeze_293: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_517: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_511, unsqueeze_293);  mul_511 = unsqueeze_293 = None
    unsqueeze_294: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_222, -1);  primals_222 = None
    unsqueeze_295: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_392: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_517, unsqueeze_295);  mul_517 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_70: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_392);  add_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_74: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_70, primals_223, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_393: "i64[]" = torch.ops.aten.add.Tensor(primals_539, 1)
    var_mean_74 = torch.ops.aten.var_mean.correction(convolution_74, [0, 2, 3], correction = 0, keepdim = True)
    getitem_150: "f32[1, 2048, 1, 1]" = var_mean_74[0]
    getitem_151: "f32[1, 2048, 1, 1]" = var_mean_74[1];  var_mean_74 = None
    add_394: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
    rsqrt_74: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_394);  add_394 = None
    sub_74: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, getitem_151)
    mul_518: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
    squeeze_222: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
    squeeze_223: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
    mul_519: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
    mul_520: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_537, 0.9)
    add_395: "f32[2048]" = torch.ops.aten.add.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
    squeeze_224: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
    mul_521: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_224, 1.0006381620931717);  squeeze_224 = None
    mul_522: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_521, 0.1);  mul_521 = None
    mul_523: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_538, 0.9)
    add_396: "f32[2048]" = torch.ops.aten.add.Tensor(mul_522, mul_523);  mul_522 = mul_523 = None
    unsqueeze_296: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_224, -1)
    unsqueeze_297: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    mul_524: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_518, unsqueeze_297);  mul_518 = unsqueeze_297 = None
    unsqueeze_298: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_225, -1);  primals_225 = None
    unsqueeze_299: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    add_397: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_524, unsqueeze_299);  mul_524 = unsqueeze_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_71: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_397);  add_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_75: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_71, primals_226, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_398: "i64[]" = torch.ops.aten.add.Tensor(primals_542, 1)
    var_mean_75 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
    getitem_152: "f32[1, 1024, 1, 1]" = var_mean_75[0]
    getitem_153: "f32[1, 1024, 1, 1]" = var_mean_75[1];  var_mean_75 = None
    add_399: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05)
    rsqrt_75: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_399);  add_399 = None
    sub_75: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_153)
    mul_525: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
    squeeze_225: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
    squeeze_226: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
    mul_526: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1)
    mul_527: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_540, 0.9)
    add_400: "f32[1024]" = torch.ops.aten.add.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
    squeeze_227: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
    mul_528: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_227, 1.0006381620931717);  squeeze_227 = None
    mul_529: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_528, 0.1);  mul_528 = None
    mul_530: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_541, 0.9)
    add_401: "f32[1024]" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    unsqueeze_300: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_227, -1)
    unsqueeze_301: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_531: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_301);  mul_525 = unsqueeze_301 = None
    unsqueeze_302: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1);  primals_228 = None
    unsqueeze_303: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_402: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_531, unsqueeze_303);  mul_531 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_403: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_402, relu_69);  add_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_72: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_403);  add_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_76: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_72, primals_229, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_404: "i64[]" = torch.ops.aten.add.Tensor(primals_545, 1)
    var_mean_76 = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
    getitem_154: "f32[1, 2048, 1, 1]" = var_mean_76[0]
    getitem_155: "f32[1, 2048, 1, 1]" = var_mean_76[1];  var_mean_76 = None
    add_405: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05)
    rsqrt_76: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_405);  add_405 = None
    sub_76: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_155)
    mul_532: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
    squeeze_228: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
    squeeze_229: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
    mul_533: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1)
    mul_534: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_543, 0.9)
    add_406: "f32[2048]" = torch.ops.aten.add.Tensor(mul_533, mul_534);  mul_533 = mul_534 = None
    squeeze_230: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
    mul_535: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_230, 1.0006381620931717);  squeeze_230 = None
    mul_536: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_535, 0.1);  mul_535 = None
    mul_537: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_544, 0.9)
    add_407: "f32[2048]" = torch.ops.aten.add.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
    unsqueeze_304: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_230, -1)
    unsqueeze_305: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    mul_538: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_305);  mul_532 = unsqueeze_305 = None
    unsqueeze_306: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_231, -1);  primals_231 = None
    unsqueeze_307: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    add_408: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_538, unsqueeze_307);  mul_538 = unsqueeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_73: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_408);  add_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_77: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_73, primals_232, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_409: "i64[]" = torch.ops.aten.add.Tensor(primals_548, 1)
    var_mean_77 = torch.ops.aten.var_mean.correction(convolution_77, [0, 2, 3], correction = 0, keepdim = True)
    getitem_156: "f32[1, 2048, 1, 1]" = var_mean_77[0]
    getitem_157: "f32[1, 2048, 1, 1]" = var_mean_77[1];  var_mean_77 = None
    add_410: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
    rsqrt_77: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_410);  add_410 = None
    sub_77: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, getitem_157)
    mul_539: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
    squeeze_231: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
    squeeze_232: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
    mul_540: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
    mul_541: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_546, 0.9)
    add_411: "f32[2048]" = torch.ops.aten.add.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
    squeeze_233: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
    mul_542: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_233, 1.0006381620931717);  squeeze_233 = None
    mul_543: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_542, 0.1);  mul_542 = None
    mul_544: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_547, 0.9)
    add_412: "f32[2048]" = torch.ops.aten.add.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
    unsqueeze_308: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_233, -1)
    unsqueeze_309: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_545: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_539, unsqueeze_309);  mul_539 = unsqueeze_309 = None
    unsqueeze_310: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_234, -1);  primals_234 = None
    unsqueeze_311: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_413: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_311);  mul_545 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_74: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_413);  add_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_78: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_74, primals_235, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_414: "i64[]" = torch.ops.aten.add.Tensor(primals_551, 1)
    var_mean_78 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
    getitem_158: "f32[1, 1024, 1, 1]" = var_mean_78[0]
    getitem_159: "f32[1, 1024, 1, 1]" = var_mean_78[1];  var_mean_78 = None
    add_415: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
    rsqrt_78: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_415);  add_415 = None
    sub_78: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_159)
    mul_546: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
    squeeze_234: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
    squeeze_235: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
    mul_547: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1)
    mul_548: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_549, 0.9)
    add_416: "f32[1024]" = torch.ops.aten.add.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
    squeeze_236: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
    mul_549: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_236, 1.0006381620931717);  squeeze_236 = None
    mul_550: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_549, 0.1);  mul_549 = None
    mul_551: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_550, 0.9)
    add_417: "f32[1024]" = torch.ops.aten.add.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
    unsqueeze_312: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_236, -1)
    unsqueeze_313: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    mul_552: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_546, unsqueeze_313);  mul_546 = unsqueeze_313 = None
    unsqueeze_314: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_237, -1);  primals_237 = None
    unsqueeze_315: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    add_418: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_552, unsqueeze_315);  mul_552 = unsqueeze_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_419: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_418, relu_72);  add_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_75: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_419);  add_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_79: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_75, primals_238, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_420: "i64[]" = torch.ops.aten.add.Tensor(primals_554, 1)
    var_mean_79 = torch.ops.aten.var_mean.correction(convolution_79, [0, 2, 3], correction = 0, keepdim = True)
    getitem_160: "f32[1, 2048, 1, 1]" = var_mean_79[0]
    getitem_161: "f32[1, 2048, 1, 1]" = var_mean_79[1];  var_mean_79 = None
    add_421: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05)
    rsqrt_79: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_421);  add_421 = None
    sub_79: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, getitem_161)
    mul_553: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
    squeeze_237: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3]);  getitem_161 = None
    squeeze_238: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
    mul_554: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1)
    mul_555: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_552, 0.9)
    add_422: "f32[2048]" = torch.ops.aten.add.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
    squeeze_239: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_160, [0, 2, 3]);  getitem_160 = None
    mul_556: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_239, 1.0006381620931717);  squeeze_239 = None
    mul_557: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_556, 0.1);  mul_556 = None
    mul_558: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_553, 0.9)
    add_423: "f32[2048]" = torch.ops.aten.add.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
    unsqueeze_316: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_239, -1)
    unsqueeze_317: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_559: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_317);  mul_553 = unsqueeze_317 = None
    unsqueeze_318: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_240, -1);  primals_240 = None
    unsqueeze_319: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_424: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_559, unsqueeze_319);  mul_559 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_76: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_424);  add_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_80: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_76, primals_241, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_425: "i64[]" = torch.ops.aten.add.Tensor(primals_557, 1)
    var_mean_80 = torch.ops.aten.var_mean.correction(convolution_80, [0, 2, 3], correction = 0, keepdim = True)
    getitem_162: "f32[1, 2048, 1, 1]" = var_mean_80[0]
    getitem_163: "f32[1, 2048, 1, 1]" = var_mean_80[1];  var_mean_80 = None
    add_426: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05)
    rsqrt_80: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_426);  add_426 = None
    sub_80: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_163)
    mul_560: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = None
    squeeze_240: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
    squeeze_241: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
    mul_561: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_240, 0.1)
    mul_562: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_555, 0.9)
    add_427: "f32[2048]" = torch.ops.aten.add.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    squeeze_242: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
    mul_563: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_242, 1.0006381620931717);  squeeze_242 = None
    mul_564: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_563, 0.1);  mul_563 = None
    mul_565: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_556, 0.9)
    add_428: "f32[2048]" = torch.ops.aten.add.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_320: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_242, -1)
    unsqueeze_321: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    mul_566: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_321);  mul_560 = unsqueeze_321 = None
    unsqueeze_322: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_243, -1);  primals_243 = None
    unsqueeze_323: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    add_429: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_323);  mul_566 = unsqueeze_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_77: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_429);  add_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_81: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_77, primals_244, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_430: "i64[]" = torch.ops.aten.add.Tensor(primals_560, 1)
    var_mean_81 = torch.ops.aten.var_mean.correction(convolution_81, [0, 2, 3], correction = 0, keepdim = True)
    getitem_164: "f32[1, 1024, 1, 1]" = var_mean_81[0]
    getitem_165: "f32[1, 1024, 1, 1]" = var_mean_81[1];  var_mean_81 = None
    add_431: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-05)
    rsqrt_81: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_431);  add_431 = None
    sub_81: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, getitem_165)
    mul_567: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = None
    squeeze_243: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
    squeeze_244: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
    mul_568: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_243, 0.1)
    mul_569: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_558, 0.9)
    add_432: "f32[1024]" = torch.ops.aten.add.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
    squeeze_245: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
    mul_570: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_245, 1.0006381620931717);  squeeze_245 = None
    mul_571: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_570, 0.1);  mul_570 = None
    mul_572: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_559, 0.9)
    add_433: "f32[1024]" = torch.ops.aten.add.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
    unsqueeze_324: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_245, -1)
    unsqueeze_325: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_573: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_567, unsqueeze_325);  mul_567 = unsqueeze_325 = None
    unsqueeze_326: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_246, -1);  primals_246 = None
    unsqueeze_327: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_434: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_573, unsqueeze_327);  mul_573 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_435: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_434, relu_75);  add_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_78: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_435);  add_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_82: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_78, primals_247, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_436: "i64[]" = torch.ops.aten.add.Tensor(primals_563, 1)
    var_mean_82 = torch.ops.aten.var_mean.correction(convolution_82, [0, 2, 3], correction = 0, keepdim = True)
    getitem_166: "f32[1, 2048, 1, 1]" = var_mean_82[0]
    getitem_167: "f32[1, 2048, 1, 1]" = var_mean_82[1];  var_mean_82 = None
    add_437: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05)
    rsqrt_82: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_437);  add_437 = None
    sub_82: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, getitem_167)
    mul_574: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
    squeeze_246: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
    squeeze_247: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
    mul_575: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_246, 0.1)
    mul_576: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_561, 0.9)
    add_438: "f32[2048]" = torch.ops.aten.add.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
    squeeze_248: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
    mul_577: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_248, 1.0006381620931717);  squeeze_248 = None
    mul_578: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_577, 0.1);  mul_577 = None
    mul_579: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_562, 0.9)
    add_439: "f32[2048]" = torch.ops.aten.add.Tensor(mul_578, mul_579);  mul_578 = mul_579 = None
    unsqueeze_328: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_248, -1)
    unsqueeze_329: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    mul_580: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_329);  mul_574 = unsqueeze_329 = None
    unsqueeze_330: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_249, -1);  primals_249 = None
    unsqueeze_331: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    add_440: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_580, unsqueeze_331);  mul_580 = unsqueeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_79: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_440);  add_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_83: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_79, primals_250, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_441: "i64[]" = torch.ops.aten.add.Tensor(primals_566, 1)
    var_mean_83 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
    getitem_168: "f32[1, 2048, 1, 1]" = var_mean_83[0]
    getitem_169: "f32[1, 2048, 1, 1]" = var_mean_83[1];  var_mean_83 = None
    add_442: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-05)
    rsqrt_83: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_442);  add_442 = None
    sub_83: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_169)
    mul_581: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
    squeeze_249: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
    squeeze_250: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
    mul_582: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_249, 0.1)
    mul_583: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_564, 0.9)
    add_443: "f32[2048]" = torch.ops.aten.add.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    squeeze_251: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
    mul_584: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_251, 1.0006381620931717);  squeeze_251 = None
    mul_585: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_584, 0.1);  mul_584 = None
    mul_586: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_565, 0.9)
    add_444: "f32[2048]" = torch.ops.aten.add.Tensor(mul_585, mul_586);  mul_585 = mul_586 = None
    unsqueeze_332: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_251, -1)
    unsqueeze_333: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_587: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_581, unsqueeze_333);  mul_581 = unsqueeze_333 = None
    unsqueeze_334: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_252, -1);  primals_252 = None
    unsqueeze_335: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_445: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_335);  mul_587 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_80: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_445);  add_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_84: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_80, primals_253, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_446: "i64[]" = torch.ops.aten.add.Tensor(primals_569, 1)
    var_mean_84 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
    getitem_170: "f32[1, 1024, 1, 1]" = var_mean_84[0]
    getitem_171: "f32[1, 1024, 1, 1]" = var_mean_84[1];  var_mean_84 = None
    add_447: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-05)
    rsqrt_84: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_447);  add_447 = None
    sub_84: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_171)
    mul_588: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
    squeeze_252: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
    squeeze_253: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
    mul_589: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_252, 0.1)
    mul_590: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_567, 0.9)
    add_448: "f32[1024]" = torch.ops.aten.add.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
    squeeze_254: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_170, [0, 2, 3]);  getitem_170 = None
    mul_591: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_254, 1.0006381620931717);  squeeze_254 = None
    mul_592: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_591, 0.1);  mul_591 = None
    mul_593: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_568, 0.9)
    add_449: "f32[1024]" = torch.ops.aten.add.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
    unsqueeze_336: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_254, -1)
    unsqueeze_337: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    mul_594: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_588, unsqueeze_337);  mul_588 = unsqueeze_337 = None
    unsqueeze_338: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_255, -1);  primals_255 = None
    unsqueeze_339: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    add_450: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_594, unsqueeze_339);  mul_594 = unsqueeze_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_451: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_450, relu_78);  add_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_81: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_451);  add_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_85: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_81, primals_256, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_452: "i64[]" = torch.ops.aten.add.Tensor(primals_572, 1)
    var_mean_85 = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True)
    getitem_172: "f32[1, 2048, 1, 1]" = var_mean_85[0]
    getitem_173: "f32[1, 2048, 1, 1]" = var_mean_85[1];  var_mean_85 = None
    add_453: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05)
    rsqrt_85: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_453);  add_453 = None
    sub_85: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_173)
    mul_595: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = None
    squeeze_255: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3]);  getitem_173 = None
    squeeze_256: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
    mul_596: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_255, 0.1)
    mul_597: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_570, 0.9)
    add_454: "f32[2048]" = torch.ops.aten.add.Tensor(mul_596, mul_597);  mul_596 = mul_597 = None
    squeeze_257: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_172, [0, 2, 3]);  getitem_172 = None
    mul_598: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_257, 1.0006381620931717);  squeeze_257 = None
    mul_599: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_598, 0.1);  mul_598 = None
    mul_600: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_571, 0.9)
    add_455: "f32[2048]" = torch.ops.aten.add.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
    unsqueeze_340: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_257, -1)
    unsqueeze_341: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_601: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_341);  mul_595 = unsqueeze_341 = None
    unsqueeze_342: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_258, -1);  primals_258 = None
    unsqueeze_343: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_456: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_601, unsqueeze_343);  mul_601 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_82: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_456);  add_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_86: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_82, primals_259, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_457: "i64[]" = torch.ops.aten.add.Tensor(primals_575, 1)
    var_mean_86 = torch.ops.aten.var_mean.correction(convolution_86, [0, 2, 3], correction = 0, keepdim = True)
    getitem_174: "f32[1, 2048, 1, 1]" = var_mean_86[0]
    getitem_175: "f32[1, 2048, 1, 1]" = var_mean_86[1];  var_mean_86 = None
    add_458: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05)
    rsqrt_86: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_458);  add_458 = None
    sub_86: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, getitem_175)
    mul_602: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = None
    squeeze_258: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
    squeeze_259: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_86, [0, 2, 3]);  rsqrt_86 = None
    mul_603: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_258, 0.1)
    mul_604: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_573, 0.9)
    add_459: "f32[2048]" = torch.ops.aten.add.Tensor(mul_603, mul_604);  mul_603 = mul_604 = None
    squeeze_260: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_174, [0, 2, 3]);  getitem_174 = None
    mul_605: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_260, 1.0006381620931717);  squeeze_260 = None
    mul_606: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_605, 0.1);  mul_605 = None
    mul_607: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_574, 0.9)
    add_460: "f32[2048]" = torch.ops.aten.add.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
    unsqueeze_344: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_260, -1)
    unsqueeze_345: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    mul_608: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_602, unsqueeze_345);  mul_602 = unsqueeze_345 = None
    unsqueeze_346: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_261, -1);  primals_261 = None
    unsqueeze_347: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    add_461: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_608, unsqueeze_347);  mul_608 = unsqueeze_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_83: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_461);  add_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_87: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_83, primals_262, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_462: "i64[]" = torch.ops.aten.add.Tensor(primals_578, 1)
    var_mean_87 = torch.ops.aten.var_mean.correction(convolution_87, [0, 2, 3], correction = 0, keepdim = True)
    getitem_176: "f32[1, 1024, 1, 1]" = var_mean_87[0]
    getitem_177: "f32[1, 1024, 1, 1]" = var_mean_87[1];  var_mean_87 = None
    add_463: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05)
    rsqrt_87: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_463);  add_463 = None
    sub_87: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, getitem_177)
    mul_609: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = None
    squeeze_261: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_177, [0, 2, 3]);  getitem_177 = None
    squeeze_262: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
    mul_610: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_261, 0.1)
    mul_611: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_576, 0.9)
    add_464: "f32[1024]" = torch.ops.aten.add.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
    squeeze_263: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_176, [0, 2, 3]);  getitem_176 = None
    mul_612: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_263, 1.0006381620931717);  squeeze_263 = None
    mul_613: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_612, 0.1);  mul_612 = None
    mul_614: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_577, 0.9)
    add_465: "f32[1024]" = torch.ops.aten.add.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
    unsqueeze_348: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_263, -1)
    unsqueeze_349: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_615: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_609, unsqueeze_349);  mul_609 = unsqueeze_349 = None
    unsqueeze_350: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_264, -1);  primals_264 = None
    unsqueeze_351: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_466: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_615, unsqueeze_351);  mul_615 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_467: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_466, relu_81);  add_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_84: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_467);  add_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_88: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_84, primals_265, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_468: "i64[]" = torch.ops.aten.add.Tensor(primals_581, 1)
    var_mean_88 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
    getitem_178: "f32[1, 2048, 1, 1]" = var_mean_88[0]
    getitem_179: "f32[1, 2048, 1, 1]" = var_mean_88[1];  var_mean_88 = None
    add_469: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05)
    rsqrt_88: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_469);  add_469 = None
    sub_88: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_179)
    mul_616: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = None
    squeeze_264: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
    squeeze_265: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
    mul_617: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_264, 0.1)
    mul_618: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_579, 0.9)
    add_470: "f32[2048]" = torch.ops.aten.add.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
    squeeze_266: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_178, [0, 2, 3]);  getitem_178 = None
    mul_619: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_266, 1.0006381620931717);  squeeze_266 = None
    mul_620: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_619, 0.1);  mul_619 = None
    mul_621: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_580, 0.9)
    add_471: "f32[2048]" = torch.ops.aten.add.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
    unsqueeze_352: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_266, -1)
    unsqueeze_353: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    mul_622: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_353);  mul_616 = unsqueeze_353 = None
    unsqueeze_354: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_267, -1);  primals_267 = None
    unsqueeze_355: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    add_472: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_622, unsqueeze_355);  mul_622 = unsqueeze_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_85: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_472);  add_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_89: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_85, primals_268, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_473: "i64[]" = torch.ops.aten.add.Tensor(primals_584, 1)
    var_mean_89 = torch.ops.aten.var_mean.correction(convolution_89, [0, 2, 3], correction = 0, keepdim = True)
    getitem_180: "f32[1, 2048, 1, 1]" = var_mean_89[0]
    getitem_181: "f32[1, 2048, 1, 1]" = var_mean_89[1];  var_mean_89 = None
    add_474: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-05)
    rsqrt_89: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_474);  add_474 = None
    sub_89: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, getitem_181)
    mul_623: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = None
    squeeze_267: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_181, [0, 2, 3]);  getitem_181 = None
    squeeze_268: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_89, [0, 2, 3]);  rsqrt_89 = None
    mul_624: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_267, 0.1)
    mul_625: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_582, 0.9)
    add_475: "f32[2048]" = torch.ops.aten.add.Tensor(mul_624, mul_625);  mul_624 = mul_625 = None
    squeeze_269: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_180, [0, 2, 3]);  getitem_180 = None
    mul_626: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_269, 1.0006381620931717);  squeeze_269 = None
    mul_627: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_626, 0.1);  mul_626 = None
    mul_628: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_583, 0.9)
    add_476: "f32[2048]" = torch.ops.aten.add.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
    unsqueeze_356: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_269, -1)
    unsqueeze_357: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_629: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_623, unsqueeze_357);  mul_623 = unsqueeze_357 = None
    unsqueeze_358: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_270, -1);  primals_270 = None
    unsqueeze_359: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_477: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_629, unsqueeze_359);  mul_629 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_86: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_477);  add_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_90: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_86, primals_271, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_478: "i64[]" = torch.ops.aten.add.Tensor(primals_587, 1)
    var_mean_90 = torch.ops.aten.var_mean.correction(convolution_90, [0, 2, 3], correction = 0, keepdim = True)
    getitem_182: "f32[1, 1024, 1, 1]" = var_mean_90[0]
    getitem_183: "f32[1, 1024, 1, 1]" = var_mean_90[1];  var_mean_90 = None
    add_479: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-05)
    rsqrt_90: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_479);  add_479 = None
    sub_90: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, getitem_183)
    mul_630: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = None
    squeeze_270: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3]);  getitem_183 = None
    squeeze_271: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_90, [0, 2, 3]);  rsqrt_90 = None
    mul_631: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_270, 0.1)
    mul_632: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_585, 0.9)
    add_480: "f32[1024]" = torch.ops.aten.add.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
    squeeze_272: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_182, [0, 2, 3]);  getitem_182 = None
    mul_633: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_272, 1.0006381620931717);  squeeze_272 = None
    mul_634: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_633, 0.1);  mul_633 = None
    mul_635: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_586, 0.9)
    add_481: "f32[1024]" = torch.ops.aten.add.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
    unsqueeze_360: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_272, -1)
    unsqueeze_361: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    mul_636: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_630, unsqueeze_361);  mul_630 = unsqueeze_361 = None
    unsqueeze_362: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_273, -1);  primals_273 = None
    unsqueeze_363: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    add_482: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_636, unsqueeze_363);  mul_636 = unsqueeze_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_483: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_482, relu_84);  add_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_87: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_483);  add_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_91: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_87, primals_274, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_484: "i64[]" = torch.ops.aten.add.Tensor(primals_590, 1)
    var_mean_91 = torch.ops.aten.var_mean.correction(convolution_91, [0, 2, 3], correction = 0, keepdim = True)
    getitem_184: "f32[1, 2048, 1, 1]" = var_mean_91[0]
    getitem_185: "f32[1, 2048, 1, 1]" = var_mean_91[1];  var_mean_91 = None
    add_485: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-05)
    rsqrt_91: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_485);  add_485 = None
    sub_91: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, getitem_185)
    mul_637: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = None
    squeeze_273: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_185, [0, 2, 3]);  getitem_185 = None
    squeeze_274: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_91, [0, 2, 3]);  rsqrt_91 = None
    mul_638: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_273, 0.1)
    mul_639: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_588, 0.9)
    add_486: "f32[2048]" = torch.ops.aten.add.Tensor(mul_638, mul_639);  mul_638 = mul_639 = None
    squeeze_275: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_184, [0, 2, 3]);  getitem_184 = None
    mul_640: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_275, 1.0006381620931717);  squeeze_275 = None
    mul_641: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_640, 0.1);  mul_640 = None
    mul_642: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_589, 0.9)
    add_487: "f32[2048]" = torch.ops.aten.add.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
    unsqueeze_364: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_275, -1)
    unsqueeze_365: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_643: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_365);  mul_637 = unsqueeze_365 = None
    unsqueeze_366: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_276, -1);  primals_276 = None
    unsqueeze_367: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_488: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_643, unsqueeze_367);  mul_643 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_88: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_488);  add_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_92: "f32[8, 2048, 14, 14]" = torch.ops.aten.convolution.default(relu_88, primals_277, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_489: "i64[]" = torch.ops.aten.add.Tensor(primals_593, 1)
    var_mean_92 = torch.ops.aten.var_mean.correction(convolution_92, [0, 2, 3], correction = 0, keepdim = True)
    getitem_186: "f32[1, 2048, 1, 1]" = var_mean_92[0]
    getitem_187: "f32[1, 2048, 1, 1]" = var_mean_92[1];  var_mean_92 = None
    add_490: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05)
    rsqrt_92: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_490);  add_490 = None
    sub_92: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_187)
    mul_644: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = None
    squeeze_276: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
    squeeze_277: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_92, [0, 2, 3]);  rsqrt_92 = None
    mul_645: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_276, 0.1)
    mul_646: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_591, 0.9)
    add_491: "f32[2048]" = torch.ops.aten.add.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
    squeeze_278: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_186, [0, 2, 3]);  getitem_186 = None
    mul_647: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_278, 1.0006381620931717);  squeeze_278 = None
    mul_648: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_647, 0.1);  mul_647 = None
    mul_649: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_592, 0.9)
    add_492: "f32[2048]" = torch.ops.aten.add.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
    unsqueeze_368: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_278, -1)
    unsqueeze_369: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    mul_650: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(mul_644, unsqueeze_369);  mul_644 = unsqueeze_369 = None
    unsqueeze_370: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_279, -1);  primals_279 = None
    unsqueeze_371: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    add_493: "f32[8, 2048, 14, 14]" = torch.ops.aten.add.Tensor(mul_650, unsqueeze_371);  mul_650 = unsqueeze_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_89: "f32[8, 2048, 14, 14]" = torch.ops.aten.relu.default(add_493);  add_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_93: "f32[8, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_89, primals_280, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_494: "i64[]" = torch.ops.aten.add.Tensor(primals_596, 1)
    var_mean_93 = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True)
    getitem_188: "f32[1, 1024, 1, 1]" = var_mean_93[0]
    getitem_189: "f32[1, 1024, 1, 1]" = var_mean_93[1];  var_mean_93 = None
    add_495: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-05)
    rsqrt_93: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_495);  add_495 = None
    sub_93: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_189)
    mul_651: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = None
    squeeze_279: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
    squeeze_280: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
    mul_652: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_279, 0.1)
    mul_653: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_594, 0.9)
    add_496: "f32[1024]" = torch.ops.aten.add.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
    squeeze_281: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_188, [0, 2, 3]);  getitem_188 = None
    mul_654: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_281, 1.0006381620931717);  squeeze_281 = None
    mul_655: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_654, 0.1);  mul_654 = None
    mul_656: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_595, 0.9)
    add_497: "f32[1024]" = torch.ops.aten.add.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
    unsqueeze_372: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_281, -1)
    unsqueeze_373: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_657: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_651, unsqueeze_373);  mul_651 = unsqueeze_373 = None
    unsqueeze_374: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_282, -1);  primals_282 = None
    unsqueeze_375: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_498: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_657, unsqueeze_375);  mul_657 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_499: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_498, relu_87);  add_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_90: "f32[8, 1024, 14, 14]" = torch.ops.aten.relu.default(add_499);  add_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_94: "f32[8, 4096, 14, 14]" = torch.ops.aten.convolution.default(relu_90, primals_283, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_500: "i64[]" = torch.ops.aten.add.Tensor(primals_599, 1)
    var_mean_94 = torch.ops.aten.var_mean.correction(convolution_94, [0, 2, 3], correction = 0, keepdim = True)
    getitem_190: "f32[1, 4096, 1, 1]" = var_mean_94[0]
    getitem_191: "f32[1, 4096, 1, 1]" = var_mean_94[1];  var_mean_94 = None
    add_501: "f32[1, 4096, 1, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-05)
    rsqrt_94: "f32[1, 4096, 1, 1]" = torch.ops.aten.rsqrt.default(add_501);  add_501 = None
    sub_94: "f32[8, 4096, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, getitem_191)
    mul_658: "f32[8, 4096, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = None
    squeeze_282: "f32[4096]" = torch.ops.aten.squeeze.dims(getitem_191, [0, 2, 3]);  getitem_191 = None
    squeeze_283: "f32[4096]" = torch.ops.aten.squeeze.dims(rsqrt_94, [0, 2, 3]);  rsqrt_94 = None
    mul_659: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_282, 0.1)
    mul_660: "f32[4096]" = torch.ops.aten.mul.Tensor(primals_597, 0.9)
    add_502: "f32[4096]" = torch.ops.aten.add.Tensor(mul_659, mul_660);  mul_659 = mul_660 = None
    squeeze_284: "f32[4096]" = torch.ops.aten.squeeze.dims(getitem_190, [0, 2, 3]);  getitem_190 = None
    mul_661: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_284, 1.0006381620931717);  squeeze_284 = None
    mul_662: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_661, 0.1);  mul_661 = None
    mul_663: "f32[4096]" = torch.ops.aten.mul.Tensor(primals_598, 0.9)
    add_503: "f32[4096]" = torch.ops.aten.add.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
    unsqueeze_376: "f32[4096, 1]" = torch.ops.aten.unsqueeze.default(primals_284, -1)
    unsqueeze_377: "f32[4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    mul_664: "f32[8, 4096, 14, 14]" = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_377);  mul_658 = unsqueeze_377 = None
    unsqueeze_378: "f32[4096, 1]" = torch.ops.aten.unsqueeze.default(primals_285, -1);  primals_285 = None
    unsqueeze_379: "f32[4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    add_504: "f32[8, 4096, 14, 14]" = torch.ops.aten.add.Tensor(mul_664, unsqueeze_379);  mul_664 = unsqueeze_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_91: "f32[8, 4096, 14, 14]" = torch.ops.aten.relu.default(add_504);  add_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_95: "f32[8, 4096, 7, 7]" = torch.ops.aten.convolution.default(relu_91, primals_286, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_505: "i64[]" = torch.ops.aten.add.Tensor(primals_602, 1)
    var_mean_95 = torch.ops.aten.var_mean.correction(convolution_95, [0, 2, 3], correction = 0, keepdim = True)
    getitem_192: "f32[1, 4096, 1, 1]" = var_mean_95[0]
    getitem_193: "f32[1, 4096, 1, 1]" = var_mean_95[1];  var_mean_95 = None
    add_506: "f32[1, 4096, 1, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05)
    rsqrt_95: "f32[1, 4096, 1, 1]" = torch.ops.aten.rsqrt.default(add_506);  add_506 = None
    sub_95: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_95, getitem_193)
    mul_665: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = None
    squeeze_285: "f32[4096]" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
    squeeze_286: "f32[4096]" = torch.ops.aten.squeeze.dims(rsqrt_95, [0, 2, 3]);  rsqrt_95 = None
    mul_666: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_285, 0.1)
    mul_667: "f32[4096]" = torch.ops.aten.mul.Tensor(primals_600, 0.9)
    add_507: "f32[4096]" = torch.ops.aten.add.Tensor(mul_666, mul_667);  mul_666 = mul_667 = None
    squeeze_287: "f32[4096]" = torch.ops.aten.squeeze.dims(getitem_192, [0, 2, 3]);  getitem_192 = None
    mul_668: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_287, 1.0025575447570332);  squeeze_287 = None
    mul_669: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_668, 0.1);  mul_668 = None
    mul_670: "f32[4096]" = torch.ops.aten.mul.Tensor(primals_601, 0.9)
    add_508: "f32[4096]" = torch.ops.aten.add.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    unsqueeze_380: "f32[4096, 1]" = torch.ops.aten.unsqueeze.default(primals_287, -1)
    unsqueeze_381: "f32[4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_671: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(mul_665, unsqueeze_381);  mul_665 = unsqueeze_381 = None
    unsqueeze_382: "f32[4096, 1]" = torch.ops.aten.unsqueeze.default(primals_288, -1);  primals_288 = None
    unsqueeze_383: "f32[4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_509: "f32[8, 4096, 7, 7]" = torch.ops.aten.add.Tensor(mul_671, unsqueeze_383);  mul_671 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_92: "f32[8, 4096, 7, 7]" = torch.ops.aten.relu.default(add_509);  add_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_96: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_92, primals_289, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_510: "i64[]" = torch.ops.aten.add.Tensor(primals_605, 1)
    var_mean_96 = torch.ops.aten.var_mean.correction(convolution_96, [0, 2, 3], correction = 0, keepdim = True)
    getitem_194: "f32[1, 2048, 1, 1]" = var_mean_96[0]
    getitem_195: "f32[1, 2048, 1, 1]" = var_mean_96[1];  var_mean_96 = None
    add_511: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-05)
    rsqrt_96: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_511);  add_511 = None
    sub_96: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_96, getitem_195)
    mul_672: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = None
    squeeze_288: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3]);  getitem_195 = None
    squeeze_289: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_96, [0, 2, 3]);  rsqrt_96 = None
    mul_673: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_288, 0.1)
    mul_674: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_603, 0.9)
    add_512: "f32[2048]" = torch.ops.aten.add.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
    squeeze_290: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_194, [0, 2, 3]);  getitem_194 = None
    mul_675: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_290, 1.0025575447570332);  squeeze_290 = None
    mul_676: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_675, 0.1);  mul_675 = None
    mul_677: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_604, 0.9)
    add_513: "f32[2048]" = torch.ops.aten.add.Tensor(mul_676, mul_677);  mul_676 = mul_677 = None
    unsqueeze_384: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_290, -1)
    unsqueeze_385: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    mul_678: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_672, unsqueeze_385);  mul_672 = unsqueeze_385 = None
    unsqueeze_386: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_291, -1);  primals_291 = None
    unsqueeze_387: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    add_514: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_678, unsqueeze_387);  mul_678 = unsqueeze_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    convolution_97: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_90, primals_292, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    add_515: "i64[]" = torch.ops.aten.add.Tensor(primals_608, 1)
    var_mean_97 = torch.ops.aten.var_mean.correction(convolution_97, [0, 2, 3], correction = 0, keepdim = True)
    getitem_196: "f32[1, 2048, 1, 1]" = var_mean_97[0]
    getitem_197: "f32[1, 2048, 1, 1]" = var_mean_97[1];  var_mean_97 = None
    add_516: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-05)
    rsqrt_97: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_516);  add_516 = None
    sub_97: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_97, getitem_197)
    mul_679: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = None
    squeeze_291: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_197, [0, 2, 3]);  getitem_197 = None
    squeeze_292: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_97, [0, 2, 3]);  rsqrt_97 = None
    mul_680: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_291, 0.1)
    mul_681: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_606, 0.9)
    add_517: "f32[2048]" = torch.ops.aten.add.Tensor(mul_680, mul_681);  mul_680 = mul_681 = None
    squeeze_293: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_196, [0, 2, 3]);  getitem_196 = None
    mul_682: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_293, 1.0025575447570332);  squeeze_293 = None
    mul_683: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_682, 0.1);  mul_682 = None
    mul_684: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_607, 0.9)
    add_518: "f32[2048]" = torch.ops.aten.add.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
    unsqueeze_388: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_293, -1)
    unsqueeze_389: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_685: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_679, unsqueeze_389);  mul_679 = unsqueeze_389 = None
    unsqueeze_390: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_294, -1);  primals_294 = None
    unsqueeze_391: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_519: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_685, unsqueeze_391);  mul_685 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_520: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_514, add_519);  add_514 = add_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_93: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_520);  add_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_98: "f32[8, 4096, 7, 7]" = torch.ops.aten.convolution.default(relu_93, primals_295, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_521: "i64[]" = torch.ops.aten.add.Tensor(primals_611, 1)
    var_mean_98 = torch.ops.aten.var_mean.correction(convolution_98, [0, 2, 3], correction = 0, keepdim = True)
    getitem_198: "f32[1, 4096, 1, 1]" = var_mean_98[0]
    getitem_199: "f32[1, 4096, 1, 1]" = var_mean_98[1];  var_mean_98 = None
    add_522: "f32[1, 4096, 1, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05)
    rsqrt_98: "f32[1, 4096, 1, 1]" = torch.ops.aten.rsqrt.default(add_522);  add_522 = None
    sub_98: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, getitem_199)
    mul_686: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = None
    squeeze_294: "f32[4096]" = torch.ops.aten.squeeze.dims(getitem_199, [0, 2, 3]);  getitem_199 = None
    squeeze_295: "f32[4096]" = torch.ops.aten.squeeze.dims(rsqrt_98, [0, 2, 3]);  rsqrt_98 = None
    mul_687: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_294, 0.1)
    mul_688: "f32[4096]" = torch.ops.aten.mul.Tensor(primals_609, 0.9)
    add_523: "f32[4096]" = torch.ops.aten.add.Tensor(mul_687, mul_688);  mul_687 = mul_688 = None
    squeeze_296: "f32[4096]" = torch.ops.aten.squeeze.dims(getitem_198, [0, 2, 3]);  getitem_198 = None
    mul_689: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_296, 1.0025575447570332);  squeeze_296 = None
    mul_690: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_689, 0.1);  mul_689 = None
    mul_691: "f32[4096]" = torch.ops.aten.mul.Tensor(primals_610, 0.9)
    add_524: "f32[4096]" = torch.ops.aten.add.Tensor(mul_690, mul_691);  mul_690 = mul_691 = None
    unsqueeze_392: "f32[4096, 1]" = torch.ops.aten.unsqueeze.default(primals_296, -1)
    unsqueeze_393: "f32[4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    mul_692: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(mul_686, unsqueeze_393);  mul_686 = unsqueeze_393 = None
    unsqueeze_394: "f32[4096, 1]" = torch.ops.aten.unsqueeze.default(primals_297, -1);  primals_297 = None
    unsqueeze_395: "f32[4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    add_525: "f32[8, 4096, 7, 7]" = torch.ops.aten.add.Tensor(mul_692, unsqueeze_395);  mul_692 = unsqueeze_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_94: "f32[8, 4096, 7, 7]" = torch.ops.aten.relu.default(add_525);  add_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_99: "f32[8, 4096, 7, 7]" = torch.ops.aten.convolution.default(relu_94, primals_298, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_526: "i64[]" = torch.ops.aten.add.Tensor(primals_614, 1)
    var_mean_99 = torch.ops.aten.var_mean.correction(convolution_99, [0, 2, 3], correction = 0, keepdim = True)
    getitem_200: "f32[1, 4096, 1, 1]" = var_mean_99[0]
    getitem_201: "f32[1, 4096, 1, 1]" = var_mean_99[1];  var_mean_99 = None
    add_527: "f32[1, 4096, 1, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-05)
    rsqrt_99: "f32[1, 4096, 1, 1]" = torch.ops.aten.rsqrt.default(add_527);  add_527 = None
    sub_99: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, getitem_201)
    mul_693: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = None
    squeeze_297: "f32[4096]" = torch.ops.aten.squeeze.dims(getitem_201, [0, 2, 3]);  getitem_201 = None
    squeeze_298: "f32[4096]" = torch.ops.aten.squeeze.dims(rsqrt_99, [0, 2, 3]);  rsqrt_99 = None
    mul_694: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_297, 0.1)
    mul_695: "f32[4096]" = torch.ops.aten.mul.Tensor(primals_612, 0.9)
    add_528: "f32[4096]" = torch.ops.aten.add.Tensor(mul_694, mul_695);  mul_694 = mul_695 = None
    squeeze_299: "f32[4096]" = torch.ops.aten.squeeze.dims(getitem_200, [0, 2, 3]);  getitem_200 = None
    mul_696: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_299, 1.0025575447570332);  squeeze_299 = None
    mul_697: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_696, 0.1);  mul_696 = None
    mul_698: "f32[4096]" = torch.ops.aten.mul.Tensor(primals_613, 0.9)
    add_529: "f32[4096]" = torch.ops.aten.add.Tensor(mul_697, mul_698);  mul_697 = mul_698 = None
    unsqueeze_396: "f32[4096, 1]" = torch.ops.aten.unsqueeze.default(primals_299, -1)
    unsqueeze_397: "f32[4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_699: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(mul_693, unsqueeze_397);  mul_693 = unsqueeze_397 = None
    unsqueeze_398: "f32[4096, 1]" = torch.ops.aten.unsqueeze.default(primals_300, -1);  primals_300 = None
    unsqueeze_399: "f32[4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_530: "f32[8, 4096, 7, 7]" = torch.ops.aten.add.Tensor(mul_699, unsqueeze_399);  mul_699 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_95: "f32[8, 4096, 7, 7]" = torch.ops.aten.relu.default(add_530);  add_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_100: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_95, primals_301, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_531: "i64[]" = torch.ops.aten.add.Tensor(primals_617, 1)
    var_mean_100 = torch.ops.aten.var_mean.correction(convolution_100, [0, 2, 3], correction = 0, keepdim = True)
    getitem_202: "f32[1, 2048, 1, 1]" = var_mean_100[0]
    getitem_203: "f32[1, 2048, 1, 1]" = var_mean_100[1];  var_mean_100 = None
    add_532: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_202, 1e-05)
    rsqrt_100: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_532);  add_532 = None
    sub_100: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_100, getitem_203)
    mul_700: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = None
    squeeze_300: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_203, [0, 2, 3]);  getitem_203 = None
    squeeze_301: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_100, [0, 2, 3]);  rsqrt_100 = None
    mul_701: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_300, 0.1)
    mul_702: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_615, 0.9)
    add_533: "f32[2048]" = torch.ops.aten.add.Tensor(mul_701, mul_702);  mul_701 = mul_702 = None
    squeeze_302: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_202, [0, 2, 3]);  getitem_202 = None
    mul_703: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_302, 1.0025575447570332);  squeeze_302 = None
    mul_704: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_703, 0.1);  mul_703 = None
    mul_705: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_616, 0.9)
    add_534: "f32[2048]" = torch.ops.aten.add.Tensor(mul_704, mul_705);  mul_704 = mul_705 = None
    unsqueeze_400: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_302, -1)
    unsqueeze_401: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    mul_706: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_700, unsqueeze_401);  mul_700 = unsqueeze_401 = None
    unsqueeze_402: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_303, -1);  primals_303 = None
    unsqueeze_403: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    add_535: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_706, unsqueeze_403);  mul_706 = unsqueeze_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_536: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_535, relu_93);  add_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_96: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_536);  add_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_101: "f32[8, 4096, 7, 7]" = torch.ops.aten.convolution.default(relu_96, primals_304, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    add_537: "i64[]" = torch.ops.aten.add.Tensor(primals_620, 1)
    var_mean_101 = torch.ops.aten.var_mean.correction(convolution_101, [0, 2, 3], correction = 0, keepdim = True)
    getitem_204: "f32[1, 4096, 1, 1]" = var_mean_101[0]
    getitem_205: "f32[1, 4096, 1, 1]" = var_mean_101[1];  var_mean_101 = None
    add_538: "f32[1, 4096, 1, 1]" = torch.ops.aten.add.Tensor(getitem_204, 1e-05)
    rsqrt_101: "f32[1, 4096, 1, 1]" = torch.ops.aten.rsqrt.default(add_538);  add_538 = None
    sub_101: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_101, getitem_205)
    mul_707: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = None
    squeeze_303: "f32[4096]" = torch.ops.aten.squeeze.dims(getitem_205, [0, 2, 3]);  getitem_205 = None
    squeeze_304: "f32[4096]" = torch.ops.aten.squeeze.dims(rsqrt_101, [0, 2, 3]);  rsqrt_101 = None
    mul_708: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_303, 0.1)
    mul_709: "f32[4096]" = torch.ops.aten.mul.Tensor(primals_618, 0.9)
    add_539: "f32[4096]" = torch.ops.aten.add.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
    squeeze_305: "f32[4096]" = torch.ops.aten.squeeze.dims(getitem_204, [0, 2, 3]);  getitem_204 = None
    mul_710: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_305, 1.0025575447570332);  squeeze_305 = None
    mul_711: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_710, 0.1);  mul_710 = None
    mul_712: "f32[4096]" = torch.ops.aten.mul.Tensor(primals_619, 0.9)
    add_540: "f32[4096]" = torch.ops.aten.add.Tensor(mul_711, mul_712);  mul_711 = mul_712 = None
    unsqueeze_404: "f32[4096, 1]" = torch.ops.aten.unsqueeze.default(primals_305, -1)
    unsqueeze_405: "f32[4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_713: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(mul_707, unsqueeze_405);  mul_707 = unsqueeze_405 = None
    unsqueeze_406: "f32[4096, 1]" = torch.ops.aten.unsqueeze.default(primals_306, -1);  primals_306 = None
    unsqueeze_407: "f32[4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_541: "f32[8, 4096, 7, 7]" = torch.ops.aten.add.Tensor(mul_713, unsqueeze_407);  mul_713 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    relu_97: "f32[8, 4096, 7, 7]" = torch.ops.aten.relu.default(add_541);  add_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_102: "f32[8, 4096, 7, 7]" = torch.ops.aten.convolution.default(relu_97, primals_307, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    add_542: "i64[]" = torch.ops.aten.add.Tensor(primals_623, 1)
    var_mean_102 = torch.ops.aten.var_mean.correction(convolution_102, [0, 2, 3], correction = 0, keepdim = True)
    getitem_206: "f32[1, 4096, 1, 1]" = var_mean_102[0]
    getitem_207: "f32[1, 4096, 1, 1]" = var_mean_102[1];  var_mean_102 = None
    add_543: "f32[1, 4096, 1, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-05)
    rsqrt_102: "f32[1, 4096, 1, 1]" = torch.ops.aten.rsqrt.default(add_543);  add_543 = None
    sub_102: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, getitem_207)
    mul_714: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = None
    squeeze_306: "f32[4096]" = torch.ops.aten.squeeze.dims(getitem_207, [0, 2, 3]);  getitem_207 = None
    squeeze_307: "f32[4096]" = torch.ops.aten.squeeze.dims(rsqrt_102, [0, 2, 3]);  rsqrt_102 = None
    mul_715: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_306, 0.1)
    mul_716: "f32[4096]" = torch.ops.aten.mul.Tensor(primals_621, 0.9)
    add_544: "f32[4096]" = torch.ops.aten.add.Tensor(mul_715, mul_716);  mul_715 = mul_716 = None
    squeeze_308: "f32[4096]" = torch.ops.aten.squeeze.dims(getitem_206, [0, 2, 3]);  getitem_206 = None
    mul_717: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_308, 1.0025575447570332);  squeeze_308 = None
    mul_718: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_717, 0.1);  mul_717 = None
    mul_719: "f32[4096]" = torch.ops.aten.mul.Tensor(primals_622, 0.9)
    add_545: "f32[4096]" = torch.ops.aten.add.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
    unsqueeze_408: "f32[4096, 1]" = torch.ops.aten.unsqueeze.default(primals_308, -1)
    unsqueeze_409: "f32[4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    mul_720: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(mul_714, unsqueeze_409);  mul_714 = unsqueeze_409 = None
    unsqueeze_410: "f32[4096, 1]" = torch.ops.aten.unsqueeze.default(primals_309, -1);  primals_309 = None
    unsqueeze_411: "f32[4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    add_546: "f32[8, 4096, 7, 7]" = torch.ops.aten.add.Tensor(mul_720, unsqueeze_411);  mul_720 = unsqueeze_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    relu_98: "f32[8, 4096, 7, 7]" = torch.ops.aten.relu.default(add_546);  add_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_103: "f32[8, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_98, primals_310, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    add_547: "i64[]" = torch.ops.aten.add.Tensor(primals_626, 1)
    var_mean_103 = torch.ops.aten.var_mean.correction(convolution_103, [0, 2, 3], correction = 0, keepdim = True)
    getitem_208: "f32[1, 2048, 1, 1]" = var_mean_103[0]
    getitem_209: "f32[1, 2048, 1, 1]" = var_mean_103[1];  var_mean_103 = None
    add_548: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-05)
    rsqrt_103: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_548);  add_548 = None
    sub_103: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_103, getitem_209)
    mul_721: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = None
    squeeze_309: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_209, [0, 2, 3]);  getitem_209 = None
    squeeze_310: "f32[2048]" = torch.ops.aten.squeeze.dims(rsqrt_103, [0, 2, 3]);  rsqrt_103 = None
    mul_722: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_309, 0.1)
    mul_723: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_624, 0.9)
    add_549: "f32[2048]" = torch.ops.aten.add.Tensor(mul_722, mul_723);  mul_722 = mul_723 = None
    squeeze_311: "f32[2048]" = torch.ops.aten.squeeze.dims(getitem_208, [0, 2, 3]);  getitem_208 = None
    mul_724: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_311, 1.0025575447570332);  squeeze_311 = None
    mul_725: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_724, 0.1);  mul_724 = None
    mul_726: "f32[2048]" = torch.ops.aten.mul.Tensor(primals_625, 0.9)
    add_550: "f32[2048]" = torch.ops.aten.add.Tensor(mul_725, mul_726);  mul_725 = mul_726 = None
    unsqueeze_412: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_311, -1)
    unsqueeze_413: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_727: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_721, unsqueeze_413);  mul_721 = unsqueeze_413 = None
    unsqueeze_414: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_312, -1);  primals_312 = None
    unsqueeze_415: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_551: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_727, unsqueeze_415);  mul_727 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:201, code: x += shortcut
    add_552: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_551, relu_96);  add_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    relu_99: "f32[8, 2048, 7, 7]" = torch.ops.aten.relu.default(add_552);  add_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_99, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view: "f32[8, 2048]" = torch.ops.aten.view.default(mean, [8, 2048]);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:538, code: return x if pre_logits else self.fc(x)
    permute: "f32[2048, 1000]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_314, view, permute);  primals_314 = None
    permute_1: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[8, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[2048, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 2048, 1, 1]" = torch.ops.aten.view.default(mm, [8, 2048, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 2048, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 2048, 7, 7]);  view_2 = None
    div: "f32[8, 2048, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_101: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_99);  relu_99 = None
    alias_102: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_101);  alias_101 = None
    le: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_102, 0);  alias_102 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_416: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_309, 0);  squeeze_309 = None
    unsqueeze_417: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    sum_2: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_104: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_418)
    mul_728: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_104);  sub_104 = None
    sum_3: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_728, [0, 2, 3]);  mul_728 = None
    mul_729: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_419: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
    unsqueeze_420: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_730: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_731: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_310, squeeze_310)
    mul_732: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_730, mul_731);  mul_730 = mul_731 = None
    unsqueeze_422: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_732, 0);  mul_732 = None
    unsqueeze_423: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_733: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_310, primals_311);  primals_311 = None
    unsqueeze_425: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_733, 0);  mul_733 = None
    unsqueeze_426: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    sub_105: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_418);  convolution_103 = unsqueeze_418 = None
    mul_734: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_424);  sub_105 = unsqueeze_424 = None
    sub_106: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_734);  mul_734 = None
    sub_107: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_421);  sub_106 = unsqueeze_421 = None
    mul_735: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_427);  sub_107 = unsqueeze_427 = None
    mul_736: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_310);  sum_3 = squeeze_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_735, relu_98, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_735 = primals_310 = None
    getitem_210: "f32[8, 4096, 7, 7]" = convolution_backward[0]
    getitem_211: "f32[2048, 4096, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_104: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(relu_98);  relu_98 = None
    alias_105: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(alias_104);  alias_104 = None
    le_1: "b8[8, 4096, 7, 7]" = torch.ops.aten.le.Scalar(alias_105, 0);  alias_105 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[8, 4096, 7, 7]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, getitem_210);  le_1 = scalar_tensor_1 = getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_428: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(squeeze_306, 0);  squeeze_306 = None
    unsqueeze_429: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    sum_4: "f32[4096]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_108: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_430)
    mul_737: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_108);  sub_108 = None
    sum_5: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 2, 3]);  mul_737 = None
    mul_738: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_431: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_432: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_739: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_740: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_307, squeeze_307)
    mul_741: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
    unsqueeze_434: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
    unsqueeze_435: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_742: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_307, primals_308);  primals_308 = None
    unsqueeze_437: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
    unsqueeze_438: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    sub_109: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_430);  convolution_102 = unsqueeze_430 = None
    mul_743: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_436);  sub_109 = unsqueeze_436 = None
    sub_110: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_743);  where_1 = mul_743 = None
    sub_111: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_433);  sub_110 = unsqueeze_433 = None
    mul_744: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_439);  sub_111 = unsqueeze_439 = None
    mul_745: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_307);  sum_5 = squeeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_744, relu_97, primals_307, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_744 = primals_307 = None
    getitem_213: "f32[8, 4096, 7, 7]" = convolution_backward_1[0]
    getitem_214: "f32[4096, 128, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_107: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(relu_97);  relu_97 = None
    alias_108: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(alias_107);  alias_107 = None
    le_2: "b8[8, 4096, 7, 7]" = torch.ops.aten.le.Scalar(alias_108, 0);  alias_108 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[8, 4096, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_213);  le_2 = scalar_tensor_2 = getitem_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_440: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(squeeze_303, 0);  squeeze_303 = None
    unsqueeze_441: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    sum_6: "f32[4096]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_112: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_442)
    mul_746: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_112);  sub_112 = None
    sum_7: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_746, [0, 2, 3]);  mul_746 = None
    mul_747: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_443: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
    unsqueeze_444: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_748: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_749: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_304, squeeze_304)
    mul_750: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_748, mul_749);  mul_748 = mul_749 = None
    unsqueeze_446: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_750, 0);  mul_750 = None
    unsqueeze_447: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_751: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_304, primals_305);  primals_305 = None
    unsqueeze_449: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
    unsqueeze_450: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    sub_113: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_442);  convolution_101 = unsqueeze_442 = None
    mul_752: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_448);  sub_113 = unsqueeze_448 = None
    sub_114: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_752);  where_2 = mul_752 = None
    sub_115: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_445);  sub_114 = unsqueeze_445 = None
    mul_753: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_451);  sub_115 = unsqueeze_451 = None
    mul_754: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_304);  sum_7 = squeeze_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_753, relu_96, primals_304, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_753 = primals_304 = None
    getitem_216: "f32[8, 2048, 7, 7]" = convolution_backward_2[0]
    getitem_217: "f32[4096, 2048, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_553: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where, getitem_216);  where = getitem_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_110: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_96);  relu_96 = None
    alias_111: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_110);  alias_110 = None
    le_3: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_111, 0);  alias_111 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, add_553);  le_3 = scalar_tensor_3 = add_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_452: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_300, 0);  squeeze_300 = None
    unsqueeze_453: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    sum_8: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_116: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_454)
    mul_755: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_116);  sub_116 = None
    sum_9: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_755, [0, 2, 3]);  mul_755 = None
    mul_756: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_455: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
    unsqueeze_456: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_757: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_758: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_301, squeeze_301)
    mul_759: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_757, mul_758);  mul_757 = mul_758 = None
    unsqueeze_458: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_759, 0);  mul_759 = None
    unsqueeze_459: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_760: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_301, primals_302);  primals_302 = None
    unsqueeze_461: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
    unsqueeze_462: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    sub_117: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_454);  convolution_100 = unsqueeze_454 = None
    mul_761: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_460);  sub_117 = unsqueeze_460 = None
    sub_118: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_761);  mul_761 = None
    sub_119: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_457);  sub_118 = unsqueeze_457 = None
    mul_762: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_463);  sub_119 = unsqueeze_463 = None
    mul_763: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_301);  sum_9 = squeeze_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_762, relu_95, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_762 = primals_301 = None
    getitem_219: "f32[8, 4096, 7, 7]" = convolution_backward_3[0]
    getitem_220: "f32[2048, 4096, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_113: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(relu_95);  relu_95 = None
    alias_114: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(alias_113);  alias_113 = None
    le_4: "b8[8, 4096, 7, 7]" = torch.ops.aten.le.Scalar(alias_114, 0);  alias_114 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[8, 4096, 7, 7]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, getitem_219);  le_4 = scalar_tensor_4 = getitem_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_464: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(squeeze_297, 0);  squeeze_297 = None
    unsqueeze_465: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    sum_10: "f32[4096]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_120: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_466)
    mul_764: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_120);  sub_120 = None
    sum_11: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_764, [0, 2, 3]);  mul_764 = None
    mul_765: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_467: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
    unsqueeze_468: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_766: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_767: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_298, squeeze_298)
    mul_768: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_766, mul_767);  mul_766 = mul_767 = None
    unsqueeze_470: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_768, 0);  mul_768 = None
    unsqueeze_471: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_769: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_298, primals_299);  primals_299 = None
    unsqueeze_473: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_769, 0);  mul_769 = None
    unsqueeze_474: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    sub_121: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_466);  convolution_99 = unsqueeze_466 = None
    mul_770: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_472);  sub_121 = unsqueeze_472 = None
    sub_122: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_770);  where_4 = mul_770 = None
    sub_123: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_469);  sub_122 = unsqueeze_469 = None
    mul_771: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_475);  sub_123 = unsqueeze_475 = None
    mul_772: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_298);  sum_11 = squeeze_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_771, relu_94, primals_298, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_771 = primals_298 = None
    getitem_222: "f32[8, 4096, 7, 7]" = convolution_backward_4[0]
    getitem_223: "f32[4096, 128, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_116: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(relu_94);  relu_94 = None
    alias_117: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(alias_116);  alias_116 = None
    le_5: "b8[8, 4096, 7, 7]" = torch.ops.aten.le.Scalar(alias_117, 0);  alias_117 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[8, 4096, 7, 7]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_222);  le_5 = scalar_tensor_5 = getitem_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_476: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(squeeze_294, 0);  squeeze_294 = None
    unsqueeze_477: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    sum_12: "f32[4096]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_124: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_478)
    mul_773: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_124);  sub_124 = None
    sum_13: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_773, [0, 2, 3]);  mul_773 = None
    mul_774: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_479: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
    unsqueeze_480: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_775: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_776: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_295, squeeze_295)
    mul_777: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_775, mul_776);  mul_775 = mul_776 = None
    unsqueeze_482: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
    unsqueeze_483: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_778: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_295, primals_296);  primals_296 = None
    unsqueeze_485: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_778, 0);  mul_778 = None
    unsqueeze_486: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    sub_125: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_478);  convolution_98 = unsqueeze_478 = None
    mul_779: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_484);  sub_125 = unsqueeze_484 = None
    sub_126: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_779);  where_5 = mul_779 = None
    sub_127: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_481);  sub_126 = unsqueeze_481 = None
    mul_780: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_487);  sub_127 = unsqueeze_487 = None
    mul_781: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_295);  sum_13 = squeeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_780, relu_93, primals_295, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_780 = primals_295 = None
    getitem_225: "f32[8, 2048, 7, 7]" = convolution_backward_5[0]
    getitem_226: "f32[4096, 2048, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_554: "f32[8, 2048, 7, 7]" = torch.ops.aten.add.Tensor(where_3, getitem_225);  where_3 = getitem_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_119: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(relu_93);  relu_93 = None
    alias_120: "f32[8, 2048, 7, 7]" = torch.ops.aten.alias.default(alias_119);  alias_119 = None
    le_6: "b8[8, 2048, 7, 7]" = torch.ops.aten.le.Scalar(alias_120, 0);  alias_120 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "f32[8, 2048, 7, 7]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, add_554);  le_6 = scalar_tensor_6 = add_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    unsqueeze_488: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_291, 0);  squeeze_291 = None
    unsqueeze_489: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    sum_14: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_128: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_490)
    mul_782: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_128);  sub_128 = None
    sum_15: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_782, [0, 2, 3]);  mul_782 = None
    mul_783: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_491: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
    unsqueeze_492: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_784: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_785: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_292, squeeze_292)
    mul_786: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_784, mul_785);  mul_784 = mul_785 = None
    unsqueeze_494: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
    unsqueeze_495: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_787: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_292, primals_293);  primals_293 = None
    unsqueeze_497: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_787, 0);  mul_787 = None
    unsqueeze_498: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    sub_129: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_490);  convolution_97 = unsqueeze_490 = None
    mul_788: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_496);  sub_129 = unsqueeze_496 = None
    sub_130: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_788);  mul_788 = None
    sub_131: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_493);  sub_130 = unsqueeze_493 = None
    mul_789: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_499);  sub_131 = unsqueeze_499 = None
    mul_790: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_292);  sum_15 = squeeze_292 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_789, relu_90, primals_292, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_789 = primals_292 = None
    getitem_228: "f32[8, 1024, 14, 14]" = convolution_backward_6[0]
    getitem_229: "f32[2048, 1024, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_500: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_288, 0);  squeeze_288 = None
    unsqueeze_501: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    sum_16: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_132: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_502)
    mul_791: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_132);  sub_132 = None
    sum_17: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_791, [0, 2, 3]);  mul_791 = None
    mul_792: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_503: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
    unsqueeze_504: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_793: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_794: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_289, squeeze_289)
    mul_795: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_793, mul_794);  mul_793 = mul_794 = None
    unsqueeze_506: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
    unsqueeze_507: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_796: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_289, primals_290);  primals_290 = None
    unsqueeze_509: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_796, 0);  mul_796 = None
    unsqueeze_510: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    sub_133: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_502);  convolution_96 = unsqueeze_502 = None
    mul_797: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_508);  sub_133 = unsqueeze_508 = None
    sub_134: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_797);  where_6 = mul_797 = None
    sub_135: "f32[8, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_505);  sub_134 = unsqueeze_505 = None
    mul_798: "f32[8, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_511);  sub_135 = unsqueeze_511 = None
    mul_799: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_289);  sum_17 = squeeze_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_798, relu_92, primals_289, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_798 = primals_289 = None
    getitem_231: "f32[8, 4096, 7, 7]" = convolution_backward_7[0]
    getitem_232: "f32[2048, 4096, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_122: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(relu_92);  relu_92 = None
    alias_123: "f32[8, 4096, 7, 7]" = torch.ops.aten.alias.default(alias_122);  alias_122 = None
    le_7: "b8[8, 4096, 7, 7]" = torch.ops.aten.le.Scalar(alias_123, 0);  alias_123 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[8, 4096, 7, 7]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, getitem_231);  le_7 = scalar_tensor_7 = getitem_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_512: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(squeeze_285, 0);  squeeze_285 = None
    unsqueeze_513: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    sum_18: "f32[4096]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_136: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_514)
    mul_800: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_136);  sub_136 = None
    sum_19: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_800, [0, 2, 3]);  mul_800 = None
    mul_801: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_515: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
    unsqueeze_516: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
    unsqueeze_517: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
    mul_802: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_803: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
    mul_804: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_802, mul_803);  mul_802 = mul_803 = None
    unsqueeze_518: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
    unsqueeze_519: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
    unsqueeze_520: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
    mul_805: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_286, primals_287);  primals_287 = None
    unsqueeze_521: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_805, 0);  mul_805 = None
    unsqueeze_522: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    sub_137: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_514);  convolution_95 = unsqueeze_514 = None
    mul_806: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_520);  sub_137 = unsqueeze_520 = None
    sub_138: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_806);  where_7 = mul_806 = None
    sub_139: "f32[8, 4096, 7, 7]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_517);  sub_138 = unsqueeze_517 = None
    mul_807: "f32[8, 4096, 7, 7]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_523);  sub_139 = unsqueeze_523 = None
    mul_808: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_286);  sum_19 = squeeze_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_807, relu_91, primals_286, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_807 = primals_286 = None
    getitem_234: "f32[8, 4096, 14, 14]" = convolution_backward_8[0]
    getitem_235: "f32[4096, 128, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_125: "f32[8, 4096, 14, 14]" = torch.ops.aten.alias.default(relu_91);  relu_91 = None
    alias_126: "f32[8, 4096, 14, 14]" = torch.ops.aten.alias.default(alias_125);  alias_125 = None
    le_8: "b8[8, 4096, 14, 14]" = torch.ops.aten.le.Scalar(alias_126, 0);  alias_126 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[8, 4096, 14, 14]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, getitem_234);  le_8 = scalar_tensor_8 = getitem_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_524: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(squeeze_282, 0);  squeeze_282 = None
    unsqueeze_525: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    sum_20: "f32[4096]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_140: "f32[8, 4096, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_526)
    mul_809: "f32[8, 4096, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_140);  sub_140 = None
    sum_21: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_809, [0, 2, 3]);  mul_809 = None
    mul_810: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
    unsqueeze_527: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
    unsqueeze_528: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
    unsqueeze_529: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
    mul_811: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
    mul_812: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
    mul_813: "f32[4096]" = torch.ops.aten.mul.Tensor(mul_811, mul_812);  mul_811 = mul_812 = None
    unsqueeze_530: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_813, 0);  mul_813 = None
    unsqueeze_531: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
    unsqueeze_532: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
    mul_814: "f32[4096]" = torch.ops.aten.mul.Tensor(squeeze_283, primals_284);  primals_284 = None
    unsqueeze_533: "f32[1, 4096]" = torch.ops.aten.unsqueeze.default(mul_814, 0);  mul_814 = None
    unsqueeze_534: "f32[1, 4096, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 4096, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    sub_141: "f32[8, 4096, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_526);  convolution_94 = unsqueeze_526 = None
    mul_815: "f32[8, 4096, 14, 14]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_532);  sub_141 = unsqueeze_532 = None
    sub_142: "f32[8, 4096, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_815);  where_8 = mul_815 = None
    sub_143: "f32[8, 4096, 14, 14]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_529);  sub_142 = unsqueeze_529 = None
    mul_816: "f32[8, 4096, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_535);  sub_143 = unsqueeze_535 = None
    mul_817: "f32[4096]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_283);  sum_21 = squeeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_816, relu_90, primals_283, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_816 = primals_283 = None
    getitem_237: "f32[8, 1024, 14, 14]" = convolution_backward_9[0]
    getitem_238: "f32[4096, 1024, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_555: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(getitem_228, getitem_237);  getitem_228 = getitem_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_128: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_90);  relu_90 = None
    alias_129: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_128);  alias_128 = None
    le_9: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_129, 0);  alias_129 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, add_555);  le_9 = scalar_tensor_9 = add_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_536: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_279, 0);  squeeze_279 = None
    unsqueeze_537: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    sum_22: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_144: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_538)
    mul_818: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_144);  sub_144 = None
    sum_23: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_818, [0, 2, 3]);  mul_818 = None
    mul_819: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
    unsqueeze_539: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
    unsqueeze_540: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
    unsqueeze_541: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
    mul_820: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
    mul_821: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
    mul_822: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_820, mul_821);  mul_820 = mul_821 = None
    unsqueeze_542: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
    unsqueeze_543: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
    unsqueeze_544: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
    mul_823: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_280, primals_281);  primals_281 = None
    unsqueeze_545: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_823, 0);  mul_823 = None
    unsqueeze_546: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    sub_145: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_538);  convolution_93 = unsqueeze_538 = None
    mul_824: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_544);  sub_145 = unsqueeze_544 = None
    sub_146: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_9, mul_824);  mul_824 = None
    sub_147: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_541);  sub_146 = unsqueeze_541 = None
    mul_825: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_547);  sub_147 = unsqueeze_547 = None
    mul_826: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_280);  sum_23 = squeeze_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_825, relu_89, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_825 = primals_280 = None
    getitem_240: "f32[8, 2048, 14, 14]" = convolution_backward_10[0]
    getitem_241: "f32[1024, 2048, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_131: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_89);  relu_89 = None
    alias_132: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_131);  alias_131 = None
    le_10: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_132, 0);  alias_132 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, getitem_240);  le_10 = scalar_tensor_10 = getitem_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_548: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_276, 0);  squeeze_276 = None
    unsqueeze_549: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    sum_24: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_148: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_550)
    mul_827: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_148);  sub_148 = None
    sum_25: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_827, [0, 2, 3]);  mul_827 = None
    mul_828: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
    unsqueeze_551: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
    unsqueeze_552: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
    unsqueeze_553: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
    mul_829: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
    mul_830: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
    mul_831: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_829, mul_830);  mul_829 = mul_830 = None
    unsqueeze_554: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
    unsqueeze_555: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
    unsqueeze_556: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
    mul_832: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_277, primals_278);  primals_278 = None
    unsqueeze_557: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
    unsqueeze_558: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    sub_149: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_550);  convolution_92 = unsqueeze_550 = None
    mul_833: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_556);  sub_149 = unsqueeze_556 = None
    sub_150: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_833);  where_10 = mul_833 = None
    sub_151: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_553);  sub_150 = unsqueeze_553 = None
    mul_834: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_559);  sub_151 = unsqueeze_559 = None
    mul_835: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_277);  sum_25 = squeeze_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_834, relu_88, primals_277, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_834 = primals_277 = None
    getitem_243: "f32[8, 2048, 14, 14]" = convolution_backward_11[0]
    getitem_244: "f32[2048, 64, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_134: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_88);  relu_88 = None
    alias_135: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_134);  alias_134 = None
    le_11: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_135, 0);  alias_135 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, getitem_243);  le_11 = scalar_tensor_11 = getitem_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_560: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_273, 0);  squeeze_273 = None
    unsqueeze_561: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    sum_26: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_152: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_562)
    mul_836: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_152);  sub_152 = None
    sum_27: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 2, 3]);  mul_836 = None
    mul_837: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_26, 0.0006377551020408163)
    unsqueeze_563: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_837, 0);  mul_837 = None
    unsqueeze_564: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
    unsqueeze_565: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
    mul_838: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_27, 0.0006377551020408163)
    mul_839: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
    mul_840: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_838, mul_839);  mul_838 = mul_839 = None
    unsqueeze_566: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
    unsqueeze_567: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
    unsqueeze_568: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
    mul_841: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_274, primals_275);  primals_275 = None
    unsqueeze_569: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_841, 0);  mul_841 = None
    unsqueeze_570: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    sub_153: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_562);  convolution_91 = unsqueeze_562 = None
    mul_842: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_568);  sub_153 = unsqueeze_568 = None
    sub_154: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_842);  where_11 = mul_842 = None
    sub_155: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_565);  sub_154 = unsqueeze_565 = None
    mul_843: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_571);  sub_155 = unsqueeze_571 = None
    mul_844: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_274);  sum_27 = squeeze_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_843, relu_87, primals_274, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_843 = primals_274 = None
    getitem_246: "f32[8, 1024, 14, 14]" = convolution_backward_12[0]
    getitem_247: "f32[2048, 1024, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_556: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_9, getitem_246);  where_9 = getitem_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_137: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_87);  relu_87 = None
    alias_138: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_137);  alias_137 = None
    le_12: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_138, 0);  alias_138 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_12: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, add_556);  le_12 = scalar_tensor_12 = add_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_572: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_270, 0);  squeeze_270 = None
    unsqueeze_573: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    sum_28: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_156: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_574)
    mul_845: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_156);  sub_156 = None
    sum_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_845, [0, 2, 3]);  mul_845 = None
    mul_846: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_28, 0.0006377551020408163)
    unsqueeze_575: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
    unsqueeze_576: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_847: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, 0.0006377551020408163)
    mul_848: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
    mul_849: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_847, mul_848);  mul_847 = mul_848 = None
    unsqueeze_578: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_849, 0);  mul_849 = None
    unsqueeze_579: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_850: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_271, primals_272);  primals_272 = None
    unsqueeze_581: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_850, 0);  mul_850 = None
    unsqueeze_582: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    sub_157: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_574);  convolution_90 = unsqueeze_574 = None
    mul_851: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_580);  sub_157 = unsqueeze_580 = None
    sub_158: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_851);  mul_851 = None
    sub_159: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_577);  sub_158 = unsqueeze_577 = None
    mul_852: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_583);  sub_159 = unsqueeze_583 = None
    mul_853: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_271);  sum_29 = squeeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_852, relu_86, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_852 = primals_271 = None
    getitem_249: "f32[8, 2048, 14, 14]" = convolution_backward_13[0]
    getitem_250: "f32[1024, 2048, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_140: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_86);  relu_86 = None
    alias_141: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_140);  alias_140 = None
    le_13: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_141, 0);  alias_141 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_13: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, getitem_249);  le_13 = scalar_tensor_13 = getitem_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_584: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_267, 0);  squeeze_267 = None
    unsqueeze_585: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    sum_30: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_160: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_586)
    mul_854: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_160);  sub_160 = None
    sum_31: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_854, [0, 2, 3]);  mul_854 = None
    mul_855: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_30, 0.0006377551020408163)
    unsqueeze_587: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
    unsqueeze_588: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
    unsqueeze_589: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
    mul_856: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_31, 0.0006377551020408163)
    mul_857: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
    mul_858: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_856, mul_857);  mul_856 = mul_857 = None
    unsqueeze_590: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_858, 0);  mul_858 = None
    unsqueeze_591: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
    unsqueeze_592: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
    mul_859: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_268, primals_269);  primals_269 = None
    unsqueeze_593: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_859, 0);  mul_859 = None
    unsqueeze_594: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    sub_161: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_586);  convolution_89 = unsqueeze_586 = None
    mul_860: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_592);  sub_161 = unsqueeze_592 = None
    sub_162: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_860);  where_13 = mul_860 = None
    sub_163: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_589);  sub_162 = unsqueeze_589 = None
    mul_861: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_595);  sub_163 = unsqueeze_595 = None
    mul_862: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_268);  sum_31 = squeeze_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_861, relu_85, primals_268, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_861 = primals_268 = None
    getitem_252: "f32[8, 2048, 14, 14]" = convolution_backward_14[0]
    getitem_253: "f32[2048, 64, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_143: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_85);  relu_85 = None
    alias_144: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_143);  alias_143 = None
    le_14: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_144, 0);  alias_144 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_14: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, getitem_252);  le_14 = scalar_tensor_14 = getitem_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_596: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_264, 0);  squeeze_264 = None
    unsqueeze_597: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    sum_32: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_164: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_598)
    mul_863: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_164);  sub_164 = None
    sum_33: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_863, [0, 2, 3]);  mul_863 = None
    mul_864: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_599: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
    unsqueeze_600: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_865: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_866: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
    mul_867: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_865, mul_866);  mul_865 = mul_866 = None
    unsqueeze_602: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_867, 0);  mul_867 = None
    unsqueeze_603: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_868: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_265, primals_266);  primals_266 = None
    unsqueeze_605: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_868, 0);  mul_868 = None
    unsqueeze_606: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    sub_165: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_598);  convolution_88 = unsqueeze_598 = None
    mul_869: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_604);  sub_165 = unsqueeze_604 = None
    sub_166: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_869);  where_14 = mul_869 = None
    sub_167: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_601);  sub_166 = unsqueeze_601 = None
    mul_870: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_607);  sub_167 = unsqueeze_607 = None
    mul_871: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_265);  sum_33 = squeeze_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_870, relu_84, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_870 = primals_265 = None
    getitem_255: "f32[8, 1024, 14, 14]" = convolution_backward_15[0]
    getitem_256: "f32[2048, 1024, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_557: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_12, getitem_255);  where_12 = getitem_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_146: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_84);  relu_84 = None
    alias_147: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_146);  alias_146 = None
    le_15: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_147, 0);  alias_147 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_15: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, add_557);  le_15 = scalar_tensor_15 = add_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_608: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_261, 0);  squeeze_261 = None
    unsqueeze_609: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_168: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_610)
    mul_872: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_168);  sub_168 = None
    sum_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_872, [0, 2, 3]);  mul_872 = None
    mul_873: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    unsqueeze_611: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_873, 0);  mul_873 = None
    unsqueeze_612: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
    unsqueeze_613: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
    mul_874: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    mul_875: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
    mul_876: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_874, mul_875);  mul_874 = mul_875 = None
    unsqueeze_614: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_876, 0);  mul_876 = None
    unsqueeze_615: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
    unsqueeze_616: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
    mul_877: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_262, primals_263);  primals_263 = None
    unsqueeze_617: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_877, 0);  mul_877 = None
    unsqueeze_618: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    sub_169: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_610);  convolution_87 = unsqueeze_610 = None
    mul_878: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_616);  sub_169 = unsqueeze_616 = None
    sub_170: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_878);  mul_878 = None
    sub_171: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_613);  sub_170 = unsqueeze_613 = None
    mul_879: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_619);  sub_171 = unsqueeze_619 = None
    mul_880: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_262);  sum_35 = squeeze_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_879, relu_83, primals_262, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_879 = primals_262 = None
    getitem_258: "f32[8, 2048, 14, 14]" = convolution_backward_16[0]
    getitem_259: "f32[1024, 2048, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_149: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_83);  relu_83 = None
    alias_150: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_149);  alias_149 = None
    le_16: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_150, 0);  alias_150 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_16: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, getitem_258);  le_16 = scalar_tensor_16 = getitem_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_620: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_258, 0);  squeeze_258 = None
    unsqueeze_621: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    sum_36: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_172: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_622)
    mul_881: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_172);  sub_172 = None
    sum_37: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_881, [0, 2, 3]);  mul_881 = None
    mul_882: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    unsqueeze_623: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_882, 0);  mul_882 = None
    unsqueeze_624: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
    unsqueeze_625: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
    mul_883: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    mul_884: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
    mul_885: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_883, mul_884);  mul_883 = mul_884 = None
    unsqueeze_626: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_885, 0);  mul_885 = None
    unsqueeze_627: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
    unsqueeze_628: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
    mul_886: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_259, primals_260);  primals_260 = None
    unsqueeze_629: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_886, 0);  mul_886 = None
    unsqueeze_630: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    sub_173: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_622);  convolution_86 = unsqueeze_622 = None
    mul_887: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_628);  sub_173 = unsqueeze_628 = None
    sub_174: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_887);  where_16 = mul_887 = None
    sub_175: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_625);  sub_174 = unsqueeze_625 = None
    mul_888: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_631);  sub_175 = unsqueeze_631 = None
    mul_889: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_259);  sum_37 = squeeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_888, relu_82, primals_259, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_888 = primals_259 = None
    getitem_261: "f32[8, 2048, 14, 14]" = convolution_backward_17[0]
    getitem_262: "f32[2048, 64, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_152: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_82);  relu_82 = None
    alias_153: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_152);  alias_152 = None
    le_17: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_153, 0);  alias_153 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_17: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, getitem_261);  le_17 = scalar_tensor_17 = getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_632: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_255, 0);  squeeze_255 = None
    unsqueeze_633: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    sum_38: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_176: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_634)
    mul_890: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_176);  sub_176 = None
    sum_39: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_890, [0, 2, 3]);  mul_890 = None
    mul_891: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    unsqueeze_635: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
    unsqueeze_636: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
    unsqueeze_637: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
    mul_892: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    mul_893: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
    mul_894: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_892, mul_893);  mul_892 = mul_893 = None
    unsqueeze_638: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_894, 0);  mul_894 = None
    unsqueeze_639: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
    unsqueeze_640: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
    mul_895: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_256, primals_257);  primals_257 = None
    unsqueeze_641: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_895, 0);  mul_895 = None
    unsqueeze_642: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    sub_177: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_634);  convolution_85 = unsqueeze_634 = None
    mul_896: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_640);  sub_177 = unsqueeze_640 = None
    sub_178: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_896);  where_17 = mul_896 = None
    sub_179: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_637);  sub_178 = unsqueeze_637 = None
    mul_897: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_643);  sub_179 = unsqueeze_643 = None
    mul_898: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_256);  sum_39 = squeeze_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_897, relu_81, primals_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_897 = primals_256 = None
    getitem_264: "f32[8, 1024, 14, 14]" = convolution_backward_18[0]
    getitem_265: "f32[2048, 1024, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_558: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_15, getitem_264);  where_15 = getitem_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_155: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_81);  relu_81 = None
    alias_156: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_155);  alias_155 = None
    le_18: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_156, 0);  alias_156 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_18: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, add_558);  le_18 = scalar_tensor_18 = add_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_644: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
    unsqueeze_645: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    sum_40: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_180: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_646)
    mul_899: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_180);  sub_180 = None
    sum_41: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_899, [0, 2, 3]);  mul_899 = None
    mul_900: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    unsqueeze_647: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_900, 0);  mul_900 = None
    unsqueeze_648: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
    unsqueeze_649: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
    mul_901: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    mul_902: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
    mul_903: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_901, mul_902);  mul_901 = mul_902 = None
    unsqueeze_650: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_903, 0);  mul_903 = None
    unsqueeze_651: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
    unsqueeze_652: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
    mul_904: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_254);  primals_254 = None
    unsqueeze_653: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_904, 0);  mul_904 = None
    unsqueeze_654: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    sub_181: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_646);  convolution_84 = unsqueeze_646 = None
    mul_905: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_652);  sub_181 = unsqueeze_652 = None
    sub_182: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_905);  mul_905 = None
    sub_183: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_649);  sub_182 = unsqueeze_649 = None
    mul_906: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_655);  sub_183 = unsqueeze_655 = None
    mul_907: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_253);  sum_41 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_906, relu_80, primals_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_906 = primals_253 = None
    getitem_267: "f32[8, 2048, 14, 14]" = convolution_backward_19[0]
    getitem_268: "f32[1024, 2048, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_158: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_80);  relu_80 = None
    alias_159: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_158);  alias_158 = None
    le_19: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_159, 0);  alias_159 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_19: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, getitem_267);  le_19 = scalar_tensor_19 = getitem_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_656: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
    unsqueeze_657: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    sum_42: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_184: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_658)
    mul_908: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_184);  sub_184 = None
    sum_43: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_908, [0, 2, 3]);  mul_908 = None
    mul_909: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    unsqueeze_659: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
    unsqueeze_660: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
    unsqueeze_661: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
    mul_910: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    mul_911: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
    mul_912: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_910, mul_911);  mul_910 = mul_911 = None
    unsqueeze_662: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_912, 0);  mul_912 = None
    unsqueeze_663: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
    unsqueeze_664: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
    mul_913: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_251);  primals_251 = None
    unsqueeze_665: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_913, 0);  mul_913 = None
    unsqueeze_666: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    sub_185: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_658);  convolution_83 = unsqueeze_658 = None
    mul_914: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_664);  sub_185 = unsqueeze_664 = None
    sub_186: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_914);  where_19 = mul_914 = None
    sub_187: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_661);  sub_186 = unsqueeze_661 = None
    mul_915: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_667);  sub_187 = unsqueeze_667 = None
    mul_916: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_250);  sum_43 = squeeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_915, relu_79, primals_250, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_915 = primals_250 = None
    getitem_270: "f32[8, 2048, 14, 14]" = convolution_backward_20[0]
    getitem_271: "f32[2048, 64, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_161: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_79);  relu_79 = None
    alias_162: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_161);  alias_161 = None
    le_20: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_162, 0);  alias_162 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_20: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, getitem_270);  le_20 = scalar_tensor_20 = getitem_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_668: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
    unsqueeze_669: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    sum_44: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_188: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_670)
    mul_917: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_188);  sub_188 = None
    sum_45: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_917, [0, 2, 3]);  mul_917 = None
    mul_918: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    unsqueeze_671: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
    unsqueeze_672: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
    unsqueeze_673: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
    mul_919: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    mul_920: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
    mul_921: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_919, mul_920);  mul_919 = mul_920 = None
    unsqueeze_674: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_921, 0);  mul_921 = None
    unsqueeze_675: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
    unsqueeze_676: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
    mul_922: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_248);  primals_248 = None
    unsqueeze_677: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_922, 0);  mul_922 = None
    unsqueeze_678: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    sub_189: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_670);  convolution_82 = unsqueeze_670 = None
    mul_923: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_676);  sub_189 = unsqueeze_676 = None
    sub_190: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_923);  where_20 = mul_923 = None
    sub_191: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_673);  sub_190 = unsqueeze_673 = None
    mul_924: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_679);  sub_191 = unsqueeze_679 = None
    mul_925: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_247);  sum_45 = squeeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_924, relu_78, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_924 = primals_247 = None
    getitem_273: "f32[8, 1024, 14, 14]" = convolution_backward_21[0]
    getitem_274: "f32[2048, 1024, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_559: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_18, getitem_273);  where_18 = getitem_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_164: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_78);  relu_78 = None
    alias_165: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_164);  alias_164 = None
    le_21: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_165, 0);  alias_165 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_21: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, add_559);  le_21 = scalar_tensor_21 = add_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_680: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
    unsqueeze_681: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    sum_46: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_192: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_682)
    mul_926: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_192);  sub_192 = None
    sum_47: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_926, [0, 2, 3]);  mul_926 = None
    mul_927: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    unsqueeze_683: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
    unsqueeze_684: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
    unsqueeze_685: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
    mul_928: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_929: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
    mul_930: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_928, mul_929);  mul_928 = mul_929 = None
    unsqueeze_686: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_930, 0);  mul_930 = None
    unsqueeze_687: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
    unsqueeze_688: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
    mul_931: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_245);  primals_245 = None
    unsqueeze_689: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
    unsqueeze_690: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    sub_193: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_682);  convolution_81 = unsqueeze_682 = None
    mul_932: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_688);  sub_193 = unsqueeze_688 = None
    sub_194: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_932);  mul_932 = None
    sub_195: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_685);  sub_194 = unsqueeze_685 = None
    mul_933: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_691);  sub_195 = unsqueeze_691 = None
    mul_934: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_244);  sum_47 = squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_933, relu_77, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_933 = primals_244 = None
    getitem_276: "f32[8, 2048, 14, 14]" = convolution_backward_22[0]
    getitem_277: "f32[1024, 2048, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_167: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_77);  relu_77 = None
    alias_168: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_167);  alias_167 = None
    le_22: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_168, 0);  alias_168 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_22: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, getitem_276);  le_22 = scalar_tensor_22 = getitem_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_692: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
    unsqueeze_693: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    sum_48: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_196: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_694)
    mul_935: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_196);  sub_196 = None
    sum_49: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_935, [0, 2, 3]);  mul_935 = None
    mul_936: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_695: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
    unsqueeze_696: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
    unsqueeze_697: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
    mul_937: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_938: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
    mul_939: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_937, mul_938);  mul_937 = mul_938 = None
    unsqueeze_698: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_939, 0);  mul_939 = None
    unsqueeze_699: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
    unsqueeze_700: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
    mul_940: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_242);  primals_242 = None
    unsqueeze_701: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_940, 0);  mul_940 = None
    unsqueeze_702: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    sub_197: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_694);  convolution_80 = unsqueeze_694 = None
    mul_941: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_700);  sub_197 = unsqueeze_700 = None
    sub_198: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_22, mul_941);  where_22 = mul_941 = None
    sub_199: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_697);  sub_198 = unsqueeze_697 = None
    mul_942: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_703);  sub_199 = unsqueeze_703 = None
    mul_943: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_241);  sum_49 = squeeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_942, relu_76, primals_241, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_942 = primals_241 = None
    getitem_279: "f32[8, 2048, 14, 14]" = convolution_backward_23[0]
    getitem_280: "f32[2048, 64, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_170: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_76);  relu_76 = None
    alias_171: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_170);  alias_170 = None
    le_23: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_171, 0);  alias_171 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_23: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, getitem_279);  le_23 = scalar_tensor_23 = getitem_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_704: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
    unsqueeze_705: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
    unsqueeze_706: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
    sum_50: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_200: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_706)
    mul_944: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_200);  sub_200 = None
    sum_51: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_944, [0, 2, 3]);  mul_944 = None
    mul_945: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    unsqueeze_707: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_945, 0);  mul_945 = None
    unsqueeze_708: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
    unsqueeze_709: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
    mul_946: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_947: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_948: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_946, mul_947);  mul_946 = mul_947 = None
    unsqueeze_710: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_948, 0);  mul_948 = None
    unsqueeze_711: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
    unsqueeze_712: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
    mul_949: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_239);  primals_239 = None
    unsqueeze_713: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_949, 0);  mul_949 = None
    unsqueeze_714: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    sub_201: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_706);  convolution_79 = unsqueeze_706 = None
    mul_950: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_712);  sub_201 = unsqueeze_712 = None
    sub_202: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_950);  where_23 = mul_950 = None
    sub_203: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_709);  sub_202 = unsqueeze_709 = None
    mul_951: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_715);  sub_203 = unsqueeze_715 = None
    mul_952: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_238);  sum_51 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_951, relu_75, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_951 = primals_238 = None
    getitem_282: "f32[8, 1024, 14, 14]" = convolution_backward_24[0]
    getitem_283: "f32[2048, 1024, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_560: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_21, getitem_282);  where_21 = getitem_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_173: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_75);  relu_75 = None
    alias_174: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_173);  alias_173 = None
    le_24: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_174, 0);  alias_174 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_24: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, add_560);  le_24 = scalar_tensor_24 = add_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_716: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
    unsqueeze_717: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
    unsqueeze_718: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
    sum_52: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_204: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_718)
    mul_953: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_204);  sub_204 = None
    sum_53: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_953, [0, 2, 3]);  mul_953 = None
    mul_954: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    unsqueeze_719: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
    unsqueeze_720: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
    unsqueeze_721: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
    mul_955: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    mul_956: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_957: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_955, mul_956);  mul_955 = mul_956 = None
    unsqueeze_722: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_957, 0);  mul_957 = None
    unsqueeze_723: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
    unsqueeze_724: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
    mul_958: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_236);  primals_236 = None
    unsqueeze_725: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_958, 0);  mul_958 = None
    unsqueeze_726: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    sub_205: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_718);  convolution_78 = unsqueeze_718 = None
    mul_959: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_724);  sub_205 = unsqueeze_724 = None
    sub_206: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_959);  mul_959 = None
    sub_207: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_721);  sub_206 = unsqueeze_721 = None
    mul_960: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_727);  sub_207 = unsqueeze_727 = None
    mul_961: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_235);  sum_53 = squeeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_960, relu_74, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_960 = primals_235 = None
    getitem_285: "f32[8, 2048, 14, 14]" = convolution_backward_25[0]
    getitem_286: "f32[1024, 2048, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_176: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_74);  relu_74 = None
    alias_177: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_176);  alias_176 = None
    le_25: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_177, 0);  alias_177 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_25: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, getitem_285);  le_25 = scalar_tensor_25 = getitem_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_728: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
    unsqueeze_729: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
    unsqueeze_730: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
    sum_54: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_208: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_730)
    mul_962: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_208);  sub_208 = None
    sum_55: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_962, [0, 2, 3]);  mul_962 = None
    mul_963: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    unsqueeze_731: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
    unsqueeze_732: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
    unsqueeze_733: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
    mul_964: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    mul_965: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_966: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_964, mul_965);  mul_964 = mul_965 = None
    unsqueeze_734: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_966, 0);  mul_966 = None
    unsqueeze_735: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
    unsqueeze_736: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
    mul_967: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_233);  primals_233 = None
    unsqueeze_737: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_967, 0);  mul_967 = None
    unsqueeze_738: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    sub_209: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_730);  convolution_77 = unsqueeze_730 = None
    mul_968: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_736);  sub_209 = unsqueeze_736 = None
    sub_210: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_968);  where_25 = mul_968 = None
    sub_211: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_733);  sub_210 = unsqueeze_733 = None
    mul_969: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_739);  sub_211 = unsqueeze_739 = None
    mul_970: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_232);  sum_55 = squeeze_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_969, relu_73, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_969 = primals_232 = None
    getitem_288: "f32[8, 2048, 14, 14]" = convolution_backward_26[0]
    getitem_289: "f32[2048, 64, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_179: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_73);  relu_73 = None
    alias_180: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_179);  alias_179 = None
    le_26: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_180, 0);  alias_180 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_26: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_26, scalar_tensor_26, getitem_288);  le_26 = scalar_tensor_26 = getitem_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_740: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
    unsqueeze_741: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
    unsqueeze_742: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
    sum_56: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_212: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_742)
    mul_971: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_212);  sub_212 = None
    sum_57: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_971, [0, 2, 3]);  mul_971 = None
    mul_972: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_743: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
    unsqueeze_744: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
    unsqueeze_745: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
    mul_973: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_974: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_975: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_973, mul_974);  mul_973 = mul_974 = None
    unsqueeze_746: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_975, 0);  mul_975 = None
    unsqueeze_747: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
    unsqueeze_748: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
    mul_976: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_230);  primals_230 = None
    unsqueeze_749: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_976, 0);  mul_976 = None
    unsqueeze_750: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    sub_213: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_742);  convolution_76 = unsqueeze_742 = None
    mul_977: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_748);  sub_213 = unsqueeze_748 = None
    sub_214: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_26, mul_977);  where_26 = mul_977 = None
    sub_215: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_745);  sub_214 = unsqueeze_745 = None
    mul_978: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_751);  sub_215 = unsqueeze_751 = None
    mul_979: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_229);  sum_57 = squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_978, relu_72, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_978 = primals_229 = None
    getitem_291: "f32[8, 1024, 14, 14]" = convolution_backward_27[0]
    getitem_292: "f32[2048, 1024, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_561: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_24, getitem_291);  where_24 = getitem_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_182: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_72);  relu_72 = None
    alias_183: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_182);  alias_182 = None
    le_27: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_183, 0);  alias_183 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_27: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_27, scalar_tensor_27, add_561);  le_27 = scalar_tensor_27 = add_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_752: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
    unsqueeze_753: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
    unsqueeze_754: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
    sum_58: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_216: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_754)
    mul_980: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_216);  sub_216 = None
    sum_59: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_980, [0, 2, 3]);  mul_980 = None
    mul_981: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_755: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
    unsqueeze_756: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
    unsqueeze_757: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
    mul_982: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_983: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_984: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_982, mul_983);  mul_982 = mul_983 = None
    unsqueeze_758: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_984, 0);  mul_984 = None
    unsqueeze_759: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
    unsqueeze_760: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
    mul_985: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_227);  primals_227 = None
    unsqueeze_761: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_985, 0);  mul_985 = None
    unsqueeze_762: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    sub_217: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_754);  convolution_75 = unsqueeze_754 = None
    mul_986: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_760);  sub_217 = unsqueeze_760 = None
    sub_218: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_986);  mul_986 = None
    sub_219: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_757);  sub_218 = unsqueeze_757 = None
    mul_987: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_763);  sub_219 = unsqueeze_763 = None
    mul_988: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_226);  sum_59 = squeeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_987, relu_71, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_987 = primals_226 = None
    getitem_294: "f32[8, 2048, 14, 14]" = convolution_backward_28[0]
    getitem_295: "f32[1024, 2048, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_185: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_71);  relu_71 = None
    alias_186: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_185);  alias_185 = None
    le_28: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_186, 0);  alias_186 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_28: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_28, scalar_tensor_28, getitem_294);  le_28 = scalar_tensor_28 = getitem_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_764: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
    unsqueeze_765: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
    unsqueeze_766: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
    sum_60: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_220: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_766)
    mul_989: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_220);  sub_220 = None
    sum_61: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_989, [0, 2, 3]);  mul_989 = None
    mul_990: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_767: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_990, 0);  mul_990 = None
    unsqueeze_768: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
    unsqueeze_769: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
    mul_991: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_992: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_993: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_991, mul_992);  mul_991 = mul_992 = None
    unsqueeze_770: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_993, 0);  mul_993 = None
    unsqueeze_771: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
    unsqueeze_772: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
    mul_994: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_224);  primals_224 = None
    unsqueeze_773: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_994, 0);  mul_994 = None
    unsqueeze_774: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    sub_221: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_766);  convolution_74 = unsqueeze_766 = None
    mul_995: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_772);  sub_221 = unsqueeze_772 = None
    sub_222: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_28, mul_995);  where_28 = mul_995 = None
    sub_223: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_769);  sub_222 = unsqueeze_769 = None
    mul_996: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_775);  sub_223 = unsqueeze_775 = None
    mul_997: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_223);  sum_61 = squeeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_996, relu_70, primals_223, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_996 = primals_223 = None
    getitem_297: "f32[8, 2048, 14, 14]" = convolution_backward_29[0]
    getitem_298: "f32[2048, 64, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_188: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_70);  relu_70 = None
    alias_189: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_188);  alias_188 = None
    le_29: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_189, 0);  alias_189 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_29: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_29, scalar_tensor_29, getitem_297);  le_29 = scalar_tensor_29 = getitem_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_776: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
    unsqueeze_777: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
    unsqueeze_778: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
    sum_62: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_224: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_778)
    mul_998: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_29, sub_224);  sub_224 = None
    sum_63: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_998, [0, 2, 3]);  mul_998 = None
    mul_999: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_62, 0.0006377551020408163)
    unsqueeze_779: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_999, 0);  mul_999 = None
    unsqueeze_780: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
    unsqueeze_781: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
    mul_1000: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    mul_1001: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_1002: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1000, mul_1001);  mul_1000 = mul_1001 = None
    unsqueeze_782: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1002, 0);  mul_1002 = None
    unsqueeze_783: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
    unsqueeze_784: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
    mul_1003: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_221);  primals_221 = None
    unsqueeze_785: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1003, 0);  mul_1003 = None
    unsqueeze_786: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    sub_225: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_778);  convolution_73 = unsqueeze_778 = None
    mul_1004: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_784);  sub_225 = unsqueeze_784 = None
    sub_226: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_29, mul_1004);  where_29 = mul_1004 = None
    sub_227: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_781);  sub_226 = unsqueeze_781 = None
    mul_1005: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_787);  sub_227 = unsqueeze_787 = None
    mul_1006: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_220);  sum_63 = squeeze_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_1005, relu_69, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1005 = primals_220 = None
    getitem_300: "f32[8, 1024, 14, 14]" = convolution_backward_30[0]
    getitem_301: "f32[2048, 1024, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_562: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_27, getitem_300);  where_27 = getitem_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_191: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_69);  relu_69 = None
    alias_192: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_191);  alias_191 = None
    le_30: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_192, 0);  alias_192 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_30: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_30, scalar_tensor_30, add_562);  le_30 = scalar_tensor_30 = add_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_788: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
    unsqueeze_789: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
    unsqueeze_790: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
    sum_64: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_228: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_790)
    mul_1007: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_228);  sub_228 = None
    sum_65: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1007, [0, 2, 3]);  mul_1007 = None
    mul_1008: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    unsqueeze_791: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
    unsqueeze_792: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
    unsqueeze_793: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
    mul_1009: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    mul_1010: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_1011: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1009, mul_1010);  mul_1009 = mul_1010 = None
    unsqueeze_794: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1011, 0);  mul_1011 = None
    unsqueeze_795: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
    unsqueeze_796: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
    mul_1012: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_218);  primals_218 = None
    unsqueeze_797: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1012, 0);  mul_1012 = None
    unsqueeze_798: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    sub_229: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_790);  convolution_72 = unsqueeze_790 = None
    mul_1013: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_796);  sub_229 = unsqueeze_796 = None
    sub_230: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_30, mul_1013);  mul_1013 = None
    sub_231: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_793);  sub_230 = unsqueeze_793 = None
    mul_1014: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_799);  sub_231 = unsqueeze_799 = None
    mul_1015: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_217);  sum_65 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_1014, relu_68, primals_217, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1014 = primals_217 = None
    getitem_303: "f32[8, 2048, 14, 14]" = convolution_backward_31[0]
    getitem_304: "f32[1024, 2048, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_194: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_68);  relu_68 = None
    alias_195: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_194);  alias_194 = None
    le_31: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_195, 0);  alias_195 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_31: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_31, scalar_tensor_31, getitem_303);  le_31 = scalar_tensor_31 = getitem_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_800: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
    unsqueeze_801: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
    unsqueeze_802: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
    sum_66: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_232: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_802)
    mul_1016: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_232);  sub_232 = None
    sum_67: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1016, [0, 2, 3]);  mul_1016 = None
    mul_1017: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    unsqueeze_803: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1017, 0);  mul_1017 = None
    unsqueeze_804: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
    unsqueeze_805: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
    mul_1018: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    mul_1019: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_1020: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1018, mul_1019);  mul_1018 = mul_1019 = None
    unsqueeze_806: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1020, 0);  mul_1020 = None
    unsqueeze_807: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
    unsqueeze_808: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
    mul_1021: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_215);  primals_215 = None
    unsqueeze_809: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1021, 0);  mul_1021 = None
    unsqueeze_810: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    sub_233: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_802);  convolution_71 = unsqueeze_802 = None
    mul_1022: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_808);  sub_233 = unsqueeze_808 = None
    sub_234: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_1022);  where_31 = mul_1022 = None
    sub_235: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_805);  sub_234 = unsqueeze_805 = None
    mul_1023: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_811);  sub_235 = unsqueeze_811 = None
    mul_1024: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_214);  sum_67 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_1023, relu_67, primals_214, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1023 = primals_214 = None
    getitem_306: "f32[8, 2048, 14, 14]" = convolution_backward_32[0]
    getitem_307: "f32[2048, 64, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_197: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_67);  relu_67 = None
    alias_198: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_197);  alias_197 = None
    le_32: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_198, 0);  alias_198 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_32: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_32, scalar_tensor_32, getitem_306);  le_32 = scalar_tensor_32 = getitem_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_812: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
    unsqueeze_813: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
    unsqueeze_814: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
    sum_68: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_236: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_814)
    mul_1025: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_236);  sub_236 = None
    sum_69: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1025, [0, 2, 3]);  mul_1025 = None
    mul_1026: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    unsqueeze_815: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1026, 0);  mul_1026 = None
    unsqueeze_816: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
    unsqueeze_817: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
    mul_1027: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    mul_1028: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_1029: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1027, mul_1028);  mul_1027 = mul_1028 = None
    unsqueeze_818: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1029, 0);  mul_1029 = None
    unsqueeze_819: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
    unsqueeze_820: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
    mul_1030: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_212);  primals_212 = None
    unsqueeze_821: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1030, 0);  mul_1030 = None
    unsqueeze_822: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    sub_237: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_814);  convolution_70 = unsqueeze_814 = None
    mul_1031: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_820);  sub_237 = unsqueeze_820 = None
    sub_238: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_32, mul_1031);  where_32 = mul_1031 = None
    sub_239: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_817);  sub_238 = unsqueeze_817 = None
    mul_1032: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_823);  sub_239 = unsqueeze_823 = None
    mul_1033: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_211);  sum_69 = squeeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_1032, relu_66, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1032 = primals_211 = None
    getitem_309: "f32[8, 1024, 14, 14]" = convolution_backward_33[0]
    getitem_310: "f32[2048, 1024, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_563: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_30, getitem_309);  where_30 = getitem_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_200: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_66);  relu_66 = None
    alias_201: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_200);  alias_200 = None
    le_33: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_201, 0);  alias_201 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_33: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_33, scalar_tensor_33, add_563);  le_33 = scalar_tensor_33 = add_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_824: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
    unsqueeze_825: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
    unsqueeze_826: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
    sum_70: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_240: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_826)
    mul_1034: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_240);  sub_240 = None
    sum_71: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1034, [0, 2, 3]);  mul_1034 = None
    mul_1035: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_827: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1035, 0);  mul_1035 = None
    unsqueeze_828: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
    unsqueeze_829: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
    mul_1036: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_1037: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_1038: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1036, mul_1037);  mul_1036 = mul_1037 = None
    unsqueeze_830: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1038, 0);  mul_1038 = None
    unsqueeze_831: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
    unsqueeze_832: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
    mul_1039: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_209);  primals_209 = None
    unsqueeze_833: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1039, 0);  mul_1039 = None
    unsqueeze_834: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    sub_241: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_826);  convolution_69 = unsqueeze_826 = None
    mul_1040: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_832);  sub_241 = unsqueeze_832 = None
    sub_242: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_33, mul_1040);  mul_1040 = None
    sub_243: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_829);  sub_242 = unsqueeze_829 = None
    mul_1041: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_835);  sub_243 = unsqueeze_835 = None
    mul_1042: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_208);  sum_71 = squeeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_1041, relu_65, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1041 = primals_208 = None
    getitem_312: "f32[8, 2048, 14, 14]" = convolution_backward_34[0]
    getitem_313: "f32[1024, 2048, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_203: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_65);  relu_65 = None
    alias_204: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_203);  alias_203 = None
    le_34: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_204, 0);  alias_204 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_34: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_34, scalar_tensor_34, getitem_312);  le_34 = scalar_tensor_34 = getitem_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_836: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
    unsqueeze_837: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
    unsqueeze_838: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
    sum_72: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_244: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_838)
    mul_1043: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_244);  sub_244 = None
    sum_73: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1043, [0, 2, 3]);  mul_1043 = None
    mul_1044: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_839: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1044, 0);  mul_1044 = None
    unsqueeze_840: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
    unsqueeze_841: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
    mul_1045: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_1046: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_1047: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1045, mul_1046);  mul_1045 = mul_1046 = None
    unsqueeze_842: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1047, 0);  mul_1047 = None
    unsqueeze_843: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
    unsqueeze_844: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
    mul_1048: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_206);  primals_206 = None
    unsqueeze_845: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1048, 0);  mul_1048 = None
    unsqueeze_846: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    sub_245: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_838);  convolution_68 = unsqueeze_838 = None
    mul_1049: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_844);  sub_245 = unsqueeze_844 = None
    sub_246: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_34, mul_1049);  where_34 = mul_1049 = None
    sub_247: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_841);  sub_246 = unsqueeze_841 = None
    mul_1050: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_847);  sub_247 = unsqueeze_847 = None
    mul_1051: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_205);  sum_73 = squeeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_1050, relu_64, primals_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1050 = primals_205 = None
    getitem_315: "f32[8, 2048, 14, 14]" = convolution_backward_35[0]
    getitem_316: "f32[2048, 64, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_206: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_64);  relu_64 = None
    alias_207: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_206);  alias_206 = None
    le_35: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_207, 0);  alias_207 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_35: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_35, scalar_tensor_35, getitem_315);  le_35 = scalar_tensor_35 = getitem_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_848: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
    unsqueeze_849: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
    unsqueeze_850: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
    sum_74: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_248: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_850)
    mul_1052: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_248);  sub_248 = None
    sum_75: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1052, [0, 2, 3]);  mul_1052 = None
    mul_1053: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_851: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1053, 0);  mul_1053 = None
    unsqueeze_852: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 2);  unsqueeze_851 = None
    unsqueeze_853: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 3);  unsqueeze_852 = None
    mul_1054: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_1055: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_1056: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1054, mul_1055);  mul_1054 = mul_1055 = None
    unsqueeze_854: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1056, 0);  mul_1056 = None
    unsqueeze_855: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 2);  unsqueeze_854 = None
    unsqueeze_856: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 3);  unsqueeze_855 = None
    mul_1057: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_203);  primals_203 = None
    unsqueeze_857: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1057, 0);  mul_1057 = None
    unsqueeze_858: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    sub_249: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_850);  convolution_67 = unsqueeze_850 = None
    mul_1058: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_856);  sub_249 = unsqueeze_856 = None
    sub_250: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_1058);  where_35 = mul_1058 = None
    sub_251: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_853);  sub_250 = unsqueeze_853 = None
    mul_1059: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_859);  sub_251 = unsqueeze_859 = None
    mul_1060: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_202);  sum_75 = squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_1059, relu_63, primals_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1059 = primals_202 = None
    getitem_318: "f32[8, 1024, 14, 14]" = convolution_backward_36[0]
    getitem_319: "f32[2048, 1024, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_564: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_33, getitem_318);  where_33 = getitem_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_209: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_63);  relu_63 = None
    alias_210: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_209);  alias_209 = None
    le_36: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_210, 0);  alias_210 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_36: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_36, scalar_tensor_36, add_564);  le_36 = scalar_tensor_36 = add_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_860: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
    unsqueeze_861: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
    unsqueeze_862: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None
    sum_76: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_252: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_862)
    mul_1061: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_252);  sub_252 = None
    sum_77: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1061, [0, 2, 3]);  mul_1061 = None
    mul_1062: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    unsqueeze_863: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1062, 0);  mul_1062 = None
    unsqueeze_864: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 2);  unsqueeze_863 = None
    unsqueeze_865: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 3);  unsqueeze_864 = None
    mul_1063: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    mul_1064: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_1065: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1063, mul_1064);  mul_1063 = mul_1064 = None
    unsqueeze_866: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1065, 0);  mul_1065 = None
    unsqueeze_867: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 2);  unsqueeze_866 = None
    unsqueeze_868: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 3);  unsqueeze_867 = None
    mul_1066: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_200);  primals_200 = None
    unsqueeze_869: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1066, 0);  mul_1066 = None
    unsqueeze_870: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    sub_253: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_862);  convolution_66 = unsqueeze_862 = None
    mul_1067: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_868);  sub_253 = unsqueeze_868 = None
    sub_254: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_36, mul_1067);  mul_1067 = None
    sub_255: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_865);  sub_254 = unsqueeze_865 = None
    mul_1068: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_871);  sub_255 = unsqueeze_871 = None
    mul_1069: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_199);  sum_77 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_1068, relu_62, primals_199, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1068 = primals_199 = None
    getitem_321: "f32[8, 2048, 14, 14]" = convolution_backward_37[0]
    getitem_322: "f32[1024, 2048, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_212: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_62);  relu_62 = None
    alias_213: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_212);  alias_212 = None
    le_37: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_213, 0);  alias_213 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_37: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_37, scalar_tensor_37, getitem_321);  le_37 = scalar_tensor_37 = getitem_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_872: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
    unsqueeze_873: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
    unsqueeze_874: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None
    sum_78: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_256: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_874)
    mul_1070: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_256);  sub_256 = None
    sum_79: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1070, [0, 2, 3]);  mul_1070 = None
    mul_1071: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    unsqueeze_875: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1071, 0);  mul_1071 = None
    unsqueeze_876: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 2);  unsqueeze_875 = None
    unsqueeze_877: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 3);  unsqueeze_876 = None
    mul_1072: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    mul_1073: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_1074: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1072, mul_1073);  mul_1072 = mul_1073 = None
    unsqueeze_878: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1074, 0);  mul_1074 = None
    unsqueeze_879: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 2);  unsqueeze_878 = None
    unsqueeze_880: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 3);  unsqueeze_879 = None
    mul_1075: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_197);  primals_197 = None
    unsqueeze_881: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1075, 0);  mul_1075 = None
    unsqueeze_882: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    sub_257: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_874);  convolution_65 = unsqueeze_874 = None
    mul_1076: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_880);  sub_257 = unsqueeze_880 = None
    sub_258: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_37, mul_1076);  where_37 = mul_1076 = None
    sub_259: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_877);  sub_258 = unsqueeze_877 = None
    mul_1077: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_883);  sub_259 = unsqueeze_883 = None
    mul_1078: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_196);  sum_79 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_1077, relu_61, primals_196, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1077 = primals_196 = None
    getitem_324: "f32[8, 2048, 14, 14]" = convolution_backward_38[0]
    getitem_325: "f32[2048, 64, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_215: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_61);  relu_61 = None
    alias_216: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_215);  alias_215 = None
    le_38: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_216, 0);  alias_216 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_38: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_38, scalar_tensor_38, getitem_324);  le_38 = scalar_tensor_38 = getitem_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_884: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_885: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
    unsqueeze_886: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
    sum_80: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_260: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_886)
    mul_1079: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_260);  sub_260 = None
    sum_81: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1079, [0, 2, 3]);  mul_1079 = None
    mul_1080: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_80, 0.0006377551020408163)
    unsqueeze_887: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1080, 0);  mul_1080 = None
    unsqueeze_888: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 2);  unsqueeze_887 = None
    unsqueeze_889: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 3);  unsqueeze_888 = None
    mul_1081: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    mul_1082: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_1083: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1081, mul_1082);  mul_1081 = mul_1082 = None
    unsqueeze_890: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1083, 0);  mul_1083 = None
    unsqueeze_891: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 2);  unsqueeze_890 = None
    unsqueeze_892: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 3);  unsqueeze_891 = None
    mul_1084: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_194);  primals_194 = None
    unsqueeze_893: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1084, 0);  mul_1084 = None
    unsqueeze_894: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    sub_261: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_886);  convolution_64 = unsqueeze_886 = None
    mul_1085: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_892);  sub_261 = unsqueeze_892 = None
    sub_262: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_38, mul_1085);  where_38 = mul_1085 = None
    sub_263: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_889);  sub_262 = unsqueeze_889 = None
    mul_1086: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_895);  sub_263 = unsqueeze_895 = None
    mul_1087: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_193);  sum_81 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_1086, relu_60, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1086 = primals_193 = None
    getitem_327: "f32[8, 1024, 14, 14]" = convolution_backward_39[0]
    getitem_328: "f32[2048, 1024, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_565: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_36, getitem_327);  where_36 = getitem_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_218: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_60);  relu_60 = None
    alias_219: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_218);  alias_218 = None
    le_39: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_219, 0);  alias_219 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_39: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_39, scalar_tensor_39, add_565);  le_39 = scalar_tensor_39 = add_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_896: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_897: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
    unsqueeze_898: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None
    sum_82: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_264: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_898)
    mul_1088: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_264);  sub_264 = None
    sum_83: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1088, [0, 2, 3]);  mul_1088 = None
    mul_1089: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    unsqueeze_899: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1089, 0);  mul_1089 = None
    unsqueeze_900: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 2);  unsqueeze_899 = None
    unsqueeze_901: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 3);  unsqueeze_900 = None
    mul_1090: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_83, 0.0006377551020408163)
    mul_1091: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_1092: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1090, mul_1091);  mul_1090 = mul_1091 = None
    unsqueeze_902: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1092, 0);  mul_1092 = None
    unsqueeze_903: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 2);  unsqueeze_902 = None
    unsqueeze_904: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 3);  unsqueeze_903 = None
    mul_1093: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_191);  primals_191 = None
    unsqueeze_905: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1093, 0);  mul_1093 = None
    unsqueeze_906: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    sub_265: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_898);  convolution_63 = unsqueeze_898 = None
    mul_1094: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_904);  sub_265 = unsqueeze_904 = None
    sub_266: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_39, mul_1094);  mul_1094 = None
    sub_267: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_901);  sub_266 = unsqueeze_901 = None
    mul_1095: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_907);  sub_267 = unsqueeze_907 = None
    mul_1096: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_190);  sum_83 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_1095, relu_59, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1095 = primals_190 = None
    getitem_330: "f32[8, 2048, 14, 14]" = convolution_backward_40[0]
    getitem_331: "f32[1024, 2048, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_221: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_59);  relu_59 = None
    alias_222: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_221);  alias_221 = None
    le_40: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_222, 0);  alias_222 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_40: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_40, scalar_tensor_40, getitem_330);  le_40 = scalar_tensor_40 = getitem_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_908: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_909: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
    unsqueeze_910: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None
    sum_84: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_268: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_910)
    mul_1097: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_268);  sub_268 = None
    sum_85: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1097, [0, 2, 3]);  mul_1097 = None
    mul_1098: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    unsqueeze_911: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1098, 0);  mul_1098 = None
    unsqueeze_912: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 2);  unsqueeze_911 = None
    unsqueeze_913: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 3);  unsqueeze_912 = None
    mul_1099: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    mul_1100: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_1101: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1099, mul_1100);  mul_1099 = mul_1100 = None
    unsqueeze_914: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1101, 0);  mul_1101 = None
    unsqueeze_915: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 2);  unsqueeze_914 = None
    unsqueeze_916: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 3);  unsqueeze_915 = None
    mul_1102: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_188);  primals_188 = None
    unsqueeze_917: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1102, 0);  mul_1102 = None
    unsqueeze_918: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    sub_269: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_910);  convolution_62 = unsqueeze_910 = None
    mul_1103: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_916);  sub_269 = unsqueeze_916 = None
    sub_270: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_1103);  where_40 = mul_1103 = None
    sub_271: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_270, unsqueeze_913);  sub_270 = unsqueeze_913 = None
    mul_1104: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_919);  sub_271 = unsqueeze_919 = None
    mul_1105: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_187);  sum_85 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_1104, relu_58, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1104 = primals_187 = None
    getitem_333: "f32[8, 2048, 14, 14]" = convolution_backward_41[0]
    getitem_334: "f32[2048, 64, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_224: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_58);  relu_58 = None
    alias_225: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_224);  alias_224 = None
    le_41: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_225, 0);  alias_225 = None
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_41: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_41, scalar_tensor_41, getitem_333);  le_41 = scalar_tensor_41 = getitem_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_920: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_921: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 2);  unsqueeze_920 = None
    unsqueeze_922: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 3);  unsqueeze_921 = None
    sum_86: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_272: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_922)
    mul_1106: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, sub_272);  sub_272 = None
    sum_87: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1106, [0, 2, 3]);  mul_1106 = None
    mul_1107: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006377551020408163)
    unsqueeze_923: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1107, 0);  mul_1107 = None
    unsqueeze_924: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 2);  unsqueeze_923 = None
    unsqueeze_925: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 3);  unsqueeze_924 = None
    mul_1108: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006377551020408163)
    mul_1109: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_1110: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1108, mul_1109);  mul_1108 = mul_1109 = None
    unsqueeze_926: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1110, 0);  mul_1110 = None
    unsqueeze_927: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 2);  unsqueeze_926 = None
    unsqueeze_928: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 3);  unsqueeze_927 = None
    mul_1111: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_185);  primals_185 = None
    unsqueeze_929: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1111, 0);  mul_1111 = None
    unsqueeze_930: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
    unsqueeze_931: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
    sub_273: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_922);  convolution_61 = unsqueeze_922 = None
    mul_1112: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_928);  sub_273 = unsqueeze_928 = None
    sub_274: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_41, mul_1112);  where_41 = mul_1112 = None
    sub_275: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_274, unsqueeze_925);  sub_274 = unsqueeze_925 = None
    mul_1113: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_931);  sub_275 = unsqueeze_931 = None
    mul_1114: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_184);  sum_87 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_1113, relu_57, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1113 = primals_184 = None
    getitem_336: "f32[8, 1024, 14, 14]" = convolution_backward_42[0]
    getitem_337: "f32[2048, 1024, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_566: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_39, getitem_336);  where_39 = getitem_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_227: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_57);  relu_57 = None
    alias_228: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_227);  alias_227 = None
    le_42: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_228, 0);  alias_228 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_42: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_42, scalar_tensor_42, add_566);  le_42 = scalar_tensor_42 = add_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_932: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_933: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 2);  unsqueeze_932 = None
    unsqueeze_934: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 3);  unsqueeze_933 = None
    sum_88: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_276: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_934)
    mul_1115: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_276);  sub_276 = None
    sum_89: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1115, [0, 2, 3]);  mul_1115 = None
    mul_1116: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_88, 0.0006377551020408163)
    unsqueeze_935: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1116, 0);  mul_1116 = None
    unsqueeze_936: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 2);  unsqueeze_935 = None
    unsqueeze_937: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 3);  unsqueeze_936 = None
    mul_1117: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_89, 0.0006377551020408163)
    mul_1118: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_1119: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1117, mul_1118);  mul_1117 = mul_1118 = None
    unsqueeze_938: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1119, 0);  mul_1119 = None
    unsqueeze_939: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 2);  unsqueeze_938 = None
    unsqueeze_940: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 3);  unsqueeze_939 = None
    mul_1120: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_182);  primals_182 = None
    unsqueeze_941: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1120, 0);  mul_1120 = None
    unsqueeze_942: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
    unsqueeze_943: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
    sub_277: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_934);  convolution_60 = unsqueeze_934 = None
    mul_1121: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_940);  sub_277 = unsqueeze_940 = None
    sub_278: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_42, mul_1121);  mul_1121 = None
    sub_279: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_278, unsqueeze_937);  sub_278 = unsqueeze_937 = None
    mul_1122: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_943);  sub_279 = unsqueeze_943 = None
    mul_1123: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_181);  sum_89 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1122, relu_56, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1122 = primals_181 = None
    getitem_339: "f32[8, 2048, 14, 14]" = convolution_backward_43[0]
    getitem_340: "f32[1024, 2048, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_230: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_56);  relu_56 = None
    alias_231: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_230);  alias_230 = None
    le_43: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_231, 0);  alias_231 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_43: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_43, scalar_tensor_43, getitem_339);  le_43 = scalar_tensor_43 = getitem_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_944: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_945: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 2);  unsqueeze_944 = None
    unsqueeze_946: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 3);  unsqueeze_945 = None
    sum_90: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_280: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_946)
    mul_1124: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_280);  sub_280 = None
    sum_91: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1124, [0, 2, 3]);  mul_1124 = None
    mul_1125: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_90, 0.0006377551020408163)
    unsqueeze_947: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1125, 0);  mul_1125 = None
    unsqueeze_948: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 2);  unsqueeze_947 = None
    unsqueeze_949: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 3);  unsqueeze_948 = None
    mul_1126: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_91, 0.0006377551020408163)
    mul_1127: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_1128: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1126, mul_1127);  mul_1126 = mul_1127 = None
    unsqueeze_950: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1128, 0);  mul_1128 = None
    unsqueeze_951: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 2);  unsqueeze_950 = None
    unsqueeze_952: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 3);  unsqueeze_951 = None
    mul_1129: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_179);  primals_179 = None
    unsqueeze_953: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1129, 0);  mul_1129 = None
    unsqueeze_954: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
    unsqueeze_955: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
    sub_281: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_946);  convolution_59 = unsqueeze_946 = None
    mul_1130: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_952);  sub_281 = unsqueeze_952 = None
    sub_282: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_43, mul_1130);  where_43 = mul_1130 = None
    sub_283: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_282, unsqueeze_949);  sub_282 = unsqueeze_949 = None
    mul_1131: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_955);  sub_283 = unsqueeze_955 = None
    mul_1132: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_178);  sum_91 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1131, relu_55, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1131 = primals_178 = None
    getitem_342: "f32[8, 2048, 14, 14]" = convolution_backward_44[0]
    getitem_343: "f32[2048, 64, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_233: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_55);  relu_55 = None
    alias_234: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_233);  alias_233 = None
    le_44: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_234, 0);  alias_234 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_44: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_44, scalar_tensor_44, getitem_342);  le_44 = scalar_tensor_44 = getitem_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_956: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_957: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 2);  unsqueeze_956 = None
    unsqueeze_958: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 3);  unsqueeze_957 = None
    sum_92: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_284: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_958)
    mul_1133: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_284);  sub_284 = None
    sum_93: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1133, [0, 2, 3]);  mul_1133 = None
    mul_1134: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_92, 0.0006377551020408163)
    unsqueeze_959: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1134, 0);  mul_1134 = None
    unsqueeze_960: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 2);  unsqueeze_959 = None
    unsqueeze_961: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 3);  unsqueeze_960 = None
    mul_1135: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_93, 0.0006377551020408163)
    mul_1136: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_1137: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1135, mul_1136);  mul_1135 = mul_1136 = None
    unsqueeze_962: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1137, 0);  mul_1137 = None
    unsqueeze_963: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 2);  unsqueeze_962 = None
    unsqueeze_964: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 3);  unsqueeze_963 = None
    mul_1138: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_176);  primals_176 = None
    unsqueeze_965: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1138, 0);  mul_1138 = None
    unsqueeze_966: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
    unsqueeze_967: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
    sub_285: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_958);  convolution_58 = unsqueeze_958 = None
    mul_1139: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_964);  sub_285 = unsqueeze_964 = None
    sub_286: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_44, mul_1139);  where_44 = mul_1139 = None
    sub_287: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_286, unsqueeze_961);  sub_286 = unsqueeze_961 = None
    mul_1140: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_967);  sub_287 = unsqueeze_967 = None
    mul_1141: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_175);  sum_93 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1140, relu_54, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1140 = primals_175 = None
    getitem_345: "f32[8, 1024, 14, 14]" = convolution_backward_45[0]
    getitem_346: "f32[2048, 1024, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_567: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_42, getitem_345);  where_42 = getitem_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_236: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_54);  relu_54 = None
    alias_237: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_236);  alias_236 = None
    le_45: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_237, 0);  alias_237 = None
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_45: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_45, scalar_tensor_45, add_567);  le_45 = scalar_tensor_45 = add_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_968: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_969: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
    unsqueeze_970: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None
    sum_94: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_288: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_970)
    mul_1142: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, sub_288);  sub_288 = None
    sum_95: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1142, [0, 2, 3]);  mul_1142 = None
    mul_1143: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_94, 0.0006377551020408163)
    unsqueeze_971: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1143, 0);  mul_1143 = None
    unsqueeze_972: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 2);  unsqueeze_971 = None
    unsqueeze_973: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 3);  unsqueeze_972 = None
    mul_1144: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_95, 0.0006377551020408163)
    mul_1145: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_1146: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1144, mul_1145);  mul_1144 = mul_1145 = None
    unsqueeze_974: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1146, 0);  mul_1146 = None
    unsqueeze_975: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 2);  unsqueeze_974 = None
    unsqueeze_976: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 3);  unsqueeze_975 = None
    mul_1147: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_173);  primals_173 = None
    unsqueeze_977: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1147, 0);  mul_1147 = None
    unsqueeze_978: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    sub_289: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_970);  convolution_57 = unsqueeze_970 = None
    mul_1148: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_976);  sub_289 = unsqueeze_976 = None
    sub_290: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_45, mul_1148);  mul_1148 = None
    sub_291: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_290, unsqueeze_973);  sub_290 = unsqueeze_973 = None
    mul_1149: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_979);  sub_291 = unsqueeze_979 = None
    mul_1150: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_172);  sum_95 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1149, relu_53, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1149 = primals_172 = None
    getitem_348: "f32[8, 2048, 14, 14]" = convolution_backward_46[0]
    getitem_349: "f32[1024, 2048, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_239: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_53);  relu_53 = None
    alias_240: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_239);  alias_239 = None
    le_46: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_240, 0);  alias_240 = None
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_46: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_46, scalar_tensor_46, getitem_348);  le_46 = scalar_tensor_46 = getitem_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_980: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_981: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 2);  unsqueeze_980 = None
    unsqueeze_982: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 3);  unsqueeze_981 = None
    sum_96: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_292: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_982)
    mul_1151: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, sub_292);  sub_292 = None
    sum_97: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1151, [0, 2, 3]);  mul_1151 = None
    mul_1152: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_96, 0.0006377551020408163)
    unsqueeze_983: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1152, 0);  mul_1152 = None
    unsqueeze_984: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 2);  unsqueeze_983 = None
    unsqueeze_985: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 3);  unsqueeze_984 = None
    mul_1153: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_97, 0.0006377551020408163)
    mul_1154: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_1155: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1153, mul_1154);  mul_1153 = mul_1154 = None
    unsqueeze_986: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1155, 0);  mul_1155 = None
    unsqueeze_987: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 2);  unsqueeze_986 = None
    unsqueeze_988: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 3);  unsqueeze_987 = None
    mul_1156: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_170);  primals_170 = None
    unsqueeze_989: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1156, 0);  mul_1156 = None
    unsqueeze_990: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
    unsqueeze_991: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
    sub_293: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_982);  convolution_56 = unsqueeze_982 = None
    mul_1157: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_988);  sub_293 = unsqueeze_988 = None
    sub_294: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_46, mul_1157);  where_46 = mul_1157 = None
    sub_295: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_294, unsqueeze_985);  sub_294 = unsqueeze_985 = None
    mul_1158: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_991);  sub_295 = unsqueeze_991 = None
    mul_1159: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_169);  sum_97 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1158, relu_52, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1158 = primals_169 = None
    getitem_351: "f32[8, 2048, 14, 14]" = convolution_backward_47[0]
    getitem_352: "f32[2048, 64, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_242: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_52);  relu_52 = None
    alias_243: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_242);  alias_242 = None
    le_47: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_243, 0);  alias_243 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_47: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_47, scalar_tensor_47, getitem_351);  le_47 = scalar_tensor_47 = getitem_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_992: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_993: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 2);  unsqueeze_992 = None
    unsqueeze_994: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 3);  unsqueeze_993 = None
    sum_98: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_296: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_994)
    mul_1160: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, sub_296);  sub_296 = None
    sum_99: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1160, [0, 2, 3]);  mul_1160 = None
    mul_1161: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_98, 0.0006377551020408163)
    unsqueeze_995: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1161, 0);  mul_1161 = None
    unsqueeze_996: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 2);  unsqueeze_995 = None
    unsqueeze_997: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 3);  unsqueeze_996 = None
    mul_1162: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_99, 0.0006377551020408163)
    mul_1163: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_1164: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1162, mul_1163);  mul_1162 = mul_1163 = None
    unsqueeze_998: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1164, 0);  mul_1164 = None
    unsqueeze_999: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 2);  unsqueeze_998 = None
    unsqueeze_1000: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 3);  unsqueeze_999 = None
    mul_1165: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_167);  primals_167 = None
    unsqueeze_1001: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1165, 0);  mul_1165 = None
    unsqueeze_1002: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 2);  unsqueeze_1001 = None
    unsqueeze_1003: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 3);  unsqueeze_1002 = None
    sub_297: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_994);  convolution_55 = unsqueeze_994 = None
    mul_1166: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_1000);  sub_297 = unsqueeze_1000 = None
    sub_298: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_47, mul_1166);  where_47 = mul_1166 = None
    sub_299: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_298, unsqueeze_997);  sub_298 = unsqueeze_997 = None
    mul_1167: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_1003);  sub_299 = unsqueeze_1003 = None
    mul_1168: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_166);  sum_99 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1167, relu_51, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1167 = primals_166 = None
    getitem_354: "f32[8, 1024, 14, 14]" = convolution_backward_48[0]
    getitem_355: "f32[2048, 1024, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_568: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_45, getitem_354);  where_45 = getitem_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_245: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_51);  relu_51 = None
    alias_246: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_245);  alias_245 = None
    le_48: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_246, 0);  alias_246 = None
    scalar_tensor_48: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_48: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_48, scalar_tensor_48, add_568);  le_48 = scalar_tensor_48 = add_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1004: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_1005: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 2);  unsqueeze_1004 = None
    unsqueeze_1006: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 3);  unsqueeze_1005 = None
    sum_100: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_300: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_1006)
    mul_1169: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_48, sub_300);  sub_300 = None
    sum_101: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1169, [0, 2, 3]);  mul_1169 = None
    mul_1170: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_100, 0.0006377551020408163)
    unsqueeze_1007: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1170, 0);  mul_1170 = None
    unsqueeze_1008: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 2);  unsqueeze_1007 = None
    unsqueeze_1009: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 3);  unsqueeze_1008 = None
    mul_1171: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_101, 0.0006377551020408163)
    mul_1172: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_1173: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1171, mul_1172);  mul_1171 = mul_1172 = None
    unsqueeze_1010: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1173, 0);  mul_1173 = None
    unsqueeze_1011: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 2);  unsqueeze_1010 = None
    unsqueeze_1012: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 3);  unsqueeze_1011 = None
    mul_1174: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_164);  primals_164 = None
    unsqueeze_1013: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1174, 0);  mul_1174 = None
    unsqueeze_1014: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 2);  unsqueeze_1013 = None
    unsqueeze_1015: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 3);  unsqueeze_1014 = None
    sub_301: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_1006);  convolution_54 = unsqueeze_1006 = None
    mul_1175: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_1012);  sub_301 = unsqueeze_1012 = None
    sub_302: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_48, mul_1175);  mul_1175 = None
    sub_303: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_1009);  sub_302 = unsqueeze_1009 = None
    mul_1176: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_1015);  sub_303 = unsqueeze_1015 = None
    mul_1177: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_163);  sum_101 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1176, relu_50, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1176 = primals_163 = None
    getitem_357: "f32[8, 2048, 14, 14]" = convolution_backward_49[0]
    getitem_358: "f32[1024, 2048, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_248: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_50);  relu_50 = None
    alias_249: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_248);  alias_248 = None
    le_49: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_249, 0);  alias_249 = None
    scalar_tensor_49: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_49: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_49, scalar_tensor_49, getitem_357);  le_49 = scalar_tensor_49 = getitem_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1016: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_1017: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 2);  unsqueeze_1016 = None
    unsqueeze_1018: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 3);  unsqueeze_1017 = None
    sum_102: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_304: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_1018)
    mul_1178: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_49, sub_304);  sub_304 = None
    sum_103: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1178, [0, 2, 3]);  mul_1178 = None
    mul_1179: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_102, 0.0006377551020408163)
    unsqueeze_1019: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1179, 0);  mul_1179 = None
    unsqueeze_1020: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 2);  unsqueeze_1019 = None
    unsqueeze_1021: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 3);  unsqueeze_1020 = None
    mul_1180: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_103, 0.0006377551020408163)
    mul_1181: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_1182: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1180, mul_1181);  mul_1180 = mul_1181 = None
    unsqueeze_1022: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1182, 0);  mul_1182 = None
    unsqueeze_1023: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 2);  unsqueeze_1022 = None
    unsqueeze_1024: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 3);  unsqueeze_1023 = None
    mul_1183: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_161);  primals_161 = None
    unsqueeze_1025: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1183, 0);  mul_1183 = None
    unsqueeze_1026: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 2);  unsqueeze_1025 = None
    unsqueeze_1027: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 3);  unsqueeze_1026 = None
    sub_305: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_1018);  convolution_53 = unsqueeze_1018 = None
    mul_1184: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_1024);  sub_305 = unsqueeze_1024 = None
    sub_306: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_49, mul_1184);  where_49 = mul_1184 = None
    sub_307: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_306, unsqueeze_1021);  sub_306 = unsqueeze_1021 = None
    mul_1185: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_1027);  sub_307 = unsqueeze_1027 = None
    mul_1186: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_160);  sum_103 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1185, relu_49, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1185 = primals_160 = None
    getitem_360: "f32[8, 2048, 14, 14]" = convolution_backward_50[0]
    getitem_361: "f32[2048, 64, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_251: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_49);  relu_49 = None
    alias_252: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_251);  alias_251 = None
    le_50: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_252, 0);  alias_252 = None
    scalar_tensor_50: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_50: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_50, scalar_tensor_50, getitem_360);  le_50 = scalar_tensor_50 = getitem_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1028: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_1029: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 2);  unsqueeze_1028 = None
    unsqueeze_1030: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 3);  unsqueeze_1029 = None
    sum_104: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_308: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_1030)
    mul_1187: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_50, sub_308);  sub_308 = None
    sum_105: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1187, [0, 2, 3]);  mul_1187 = None
    mul_1188: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_104, 0.0006377551020408163)
    unsqueeze_1031: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1188, 0);  mul_1188 = None
    unsqueeze_1032: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 2);  unsqueeze_1031 = None
    unsqueeze_1033: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 3);  unsqueeze_1032 = None
    mul_1189: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_105, 0.0006377551020408163)
    mul_1190: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_1191: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1189, mul_1190);  mul_1189 = mul_1190 = None
    unsqueeze_1034: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1191, 0);  mul_1191 = None
    unsqueeze_1035: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 2);  unsqueeze_1034 = None
    unsqueeze_1036: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 3);  unsqueeze_1035 = None
    mul_1192: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_158);  primals_158 = None
    unsqueeze_1037: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1192, 0);  mul_1192 = None
    unsqueeze_1038: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 2);  unsqueeze_1037 = None
    unsqueeze_1039: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 3);  unsqueeze_1038 = None
    sub_309: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_1030);  convolution_52 = unsqueeze_1030 = None
    mul_1193: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_1036);  sub_309 = unsqueeze_1036 = None
    sub_310: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_50, mul_1193);  where_50 = mul_1193 = None
    sub_311: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_310, unsqueeze_1033);  sub_310 = unsqueeze_1033 = None
    mul_1194: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_1039);  sub_311 = unsqueeze_1039 = None
    mul_1195: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_157);  sum_105 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1194, relu_48, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1194 = primals_157 = None
    getitem_363: "f32[8, 1024, 14, 14]" = convolution_backward_51[0]
    getitem_364: "f32[2048, 1024, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_569: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_48, getitem_363);  where_48 = getitem_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_254: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_48);  relu_48 = None
    alias_255: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_254);  alias_254 = None
    le_51: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_255, 0);  alias_255 = None
    scalar_tensor_51: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_51: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_51, scalar_tensor_51, add_569);  le_51 = scalar_tensor_51 = add_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1040: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_1041: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 2);  unsqueeze_1040 = None
    unsqueeze_1042: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 3);  unsqueeze_1041 = None
    sum_106: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_312: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_1042)
    mul_1196: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_51, sub_312);  sub_312 = None
    sum_107: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1196, [0, 2, 3]);  mul_1196 = None
    mul_1197: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_106, 0.0006377551020408163)
    unsqueeze_1043: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1197, 0);  mul_1197 = None
    unsqueeze_1044: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 2);  unsqueeze_1043 = None
    unsqueeze_1045: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, 3);  unsqueeze_1044 = None
    mul_1198: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_107, 0.0006377551020408163)
    mul_1199: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_1200: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1198, mul_1199);  mul_1198 = mul_1199 = None
    unsqueeze_1046: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1200, 0);  mul_1200 = None
    unsqueeze_1047: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 2);  unsqueeze_1046 = None
    unsqueeze_1048: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 3);  unsqueeze_1047 = None
    mul_1201: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_155);  primals_155 = None
    unsqueeze_1049: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1201, 0);  mul_1201 = None
    unsqueeze_1050: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 2);  unsqueeze_1049 = None
    unsqueeze_1051: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 3);  unsqueeze_1050 = None
    sub_313: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_1042);  convolution_51 = unsqueeze_1042 = None
    mul_1202: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_1048);  sub_313 = unsqueeze_1048 = None
    sub_314: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_51, mul_1202);  mul_1202 = None
    sub_315: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_314, unsqueeze_1045);  sub_314 = unsqueeze_1045 = None
    mul_1203: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_1051);  sub_315 = unsqueeze_1051 = None
    mul_1204: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_154);  sum_107 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1203, relu_47, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1203 = primals_154 = None
    getitem_366: "f32[8, 2048, 14, 14]" = convolution_backward_52[0]
    getitem_367: "f32[1024, 2048, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_257: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_47);  relu_47 = None
    alias_258: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_257);  alias_257 = None
    le_52: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_258, 0);  alias_258 = None
    scalar_tensor_52: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_52: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_52, scalar_tensor_52, getitem_366);  le_52 = scalar_tensor_52 = getitem_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1052: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_1053: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 2);  unsqueeze_1052 = None
    unsqueeze_1054: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 3);  unsqueeze_1053 = None
    sum_108: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_316: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_1054)
    mul_1205: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_52, sub_316);  sub_316 = None
    sum_109: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1205, [0, 2, 3]);  mul_1205 = None
    mul_1206: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_108, 0.0006377551020408163)
    unsqueeze_1055: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1206, 0);  mul_1206 = None
    unsqueeze_1056: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 2);  unsqueeze_1055 = None
    unsqueeze_1057: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, 3);  unsqueeze_1056 = None
    mul_1207: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_109, 0.0006377551020408163)
    mul_1208: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_1209: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1207, mul_1208);  mul_1207 = mul_1208 = None
    unsqueeze_1058: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1209, 0);  mul_1209 = None
    unsqueeze_1059: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 2);  unsqueeze_1058 = None
    unsqueeze_1060: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 3);  unsqueeze_1059 = None
    mul_1210: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_152);  primals_152 = None
    unsqueeze_1061: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1210, 0);  mul_1210 = None
    unsqueeze_1062: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 2);  unsqueeze_1061 = None
    unsqueeze_1063: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 3);  unsqueeze_1062 = None
    sub_317: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_1054);  convolution_50 = unsqueeze_1054 = None
    mul_1211: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_1060);  sub_317 = unsqueeze_1060 = None
    sub_318: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_52, mul_1211);  where_52 = mul_1211 = None
    sub_319: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_318, unsqueeze_1057);  sub_318 = unsqueeze_1057 = None
    mul_1212: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_1063);  sub_319 = unsqueeze_1063 = None
    mul_1213: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_151);  sum_109 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1212, relu_46, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1212 = primals_151 = None
    getitem_369: "f32[8, 2048, 14, 14]" = convolution_backward_53[0]
    getitem_370: "f32[2048, 64, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_260: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_46);  relu_46 = None
    alias_261: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_260);  alias_260 = None
    le_53: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_261, 0);  alias_261 = None
    scalar_tensor_53: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_53: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_53, scalar_tensor_53, getitem_369);  le_53 = scalar_tensor_53 = getitem_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1064: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_1065: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 2);  unsqueeze_1064 = None
    unsqueeze_1066: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 3);  unsqueeze_1065 = None
    sum_110: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_320: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_1066)
    mul_1214: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_53, sub_320);  sub_320 = None
    sum_111: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1214, [0, 2, 3]);  mul_1214 = None
    mul_1215: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_110, 0.0006377551020408163)
    unsqueeze_1067: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1215, 0);  mul_1215 = None
    unsqueeze_1068: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 2);  unsqueeze_1067 = None
    unsqueeze_1069: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, 3);  unsqueeze_1068 = None
    mul_1216: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_111, 0.0006377551020408163)
    mul_1217: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_1218: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1216, mul_1217);  mul_1216 = mul_1217 = None
    unsqueeze_1070: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1218, 0);  mul_1218 = None
    unsqueeze_1071: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 2);  unsqueeze_1070 = None
    unsqueeze_1072: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 3);  unsqueeze_1071 = None
    mul_1219: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_149);  primals_149 = None
    unsqueeze_1073: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1219, 0);  mul_1219 = None
    unsqueeze_1074: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 2);  unsqueeze_1073 = None
    unsqueeze_1075: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 3);  unsqueeze_1074 = None
    sub_321: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_1066);  convolution_49 = unsqueeze_1066 = None
    mul_1220: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_1072);  sub_321 = unsqueeze_1072 = None
    sub_322: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_53, mul_1220);  where_53 = mul_1220 = None
    sub_323: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_322, unsqueeze_1069);  sub_322 = unsqueeze_1069 = None
    mul_1221: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_1075);  sub_323 = unsqueeze_1075 = None
    mul_1222: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_148);  sum_111 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1221, relu_45, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1221 = primals_148 = None
    getitem_372: "f32[8, 1024, 14, 14]" = convolution_backward_54[0]
    getitem_373: "f32[2048, 1024, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_570: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_51, getitem_372);  where_51 = getitem_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_263: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_45);  relu_45 = None
    alias_264: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_263);  alias_263 = None
    le_54: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_264, 0);  alias_264 = None
    scalar_tensor_54: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_54: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_54, scalar_tensor_54, add_570);  le_54 = scalar_tensor_54 = add_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1076: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_1077: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, 2);  unsqueeze_1076 = None
    unsqueeze_1078: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 3);  unsqueeze_1077 = None
    sum_112: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_324: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_1078)
    mul_1223: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_54, sub_324);  sub_324 = None
    sum_113: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1223, [0, 2, 3]);  mul_1223 = None
    mul_1224: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_112, 0.0006377551020408163)
    unsqueeze_1079: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1224, 0);  mul_1224 = None
    unsqueeze_1080: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1079, 2);  unsqueeze_1079 = None
    unsqueeze_1081: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, 3);  unsqueeze_1080 = None
    mul_1225: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_113, 0.0006377551020408163)
    mul_1226: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_1227: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1225, mul_1226);  mul_1225 = mul_1226 = None
    unsqueeze_1082: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1227, 0);  mul_1227 = None
    unsqueeze_1083: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 2);  unsqueeze_1082 = None
    unsqueeze_1084: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 3);  unsqueeze_1083 = None
    mul_1228: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_146);  primals_146 = None
    unsqueeze_1085: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1228, 0);  mul_1228 = None
    unsqueeze_1086: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 2);  unsqueeze_1085 = None
    unsqueeze_1087: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 3);  unsqueeze_1086 = None
    sub_325: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_1078);  convolution_48 = unsqueeze_1078 = None
    mul_1229: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_1084);  sub_325 = unsqueeze_1084 = None
    sub_326: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_54, mul_1229);  mul_1229 = None
    sub_327: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_326, unsqueeze_1081);  sub_326 = unsqueeze_1081 = None
    mul_1230: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_1087);  sub_327 = unsqueeze_1087 = None
    mul_1231: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_145);  sum_113 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1230, relu_44, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1230 = primals_145 = None
    getitem_375: "f32[8, 2048, 14, 14]" = convolution_backward_55[0]
    getitem_376: "f32[1024, 2048, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_266: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_44);  relu_44 = None
    alias_267: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_266);  alias_266 = None
    le_55: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_267, 0);  alias_267 = None
    scalar_tensor_55: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_55: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_55, scalar_tensor_55, getitem_375);  le_55 = scalar_tensor_55 = getitem_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1088: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_1089: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, 2);  unsqueeze_1088 = None
    unsqueeze_1090: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 3);  unsqueeze_1089 = None
    sum_114: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_328: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_1090)
    mul_1232: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_55, sub_328);  sub_328 = None
    sum_115: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1232, [0, 2, 3]);  mul_1232 = None
    mul_1233: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_114, 0.0006377551020408163)
    unsqueeze_1091: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1233, 0);  mul_1233 = None
    unsqueeze_1092: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1091, 2);  unsqueeze_1091 = None
    unsqueeze_1093: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, 3);  unsqueeze_1092 = None
    mul_1234: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_115, 0.0006377551020408163)
    mul_1235: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_1236: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1234, mul_1235);  mul_1234 = mul_1235 = None
    unsqueeze_1094: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1236, 0);  mul_1236 = None
    unsqueeze_1095: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 2);  unsqueeze_1094 = None
    unsqueeze_1096: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 3);  unsqueeze_1095 = None
    mul_1237: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_143);  primals_143 = None
    unsqueeze_1097: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1237, 0);  mul_1237 = None
    unsqueeze_1098: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 2);  unsqueeze_1097 = None
    unsqueeze_1099: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 3);  unsqueeze_1098 = None
    sub_329: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_1090);  convolution_47 = unsqueeze_1090 = None
    mul_1238: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_1096);  sub_329 = unsqueeze_1096 = None
    sub_330: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_55, mul_1238);  where_55 = mul_1238 = None
    sub_331: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_330, unsqueeze_1093);  sub_330 = unsqueeze_1093 = None
    mul_1239: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_1099);  sub_331 = unsqueeze_1099 = None
    mul_1240: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_142);  sum_115 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1239, relu_43, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1239 = primals_142 = None
    getitem_378: "f32[8, 2048, 14, 14]" = convolution_backward_56[0]
    getitem_379: "f32[2048, 64, 3, 3]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_269: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_43);  relu_43 = None
    alias_270: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_269);  alias_269 = None
    le_56: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_270, 0);  alias_270 = None
    scalar_tensor_56: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_56: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_56, scalar_tensor_56, getitem_378);  le_56 = scalar_tensor_56 = getitem_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1100: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_1101: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, 2);  unsqueeze_1100 = None
    unsqueeze_1102: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 3);  unsqueeze_1101 = None
    sum_116: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_332: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_1102)
    mul_1241: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_56, sub_332);  sub_332 = None
    sum_117: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1241, [0, 2, 3]);  mul_1241 = None
    mul_1242: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_116, 0.0006377551020408163)
    unsqueeze_1103: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1242, 0);  mul_1242 = None
    unsqueeze_1104: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1103, 2);  unsqueeze_1103 = None
    unsqueeze_1105: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, 3);  unsqueeze_1104 = None
    mul_1243: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_117, 0.0006377551020408163)
    mul_1244: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_1245: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1243, mul_1244);  mul_1243 = mul_1244 = None
    unsqueeze_1106: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1245, 0);  mul_1245 = None
    unsqueeze_1107: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 2);  unsqueeze_1106 = None
    unsqueeze_1108: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 3);  unsqueeze_1107 = None
    mul_1246: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_140);  primals_140 = None
    unsqueeze_1109: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1246, 0);  mul_1246 = None
    unsqueeze_1110: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 2);  unsqueeze_1109 = None
    unsqueeze_1111: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 3);  unsqueeze_1110 = None
    sub_333: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_1102);  convolution_46 = unsqueeze_1102 = None
    mul_1247: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_1108);  sub_333 = unsqueeze_1108 = None
    sub_334: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_56, mul_1247);  where_56 = mul_1247 = None
    sub_335: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_334, unsqueeze_1105);  sub_334 = unsqueeze_1105 = None
    mul_1248: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_1111);  sub_335 = unsqueeze_1111 = None
    mul_1249: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_139);  sum_117 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1248, relu_42, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1248 = primals_139 = None
    getitem_381: "f32[8, 1024, 14, 14]" = convolution_backward_57[0]
    getitem_382: "f32[2048, 1024, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_571: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_54, getitem_381);  where_54 = getitem_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_272: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
    alias_273: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_272);  alias_272 = None
    le_57: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_273, 0);  alias_273 = None
    scalar_tensor_57: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_57: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_57, scalar_tensor_57, add_571);  le_57 = scalar_tensor_57 = add_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1112: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_1113: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, 2);  unsqueeze_1112 = None
    unsqueeze_1114: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 3);  unsqueeze_1113 = None
    sum_118: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_336: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_1114)
    mul_1250: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_57, sub_336);  sub_336 = None
    sum_119: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1250, [0, 2, 3]);  mul_1250 = None
    mul_1251: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_118, 0.0006377551020408163)
    unsqueeze_1115: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1251, 0);  mul_1251 = None
    unsqueeze_1116: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1115, 2);  unsqueeze_1115 = None
    unsqueeze_1117: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, 3);  unsqueeze_1116 = None
    mul_1252: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_119, 0.0006377551020408163)
    mul_1253: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_1254: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1252, mul_1253);  mul_1252 = mul_1253 = None
    unsqueeze_1118: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1254, 0);  mul_1254 = None
    unsqueeze_1119: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 2);  unsqueeze_1118 = None
    unsqueeze_1120: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 3);  unsqueeze_1119 = None
    mul_1255: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_137);  primals_137 = None
    unsqueeze_1121: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1255, 0);  mul_1255 = None
    unsqueeze_1122: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 2);  unsqueeze_1121 = None
    unsqueeze_1123: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 3);  unsqueeze_1122 = None
    sub_337: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_1114);  convolution_45 = unsqueeze_1114 = None
    mul_1256: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_1120);  sub_337 = unsqueeze_1120 = None
    sub_338: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_57, mul_1256);  mul_1256 = None
    sub_339: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_338, unsqueeze_1117);  sub_338 = unsqueeze_1117 = None
    mul_1257: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_1123);  sub_339 = unsqueeze_1123 = None
    mul_1258: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_136);  sum_119 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1257, relu_41, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1257 = primals_136 = None
    getitem_384: "f32[8, 2048, 14, 14]" = convolution_backward_58[0]
    getitem_385: "f32[1024, 2048, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_275: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_276: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_275);  alias_275 = None
    le_58: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_276, 0);  alias_276 = None
    scalar_tensor_58: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_58: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_58, scalar_tensor_58, getitem_384);  le_58 = scalar_tensor_58 = getitem_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1124: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_1125: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, 2);  unsqueeze_1124 = None
    unsqueeze_1126: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 3);  unsqueeze_1125 = None
    sum_120: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_340: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_1126)
    mul_1259: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_58, sub_340);  sub_340 = None
    sum_121: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1259, [0, 2, 3]);  mul_1259 = None
    mul_1260: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_120, 0.0006377551020408163)
    unsqueeze_1127: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1260, 0);  mul_1260 = None
    unsqueeze_1128: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1127, 2);  unsqueeze_1127 = None
    unsqueeze_1129: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, 3);  unsqueeze_1128 = None
    mul_1261: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_121, 0.0006377551020408163)
    mul_1262: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_1263: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1261, mul_1262);  mul_1261 = mul_1262 = None
    unsqueeze_1130: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1263, 0);  mul_1263 = None
    unsqueeze_1131: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 2);  unsqueeze_1130 = None
    unsqueeze_1132: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 3);  unsqueeze_1131 = None
    mul_1264: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_134);  primals_134 = None
    unsqueeze_1133: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1264, 0);  mul_1264 = None
    unsqueeze_1134: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 2);  unsqueeze_1133 = None
    unsqueeze_1135: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 3);  unsqueeze_1134 = None
    sub_341: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_1126);  convolution_44 = unsqueeze_1126 = None
    mul_1265: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_1132);  sub_341 = unsqueeze_1132 = None
    sub_342: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_58, mul_1265);  where_58 = mul_1265 = None
    sub_343: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_342, unsqueeze_1129);  sub_342 = unsqueeze_1129 = None
    mul_1266: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_1135);  sub_343 = unsqueeze_1135 = None
    mul_1267: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_133);  sum_121 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1266, relu_40, primals_133, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1266 = primals_133 = None
    getitem_387: "f32[8, 2048, 14, 14]" = convolution_backward_59[0]
    getitem_388: "f32[2048, 64, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_278: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_40);  relu_40 = None
    alias_279: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_278);  alias_278 = None
    le_59: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_279, 0);  alias_279 = None
    scalar_tensor_59: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_59: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_59, scalar_tensor_59, getitem_387);  le_59 = scalar_tensor_59 = getitem_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1136: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_1137: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, 2);  unsqueeze_1136 = None
    unsqueeze_1138: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 3);  unsqueeze_1137 = None
    sum_122: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_344: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_1138)
    mul_1268: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_59, sub_344);  sub_344 = None
    sum_123: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1268, [0, 2, 3]);  mul_1268 = None
    mul_1269: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_122, 0.0006377551020408163)
    unsqueeze_1139: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1269, 0);  mul_1269 = None
    unsqueeze_1140: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1139, 2);  unsqueeze_1139 = None
    unsqueeze_1141: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, 3);  unsqueeze_1140 = None
    mul_1270: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_123, 0.0006377551020408163)
    mul_1271: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_1272: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1270, mul_1271);  mul_1270 = mul_1271 = None
    unsqueeze_1142: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1272, 0);  mul_1272 = None
    unsqueeze_1143: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 2);  unsqueeze_1142 = None
    unsqueeze_1144: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 3);  unsqueeze_1143 = None
    mul_1273: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_131);  primals_131 = None
    unsqueeze_1145: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1273, 0);  mul_1273 = None
    unsqueeze_1146: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 2);  unsqueeze_1145 = None
    unsqueeze_1147: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 3);  unsqueeze_1146 = None
    sub_345: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_1138);  convolution_43 = unsqueeze_1138 = None
    mul_1274: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_1144);  sub_345 = unsqueeze_1144 = None
    sub_346: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_59, mul_1274);  where_59 = mul_1274 = None
    sub_347: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_346, unsqueeze_1141);  sub_346 = unsqueeze_1141 = None
    mul_1275: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_1147);  sub_347 = unsqueeze_1147 = None
    mul_1276: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_130);  sum_123 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1275, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1275 = primals_130 = None
    getitem_390: "f32[8, 1024, 14, 14]" = convolution_backward_60[0]
    getitem_391: "f32[2048, 1024, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_572: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_57, getitem_390);  where_57 = getitem_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_281: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_282: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_281);  alias_281 = None
    le_60: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_282, 0);  alias_282 = None
    scalar_tensor_60: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_60: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_60, scalar_tensor_60, add_572);  le_60 = scalar_tensor_60 = add_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1148: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_1149: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, 2);  unsqueeze_1148 = None
    unsqueeze_1150: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 3);  unsqueeze_1149 = None
    sum_124: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_348: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_1150)
    mul_1277: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_60, sub_348);  sub_348 = None
    sum_125: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1277, [0, 2, 3]);  mul_1277 = None
    mul_1278: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_124, 0.0006377551020408163)
    unsqueeze_1151: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1278, 0);  mul_1278 = None
    unsqueeze_1152: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1151, 2);  unsqueeze_1151 = None
    unsqueeze_1153: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, 3);  unsqueeze_1152 = None
    mul_1279: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_125, 0.0006377551020408163)
    mul_1280: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_1281: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1279, mul_1280);  mul_1279 = mul_1280 = None
    unsqueeze_1154: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1281, 0);  mul_1281 = None
    unsqueeze_1155: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 2);  unsqueeze_1154 = None
    unsqueeze_1156: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 3);  unsqueeze_1155 = None
    mul_1282: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_128);  primals_128 = None
    unsqueeze_1157: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1282, 0);  mul_1282 = None
    unsqueeze_1158: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 2);  unsqueeze_1157 = None
    unsqueeze_1159: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 3);  unsqueeze_1158 = None
    sub_349: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_1150);  convolution_42 = unsqueeze_1150 = None
    mul_1283: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_1156);  sub_349 = unsqueeze_1156 = None
    sub_350: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_60, mul_1283);  mul_1283 = None
    sub_351: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_350, unsqueeze_1153);  sub_350 = unsqueeze_1153 = None
    mul_1284: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_1159);  sub_351 = unsqueeze_1159 = None
    mul_1285: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_127);  sum_125 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1284, relu_38, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1284 = primals_127 = None
    getitem_393: "f32[8, 2048, 14, 14]" = convolution_backward_61[0]
    getitem_394: "f32[1024, 2048, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_284: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_285: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_284);  alias_284 = None
    le_61: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_285, 0);  alias_285 = None
    scalar_tensor_61: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_61: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_61, scalar_tensor_61, getitem_393);  le_61 = scalar_tensor_61 = getitem_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1160: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_1161: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, 2);  unsqueeze_1160 = None
    unsqueeze_1162: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 3);  unsqueeze_1161 = None
    sum_126: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_352: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1162)
    mul_1286: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_61, sub_352);  sub_352 = None
    sum_127: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1286, [0, 2, 3]);  mul_1286 = None
    mul_1287: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_126, 0.0006377551020408163)
    unsqueeze_1163: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1287, 0);  mul_1287 = None
    unsqueeze_1164: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1163, 2);  unsqueeze_1163 = None
    unsqueeze_1165: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, 3);  unsqueeze_1164 = None
    mul_1288: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_127, 0.0006377551020408163)
    mul_1289: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_1290: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1288, mul_1289);  mul_1288 = mul_1289 = None
    unsqueeze_1166: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1290, 0);  mul_1290 = None
    unsqueeze_1167: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 2);  unsqueeze_1166 = None
    unsqueeze_1168: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 3);  unsqueeze_1167 = None
    mul_1291: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_125);  primals_125 = None
    unsqueeze_1169: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1291, 0);  mul_1291 = None
    unsqueeze_1170: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 2);  unsqueeze_1169 = None
    unsqueeze_1171: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 3);  unsqueeze_1170 = None
    sub_353: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1162);  convolution_41 = unsqueeze_1162 = None
    mul_1292: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_353, unsqueeze_1168);  sub_353 = unsqueeze_1168 = None
    sub_354: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_61, mul_1292);  where_61 = mul_1292 = None
    sub_355: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_354, unsqueeze_1165);  sub_354 = unsqueeze_1165 = None
    mul_1293: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_355, unsqueeze_1171);  sub_355 = unsqueeze_1171 = None
    mul_1294: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_124);  sum_127 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1293, relu_37, primals_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1293 = primals_124 = None
    getitem_396: "f32[8, 2048, 14, 14]" = convolution_backward_62[0]
    getitem_397: "f32[2048, 64, 3, 3]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_287: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_288: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_287);  alias_287 = None
    le_62: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_288, 0);  alias_288 = None
    scalar_tensor_62: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_62: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_62, scalar_tensor_62, getitem_396);  le_62 = scalar_tensor_62 = getitem_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1172: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_1173: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, 2);  unsqueeze_1172 = None
    unsqueeze_1174: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 3);  unsqueeze_1173 = None
    sum_128: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_356: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_1174)
    mul_1295: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_62, sub_356);  sub_356 = None
    sum_129: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1295, [0, 2, 3]);  mul_1295 = None
    mul_1296: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_128, 0.0006377551020408163)
    unsqueeze_1175: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1296, 0);  mul_1296 = None
    unsqueeze_1176: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1175, 2);  unsqueeze_1175 = None
    unsqueeze_1177: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, 3);  unsqueeze_1176 = None
    mul_1297: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_129, 0.0006377551020408163)
    mul_1298: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_1299: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1297, mul_1298);  mul_1297 = mul_1298 = None
    unsqueeze_1178: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1299, 0);  mul_1299 = None
    unsqueeze_1179: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 2);  unsqueeze_1178 = None
    unsqueeze_1180: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 3);  unsqueeze_1179 = None
    mul_1300: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_122);  primals_122 = None
    unsqueeze_1181: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1300, 0);  mul_1300 = None
    unsqueeze_1182: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 2);  unsqueeze_1181 = None
    unsqueeze_1183: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 3);  unsqueeze_1182 = None
    sub_357: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_1174);  convolution_40 = unsqueeze_1174 = None
    mul_1301: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_357, unsqueeze_1180);  sub_357 = unsqueeze_1180 = None
    sub_358: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_62, mul_1301);  where_62 = mul_1301 = None
    sub_359: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_358, unsqueeze_1177);  sub_358 = unsqueeze_1177 = None
    mul_1302: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_359, unsqueeze_1183);  sub_359 = unsqueeze_1183 = None
    mul_1303: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_121);  sum_129 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1302, relu_36, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1302 = primals_121 = None
    getitem_399: "f32[8, 1024, 14, 14]" = convolution_backward_63[0]
    getitem_400: "f32[2048, 1024, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_573: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_60, getitem_399);  where_60 = getitem_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_290: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_291: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_290);  alias_290 = None
    le_63: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_291, 0);  alias_291 = None
    scalar_tensor_63: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_63: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_63, scalar_tensor_63, add_573);  le_63 = scalar_tensor_63 = add_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1184: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_1185: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, 2);  unsqueeze_1184 = None
    unsqueeze_1186: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 3);  unsqueeze_1185 = None
    sum_130: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_360: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_1186)
    mul_1304: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_63, sub_360);  sub_360 = None
    sum_131: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1304, [0, 2, 3]);  mul_1304 = None
    mul_1305: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_130, 0.0006377551020408163)
    unsqueeze_1187: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1305, 0);  mul_1305 = None
    unsqueeze_1188: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1187, 2);  unsqueeze_1187 = None
    unsqueeze_1189: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, 3);  unsqueeze_1188 = None
    mul_1306: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_131, 0.0006377551020408163)
    mul_1307: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_1308: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1306, mul_1307);  mul_1306 = mul_1307 = None
    unsqueeze_1190: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1308, 0);  mul_1308 = None
    unsqueeze_1191: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 2);  unsqueeze_1190 = None
    unsqueeze_1192: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 3);  unsqueeze_1191 = None
    mul_1309: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_119);  primals_119 = None
    unsqueeze_1193: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1309, 0);  mul_1309 = None
    unsqueeze_1194: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 2);  unsqueeze_1193 = None
    unsqueeze_1195: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 3);  unsqueeze_1194 = None
    sub_361: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_1186);  convolution_39 = unsqueeze_1186 = None
    mul_1310: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_361, unsqueeze_1192);  sub_361 = unsqueeze_1192 = None
    sub_362: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_63, mul_1310);  mul_1310 = None
    sub_363: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_362, unsqueeze_1189);  sub_362 = unsqueeze_1189 = None
    mul_1311: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_363, unsqueeze_1195);  sub_363 = unsqueeze_1195 = None
    mul_1312: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_118);  sum_131 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1311, relu_35, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1311 = primals_118 = None
    getitem_402: "f32[8, 2048, 14, 14]" = convolution_backward_64[0]
    getitem_403: "f32[1024, 2048, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_293: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_294: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_293);  alias_293 = None
    le_64: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_294, 0);  alias_294 = None
    scalar_tensor_64: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_64: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_64, scalar_tensor_64, getitem_402);  le_64 = scalar_tensor_64 = getitem_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1196: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_1197: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, 2);  unsqueeze_1196 = None
    unsqueeze_1198: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 3);  unsqueeze_1197 = None
    sum_132: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_364: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_1198)
    mul_1313: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_64, sub_364);  sub_364 = None
    sum_133: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1313, [0, 2, 3]);  mul_1313 = None
    mul_1314: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_132, 0.0006377551020408163)
    unsqueeze_1199: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1314, 0);  mul_1314 = None
    unsqueeze_1200: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1199, 2);  unsqueeze_1199 = None
    unsqueeze_1201: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, 3);  unsqueeze_1200 = None
    mul_1315: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_133, 0.0006377551020408163)
    mul_1316: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_1317: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1315, mul_1316);  mul_1315 = mul_1316 = None
    unsqueeze_1202: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1317, 0);  mul_1317 = None
    unsqueeze_1203: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 2);  unsqueeze_1202 = None
    unsqueeze_1204: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 3);  unsqueeze_1203 = None
    mul_1318: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_116);  primals_116 = None
    unsqueeze_1205: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1318, 0);  mul_1318 = None
    unsqueeze_1206: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 2);  unsqueeze_1205 = None
    unsqueeze_1207: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 3);  unsqueeze_1206 = None
    sub_365: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_1198);  convolution_38 = unsqueeze_1198 = None
    mul_1319: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_365, unsqueeze_1204);  sub_365 = unsqueeze_1204 = None
    sub_366: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_64, mul_1319);  where_64 = mul_1319 = None
    sub_367: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_366, unsqueeze_1201);  sub_366 = unsqueeze_1201 = None
    mul_1320: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_367, unsqueeze_1207);  sub_367 = unsqueeze_1207 = None
    mul_1321: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_115);  sum_133 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1320, relu_34, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1320 = primals_115 = None
    getitem_405: "f32[8, 2048, 14, 14]" = convolution_backward_65[0]
    getitem_406: "f32[2048, 64, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_296: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_297: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_296);  alias_296 = None
    le_65: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_297, 0);  alias_297 = None
    scalar_tensor_65: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_65: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_65, scalar_tensor_65, getitem_405);  le_65 = scalar_tensor_65 = getitem_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1208: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_1209: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, 2);  unsqueeze_1208 = None
    unsqueeze_1210: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 3);  unsqueeze_1209 = None
    sum_134: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_368: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1210)
    mul_1322: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_65, sub_368);  sub_368 = None
    sum_135: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1322, [0, 2, 3]);  mul_1322 = None
    mul_1323: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_134, 0.0006377551020408163)
    unsqueeze_1211: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1323, 0);  mul_1323 = None
    unsqueeze_1212: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1211, 2);  unsqueeze_1211 = None
    unsqueeze_1213: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, 3);  unsqueeze_1212 = None
    mul_1324: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_135, 0.0006377551020408163)
    mul_1325: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_1326: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1324, mul_1325);  mul_1324 = mul_1325 = None
    unsqueeze_1214: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1326, 0);  mul_1326 = None
    unsqueeze_1215: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 2);  unsqueeze_1214 = None
    unsqueeze_1216: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 3);  unsqueeze_1215 = None
    mul_1327: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_113);  primals_113 = None
    unsqueeze_1217: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1327, 0);  mul_1327 = None
    unsqueeze_1218: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 2);  unsqueeze_1217 = None
    unsqueeze_1219: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 3);  unsqueeze_1218 = None
    sub_369: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1210);  convolution_37 = unsqueeze_1210 = None
    mul_1328: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_369, unsqueeze_1216);  sub_369 = unsqueeze_1216 = None
    sub_370: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_65, mul_1328);  where_65 = mul_1328 = None
    sub_371: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_370, unsqueeze_1213);  sub_370 = unsqueeze_1213 = None
    mul_1329: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_371, unsqueeze_1219);  sub_371 = unsqueeze_1219 = None
    mul_1330: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_112);  sum_135 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1329, relu_33, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1329 = primals_112 = None
    getitem_408: "f32[8, 1024, 14, 14]" = convolution_backward_66[0]
    getitem_409: "f32[2048, 1024, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_574: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_63, getitem_408);  where_63 = getitem_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_299: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_300: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_299);  alias_299 = None
    le_66: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_300, 0);  alias_300 = None
    scalar_tensor_66: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_66: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_66, scalar_tensor_66, add_574);  le_66 = scalar_tensor_66 = add_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1220: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_1221: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, 2);  unsqueeze_1220 = None
    unsqueeze_1222: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 3);  unsqueeze_1221 = None
    sum_136: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_372: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1222)
    mul_1331: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_66, sub_372);  sub_372 = None
    sum_137: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1331, [0, 2, 3]);  mul_1331 = None
    mul_1332: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_136, 0.0006377551020408163)
    unsqueeze_1223: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1332, 0);  mul_1332 = None
    unsqueeze_1224: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1223, 2);  unsqueeze_1223 = None
    unsqueeze_1225: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, 3);  unsqueeze_1224 = None
    mul_1333: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_137, 0.0006377551020408163)
    mul_1334: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_1335: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1333, mul_1334);  mul_1333 = mul_1334 = None
    unsqueeze_1226: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1335, 0);  mul_1335 = None
    unsqueeze_1227: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 2);  unsqueeze_1226 = None
    unsqueeze_1228: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 3);  unsqueeze_1227 = None
    mul_1336: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_110);  primals_110 = None
    unsqueeze_1229: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1336, 0);  mul_1336 = None
    unsqueeze_1230: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 2);  unsqueeze_1229 = None
    unsqueeze_1231: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 3);  unsqueeze_1230 = None
    sub_373: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1222);  convolution_36 = unsqueeze_1222 = None
    mul_1337: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_373, unsqueeze_1228);  sub_373 = unsqueeze_1228 = None
    sub_374: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_66, mul_1337);  mul_1337 = None
    sub_375: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_374, unsqueeze_1225);  sub_374 = unsqueeze_1225 = None
    mul_1338: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_375, unsqueeze_1231);  sub_375 = unsqueeze_1231 = None
    mul_1339: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_137, squeeze_109);  sum_137 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1338, relu_32, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1338 = primals_109 = None
    getitem_411: "f32[8, 2048, 14, 14]" = convolution_backward_67[0]
    getitem_412: "f32[1024, 2048, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_302: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_303: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_302);  alias_302 = None
    le_67: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_303, 0);  alias_303 = None
    scalar_tensor_67: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_67: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_67, scalar_tensor_67, getitem_411);  le_67 = scalar_tensor_67 = getitem_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1232: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_1233: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, 2);  unsqueeze_1232 = None
    unsqueeze_1234: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 3);  unsqueeze_1233 = None
    sum_138: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_376: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_1234)
    mul_1340: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_67, sub_376);  sub_376 = None
    sum_139: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1340, [0, 2, 3]);  mul_1340 = None
    mul_1341: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_138, 0.0006377551020408163)
    unsqueeze_1235: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1341, 0);  mul_1341 = None
    unsqueeze_1236: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1235, 2);  unsqueeze_1235 = None
    unsqueeze_1237: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, 3);  unsqueeze_1236 = None
    mul_1342: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_139, 0.0006377551020408163)
    mul_1343: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_1344: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1342, mul_1343);  mul_1342 = mul_1343 = None
    unsqueeze_1238: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1344, 0);  mul_1344 = None
    unsqueeze_1239: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 2);  unsqueeze_1238 = None
    unsqueeze_1240: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 3);  unsqueeze_1239 = None
    mul_1345: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_107);  primals_107 = None
    unsqueeze_1241: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1345, 0);  mul_1345 = None
    unsqueeze_1242: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 2);  unsqueeze_1241 = None
    unsqueeze_1243: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 3);  unsqueeze_1242 = None
    sub_377: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_1234);  convolution_35 = unsqueeze_1234 = None
    mul_1346: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_377, unsqueeze_1240);  sub_377 = unsqueeze_1240 = None
    sub_378: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_67, mul_1346);  where_67 = mul_1346 = None
    sub_379: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_378, unsqueeze_1237);  sub_378 = unsqueeze_1237 = None
    mul_1347: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_379, unsqueeze_1243);  sub_379 = unsqueeze_1243 = None
    mul_1348: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_139, squeeze_106);  sum_139 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1347, relu_31, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1347 = primals_106 = None
    getitem_414: "f32[8, 2048, 14, 14]" = convolution_backward_68[0]
    getitem_415: "f32[2048, 64, 3, 3]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_305: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_306: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_305);  alias_305 = None
    le_68: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_306, 0);  alias_306 = None
    scalar_tensor_68: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_68: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_68, scalar_tensor_68, getitem_414);  le_68 = scalar_tensor_68 = getitem_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1244: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_1245: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, 2);  unsqueeze_1244 = None
    unsqueeze_1246: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 3);  unsqueeze_1245 = None
    sum_140: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_380: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_1246)
    mul_1349: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_68, sub_380);  sub_380 = None
    sum_141: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1349, [0, 2, 3]);  mul_1349 = None
    mul_1350: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_140, 0.0006377551020408163)
    unsqueeze_1247: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1350, 0);  mul_1350 = None
    unsqueeze_1248: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1247, 2);  unsqueeze_1247 = None
    unsqueeze_1249: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, 3);  unsqueeze_1248 = None
    mul_1351: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_141, 0.0006377551020408163)
    mul_1352: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_1353: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1351, mul_1352);  mul_1351 = mul_1352 = None
    unsqueeze_1250: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1353, 0);  mul_1353 = None
    unsqueeze_1251: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, 2);  unsqueeze_1250 = None
    unsqueeze_1252: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 3);  unsqueeze_1251 = None
    mul_1354: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_104);  primals_104 = None
    unsqueeze_1253: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1354, 0);  mul_1354 = None
    unsqueeze_1254: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 2);  unsqueeze_1253 = None
    unsqueeze_1255: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 3);  unsqueeze_1254 = None
    sub_381: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_1246);  convolution_34 = unsqueeze_1246 = None
    mul_1355: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_381, unsqueeze_1252);  sub_381 = unsqueeze_1252 = None
    sub_382: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_68, mul_1355);  where_68 = mul_1355 = None
    sub_383: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_382, unsqueeze_1249);  sub_382 = unsqueeze_1249 = None
    mul_1356: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_383, unsqueeze_1255);  sub_383 = unsqueeze_1255 = None
    mul_1357: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_141, squeeze_103);  sum_141 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1356, relu_30, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1356 = primals_103 = None
    getitem_417: "f32[8, 1024, 14, 14]" = convolution_backward_69[0]
    getitem_418: "f32[2048, 1024, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_575: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_66, getitem_417);  where_66 = getitem_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_308: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_309: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_308);  alias_308 = None
    le_69: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_309, 0);  alias_309 = None
    scalar_tensor_69: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_69: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_69, scalar_tensor_69, add_575);  le_69 = scalar_tensor_69 = add_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1256: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_1257: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, 2);  unsqueeze_1256 = None
    unsqueeze_1258: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 3);  unsqueeze_1257 = None
    sum_142: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_384: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1258)
    mul_1358: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_69, sub_384);  sub_384 = None
    sum_143: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1358, [0, 2, 3]);  mul_1358 = None
    mul_1359: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_142, 0.0006377551020408163)
    unsqueeze_1259: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1359, 0);  mul_1359 = None
    unsqueeze_1260: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1259, 2);  unsqueeze_1259 = None
    unsqueeze_1261: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, 3);  unsqueeze_1260 = None
    mul_1360: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_143, 0.0006377551020408163)
    mul_1361: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_1362: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1360, mul_1361);  mul_1360 = mul_1361 = None
    unsqueeze_1262: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1362, 0);  mul_1362 = None
    unsqueeze_1263: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, 2);  unsqueeze_1262 = None
    unsqueeze_1264: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 3);  unsqueeze_1263 = None
    mul_1363: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_101);  primals_101 = None
    unsqueeze_1265: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1363, 0);  mul_1363 = None
    unsqueeze_1266: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 2);  unsqueeze_1265 = None
    unsqueeze_1267: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 3);  unsqueeze_1266 = None
    sub_385: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1258);  convolution_33 = unsqueeze_1258 = None
    mul_1364: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_385, unsqueeze_1264);  sub_385 = unsqueeze_1264 = None
    sub_386: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_69, mul_1364);  mul_1364 = None
    sub_387: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_386, unsqueeze_1261);  sub_386 = unsqueeze_1261 = None
    mul_1365: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_387, unsqueeze_1267);  sub_387 = unsqueeze_1267 = None
    mul_1366: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_143, squeeze_100);  sum_143 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1365, relu_29, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1365 = primals_100 = None
    getitem_420: "f32[8, 2048, 14, 14]" = convolution_backward_70[0]
    getitem_421: "f32[1024, 2048, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_311: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_312: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_311);  alias_311 = None
    le_70: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_312, 0);  alias_312 = None
    scalar_tensor_70: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_70: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_70, scalar_tensor_70, getitem_420);  le_70 = scalar_tensor_70 = getitem_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1268: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_1269: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, 2);  unsqueeze_1268 = None
    unsqueeze_1270: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 3);  unsqueeze_1269 = None
    sum_144: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_388: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1270)
    mul_1367: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_70, sub_388);  sub_388 = None
    sum_145: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1367, [0, 2, 3]);  mul_1367 = None
    mul_1368: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_144, 0.0006377551020408163)
    unsqueeze_1271: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1368, 0);  mul_1368 = None
    unsqueeze_1272: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1271, 2);  unsqueeze_1271 = None
    unsqueeze_1273: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, 3);  unsqueeze_1272 = None
    mul_1369: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_145, 0.0006377551020408163)
    mul_1370: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_1371: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1369, mul_1370);  mul_1369 = mul_1370 = None
    unsqueeze_1274: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1371, 0);  mul_1371 = None
    unsqueeze_1275: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, 2);  unsqueeze_1274 = None
    unsqueeze_1276: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 3);  unsqueeze_1275 = None
    mul_1372: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_98);  primals_98 = None
    unsqueeze_1277: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1372, 0);  mul_1372 = None
    unsqueeze_1278: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 2);  unsqueeze_1277 = None
    unsqueeze_1279: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 3);  unsqueeze_1278 = None
    sub_389: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1270);  convolution_32 = unsqueeze_1270 = None
    mul_1373: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_389, unsqueeze_1276);  sub_389 = unsqueeze_1276 = None
    sub_390: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_70, mul_1373);  where_70 = mul_1373 = None
    sub_391: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_390, unsqueeze_1273);  sub_390 = unsqueeze_1273 = None
    mul_1374: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_391, unsqueeze_1279);  sub_391 = unsqueeze_1279 = None
    mul_1375: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_97);  sum_145 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1374, relu_28, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1374 = primals_97 = None
    getitem_423: "f32[8, 2048, 14, 14]" = convolution_backward_71[0]
    getitem_424: "f32[2048, 64, 3, 3]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_314: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_315: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_314);  alias_314 = None
    le_71: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_315, 0);  alias_315 = None
    scalar_tensor_71: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_71: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_71, scalar_tensor_71, getitem_423);  le_71 = scalar_tensor_71 = getitem_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1280: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_1281: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, 2);  unsqueeze_1280 = None
    unsqueeze_1282: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 3);  unsqueeze_1281 = None
    sum_146: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_392: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1282)
    mul_1376: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_71, sub_392);  sub_392 = None
    sum_147: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1376, [0, 2, 3]);  mul_1376 = None
    mul_1377: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_146, 0.0006377551020408163)
    unsqueeze_1283: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1377, 0);  mul_1377 = None
    unsqueeze_1284: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1283, 2);  unsqueeze_1283 = None
    unsqueeze_1285: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, 3);  unsqueeze_1284 = None
    mul_1378: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_147, 0.0006377551020408163)
    mul_1379: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_1380: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1378, mul_1379);  mul_1378 = mul_1379 = None
    unsqueeze_1286: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1380, 0);  mul_1380 = None
    unsqueeze_1287: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, 2);  unsqueeze_1286 = None
    unsqueeze_1288: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1287, 3);  unsqueeze_1287 = None
    mul_1381: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_95);  primals_95 = None
    unsqueeze_1289: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1381, 0);  mul_1381 = None
    unsqueeze_1290: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1289, 2);  unsqueeze_1289 = None
    unsqueeze_1291: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 3);  unsqueeze_1290 = None
    sub_393: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1282);  convolution_31 = unsqueeze_1282 = None
    mul_1382: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_393, unsqueeze_1288);  sub_393 = unsqueeze_1288 = None
    sub_394: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_71, mul_1382);  where_71 = mul_1382 = None
    sub_395: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_394, unsqueeze_1285);  sub_394 = unsqueeze_1285 = None
    mul_1383: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_395, unsqueeze_1291);  sub_395 = unsqueeze_1291 = None
    mul_1384: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_94);  sum_147 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1383, relu_27, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1383 = primals_94 = None
    getitem_426: "f32[8, 1024, 14, 14]" = convolution_backward_72[0]
    getitem_427: "f32[2048, 1024, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_576: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_69, getitem_426);  where_69 = getitem_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_317: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_318: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_317);  alias_317 = None
    le_72: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_318, 0);  alias_318 = None
    scalar_tensor_72: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_72: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_72, scalar_tensor_72, add_576);  le_72 = scalar_tensor_72 = add_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1292: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_1293: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, 2);  unsqueeze_1292 = None
    unsqueeze_1294: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1293, 3);  unsqueeze_1293 = None
    sum_148: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_396: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_1294)
    mul_1385: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_72, sub_396);  sub_396 = None
    sum_149: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1385, [0, 2, 3]);  mul_1385 = None
    mul_1386: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_148, 0.0006377551020408163)
    unsqueeze_1295: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1386, 0);  mul_1386 = None
    unsqueeze_1296: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1295, 2);  unsqueeze_1295 = None
    unsqueeze_1297: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, 3);  unsqueeze_1296 = None
    mul_1387: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_149, 0.0006377551020408163)
    mul_1388: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_1389: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1387, mul_1388);  mul_1387 = mul_1388 = None
    unsqueeze_1298: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1389, 0);  mul_1389 = None
    unsqueeze_1299: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, 2);  unsqueeze_1298 = None
    unsqueeze_1300: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1299, 3);  unsqueeze_1299 = None
    mul_1390: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_92);  primals_92 = None
    unsqueeze_1301: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1390, 0);  mul_1390 = None
    unsqueeze_1302: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1301, 2);  unsqueeze_1301 = None
    unsqueeze_1303: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 3);  unsqueeze_1302 = None
    sub_397: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_1294);  convolution_30 = unsqueeze_1294 = None
    mul_1391: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_397, unsqueeze_1300);  sub_397 = unsqueeze_1300 = None
    sub_398: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_72, mul_1391);  mul_1391 = None
    sub_399: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_398, unsqueeze_1297);  sub_398 = unsqueeze_1297 = None
    mul_1392: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_399, unsqueeze_1303);  sub_399 = unsqueeze_1303 = None
    mul_1393: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_149, squeeze_91);  sum_149 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1392, relu_26, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1392 = primals_91 = None
    getitem_429: "f32[8, 2048, 14, 14]" = convolution_backward_73[0]
    getitem_430: "f32[1024, 2048, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_320: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_321: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_320);  alias_320 = None
    le_73: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_321, 0);  alias_321 = None
    scalar_tensor_73: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_73: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_73, scalar_tensor_73, getitem_429);  le_73 = scalar_tensor_73 = getitem_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1304: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_1305: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, 2);  unsqueeze_1304 = None
    unsqueeze_1306: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1305, 3);  unsqueeze_1305 = None
    sum_150: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_400: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1306)
    mul_1394: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_73, sub_400);  sub_400 = None
    sum_151: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1394, [0, 2, 3]);  mul_1394 = None
    mul_1395: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_150, 0.0006377551020408163)
    unsqueeze_1307: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1395, 0);  mul_1395 = None
    unsqueeze_1308: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1307, 2);  unsqueeze_1307 = None
    unsqueeze_1309: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, 3);  unsqueeze_1308 = None
    mul_1396: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_151, 0.0006377551020408163)
    mul_1397: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_1398: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1396, mul_1397);  mul_1396 = mul_1397 = None
    unsqueeze_1310: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1398, 0);  mul_1398 = None
    unsqueeze_1311: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, 2);  unsqueeze_1310 = None
    unsqueeze_1312: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1311, 3);  unsqueeze_1311 = None
    mul_1399: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_89);  primals_89 = None
    unsqueeze_1313: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1399, 0);  mul_1399 = None
    unsqueeze_1314: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1313, 2);  unsqueeze_1313 = None
    unsqueeze_1315: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 3);  unsqueeze_1314 = None
    sub_401: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1306);  convolution_29 = unsqueeze_1306 = None
    mul_1400: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_401, unsqueeze_1312);  sub_401 = unsqueeze_1312 = None
    sub_402: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_73, mul_1400);  where_73 = mul_1400 = None
    sub_403: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_402, unsqueeze_1309);  sub_402 = unsqueeze_1309 = None
    mul_1401: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_403, unsqueeze_1315);  sub_403 = unsqueeze_1315 = None
    mul_1402: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_151, squeeze_88);  sum_151 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1401, relu_25, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1401 = primals_88 = None
    getitem_432: "f32[8, 2048, 14, 14]" = convolution_backward_74[0]
    getitem_433: "f32[2048, 64, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_323: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_324: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_323);  alias_323 = None
    le_74: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_324, 0);  alias_324 = None
    scalar_tensor_74: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_74: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_74, scalar_tensor_74, getitem_432);  le_74 = scalar_tensor_74 = getitem_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1316: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_1317: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, 2);  unsqueeze_1316 = None
    unsqueeze_1318: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1317, 3);  unsqueeze_1317 = None
    sum_152: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_404: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1318)
    mul_1403: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_74, sub_404);  sub_404 = None
    sum_153: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1403, [0, 2, 3]);  mul_1403 = None
    mul_1404: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_152, 0.0006377551020408163)
    unsqueeze_1319: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1404, 0);  mul_1404 = None
    unsqueeze_1320: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1319, 2);  unsqueeze_1319 = None
    unsqueeze_1321: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, 3);  unsqueeze_1320 = None
    mul_1405: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_153, 0.0006377551020408163)
    mul_1406: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_1407: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1405, mul_1406);  mul_1405 = mul_1406 = None
    unsqueeze_1322: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1407, 0);  mul_1407 = None
    unsqueeze_1323: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, 2);  unsqueeze_1322 = None
    unsqueeze_1324: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1323, 3);  unsqueeze_1323 = None
    mul_1408: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_86);  primals_86 = None
    unsqueeze_1325: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1408, 0);  mul_1408 = None
    unsqueeze_1326: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1325, 2);  unsqueeze_1325 = None
    unsqueeze_1327: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 3);  unsqueeze_1326 = None
    sub_405: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1318);  convolution_28 = unsqueeze_1318 = None
    mul_1409: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_405, unsqueeze_1324);  sub_405 = unsqueeze_1324 = None
    sub_406: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_74, mul_1409);  where_74 = mul_1409 = None
    sub_407: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_406, unsqueeze_1321);  sub_406 = unsqueeze_1321 = None
    mul_1410: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_407, unsqueeze_1327);  sub_407 = unsqueeze_1327 = None
    mul_1411: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_85);  sum_153 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1410, relu_24, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1410 = primals_85 = None
    getitem_435: "f32[8, 1024, 14, 14]" = convolution_backward_75[0]
    getitem_436: "f32[2048, 1024, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_577: "f32[8, 1024, 14, 14]" = torch.ops.aten.add.Tensor(where_72, getitem_435);  where_72 = getitem_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_326: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_327: "f32[8, 1024, 14, 14]" = torch.ops.aten.alias.default(alias_326);  alias_326 = None
    le_75: "b8[8, 1024, 14, 14]" = torch.ops.aten.le.Scalar(alias_327, 0);  alias_327 = None
    scalar_tensor_75: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_75: "f32[8, 1024, 14, 14]" = torch.ops.aten.where.self(le_75, scalar_tensor_75, add_577);  le_75 = scalar_tensor_75 = add_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    unsqueeze_1328: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_1329: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, 2);  unsqueeze_1328 = None
    unsqueeze_1330: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1329, 3);  unsqueeze_1329 = None
    sum_154: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_408: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1330)
    mul_1412: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_75, sub_408);  sub_408 = None
    sum_155: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1412, [0, 2, 3]);  mul_1412 = None
    mul_1413: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_154, 0.0006377551020408163)
    unsqueeze_1331: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1413, 0);  mul_1413 = None
    unsqueeze_1332: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1331, 2);  unsqueeze_1331 = None
    unsqueeze_1333: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, 3);  unsqueeze_1332 = None
    mul_1414: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_155, 0.0006377551020408163)
    mul_1415: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_1416: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1414, mul_1415);  mul_1414 = mul_1415 = None
    unsqueeze_1334: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1416, 0);  mul_1416 = None
    unsqueeze_1335: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, 2);  unsqueeze_1334 = None
    unsqueeze_1336: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1335, 3);  unsqueeze_1335 = None
    mul_1417: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_83);  primals_83 = None
    unsqueeze_1337: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1417, 0);  mul_1417 = None
    unsqueeze_1338: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1337, 2);  unsqueeze_1337 = None
    unsqueeze_1339: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 3);  unsqueeze_1338 = None
    sub_409: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1330);  convolution_27 = unsqueeze_1330 = None
    mul_1418: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_409, unsqueeze_1336);  sub_409 = unsqueeze_1336 = None
    sub_410: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_75, mul_1418);  mul_1418 = None
    sub_411: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_410, unsqueeze_1333);  sub_410 = unsqueeze_1333 = None
    mul_1419: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_411, unsqueeze_1339);  sub_411 = unsqueeze_1339 = None
    mul_1420: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_82);  sum_155 = squeeze_82 = None
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1419, relu_21, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1419 = primals_82 = None
    getitem_438: "f32[8, 512, 28, 28]" = convolution_backward_76[0]
    getitem_439: "f32[1024, 512, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1340: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_1341: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, 2);  unsqueeze_1340 = None
    unsqueeze_1342: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 3);  unsqueeze_1341 = None
    sum_156: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_412: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1342)
    mul_1421: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(where_75, sub_412);  sub_412 = None
    sum_157: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1421, [0, 2, 3]);  mul_1421 = None
    mul_1422: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_156, 0.0006377551020408163)
    unsqueeze_1343: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1422, 0);  mul_1422 = None
    unsqueeze_1344: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1343, 2);  unsqueeze_1343 = None
    unsqueeze_1345: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, 3);  unsqueeze_1344 = None
    mul_1423: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_157, 0.0006377551020408163)
    mul_1424: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_1425: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1423, mul_1424);  mul_1423 = mul_1424 = None
    unsqueeze_1346: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1425, 0);  mul_1425 = None
    unsqueeze_1347: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, 2);  unsqueeze_1346 = None
    unsqueeze_1348: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1347, 3);  unsqueeze_1347 = None
    mul_1426: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_80);  primals_80 = None
    unsqueeze_1349: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1426, 0);  mul_1426 = None
    unsqueeze_1350: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1349, 2);  unsqueeze_1349 = None
    unsqueeze_1351: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 3);  unsqueeze_1350 = None
    sub_413: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1342);  convolution_26 = unsqueeze_1342 = None
    mul_1427: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_413, unsqueeze_1348);  sub_413 = unsqueeze_1348 = None
    sub_414: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(where_75, mul_1427);  where_75 = mul_1427 = None
    sub_415: "f32[8, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(sub_414, unsqueeze_1345);  sub_414 = unsqueeze_1345 = None
    mul_1428: "f32[8, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_415, unsqueeze_1351);  sub_415 = unsqueeze_1351 = None
    mul_1429: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_79);  sum_157 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1428, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1428 = primals_79 = None
    getitem_441: "f32[8, 2048, 14, 14]" = convolution_backward_77[0]
    getitem_442: "f32[1024, 2048, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_329: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_330: "f32[8, 2048, 14, 14]" = torch.ops.aten.alias.default(alias_329);  alias_329 = None
    le_76: "b8[8, 2048, 14, 14]" = torch.ops.aten.le.Scalar(alias_330, 0);  alias_330 = None
    scalar_tensor_76: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_76: "f32[8, 2048, 14, 14]" = torch.ops.aten.where.self(le_76, scalar_tensor_76, getitem_441);  le_76 = scalar_tensor_76 = getitem_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1352: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_1353: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, 2);  unsqueeze_1352 = None
    unsqueeze_1354: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 3);  unsqueeze_1353 = None
    sum_158: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_416: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_1354)
    mul_1430: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(where_76, sub_416);  sub_416 = None
    sum_159: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1430, [0, 2, 3]);  mul_1430 = None
    mul_1431: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_158, 0.0006377551020408163)
    unsqueeze_1355: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1431, 0);  mul_1431 = None
    unsqueeze_1356: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1355, 2);  unsqueeze_1355 = None
    unsqueeze_1357: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, 3);  unsqueeze_1356 = None
    mul_1432: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_159, 0.0006377551020408163)
    mul_1433: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_1434: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1432, mul_1433);  mul_1432 = mul_1433 = None
    unsqueeze_1358: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1434, 0);  mul_1434 = None
    unsqueeze_1359: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, 2);  unsqueeze_1358 = None
    unsqueeze_1360: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1359, 3);  unsqueeze_1359 = None
    mul_1435: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_77);  primals_77 = None
    unsqueeze_1361: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1435, 0);  mul_1435 = None
    unsqueeze_1362: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1361, 2);  unsqueeze_1361 = None
    unsqueeze_1363: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 3);  unsqueeze_1362 = None
    sub_417: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_1354);  convolution_25 = unsqueeze_1354 = None
    mul_1436: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_417, unsqueeze_1360);  sub_417 = unsqueeze_1360 = None
    sub_418: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(where_76, mul_1436);  where_76 = mul_1436 = None
    sub_419: "f32[8, 2048, 14, 14]" = torch.ops.aten.sub.Tensor(sub_418, unsqueeze_1357);  sub_418 = unsqueeze_1357 = None
    mul_1437: "f32[8, 2048, 14, 14]" = torch.ops.aten.mul.Tensor(sub_419, unsqueeze_1363);  sub_419 = unsqueeze_1363 = None
    mul_1438: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_76);  sum_159 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1437, relu_22, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1437 = primals_76 = None
    getitem_444: "f32[8, 2048, 28, 28]" = convolution_backward_78[0]
    getitem_445: "f32[2048, 64, 3, 3]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_332: "f32[8, 2048, 28, 28]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_333: "f32[8, 2048, 28, 28]" = torch.ops.aten.alias.default(alias_332);  alias_332 = None
    le_77: "b8[8, 2048, 28, 28]" = torch.ops.aten.le.Scalar(alias_333, 0);  alias_333 = None
    scalar_tensor_77: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_77: "f32[8, 2048, 28, 28]" = torch.ops.aten.where.self(le_77, scalar_tensor_77, getitem_444);  le_77 = scalar_tensor_77 = getitem_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1364: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_1365: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, 2);  unsqueeze_1364 = None
    unsqueeze_1366: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 3);  unsqueeze_1365 = None
    sum_160: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_420: "f32[8, 2048, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1366)
    mul_1439: "f32[8, 2048, 28, 28]" = torch.ops.aten.mul.Tensor(where_77, sub_420);  sub_420 = None
    sum_161: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_1439, [0, 2, 3]);  mul_1439 = None
    mul_1440: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_160, 0.00015943877551020407)
    unsqueeze_1367: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1440, 0);  mul_1440 = None
    unsqueeze_1368: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1367, 2);  unsqueeze_1367 = None
    unsqueeze_1369: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, 3);  unsqueeze_1368 = None
    mul_1441: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_161, 0.00015943877551020407)
    mul_1442: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_1443: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_1441, mul_1442);  mul_1441 = mul_1442 = None
    unsqueeze_1370: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1443, 0);  mul_1443 = None
    unsqueeze_1371: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, 2);  unsqueeze_1370 = None
    unsqueeze_1372: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1371, 3);  unsqueeze_1371 = None
    mul_1444: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_74);  primals_74 = None
    unsqueeze_1373: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_1444, 0);  mul_1444 = None
    unsqueeze_1374: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1373, 2);  unsqueeze_1373 = None
    unsqueeze_1375: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 3);  unsqueeze_1374 = None
    sub_421: "f32[8, 2048, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1366);  convolution_24 = unsqueeze_1366 = None
    mul_1445: "f32[8, 2048, 28, 28]" = torch.ops.aten.mul.Tensor(sub_421, unsqueeze_1372);  sub_421 = unsqueeze_1372 = None
    sub_422: "f32[8, 2048, 28, 28]" = torch.ops.aten.sub.Tensor(where_77, mul_1445);  where_77 = mul_1445 = None
    sub_423: "f32[8, 2048, 28, 28]" = torch.ops.aten.sub.Tensor(sub_422, unsqueeze_1369);  sub_422 = unsqueeze_1369 = None
    mul_1446: "f32[8, 2048, 28, 28]" = torch.ops.aten.mul.Tensor(sub_423, unsqueeze_1375);  sub_423 = unsqueeze_1375 = None
    mul_1447: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_73);  sum_161 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1446, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1446 = primals_73 = None
    getitem_447: "f32[8, 512, 28, 28]" = convolution_backward_79[0]
    getitem_448: "f32[2048, 512, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_578: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(getitem_438, getitem_447);  getitem_438 = getitem_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_335: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_336: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_335);  alias_335 = None
    le_78: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_336, 0);  alias_336 = None
    scalar_tensor_78: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_78: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_78, scalar_tensor_78, add_578);  le_78 = scalar_tensor_78 = add_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1376: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_1377: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, 2);  unsqueeze_1376 = None
    unsqueeze_1378: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 3);  unsqueeze_1377 = None
    sum_162: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_424: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1378)
    mul_1448: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_78, sub_424);  sub_424 = None
    sum_163: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1448, [0, 2, 3]);  mul_1448 = None
    mul_1449: "f32[512]" = torch.ops.aten.mul.Tensor(sum_162, 0.00015943877551020407)
    unsqueeze_1379: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1449, 0);  mul_1449 = None
    unsqueeze_1380: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1379, 2);  unsqueeze_1379 = None
    unsqueeze_1381: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, 3);  unsqueeze_1380 = None
    mul_1450: "f32[512]" = torch.ops.aten.mul.Tensor(sum_163, 0.00015943877551020407)
    mul_1451: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_1452: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1450, mul_1451);  mul_1450 = mul_1451 = None
    unsqueeze_1382: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1452, 0);  mul_1452 = None
    unsqueeze_1383: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, 2);  unsqueeze_1382 = None
    unsqueeze_1384: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1383, 3);  unsqueeze_1383 = None
    mul_1453: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_71);  primals_71 = None
    unsqueeze_1385: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1453, 0);  mul_1453 = None
    unsqueeze_1386: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1385, 2);  unsqueeze_1385 = None
    unsqueeze_1387: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 3);  unsqueeze_1386 = None
    sub_425: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1378);  convolution_23 = unsqueeze_1378 = None
    mul_1454: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_425, unsqueeze_1384);  sub_425 = unsqueeze_1384 = None
    sub_426: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_78, mul_1454);  mul_1454 = None
    sub_427: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_426, unsqueeze_1381);  sub_426 = unsqueeze_1381 = None
    mul_1455: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_427, unsqueeze_1387);  sub_427 = unsqueeze_1387 = None
    mul_1456: "f32[512]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_70);  sum_163 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1455, relu_20, primals_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1455 = primals_70 = None
    getitem_450: "f32[8, 1024, 28, 28]" = convolution_backward_80[0]
    getitem_451: "f32[512, 1024, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_338: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_339: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_338);  alias_338 = None
    le_79: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_339, 0);  alias_339 = None
    scalar_tensor_79: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_79: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_79, scalar_tensor_79, getitem_450);  le_79 = scalar_tensor_79 = getitem_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1388: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_1389: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, 2);  unsqueeze_1388 = None
    unsqueeze_1390: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1389, 3);  unsqueeze_1389 = None
    sum_164: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_428: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1390)
    mul_1457: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_79, sub_428);  sub_428 = None
    sum_165: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1457, [0, 2, 3]);  mul_1457 = None
    mul_1458: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_164, 0.00015943877551020407)
    unsqueeze_1391: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1458, 0);  mul_1458 = None
    unsqueeze_1392: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1391, 2);  unsqueeze_1391 = None
    unsqueeze_1393: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1392, 3);  unsqueeze_1392 = None
    mul_1459: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_165, 0.00015943877551020407)
    mul_1460: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_1461: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1459, mul_1460);  mul_1459 = mul_1460 = None
    unsqueeze_1394: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1461, 0);  mul_1461 = None
    unsqueeze_1395: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, 2);  unsqueeze_1394 = None
    unsqueeze_1396: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1395, 3);  unsqueeze_1395 = None
    mul_1462: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_68);  primals_68 = None
    unsqueeze_1397: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1462, 0);  mul_1462 = None
    unsqueeze_1398: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1397, 2);  unsqueeze_1397 = None
    unsqueeze_1399: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, 3);  unsqueeze_1398 = None
    sub_429: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1390);  convolution_22 = unsqueeze_1390 = None
    mul_1463: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_429, unsqueeze_1396);  sub_429 = unsqueeze_1396 = None
    sub_430: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_79, mul_1463);  where_79 = mul_1463 = None
    sub_431: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_430, unsqueeze_1393);  sub_430 = unsqueeze_1393 = None
    mul_1464: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_431, unsqueeze_1399);  sub_431 = unsqueeze_1399 = None
    mul_1465: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_67);  sum_165 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1464, relu_19, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1464 = primals_67 = None
    getitem_453: "f32[8, 1024, 28, 28]" = convolution_backward_81[0]
    getitem_454: "f32[1024, 32, 3, 3]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_341: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_342: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_341);  alias_341 = None
    le_80: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_342, 0);  alias_342 = None
    scalar_tensor_80: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_80: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_80, scalar_tensor_80, getitem_453);  le_80 = scalar_tensor_80 = getitem_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1400: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_1401: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, 2);  unsqueeze_1400 = None
    unsqueeze_1402: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1401, 3);  unsqueeze_1401 = None
    sum_166: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_432: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1402)
    mul_1466: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_80, sub_432);  sub_432 = None
    sum_167: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1466, [0, 2, 3]);  mul_1466 = None
    mul_1467: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_166, 0.00015943877551020407)
    unsqueeze_1403: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1467, 0);  mul_1467 = None
    unsqueeze_1404: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1403, 2);  unsqueeze_1403 = None
    unsqueeze_1405: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1404, 3);  unsqueeze_1404 = None
    mul_1468: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_167, 0.00015943877551020407)
    mul_1469: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_1470: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1468, mul_1469);  mul_1468 = mul_1469 = None
    unsqueeze_1406: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1470, 0);  mul_1470 = None
    unsqueeze_1407: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, 2);  unsqueeze_1406 = None
    unsqueeze_1408: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1407, 3);  unsqueeze_1407 = None
    mul_1471: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_65);  primals_65 = None
    unsqueeze_1409: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1471, 0);  mul_1471 = None
    unsqueeze_1410: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1409, 2);  unsqueeze_1409 = None
    unsqueeze_1411: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, 3);  unsqueeze_1410 = None
    sub_433: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1402);  convolution_21 = unsqueeze_1402 = None
    mul_1472: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_433, unsqueeze_1408);  sub_433 = unsqueeze_1408 = None
    sub_434: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_80, mul_1472);  where_80 = mul_1472 = None
    sub_435: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_434, unsqueeze_1405);  sub_434 = unsqueeze_1405 = None
    mul_1473: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_435, unsqueeze_1411);  sub_435 = unsqueeze_1411 = None
    mul_1474: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_64);  sum_167 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1473, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1473 = primals_64 = None
    getitem_456: "f32[8, 512, 28, 28]" = convolution_backward_82[0]
    getitem_457: "f32[1024, 512, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_579: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_78, getitem_456);  where_78 = getitem_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_344: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_345: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_344);  alias_344 = None
    le_81: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_345, 0);  alias_345 = None
    scalar_tensor_81: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_81: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_81, scalar_tensor_81, add_579);  le_81 = scalar_tensor_81 = add_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1412: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_1413: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, 2);  unsqueeze_1412 = None
    unsqueeze_1414: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1413, 3);  unsqueeze_1413 = None
    sum_168: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
    sub_436: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_1414)
    mul_1475: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_81, sub_436);  sub_436 = None
    sum_169: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1475, [0, 2, 3]);  mul_1475 = None
    mul_1476: "f32[512]" = torch.ops.aten.mul.Tensor(sum_168, 0.00015943877551020407)
    unsqueeze_1415: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1476, 0);  mul_1476 = None
    unsqueeze_1416: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1415, 2);  unsqueeze_1415 = None
    unsqueeze_1417: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1416, 3);  unsqueeze_1416 = None
    mul_1477: "f32[512]" = torch.ops.aten.mul.Tensor(sum_169, 0.00015943877551020407)
    mul_1478: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_1479: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1477, mul_1478);  mul_1477 = mul_1478 = None
    unsqueeze_1418: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1479, 0);  mul_1479 = None
    unsqueeze_1419: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, 2);  unsqueeze_1418 = None
    unsqueeze_1420: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1419, 3);  unsqueeze_1419 = None
    mul_1480: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_62);  primals_62 = None
    unsqueeze_1421: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1480, 0);  mul_1480 = None
    unsqueeze_1422: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1421, 2);  unsqueeze_1421 = None
    unsqueeze_1423: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, 3);  unsqueeze_1422 = None
    sub_437: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_1414);  convolution_20 = unsqueeze_1414 = None
    mul_1481: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_437, unsqueeze_1420);  sub_437 = unsqueeze_1420 = None
    sub_438: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_81, mul_1481);  mul_1481 = None
    sub_439: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_438, unsqueeze_1417);  sub_438 = unsqueeze_1417 = None
    mul_1482: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_439, unsqueeze_1423);  sub_439 = unsqueeze_1423 = None
    mul_1483: "f32[512]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_61);  sum_169 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1482, relu_17, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1482 = primals_61 = None
    getitem_459: "f32[8, 1024, 28, 28]" = convolution_backward_83[0]
    getitem_460: "f32[512, 1024, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_347: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_348: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_347);  alias_347 = None
    le_82: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_348, 0);  alias_348 = None
    scalar_tensor_82: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_82: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_82, scalar_tensor_82, getitem_459);  le_82 = scalar_tensor_82 = getitem_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1424: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_1425: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, 2);  unsqueeze_1424 = None
    unsqueeze_1426: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1425, 3);  unsqueeze_1425 = None
    sum_170: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
    sub_440: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1426)
    mul_1484: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_82, sub_440);  sub_440 = None
    sum_171: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1484, [0, 2, 3]);  mul_1484 = None
    mul_1485: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_170, 0.00015943877551020407)
    unsqueeze_1427: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1485, 0);  mul_1485 = None
    unsqueeze_1428: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1427, 2);  unsqueeze_1427 = None
    unsqueeze_1429: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1428, 3);  unsqueeze_1428 = None
    mul_1486: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_171, 0.00015943877551020407)
    mul_1487: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_1488: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1486, mul_1487);  mul_1486 = mul_1487 = None
    unsqueeze_1430: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1488, 0);  mul_1488 = None
    unsqueeze_1431: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1430, 2);  unsqueeze_1430 = None
    unsqueeze_1432: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1431, 3);  unsqueeze_1431 = None
    mul_1489: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_59);  primals_59 = None
    unsqueeze_1433: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1489, 0);  mul_1489 = None
    unsqueeze_1434: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1433, 2);  unsqueeze_1433 = None
    unsqueeze_1435: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, 3);  unsqueeze_1434 = None
    sub_441: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1426);  convolution_19 = unsqueeze_1426 = None
    mul_1490: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_441, unsqueeze_1432);  sub_441 = unsqueeze_1432 = None
    sub_442: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_82, mul_1490);  where_82 = mul_1490 = None
    sub_443: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_442, unsqueeze_1429);  sub_442 = unsqueeze_1429 = None
    mul_1491: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_443, unsqueeze_1435);  sub_443 = unsqueeze_1435 = None
    mul_1492: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_58);  sum_171 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1491, relu_16, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1491 = primals_58 = None
    getitem_462: "f32[8, 1024, 28, 28]" = convolution_backward_84[0]
    getitem_463: "f32[1024, 32, 3, 3]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_350: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_351: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_350);  alias_350 = None
    le_83: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_351, 0);  alias_351 = None
    scalar_tensor_83: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_83: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_83, scalar_tensor_83, getitem_462);  le_83 = scalar_tensor_83 = getitem_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1436: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_1437: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1436, 2);  unsqueeze_1436 = None
    unsqueeze_1438: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1437, 3);  unsqueeze_1437 = None
    sum_172: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
    sub_444: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1438)
    mul_1493: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_83, sub_444);  sub_444 = None
    sum_173: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1493, [0, 2, 3]);  mul_1493 = None
    mul_1494: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_172, 0.00015943877551020407)
    unsqueeze_1439: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1494, 0);  mul_1494 = None
    unsqueeze_1440: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1439, 2);  unsqueeze_1439 = None
    unsqueeze_1441: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1440, 3);  unsqueeze_1440 = None
    mul_1495: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_173, 0.00015943877551020407)
    mul_1496: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1497: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1495, mul_1496);  mul_1495 = mul_1496 = None
    unsqueeze_1442: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1497, 0);  mul_1497 = None
    unsqueeze_1443: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1442, 2);  unsqueeze_1442 = None
    unsqueeze_1444: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1443, 3);  unsqueeze_1443 = None
    mul_1498: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_56);  primals_56 = None
    unsqueeze_1445: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1498, 0);  mul_1498 = None
    unsqueeze_1446: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1445, 2);  unsqueeze_1445 = None
    unsqueeze_1447: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, 3);  unsqueeze_1446 = None
    sub_445: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1438);  convolution_18 = unsqueeze_1438 = None
    mul_1499: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_445, unsqueeze_1444);  sub_445 = unsqueeze_1444 = None
    sub_446: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_83, mul_1499);  where_83 = mul_1499 = None
    sub_447: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_446, unsqueeze_1441);  sub_446 = unsqueeze_1441 = None
    mul_1500: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_447, unsqueeze_1447);  sub_447 = unsqueeze_1447 = None
    mul_1501: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_55);  sum_173 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1500, relu_15, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1500 = primals_55 = None
    getitem_465: "f32[8, 512, 28, 28]" = convolution_backward_85[0]
    getitem_466: "f32[1024, 512, 1, 1]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_580: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_81, getitem_465);  where_81 = getitem_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_353: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_354: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_353);  alias_353 = None
    le_84: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_354, 0);  alias_354 = None
    scalar_tensor_84: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_84: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_84, scalar_tensor_84, add_580);  le_84 = scalar_tensor_84 = add_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1448: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_1449: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, 2);  unsqueeze_1448 = None
    unsqueeze_1450: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1449, 3);  unsqueeze_1449 = None
    sum_174: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
    sub_448: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1450)
    mul_1502: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_84, sub_448);  sub_448 = None
    sum_175: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1502, [0, 2, 3]);  mul_1502 = None
    mul_1503: "f32[512]" = torch.ops.aten.mul.Tensor(sum_174, 0.00015943877551020407)
    unsqueeze_1451: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1503, 0);  mul_1503 = None
    unsqueeze_1452: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1451, 2);  unsqueeze_1451 = None
    unsqueeze_1453: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1452, 3);  unsqueeze_1452 = None
    mul_1504: "f32[512]" = torch.ops.aten.mul.Tensor(sum_175, 0.00015943877551020407)
    mul_1505: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1506: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1504, mul_1505);  mul_1504 = mul_1505 = None
    unsqueeze_1454: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1506, 0);  mul_1506 = None
    unsqueeze_1455: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, 2);  unsqueeze_1454 = None
    unsqueeze_1456: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1455, 3);  unsqueeze_1455 = None
    mul_1507: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_53);  primals_53 = None
    unsqueeze_1457: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1507, 0);  mul_1507 = None
    unsqueeze_1458: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1457, 2);  unsqueeze_1457 = None
    unsqueeze_1459: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, 3);  unsqueeze_1458 = None
    sub_449: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1450);  convolution_17 = unsqueeze_1450 = None
    mul_1508: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_449, unsqueeze_1456);  sub_449 = unsqueeze_1456 = None
    sub_450: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_84, mul_1508);  mul_1508 = None
    sub_451: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_450, unsqueeze_1453);  sub_450 = unsqueeze_1453 = None
    mul_1509: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_451, unsqueeze_1459);  sub_451 = unsqueeze_1459 = None
    mul_1510: "f32[512]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_52);  sum_175 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1509, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1509 = primals_52 = None
    getitem_468: "f32[8, 1024, 28, 28]" = convolution_backward_86[0]
    getitem_469: "f32[512, 1024, 1, 1]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_356: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_357: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_356);  alias_356 = None
    le_85: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_357, 0);  alias_357 = None
    scalar_tensor_85: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_85: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_85, scalar_tensor_85, getitem_468);  le_85 = scalar_tensor_85 = getitem_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1460: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_1461: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, 2);  unsqueeze_1460 = None
    unsqueeze_1462: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1461, 3);  unsqueeze_1461 = None
    sum_176: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_452: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1462)
    mul_1511: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_85, sub_452);  sub_452 = None
    sum_177: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1511, [0, 2, 3]);  mul_1511 = None
    mul_1512: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_176, 0.00015943877551020407)
    unsqueeze_1463: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1512, 0);  mul_1512 = None
    unsqueeze_1464: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1463, 2);  unsqueeze_1463 = None
    unsqueeze_1465: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1464, 3);  unsqueeze_1464 = None
    mul_1513: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_177, 0.00015943877551020407)
    mul_1514: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1515: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1513, mul_1514);  mul_1513 = mul_1514 = None
    unsqueeze_1466: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1515, 0);  mul_1515 = None
    unsqueeze_1467: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, 2);  unsqueeze_1466 = None
    unsqueeze_1468: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1467, 3);  unsqueeze_1467 = None
    mul_1516: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_50);  primals_50 = None
    unsqueeze_1469: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1516, 0);  mul_1516 = None
    unsqueeze_1470: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1469, 2);  unsqueeze_1469 = None
    unsqueeze_1471: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, 3);  unsqueeze_1470 = None
    sub_453: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1462);  convolution_16 = unsqueeze_1462 = None
    mul_1517: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_453, unsqueeze_1468);  sub_453 = unsqueeze_1468 = None
    sub_454: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_85, mul_1517);  where_85 = mul_1517 = None
    sub_455: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_454, unsqueeze_1465);  sub_454 = unsqueeze_1465 = None
    mul_1518: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_455, unsqueeze_1471);  sub_455 = unsqueeze_1471 = None
    mul_1519: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_49);  sum_177 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1518, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1518 = primals_49 = None
    getitem_471: "f32[8, 1024, 28, 28]" = convolution_backward_87[0]
    getitem_472: "f32[1024, 32, 3, 3]" = convolution_backward_87[1];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_359: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_360: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_359);  alias_359 = None
    le_86: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_360, 0);  alias_360 = None
    scalar_tensor_86: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_86: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_86, scalar_tensor_86, getitem_471);  le_86 = scalar_tensor_86 = getitem_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1472: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_1473: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, 2);  unsqueeze_1472 = None
    unsqueeze_1474: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1473, 3);  unsqueeze_1473 = None
    sum_178: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
    sub_456: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1474)
    mul_1520: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_86, sub_456);  sub_456 = None
    sum_179: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1520, [0, 2, 3]);  mul_1520 = None
    mul_1521: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_178, 0.00015943877551020407)
    unsqueeze_1475: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1521, 0);  mul_1521 = None
    unsqueeze_1476: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1475, 2);  unsqueeze_1475 = None
    unsqueeze_1477: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1476, 3);  unsqueeze_1476 = None
    mul_1522: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_179, 0.00015943877551020407)
    mul_1523: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1524: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1522, mul_1523);  mul_1522 = mul_1523 = None
    unsqueeze_1478: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1524, 0);  mul_1524 = None
    unsqueeze_1479: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, 2);  unsqueeze_1478 = None
    unsqueeze_1480: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1479, 3);  unsqueeze_1479 = None
    mul_1525: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_47);  primals_47 = None
    unsqueeze_1481: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1525, 0);  mul_1525 = None
    unsqueeze_1482: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1481, 2);  unsqueeze_1481 = None
    unsqueeze_1483: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, 3);  unsqueeze_1482 = None
    sub_457: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1474);  convolution_15 = unsqueeze_1474 = None
    mul_1526: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_457, unsqueeze_1480);  sub_457 = unsqueeze_1480 = None
    sub_458: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_86, mul_1526);  where_86 = mul_1526 = None
    sub_459: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_458, unsqueeze_1477);  sub_458 = unsqueeze_1477 = None
    mul_1527: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_459, unsqueeze_1483);  sub_459 = unsqueeze_1483 = None
    mul_1528: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_46);  sum_179 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1527, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1527 = primals_46 = None
    getitem_474: "f32[8, 512, 28, 28]" = convolution_backward_88[0]
    getitem_475: "f32[1024, 512, 1, 1]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_581: "f32[8, 512, 28, 28]" = torch.ops.aten.add.Tensor(where_84, getitem_474);  where_84 = getitem_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_362: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_363: "f32[8, 512, 28, 28]" = torch.ops.aten.alias.default(alias_362);  alias_362 = None
    le_87: "b8[8, 512, 28, 28]" = torch.ops.aten.le.Scalar(alias_363, 0);  alias_363 = None
    scalar_tensor_87: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_87: "f32[8, 512, 28, 28]" = torch.ops.aten.where.self(le_87, scalar_tensor_87, add_581);  le_87 = scalar_tensor_87 = add_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    unsqueeze_1484: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_1485: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, 2);  unsqueeze_1484 = None
    unsqueeze_1486: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1485, 3);  unsqueeze_1485 = None
    sum_180: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_460: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1486)
    mul_1529: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_87, sub_460);  sub_460 = None
    sum_181: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1529, [0, 2, 3]);  mul_1529 = None
    mul_1530: "f32[512]" = torch.ops.aten.mul.Tensor(sum_180, 0.00015943877551020407)
    unsqueeze_1487: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1530, 0);  mul_1530 = None
    unsqueeze_1488: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1487, 2);  unsqueeze_1487 = None
    unsqueeze_1489: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1488, 3);  unsqueeze_1488 = None
    mul_1531: "f32[512]" = torch.ops.aten.mul.Tensor(sum_181, 0.00015943877551020407)
    mul_1532: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1533: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1531, mul_1532);  mul_1531 = mul_1532 = None
    unsqueeze_1490: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1533, 0);  mul_1533 = None
    unsqueeze_1491: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, 2);  unsqueeze_1490 = None
    unsqueeze_1492: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1491, 3);  unsqueeze_1491 = None
    mul_1534: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_44);  primals_44 = None
    unsqueeze_1493: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1534, 0);  mul_1534 = None
    unsqueeze_1494: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1493, 2);  unsqueeze_1493 = None
    unsqueeze_1495: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, 3);  unsqueeze_1494 = None
    sub_461: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1486);  convolution_14 = unsqueeze_1486 = None
    mul_1535: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_461, unsqueeze_1492);  sub_461 = unsqueeze_1492 = None
    sub_462: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_87, mul_1535);  mul_1535 = None
    sub_463: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_462, unsqueeze_1489);  sub_462 = unsqueeze_1489 = None
    mul_1536: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_463, unsqueeze_1495);  sub_463 = unsqueeze_1495 = None
    mul_1537: "f32[512]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_43);  sum_181 = squeeze_43 = None
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1536, relu_9, primals_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1536 = primals_43 = None
    getitem_477: "f32[8, 256, 56, 56]" = convolution_backward_89[0]
    getitem_478: "f32[512, 256, 1, 1]" = convolution_backward_89[1];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1496: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_1497: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1496, 2);  unsqueeze_1496 = None
    unsqueeze_1498: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1497, 3);  unsqueeze_1497 = None
    sum_182: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_464: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1498)
    mul_1538: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_87, sub_464);  sub_464 = None
    sum_183: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1538, [0, 2, 3]);  mul_1538 = None
    mul_1539: "f32[512]" = torch.ops.aten.mul.Tensor(sum_182, 0.00015943877551020407)
    unsqueeze_1499: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1539, 0);  mul_1539 = None
    unsqueeze_1500: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1499, 2);  unsqueeze_1499 = None
    unsqueeze_1501: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1500, 3);  unsqueeze_1500 = None
    mul_1540: "f32[512]" = torch.ops.aten.mul.Tensor(sum_183, 0.00015943877551020407)
    mul_1541: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1542: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1540, mul_1541);  mul_1540 = mul_1541 = None
    unsqueeze_1502: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1542, 0);  mul_1542 = None
    unsqueeze_1503: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1502, 2);  unsqueeze_1502 = None
    unsqueeze_1504: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1503, 3);  unsqueeze_1503 = None
    mul_1543: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_41);  primals_41 = None
    unsqueeze_1505: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1543, 0);  mul_1543 = None
    unsqueeze_1506: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1505, 2);  unsqueeze_1505 = None
    unsqueeze_1507: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, 3);  unsqueeze_1506 = None
    sub_465: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1498);  convolution_13 = unsqueeze_1498 = None
    mul_1544: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_465, unsqueeze_1504);  sub_465 = unsqueeze_1504 = None
    sub_466: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_87, mul_1544);  where_87 = mul_1544 = None
    sub_467: "f32[8, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_466, unsqueeze_1501);  sub_466 = unsqueeze_1501 = None
    mul_1545: "f32[8, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_467, unsqueeze_1507);  sub_467 = unsqueeze_1507 = None
    mul_1546: "f32[512]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_40);  sum_183 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1545, relu_11, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1545 = primals_40 = None
    getitem_480: "f32[8, 1024, 28, 28]" = convolution_backward_90[0]
    getitem_481: "f32[512, 1024, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_365: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_366: "f32[8, 1024, 28, 28]" = torch.ops.aten.alias.default(alias_365);  alias_365 = None
    le_88: "b8[8, 1024, 28, 28]" = torch.ops.aten.le.Scalar(alias_366, 0);  alias_366 = None
    scalar_tensor_88: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_88: "f32[8, 1024, 28, 28]" = torch.ops.aten.where.self(le_88, scalar_tensor_88, getitem_480);  le_88 = scalar_tensor_88 = getitem_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1508: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_1509: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1508, 2);  unsqueeze_1508 = None
    unsqueeze_1510: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1509, 3);  unsqueeze_1509 = None
    sum_184: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_468: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1510)
    mul_1547: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(where_88, sub_468);  sub_468 = None
    sum_185: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1547, [0, 2, 3]);  mul_1547 = None
    mul_1548: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_184, 0.00015943877551020407)
    unsqueeze_1511: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1548, 0);  mul_1548 = None
    unsqueeze_1512: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1511, 2);  unsqueeze_1511 = None
    unsqueeze_1513: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1512, 3);  unsqueeze_1512 = None
    mul_1549: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_185, 0.00015943877551020407)
    mul_1550: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1551: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1549, mul_1550);  mul_1549 = mul_1550 = None
    unsqueeze_1514: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1551, 0);  mul_1551 = None
    unsqueeze_1515: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1514, 2);  unsqueeze_1514 = None
    unsqueeze_1516: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1515, 3);  unsqueeze_1515 = None
    mul_1552: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_38);  primals_38 = None
    unsqueeze_1517: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1552, 0);  mul_1552 = None
    unsqueeze_1518: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1517, 2);  unsqueeze_1517 = None
    unsqueeze_1519: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, 3);  unsqueeze_1518 = None
    sub_469: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1510);  convolution_12 = unsqueeze_1510 = None
    mul_1553: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_469, unsqueeze_1516);  sub_469 = unsqueeze_1516 = None
    sub_470: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(where_88, mul_1553);  where_88 = mul_1553 = None
    sub_471: "f32[8, 1024, 28, 28]" = torch.ops.aten.sub.Tensor(sub_470, unsqueeze_1513);  sub_470 = unsqueeze_1513 = None
    mul_1554: "f32[8, 1024, 28, 28]" = torch.ops.aten.mul.Tensor(sub_471, unsqueeze_1519);  sub_471 = unsqueeze_1519 = None
    mul_1555: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_185, squeeze_37);  sum_185 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1554, relu_10, primals_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1554 = primals_37 = None
    getitem_483: "f32[8, 1024, 56, 56]" = convolution_backward_91[0]
    getitem_484: "f32[1024, 32, 3, 3]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_368: "f32[8, 1024, 56, 56]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_369: "f32[8, 1024, 56, 56]" = torch.ops.aten.alias.default(alias_368);  alias_368 = None
    le_89: "b8[8, 1024, 56, 56]" = torch.ops.aten.le.Scalar(alias_369, 0);  alias_369 = None
    scalar_tensor_89: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_89: "f32[8, 1024, 56, 56]" = torch.ops.aten.where.self(le_89, scalar_tensor_89, getitem_483);  le_89 = scalar_tensor_89 = getitem_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1520: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_1521: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1520, 2);  unsqueeze_1520 = None
    unsqueeze_1522: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1521, 3);  unsqueeze_1521 = None
    sum_186: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
    sub_472: "f32[8, 1024, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1522)
    mul_1556: "f32[8, 1024, 56, 56]" = torch.ops.aten.mul.Tensor(where_89, sub_472);  sub_472 = None
    sum_187: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1556, [0, 2, 3]);  mul_1556 = None
    mul_1557: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_186, 3.985969387755102e-05)
    unsqueeze_1523: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1557, 0);  mul_1557 = None
    unsqueeze_1524: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1523, 2);  unsqueeze_1523 = None
    unsqueeze_1525: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1524, 3);  unsqueeze_1524 = None
    mul_1558: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_187, 3.985969387755102e-05)
    mul_1559: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1560: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_1558, mul_1559);  mul_1558 = mul_1559 = None
    unsqueeze_1526: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1560, 0);  mul_1560 = None
    unsqueeze_1527: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1526, 2);  unsqueeze_1526 = None
    unsqueeze_1528: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1527, 3);  unsqueeze_1527 = None
    mul_1561: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_35);  primals_35 = None
    unsqueeze_1529: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_1561, 0);  mul_1561 = None
    unsqueeze_1530: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1529, 2);  unsqueeze_1529 = None
    unsqueeze_1531: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, 3);  unsqueeze_1530 = None
    sub_473: "f32[8, 1024, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1522);  convolution_11 = unsqueeze_1522 = None
    mul_1562: "f32[8, 1024, 56, 56]" = torch.ops.aten.mul.Tensor(sub_473, unsqueeze_1528);  sub_473 = unsqueeze_1528 = None
    sub_474: "f32[8, 1024, 56, 56]" = torch.ops.aten.sub.Tensor(where_89, mul_1562);  where_89 = mul_1562 = None
    sub_475: "f32[8, 1024, 56, 56]" = torch.ops.aten.sub.Tensor(sub_474, unsqueeze_1525);  sub_474 = unsqueeze_1525 = None
    mul_1563: "f32[8, 1024, 56, 56]" = torch.ops.aten.mul.Tensor(sub_475, unsqueeze_1531);  sub_475 = unsqueeze_1531 = None
    mul_1564: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_187, squeeze_34);  sum_187 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1563, relu_9, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1563 = primals_34 = None
    getitem_486: "f32[8, 256, 56, 56]" = convolution_backward_92[0]
    getitem_487: "f32[1024, 256, 1, 1]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_582: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(getitem_477, getitem_486);  getitem_477 = getitem_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_371: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_372: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_371);  alias_371 = None
    le_90: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_372, 0);  alias_372 = None
    scalar_tensor_90: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_90: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_90, scalar_tensor_90, add_582);  le_90 = scalar_tensor_90 = add_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1532: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_1533: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1532, 2);  unsqueeze_1532 = None
    unsqueeze_1534: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1533, 3);  unsqueeze_1533 = None
    sum_188: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
    sub_476: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1534)
    mul_1565: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_90, sub_476);  sub_476 = None
    sum_189: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1565, [0, 2, 3]);  mul_1565 = None
    mul_1566: "f32[256]" = torch.ops.aten.mul.Tensor(sum_188, 3.985969387755102e-05)
    unsqueeze_1535: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1566, 0);  mul_1566 = None
    unsqueeze_1536: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1535, 2);  unsqueeze_1535 = None
    unsqueeze_1537: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1536, 3);  unsqueeze_1536 = None
    mul_1567: "f32[256]" = torch.ops.aten.mul.Tensor(sum_189, 3.985969387755102e-05)
    mul_1568: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1569: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1567, mul_1568);  mul_1567 = mul_1568 = None
    unsqueeze_1538: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1569, 0);  mul_1569 = None
    unsqueeze_1539: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1538, 2);  unsqueeze_1538 = None
    unsqueeze_1540: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1539, 3);  unsqueeze_1539 = None
    mul_1570: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_32);  primals_32 = None
    unsqueeze_1541: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1570, 0);  mul_1570 = None
    unsqueeze_1542: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1541, 2);  unsqueeze_1541 = None
    unsqueeze_1543: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, 3);  unsqueeze_1542 = None
    sub_477: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1534);  convolution_10 = unsqueeze_1534 = None
    mul_1571: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_477, unsqueeze_1540);  sub_477 = unsqueeze_1540 = None
    sub_478: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_90, mul_1571);  mul_1571 = None
    sub_479: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_478, unsqueeze_1537);  sub_478 = unsqueeze_1537 = None
    mul_1572: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_479, unsqueeze_1543);  sub_479 = unsqueeze_1543 = None
    mul_1573: "f32[256]" = torch.ops.aten.mul.Tensor(sum_189, squeeze_31);  sum_189 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1572, relu_8, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1572 = primals_31 = None
    getitem_489: "f32[8, 512, 56, 56]" = convolution_backward_93[0]
    getitem_490: "f32[256, 512, 1, 1]" = convolution_backward_93[1];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_374: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_375: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(alias_374);  alias_374 = None
    le_91: "b8[8, 512, 56, 56]" = torch.ops.aten.le.Scalar(alias_375, 0);  alias_375 = None
    scalar_tensor_91: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_91: "f32[8, 512, 56, 56]" = torch.ops.aten.where.self(le_91, scalar_tensor_91, getitem_489);  le_91 = scalar_tensor_91 = getitem_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1544: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_1545: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1544, 2);  unsqueeze_1544 = None
    unsqueeze_1546: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1545, 3);  unsqueeze_1545 = None
    sum_190: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
    sub_480: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1546)
    mul_1574: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(where_91, sub_480);  sub_480 = None
    sum_191: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1574, [0, 2, 3]);  mul_1574 = None
    mul_1575: "f32[512]" = torch.ops.aten.mul.Tensor(sum_190, 3.985969387755102e-05)
    unsqueeze_1547: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1575, 0);  mul_1575 = None
    unsqueeze_1548: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1547, 2);  unsqueeze_1547 = None
    unsqueeze_1549: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1548, 3);  unsqueeze_1548 = None
    mul_1576: "f32[512]" = torch.ops.aten.mul.Tensor(sum_191, 3.985969387755102e-05)
    mul_1577: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1578: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1576, mul_1577);  mul_1576 = mul_1577 = None
    unsqueeze_1550: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1578, 0);  mul_1578 = None
    unsqueeze_1551: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1550, 2);  unsqueeze_1550 = None
    unsqueeze_1552: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1551, 3);  unsqueeze_1551 = None
    mul_1579: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_29);  primals_29 = None
    unsqueeze_1553: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1579, 0);  mul_1579 = None
    unsqueeze_1554: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1553, 2);  unsqueeze_1553 = None
    unsqueeze_1555: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, 3);  unsqueeze_1554 = None
    sub_481: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1546);  convolution_9 = unsqueeze_1546 = None
    mul_1580: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_481, unsqueeze_1552);  sub_481 = unsqueeze_1552 = None
    sub_482: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(where_91, mul_1580);  where_91 = mul_1580 = None
    sub_483: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(sub_482, unsqueeze_1549);  sub_482 = unsqueeze_1549 = None
    mul_1581: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_483, unsqueeze_1555);  sub_483 = unsqueeze_1555 = None
    mul_1582: "f32[512]" = torch.ops.aten.mul.Tensor(sum_191, squeeze_28);  sum_191 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1581, relu_7, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1581 = primals_28 = None
    getitem_492: "f32[8, 512, 56, 56]" = convolution_backward_94[0]
    getitem_493: "f32[512, 16, 3, 3]" = convolution_backward_94[1];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_377: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_378: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(alias_377);  alias_377 = None
    le_92: "b8[8, 512, 56, 56]" = torch.ops.aten.le.Scalar(alias_378, 0);  alias_378 = None
    scalar_tensor_92: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_92: "f32[8, 512, 56, 56]" = torch.ops.aten.where.self(le_92, scalar_tensor_92, getitem_492);  le_92 = scalar_tensor_92 = getitem_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1556: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_1557: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1556, 2);  unsqueeze_1556 = None
    unsqueeze_1558: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1557, 3);  unsqueeze_1557 = None
    sum_192: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_484: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1558)
    mul_1583: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(where_92, sub_484);  sub_484 = None
    sum_193: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1583, [0, 2, 3]);  mul_1583 = None
    mul_1584: "f32[512]" = torch.ops.aten.mul.Tensor(sum_192, 3.985969387755102e-05)
    unsqueeze_1559: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1584, 0);  mul_1584 = None
    unsqueeze_1560: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1559, 2);  unsqueeze_1559 = None
    unsqueeze_1561: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1560, 3);  unsqueeze_1560 = None
    mul_1585: "f32[512]" = torch.ops.aten.mul.Tensor(sum_193, 3.985969387755102e-05)
    mul_1586: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1587: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1585, mul_1586);  mul_1585 = mul_1586 = None
    unsqueeze_1562: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1587, 0);  mul_1587 = None
    unsqueeze_1563: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1562, 2);  unsqueeze_1562 = None
    unsqueeze_1564: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1563, 3);  unsqueeze_1563 = None
    mul_1588: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_26);  primals_26 = None
    unsqueeze_1565: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1588, 0);  mul_1588 = None
    unsqueeze_1566: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1565, 2);  unsqueeze_1565 = None
    unsqueeze_1567: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, 3);  unsqueeze_1566 = None
    sub_485: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1558);  convolution_8 = unsqueeze_1558 = None
    mul_1589: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_485, unsqueeze_1564);  sub_485 = unsqueeze_1564 = None
    sub_486: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(where_92, mul_1589);  where_92 = mul_1589 = None
    sub_487: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(sub_486, unsqueeze_1561);  sub_486 = unsqueeze_1561 = None
    mul_1590: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_487, unsqueeze_1567);  sub_487 = unsqueeze_1567 = None
    mul_1591: "f32[512]" = torch.ops.aten.mul.Tensor(sum_193, squeeze_25);  sum_193 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1590, relu_6, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1590 = primals_25 = None
    getitem_495: "f32[8, 256, 56, 56]" = convolution_backward_95[0]
    getitem_496: "f32[512, 256, 1, 1]" = convolution_backward_95[1];  convolution_backward_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_583: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_90, getitem_495);  where_90 = getitem_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_380: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_381: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_380);  alias_380 = None
    le_93: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_381, 0);  alias_381 = None
    scalar_tensor_93: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_93: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_93, scalar_tensor_93, add_583);  le_93 = scalar_tensor_93 = add_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1568: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_1569: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1568, 2);  unsqueeze_1568 = None
    unsqueeze_1570: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1569, 3);  unsqueeze_1569 = None
    sum_194: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
    sub_488: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1570)
    mul_1592: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_93, sub_488);  sub_488 = None
    sum_195: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1592, [0, 2, 3]);  mul_1592 = None
    mul_1593: "f32[256]" = torch.ops.aten.mul.Tensor(sum_194, 3.985969387755102e-05)
    unsqueeze_1571: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1593, 0);  mul_1593 = None
    unsqueeze_1572: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1571, 2);  unsqueeze_1571 = None
    unsqueeze_1573: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1572, 3);  unsqueeze_1572 = None
    mul_1594: "f32[256]" = torch.ops.aten.mul.Tensor(sum_195, 3.985969387755102e-05)
    mul_1595: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1596: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1594, mul_1595);  mul_1594 = mul_1595 = None
    unsqueeze_1574: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1596, 0);  mul_1596 = None
    unsqueeze_1575: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1574, 2);  unsqueeze_1574 = None
    unsqueeze_1576: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1575, 3);  unsqueeze_1575 = None
    mul_1597: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_23);  primals_23 = None
    unsqueeze_1577: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1597, 0);  mul_1597 = None
    unsqueeze_1578: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1577, 2);  unsqueeze_1577 = None
    unsqueeze_1579: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, 3);  unsqueeze_1578 = None
    sub_489: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1570);  convolution_7 = unsqueeze_1570 = None
    mul_1598: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_489, unsqueeze_1576);  sub_489 = unsqueeze_1576 = None
    sub_490: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_93, mul_1598);  mul_1598 = None
    sub_491: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_490, unsqueeze_1573);  sub_490 = unsqueeze_1573 = None
    mul_1599: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_491, unsqueeze_1579);  sub_491 = unsqueeze_1579 = None
    mul_1600: "f32[256]" = torch.ops.aten.mul.Tensor(sum_195, squeeze_22);  sum_195 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(mul_1599, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1599 = primals_22 = None
    getitem_498: "f32[8, 512, 56, 56]" = convolution_backward_96[0]
    getitem_499: "f32[256, 512, 1, 1]" = convolution_backward_96[1];  convolution_backward_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_383: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_384: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(alias_383);  alias_383 = None
    le_94: "b8[8, 512, 56, 56]" = torch.ops.aten.le.Scalar(alias_384, 0);  alias_384 = None
    scalar_tensor_94: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_94: "f32[8, 512, 56, 56]" = torch.ops.aten.where.self(le_94, scalar_tensor_94, getitem_498);  le_94 = scalar_tensor_94 = getitem_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1580: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_1581: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1580, 2);  unsqueeze_1580 = None
    unsqueeze_1582: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1581, 3);  unsqueeze_1581 = None
    sum_196: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_492: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1582)
    mul_1601: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(where_94, sub_492);  sub_492 = None
    sum_197: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1601, [0, 2, 3]);  mul_1601 = None
    mul_1602: "f32[512]" = torch.ops.aten.mul.Tensor(sum_196, 3.985969387755102e-05)
    unsqueeze_1583: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1602, 0);  mul_1602 = None
    unsqueeze_1584: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1583, 2);  unsqueeze_1583 = None
    unsqueeze_1585: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1584, 3);  unsqueeze_1584 = None
    mul_1603: "f32[512]" = torch.ops.aten.mul.Tensor(sum_197, 3.985969387755102e-05)
    mul_1604: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1605: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1603, mul_1604);  mul_1603 = mul_1604 = None
    unsqueeze_1586: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1605, 0);  mul_1605 = None
    unsqueeze_1587: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1586, 2);  unsqueeze_1586 = None
    unsqueeze_1588: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1587, 3);  unsqueeze_1587 = None
    mul_1606: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_20);  primals_20 = None
    unsqueeze_1589: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1606, 0);  mul_1606 = None
    unsqueeze_1590: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1589, 2);  unsqueeze_1589 = None
    unsqueeze_1591: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, 3);  unsqueeze_1590 = None
    sub_493: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1582);  convolution_6 = unsqueeze_1582 = None
    mul_1607: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_493, unsqueeze_1588);  sub_493 = unsqueeze_1588 = None
    sub_494: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(where_94, mul_1607);  where_94 = mul_1607 = None
    sub_495: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(sub_494, unsqueeze_1585);  sub_494 = unsqueeze_1585 = None
    mul_1608: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_495, unsqueeze_1591);  sub_495 = unsqueeze_1591 = None
    mul_1609: "f32[512]" = torch.ops.aten.mul.Tensor(sum_197, squeeze_19);  sum_197 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(mul_1608, relu_4, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1608 = primals_19 = None
    getitem_501: "f32[8, 512, 56, 56]" = convolution_backward_97[0]
    getitem_502: "f32[512, 16, 3, 3]" = convolution_backward_97[1];  convolution_backward_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_386: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_387: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(alias_386);  alias_386 = None
    le_95: "b8[8, 512, 56, 56]" = torch.ops.aten.le.Scalar(alias_387, 0);  alias_387 = None
    scalar_tensor_95: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_95: "f32[8, 512, 56, 56]" = torch.ops.aten.where.self(le_95, scalar_tensor_95, getitem_501);  le_95 = scalar_tensor_95 = getitem_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1592: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_1593: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1592, 2);  unsqueeze_1592 = None
    unsqueeze_1594: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1593, 3);  unsqueeze_1593 = None
    sum_198: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
    sub_496: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1594)
    mul_1610: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(where_95, sub_496);  sub_496 = None
    sum_199: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1610, [0, 2, 3]);  mul_1610 = None
    mul_1611: "f32[512]" = torch.ops.aten.mul.Tensor(sum_198, 3.985969387755102e-05)
    unsqueeze_1595: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1611, 0);  mul_1611 = None
    unsqueeze_1596: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1595, 2);  unsqueeze_1595 = None
    unsqueeze_1597: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1596, 3);  unsqueeze_1596 = None
    mul_1612: "f32[512]" = torch.ops.aten.mul.Tensor(sum_199, 3.985969387755102e-05)
    mul_1613: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1614: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1612, mul_1613);  mul_1612 = mul_1613 = None
    unsqueeze_1598: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1614, 0);  mul_1614 = None
    unsqueeze_1599: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1598, 2);  unsqueeze_1598 = None
    unsqueeze_1600: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1599, 3);  unsqueeze_1599 = None
    mul_1615: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_17);  primals_17 = None
    unsqueeze_1601: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1615, 0);  mul_1615 = None
    unsqueeze_1602: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1601, 2);  unsqueeze_1601 = None
    unsqueeze_1603: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, 3);  unsqueeze_1602 = None
    sub_497: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1594);  convolution_5 = unsqueeze_1594 = None
    mul_1616: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_497, unsqueeze_1600);  sub_497 = unsqueeze_1600 = None
    sub_498: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(where_95, mul_1616);  where_95 = mul_1616 = None
    sub_499: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(sub_498, unsqueeze_1597);  sub_498 = unsqueeze_1597 = None
    mul_1617: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_499, unsqueeze_1603);  sub_499 = unsqueeze_1603 = None
    mul_1618: "f32[512]" = torch.ops.aten.mul.Tensor(sum_199, squeeze_16);  sum_199 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(mul_1617, relu_3, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1617 = primals_16 = None
    getitem_504: "f32[8, 256, 56, 56]" = convolution_backward_98[0]
    getitem_505: "f32[512, 256, 1, 1]" = convolution_backward_98[1];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_584: "f32[8, 256, 56, 56]" = torch.ops.aten.add.Tensor(where_93, getitem_504);  where_93 = getitem_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:202, code: x = self.act3(x)
    alias_389: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_390: "f32[8, 256, 56, 56]" = torch.ops.aten.alias.default(alias_389);  alias_389 = None
    le_96: "b8[8, 256, 56, 56]" = torch.ops.aten.le.Scalar(alias_390, 0);  alias_390 = None
    scalar_tensor_96: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_96: "f32[8, 256, 56, 56]" = torch.ops.aten.where.self(le_96, scalar_tensor_96, add_584);  le_96 = scalar_tensor_96 = add_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:200, code: shortcut = self.downsample(shortcut)
    unsqueeze_1604: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_1605: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1604, 2);  unsqueeze_1604 = None
    unsqueeze_1606: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1605, 3);  unsqueeze_1605 = None
    sum_200: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_500: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1606)
    mul_1619: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_96, sub_500);  sub_500 = None
    sum_201: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1619, [0, 2, 3]);  mul_1619 = None
    mul_1620: "f32[256]" = torch.ops.aten.mul.Tensor(sum_200, 3.985969387755102e-05)
    unsqueeze_1607: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1620, 0);  mul_1620 = None
    unsqueeze_1608: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1607, 2);  unsqueeze_1607 = None
    unsqueeze_1609: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1608, 3);  unsqueeze_1608 = None
    mul_1621: "f32[256]" = torch.ops.aten.mul.Tensor(sum_201, 3.985969387755102e-05)
    mul_1622: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1623: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1621, mul_1622);  mul_1621 = mul_1622 = None
    unsqueeze_1610: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1623, 0);  mul_1623 = None
    unsqueeze_1611: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1610, 2);  unsqueeze_1610 = None
    unsqueeze_1612: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1611, 3);  unsqueeze_1611 = None
    mul_1624: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
    unsqueeze_1613: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1624, 0);  mul_1624 = None
    unsqueeze_1614: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1613, 2);  unsqueeze_1613 = None
    unsqueeze_1615: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, 3);  unsqueeze_1614 = None
    sub_501: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1606);  convolution_4 = unsqueeze_1606 = None
    mul_1625: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_501, unsqueeze_1612);  sub_501 = unsqueeze_1612 = None
    sub_502: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_96, mul_1625);  mul_1625 = None
    sub_503: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_502, unsqueeze_1609);  sub_502 = unsqueeze_1609 = None
    mul_1626: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_503, unsqueeze_1615);  sub_503 = unsqueeze_1615 = None
    mul_1627: "f32[256]" = torch.ops.aten.mul.Tensor(sum_201, squeeze_13);  sum_201 = squeeze_13 = None
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(mul_1626, getitem_2, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1626 = primals_13 = None
    getitem_507: "f32[8, 64, 56, 56]" = convolution_backward_99[0]
    getitem_508: "f32[256, 64, 1, 1]" = convolution_backward_99[1];  convolution_backward_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:191, code: x = self.bn3(x)
    unsqueeze_1616: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_1617: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1616, 2);  unsqueeze_1616 = None
    unsqueeze_1618: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1617, 3);  unsqueeze_1617 = None
    sum_202: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_504: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1618)
    mul_1628: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_96, sub_504);  sub_504 = None
    sum_203: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1628, [0, 2, 3]);  mul_1628 = None
    mul_1629: "f32[256]" = torch.ops.aten.mul.Tensor(sum_202, 3.985969387755102e-05)
    unsqueeze_1619: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1629, 0);  mul_1629 = None
    unsqueeze_1620: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1619, 2);  unsqueeze_1619 = None
    unsqueeze_1621: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1620, 3);  unsqueeze_1620 = None
    mul_1630: "f32[256]" = torch.ops.aten.mul.Tensor(sum_203, 3.985969387755102e-05)
    mul_1631: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1632: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1630, mul_1631);  mul_1630 = mul_1631 = None
    unsqueeze_1622: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1632, 0);  mul_1632 = None
    unsqueeze_1623: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1622, 2);  unsqueeze_1622 = None
    unsqueeze_1624: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1623, 3);  unsqueeze_1623 = None
    mul_1633: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
    unsqueeze_1625: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1633, 0);  mul_1633 = None
    unsqueeze_1626: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1625, 2);  unsqueeze_1625 = None
    unsqueeze_1627: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, 3);  unsqueeze_1626 = None
    sub_505: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1618);  convolution_3 = unsqueeze_1618 = None
    mul_1634: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_505, unsqueeze_1624);  sub_505 = unsqueeze_1624 = None
    sub_506: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_96, mul_1634);  where_96 = mul_1634 = None
    sub_507: "f32[8, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_506, unsqueeze_1621);  sub_506 = unsqueeze_1621 = None
    mul_1635: "f32[8, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_507, unsqueeze_1627);  sub_507 = unsqueeze_1627 = None
    mul_1636: "f32[256]" = torch.ops.aten.mul.Tensor(sum_203, squeeze_10);  sum_203 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:190, code: x = self.conv3(x)
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(mul_1635, relu_2, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1635 = primals_10 = None
    getitem_510: "f32[8, 512, 56, 56]" = convolution_backward_100[0]
    getitem_511: "f32[256, 512, 1, 1]" = convolution_backward_100[1];  convolution_backward_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:187, code: x = self.act2(x)
    alias_392: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_393: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(alias_392);  alias_392 = None
    le_97: "b8[8, 512, 56, 56]" = torch.ops.aten.le.Scalar(alias_393, 0);  alias_393 = None
    scalar_tensor_97: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_97: "f32[8, 512, 56, 56]" = torch.ops.aten.where.self(le_97, scalar_tensor_97, getitem_510);  le_97 = scalar_tensor_97 = getitem_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:185, code: x = self.bn2(x)
    unsqueeze_1628: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_1629: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1628, 2);  unsqueeze_1628 = None
    unsqueeze_1630: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1629, 3);  unsqueeze_1629 = None
    sum_204: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_97, [0, 2, 3])
    sub_508: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1630)
    mul_1637: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(where_97, sub_508);  sub_508 = None
    sum_205: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1637, [0, 2, 3]);  mul_1637 = None
    mul_1638: "f32[512]" = torch.ops.aten.mul.Tensor(sum_204, 3.985969387755102e-05)
    unsqueeze_1631: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1638, 0);  mul_1638 = None
    unsqueeze_1632: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1631, 2);  unsqueeze_1631 = None
    unsqueeze_1633: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1632, 3);  unsqueeze_1632 = None
    mul_1639: "f32[512]" = torch.ops.aten.mul.Tensor(sum_205, 3.985969387755102e-05)
    mul_1640: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1641: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1639, mul_1640);  mul_1639 = mul_1640 = None
    unsqueeze_1634: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1641, 0);  mul_1641 = None
    unsqueeze_1635: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1634, 2);  unsqueeze_1634 = None
    unsqueeze_1636: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1635, 3);  unsqueeze_1635 = None
    mul_1642: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
    unsqueeze_1637: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1642, 0);  mul_1642 = None
    unsqueeze_1638: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1637, 2);  unsqueeze_1637 = None
    unsqueeze_1639: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, 3);  unsqueeze_1638 = None
    sub_509: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1630);  convolution_2 = unsqueeze_1630 = None
    mul_1643: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_509, unsqueeze_1636);  sub_509 = unsqueeze_1636 = None
    sub_510: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(where_97, mul_1643);  where_97 = mul_1643 = None
    sub_511: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(sub_510, unsqueeze_1633);  sub_510 = unsqueeze_1633 = None
    mul_1644: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_511, unsqueeze_1639);  sub_511 = unsqueeze_1639 = None
    mul_1645: "f32[512]" = torch.ops.aten.mul.Tensor(sum_205, squeeze_7);  sum_205 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:184, code: x = self.conv2(x)
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(mul_1644, relu_1, primals_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1644 = primals_7 = None
    getitem_513: "f32[8, 512, 56, 56]" = convolution_backward_101[0]
    getitem_514: "f32[512, 16, 3, 3]" = convolution_backward_101[1];  convolution_backward_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:182, code: x = self.act1(x)
    alias_395: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_396: "f32[8, 512, 56, 56]" = torch.ops.aten.alias.default(alias_395);  alias_395 = None
    le_98: "b8[8, 512, 56, 56]" = torch.ops.aten.le.Scalar(alias_396, 0);  alias_396 = None
    scalar_tensor_98: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_98: "f32[8, 512, 56, 56]" = torch.ops.aten.where.self(le_98, scalar_tensor_98, getitem_513);  le_98 = scalar_tensor_98 = getitem_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:181, code: x = self.bn1(x)
    unsqueeze_1640: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_1641: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1640, 2);  unsqueeze_1640 = None
    unsqueeze_1642: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1641, 3);  unsqueeze_1641 = None
    sum_206: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_98, [0, 2, 3])
    sub_512: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1642)
    mul_1646: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(where_98, sub_512);  sub_512 = None
    sum_207: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1646, [0, 2, 3]);  mul_1646 = None
    mul_1647: "f32[512]" = torch.ops.aten.mul.Tensor(sum_206, 3.985969387755102e-05)
    unsqueeze_1643: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1647, 0);  mul_1647 = None
    unsqueeze_1644: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1643, 2);  unsqueeze_1643 = None
    unsqueeze_1645: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1644, 3);  unsqueeze_1644 = None
    mul_1648: "f32[512]" = torch.ops.aten.mul.Tensor(sum_207, 3.985969387755102e-05)
    mul_1649: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1650: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1648, mul_1649);  mul_1648 = mul_1649 = None
    unsqueeze_1646: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1650, 0);  mul_1650 = None
    unsqueeze_1647: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1646, 2);  unsqueeze_1646 = None
    unsqueeze_1648: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1647, 3);  unsqueeze_1647 = None
    mul_1651: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
    unsqueeze_1649: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1651, 0);  mul_1651 = None
    unsqueeze_1650: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1649, 2);  unsqueeze_1649 = None
    unsqueeze_1651: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, 3);  unsqueeze_1650 = None
    sub_513: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1642);  convolution_1 = unsqueeze_1642 = None
    mul_1652: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_513, unsqueeze_1648);  sub_513 = unsqueeze_1648 = None
    sub_514: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(where_98, mul_1652);  where_98 = mul_1652 = None
    sub_515: "f32[8, 512, 56, 56]" = torch.ops.aten.sub.Tensor(sub_514, unsqueeze_1645);  sub_514 = unsqueeze_1645 = None
    mul_1653: "f32[8, 512, 56, 56]" = torch.ops.aten.mul.Tensor(sub_515, unsqueeze_1651);  sub_515 = unsqueeze_1651 = None
    mul_1654: "f32[512]" = torch.ops.aten.mul.Tensor(sum_207, squeeze_4);  sum_207 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(mul_1653, getitem_2, primals_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1653 = getitem_2 = primals_4 = None
    getitem_516: "f32[8, 64, 56, 56]" = convolution_backward_102[0]
    getitem_517: "f32[512, 64, 1, 1]" = convolution_backward_102[1];  convolution_backward_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:180, code: x = self.conv1(x)
    add_585: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(getitem_507, getitem_516);  getitem_507 = getitem_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:523, code: x = self.maxpool(x)
    max_pool2d_with_indices_backward: "f32[8, 64, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_585, relu, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_3);  add_585 = getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:522, code: x = self.act1(x)
    alias_398: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_399: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_398);  alias_398 = None
    le_99: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_399, 0);  alias_399 = None
    scalar_tensor_99: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_99: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_99, scalar_tensor_99, max_pool2d_with_indices_backward);  le_99 = scalar_tensor_99 = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:521, code: x = self.bn1(x)
    unsqueeze_1652: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_1653: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1652, 2);  unsqueeze_1652 = None
    unsqueeze_1654: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1653, 3);  unsqueeze_1653 = None
    sum_208: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_99, [0, 2, 3])
    sub_516: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1654)
    mul_1655: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_99, sub_516);  sub_516 = None
    sum_209: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1655, [0, 2, 3]);  mul_1655 = None
    mul_1656: "f32[64]" = torch.ops.aten.mul.Tensor(sum_208, 9.964923469387754e-06)
    unsqueeze_1655: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1656, 0);  mul_1656 = None
    unsqueeze_1656: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1655, 2);  unsqueeze_1655 = None
    unsqueeze_1657: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1656, 3);  unsqueeze_1656 = None
    mul_1657: "f32[64]" = torch.ops.aten.mul.Tensor(sum_209, 9.964923469387754e-06)
    mul_1658: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1659: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1657, mul_1658);  mul_1657 = mul_1658 = None
    unsqueeze_1658: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1659, 0);  mul_1659 = None
    unsqueeze_1659: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1658, 2);  unsqueeze_1658 = None
    unsqueeze_1660: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1659, 3);  unsqueeze_1659 = None
    mul_1660: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
    unsqueeze_1661: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1660, 0);  mul_1660 = None
    unsqueeze_1662: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1661, 2);  unsqueeze_1661 = None
    unsqueeze_1663: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, 3);  unsqueeze_1662 = None
    sub_517: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1654);  convolution = unsqueeze_1654 = None
    mul_1661: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_517, unsqueeze_1660);  sub_517 = unsqueeze_1660 = None
    sub_518: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_99, mul_1661);  where_99 = mul_1661 = None
    sub_519: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_518, unsqueeze_1657);  sub_518 = unsqueeze_1657 = None
    mul_1662: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_519, unsqueeze_1663);  sub_519 = unsqueeze_1663 = None
    mul_1663: "f32[64]" = torch.ops.aten.mul.Tensor(sum_209, squeeze_1);  sum_209 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/resnet.py:520, code: x = self.conv1(x)
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(mul_1662, primals_627, primals_1, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1662 = primals_627 = primals_1 = None
    getitem_520: "f32[64, 3, 7, 7]" = convolution_backward_103[1];  convolution_backward_103 = None
    
    # No stacktrace found for following nodes
    copy_: "f32[64]" = torch.ops.aten.copy_.default(primals_315, add_2);  primals_315 = add_2 = None
    copy__1: "f32[64]" = torch.ops.aten.copy_.default(primals_316, add_3);  primals_316 = add_3 = None
    copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_317, add);  primals_317 = add = None
    copy__3: "f32[512]" = torch.ops.aten.copy_.default(primals_318, add_7);  primals_318 = add_7 = None
    copy__4: "f32[512]" = torch.ops.aten.copy_.default(primals_319, add_8);  primals_319 = add_8 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_320, add_5);  primals_320 = add_5 = None
    copy__6: "f32[512]" = torch.ops.aten.copy_.default(primals_321, add_12);  primals_321 = add_12 = None
    copy__7: "f32[512]" = torch.ops.aten.copy_.default(primals_322, add_13);  primals_322 = add_13 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_323, add_10);  primals_323 = add_10 = None
    copy__9: "f32[256]" = torch.ops.aten.copy_.default(primals_324, add_17);  primals_324 = add_17 = None
    copy__10: "f32[256]" = torch.ops.aten.copy_.default(primals_325, add_18);  primals_325 = add_18 = None
    copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_326, add_15);  primals_326 = add_15 = None
    copy__12: "f32[256]" = torch.ops.aten.copy_.default(primals_327, add_22);  primals_327 = add_22 = None
    copy__13: "f32[256]" = torch.ops.aten.copy_.default(primals_328, add_23);  primals_328 = add_23 = None
    copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_329, add_20);  primals_329 = add_20 = None
    copy__15: "f32[512]" = torch.ops.aten.copy_.default(primals_330, add_28);  primals_330 = add_28 = None
    copy__16: "f32[512]" = torch.ops.aten.copy_.default(primals_331, add_29);  primals_331 = add_29 = None
    copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_332, add_26);  primals_332 = add_26 = None
    copy__18: "f32[512]" = torch.ops.aten.copy_.default(primals_333, add_33);  primals_333 = add_33 = None
    copy__19: "f32[512]" = torch.ops.aten.copy_.default(primals_334, add_34);  primals_334 = add_34 = None
    copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_335, add_31);  primals_335 = add_31 = None
    copy__21: "f32[256]" = torch.ops.aten.copy_.default(primals_336, add_38);  primals_336 = add_38 = None
    copy__22: "f32[256]" = torch.ops.aten.copy_.default(primals_337, add_39);  primals_337 = add_39 = None
    copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_338, add_36);  primals_338 = add_36 = None
    copy__24: "f32[512]" = torch.ops.aten.copy_.default(primals_339, add_44);  primals_339 = add_44 = None
    copy__25: "f32[512]" = torch.ops.aten.copy_.default(primals_340, add_45);  primals_340 = add_45 = None
    copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_341, add_42);  primals_341 = add_42 = None
    copy__27: "f32[512]" = torch.ops.aten.copy_.default(primals_342, add_49);  primals_342 = add_49 = None
    copy__28: "f32[512]" = torch.ops.aten.copy_.default(primals_343, add_50);  primals_343 = add_50 = None
    copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_344, add_47);  primals_344 = add_47 = None
    copy__30: "f32[256]" = torch.ops.aten.copy_.default(primals_345, add_54);  primals_345 = add_54 = None
    copy__31: "f32[256]" = torch.ops.aten.copy_.default(primals_346, add_55);  primals_346 = add_55 = None
    copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_347, add_52);  primals_347 = add_52 = None
    copy__33: "f32[1024]" = torch.ops.aten.copy_.default(primals_348, add_60);  primals_348 = add_60 = None
    copy__34: "f32[1024]" = torch.ops.aten.copy_.default(primals_349, add_61);  primals_349 = add_61 = None
    copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_350, add_58);  primals_350 = add_58 = None
    copy__36: "f32[1024]" = torch.ops.aten.copy_.default(primals_351, add_65);  primals_351 = add_65 = None
    copy__37: "f32[1024]" = torch.ops.aten.copy_.default(primals_352, add_66);  primals_352 = add_66 = None
    copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_353, add_63);  primals_353 = add_63 = None
    copy__39: "f32[512]" = torch.ops.aten.copy_.default(primals_354, add_70);  primals_354 = add_70 = None
    copy__40: "f32[512]" = torch.ops.aten.copy_.default(primals_355, add_71);  primals_355 = add_71 = None
    copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_356, add_68);  primals_356 = add_68 = None
    copy__42: "f32[512]" = torch.ops.aten.copy_.default(primals_357, add_75);  primals_357 = add_75 = None
    copy__43: "f32[512]" = torch.ops.aten.copy_.default(primals_358, add_76);  primals_358 = add_76 = None
    copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_359, add_73);  primals_359 = add_73 = None
    copy__45: "f32[1024]" = torch.ops.aten.copy_.default(primals_360, add_81);  primals_360 = add_81 = None
    copy__46: "f32[1024]" = torch.ops.aten.copy_.default(primals_361, add_82);  primals_361 = add_82 = None
    copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_362, add_79);  primals_362 = add_79 = None
    copy__48: "f32[1024]" = torch.ops.aten.copy_.default(primals_363, add_86);  primals_363 = add_86 = None
    copy__49: "f32[1024]" = torch.ops.aten.copy_.default(primals_364, add_87);  primals_364 = add_87 = None
    copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_365, add_84);  primals_365 = add_84 = None
    copy__51: "f32[512]" = torch.ops.aten.copy_.default(primals_366, add_91);  primals_366 = add_91 = None
    copy__52: "f32[512]" = torch.ops.aten.copy_.default(primals_367, add_92);  primals_367 = add_92 = None
    copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_368, add_89);  primals_368 = add_89 = None
    copy__54: "f32[1024]" = torch.ops.aten.copy_.default(primals_369, add_97);  primals_369 = add_97 = None
    copy__55: "f32[1024]" = torch.ops.aten.copy_.default(primals_370, add_98);  primals_370 = add_98 = None
    copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_371, add_95);  primals_371 = add_95 = None
    copy__57: "f32[1024]" = torch.ops.aten.copy_.default(primals_372, add_102);  primals_372 = add_102 = None
    copy__58: "f32[1024]" = torch.ops.aten.copy_.default(primals_373, add_103);  primals_373 = add_103 = None
    copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_374, add_100);  primals_374 = add_100 = None
    copy__60: "f32[512]" = torch.ops.aten.copy_.default(primals_375, add_107);  primals_375 = add_107 = None
    copy__61: "f32[512]" = torch.ops.aten.copy_.default(primals_376, add_108);  primals_376 = add_108 = None
    copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_377, add_105);  primals_377 = add_105 = None
    copy__63: "f32[1024]" = torch.ops.aten.copy_.default(primals_378, add_113);  primals_378 = add_113 = None
    copy__64: "f32[1024]" = torch.ops.aten.copy_.default(primals_379, add_114);  primals_379 = add_114 = None
    copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_380, add_111);  primals_380 = add_111 = None
    copy__66: "f32[1024]" = torch.ops.aten.copy_.default(primals_381, add_118);  primals_381 = add_118 = None
    copy__67: "f32[1024]" = torch.ops.aten.copy_.default(primals_382, add_119);  primals_382 = add_119 = None
    copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_383, add_116);  primals_383 = add_116 = None
    copy__69: "f32[512]" = torch.ops.aten.copy_.default(primals_384, add_123);  primals_384 = add_123 = None
    copy__70: "f32[512]" = torch.ops.aten.copy_.default(primals_385, add_124);  primals_385 = add_124 = None
    copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_386, add_121);  primals_386 = add_121 = None
    copy__72: "f32[2048]" = torch.ops.aten.copy_.default(primals_387, add_129);  primals_387 = add_129 = None
    copy__73: "f32[2048]" = torch.ops.aten.copy_.default(primals_388, add_130);  primals_388 = add_130 = None
    copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_389, add_127);  primals_389 = add_127 = None
    copy__75: "f32[2048]" = torch.ops.aten.copy_.default(primals_390, add_134);  primals_390 = add_134 = None
    copy__76: "f32[2048]" = torch.ops.aten.copy_.default(primals_391, add_135);  primals_391 = add_135 = None
    copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_392, add_132);  primals_392 = add_132 = None
    copy__78: "f32[1024]" = torch.ops.aten.copy_.default(primals_393, add_139);  primals_393 = add_139 = None
    copy__79: "f32[1024]" = torch.ops.aten.copy_.default(primals_394, add_140);  primals_394 = add_140 = None
    copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_395, add_137);  primals_395 = add_137 = None
    copy__81: "f32[1024]" = torch.ops.aten.copy_.default(primals_396, add_144);  primals_396 = add_144 = None
    copy__82: "f32[1024]" = torch.ops.aten.copy_.default(primals_397, add_145);  primals_397 = add_145 = None
    copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_398, add_142);  primals_398 = add_142 = None
    copy__84: "f32[2048]" = torch.ops.aten.copy_.default(primals_399, add_150);  primals_399 = add_150 = None
    copy__85: "f32[2048]" = torch.ops.aten.copy_.default(primals_400, add_151);  primals_400 = add_151 = None
    copy__86: "i64[]" = torch.ops.aten.copy_.default(primals_401, add_148);  primals_401 = add_148 = None
    copy__87: "f32[2048]" = torch.ops.aten.copy_.default(primals_402, add_155);  primals_402 = add_155 = None
    copy__88: "f32[2048]" = torch.ops.aten.copy_.default(primals_403, add_156);  primals_403 = add_156 = None
    copy__89: "i64[]" = torch.ops.aten.copy_.default(primals_404, add_153);  primals_404 = add_153 = None
    copy__90: "f32[1024]" = torch.ops.aten.copy_.default(primals_405, add_160);  primals_405 = add_160 = None
    copy__91: "f32[1024]" = torch.ops.aten.copy_.default(primals_406, add_161);  primals_406 = add_161 = None
    copy__92: "i64[]" = torch.ops.aten.copy_.default(primals_407, add_158);  primals_407 = add_158 = None
    copy__93: "f32[2048]" = torch.ops.aten.copy_.default(primals_408, add_166);  primals_408 = add_166 = None
    copy__94: "f32[2048]" = torch.ops.aten.copy_.default(primals_409, add_167);  primals_409 = add_167 = None
    copy__95: "i64[]" = torch.ops.aten.copy_.default(primals_410, add_164);  primals_410 = add_164 = None
    copy__96: "f32[2048]" = torch.ops.aten.copy_.default(primals_411, add_171);  primals_411 = add_171 = None
    copy__97: "f32[2048]" = torch.ops.aten.copy_.default(primals_412, add_172);  primals_412 = add_172 = None
    copy__98: "i64[]" = torch.ops.aten.copy_.default(primals_413, add_169);  primals_413 = add_169 = None
    copy__99: "f32[1024]" = torch.ops.aten.copy_.default(primals_414, add_176);  primals_414 = add_176 = None
    copy__100: "f32[1024]" = torch.ops.aten.copy_.default(primals_415, add_177);  primals_415 = add_177 = None
    copy__101: "i64[]" = torch.ops.aten.copy_.default(primals_416, add_174);  primals_416 = add_174 = None
    copy__102: "f32[2048]" = torch.ops.aten.copy_.default(primals_417, add_182);  primals_417 = add_182 = None
    copy__103: "f32[2048]" = torch.ops.aten.copy_.default(primals_418, add_183);  primals_418 = add_183 = None
    copy__104: "i64[]" = torch.ops.aten.copy_.default(primals_419, add_180);  primals_419 = add_180 = None
    copy__105: "f32[2048]" = torch.ops.aten.copy_.default(primals_420, add_187);  primals_420 = add_187 = None
    copy__106: "f32[2048]" = torch.ops.aten.copy_.default(primals_421, add_188);  primals_421 = add_188 = None
    copy__107: "i64[]" = torch.ops.aten.copy_.default(primals_422, add_185);  primals_422 = add_185 = None
    copy__108: "f32[1024]" = torch.ops.aten.copy_.default(primals_423, add_192);  primals_423 = add_192 = None
    copy__109: "f32[1024]" = torch.ops.aten.copy_.default(primals_424, add_193);  primals_424 = add_193 = None
    copy__110: "i64[]" = torch.ops.aten.copy_.default(primals_425, add_190);  primals_425 = add_190 = None
    copy__111: "f32[2048]" = torch.ops.aten.copy_.default(primals_426, add_198);  primals_426 = add_198 = None
    copy__112: "f32[2048]" = torch.ops.aten.copy_.default(primals_427, add_199);  primals_427 = add_199 = None
    copy__113: "i64[]" = torch.ops.aten.copy_.default(primals_428, add_196);  primals_428 = add_196 = None
    copy__114: "f32[2048]" = torch.ops.aten.copy_.default(primals_429, add_203);  primals_429 = add_203 = None
    copy__115: "f32[2048]" = torch.ops.aten.copy_.default(primals_430, add_204);  primals_430 = add_204 = None
    copy__116: "i64[]" = torch.ops.aten.copy_.default(primals_431, add_201);  primals_431 = add_201 = None
    copy__117: "f32[1024]" = torch.ops.aten.copy_.default(primals_432, add_208);  primals_432 = add_208 = None
    copy__118: "f32[1024]" = torch.ops.aten.copy_.default(primals_433, add_209);  primals_433 = add_209 = None
    copy__119: "i64[]" = torch.ops.aten.copy_.default(primals_434, add_206);  primals_434 = add_206 = None
    copy__120: "f32[2048]" = torch.ops.aten.copy_.default(primals_435, add_214);  primals_435 = add_214 = None
    copy__121: "f32[2048]" = torch.ops.aten.copy_.default(primals_436, add_215);  primals_436 = add_215 = None
    copy__122: "i64[]" = torch.ops.aten.copy_.default(primals_437, add_212);  primals_437 = add_212 = None
    copy__123: "f32[2048]" = torch.ops.aten.copy_.default(primals_438, add_219);  primals_438 = add_219 = None
    copy__124: "f32[2048]" = torch.ops.aten.copy_.default(primals_439, add_220);  primals_439 = add_220 = None
    copy__125: "i64[]" = torch.ops.aten.copy_.default(primals_440, add_217);  primals_440 = add_217 = None
    copy__126: "f32[1024]" = torch.ops.aten.copy_.default(primals_441, add_224);  primals_441 = add_224 = None
    copy__127: "f32[1024]" = torch.ops.aten.copy_.default(primals_442, add_225);  primals_442 = add_225 = None
    copy__128: "i64[]" = torch.ops.aten.copy_.default(primals_443, add_222);  primals_443 = add_222 = None
    copy__129: "f32[2048]" = torch.ops.aten.copy_.default(primals_444, add_230);  primals_444 = add_230 = None
    copy__130: "f32[2048]" = torch.ops.aten.copy_.default(primals_445, add_231);  primals_445 = add_231 = None
    copy__131: "i64[]" = torch.ops.aten.copy_.default(primals_446, add_228);  primals_446 = add_228 = None
    copy__132: "f32[2048]" = torch.ops.aten.copy_.default(primals_447, add_235);  primals_447 = add_235 = None
    copy__133: "f32[2048]" = torch.ops.aten.copy_.default(primals_448, add_236);  primals_448 = add_236 = None
    copy__134: "i64[]" = torch.ops.aten.copy_.default(primals_449, add_233);  primals_449 = add_233 = None
    copy__135: "f32[1024]" = torch.ops.aten.copy_.default(primals_450, add_240);  primals_450 = add_240 = None
    copy__136: "f32[1024]" = torch.ops.aten.copy_.default(primals_451, add_241);  primals_451 = add_241 = None
    copy__137: "i64[]" = torch.ops.aten.copy_.default(primals_452, add_238);  primals_452 = add_238 = None
    copy__138: "f32[2048]" = torch.ops.aten.copy_.default(primals_453, add_246);  primals_453 = add_246 = None
    copy__139: "f32[2048]" = torch.ops.aten.copy_.default(primals_454, add_247);  primals_454 = add_247 = None
    copy__140: "i64[]" = torch.ops.aten.copy_.default(primals_455, add_244);  primals_455 = add_244 = None
    copy__141: "f32[2048]" = torch.ops.aten.copy_.default(primals_456, add_251);  primals_456 = add_251 = None
    copy__142: "f32[2048]" = torch.ops.aten.copy_.default(primals_457, add_252);  primals_457 = add_252 = None
    copy__143: "i64[]" = torch.ops.aten.copy_.default(primals_458, add_249);  primals_458 = add_249 = None
    copy__144: "f32[1024]" = torch.ops.aten.copy_.default(primals_459, add_256);  primals_459 = add_256 = None
    copy__145: "f32[1024]" = torch.ops.aten.copy_.default(primals_460, add_257);  primals_460 = add_257 = None
    copy__146: "i64[]" = torch.ops.aten.copy_.default(primals_461, add_254);  primals_461 = add_254 = None
    copy__147: "f32[2048]" = torch.ops.aten.copy_.default(primals_462, add_262);  primals_462 = add_262 = None
    copy__148: "f32[2048]" = torch.ops.aten.copy_.default(primals_463, add_263);  primals_463 = add_263 = None
    copy__149: "i64[]" = torch.ops.aten.copy_.default(primals_464, add_260);  primals_464 = add_260 = None
    copy__150: "f32[2048]" = torch.ops.aten.copy_.default(primals_465, add_267);  primals_465 = add_267 = None
    copy__151: "f32[2048]" = torch.ops.aten.copy_.default(primals_466, add_268);  primals_466 = add_268 = None
    copy__152: "i64[]" = torch.ops.aten.copy_.default(primals_467, add_265);  primals_467 = add_265 = None
    copy__153: "f32[1024]" = torch.ops.aten.copy_.default(primals_468, add_272);  primals_468 = add_272 = None
    copy__154: "f32[1024]" = torch.ops.aten.copy_.default(primals_469, add_273);  primals_469 = add_273 = None
    copy__155: "i64[]" = torch.ops.aten.copy_.default(primals_470, add_270);  primals_470 = add_270 = None
    copy__156: "f32[2048]" = torch.ops.aten.copy_.default(primals_471, add_278);  primals_471 = add_278 = None
    copy__157: "f32[2048]" = torch.ops.aten.copy_.default(primals_472, add_279);  primals_472 = add_279 = None
    copy__158: "i64[]" = torch.ops.aten.copy_.default(primals_473, add_276);  primals_473 = add_276 = None
    copy__159: "f32[2048]" = torch.ops.aten.copy_.default(primals_474, add_283);  primals_474 = add_283 = None
    copy__160: "f32[2048]" = torch.ops.aten.copy_.default(primals_475, add_284);  primals_475 = add_284 = None
    copy__161: "i64[]" = torch.ops.aten.copy_.default(primals_476, add_281);  primals_476 = add_281 = None
    copy__162: "f32[1024]" = torch.ops.aten.copy_.default(primals_477, add_288);  primals_477 = add_288 = None
    copy__163: "f32[1024]" = torch.ops.aten.copy_.default(primals_478, add_289);  primals_478 = add_289 = None
    copy__164: "i64[]" = torch.ops.aten.copy_.default(primals_479, add_286);  primals_479 = add_286 = None
    copy__165: "f32[2048]" = torch.ops.aten.copy_.default(primals_480, add_294);  primals_480 = add_294 = None
    copy__166: "f32[2048]" = torch.ops.aten.copy_.default(primals_481, add_295);  primals_481 = add_295 = None
    copy__167: "i64[]" = torch.ops.aten.copy_.default(primals_482, add_292);  primals_482 = add_292 = None
    copy__168: "f32[2048]" = torch.ops.aten.copy_.default(primals_483, add_299);  primals_483 = add_299 = None
    copy__169: "f32[2048]" = torch.ops.aten.copy_.default(primals_484, add_300);  primals_484 = add_300 = None
    copy__170: "i64[]" = torch.ops.aten.copy_.default(primals_485, add_297);  primals_485 = add_297 = None
    copy__171: "f32[1024]" = torch.ops.aten.copy_.default(primals_486, add_304);  primals_486 = add_304 = None
    copy__172: "f32[1024]" = torch.ops.aten.copy_.default(primals_487, add_305);  primals_487 = add_305 = None
    copy__173: "i64[]" = torch.ops.aten.copy_.default(primals_488, add_302);  primals_488 = add_302 = None
    copy__174: "f32[2048]" = torch.ops.aten.copy_.default(primals_489, add_310);  primals_489 = add_310 = None
    copy__175: "f32[2048]" = torch.ops.aten.copy_.default(primals_490, add_311);  primals_490 = add_311 = None
    copy__176: "i64[]" = torch.ops.aten.copy_.default(primals_491, add_308);  primals_491 = add_308 = None
    copy__177: "f32[2048]" = torch.ops.aten.copy_.default(primals_492, add_315);  primals_492 = add_315 = None
    copy__178: "f32[2048]" = torch.ops.aten.copy_.default(primals_493, add_316);  primals_493 = add_316 = None
    copy__179: "i64[]" = torch.ops.aten.copy_.default(primals_494, add_313);  primals_494 = add_313 = None
    copy__180: "f32[1024]" = torch.ops.aten.copy_.default(primals_495, add_320);  primals_495 = add_320 = None
    copy__181: "f32[1024]" = torch.ops.aten.copy_.default(primals_496, add_321);  primals_496 = add_321 = None
    copy__182: "i64[]" = torch.ops.aten.copy_.default(primals_497, add_318);  primals_497 = add_318 = None
    copy__183: "f32[2048]" = torch.ops.aten.copy_.default(primals_498, add_326);  primals_498 = add_326 = None
    copy__184: "f32[2048]" = torch.ops.aten.copy_.default(primals_499, add_327);  primals_499 = add_327 = None
    copy__185: "i64[]" = torch.ops.aten.copy_.default(primals_500, add_324);  primals_500 = add_324 = None
    copy__186: "f32[2048]" = torch.ops.aten.copy_.default(primals_501, add_331);  primals_501 = add_331 = None
    copy__187: "f32[2048]" = torch.ops.aten.copy_.default(primals_502, add_332);  primals_502 = add_332 = None
    copy__188: "i64[]" = torch.ops.aten.copy_.default(primals_503, add_329);  primals_503 = add_329 = None
    copy__189: "f32[1024]" = torch.ops.aten.copy_.default(primals_504, add_336);  primals_504 = add_336 = None
    copy__190: "f32[1024]" = torch.ops.aten.copy_.default(primals_505, add_337);  primals_505 = add_337 = None
    copy__191: "i64[]" = torch.ops.aten.copy_.default(primals_506, add_334);  primals_506 = add_334 = None
    copy__192: "f32[2048]" = torch.ops.aten.copy_.default(primals_507, add_342);  primals_507 = add_342 = None
    copy__193: "f32[2048]" = torch.ops.aten.copy_.default(primals_508, add_343);  primals_508 = add_343 = None
    copy__194: "i64[]" = torch.ops.aten.copy_.default(primals_509, add_340);  primals_509 = add_340 = None
    copy__195: "f32[2048]" = torch.ops.aten.copy_.default(primals_510, add_347);  primals_510 = add_347 = None
    copy__196: "f32[2048]" = torch.ops.aten.copy_.default(primals_511, add_348);  primals_511 = add_348 = None
    copy__197: "i64[]" = torch.ops.aten.copy_.default(primals_512, add_345);  primals_512 = add_345 = None
    copy__198: "f32[1024]" = torch.ops.aten.copy_.default(primals_513, add_352);  primals_513 = add_352 = None
    copy__199: "f32[1024]" = torch.ops.aten.copy_.default(primals_514, add_353);  primals_514 = add_353 = None
    copy__200: "i64[]" = torch.ops.aten.copy_.default(primals_515, add_350);  primals_515 = add_350 = None
    copy__201: "f32[2048]" = torch.ops.aten.copy_.default(primals_516, add_358);  primals_516 = add_358 = None
    copy__202: "f32[2048]" = torch.ops.aten.copy_.default(primals_517, add_359);  primals_517 = add_359 = None
    copy__203: "i64[]" = torch.ops.aten.copy_.default(primals_518, add_356);  primals_518 = add_356 = None
    copy__204: "f32[2048]" = torch.ops.aten.copy_.default(primals_519, add_363);  primals_519 = add_363 = None
    copy__205: "f32[2048]" = torch.ops.aten.copy_.default(primals_520, add_364);  primals_520 = add_364 = None
    copy__206: "i64[]" = torch.ops.aten.copy_.default(primals_521, add_361);  primals_521 = add_361 = None
    copy__207: "f32[1024]" = torch.ops.aten.copy_.default(primals_522, add_368);  primals_522 = add_368 = None
    copy__208: "f32[1024]" = torch.ops.aten.copy_.default(primals_523, add_369);  primals_523 = add_369 = None
    copy__209: "i64[]" = torch.ops.aten.copy_.default(primals_524, add_366);  primals_524 = add_366 = None
    copy__210: "f32[2048]" = torch.ops.aten.copy_.default(primals_525, add_374);  primals_525 = add_374 = None
    copy__211: "f32[2048]" = torch.ops.aten.copy_.default(primals_526, add_375);  primals_526 = add_375 = None
    copy__212: "i64[]" = torch.ops.aten.copy_.default(primals_527, add_372);  primals_527 = add_372 = None
    copy__213: "f32[2048]" = torch.ops.aten.copy_.default(primals_528, add_379);  primals_528 = add_379 = None
    copy__214: "f32[2048]" = torch.ops.aten.copy_.default(primals_529, add_380);  primals_529 = add_380 = None
    copy__215: "i64[]" = torch.ops.aten.copy_.default(primals_530, add_377);  primals_530 = add_377 = None
    copy__216: "f32[1024]" = torch.ops.aten.copy_.default(primals_531, add_384);  primals_531 = add_384 = None
    copy__217: "f32[1024]" = torch.ops.aten.copy_.default(primals_532, add_385);  primals_532 = add_385 = None
    copy__218: "i64[]" = torch.ops.aten.copy_.default(primals_533, add_382);  primals_533 = add_382 = None
    copy__219: "f32[2048]" = torch.ops.aten.copy_.default(primals_534, add_390);  primals_534 = add_390 = None
    copy__220: "f32[2048]" = torch.ops.aten.copy_.default(primals_535, add_391);  primals_535 = add_391 = None
    copy__221: "i64[]" = torch.ops.aten.copy_.default(primals_536, add_388);  primals_536 = add_388 = None
    copy__222: "f32[2048]" = torch.ops.aten.copy_.default(primals_537, add_395);  primals_537 = add_395 = None
    copy__223: "f32[2048]" = torch.ops.aten.copy_.default(primals_538, add_396);  primals_538 = add_396 = None
    copy__224: "i64[]" = torch.ops.aten.copy_.default(primals_539, add_393);  primals_539 = add_393 = None
    copy__225: "f32[1024]" = torch.ops.aten.copy_.default(primals_540, add_400);  primals_540 = add_400 = None
    copy__226: "f32[1024]" = torch.ops.aten.copy_.default(primals_541, add_401);  primals_541 = add_401 = None
    copy__227: "i64[]" = torch.ops.aten.copy_.default(primals_542, add_398);  primals_542 = add_398 = None
    copy__228: "f32[2048]" = torch.ops.aten.copy_.default(primals_543, add_406);  primals_543 = add_406 = None
    copy__229: "f32[2048]" = torch.ops.aten.copy_.default(primals_544, add_407);  primals_544 = add_407 = None
    copy__230: "i64[]" = torch.ops.aten.copy_.default(primals_545, add_404);  primals_545 = add_404 = None
    copy__231: "f32[2048]" = torch.ops.aten.copy_.default(primals_546, add_411);  primals_546 = add_411 = None
    copy__232: "f32[2048]" = torch.ops.aten.copy_.default(primals_547, add_412);  primals_547 = add_412 = None
    copy__233: "i64[]" = torch.ops.aten.copy_.default(primals_548, add_409);  primals_548 = add_409 = None
    copy__234: "f32[1024]" = torch.ops.aten.copy_.default(primals_549, add_416);  primals_549 = add_416 = None
    copy__235: "f32[1024]" = torch.ops.aten.copy_.default(primals_550, add_417);  primals_550 = add_417 = None
    copy__236: "i64[]" = torch.ops.aten.copy_.default(primals_551, add_414);  primals_551 = add_414 = None
    copy__237: "f32[2048]" = torch.ops.aten.copy_.default(primals_552, add_422);  primals_552 = add_422 = None
    copy__238: "f32[2048]" = torch.ops.aten.copy_.default(primals_553, add_423);  primals_553 = add_423 = None
    copy__239: "i64[]" = torch.ops.aten.copy_.default(primals_554, add_420);  primals_554 = add_420 = None
    copy__240: "f32[2048]" = torch.ops.aten.copy_.default(primals_555, add_427);  primals_555 = add_427 = None
    copy__241: "f32[2048]" = torch.ops.aten.copy_.default(primals_556, add_428);  primals_556 = add_428 = None
    copy__242: "i64[]" = torch.ops.aten.copy_.default(primals_557, add_425);  primals_557 = add_425 = None
    copy__243: "f32[1024]" = torch.ops.aten.copy_.default(primals_558, add_432);  primals_558 = add_432 = None
    copy__244: "f32[1024]" = torch.ops.aten.copy_.default(primals_559, add_433);  primals_559 = add_433 = None
    copy__245: "i64[]" = torch.ops.aten.copy_.default(primals_560, add_430);  primals_560 = add_430 = None
    copy__246: "f32[2048]" = torch.ops.aten.copy_.default(primals_561, add_438);  primals_561 = add_438 = None
    copy__247: "f32[2048]" = torch.ops.aten.copy_.default(primals_562, add_439);  primals_562 = add_439 = None
    copy__248: "i64[]" = torch.ops.aten.copy_.default(primals_563, add_436);  primals_563 = add_436 = None
    copy__249: "f32[2048]" = torch.ops.aten.copy_.default(primals_564, add_443);  primals_564 = add_443 = None
    copy__250: "f32[2048]" = torch.ops.aten.copy_.default(primals_565, add_444);  primals_565 = add_444 = None
    copy__251: "i64[]" = torch.ops.aten.copy_.default(primals_566, add_441);  primals_566 = add_441 = None
    copy__252: "f32[1024]" = torch.ops.aten.copy_.default(primals_567, add_448);  primals_567 = add_448 = None
    copy__253: "f32[1024]" = torch.ops.aten.copy_.default(primals_568, add_449);  primals_568 = add_449 = None
    copy__254: "i64[]" = torch.ops.aten.copy_.default(primals_569, add_446);  primals_569 = add_446 = None
    copy__255: "f32[2048]" = torch.ops.aten.copy_.default(primals_570, add_454);  primals_570 = add_454 = None
    copy__256: "f32[2048]" = torch.ops.aten.copy_.default(primals_571, add_455);  primals_571 = add_455 = None
    copy__257: "i64[]" = torch.ops.aten.copy_.default(primals_572, add_452);  primals_572 = add_452 = None
    copy__258: "f32[2048]" = torch.ops.aten.copy_.default(primals_573, add_459);  primals_573 = add_459 = None
    copy__259: "f32[2048]" = torch.ops.aten.copy_.default(primals_574, add_460);  primals_574 = add_460 = None
    copy__260: "i64[]" = torch.ops.aten.copy_.default(primals_575, add_457);  primals_575 = add_457 = None
    copy__261: "f32[1024]" = torch.ops.aten.copy_.default(primals_576, add_464);  primals_576 = add_464 = None
    copy__262: "f32[1024]" = torch.ops.aten.copy_.default(primals_577, add_465);  primals_577 = add_465 = None
    copy__263: "i64[]" = torch.ops.aten.copy_.default(primals_578, add_462);  primals_578 = add_462 = None
    copy__264: "f32[2048]" = torch.ops.aten.copy_.default(primals_579, add_470);  primals_579 = add_470 = None
    copy__265: "f32[2048]" = torch.ops.aten.copy_.default(primals_580, add_471);  primals_580 = add_471 = None
    copy__266: "i64[]" = torch.ops.aten.copy_.default(primals_581, add_468);  primals_581 = add_468 = None
    copy__267: "f32[2048]" = torch.ops.aten.copy_.default(primals_582, add_475);  primals_582 = add_475 = None
    copy__268: "f32[2048]" = torch.ops.aten.copy_.default(primals_583, add_476);  primals_583 = add_476 = None
    copy__269: "i64[]" = torch.ops.aten.copy_.default(primals_584, add_473);  primals_584 = add_473 = None
    copy__270: "f32[1024]" = torch.ops.aten.copy_.default(primals_585, add_480);  primals_585 = add_480 = None
    copy__271: "f32[1024]" = torch.ops.aten.copy_.default(primals_586, add_481);  primals_586 = add_481 = None
    copy__272: "i64[]" = torch.ops.aten.copy_.default(primals_587, add_478);  primals_587 = add_478 = None
    copy__273: "f32[2048]" = torch.ops.aten.copy_.default(primals_588, add_486);  primals_588 = add_486 = None
    copy__274: "f32[2048]" = torch.ops.aten.copy_.default(primals_589, add_487);  primals_589 = add_487 = None
    copy__275: "i64[]" = torch.ops.aten.copy_.default(primals_590, add_484);  primals_590 = add_484 = None
    copy__276: "f32[2048]" = torch.ops.aten.copy_.default(primals_591, add_491);  primals_591 = add_491 = None
    copy__277: "f32[2048]" = torch.ops.aten.copy_.default(primals_592, add_492);  primals_592 = add_492 = None
    copy__278: "i64[]" = torch.ops.aten.copy_.default(primals_593, add_489);  primals_593 = add_489 = None
    copy__279: "f32[1024]" = torch.ops.aten.copy_.default(primals_594, add_496);  primals_594 = add_496 = None
    copy__280: "f32[1024]" = torch.ops.aten.copy_.default(primals_595, add_497);  primals_595 = add_497 = None
    copy__281: "i64[]" = torch.ops.aten.copy_.default(primals_596, add_494);  primals_596 = add_494 = None
    copy__282: "f32[4096]" = torch.ops.aten.copy_.default(primals_597, add_502);  primals_597 = add_502 = None
    copy__283: "f32[4096]" = torch.ops.aten.copy_.default(primals_598, add_503);  primals_598 = add_503 = None
    copy__284: "i64[]" = torch.ops.aten.copy_.default(primals_599, add_500);  primals_599 = add_500 = None
    copy__285: "f32[4096]" = torch.ops.aten.copy_.default(primals_600, add_507);  primals_600 = add_507 = None
    copy__286: "f32[4096]" = torch.ops.aten.copy_.default(primals_601, add_508);  primals_601 = add_508 = None
    copy__287: "i64[]" = torch.ops.aten.copy_.default(primals_602, add_505);  primals_602 = add_505 = None
    copy__288: "f32[2048]" = torch.ops.aten.copy_.default(primals_603, add_512);  primals_603 = add_512 = None
    copy__289: "f32[2048]" = torch.ops.aten.copy_.default(primals_604, add_513);  primals_604 = add_513 = None
    copy__290: "i64[]" = torch.ops.aten.copy_.default(primals_605, add_510);  primals_605 = add_510 = None
    copy__291: "f32[2048]" = torch.ops.aten.copy_.default(primals_606, add_517);  primals_606 = add_517 = None
    copy__292: "f32[2048]" = torch.ops.aten.copy_.default(primals_607, add_518);  primals_607 = add_518 = None
    copy__293: "i64[]" = torch.ops.aten.copy_.default(primals_608, add_515);  primals_608 = add_515 = None
    copy__294: "f32[4096]" = torch.ops.aten.copy_.default(primals_609, add_523);  primals_609 = add_523 = None
    copy__295: "f32[4096]" = torch.ops.aten.copy_.default(primals_610, add_524);  primals_610 = add_524 = None
    copy__296: "i64[]" = torch.ops.aten.copy_.default(primals_611, add_521);  primals_611 = add_521 = None
    copy__297: "f32[4096]" = torch.ops.aten.copy_.default(primals_612, add_528);  primals_612 = add_528 = None
    copy__298: "f32[4096]" = torch.ops.aten.copy_.default(primals_613, add_529);  primals_613 = add_529 = None
    copy__299: "i64[]" = torch.ops.aten.copy_.default(primals_614, add_526);  primals_614 = add_526 = None
    copy__300: "f32[2048]" = torch.ops.aten.copy_.default(primals_615, add_533);  primals_615 = add_533 = None
    copy__301: "f32[2048]" = torch.ops.aten.copy_.default(primals_616, add_534);  primals_616 = add_534 = None
    copy__302: "i64[]" = torch.ops.aten.copy_.default(primals_617, add_531);  primals_617 = add_531 = None
    copy__303: "f32[4096]" = torch.ops.aten.copy_.default(primals_618, add_539);  primals_618 = add_539 = None
    copy__304: "f32[4096]" = torch.ops.aten.copy_.default(primals_619, add_540);  primals_619 = add_540 = None
    copy__305: "i64[]" = torch.ops.aten.copy_.default(primals_620, add_537);  primals_620 = add_537 = None
    copy__306: "f32[4096]" = torch.ops.aten.copy_.default(primals_621, add_544);  primals_621 = add_544 = None
    copy__307: "f32[4096]" = torch.ops.aten.copy_.default(primals_622, add_545);  primals_622 = add_545 = None
    copy__308: "i64[]" = torch.ops.aten.copy_.default(primals_623, add_542);  primals_623 = add_542 = None
    copy__309: "f32[2048]" = torch.ops.aten.copy_.default(primals_624, add_549);  primals_624 = add_549 = None
    copy__310: "f32[2048]" = torch.ops.aten.copy_.default(primals_625, add_550);  primals_625 = add_550 = None
    copy__311: "i64[]" = torch.ops.aten.copy_.default(primals_626, add_547);  primals_626 = add_547 = None
    return pytree.tree_unflatten([addmm, getitem_520, mul_1663, sum_208, getitem_517, mul_1654, sum_206, getitem_514, mul_1645, sum_204, getitem_511, mul_1636, sum_202, getitem_508, mul_1627, sum_200, getitem_505, mul_1618, sum_198, getitem_502, mul_1609, sum_196, getitem_499, mul_1600, sum_194, getitem_496, mul_1591, sum_192, getitem_493, mul_1582, sum_190, getitem_490, mul_1573, sum_188, getitem_487, mul_1564, sum_186, getitem_484, mul_1555, sum_184, getitem_481, mul_1546, sum_182, getitem_478, mul_1537, sum_180, getitem_475, mul_1528, sum_178, getitem_472, mul_1519, sum_176, getitem_469, mul_1510, sum_174, getitem_466, mul_1501, sum_172, getitem_463, mul_1492, sum_170, getitem_460, mul_1483, sum_168, getitem_457, mul_1474, sum_166, getitem_454, mul_1465, sum_164, getitem_451, mul_1456, sum_162, getitem_448, mul_1447, sum_160, getitem_445, mul_1438, sum_158, getitem_442, mul_1429, sum_156, getitem_439, mul_1420, sum_154, getitem_436, mul_1411, sum_152, getitem_433, mul_1402, sum_150, getitem_430, mul_1393, sum_148, getitem_427, mul_1384, sum_146, getitem_424, mul_1375, sum_144, getitem_421, mul_1366, sum_142, getitem_418, mul_1357, sum_140, getitem_415, mul_1348, sum_138, getitem_412, mul_1339, sum_136, getitem_409, mul_1330, sum_134, getitem_406, mul_1321, sum_132, getitem_403, mul_1312, sum_130, getitem_400, mul_1303, sum_128, getitem_397, mul_1294, sum_126, getitem_394, mul_1285, sum_124, getitem_391, mul_1276, sum_122, getitem_388, mul_1267, sum_120, getitem_385, mul_1258, sum_118, getitem_382, mul_1249, sum_116, getitem_379, mul_1240, sum_114, getitem_376, mul_1231, sum_112, getitem_373, mul_1222, sum_110, getitem_370, mul_1213, sum_108, getitem_367, mul_1204, sum_106, getitem_364, mul_1195, sum_104, getitem_361, mul_1186, sum_102, getitem_358, mul_1177, sum_100, getitem_355, mul_1168, sum_98, getitem_352, mul_1159, sum_96, getitem_349, mul_1150, sum_94, getitem_346, mul_1141, sum_92, getitem_343, mul_1132, sum_90, getitem_340, mul_1123, sum_88, getitem_337, mul_1114, sum_86, getitem_334, mul_1105, sum_84, getitem_331, mul_1096, sum_82, getitem_328, mul_1087, sum_80, getitem_325, mul_1078, sum_78, getitem_322, mul_1069, sum_76, getitem_319, mul_1060, sum_74, getitem_316, mul_1051, sum_72, getitem_313, mul_1042, sum_70, getitem_310, mul_1033, sum_68, getitem_307, mul_1024, sum_66, getitem_304, mul_1015, sum_64, getitem_301, mul_1006, sum_62, getitem_298, mul_997, sum_60, getitem_295, mul_988, sum_58, getitem_292, mul_979, sum_56, getitem_289, mul_970, sum_54, getitem_286, mul_961, sum_52, getitem_283, mul_952, sum_50, getitem_280, mul_943, sum_48, getitem_277, mul_934, sum_46, getitem_274, mul_925, sum_44, getitem_271, mul_916, sum_42, getitem_268, mul_907, sum_40, getitem_265, mul_898, sum_38, getitem_262, mul_889, sum_36, getitem_259, mul_880, sum_34, getitem_256, mul_871, sum_32, getitem_253, mul_862, sum_30, getitem_250, mul_853, sum_28, getitem_247, mul_844, sum_26, getitem_244, mul_835, sum_24, getitem_241, mul_826, sum_22, getitem_238, mul_817, sum_20, getitem_235, mul_808, sum_18, getitem_232, mul_799, sum_16, getitem_229, mul_790, sum_14, getitem_226, mul_781, sum_12, getitem_223, mul_772, sum_10, getitem_220, mul_763, sum_8, getitem_217, mul_754, sum_6, getitem_214, mul_745, sum_4, getitem_211, mul_736, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    