from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[16, 3, 7, 7]"; primals_2: "f32[16]"; primals_3: "f32[16]"; primals_4: "f32[16, 16, 3, 3]"; primals_5: "f32[16]"; primals_6: "f32[16]"; primals_7: "f32[32, 16, 3, 3]"; primals_8: "f32[32]"; primals_9: "f32[32]"; primals_10: "f32[128, 32, 1, 1]"; primals_11: "f32[128]"; primals_12: "f32[128]"; primals_13: "f32[64, 32, 1, 1]"; primals_14: "f32[64]"; primals_15: "f32[64]"; primals_16: "f32[64, 64, 3, 3]"; primals_17: "f32[64]"; primals_18: "f32[64]"; primals_19: "f32[128, 64, 1, 1]"; primals_20: "f32[128]"; primals_21: "f32[128]"; primals_22: "f32[64, 128, 1, 1]"; primals_23: "f32[64]"; primals_24: "f32[64]"; primals_25: "f32[64, 64, 3, 3]"; primals_26: "f32[64]"; primals_27: "f32[64]"; primals_28: "f32[128, 64, 1, 1]"; primals_29: "f32[128]"; primals_30: "f32[128]"; primals_31: "f32[128, 256, 1, 1]"; primals_32: "f32[128]"; primals_33: "f32[128]"; primals_34: "f32[256, 128, 1, 1]"; primals_35: "f32[256]"; primals_36: "f32[256]"; primals_37: "f32[128, 128, 1, 1]"; primals_38: "f32[128]"; primals_39: "f32[128]"; primals_40: "f32[128, 128, 3, 3]"; primals_41: "f32[128]"; primals_42: "f32[128]"; primals_43: "f32[256, 128, 1, 1]"; primals_44: "f32[256]"; primals_45: "f32[256]"; primals_46: "f32[128, 256, 1, 1]"; primals_47: "f32[128]"; primals_48: "f32[128]"; primals_49: "f32[128, 128, 3, 3]"; primals_50: "f32[128]"; primals_51: "f32[128]"; primals_52: "f32[256, 128, 1, 1]"; primals_53: "f32[256]"; primals_54: "f32[256]"; primals_55: "f32[256, 512, 1, 1]"; primals_56: "f32[256]"; primals_57: "f32[256]"; primals_58: "f32[128, 256, 1, 1]"; primals_59: "f32[128]"; primals_60: "f32[128]"; primals_61: "f32[128, 128, 3, 3]"; primals_62: "f32[128]"; primals_63: "f32[128]"; primals_64: "f32[256, 128, 1, 1]"; primals_65: "f32[256]"; primals_66: "f32[256]"; primals_67: "f32[128, 256, 1, 1]"; primals_68: "f32[128]"; primals_69: "f32[128]"; primals_70: "f32[128, 128, 3, 3]"; primals_71: "f32[128]"; primals_72: "f32[128]"; primals_73: "f32[256, 128, 1, 1]"; primals_74: "f32[256]"; primals_75: "f32[256]"; primals_76: "f32[256, 768, 1, 1]"; primals_77: "f32[256]"; primals_78: "f32[256]"; primals_79: "f32[128, 256, 1, 1]"; primals_80: "f32[128]"; primals_81: "f32[128]"; primals_82: "f32[128, 128, 3, 3]"; primals_83: "f32[128]"; primals_84: "f32[128]"; primals_85: "f32[256, 128, 1, 1]"; primals_86: "f32[256]"; primals_87: "f32[256]"; primals_88: "f32[128, 256, 1, 1]"; primals_89: "f32[128]"; primals_90: "f32[128]"; primals_91: "f32[128, 128, 3, 3]"; primals_92: "f32[128]"; primals_93: "f32[128]"; primals_94: "f32[256, 128, 1, 1]"; primals_95: "f32[256]"; primals_96: "f32[256]"; primals_97: "f32[256, 512, 1, 1]"; primals_98: "f32[256]"; primals_99: "f32[256]"; primals_100: "f32[128, 256, 1, 1]"; primals_101: "f32[128]"; primals_102: "f32[128]"; primals_103: "f32[128, 128, 3, 3]"; primals_104: "f32[128]"; primals_105: "f32[128]"; primals_106: "f32[256, 128, 1, 1]"; primals_107: "f32[256]"; primals_108: "f32[256]"; primals_109: "f32[128, 256, 1, 1]"; primals_110: "f32[128]"; primals_111: "f32[128]"; primals_112: "f32[128, 128, 3, 3]"; primals_113: "f32[128]"; primals_114: "f32[128]"; primals_115: "f32[256, 128, 1, 1]"; primals_116: "f32[256]"; primals_117: "f32[256]"; primals_118: "f32[256, 1152, 1, 1]"; primals_119: "f32[256]"; primals_120: "f32[256]"; primals_121: "f32[512, 256, 1, 1]"; primals_122: "f32[512]"; primals_123: "f32[512]"; primals_124: "f32[256, 256, 1, 1]"; primals_125: "f32[256]"; primals_126: "f32[256]"; primals_127: "f32[256, 256, 3, 3]"; primals_128: "f32[256]"; primals_129: "f32[256]"; primals_130: "f32[512, 256, 1, 1]"; primals_131: "f32[512]"; primals_132: "f32[512]"; primals_133: "f32[256, 512, 1, 1]"; primals_134: "f32[256]"; primals_135: "f32[256]"; primals_136: "f32[256, 256, 3, 3]"; primals_137: "f32[256]"; primals_138: "f32[256]"; primals_139: "f32[512, 256, 1, 1]"; primals_140: "f32[512]"; primals_141: "f32[512]"; primals_142: "f32[512, 1024, 1, 1]"; primals_143: "f32[512]"; primals_144: "f32[512]"; primals_145: "f32[256, 512, 1, 1]"; primals_146: "f32[256]"; primals_147: "f32[256]"; primals_148: "f32[256, 256, 3, 3]"; primals_149: "f32[256]"; primals_150: "f32[256]"; primals_151: "f32[512, 256, 1, 1]"; primals_152: "f32[512]"; primals_153: "f32[512]"; primals_154: "f32[256, 512, 1, 1]"; primals_155: "f32[256]"; primals_156: "f32[256]"; primals_157: "f32[256, 256, 3, 3]"; primals_158: "f32[256]"; primals_159: "f32[256]"; primals_160: "f32[512, 256, 1, 1]"; primals_161: "f32[512]"; primals_162: "f32[512]"; primals_163: "f32[512, 1536, 1, 1]"; primals_164: "f32[512]"; primals_165: "f32[512]"; primals_166: "f32[256, 512, 1, 1]"; primals_167: "f32[256]"; primals_168: "f32[256]"; primals_169: "f32[256, 256, 3, 3]"; primals_170: "f32[256]"; primals_171: "f32[256]"; primals_172: "f32[512, 256, 1, 1]"; primals_173: "f32[512]"; primals_174: "f32[512]"; primals_175: "f32[256, 512, 1, 1]"; primals_176: "f32[256]"; primals_177: "f32[256]"; primals_178: "f32[256, 256, 3, 3]"; primals_179: "f32[256]"; primals_180: "f32[256]"; primals_181: "f32[512, 256, 1, 1]"; primals_182: "f32[512]"; primals_183: "f32[512]"; primals_184: "f32[512, 1024, 1, 1]"; primals_185: "f32[512]"; primals_186: "f32[512]"; primals_187: "f32[256, 512, 1, 1]"; primals_188: "f32[256]"; primals_189: "f32[256]"; primals_190: "f32[256, 256, 3, 3]"; primals_191: "f32[256]"; primals_192: "f32[256]"; primals_193: "f32[512, 256, 1, 1]"; primals_194: "f32[512]"; primals_195: "f32[512]"; primals_196: "f32[256, 512, 1, 1]"; primals_197: "f32[256]"; primals_198: "f32[256]"; primals_199: "f32[256, 256, 3, 3]"; primals_200: "f32[256]"; primals_201: "f32[256]"; primals_202: "f32[512, 256, 1, 1]"; primals_203: "f32[512]"; primals_204: "f32[512]"; primals_205: "f32[512, 2048, 1, 1]"; primals_206: "f32[512]"; primals_207: "f32[512]"; primals_208: "f32[256, 512, 1, 1]"; primals_209: "f32[256]"; primals_210: "f32[256]"; primals_211: "f32[256, 256, 3, 3]"; primals_212: "f32[256]"; primals_213: "f32[256]"; primals_214: "f32[512, 256, 1, 1]"; primals_215: "f32[512]"; primals_216: "f32[512]"; primals_217: "f32[256, 512, 1, 1]"; primals_218: "f32[256]"; primals_219: "f32[256]"; primals_220: "f32[256, 256, 3, 3]"; primals_221: "f32[256]"; primals_222: "f32[256]"; primals_223: "f32[512, 256, 1, 1]"; primals_224: "f32[512]"; primals_225: "f32[512]"; primals_226: "f32[512, 1024, 1, 1]"; primals_227: "f32[512]"; primals_228: "f32[512]"; primals_229: "f32[256, 512, 1, 1]"; primals_230: "f32[256]"; primals_231: "f32[256]"; primals_232: "f32[256, 256, 3, 3]"; primals_233: "f32[256]"; primals_234: "f32[256]"; primals_235: "f32[512, 256, 1, 1]"; primals_236: "f32[512]"; primals_237: "f32[512]"; primals_238: "f32[256, 512, 1, 1]"; primals_239: "f32[256]"; primals_240: "f32[256]"; primals_241: "f32[256, 256, 3, 3]"; primals_242: "f32[256]"; primals_243: "f32[256]"; primals_244: "f32[512, 256, 1, 1]"; primals_245: "f32[512]"; primals_246: "f32[512]"; primals_247: "f32[512, 1536, 1, 1]"; primals_248: "f32[512]"; primals_249: "f32[512]"; primals_250: "f32[256, 512, 1, 1]"; primals_251: "f32[256]"; primals_252: "f32[256]"; primals_253: "f32[256, 256, 3, 3]"; primals_254: "f32[256]"; primals_255: "f32[256]"; primals_256: "f32[512, 256, 1, 1]"; primals_257: "f32[512]"; primals_258: "f32[512]"; primals_259: "f32[256, 512, 1, 1]"; primals_260: "f32[256]"; primals_261: "f32[256]"; primals_262: "f32[256, 256, 3, 3]"; primals_263: "f32[256]"; primals_264: "f32[256]"; primals_265: "f32[512, 256, 1, 1]"; primals_266: "f32[512]"; primals_267: "f32[512]"; primals_268: "f32[512, 1024, 1, 1]"; primals_269: "f32[512]"; primals_270: "f32[512]"; primals_271: "f32[256, 512, 1, 1]"; primals_272: "f32[256]"; primals_273: "f32[256]"; primals_274: "f32[256, 256, 3, 3]"; primals_275: "f32[256]"; primals_276: "f32[256]"; primals_277: "f32[512, 256, 1, 1]"; primals_278: "f32[512]"; primals_279: "f32[512]"; primals_280: "f32[256, 512, 1, 1]"; primals_281: "f32[256]"; primals_282: "f32[256]"; primals_283: "f32[256, 256, 3, 3]"; primals_284: "f32[256]"; primals_285: "f32[256]"; primals_286: "f32[512, 256, 1, 1]"; primals_287: "f32[512]"; primals_288: "f32[512]"; primals_289: "f32[512, 2816, 1, 1]"; primals_290: "f32[512]"; primals_291: "f32[512]"; primals_292: "f32[1024, 512, 1, 1]"; primals_293: "f32[1024]"; primals_294: "f32[1024]"; primals_295: "f32[512, 512, 1, 1]"; primals_296: "f32[512]"; primals_297: "f32[512]"; primals_298: "f32[512, 512, 3, 3]"; primals_299: "f32[512]"; primals_300: "f32[512]"; primals_301: "f32[1024, 512, 1, 1]"; primals_302: "f32[1024]"; primals_303: "f32[1024]"; primals_304: "f32[512, 1024, 1, 1]"; primals_305: "f32[512]"; primals_306: "f32[512]"; primals_307: "f32[512, 512, 3, 3]"; primals_308: "f32[512]"; primals_309: "f32[512]"; primals_310: "f32[1024, 512, 1, 1]"; primals_311: "f32[1024]"; primals_312: "f32[1024]"; primals_313: "f32[1024, 2560, 1, 1]"; primals_314: "f32[1024]"; primals_315: "f32[1024]"; primals_316: "f32[1000, 1024, 1, 1]"; primals_317: "f32[1000]"; primals_318: "f32[16]"; primals_319: "f32[16]"; primals_320: "i64[]"; primals_321: "f32[16]"; primals_322: "f32[16]"; primals_323: "i64[]"; primals_324: "f32[32]"; primals_325: "f32[32]"; primals_326: "i64[]"; primals_327: "f32[128]"; primals_328: "f32[128]"; primals_329: "i64[]"; primals_330: "f32[64]"; primals_331: "f32[64]"; primals_332: "i64[]"; primals_333: "f32[64]"; primals_334: "f32[64]"; primals_335: "i64[]"; primals_336: "f32[128]"; primals_337: "f32[128]"; primals_338: "i64[]"; primals_339: "f32[64]"; primals_340: "f32[64]"; primals_341: "i64[]"; primals_342: "f32[64]"; primals_343: "f32[64]"; primals_344: "i64[]"; primals_345: "f32[128]"; primals_346: "f32[128]"; primals_347: "i64[]"; primals_348: "f32[128]"; primals_349: "f32[128]"; primals_350: "i64[]"; primals_351: "f32[256]"; primals_352: "f32[256]"; primals_353: "i64[]"; primals_354: "f32[128]"; primals_355: "f32[128]"; primals_356: "i64[]"; primals_357: "f32[128]"; primals_358: "f32[128]"; primals_359: "i64[]"; primals_360: "f32[256]"; primals_361: "f32[256]"; primals_362: "i64[]"; primals_363: "f32[128]"; primals_364: "f32[128]"; primals_365: "i64[]"; primals_366: "f32[128]"; primals_367: "f32[128]"; primals_368: "i64[]"; primals_369: "f32[256]"; primals_370: "f32[256]"; primals_371: "i64[]"; primals_372: "f32[256]"; primals_373: "f32[256]"; primals_374: "i64[]"; primals_375: "f32[128]"; primals_376: "f32[128]"; primals_377: "i64[]"; primals_378: "f32[128]"; primals_379: "f32[128]"; primals_380: "i64[]"; primals_381: "f32[256]"; primals_382: "f32[256]"; primals_383: "i64[]"; primals_384: "f32[128]"; primals_385: "f32[128]"; primals_386: "i64[]"; primals_387: "f32[128]"; primals_388: "f32[128]"; primals_389: "i64[]"; primals_390: "f32[256]"; primals_391: "f32[256]"; primals_392: "i64[]"; primals_393: "f32[256]"; primals_394: "f32[256]"; primals_395: "i64[]"; primals_396: "f32[128]"; primals_397: "f32[128]"; primals_398: "i64[]"; primals_399: "f32[128]"; primals_400: "f32[128]"; primals_401: "i64[]"; primals_402: "f32[256]"; primals_403: "f32[256]"; primals_404: "i64[]"; primals_405: "f32[128]"; primals_406: "f32[128]"; primals_407: "i64[]"; primals_408: "f32[128]"; primals_409: "f32[128]"; primals_410: "i64[]"; primals_411: "f32[256]"; primals_412: "f32[256]"; primals_413: "i64[]"; primals_414: "f32[256]"; primals_415: "f32[256]"; primals_416: "i64[]"; primals_417: "f32[128]"; primals_418: "f32[128]"; primals_419: "i64[]"; primals_420: "f32[128]"; primals_421: "f32[128]"; primals_422: "i64[]"; primals_423: "f32[256]"; primals_424: "f32[256]"; primals_425: "i64[]"; primals_426: "f32[128]"; primals_427: "f32[128]"; primals_428: "i64[]"; primals_429: "f32[128]"; primals_430: "f32[128]"; primals_431: "i64[]"; primals_432: "f32[256]"; primals_433: "f32[256]"; primals_434: "i64[]"; primals_435: "f32[256]"; primals_436: "f32[256]"; primals_437: "i64[]"; primals_438: "f32[512]"; primals_439: "f32[512]"; primals_440: "i64[]"; primals_441: "f32[256]"; primals_442: "f32[256]"; primals_443: "i64[]"; primals_444: "f32[256]"; primals_445: "f32[256]"; primals_446: "i64[]"; primals_447: "f32[512]"; primals_448: "f32[512]"; primals_449: "i64[]"; primals_450: "f32[256]"; primals_451: "f32[256]"; primals_452: "i64[]"; primals_453: "f32[256]"; primals_454: "f32[256]"; primals_455: "i64[]"; primals_456: "f32[512]"; primals_457: "f32[512]"; primals_458: "i64[]"; primals_459: "f32[512]"; primals_460: "f32[512]"; primals_461: "i64[]"; primals_462: "f32[256]"; primals_463: "f32[256]"; primals_464: "i64[]"; primals_465: "f32[256]"; primals_466: "f32[256]"; primals_467: "i64[]"; primals_468: "f32[512]"; primals_469: "f32[512]"; primals_470: "i64[]"; primals_471: "f32[256]"; primals_472: "f32[256]"; primals_473: "i64[]"; primals_474: "f32[256]"; primals_475: "f32[256]"; primals_476: "i64[]"; primals_477: "f32[512]"; primals_478: "f32[512]"; primals_479: "i64[]"; primals_480: "f32[512]"; primals_481: "f32[512]"; primals_482: "i64[]"; primals_483: "f32[256]"; primals_484: "f32[256]"; primals_485: "i64[]"; primals_486: "f32[256]"; primals_487: "f32[256]"; primals_488: "i64[]"; primals_489: "f32[512]"; primals_490: "f32[512]"; primals_491: "i64[]"; primals_492: "f32[256]"; primals_493: "f32[256]"; primals_494: "i64[]"; primals_495: "f32[256]"; primals_496: "f32[256]"; primals_497: "i64[]"; primals_498: "f32[512]"; primals_499: "f32[512]"; primals_500: "i64[]"; primals_501: "f32[512]"; primals_502: "f32[512]"; primals_503: "i64[]"; primals_504: "f32[256]"; primals_505: "f32[256]"; primals_506: "i64[]"; primals_507: "f32[256]"; primals_508: "f32[256]"; primals_509: "i64[]"; primals_510: "f32[512]"; primals_511: "f32[512]"; primals_512: "i64[]"; primals_513: "f32[256]"; primals_514: "f32[256]"; primals_515: "i64[]"; primals_516: "f32[256]"; primals_517: "f32[256]"; primals_518: "i64[]"; primals_519: "f32[512]"; primals_520: "f32[512]"; primals_521: "i64[]"; primals_522: "f32[512]"; primals_523: "f32[512]"; primals_524: "i64[]"; primals_525: "f32[256]"; primals_526: "f32[256]"; primals_527: "i64[]"; primals_528: "f32[256]"; primals_529: "f32[256]"; primals_530: "i64[]"; primals_531: "f32[512]"; primals_532: "f32[512]"; primals_533: "i64[]"; primals_534: "f32[256]"; primals_535: "f32[256]"; primals_536: "i64[]"; primals_537: "f32[256]"; primals_538: "f32[256]"; primals_539: "i64[]"; primals_540: "f32[512]"; primals_541: "f32[512]"; primals_542: "i64[]"; primals_543: "f32[512]"; primals_544: "f32[512]"; primals_545: "i64[]"; primals_546: "f32[256]"; primals_547: "f32[256]"; primals_548: "i64[]"; primals_549: "f32[256]"; primals_550: "f32[256]"; primals_551: "i64[]"; primals_552: "f32[512]"; primals_553: "f32[512]"; primals_554: "i64[]"; primals_555: "f32[256]"; primals_556: "f32[256]"; primals_557: "i64[]"; primals_558: "f32[256]"; primals_559: "f32[256]"; primals_560: "i64[]"; primals_561: "f32[512]"; primals_562: "f32[512]"; primals_563: "i64[]"; primals_564: "f32[512]"; primals_565: "f32[512]"; primals_566: "i64[]"; primals_567: "f32[256]"; primals_568: "f32[256]"; primals_569: "i64[]"; primals_570: "f32[256]"; primals_571: "f32[256]"; primals_572: "i64[]"; primals_573: "f32[512]"; primals_574: "f32[512]"; primals_575: "i64[]"; primals_576: "f32[256]"; primals_577: "f32[256]"; primals_578: "i64[]"; primals_579: "f32[256]"; primals_580: "f32[256]"; primals_581: "i64[]"; primals_582: "f32[512]"; primals_583: "f32[512]"; primals_584: "i64[]"; primals_585: "f32[512]"; primals_586: "f32[512]"; primals_587: "i64[]"; primals_588: "f32[256]"; primals_589: "f32[256]"; primals_590: "i64[]"; primals_591: "f32[256]"; primals_592: "f32[256]"; primals_593: "i64[]"; primals_594: "f32[512]"; primals_595: "f32[512]"; primals_596: "i64[]"; primals_597: "f32[256]"; primals_598: "f32[256]"; primals_599: "i64[]"; primals_600: "f32[256]"; primals_601: "f32[256]"; primals_602: "i64[]"; primals_603: "f32[512]"; primals_604: "f32[512]"; primals_605: "i64[]"; primals_606: "f32[512]"; primals_607: "f32[512]"; primals_608: "i64[]"; primals_609: "f32[1024]"; primals_610: "f32[1024]"; primals_611: "i64[]"; primals_612: "f32[512]"; primals_613: "f32[512]"; primals_614: "i64[]"; primals_615: "f32[512]"; primals_616: "f32[512]"; primals_617: "i64[]"; primals_618: "f32[1024]"; primals_619: "f32[1024]"; primals_620: "i64[]"; primals_621: "f32[512]"; primals_622: "f32[512]"; primals_623: "i64[]"; primals_624: "f32[512]"; primals_625: "f32[512]"; primals_626: "i64[]"; primals_627: "f32[1024]"; primals_628: "f32[1024]"; primals_629: "i64[]"; primals_630: "f32[1024]"; primals_631: "f32[1024]"; primals_632: "i64[]"; primals_633: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:363, code: x = self.base_layer(x)
    convolution: "f32[8, 16, 224, 224]" = torch.ops.aten.convolution.default(primals_633, primals_1, None, [1, 1], [3, 3], [1, 1], False, [0, 0], 1)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_320, 1)
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 16, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 16, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(primals_318, 0.9)
    add_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000024912370735);  squeeze_2 = None
    mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[16]" = torch.ops.aten.mul.Tensor(primals_319, 0.9)
    add_3: "f32[16]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 16, 224, 224]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    relu: "f32[8, 16, 224, 224]" = torch.ops.aten.relu.default(add_4);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:364, code: x = self.level0(x)
    convolution_1: "f32[8, 16, 224, 224]" = torch.ops.aten.convolution.default(relu, primals_4, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_323, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 16, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 16, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_1: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_7: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_8: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_9: "f32[16]" = torch.ops.aten.mul.Tensor(primals_321, 0.9)
    add_7: "f32[16]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    squeeze_5: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_10: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000024912370735);  squeeze_5 = None
    mul_11: "f32[16]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
    mul_12: "f32[16]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
    add_8: "f32[16]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    unsqueeze_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_13: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
    unsqueeze_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_9: "f32[8, 16, 224, 224]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
    relu_1: "f32[8, 16, 224, 224]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:365, code: x = self.level1(x)
    convolution_2: "f32[8, 32, 112, 112]" = torch.ops.aten.convolution.default(relu_1, primals_7, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_326, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 32, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 32, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_2: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_14: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_15: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_16: "f32[32]" = torch.ops.aten.mul.Tensor(primals_324, 0.9)
    add_12: "f32[32]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
    squeeze_8: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_17: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
    mul_18: "f32[32]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
    mul_19: "f32[32]" = torch.ops.aten.mul.Tensor(primals_325, 0.9)
    add_13: "f32[32]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
    unsqueeze_8: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1)
    unsqueeze_9: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_20: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
    unsqueeze_10: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
    unsqueeze_11: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_14: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
    relu_2: "f32[8, 32, 112, 112]" = torch.ops.aten.relu.default(add_14);  add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu_2, [2, 2], [2, 2])
    getitem_6: "f32[8, 32, 56, 56]" = max_pool2d_with_indices[0]
    getitem_7: "i64[8, 32, 56, 56]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    convolution_3: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(getitem_6, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_329, 1)
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1, 1]" = var_mean_3[0]
    getitem_9: "f32[1, 128, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_16: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_3: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_3: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
    mul_21: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_10: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_22: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_23: "f32[128]" = torch.ops.aten.mul.Tensor(primals_327, 0.9)
    add_17: "f32[128]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    squeeze_11: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_24: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000398612827361);  squeeze_11 = None
    mul_25: "f32[128]" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
    mul_26: "f32[128]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
    add_18: "f32[128]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
    unsqueeze_12: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_13: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_27: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
    unsqueeze_14: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_15: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_19: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_4: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(relu_2, primals_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_332, 1)
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 64, 1, 1]" = var_mean_4[0]
    getitem_11: "f32[1, 64, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_21: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_4: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_4: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
    mul_28: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_13: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_29: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_30: "f32[64]" = torch.ops.aten.mul.Tensor(primals_330, 0.9)
    add_22: "f32[64]" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
    squeeze_14: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_31: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.00000996502277);  squeeze_14 = None
    mul_32: "f32[64]" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
    mul_33: "f32[64]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
    add_23: "f32[64]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1)
    unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_34: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
    unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
    unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_24: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_3: "f32[8, 64, 112, 112]" = torch.ops.aten.relu.default(add_24);  add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_5: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_16, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_25: "i64[]" = torch.ops.aten.add.Tensor(primals_335, 1)
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 64, 1, 1]" = var_mean_5[0]
    getitem_13: "f32[1, 64, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_26: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_5: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_5: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_13)
    mul_35: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_16: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_36: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_37: "f32[64]" = torch.ops.aten.mul.Tensor(primals_333, 0.9)
    add_27: "f32[64]" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
    squeeze_17: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_38: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000398612827361);  squeeze_17 = None
    mul_39: "f32[64]" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
    mul_40: "f32[64]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
    add_28: "f32[64]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_41: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_29: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_4: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_6: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_338, 1)
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1, 1]" = var_mean_6[0]
    getitem_15: "f32[1, 128, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_31: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_6: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_15)
    mul_42: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_19: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_43: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_44: "f32[128]" = torch.ops.aten.mul.Tensor(primals_336, 0.9)
    add_32: "f32[128]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_20: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_45: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_46: "f32[128]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[128]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
    add_33: "f32[128]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_24: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1)
    unsqueeze_25: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_48: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
    unsqueeze_26: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
    unsqueeze_27: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_34: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_35: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(add_34, add_19);  add_34 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_5: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_7: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_341, 1)
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 64, 1, 1]" = var_mean_7[0]
    getitem_17: "f32[1, 64, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_37: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_7: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_7: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_17)
    mul_49: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_22: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_50: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_51: "f32[64]" = torch.ops.aten.mul.Tensor(primals_339, 0.9)
    add_38: "f32[64]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    squeeze_23: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_52: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_53: "f32[64]" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
    mul_54: "f32[64]" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
    add_39: "f32[64]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    unsqueeze_28: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_29: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_55: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
    unsqueeze_30: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_31: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_40: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_6: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_40);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_8: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_6, primals_25, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_41: "i64[]" = torch.ops.aten.add.Tensor(primals_344, 1)
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 64, 1, 1]" = var_mean_8[0]
    getitem_19: "f32[1, 64, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_42: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_8: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_8: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_19)
    mul_56: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_25: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_57: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_58: "f32[64]" = torch.ops.aten.mul.Tensor(primals_342, 0.9)
    add_43: "f32[64]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
    squeeze_26: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_59: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_60: "f32[64]" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
    mul_61: "f32[64]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
    add_44: "f32[64]" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
    unsqueeze_32: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1)
    unsqueeze_33: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_62: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
    unsqueeze_34: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
    unsqueeze_35: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_45: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_7: "f32[8, 64, 56, 56]" = torch.ops.aten.relu.default(add_45);  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_9: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_7, primals_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_46: "i64[]" = torch.ops.aten.add.Tensor(primals_347, 1)
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1, 1]" = var_mean_9[0]
    getitem_21: "f32[1, 128, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_47: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_9: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_9: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_21)
    mul_63: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_28: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_64: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_65: "f32[128]" = torch.ops.aten.mul.Tensor(primals_345, 0.9)
    add_48: "f32[128]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    squeeze_29: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_66: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_67: "f32[128]" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
    mul_68: "f32[128]" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
    add_49: "f32[128]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
    unsqueeze_36: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_37: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_69: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
    unsqueeze_38: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_39: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_50: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_51: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(add_50, relu_5);  add_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_8: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat: "f32[8, 256, 56, 56]" = torch.ops.aten.cat.default([relu_8, relu_5], 1)
    convolution_10: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(cat, primals_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_350, 1)
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1, 1]" = var_mean_10[0]
    getitem_23: "f32[1, 128, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_53: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_10: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_10: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_23)
    mul_70: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_31: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_71: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_72: "f32[128]" = torch.ops.aten.mul.Tensor(primals_348, 0.9)
    add_54: "f32[128]" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
    squeeze_32: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_73: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000398612827361);  squeeze_32 = None
    mul_74: "f32[128]" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
    mul_75: "f32[128]" = torch.ops.aten.mul.Tensor(primals_349, 0.9)
    add_55: "f32[128]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    unsqueeze_40: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1)
    unsqueeze_41: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_76: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
    unsqueeze_42: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
    unsqueeze_43: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_56: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_57: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(add_56, relu_8);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_9: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices.default(relu_9, [2, 2], [2, 2])
    getitem_24: "f32[8, 128, 28, 28]" = max_pool2d_with_indices_1[0]
    getitem_25: "i64[8, 128, 28, 28]" = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_3 = torch.ops.aten.max_pool2d_with_indices.default(relu_9, [2, 2], [2, 2])
    getitem_28: "f32[8, 128, 28, 28]" = max_pool2d_with_indices_3[0]
    getitem_29: "i64[8, 128, 28, 28]" = max_pool2d_with_indices_3[1];  max_pool2d_with_indices_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    convolution_11: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(getitem_28, primals_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_58: "i64[]" = torch.ops.aten.add.Tensor(primals_353, 1)
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 256, 1, 1]" = var_mean_11[0]
    getitem_31: "f32[1, 256, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_59: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_11: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_11: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_31)
    mul_77: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_34: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_78: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_79: "f32[256]" = torch.ops.aten.mul.Tensor(primals_351, 0.9)
    add_60: "f32[256]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_35: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_80: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001594642002871);  squeeze_35 = None
    mul_81: "f32[256]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[256]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
    add_61: "f32[256]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_44: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_45: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_83: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
    unsqueeze_46: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_47: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_62: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_12: "f32[8, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_9, primals_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_63: "i64[]" = torch.ops.aten.add.Tensor(primals_356, 1)
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1, 1]" = var_mean_12[0]
    getitem_33: "f32[1, 128, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_64: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_12: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_12: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_33)
    mul_84: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_37: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_85: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_86: "f32[128]" = torch.ops.aten.mul.Tensor(primals_354, 0.9)
    add_65: "f32[128]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_38: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_87: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000398612827361);  squeeze_38 = None
    mul_88: "f32[128]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[128]" = torch.ops.aten.mul.Tensor(primals_355, 0.9)
    add_66: "f32[128]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_48: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1)
    unsqueeze_49: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_90: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
    unsqueeze_50: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
    unsqueeze_51: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_67: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_10: "f32[8, 128, 56, 56]" = torch.ops.aten.relu.default(add_67);  add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_13: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_10, primals_40, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_359, 1)
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1, 1]" = var_mean_13[0]
    getitem_35: "f32[1, 128, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_69: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_13: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_13: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_35)
    mul_91: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_40: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_92: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_93: "f32[128]" = torch.ops.aten.mul.Tensor(primals_357, 0.9)
    add_70: "f32[128]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    squeeze_41: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_94: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001594642002871);  squeeze_41 = None
    mul_95: "f32[128]" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
    mul_96: "f32[128]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
    add_71: "f32[128]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
    unsqueeze_52: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_53: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_97: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
    unsqueeze_54: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_55: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_72: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_11: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_72);  add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_14: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_11, primals_43, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_362, 1)
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 256, 1, 1]" = var_mean_14[0]
    getitem_37: "f32[1, 256, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_74: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_14: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_14: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_37)
    mul_98: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_43: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_99: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_100: "f32[256]" = torch.ops.aten.mul.Tensor(primals_360, 0.9)
    add_75: "f32[256]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
    squeeze_44: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_101: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001594642002871);  squeeze_44 = None
    mul_102: "f32[256]" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
    mul_103: "f32[256]" = torch.ops.aten.mul.Tensor(primals_361, 0.9)
    add_76: "f32[256]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    unsqueeze_56: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1)
    unsqueeze_57: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_104: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
    unsqueeze_58: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
    unsqueeze_59: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_77: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_78: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_77, add_62);  add_77 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_12: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_78);  add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_15: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_79: "i64[]" = torch.ops.aten.add.Tensor(primals_365, 1)
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 128, 1, 1]" = var_mean_15[0]
    getitem_39: "f32[1, 128, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_80: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_15: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_15: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_39)
    mul_105: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_46: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_106: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_107: "f32[128]" = torch.ops.aten.mul.Tensor(primals_363, 0.9)
    add_81: "f32[128]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    squeeze_47: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_108: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001594642002871);  squeeze_47 = None
    mul_109: "f32[128]" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
    mul_110: "f32[128]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
    add_82: "f32[128]" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
    unsqueeze_60: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_61: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_111: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
    unsqueeze_62: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_63: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_83: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_13: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_83);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_16: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_13, primals_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_84: "i64[]" = torch.ops.aten.add.Tensor(primals_368, 1)
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1, 1]" = var_mean_16[0]
    getitem_41: "f32[1, 128, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_85: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_16: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_16: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_41)
    mul_112: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_49: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_113: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_114: "f32[128]" = torch.ops.aten.mul.Tensor(primals_366, 0.9)
    add_86: "f32[128]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
    squeeze_50: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_115: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
    mul_116: "f32[128]" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
    mul_117: "f32[128]" = torch.ops.aten.mul.Tensor(primals_367, 0.9)
    add_87: "f32[128]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    unsqueeze_64: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
    unsqueeze_65: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_118: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
    unsqueeze_66: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
    unsqueeze_67: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_88: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_14: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_88);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_17: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_14, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_89: "i64[]" = torch.ops.aten.add.Tensor(primals_371, 1)
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 256, 1, 1]" = var_mean_17[0]
    getitem_43: "f32[1, 256, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_90: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_17: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_17: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_43)
    mul_119: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_52: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_120: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_121: "f32[256]" = torch.ops.aten.mul.Tensor(primals_369, 0.9)
    add_91: "f32[256]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
    squeeze_53: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_122: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001594642002871);  squeeze_53 = None
    mul_123: "f32[256]" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
    mul_124: "f32[256]" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
    add_92: "f32[256]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    unsqueeze_68: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_69: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_125: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
    unsqueeze_70: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_71: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_93: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_94: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_93, relu_12);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_15: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_94);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_1: "f32[8, 512, 28, 28]" = torch.ops.aten.cat.default([relu_15, relu_12], 1)
    convolution_18: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(cat_1, primals_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_95: "i64[]" = torch.ops.aten.add.Tensor(primals_374, 1)
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 256, 1, 1]" = var_mean_18[0]
    getitem_45: "f32[1, 256, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_96: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_18: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_18: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_45)
    mul_126: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_55: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_127: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_128: "f32[256]" = torch.ops.aten.mul.Tensor(primals_372, 0.9)
    add_97: "f32[256]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    squeeze_56: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_129: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
    mul_130: "f32[256]" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
    mul_131: "f32[256]" = torch.ops.aten.mul.Tensor(primals_373, 0.9)
    add_98: "f32[256]" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
    unsqueeze_72: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
    unsqueeze_73: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_132: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
    unsqueeze_74: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
    unsqueeze_75: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_99: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_100: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_99, relu_15);  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_16: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_100);  add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_19: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_16, primals_58, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_101: "i64[]" = torch.ops.aten.add.Tensor(primals_377, 1)
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128, 1, 1]" = var_mean_19[0]
    getitem_47: "f32[1, 128, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_102: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_19: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_19: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_47)
    mul_133: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_58: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_134: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_135: "f32[128]" = torch.ops.aten.mul.Tensor(primals_375, 0.9)
    add_103: "f32[128]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    squeeze_59: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_136: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
    mul_137: "f32[128]" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
    mul_138: "f32[128]" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
    add_104: "f32[128]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
    unsqueeze_76: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_77: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_139: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
    unsqueeze_78: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_79: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_105: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_17: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_105);  add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_20: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_17, primals_61, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_106: "i64[]" = torch.ops.aten.add.Tensor(primals_380, 1)
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 128, 1, 1]" = var_mean_20[0]
    getitem_49: "f32[1, 128, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_107: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_20: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
    sub_20: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_49)
    mul_140: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_61: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_141: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_142: "f32[128]" = torch.ops.aten.mul.Tensor(primals_378, 0.9)
    add_108: "f32[128]" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
    squeeze_62: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_143: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
    mul_144: "f32[128]" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
    mul_145: "f32[128]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
    add_109: "f32[128]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    unsqueeze_80: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
    unsqueeze_81: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_146: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
    unsqueeze_82: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
    unsqueeze_83: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_110: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_18: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_110);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_21: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_18, primals_64, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_111: "i64[]" = torch.ops.aten.add.Tensor(primals_383, 1)
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 256, 1, 1]" = var_mean_21[0]
    getitem_51: "f32[1, 256, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_112: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_21: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_21: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_51)
    mul_147: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_64: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_148: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_149: "f32[256]" = torch.ops.aten.mul.Tensor(primals_381, 0.9)
    add_113: "f32[256]" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
    squeeze_65: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_150: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
    mul_151: "f32[256]" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
    mul_152: "f32[256]" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
    add_114: "f32[256]" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
    unsqueeze_84: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_85: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_153: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
    unsqueeze_86: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_87: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_115: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_116: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_115, relu_16);  add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_19: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_116);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_22: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_19, primals_67, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_117: "i64[]" = torch.ops.aten.add.Tensor(primals_386, 1)
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1, 1]" = var_mean_22[0]
    getitem_53: "f32[1, 128, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_118: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_22: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_22: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_53)
    mul_154: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_67: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_155: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_156: "f32[128]" = torch.ops.aten.mul.Tensor(primals_384, 0.9)
    add_119: "f32[128]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    squeeze_68: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_157: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001594642002871);  squeeze_68 = None
    mul_158: "f32[128]" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
    mul_159: "f32[128]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
    add_120: "f32[128]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    unsqueeze_88: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
    unsqueeze_89: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_160: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
    unsqueeze_90: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1);  primals_69 = None
    unsqueeze_91: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_121: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_20: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_121);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_23: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_20, primals_70, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_122: "i64[]" = torch.ops.aten.add.Tensor(primals_389, 1)
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 128, 1, 1]" = var_mean_23[0]
    getitem_55: "f32[1, 128, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_123: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_23: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_23: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_55)
    mul_161: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_70: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_162: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_163: "f32[128]" = torch.ops.aten.mul.Tensor(primals_387, 0.9)
    add_124: "f32[128]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    squeeze_71: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_164: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001594642002871);  squeeze_71 = None
    mul_165: "f32[128]" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
    mul_166: "f32[128]" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
    add_125: "f32[128]" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    unsqueeze_92: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_93: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_167: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
    unsqueeze_94: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_95: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_126: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_21: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_126);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_24: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_21, primals_73, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_127: "i64[]" = torch.ops.aten.add.Tensor(primals_392, 1)
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 256, 1, 1]" = var_mean_24[0]
    getitem_57: "f32[1, 256, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_128: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_24: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    sub_24: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_57)
    mul_168: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_73: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_169: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_170: "f32[256]" = torch.ops.aten.mul.Tensor(primals_390, 0.9)
    add_129: "f32[256]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
    squeeze_74: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_171: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001594642002871);  squeeze_74 = None
    mul_172: "f32[256]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
    mul_173: "f32[256]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
    add_130: "f32[256]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    unsqueeze_96: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1)
    unsqueeze_97: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_174: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
    unsqueeze_98: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1);  primals_75 = None
    unsqueeze_99: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_131: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_132: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_131, relu_19);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_22: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_132);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_2: "f32[8, 768, 28, 28]" = torch.ops.aten.cat.default([relu_22, relu_19, relu_16], 1)
    convolution_25: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(cat_2, primals_76, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_133: "i64[]" = torch.ops.aten.add.Tensor(primals_395, 1)
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_25, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 256, 1, 1]" = var_mean_25[0]
    getitem_59: "f32[1, 256, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_134: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_25: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_25: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_59)
    mul_175: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_76: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_176: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_177: "f32[256]" = torch.ops.aten.mul.Tensor(primals_393, 0.9)
    add_135: "f32[256]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
    squeeze_77: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_178: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001594642002871);  squeeze_77 = None
    mul_179: "f32[256]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
    mul_180: "f32[256]" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
    add_136: "f32[256]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    unsqueeze_100: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_101: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_181: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
    unsqueeze_102: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_103: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_137: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_138: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_137, relu_22);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_23: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_138);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_26: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_23, primals_79, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_139: "i64[]" = torch.ops.aten.add.Tensor(primals_398, 1)
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1, 1]" = var_mean_26[0]
    getitem_61: "f32[1, 128, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_140: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_26: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    sub_26: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_61)
    mul_182: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_79: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_183: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_184: "f32[128]" = torch.ops.aten.mul.Tensor(primals_396, 0.9)
    add_141: "f32[128]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
    squeeze_80: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_185: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001594642002871);  squeeze_80 = None
    mul_186: "f32[128]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
    mul_187: "f32[128]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
    add_142: "f32[128]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
    unsqueeze_104: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1)
    unsqueeze_105: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_188: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
    unsqueeze_106: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1);  primals_81 = None
    unsqueeze_107: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_143: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_24: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_143);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_27: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_24, primals_82, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_144: "i64[]" = torch.ops.aten.add.Tensor(primals_401, 1)
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 128, 1, 1]" = var_mean_27[0]
    getitem_63: "f32[1, 128, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_145: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_27: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    sub_27: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_63)
    mul_189: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_82: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_190: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_191: "f32[128]" = torch.ops.aten.mul.Tensor(primals_399, 0.9)
    add_146: "f32[128]" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
    squeeze_83: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_192: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0001594642002871);  squeeze_83 = None
    mul_193: "f32[128]" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
    mul_194: "f32[128]" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
    add_147: "f32[128]" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
    unsqueeze_108: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_109: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_195: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
    unsqueeze_110: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_111: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_148: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_25: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_148);  add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_28: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_25, primals_85, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_149: "i64[]" = torch.ops.aten.add.Tensor(primals_404, 1)
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 256, 1, 1]" = var_mean_28[0]
    getitem_65: "f32[1, 256, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_150: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_28: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_28: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_65)
    mul_196: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_85: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_197: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_198: "f32[256]" = torch.ops.aten.mul.Tensor(primals_402, 0.9)
    add_151: "f32[256]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
    squeeze_86: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_199: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0001594642002871);  squeeze_86 = None
    mul_200: "f32[256]" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
    mul_201: "f32[256]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
    add_152: "f32[256]" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
    unsqueeze_112: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1)
    unsqueeze_113: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_202: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
    unsqueeze_114: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1);  primals_87 = None
    unsqueeze_115: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_153: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_154: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_153, relu_23);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_26: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_154);  add_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_29: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_26, primals_88, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_155: "i64[]" = torch.ops.aten.add.Tensor(primals_407, 1)
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1, 1]" = var_mean_29[0]
    getitem_67: "f32[1, 128, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_156: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_29: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    sub_29: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_67)
    mul_203: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_88: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_204: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_205: "f32[128]" = torch.ops.aten.mul.Tensor(primals_405, 0.9)
    add_157: "f32[128]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_89: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_206: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0001594642002871);  squeeze_89 = None
    mul_207: "f32[128]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[128]" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
    add_158: "f32[128]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_116: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_117: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_209: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
    unsqueeze_118: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_119: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_159: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_27: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_159);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_30: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_27, primals_91, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_160: "i64[]" = torch.ops.aten.add.Tensor(primals_410, 1)
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_30, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1, 1]" = var_mean_30[0]
    getitem_69: "f32[1, 128, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_161: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_30: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_30: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_69)
    mul_210: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_91: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_211: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_212: "f32[128]" = torch.ops.aten.mul.Tensor(primals_408, 0.9)
    add_162: "f32[128]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
    squeeze_92: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_213: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0001594642002871);  squeeze_92 = None
    mul_214: "f32[128]" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
    mul_215: "f32[128]" = torch.ops.aten.mul.Tensor(primals_409, 0.9)
    add_163: "f32[128]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    unsqueeze_120: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1)
    unsqueeze_121: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_216: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
    unsqueeze_122: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
    unsqueeze_123: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_164: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_28: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_164);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_31: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_28, primals_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_165: "i64[]" = torch.ops.aten.add.Tensor(primals_413, 1)
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 256, 1, 1]" = var_mean_31[0]
    getitem_71: "f32[1, 256, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_166: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_31: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    sub_31: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_71)
    mul_217: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_94: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_218: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_219: "f32[256]" = torch.ops.aten.mul.Tensor(primals_411, 0.9)
    add_167: "f32[256]" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
    squeeze_95: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_220: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0001594642002871);  squeeze_95 = None
    mul_221: "f32[256]" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
    mul_222: "f32[256]" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
    add_168: "f32[256]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
    unsqueeze_124: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_125: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_223: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
    unsqueeze_126: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_127: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_169: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_170: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_169, relu_26);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_29: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_170);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_3: "f32[8, 512, 28, 28]" = torch.ops.aten.cat.default([relu_29, relu_26], 1)
    convolution_32: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(cat_3, primals_97, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_171: "i64[]" = torch.ops.aten.add.Tensor(primals_416, 1)
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 256, 1, 1]" = var_mean_32[0]
    getitem_73: "f32[1, 256, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_172: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_32: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_32: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_73)
    mul_224: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_97: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_225: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_226: "f32[256]" = torch.ops.aten.mul.Tensor(primals_414, 0.9)
    add_173: "f32[256]" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
    squeeze_98: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_227: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0001594642002871);  squeeze_98 = None
    mul_228: "f32[256]" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
    mul_229: "f32[256]" = torch.ops.aten.mul.Tensor(primals_415, 0.9)
    add_174: "f32[256]" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
    unsqueeze_128: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1)
    unsqueeze_129: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_230: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
    unsqueeze_130: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
    unsqueeze_131: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_175: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_176: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_175, relu_29);  add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_30: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_176);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_33: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_30, primals_100, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_177: "i64[]" = torch.ops.aten.add.Tensor(primals_419, 1)
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 128, 1, 1]" = var_mean_33[0]
    getitem_75: "f32[1, 128, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_178: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_33: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_33: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_75)
    mul_231: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_100: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_232: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_233: "f32[128]" = torch.ops.aten.mul.Tensor(primals_417, 0.9)
    add_179: "f32[128]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    squeeze_101: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_234: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0001594642002871);  squeeze_101 = None
    mul_235: "f32[128]" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
    mul_236: "f32[128]" = torch.ops.aten.mul.Tensor(primals_418, 0.9)
    add_180: "f32[128]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
    unsqueeze_132: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_133: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_237: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
    unsqueeze_134: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_135: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_181: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_31: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_181);  add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_34: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_31, primals_103, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_182: "i64[]" = torch.ops.aten.add.Tensor(primals_422, 1)
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1, 1]" = var_mean_34[0]
    getitem_77: "f32[1, 128, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_183: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_34: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    sub_34: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_77)
    mul_238: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_103: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_239: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_240: "f32[128]" = torch.ops.aten.mul.Tensor(primals_420, 0.9)
    add_184: "f32[128]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_104: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_241: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0001594642002871);  squeeze_104 = None
    mul_242: "f32[128]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[128]" = torch.ops.aten.mul.Tensor(primals_421, 0.9)
    add_185: "f32[128]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_136: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1)
    unsqueeze_137: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_244: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
    unsqueeze_138: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
    unsqueeze_139: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_186: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_32: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_186);  add_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_35: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_32, primals_106, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_187: "i64[]" = torch.ops.aten.add.Tensor(primals_425, 1)
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_35, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 256, 1, 1]" = var_mean_35[0]
    getitem_79: "f32[1, 256, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_188: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_35: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    sub_35: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_79)
    mul_245: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_106: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_246: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_247: "f32[256]" = torch.ops.aten.mul.Tensor(primals_423, 0.9)
    add_189: "f32[256]" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    squeeze_107: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_248: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0001594642002871);  squeeze_107 = None
    mul_249: "f32[256]" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
    mul_250: "f32[256]" = torch.ops.aten.mul.Tensor(primals_424, 0.9)
    add_190: "f32[256]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    unsqueeze_140: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_141: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_251: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
    unsqueeze_142: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_143: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_191: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_192: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_191, relu_30);  add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_33: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_192);  add_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_36: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_33, primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_193: "i64[]" = torch.ops.aten.add.Tensor(primals_428, 1)
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1, 1]" = var_mean_36[0]
    getitem_81: "f32[1, 128, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_194: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_36: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_36: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_81)
    mul_252: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_109: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_253: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_254: "f32[128]" = torch.ops.aten.mul.Tensor(primals_426, 0.9)
    add_195: "f32[128]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    squeeze_110: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_255: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0001594642002871);  squeeze_110 = None
    mul_256: "f32[128]" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
    mul_257: "f32[128]" = torch.ops.aten.mul.Tensor(primals_427, 0.9)
    add_196: "f32[128]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    unsqueeze_144: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1)
    unsqueeze_145: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_258: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
    unsqueeze_146: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
    unsqueeze_147: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_197: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_34: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_197);  add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_37: "f32[8, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_34, primals_112, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_198: "i64[]" = torch.ops.aten.add.Tensor(primals_431, 1)
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1, 1]" = var_mean_37[0]
    getitem_83: "f32[1, 128, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_199: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_37: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_37: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_83)
    mul_259: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_112: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_260: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_261: "f32[128]" = torch.ops.aten.mul.Tensor(primals_429, 0.9)
    add_200: "f32[128]" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
    squeeze_113: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_262: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0001594642002871);  squeeze_113 = None
    mul_263: "f32[128]" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
    mul_264: "f32[128]" = torch.ops.aten.mul.Tensor(primals_430, 0.9)
    add_201: "f32[128]" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
    unsqueeze_148: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_149: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_265: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
    unsqueeze_150: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_151: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_202: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_35: "f32[8, 128, 28, 28]" = torch.ops.aten.relu.default(add_202);  add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_38: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_35, primals_115, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_203: "i64[]" = torch.ops.aten.add.Tensor(primals_434, 1)
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 256, 1, 1]" = var_mean_38[0]
    getitem_85: "f32[1, 256, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_204: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_38: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
    sub_38: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_85)
    mul_266: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_115: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_267: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_268: "f32[256]" = torch.ops.aten.mul.Tensor(primals_432, 0.9)
    add_205: "f32[256]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    squeeze_116: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_269: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0001594642002871);  squeeze_116 = None
    mul_270: "f32[256]" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
    mul_271: "f32[256]" = torch.ops.aten.mul.Tensor(primals_433, 0.9)
    add_206: "f32[256]" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    unsqueeze_152: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
    unsqueeze_153: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_272: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
    unsqueeze_154: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
    unsqueeze_155: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_207: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_208: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_207, relu_33);  add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_36: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_208);  add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_4: "f32[8, 1152, 28, 28]" = torch.ops.aten.cat.default([relu_36, relu_33, getitem_24, relu_23, relu_30], 1);  getitem_24 = None
    convolution_39: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(cat_4, primals_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_209: "i64[]" = torch.ops.aten.add.Tensor(primals_437, 1)
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 256, 1, 1]" = var_mean_39[0]
    getitem_87: "f32[1, 256, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_210: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_39: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_39: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_87)
    mul_273: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_118: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_274: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_275: "f32[256]" = torch.ops.aten.mul.Tensor(primals_435, 0.9)
    add_211: "f32[256]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    squeeze_119: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_276: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0001594642002871);  squeeze_119 = None
    mul_277: "f32[256]" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
    mul_278: "f32[256]" = torch.ops.aten.mul.Tensor(primals_436, 0.9)
    add_212: "f32[256]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    unsqueeze_156: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_157: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_279: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
    unsqueeze_158: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_159: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_213: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_214: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_213, relu_36);  add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_37: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_214);  add_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_4 = torch.ops.aten.max_pool2d_with_indices.default(relu_37, [2, 2], [2, 2])
    getitem_88: "f32[8, 256, 14, 14]" = max_pool2d_with_indices_4[0]
    getitem_89: "i64[8, 256, 14, 14]" = max_pool2d_with_indices_4[1];  max_pool2d_with_indices_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_7 = torch.ops.aten.max_pool2d_with_indices.default(relu_37, [2, 2], [2, 2])
    getitem_94: "f32[8, 256, 14, 14]" = max_pool2d_with_indices_7[0]
    getitem_95: "i64[8, 256, 14, 14]" = max_pool2d_with_indices_7[1];  max_pool2d_with_indices_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    convolution_40: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(getitem_94, primals_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_215: "i64[]" = torch.ops.aten.add.Tensor(primals_440, 1)
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 512, 1, 1]" = var_mean_40[0]
    getitem_97: "f32[1, 512, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_216: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_40: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
    sub_40: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_97)
    mul_280: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_121: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_281: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_282: "f32[512]" = torch.ops.aten.mul.Tensor(primals_438, 0.9)
    add_217: "f32[512]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
    squeeze_122: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_283: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0006381620931717);  squeeze_122 = None
    mul_284: "f32[512]" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
    mul_285: "f32[512]" = torch.ops.aten.mul.Tensor(primals_439, 0.9)
    add_218: "f32[512]" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
    unsqueeze_160: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
    unsqueeze_161: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_286: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
    unsqueeze_162: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
    unsqueeze_163: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_219: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_41: "f32[8, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_37, primals_124, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_220: "i64[]" = torch.ops.aten.add.Tensor(primals_443, 1)
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 256, 1, 1]" = var_mean_41[0]
    getitem_99: "f32[1, 256, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_221: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
    rsqrt_41: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
    sub_41: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_99)
    mul_287: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_124: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_288: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_289: "f32[256]" = torch.ops.aten.mul.Tensor(primals_441, 0.9)
    add_222: "f32[256]" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    squeeze_125: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_290: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0001594642002871);  squeeze_125 = None
    mul_291: "f32[256]" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
    mul_292: "f32[256]" = torch.ops.aten.mul.Tensor(primals_442, 0.9)
    add_223: "f32[256]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    unsqueeze_164: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_165: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_293: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
    unsqueeze_166: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_167: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_224: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_38: "f32[8, 256, 28, 28]" = torch.ops.aten.relu.default(add_224);  add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_42: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_38, primals_127, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_225: "i64[]" = torch.ops.aten.add.Tensor(primals_446, 1)
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 256, 1, 1]" = var_mean_42[0]
    getitem_101: "f32[1, 256, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_226: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
    rsqrt_42: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
    sub_42: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_101)
    mul_294: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_127: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_295: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_296: "f32[256]" = torch.ops.aten.mul.Tensor(primals_444, 0.9)
    add_227: "f32[256]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_128: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_297: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0006381620931717);  squeeze_128 = None
    mul_298: "f32[256]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[256]" = torch.ops.aten.mul.Tensor(primals_445, 0.9)
    add_228: "f32[256]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_168: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
    unsqueeze_169: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_300: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
    unsqueeze_170: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
    unsqueeze_171: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_229: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_39: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_229);  add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_43: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_39, primals_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_230: "i64[]" = torch.ops.aten.add.Tensor(primals_449, 1)
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 512, 1, 1]" = var_mean_43[0]
    getitem_103: "f32[1, 512, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_231: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_43: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_43: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_103)
    mul_301: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_130: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_302: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_303: "f32[512]" = torch.ops.aten.mul.Tensor(primals_447, 0.9)
    add_232: "f32[512]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    squeeze_131: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_304: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0006381620931717);  squeeze_131 = None
    mul_305: "f32[512]" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
    mul_306: "f32[512]" = torch.ops.aten.mul.Tensor(primals_448, 0.9)
    add_233: "f32[512]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_172: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_173: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_307: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
    unsqueeze_174: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_175: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_234: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_235: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_234, add_219);  add_234 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_40: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_235);  add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_44: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_40, primals_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_236: "i64[]" = torch.ops.aten.add.Tensor(primals_452, 1)
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 256, 1, 1]" = var_mean_44[0]
    getitem_105: "f32[1, 256, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_237: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
    rsqrt_44: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
    sub_44: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_105)
    mul_308: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_133: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_309: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_310: "f32[256]" = torch.ops.aten.mul.Tensor(primals_450, 0.9)
    add_238: "f32[256]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    squeeze_134: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_311: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0006381620931717);  squeeze_134 = None
    mul_312: "f32[256]" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
    mul_313: "f32[256]" = torch.ops.aten.mul.Tensor(primals_451, 0.9)
    add_239: "f32[256]" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_176: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
    unsqueeze_177: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_314: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
    unsqueeze_178: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
    unsqueeze_179: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_240: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_41: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_240);  add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_45: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_41, primals_136, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_241: "i64[]" = torch.ops.aten.add.Tensor(primals_455, 1)
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 256, 1, 1]" = var_mean_45[0]
    getitem_107: "f32[1, 256, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_242: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_45: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
    sub_45: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_107)
    mul_315: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_136: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_316: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_317: "f32[256]" = torch.ops.aten.mul.Tensor(primals_453, 0.9)
    add_243: "f32[256]" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    squeeze_137: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_318: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0006381620931717);  squeeze_137 = None
    mul_319: "f32[256]" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
    mul_320: "f32[256]" = torch.ops.aten.mul.Tensor(primals_454, 0.9)
    add_244: "f32[256]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    unsqueeze_180: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
    unsqueeze_181: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_321: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
    unsqueeze_182: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
    unsqueeze_183: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_245: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_42: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_245);  add_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_46: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_42, primals_139, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_246: "i64[]" = torch.ops.aten.add.Tensor(primals_458, 1)
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 512, 1, 1]" = var_mean_46[0]
    getitem_109: "f32[1, 512, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_247: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
    rsqrt_46: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
    sub_46: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_109)
    mul_322: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_139: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_323: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_324: "f32[512]" = torch.ops.aten.mul.Tensor(primals_456, 0.9)
    add_248: "f32[512]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    squeeze_140: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_325: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0006381620931717);  squeeze_140 = None
    mul_326: "f32[512]" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
    mul_327: "f32[512]" = torch.ops.aten.mul.Tensor(primals_457, 0.9)
    add_249: "f32[512]" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    unsqueeze_184: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
    unsqueeze_185: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_328: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
    unsqueeze_186: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1);  primals_141 = None
    unsqueeze_187: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_250: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_251: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_250, relu_40);  add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_43: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_251);  add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_5: "f32[8, 1024, 14, 14]" = torch.ops.aten.cat.default([relu_43, relu_40], 1)
    convolution_47: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(cat_5, primals_142, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_252: "i64[]" = torch.ops.aten.add.Tensor(primals_461, 1)
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 512, 1, 1]" = var_mean_47[0]
    getitem_111: "f32[1, 512, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_253: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
    rsqrt_47: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_253);  add_253 = None
    sub_47: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_111)
    mul_329: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_142: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_330: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_331: "f32[512]" = torch.ops.aten.mul.Tensor(primals_459, 0.9)
    add_254: "f32[512]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    squeeze_143: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_332: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0006381620931717);  squeeze_143 = None
    mul_333: "f32[512]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
    mul_334: "f32[512]" = torch.ops.aten.mul.Tensor(primals_460, 0.9)
    add_255: "f32[512]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    unsqueeze_188: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_189: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_335: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
    unsqueeze_190: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_191: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_256: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_257: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_256, relu_43);  add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_44: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_257);  add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_48: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_44, primals_145, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_258: "i64[]" = torch.ops.aten.add.Tensor(primals_464, 1)
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 256, 1, 1]" = var_mean_48[0]
    getitem_113: "f32[1, 256, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_259: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_48: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_259);  add_259 = None
    sub_48: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_113)
    mul_336: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_145: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_337: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_338: "f32[256]" = torch.ops.aten.mul.Tensor(primals_462, 0.9)
    add_260: "f32[256]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
    squeeze_146: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_339: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0006381620931717);  squeeze_146 = None
    mul_340: "f32[256]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
    mul_341: "f32[256]" = torch.ops.aten.mul.Tensor(primals_463, 0.9)
    add_261: "f32[256]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
    unsqueeze_192: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1)
    unsqueeze_193: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_342: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
    unsqueeze_194: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1);  primals_147 = None
    unsqueeze_195: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_262: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_45: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_262);  add_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_49: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_45, primals_148, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_263: "i64[]" = torch.ops.aten.add.Tensor(primals_467, 1)
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 256, 1, 1]" = var_mean_49[0]
    getitem_115: "f32[1, 256, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_264: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
    rsqrt_49: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
    sub_49: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_115)
    mul_343: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_148: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_344: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_345: "f32[256]" = torch.ops.aten.mul.Tensor(primals_465, 0.9)
    add_265: "f32[256]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    squeeze_149: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_346: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0006381620931717);  squeeze_149 = None
    mul_347: "f32[256]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
    mul_348: "f32[256]" = torch.ops.aten.mul.Tensor(primals_466, 0.9)
    add_266: "f32[256]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    unsqueeze_196: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1)
    unsqueeze_197: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_349: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
    unsqueeze_198: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
    unsqueeze_199: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_267: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_46: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_267);  add_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_50: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_46, primals_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_268: "i64[]" = torch.ops.aten.add.Tensor(primals_470, 1)
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[1, 512, 1, 1]" = var_mean_50[0]
    getitem_117: "f32[1, 512, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_269: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
    rsqrt_50: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_269);  add_269 = None
    sub_50: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_117)
    mul_350: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
    squeeze_151: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_351: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_352: "f32[512]" = torch.ops.aten.mul.Tensor(primals_468, 0.9)
    add_270: "f32[512]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    squeeze_152: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
    mul_353: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0006381620931717);  squeeze_152 = None
    mul_354: "f32[512]" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
    mul_355: "f32[512]" = torch.ops.aten.mul.Tensor(primals_469, 0.9)
    add_271: "f32[512]" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_200: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1)
    unsqueeze_201: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_356: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
    unsqueeze_202: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1);  primals_153 = None
    unsqueeze_203: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_272: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_273: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_272, relu_44);  add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_47: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_273);  add_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_51: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_47, primals_154, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_274: "i64[]" = torch.ops.aten.add.Tensor(primals_473, 1)
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 256, 1, 1]" = var_mean_51[0]
    getitem_119: "f32[1, 256, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_275: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_51: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
    sub_51: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_119)
    mul_357: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_154: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_358: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_359: "f32[256]" = torch.ops.aten.mul.Tensor(primals_471, 0.9)
    add_276: "f32[256]" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
    squeeze_155: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_360: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0006381620931717);  squeeze_155 = None
    mul_361: "f32[256]" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
    mul_362: "f32[256]" = torch.ops.aten.mul.Tensor(primals_472, 0.9)
    add_277: "f32[256]" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    unsqueeze_204: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
    unsqueeze_205: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_363: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
    unsqueeze_206: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
    unsqueeze_207: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_278: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_48: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_278);  add_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_52: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_48, primals_157, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_279: "i64[]" = torch.ops.aten.add.Tensor(primals_476, 1)
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 256, 1, 1]" = var_mean_52[0]
    getitem_121: "f32[1, 256, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_280: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
    rsqrt_52: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
    sub_52: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_121)
    mul_364: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_157: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_365: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_366: "f32[256]" = torch.ops.aten.mul.Tensor(primals_474, 0.9)
    add_281: "f32[256]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    squeeze_158: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_367: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0006381620931717);  squeeze_158 = None
    mul_368: "f32[256]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
    mul_369: "f32[256]" = torch.ops.aten.mul.Tensor(primals_475, 0.9)
    add_282: "f32[256]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    unsqueeze_208: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_158, -1)
    unsqueeze_209: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_370: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
    unsqueeze_210: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1);  primals_159 = None
    unsqueeze_211: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_283: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_49: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_283);  add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_53: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_49, primals_160, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_284: "i64[]" = torch.ops.aten.add.Tensor(primals_479, 1)
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[1, 512, 1, 1]" = var_mean_53[0]
    getitem_123: "f32[1, 512, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_285: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
    rsqrt_53: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
    sub_53: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_123)
    mul_371: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    squeeze_160: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_372: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_373: "f32[512]" = torch.ops.aten.mul.Tensor(primals_477, 0.9)
    add_286: "f32[512]" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    squeeze_161: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
    mul_374: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0006381620931717);  squeeze_161 = None
    mul_375: "f32[512]" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
    mul_376: "f32[512]" = torch.ops.aten.mul.Tensor(primals_478, 0.9)
    add_287: "f32[512]" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_212: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_161, -1)
    unsqueeze_213: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_377: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
    unsqueeze_214: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1);  primals_162 = None
    unsqueeze_215: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_288: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_289: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_288, relu_47);  add_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_50: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_289);  add_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_6: "f32[8, 1536, 14, 14]" = torch.ops.aten.cat.default([relu_50, relu_47, relu_44], 1)
    convolution_54: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(cat_6, primals_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_290: "i64[]" = torch.ops.aten.add.Tensor(primals_482, 1)
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 512, 1, 1]" = var_mean_54[0]
    getitem_125: "f32[1, 512, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_291: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
    rsqrt_54: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_291);  add_291 = None
    sub_54: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_125)
    mul_378: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_163: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_379: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_380: "f32[512]" = torch.ops.aten.mul.Tensor(primals_480, 0.9)
    add_292: "f32[512]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    squeeze_164: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_381: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0006381620931717);  squeeze_164 = None
    mul_382: "f32[512]" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
    mul_383: "f32[512]" = torch.ops.aten.mul.Tensor(primals_481, 0.9)
    add_293: "f32[512]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    unsqueeze_216: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_164, -1)
    unsqueeze_217: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_384: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
    unsqueeze_218: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_165, -1);  primals_165 = None
    unsqueeze_219: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_294: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_295: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_294, relu_50);  add_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_51: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_295);  add_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_55: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_51, primals_166, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_296: "i64[]" = torch.ops.aten.add.Tensor(primals_485, 1)
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 256, 1, 1]" = var_mean_55[0]
    getitem_127: "f32[1, 256, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_297: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
    rsqrt_55: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_297);  add_297 = None
    sub_55: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_127)
    mul_385: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_166: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_386: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_387: "f32[256]" = torch.ops.aten.mul.Tensor(primals_483, 0.9)
    add_298: "f32[256]" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
    squeeze_167: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_388: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0006381620931717);  squeeze_167 = None
    mul_389: "f32[256]" = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
    mul_390: "f32[256]" = torch.ops.aten.mul.Tensor(primals_484, 0.9)
    add_299: "f32[256]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    unsqueeze_220: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1)
    unsqueeze_221: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_391: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
    unsqueeze_222: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1);  primals_168 = None
    unsqueeze_223: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_300: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_52: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_300);  add_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_56: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_52, primals_169, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_301: "i64[]" = torch.ops.aten.add.Tensor(primals_488, 1)
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[1, 256, 1, 1]" = var_mean_56[0]
    getitem_129: "f32[1, 256, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_302: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
    rsqrt_56: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_302);  add_302 = None
    sub_56: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_129)
    mul_392: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
    squeeze_169: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_393: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_394: "f32[256]" = torch.ops.aten.mul.Tensor(primals_486, 0.9)
    add_303: "f32[256]" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    squeeze_170: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
    mul_395: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0006381620931717);  squeeze_170 = None
    mul_396: "f32[256]" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
    mul_397: "f32[256]" = torch.ops.aten.mul.Tensor(primals_487, 0.9)
    add_304: "f32[256]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    unsqueeze_224: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_170, -1)
    unsqueeze_225: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_398: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_225);  mul_392 = unsqueeze_225 = None
    unsqueeze_226: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_171, -1);  primals_171 = None
    unsqueeze_227: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_305: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_227);  mul_398 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_53: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_305);  add_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_57: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_53, primals_172, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_306: "i64[]" = torch.ops.aten.add.Tensor(primals_491, 1)
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
    getitem_130: "f32[1, 512, 1, 1]" = var_mean_57[0]
    getitem_131: "f32[1, 512, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_307: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05)
    rsqrt_57: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_307);  add_307 = None
    sub_57: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_131)
    mul_399: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
    squeeze_172: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_400: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_401: "f32[512]" = torch.ops.aten.mul.Tensor(primals_489, 0.9)
    add_308: "f32[512]" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
    squeeze_173: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
    mul_402: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0006381620931717);  squeeze_173 = None
    mul_403: "f32[512]" = torch.ops.aten.mul.Tensor(mul_402, 0.1);  mul_402 = None
    mul_404: "f32[512]" = torch.ops.aten.mul.Tensor(primals_490, 0.9)
    add_309: "f32[512]" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
    unsqueeze_228: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_173, -1)
    unsqueeze_229: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_405: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_229);  mul_399 = unsqueeze_229 = None
    unsqueeze_230: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_174, -1);  primals_174 = None
    unsqueeze_231: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_310: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_405, unsqueeze_231);  mul_405 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_311: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_310, relu_51);  add_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_54: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_311);  add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_58: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_54, primals_175, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_312: "i64[]" = torch.ops.aten.add.Tensor(primals_494, 1)
    var_mean_58 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_132: "f32[1, 256, 1, 1]" = var_mean_58[0]
    getitem_133: "f32[1, 256, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_313: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05)
    rsqrt_58: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_313);  add_313 = None
    sub_58: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_133)
    mul_406: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    squeeze_174: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
    squeeze_175: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_407: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_408: "f32[256]" = torch.ops.aten.mul.Tensor(primals_492, 0.9)
    add_314: "f32[256]" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    squeeze_176: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
    mul_409: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0006381620931717);  squeeze_176 = None
    mul_410: "f32[256]" = torch.ops.aten.mul.Tensor(mul_409, 0.1);  mul_409 = None
    mul_411: "f32[256]" = torch.ops.aten.mul.Tensor(primals_493, 0.9)
    add_315: "f32[256]" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    unsqueeze_232: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_176, -1)
    unsqueeze_233: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_412: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_233);  mul_406 = unsqueeze_233 = None
    unsqueeze_234: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_177, -1);  primals_177 = None
    unsqueeze_235: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_316: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_235);  mul_412 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_55: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_316);  add_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_59: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_55, primals_178, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_317: "i64[]" = torch.ops.aten.add.Tensor(primals_497, 1)
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[1, 256, 1, 1]" = var_mean_59[0]
    getitem_135: "f32[1, 256, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_318: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05)
    rsqrt_59: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_318);  add_318 = None
    sub_59: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_135)
    mul_413: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    squeeze_177: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
    squeeze_178: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    mul_414: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_415: "f32[256]" = torch.ops.aten.mul.Tensor(primals_495, 0.9)
    add_319: "f32[256]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    squeeze_179: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
    mul_416: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0006381620931717);  squeeze_179 = None
    mul_417: "f32[256]" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
    mul_418: "f32[256]" = torch.ops.aten.mul.Tensor(primals_496, 0.9)
    add_320: "f32[256]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    unsqueeze_236: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_179, -1)
    unsqueeze_237: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_419: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_237);  mul_413 = unsqueeze_237 = None
    unsqueeze_238: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_180, -1);  primals_180 = None
    unsqueeze_239: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_321: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_239);  mul_419 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_56: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_321);  add_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_60: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_56, primals_181, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_322: "i64[]" = torch.ops.aten.add.Tensor(primals_500, 1)
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
    getitem_136: "f32[1, 512, 1, 1]" = var_mean_60[0]
    getitem_137: "f32[1, 512, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_323: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05)
    rsqrt_60: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_323);  add_323 = None
    sub_60: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_137)
    mul_420: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    squeeze_180: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
    squeeze_181: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    mul_421: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_422: "f32[512]" = torch.ops.aten.mul.Tensor(primals_498, 0.9)
    add_324: "f32[512]" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    squeeze_182: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
    mul_423: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0006381620931717);  squeeze_182 = None
    mul_424: "f32[512]" = torch.ops.aten.mul.Tensor(mul_423, 0.1);  mul_423 = None
    mul_425: "f32[512]" = torch.ops.aten.mul.Tensor(primals_499, 0.9)
    add_325: "f32[512]" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
    unsqueeze_240: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_182, -1)
    unsqueeze_241: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_426: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_241);  mul_420 = unsqueeze_241 = None
    unsqueeze_242: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_183, -1);  primals_183 = None
    unsqueeze_243: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_326: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_243);  mul_426 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_327: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_326, relu_54);  add_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_57: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_327);  add_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_7: "f32[8, 1024, 14, 14]" = torch.ops.aten.cat.default([relu_57, relu_54], 1)
    convolution_61: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(cat_7, primals_184, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_328: "i64[]" = torch.ops.aten.add.Tensor(primals_503, 1)
    var_mean_61 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_138: "f32[1, 512, 1, 1]" = var_mean_61[0]
    getitem_139: "f32[1, 512, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_329: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05)
    rsqrt_61: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_329);  add_329 = None
    sub_61: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_139)
    mul_427: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    squeeze_183: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
    squeeze_184: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_428: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_429: "f32[512]" = torch.ops.aten.mul.Tensor(primals_501, 0.9)
    add_330: "f32[512]" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    squeeze_185: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
    mul_430: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0006381620931717);  squeeze_185 = None
    mul_431: "f32[512]" = torch.ops.aten.mul.Tensor(mul_430, 0.1);  mul_430 = None
    mul_432: "f32[512]" = torch.ops.aten.mul.Tensor(primals_502, 0.9)
    add_331: "f32[512]" = torch.ops.aten.add.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
    unsqueeze_244: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_185, -1)
    unsqueeze_245: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_433: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_245);  mul_427 = unsqueeze_245 = None
    unsqueeze_246: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_186, -1);  primals_186 = None
    unsqueeze_247: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_332: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_433, unsqueeze_247);  mul_433 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_333: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_332, relu_57);  add_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_58: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_333);  add_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_62: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_58, primals_187, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_334: "i64[]" = torch.ops.aten.add.Tensor(primals_506, 1)
    var_mean_62 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
    getitem_140: "f32[1, 256, 1, 1]" = var_mean_62[0]
    getitem_141: "f32[1, 256, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_335: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05)
    rsqrt_62: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_335);  add_335 = None
    sub_62: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_141)
    mul_434: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
    squeeze_186: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
    squeeze_187: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
    mul_435: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_436: "f32[256]" = torch.ops.aten.mul.Tensor(primals_504, 0.9)
    add_336: "f32[256]" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
    squeeze_188: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
    mul_437: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0006381620931717);  squeeze_188 = None
    mul_438: "f32[256]" = torch.ops.aten.mul.Tensor(mul_437, 0.1);  mul_437 = None
    mul_439: "f32[256]" = torch.ops.aten.mul.Tensor(primals_505, 0.9)
    add_337: "f32[256]" = torch.ops.aten.add.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_248: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_188, -1)
    unsqueeze_249: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    mul_440: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_249);  mul_434 = unsqueeze_249 = None
    unsqueeze_250: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_189, -1);  primals_189 = None
    unsqueeze_251: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    add_338: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_251);  mul_440 = unsqueeze_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_59: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_338);  add_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_63: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_59, primals_190, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_339: "i64[]" = torch.ops.aten.add.Tensor(primals_509, 1)
    var_mean_63 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_142: "f32[1, 256, 1, 1]" = var_mean_63[0]
    getitem_143: "f32[1, 256, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_340: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05)
    rsqrt_63: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_340);  add_340 = None
    sub_63: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_143)
    mul_441: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
    squeeze_189: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
    squeeze_190: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
    mul_442: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_443: "f32[256]" = torch.ops.aten.mul.Tensor(primals_507, 0.9)
    add_341: "f32[256]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    squeeze_191: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
    mul_444: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0006381620931717);  squeeze_191 = None
    mul_445: "f32[256]" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
    mul_446: "f32[256]" = torch.ops.aten.mul.Tensor(primals_508, 0.9)
    add_342: "f32[256]" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
    unsqueeze_252: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_191, -1)
    unsqueeze_253: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_447: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_253);  mul_441 = unsqueeze_253 = None
    unsqueeze_254: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_192, -1);  primals_192 = None
    unsqueeze_255: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_343: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_255);  mul_447 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_60: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_343);  add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_64: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_60, primals_193, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_344: "i64[]" = torch.ops.aten.add.Tensor(primals_512, 1)
    var_mean_64 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_144: "f32[1, 512, 1, 1]" = var_mean_64[0]
    getitem_145: "f32[1, 512, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_345: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05)
    rsqrt_64: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_345);  add_345 = None
    sub_64: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_145)
    mul_448: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
    squeeze_192: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
    squeeze_193: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
    mul_449: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
    mul_450: "f32[512]" = torch.ops.aten.mul.Tensor(primals_510, 0.9)
    add_346: "f32[512]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    squeeze_194: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
    mul_451: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0006381620931717);  squeeze_194 = None
    mul_452: "f32[512]" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
    mul_453: "f32[512]" = torch.ops.aten.mul.Tensor(primals_511, 0.9)
    add_347: "f32[512]" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
    unsqueeze_256: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_194, -1)
    unsqueeze_257: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    mul_454: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_257);  mul_448 = unsqueeze_257 = None
    unsqueeze_258: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_195, -1);  primals_195 = None
    unsqueeze_259: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    add_348: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_259);  mul_454 = unsqueeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_349: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_348, relu_58);  add_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_61: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_349);  add_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_65: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_61, primals_196, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_350: "i64[]" = torch.ops.aten.add.Tensor(primals_515, 1)
    var_mean_65 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
    getitem_146: "f32[1, 256, 1, 1]" = var_mean_65[0]
    getitem_147: "f32[1, 256, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_351: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05)
    rsqrt_65: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
    sub_65: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_147)
    mul_455: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
    squeeze_195: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
    squeeze_196: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
    mul_456: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
    mul_457: "f32[256]" = torch.ops.aten.mul.Tensor(primals_513, 0.9)
    add_352: "f32[256]" = torch.ops.aten.add.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    squeeze_197: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
    mul_458: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0006381620931717);  squeeze_197 = None
    mul_459: "f32[256]" = torch.ops.aten.mul.Tensor(mul_458, 0.1);  mul_458 = None
    mul_460: "f32[256]" = torch.ops.aten.mul.Tensor(primals_514, 0.9)
    add_353: "f32[256]" = torch.ops.aten.add.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
    unsqueeze_260: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_197, -1)
    unsqueeze_261: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_461: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_455, unsqueeze_261);  mul_455 = unsqueeze_261 = None
    unsqueeze_262: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_198, -1);  primals_198 = None
    unsqueeze_263: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_354: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_263);  mul_461 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_62: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_354);  add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_66: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_62, primals_199, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_355: "i64[]" = torch.ops.aten.add.Tensor(primals_518, 1)
    var_mean_66 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_148: "f32[1, 256, 1, 1]" = var_mean_66[0]
    getitem_149: "f32[1, 256, 1, 1]" = var_mean_66[1];  var_mean_66 = None
    add_356: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05)
    rsqrt_66: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_356);  add_356 = None
    sub_66: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_149)
    mul_462: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
    squeeze_198: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
    squeeze_199: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
    mul_463: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
    mul_464: "f32[256]" = torch.ops.aten.mul.Tensor(primals_516, 0.9)
    add_357: "f32[256]" = torch.ops.aten.add.Tensor(mul_463, mul_464);  mul_463 = mul_464 = None
    squeeze_200: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
    mul_465: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0006381620931717);  squeeze_200 = None
    mul_466: "f32[256]" = torch.ops.aten.mul.Tensor(mul_465, 0.1);  mul_465 = None
    mul_467: "f32[256]" = torch.ops.aten.mul.Tensor(primals_517, 0.9)
    add_358: "f32[256]" = torch.ops.aten.add.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
    unsqueeze_264: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_200, -1)
    unsqueeze_265: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    mul_468: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_265);  mul_462 = unsqueeze_265 = None
    unsqueeze_266: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_201, -1);  primals_201 = None
    unsqueeze_267: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    add_359: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_468, unsqueeze_267);  mul_468 = unsqueeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_63: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_359);  add_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_67: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_63, primals_202, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_360: "i64[]" = torch.ops.aten.add.Tensor(primals_521, 1)
    var_mean_67 = torch.ops.aten.var_mean.correction(convolution_67, [0, 2, 3], correction = 0, keepdim = True)
    getitem_150: "f32[1, 512, 1, 1]" = var_mean_67[0]
    getitem_151: "f32[1, 512, 1, 1]" = var_mean_67[1];  var_mean_67 = None
    add_361: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
    rsqrt_67: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_361);  add_361 = None
    sub_67: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, getitem_151)
    mul_469: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
    squeeze_201: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
    squeeze_202: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
    mul_470: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
    mul_471: "f32[512]" = torch.ops.aten.mul.Tensor(primals_519, 0.9)
    add_362: "f32[512]" = torch.ops.aten.add.Tensor(mul_470, mul_471);  mul_470 = mul_471 = None
    squeeze_203: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
    mul_472: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_203, 1.0006381620931717);  squeeze_203 = None
    mul_473: "f32[512]" = torch.ops.aten.mul.Tensor(mul_472, 0.1);  mul_472 = None
    mul_474: "f32[512]" = torch.ops.aten.mul.Tensor(primals_520, 0.9)
    add_363: "f32[512]" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    unsqueeze_268: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1)
    unsqueeze_269: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_475: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_269);  mul_469 = unsqueeze_269 = None
    unsqueeze_270: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
    unsqueeze_271: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_364: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_475, unsqueeze_271);  mul_475 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_365: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_364, relu_61);  add_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_64: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_365);  add_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_8: "f32[8, 2048, 14, 14]" = torch.ops.aten.cat.default([relu_64, relu_61, relu_51, relu_58], 1)
    convolution_68: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(cat_8, primals_205, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_366: "i64[]" = torch.ops.aten.add.Tensor(primals_524, 1)
    var_mean_68 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True)
    getitem_152: "f32[1, 512, 1, 1]" = var_mean_68[0]
    getitem_153: "f32[1, 512, 1, 1]" = var_mean_68[1];  var_mean_68 = None
    add_367: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05)
    rsqrt_68: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_367);  add_367 = None
    sub_68: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_153)
    mul_476: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
    squeeze_204: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
    squeeze_205: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
    mul_477: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1)
    mul_478: "f32[512]" = torch.ops.aten.mul.Tensor(primals_522, 0.9)
    add_368: "f32[512]" = torch.ops.aten.add.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
    squeeze_206: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
    mul_479: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_206, 1.0006381620931717);  squeeze_206 = None
    mul_480: "f32[512]" = torch.ops.aten.mul.Tensor(mul_479, 0.1);  mul_479 = None
    mul_481: "f32[512]" = torch.ops.aten.mul.Tensor(primals_523, 0.9)
    add_369: "f32[512]" = torch.ops.aten.add.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
    unsqueeze_272: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_206, -1)
    unsqueeze_273: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    mul_482: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_273);  mul_476 = unsqueeze_273 = None
    unsqueeze_274: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_207, -1);  primals_207 = None
    unsqueeze_275: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    add_370: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_275);  mul_482 = unsqueeze_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_371: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_370, relu_64);  add_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_65: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_371);  add_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_69: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_65, primals_208, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_372: "i64[]" = torch.ops.aten.add.Tensor(primals_527, 1)
    var_mean_69 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
    getitem_154: "f32[1, 256, 1, 1]" = var_mean_69[0]
    getitem_155: "f32[1, 256, 1, 1]" = var_mean_69[1];  var_mean_69 = None
    add_373: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05)
    rsqrt_69: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_373);  add_373 = None
    sub_69: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_155)
    mul_483: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
    squeeze_207: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
    squeeze_208: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
    mul_484: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1)
    mul_485: "f32[256]" = torch.ops.aten.mul.Tensor(primals_525, 0.9)
    add_374: "f32[256]" = torch.ops.aten.add.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
    squeeze_209: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
    mul_486: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_209, 1.0006381620931717);  squeeze_209 = None
    mul_487: "f32[256]" = torch.ops.aten.mul.Tensor(mul_486, 0.1);  mul_486 = None
    mul_488: "f32[256]" = torch.ops.aten.mul.Tensor(primals_526, 0.9)
    add_375: "f32[256]" = torch.ops.aten.add.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
    unsqueeze_276: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_209, -1)
    unsqueeze_277: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_489: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_483, unsqueeze_277);  mul_483 = unsqueeze_277 = None
    unsqueeze_278: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_210, -1);  primals_210 = None
    unsqueeze_279: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_376: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_489, unsqueeze_279);  mul_489 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_66: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_376);  add_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_70: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_66, primals_211, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_377: "i64[]" = torch.ops.aten.add.Tensor(primals_530, 1)
    var_mean_70 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
    getitem_156: "f32[1, 256, 1, 1]" = var_mean_70[0]
    getitem_157: "f32[1, 256, 1, 1]" = var_mean_70[1];  var_mean_70 = None
    add_378: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
    rsqrt_70: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_378);  add_378 = None
    sub_70: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_157)
    mul_490: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
    squeeze_210: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
    squeeze_211: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
    mul_491: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
    mul_492: "f32[256]" = torch.ops.aten.mul.Tensor(primals_528, 0.9)
    add_379: "f32[256]" = torch.ops.aten.add.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
    squeeze_212: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
    mul_493: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_212, 1.0006381620931717);  squeeze_212 = None
    mul_494: "f32[256]" = torch.ops.aten.mul.Tensor(mul_493, 0.1);  mul_493 = None
    mul_495: "f32[256]" = torch.ops.aten.mul.Tensor(primals_529, 0.9)
    add_380: "f32[256]" = torch.ops.aten.add.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
    unsqueeze_280: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_212, -1)
    unsqueeze_281: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    mul_496: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_281);  mul_490 = unsqueeze_281 = None
    unsqueeze_282: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_213, -1);  primals_213 = None
    unsqueeze_283: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    add_381: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_496, unsqueeze_283);  mul_496 = unsqueeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_67: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_381);  add_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_71: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_67, primals_214, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_382: "i64[]" = torch.ops.aten.add.Tensor(primals_533, 1)
    var_mean_71 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
    getitem_158: "f32[1, 512, 1, 1]" = var_mean_71[0]
    getitem_159: "f32[1, 512, 1, 1]" = var_mean_71[1];  var_mean_71 = None
    add_383: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
    rsqrt_71: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_383);  add_383 = None
    sub_71: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_159)
    mul_497: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
    squeeze_213: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
    squeeze_214: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
    mul_498: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1)
    mul_499: "f32[512]" = torch.ops.aten.mul.Tensor(primals_531, 0.9)
    add_384: "f32[512]" = torch.ops.aten.add.Tensor(mul_498, mul_499);  mul_498 = mul_499 = None
    squeeze_215: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
    mul_500: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_215, 1.0006381620931717);  squeeze_215 = None
    mul_501: "f32[512]" = torch.ops.aten.mul.Tensor(mul_500, 0.1);  mul_500 = None
    mul_502: "f32[512]" = torch.ops.aten.mul.Tensor(primals_532, 0.9)
    add_385: "f32[512]" = torch.ops.aten.add.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
    unsqueeze_284: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_215, -1)
    unsqueeze_285: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_503: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_497, unsqueeze_285);  mul_497 = unsqueeze_285 = None
    unsqueeze_286: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_216, -1);  primals_216 = None
    unsqueeze_287: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_386: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_287);  mul_503 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_387: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_386, relu_65);  add_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_68: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_387);  add_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_72: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_68, primals_217, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_388: "i64[]" = torch.ops.aten.add.Tensor(primals_536, 1)
    var_mean_72 = torch.ops.aten.var_mean.correction(convolution_72, [0, 2, 3], correction = 0, keepdim = True)
    getitem_160: "f32[1, 256, 1, 1]" = var_mean_72[0]
    getitem_161: "f32[1, 256, 1, 1]" = var_mean_72[1];  var_mean_72 = None
    add_389: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05)
    rsqrt_72: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_389);  add_389 = None
    sub_72: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, getitem_161)
    mul_504: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
    squeeze_216: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3]);  getitem_161 = None
    squeeze_217: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
    mul_505: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1)
    mul_506: "f32[256]" = torch.ops.aten.mul.Tensor(primals_534, 0.9)
    add_390: "f32[256]" = torch.ops.aten.add.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    squeeze_218: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_160, [0, 2, 3]);  getitem_160 = None
    mul_507: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_218, 1.0006381620931717);  squeeze_218 = None
    mul_508: "f32[256]" = torch.ops.aten.mul.Tensor(mul_507, 0.1);  mul_507 = None
    mul_509: "f32[256]" = torch.ops.aten.mul.Tensor(primals_535, 0.9)
    add_391: "f32[256]" = torch.ops.aten.add.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
    unsqueeze_288: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_218, -1)
    unsqueeze_289: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    mul_510: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_504, unsqueeze_289);  mul_504 = unsqueeze_289 = None
    unsqueeze_290: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_219, -1);  primals_219 = None
    unsqueeze_291: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    add_392: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_510, unsqueeze_291);  mul_510 = unsqueeze_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_69: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_392);  add_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_73: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_69, primals_220, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_393: "i64[]" = torch.ops.aten.add.Tensor(primals_539, 1)
    var_mean_73 = torch.ops.aten.var_mean.correction(convolution_73, [0, 2, 3], correction = 0, keepdim = True)
    getitem_162: "f32[1, 256, 1, 1]" = var_mean_73[0]
    getitem_163: "f32[1, 256, 1, 1]" = var_mean_73[1];  var_mean_73 = None
    add_394: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05)
    rsqrt_73: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_394);  add_394 = None
    sub_73: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, getitem_163)
    mul_511: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
    squeeze_219: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
    squeeze_220: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
    mul_512: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
    mul_513: "f32[256]" = torch.ops.aten.mul.Tensor(primals_537, 0.9)
    add_395: "f32[256]" = torch.ops.aten.add.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
    squeeze_221: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
    mul_514: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_221, 1.0006381620931717);  squeeze_221 = None
    mul_515: "f32[256]" = torch.ops.aten.mul.Tensor(mul_514, 0.1);  mul_514 = None
    mul_516: "f32[256]" = torch.ops.aten.mul.Tensor(primals_538, 0.9)
    add_396: "f32[256]" = torch.ops.aten.add.Tensor(mul_515, mul_516);  mul_515 = mul_516 = None
    unsqueeze_292: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_221, -1)
    unsqueeze_293: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_517: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_511, unsqueeze_293);  mul_511 = unsqueeze_293 = None
    unsqueeze_294: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_222, -1);  primals_222 = None
    unsqueeze_295: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_397: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_517, unsqueeze_295);  mul_517 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_70: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_397);  add_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_74: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_70, primals_223, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_398: "i64[]" = torch.ops.aten.add.Tensor(primals_542, 1)
    var_mean_74 = torch.ops.aten.var_mean.correction(convolution_74, [0, 2, 3], correction = 0, keepdim = True)
    getitem_164: "f32[1, 512, 1, 1]" = var_mean_74[0]
    getitem_165: "f32[1, 512, 1, 1]" = var_mean_74[1];  var_mean_74 = None
    add_399: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-05)
    rsqrt_74: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_399);  add_399 = None
    sub_74: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, getitem_165)
    mul_518: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
    squeeze_222: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
    squeeze_223: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
    mul_519: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
    mul_520: "f32[512]" = torch.ops.aten.mul.Tensor(primals_540, 0.9)
    add_400: "f32[512]" = torch.ops.aten.add.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
    squeeze_224: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
    mul_521: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_224, 1.0006381620931717);  squeeze_224 = None
    mul_522: "f32[512]" = torch.ops.aten.mul.Tensor(mul_521, 0.1);  mul_521 = None
    mul_523: "f32[512]" = torch.ops.aten.mul.Tensor(primals_541, 0.9)
    add_401: "f32[512]" = torch.ops.aten.add.Tensor(mul_522, mul_523);  mul_522 = mul_523 = None
    unsqueeze_296: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_224, -1)
    unsqueeze_297: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    mul_524: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_518, unsqueeze_297);  mul_518 = unsqueeze_297 = None
    unsqueeze_298: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_225, -1);  primals_225 = None
    unsqueeze_299: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    add_402: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_524, unsqueeze_299);  mul_524 = unsqueeze_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_403: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_402, relu_68);  add_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_71: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_403);  add_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_9: "f32[8, 1024, 14, 14]" = torch.ops.aten.cat.default([relu_71, relu_68], 1)
    convolution_75: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(cat_9, primals_226, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_404: "i64[]" = torch.ops.aten.add.Tensor(primals_545, 1)
    var_mean_75 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
    getitem_166: "f32[1, 512, 1, 1]" = var_mean_75[0]
    getitem_167: "f32[1, 512, 1, 1]" = var_mean_75[1];  var_mean_75 = None
    add_405: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05)
    rsqrt_75: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_405);  add_405 = None
    sub_75: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_167)
    mul_525: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
    squeeze_225: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
    squeeze_226: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
    mul_526: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1)
    mul_527: "f32[512]" = torch.ops.aten.mul.Tensor(primals_543, 0.9)
    add_406: "f32[512]" = torch.ops.aten.add.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
    squeeze_227: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
    mul_528: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_227, 1.0006381620931717);  squeeze_227 = None
    mul_529: "f32[512]" = torch.ops.aten.mul.Tensor(mul_528, 0.1);  mul_528 = None
    mul_530: "f32[512]" = torch.ops.aten.mul.Tensor(primals_544, 0.9)
    add_407: "f32[512]" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    unsqueeze_300: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_227, -1)
    unsqueeze_301: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_531: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_301);  mul_525 = unsqueeze_301 = None
    unsqueeze_302: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1);  primals_228 = None
    unsqueeze_303: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_408: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_531, unsqueeze_303);  mul_531 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_409: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_408, relu_71);  add_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_72: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_409);  add_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_76: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_72, primals_229, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_410: "i64[]" = torch.ops.aten.add.Tensor(primals_548, 1)
    var_mean_76 = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
    getitem_168: "f32[1, 256, 1, 1]" = var_mean_76[0]
    getitem_169: "f32[1, 256, 1, 1]" = var_mean_76[1];  var_mean_76 = None
    add_411: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-05)
    rsqrt_76: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_411);  add_411 = None
    sub_76: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_169)
    mul_532: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
    squeeze_228: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
    squeeze_229: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
    mul_533: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1)
    mul_534: "f32[256]" = torch.ops.aten.mul.Tensor(primals_546, 0.9)
    add_412: "f32[256]" = torch.ops.aten.add.Tensor(mul_533, mul_534);  mul_533 = mul_534 = None
    squeeze_230: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
    mul_535: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_230, 1.0006381620931717);  squeeze_230 = None
    mul_536: "f32[256]" = torch.ops.aten.mul.Tensor(mul_535, 0.1);  mul_535 = None
    mul_537: "f32[256]" = torch.ops.aten.mul.Tensor(primals_547, 0.9)
    add_413: "f32[256]" = torch.ops.aten.add.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
    unsqueeze_304: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_230, -1)
    unsqueeze_305: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    mul_538: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_305);  mul_532 = unsqueeze_305 = None
    unsqueeze_306: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_231, -1);  primals_231 = None
    unsqueeze_307: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    add_414: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_538, unsqueeze_307);  mul_538 = unsqueeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_73: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_414);  add_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_77: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_73, primals_232, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_415: "i64[]" = torch.ops.aten.add.Tensor(primals_551, 1)
    var_mean_77 = torch.ops.aten.var_mean.correction(convolution_77, [0, 2, 3], correction = 0, keepdim = True)
    getitem_170: "f32[1, 256, 1, 1]" = var_mean_77[0]
    getitem_171: "f32[1, 256, 1, 1]" = var_mean_77[1];  var_mean_77 = None
    add_416: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-05)
    rsqrt_77: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_416);  add_416 = None
    sub_77: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, getitem_171)
    mul_539: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
    squeeze_231: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
    squeeze_232: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
    mul_540: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
    mul_541: "f32[256]" = torch.ops.aten.mul.Tensor(primals_549, 0.9)
    add_417: "f32[256]" = torch.ops.aten.add.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
    squeeze_233: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_170, [0, 2, 3]);  getitem_170 = None
    mul_542: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_233, 1.0006381620931717);  squeeze_233 = None
    mul_543: "f32[256]" = torch.ops.aten.mul.Tensor(mul_542, 0.1);  mul_542 = None
    mul_544: "f32[256]" = torch.ops.aten.mul.Tensor(primals_550, 0.9)
    add_418: "f32[256]" = torch.ops.aten.add.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
    unsqueeze_308: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_233, -1)
    unsqueeze_309: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_545: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_539, unsqueeze_309);  mul_539 = unsqueeze_309 = None
    unsqueeze_310: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_234, -1);  primals_234 = None
    unsqueeze_311: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_419: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_311);  mul_545 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_74: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_419);  add_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_78: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_74, primals_235, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_420: "i64[]" = torch.ops.aten.add.Tensor(primals_554, 1)
    var_mean_78 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
    getitem_172: "f32[1, 512, 1, 1]" = var_mean_78[0]
    getitem_173: "f32[1, 512, 1, 1]" = var_mean_78[1];  var_mean_78 = None
    add_421: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05)
    rsqrt_78: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_421);  add_421 = None
    sub_78: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_173)
    mul_546: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
    squeeze_234: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3]);  getitem_173 = None
    squeeze_235: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
    mul_547: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1)
    mul_548: "f32[512]" = torch.ops.aten.mul.Tensor(primals_552, 0.9)
    add_422: "f32[512]" = torch.ops.aten.add.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
    squeeze_236: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_172, [0, 2, 3]);  getitem_172 = None
    mul_549: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_236, 1.0006381620931717);  squeeze_236 = None
    mul_550: "f32[512]" = torch.ops.aten.mul.Tensor(mul_549, 0.1);  mul_549 = None
    mul_551: "f32[512]" = torch.ops.aten.mul.Tensor(primals_553, 0.9)
    add_423: "f32[512]" = torch.ops.aten.add.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
    unsqueeze_312: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_236, -1)
    unsqueeze_313: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    mul_552: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_546, unsqueeze_313);  mul_546 = unsqueeze_313 = None
    unsqueeze_314: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_237, -1);  primals_237 = None
    unsqueeze_315: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    add_424: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_552, unsqueeze_315);  mul_552 = unsqueeze_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_425: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_424, relu_72);  add_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_75: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_425);  add_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_79: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_75, primals_238, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_426: "i64[]" = torch.ops.aten.add.Tensor(primals_557, 1)
    var_mean_79 = torch.ops.aten.var_mean.correction(convolution_79, [0, 2, 3], correction = 0, keepdim = True)
    getitem_174: "f32[1, 256, 1, 1]" = var_mean_79[0]
    getitem_175: "f32[1, 256, 1, 1]" = var_mean_79[1];  var_mean_79 = None
    add_427: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-05)
    rsqrt_79: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_427);  add_427 = None
    sub_79: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, getitem_175)
    mul_553: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
    squeeze_237: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
    squeeze_238: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
    mul_554: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1)
    mul_555: "f32[256]" = torch.ops.aten.mul.Tensor(primals_555, 0.9)
    add_428: "f32[256]" = torch.ops.aten.add.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
    squeeze_239: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_174, [0, 2, 3]);  getitem_174 = None
    mul_556: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_239, 1.0006381620931717);  squeeze_239 = None
    mul_557: "f32[256]" = torch.ops.aten.mul.Tensor(mul_556, 0.1);  mul_556 = None
    mul_558: "f32[256]" = torch.ops.aten.mul.Tensor(primals_556, 0.9)
    add_429: "f32[256]" = torch.ops.aten.add.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
    unsqueeze_316: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_239, -1)
    unsqueeze_317: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_559: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_317);  mul_553 = unsqueeze_317 = None
    unsqueeze_318: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_240, -1);  primals_240 = None
    unsqueeze_319: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_430: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_559, unsqueeze_319);  mul_559 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_76: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_430);  add_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_80: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_76, primals_241, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_431: "i64[]" = torch.ops.aten.add.Tensor(primals_560, 1)
    var_mean_80 = torch.ops.aten.var_mean.correction(convolution_80, [0, 2, 3], correction = 0, keepdim = True)
    getitem_176: "f32[1, 256, 1, 1]" = var_mean_80[0]
    getitem_177: "f32[1, 256, 1, 1]" = var_mean_80[1];  var_mean_80 = None
    add_432: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05)
    rsqrt_80: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_432);  add_432 = None
    sub_80: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_177)
    mul_560: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = None
    squeeze_240: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_177, [0, 2, 3]);  getitem_177 = None
    squeeze_241: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
    mul_561: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_240, 0.1)
    mul_562: "f32[256]" = torch.ops.aten.mul.Tensor(primals_558, 0.9)
    add_433: "f32[256]" = torch.ops.aten.add.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
    squeeze_242: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_176, [0, 2, 3]);  getitem_176 = None
    mul_563: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_242, 1.0006381620931717);  squeeze_242 = None
    mul_564: "f32[256]" = torch.ops.aten.mul.Tensor(mul_563, 0.1);  mul_563 = None
    mul_565: "f32[256]" = torch.ops.aten.mul.Tensor(primals_559, 0.9)
    add_434: "f32[256]" = torch.ops.aten.add.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_320: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_242, -1)
    unsqueeze_321: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    mul_566: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_321);  mul_560 = unsqueeze_321 = None
    unsqueeze_322: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_243, -1);  primals_243 = None
    unsqueeze_323: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    add_435: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_323);  mul_566 = unsqueeze_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_77: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_435);  add_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_81: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_77, primals_244, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_436: "i64[]" = torch.ops.aten.add.Tensor(primals_563, 1)
    var_mean_81 = torch.ops.aten.var_mean.correction(convolution_81, [0, 2, 3], correction = 0, keepdim = True)
    getitem_178: "f32[1, 512, 1, 1]" = var_mean_81[0]
    getitem_179: "f32[1, 512, 1, 1]" = var_mean_81[1];  var_mean_81 = None
    add_437: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05)
    rsqrt_81: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_437);  add_437 = None
    sub_81: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, getitem_179)
    mul_567: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = None
    squeeze_243: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
    squeeze_244: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
    mul_568: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_243, 0.1)
    mul_569: "f32[512]" = torch.ops.aten.mul.Tensor(primals_561, 0.9)
    add_438: "f32[512]" = torch.ops.aten.add.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
    squeeze_245: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_178, [0, 2, 3]);  getitem_178 = None
    mul_570: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_245, 1.0006381620931717);  squeeze_245 = None
    mul_571: "f32[512]" = torch.ops.aten.mul.Tensor(mul_570, 0.1);  mul_570 = None
    mul_572: "f32[512]" = torch.ops.aten.mul.Tensor(primals_562, 0.9)
    add_439: "f32[512]" = torch.ops.aten.add.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
    unsqueeze_324: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_245, -1)
    unsqueeze_325: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_573: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_567, unsqueeze_325);  mul_567 = unsqueeze_325 = None
    unsqueeze_326: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_246, -1);  primals_246 = None
    unsqueeze_327: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_440: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_573, unsqueeze_327);  mul_573 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_441: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_440, relu_75);  add_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_78: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_441);  add_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_10: "f32[8, 1536, 14, 14]" = torch.ops.aten.cat.default([relu_78, relu_75, relu_72], 1)
    convolution_82: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(cat_10, primals_247, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_442: "i64[]" = torch.ops.aten.add.Tensor(primals_566, 1)
    var_mean_82 = torch.ops.aten.var_mean.correction(convolution_82, [0, 2, 3], correction = 0, keepdim = True)
    getitem_180: "f32[1, 512, 1, 1]" = var_mean_82[0]
    getitem_181: "f32[1, 512, 1, 1]" = var_mean_82[1];  var_mean_82 = None
    add_443: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-05)
    rsqrt_82: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_443);  add_443 = None
    sub_82: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, getitem_181)
    mul_574: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
    squeeze_246: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_181, [0, 2, 3]);  getitem_181 = None
    squeeze_247: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
    mul_575: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_246, 0.1)
    mul_576: "f32[512]" = torch.ops.aten.mul.Tensor(primals_564, 0.9)
    add_444: "f32[512]" = torch.ops.aten.add.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
    squeeze_248: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_180, [0, 2, 3]);  getitem_180 = None
    mul_577: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_248, 1.0006381620931717);  squeeze_248 = None
    mul_578: "f32[512]" = torch.ops.aten.mul.Tensor(mul_577, 0.1);  mul_577 = None
    mul_579: "f32[512]" = torch.ops.aten.mul.Tensor(primals_565, 0.9)
    add_445: "f32[512]" = torch.ops.aten.add.Tensor(mul_578, mul_579);  mul_578 = mul_579 = None
    unsqueeze_328: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_248, -1)
    unsqueeze_329: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    mul_580: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_329);  mul_574 = unsqueeze_329 = None
    unsqueeze_330: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_249, -1);  primals_249 = None
    unsqueeze_331: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    add_446: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_580, unsqueeze_331);  mul_580 = unsqueeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_447: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_446, relu_78);  add_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_79: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_447);  add_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_83: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_79, primals_250, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_448: "i64[]" = torch.ops.aten.add.Tensor(primals_569, 1)
    var_mean_83 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
    getitem_182: "f32[1, 256, 1, 1]" = var_mean_83[0]
    getitem_183: "f32[1, 256, 1, 1]" = var_mean_83[1];  var_mean_83 = None
    add_449: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-05)
    rsqrt_83: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_449);  add_449 = None
    sub_83: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_183)
    mul_581: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
    squeeze_249: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3]);  getitem_183 = None
    squeeze_250: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
    mul_582: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_249, 0.1)
    mul_583: "f32[256]" = torch.ops.aten.mul.Tensor(primals_567, 0.9)
    add_450: "f32[256]" = torch.ops.aten.add.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    squeeze_251: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_182, [0, 2, 3]);  getitem_182 = None
    mul_584: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_251, 1.0006381620931717);  squeeze_251 = None
    mul_585: "f32[256]" = torch.ops.aten.mul.Tensor(mul_584, 0.1);  mul_584 = None
    mul_586: "f32[256]" = torch.ops.aten.mul.Tensor(primals_568, 0.9)
    add_451: "f32[256]" = torch.ops.aten.add.Tensor(mul_585, mul_586);  mul_585 = mul_586 = None
    unsqueeze_332: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_251, -1)
    unsqueeze_333: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_587: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_581, unsqueeze_333);  mul_581 = unsqueeze_333 = None
    unsqueeze_334: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_252, -1);  primals_252 = None
    unsqueeze_335: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_452: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_335);  mul_587 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_80: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_452);  add_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_84: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_80, primals_253, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_453: "i64[]" = torch.ops.aten.add.Tensor(primals_572, 1)
    var_mean_84 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
    getitem_184: "f32[1, 256, 1, 1]" = var_mean_84[0]
    getitem_185: "f32[1, 256, 1, 1]" = var_mean_84[1];  var_mean_84 = None
    add_454: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-05)
    rsqrt_84: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_454);  add_454 = None
    sub_84: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_185)
    mul_588: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
    squeeze_252: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_185, [0, 2, 3]);  getitem_185 = None
    squeeze_253: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
    mul_589: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_252, 0.1)
    mul_590: "f32[256]" = torch.ops.aten.mul.Tensor(primals_570, 0.9)
    add_455: "f32[256]" = torch.ops.aten.add.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
    squeeze_254: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_184, [0, 2, 3]);  getitem_184 = None
    mul_591: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_254, 1.0006381620931717);  squeeze_254 = None
    mul_592: "f32[256]" = torch.ops.aten.mul.Tensor(mul_591, 0.1);  mul_591 = None
    mul_593: "f32[256]" = torch.ops.aten.mul.Tensor(primals_571, 0.9)
    add_456: "f32[256]" = torch.ops.aten.add.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
    unsqueeze_336: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_254, -1)
    unsqueeze_337: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    mul_594: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_588, unsqueeze_337);  mul_588 = unsqueeze_337 = None
    unsqueeze_338: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_255, -1);  primals_255 = None
    unsqueeze_339: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    add_457: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_594, unsqueeze_339);  mul_594 = unsqueeze_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_81: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_457);  add_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_85: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_81, primals_256, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_458: "i64[]" = torch.ops.aten.add.Tensor(primals_575, 1)
    var_mean_85 = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True)
    getitem_186: "f32[1, 512, 1, 1]" = var_mean_85[0]
    getitem_187: "f32[1, 512, 1, 1]" = var_mean_85[1];  var_mean_85 = None
    add_459: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05)
    rsqrt_85: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_459);  add_459 = None
    sub_85: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_187)
    mul_595: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = None
    squeeze_255: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
    squeeze_256: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
    mul_596: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_255, 0.1)
    mul_597: "f32[512]" = torch.ops.aten.mul.Tensor(primals_573, 0.9)
    add_460: "f32[512]" = torch.ops.aten.add.Tensor(mul_596, mul_597);  mul_596 = mul_597 = None
    squeeze_257: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_186, [0, 2, 3]);  getitem_186 = None
    mul_598: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_257, 1.0006381620931717);  squeeze_257 = None
    mul_599: "f32[512]" = torch.ops.aten.mul.Tensor(mul_598, 0.1);  mul_598 = None
    mul_600: "f32[512]" = torch.ops.aten.mul.Tensor(primals_574, 0.9)
    add_461: "f32[512]" = torch.ops.aten.add.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
    unsqueeze_340: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_257, -1)
    unsqueeze_341: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_601: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_341);  mul_595 = unsqueeze_341 = None
    unsqueeze_342: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_258, -1);  primals_258 = None
    unsqueeze_343: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_462: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_601, unsqueeze_343);  mul_601 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_463: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_462, relu_79);  add_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_82: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_463);  add_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_86: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_82, primals_259, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_464: "i64[]" = torch.ops.aten.add.Tensor(primals_578, 1)
    var_mean_86 = torch.ops.aten.var_mean.correction(convolution_86, [0, 2, 3], correction = 0, keepdim = True)
    getitem_188: "f32[1, 256, 1, 1]" = var_mean_86[0]
    getitem_189: "f32[1, 256, 1, 1]" = var_mean_86[1];  var_mean_86 = None
    add_465: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-05)
    rsqrt_86: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_465);  add_465 = None
    sub_86: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, getitem_189)
    mul_602: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = None
    squeeze_258: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
    squeeze_259: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_86, [0, 2, 3]);  rsqrt_86 = None
    mul_603: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_258, 0.1)
    mul_604: "f32[256]" = torch.ops.aten.mul.Tensor(primals_576, 0.9)
    add_466: "f32[256]" = torch.ops.aten.add.Tensor(mul_603, mul_604);  mul_603 = mul_604 = None
    squeeze_260: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_188, [0, 2, 3]);  getitem_188 = None
    mul_605: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_260, 1.0006381620931717);  squeeze_260 = None
    mul_606: "f32[256]" = torch.ops.aten.mul.Tensor(mul_605, 0.1);  mul_605 = None
    mul_607: "f32[256]" = torch.ops.aten.mul.Tensor(primals_577, 0.9)
    add_467: "f32[256]" = torch.ops.aten.add.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
    unsqueeze_344: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_260, -1)
    unsqueeze_345: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    mul_608: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_602, unsqueeze_345);  mul_602 = unsqueeze_345 = None
    unsqueeze_346: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_261, -1);  primals_261 = None
    unsqueeze_347: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    add_468: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_608, unsqueeze_347);  mul_608 = unsqueeze_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_83: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_468);  add_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_87: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_83, primals_262, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_469: "i64[]" = torch.ops.aten.add.Tensor(primals_581, 1)
    var_mean_87 = torch.ops.aten.var_mean.correction(convolution_87, [0, 2, 3], correction = 0, keepdim = True)
    getitem_190: "f32[1, 256, 1, 1]" = var_mean_87[0]
    getitem_191: "f32[1, 256, 1, 1]" = var_mean_87[1];  var_mean_87 = None
    add_470: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-05)
    rsqrt_87: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_470);  add_470 = None
    sub_87: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, getitem_191)
    mul_609: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = None
    squeeze_261: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_191, [0, 2, 3]);  getitem_191 = None
    squeeze_262: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
    mul_610: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_261, 0.1)
    mul_611: "f32[256]" = torch.ops.aten.mul.Tensor(primals_579, 0.9)
    add_471: "f32[256]" = torch.ops.aten.add.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
    squeeze_263: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_190, [0, 2, 3]);  getitem_190 = None
    mul_612: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_263, 1.0006381620931717);  squeeze_263 = None
    mul_613: "f32[256]" = torch.ops.aten.mul.Tensor(mul_612, 0.1);  mul_612 = None
    mul_614: "f32[256]" = torch.ops.aten.mul.Tensor(primals_580, 0.9)
    add_472: "f32[256]" = torch.ops.aten.add.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
    unsqueeze_348: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_263, -1)
    unsqueeze_349: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_615: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_609, unsqueeze_349);  mul_609 = unsqueeze_349 = None
    unsqueeze_350: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_264, -1);  primals_264 = None
    unsqueeze_351: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_473: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_615, unsqueeze_351);  mul_615 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_84: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_473);  add_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_88: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_84, primals_265, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_474: "i64[]" = torch.ops.aten.add.Tensor(primals_584, 1)
    var_mean_88 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
    getitem_192: "f32[1, 512, 1, 1]" = var_mean_88[0]
    getitem_193: "f32[1, 512, 1, 1]" = var_mean_88[1];  var_mean_88 = None
    add_475: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-05)
    rsqrt_88: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_475);  add_475 = None
    sub_88: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_193)
    mul_616: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = None
    squeeze_264: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
    squeeze_265: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
    mul_617: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_264, 0.1)
    mul_618: "f32[512]" = torch.ops.aten.mul.Tensor(primals_582, 0.9)
    add_476: "f32[512]" = torch.ops.aten.add.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
    squeeze_266: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_192, [0, 2, 3]);  getitem_192 = None
    mul_619: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_266, 1.0006381620931717);  squeeze_266 = None
    mul_620: "f32[512]" = torch.ops.aten.mul.Tensor(mul_619, 0.1);  mul_619 = None
    mul_621: "f32[512]" = torch.ops.aten.mul.Tensor(primals_583, 0.9)
    add_477: "f32[512]" = torch.ops.aten.add.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
    unsqueeze_352: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_266, -1)
    unsqueeze_353: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    mul_622: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_353);  mul_616 = unsqueeze_353 = None
    unsqueeze_354: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_267, -1);  primals_267 = None
    unsqueeze_355: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    add_478: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_622, unsqueeze_355);  mul_622 = unsqueeze_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_479: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_478, relu_82);  add_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_85: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_479);  add_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_11: "f32[8, 1024, 14, 14]" = torch.ops.aten.cat.default([relu_85, relu_82], 1)
    convolution_89: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(cat_11, primals_268, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_480: "i64[]" = torch.ops.aten.add.Tensor(primals_587, 1)
    var_mean_89 = torch.ops.aten.var_mean.correction(convolution_89, [0, 2, 3], correction = 0, keepdim = True)
    getitem_194: "f32[1, 512, 1, 1]" = var_mean_89[0]
    getitem_195: "f32[1, 512, 1, 1]" = var_mean_89[1];  var_mean_89 = None
    add_481: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_194, 1e-05)
    rsqrt_89: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_481);  add_481 = None
    sub_89: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, getitem_195)
    mul_623: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = None
    squeeze_267: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3]);  getitem_195 = None
    squeeze_268: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_89, [0, 2, 3]);  rsqrt_89 = None
    mul_624: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_267, 0.1)
    mul_625: "f32[512]" = torch.ops.aten.mul.Tensor(primals_585, 0.9)
    add_482: "f32[512]" = torch.ops.aten.add.Tensor(mul_624, mul_625);  mul_624 = mul_625 = None
    squeeze_269: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_194, [0, 2, 3]);  getitem_194 = None
    mul_626: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_269, 1.0006381620931717);  squeeze_269 = None
    mul_627: "f32[512]" = torch.ops.aten.mul.Tensor(mul_626, 0.1);  mul_626 = None
    mul_628: "f32[512]" = torch.ops.aten.mul.Tensor(primals_586, 0.9)
    add_483: "f32[512]" = torch.ops.aten.add.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
    unsqueeze_356: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_269, -1)
    unsqueeze_357: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_629: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_623, unsqueeze_357);  mul_623 = unsqueeze_357 = None
    unsqueeze_358: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_270, -1);  primals_270 = None
    unsqueeze_359: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_484: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_629, unsqueeze_359);  mul_629 = unsqueeze_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_485: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_484, relu_85);  add_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_86: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_485);  add_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_90: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_86, primals_271, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_486: "i64[]" = torch.ops.aten.add.Tensor(primals_590, 1)
    var_mean_90 = torch.ops.aten.var_mean.correction(convolution_90, [0, 2, 3], correction = 0, keepdim = True)
    getitem_196: "f32[1, 256, 1, 1]" = var_mean_90[0]
    getitem_197: "f32[1, 256, 1, 1]" = var_mean_90[1];  var_mean_90 = None
    add_487: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-05)
    rsqrt_90: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_487);  add_487 = None
    sub_90: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, getitem_197)
    mul_630: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = None
    squeeze_270: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_197, [0, 2, 3]);  getitem_197 = None
    squeeze_271: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_90, [0, 2, 3]);  rsqrt_90 = None
    mul_631: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_270, 0.1)
    mul_632: "f32[256]" = torch.ops.aten.mul.Tensor(primals_588, 0.9)
    add_488: "f32[256]" = torch.ops.aten.add.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
    squeeze_272: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_196, [0, 2, 3]);  getitem_196 = None
    mul_633: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_272, 1.0006381620931717);  squeeze_272 = None
    mul_634: "f32[256]" = torch.ops.aten.mul.Tensor(mul_633, 0.1);  mul_633 = None
    mul_635: "f32[256]" = torch.ops.aten.mul.Tensor(primals_589, 0.9)
    add_489: "f32[256]" = torch.ops.aten.add.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
    unsqueeze_360: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_272, -1)
    unsqueeze_361: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    mul_636: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_630, unsqueeze_361);  mul_630 = unsqueeze_361 = None
    unsqueeze_362: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_273, -1);  primals_273 = None
    unsqueeze_363: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    add_490: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_636, unsqueeze_363);  mul_636 = unsqueeze_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_87: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_490);  add_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_91: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_87, primals_274, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_491: "i64[]" = torch.ops.aten.add.Tensor(primals_593, 1)
    var_mean_91 = torch.ops.aten.var_mean.correction(convolution_91, [0, 2, 3], correction = 0, keepdim = True)
    getitem_198: "f32[1, 256, 1, 1]" = var_mean_91[0]
    getitem_199: "f32[1, 256, 1, 1]" = var_mean_91[1];  var_mean_91 = None
    add_492: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05)
    rsqrt_91: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_492);  add_492 = None
    sub_91: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, getitem_199)
    mul_637: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = None
    squeeze_273: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_199, [0, 2, 3]);  getitem_199 = None
    squeeze_274: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_91, [0, 2, 3]);  rsqrt_91 = None
    mul_638: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_273, 0.1)
    mul_639: "f32[256]" = torch.ops.aten.mul.Tensor(primals_591, 0.9)
    add_493: "f32[256]" = torch.ops.aten.add.Tensor(mul_638, mul_639);  mul_638 = mul_639 = None
    squeeze_275: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_198, [0, 2, 3]);  getitem_198 = None
    mul_640: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_275, 1.0006381620931717);  squeeze_275 = None
    mul_641: "f32[256]" = torch.ops.aten.mul.Tensor(mul_640, 0.1);  mul_640 = None
    mul_642: "f32[256]" = torch.ops.aten.mul.Tensor(primals_592, 0.9)
    add_494: "f32[256]" = torch.ops.aten.add.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
    unsqueeze_364: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_275, -1)
    unsqueeze_365: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_643: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_365);  mul_637 = unsqueeze_365 = None
    unsqueeze_366: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_276, -1);  primals_276 = None
    unsqueeze_367: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_495: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_643, unsqueeze_367);  mul_643 = unsqueeze_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_88: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_495);  add_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_92: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_88, primals_277, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_496: "i64[]" = torch.ops.aten.add.Tensor(primals_596, 1)
    var_mean_92 = torch.ops.aten.var_mean.correction(convolution_92, [0, 2, 3], correction = 0, keepdim = True)
    getitem_200: "f32[1, 512, 1, 1]" = var_mean_92[0]
    getitem_201: "f32[1, 512, 1, 1]" = var_mean_92[1];  var_mean_92 = None
    add_497: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-05)
    rsqrt_92: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_497);  add_497 = None
    sub_92: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_201)
    mul_644: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = None
    squeeze_276: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_201, [0, 2, 3]);  getitem_201 = None
    squeeze_277: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_92, [0, 2, 3]);  rsqrt_92 = None
    mul_645: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_276, 0.1)
    mul_646: "f32[512]" = torch.ops.aten.mul.Tensor(primals_594, 0.9)
    add_498: "f32[512]" = torch.ops.aten.add.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
    squeeze_278: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_200, [0, 2, 3]);  getitem_200 = None
    mul_647: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_278, 1.0006381620931717);  squeeze_278 = None
    mul_648: "f32[512]" = torch.ops.aten.mul.Tensor(mul_647, 0.1);  mul_647 = None
    mul_649: "f32[512]" = torch.ops.aten.mul.Tensor(primals_595, 0.9)
    add_499: "f32[512]" = torch.ops.aten.add.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
    unsqueeze_368: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_278, -1)
    unsqueeze_369: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    mul_650: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_644, unsqueeze_369);  mul_644 = unsqueeze_369 = None
    unsqueeze_370: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_279, -1);  primals_279 = None
    unsqueeze_371: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    add_500: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_650, unsqueeze_371);  mul_650 = unsqueeze_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_501: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_500, relu_86);  add_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_89: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_501);  add_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_93: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_89, primals_280, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_502: "i64[]" = torch.ops.aten.add.Tensor(primals_599, 1)
    var_mean_93 = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True)
    getitem_202: "f32[1, 256, 1, 1]" = var_mean_93[0]
    getitem_203: "f32[1, 256, 1, 1]" = var_mean_93[1];  var_mean_93 = None
    add_503: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_202, 1e-05)
    rsqrt_93: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_503);  add_503 = None
    sub_93: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_203)
    mul_651: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = None
    squeeze_279: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_203, [0, 2, 3]);  getitem_203 = None
    squeeze_280: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
    mul_652: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_279, 0.1)
    mul_653: "f32[256]" = torch.ops.aten.mul.Tensor(primals_597, 0.9)
    add_504: "f32[256]" = torch.ops.aten.add.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
    squeeze_281: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_202, [0, 2, 3]);  getitem_202 = None
    mul_654: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_281, 1.0006381620931717);  squeeze_281 = None
    mul_655: "f32[256]" = torch.ops.aten.mul.Tensor(mul_654, 0.1);  mul_654 = None
    mul_656: "f32[256]" = torch.ops.aten.mul.Tensor(primals_598, 0.9)
    add_505: "f32[256]" = torch.ops.aten.add.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
    unsqueeze_372: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_281, -1)
    unsqueeze_373: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_657: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_651, unsqueeze_373);  mul_651 = unsqueeze_373 = None
    unsqueeze_374: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_282, -1);  primals_282 = None
    unsqueeze_375: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_506: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_657, unsqueeze_375);  mul_657 = unsqueeze_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_90: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_506);  add_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_94: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_90, primals_283, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_507: "i64[]" = torch.ops.aten.add.Tensor(primals_602, 1)
    var_mean_94 = torch.ops.aten.var_mean.correction(convolution_94, [0, 2, 3], correction = 0, keepdim = True)
    getitem_204: "f32[1, 256, 1, 1]" = var_mean_94[0]
    getitem_205: "f32[1, 256, 1, 1]" = var_mean_94[1];  var_mean_94 = None
    add_508: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem_204, 1e-05)
    rsqrt_94: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_508);  add_508 = None
    sub_94: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, getitem_205)
    mul_658: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = None
    squeeze_282: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_205, [0, 2, 3]);  getitem_205 = None
    squeeze_283: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_94, [0, 2, 3]);  rsqrt_94 = None
    mul_659: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_282, 0.1)
    mul_660: "f32[256]" = torch.ops.aten.mul.Tensor(primals_600, 0.9)
    add_509: "f32[256]" = torch.ops.aten.add.Tensor(mul_659, mul_660);  mul_659 = mul_660 = None
    squeeze_284: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_204, [0, 2, 3]);  getitem_204 = None
    mul_661: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_284, 1.0006381620931717);  squeeze_284 = None
    mul_662: "f32[256]" = torch.ops.aten.mul.Tensor(mul_661, 0.1);  mul_661 = None
    mul_663: "f32[256]" = torch.ops.aten.mul.Tensor(primals_601, 0.9)
    add_510: "f32[256]" = torch.ops.aten.add.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
    unsqueeze_376: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_284, -1)
    unsqueeze_377: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    mul_664: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_377);  mul_658 = unsqueeze_377 = None
    unsqueeze_378: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_285, -1);  primals_285 = None
    unsqueeze_379: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    add_511: "f32[8, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_664, unsqueeze_379);  mul_664 = unsqueeze_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_91: "f32[8, 256, 14, 14]" = torch.ops.aten.relu.default(add_511);  add_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_95: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_91, primals_286, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_512: "i64[]" = torch.ops.aten.add.Tensor(primals_605, 1)
    var_mean_95 = torch.ops.aten.var_mean.correction(convolution_95, [0, 2, 3], correction = 0, keepdim = True)
    getitem_206: "f32[1, 512, 1, 1]" = var_mean_95[0]
    getitem_207: "f32[1, 512, 1, 1]" = var_mean_95[1];  var_mean_95 = None
    add_513: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-05)
    rsqrt_95: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_513);  add_513 = None
    sub_95: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, getitem_207)
    mul_665: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = None
    squeeze_285: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_207, [0, 2, 3]);  getitem_207 = None
    squeeze_286: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_95, [0, 2, 3]);  rsqrt_95 = None
    mul_666: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_285, 0.1)
    mul_667: "f32[512]" = torch.ops.aten.mul.Tensor(primals_603, 0.9)
    add_514: "f32[512]" = torch.ops.aten.add.Tensor(mul_666, mul_667);  mul_666 = mul_667 = None
    squeeze_287: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_206, [0, 2, 3]);  getitem_206 = None
    mul_668: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_287, 1.0006381620931717);  squeeze_287 = None
    mul_669: "f32[512]" = torch.ops.aten.mul.Tensor(mul_668, 0.1);  mul_668 = None
    mul_670: "f32[512]" = torch.ops.aten.mul.Tensor(primals_604, 0.9)
    add_515: "f32[512]" = torch.ops.aten.add.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    unsqueeze_380: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_287, -1)
    unsqueeze_381: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_671: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_665, unsqueeze_381);  mul_665 = unsqueeze_381 = None
    unsqueeze_382: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_288, -1);  primals_288 = None
    unsqueeze_383: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_516: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_671, unsqueeze_383);  mul_671 = unsqueeze_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_517: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_516, relu_89);  add_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_92: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_517);  add_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_12: "f32[8, 2816, 14, 14]" = torch.ops.aten.cat.default([relu_92, relu_89, getitem_88, relu_65, relu_79, relu_86], 1);  getitem_88 = None
    convolution_96: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(cat_12, primals_289, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_518: "i64[]" = torch.ops.aten.add.Tensor(primals_608, 1)
    var_mean_96 = torch.ops.aten.var_mean.correction(convolution_96, [0, 2, 3], correction = 0, keepdim = True)
    getitem_208: "f32[1, 512, 1, 1]" = var_mean_96[0]
    getitem_209: "f32[1, 512, 1, 1]" = var_mean_96[1];  var_mean_96 = None
    add_519: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-05)
    rsqrt_96: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_519);  add_519 = None
    sub_96: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_96, getitem_209)
    mul_672: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = None
    squeeze_288: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_209, [0, 2, 3]);  getitem_209 = None
    squeeze_289: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_96, [0, 2, 3]);  rsqrt_96 = None
    mul_673: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_288, 0.1)
    mul_674: "f32[512]" = torch.ops.aten.mul.Tensor(primals_606, 0.9)
    add_520: "f32[512]" = torch.ops.aten.add.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
    squeeze_290: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_208, [0, 2, 3]);  getitem_208 = None
    mul_675: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_290, 1.0006381620931717);  squeeze_290 = None
    mul_676: "f32[512]" = torch.ops.aten.mul.Tensor(mul_675, 0.1);  mul_675 = None
    mul_677: "f32[512]" = torch.ops.aten.mul.Tensor(primals_607, 0.9)
    add_521: "f32[512]" = torch.ops.aten.add.Tensor(mul_676, mul_677);  mul_676 = mul_677 = None
    unsqueeze_384: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_290, -1)
    unsqueeze_385: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    mul_678: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_672, unsqueeze_385);  mul_672 = unsqueeze_385 = None
    unsqueeze_386: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_291, -1);  primals_291 = None
    unsqueeze_387: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    add_522: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_678, unsqueeze_387);  mul_678 = unsqueeze_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_523: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_522, relu_92);  add_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_93: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_523);  add_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_8 = torch.ops.aten.max_pool2d_with_indices.default(relu_93, [2, 2], [2, 2])
    getitem_210: "f32[8, 512, 7, 7]" = max_pool2d_with_indices_8[0]
    getitem_211: "i64[8, 512, 7, 7]" = max_pool2d_with_indices_8[1];  max_pool2d_with_indices_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    convolution_97: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(getitem_210, primals_292, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    add_524: "i64[]" = torch.ops.aten.add.Tensor(primals_611, 1)
    var_mean_97 = torch.ops.aten.var_mean.correction(convolution_97, [0, 2, 3], correction = 0, keepdim = True)
    getitem_212: "f32[1, 1024, 1, 1]" = var_mean_97[0]
    getitem_213: "f32[1, 1024, 1, 1]" = var_mean_97[1];  var_mean_97 = None
    add_525: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_212, 1e-05)
    rsqrt_97: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_525);  add_525 = None
    sub_97: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_97, getitem_213)
    mul_679: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = None
    squeeze_291: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_213, [0, 2, 3]);  getitem_213 = None
    squeeze_292: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_97, [0, 2, 3]);  rsqrt_97 = None
    mul_680: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_291, 0.1)
    mul_681: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_609, 0.9)
    add_526: "f32[1024]" = torch.ops.aten.add.Tensor(mul_680, mul_681);  mul_680 = mul_681 = None
    squeeze_293: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_212, [0, 2, 3]);  getitem_212 = None
    mul_682: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_293, 1.0025575447570332);  squeeze_293 = None
    mul_683: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_682, 0.1);  mul_682 = None
    mul_684: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_610, 0.9)
    add_527: "f32[1024]" = torch.ops.aten.add.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
    unsqueeze_388: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_293, -1)
    unsqueeze_389: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_685: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_679, unsqueeze_389);  mul_679 = unsqueeze_389 = None
    unsqueeze_390: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_294, -1);  primals_294 = None
    unsqueeze_391: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_528: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_685, unsqueeze_391);  mul_685 = unsqueeze_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_98: "f32[8, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_93, primals_295, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_529: "i64[]" = torch.ops.aten.add.Tensor(primals_614, 1)
    var_mean_98 = torch.ops.aten.var_mean.correction(convolution_98, [0, 2, 3], correction = 0, keepdim = True)
    getitem_214: "f32[1, 512, 1, 1]" = var_mean_98[0]
    getitem_215: "f32[1, 512, 1, 1]" = var_mean_98[1];  var_mean_98 = None
    add_530: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_214, 1e-05)
    rsqrt_98: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_530);  add_530 = None
    sub_98: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, getitem_215)
    mul_686: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = None
    squeeze_294: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_215, [0, 2, 3]);  getitem_215 = None
    squeeze_295: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_98, [0, 2, 3]);  rsqrt_98 = None
    mul_687: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_294, 0.1)
    mul_688: "f32[512]" = torch.ops.aten.mul.Tensor(primals_612, 0.9)
    add_531: "f32[512]" = torch.ops.aten.add.Tensor(mul_687, mul_688);  mul_687 = mul_688 = None
    squeeze_296: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_214, [0, 2, 3]);  getitem_214 = None
    mul_689: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_296, 1.0006381620931717);  squeeze_296 = None
    mul_690: "f32[512]" = torch.ops.aten.mul.Tensor(mul_689, 0.1);  mul_689 = None
    mul_691: "f32[512]" = torch.ops.aten.mul.Tensor(primals_613, 0.9)
    add_532: "f32[512]" = torch.ops.aten.add.Tensor(mul_690, mul_691);  mul_690 = mul_691 = None
    unsqueeze_392: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_296, -1)
    unsqueeze_393: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    mul_692: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_686, unsqueeze_393);  mul_686 = unsqueeze_393 = None
    unsqueeze_394: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_297, -1);  primals_297 = None
    unsqueeze_395: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    add_533: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_692, unsqueeze_395);  mul_692 = unsqueeze_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_94: "f32[8, 512, 14, 14]" = torch.ops.aten.relu.default(add_533);  add_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_99: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_94, primals_298, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_534: "i64[]" = torch.ops.aten.add.Tensor(primals_617, 1)
    var_mean_99 = torch.ops.aten.var_mean.correction(convolution_99, [0, 2, 3], correction = 0, keepdim = True)
    getitem_216: "f32[1, 512, 1, 1]" = var_mean_99[0]
    getitem_217: "f32[1, 512, 1, 1]" = var_mean_99[1];  var_mean_99 = None
    add_535: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_216, 1e-05)
    rsqrt_99: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_535);  add_535 = None
    sub_99: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, getitem_217)
    mul_693: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = None
    squeeze_297: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_217, [0, 2, 3]);  getitem_217 = None
    squeeze_298: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_99, [0, 2, 3]);  rsqrt_99 = None
    mul_694: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_297, 0.1)
    mul_695: "f32[512]" = torch.ops.aten.mul.Tensor(primals_615, 0.9)
    add_536: "f32[512]" = torch.ops.aten.add.Tensor(mul_694, mul_695);  mul_694 = mul_695 = None
    squeeze_299: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_216, [0, 2, 3]);  getitem_216 = None
    mul_696: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_299, 1.0025575447570332);  squeeze_299 = None
    mul_697: "f32[512]" = torch.ops.aten.mul.Tensor(mul_696, 0.1);  mul_696 = None
    mul_698: "f32[512]" = torch.ops.aten.mul.Tensor(primals_616, 0.9)
    add_537: "f32[512]" = torch.ops.aten.add.Tensor(mul_697, mul_698);  mul_697 = mul_698 = None
    unsqueeze_396: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_299, -1)
    unsqueeze_397: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_699: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_693, unsqueeze_397);  mul_693 = unsqueeze_397 = None
    unsqueeze_398: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_300, -1);  primals_300 = None
    unsqueeze_399: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_538: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_699, unsqueeze_399);  mul_699 = unsqueeze_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_95: "f32[8, 512, 7, 7]" = torch.ops.aten.relu.default(add_538);  add_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_100: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(relu_95, primals_301, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_539: "i64[]" = torch.ops.aten.add.Tensor(primals_620, 1)
    var_mean_100 = torch.ops.aten.var_mean.correction(convolution_100, [0, 2, 3], correction = 0, keepdim = True)
    getitem_218: "f32[1, 1024, 1, 1]" = var_mean_100[0]
    getitem_219: "f32[1, 1024, 1, 1]" = var_mean_100[1];  var_mean_100 = None
    add_540: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-05)
    rsqrt_100: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_540);  add_540 = None
    sub_100: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_100, getitem_219)
    mul_700: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = None
    squeeze_300: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_219, [0, 2, 3]);  getitem_219 = None
    squeeze_301: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_100, [0, 2, 3]);  rsqrt_100 = None
    mul_701: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_300, 0.1)
    mul_702: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_618, 0.9)
    add_541: "f32[1024]" = torch.ops.aten.add.Tensor(mul_701, mul_702);  mul_701 = mul_702 = None
    squeeze_302: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_218, [0, 2, 3]);  getitem_218 = None
    mul_703: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_302, 1.0025575447570332);  squeeze_302 = None
    mul_704: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_703, 0.1);  mul_703 = None
    mul_705: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_619, 0.9)
    add_542: "f32[1024]" = torch.ops.aten.add.Tensor(mul_704, mul_705);  mul_704 = mul_705 = None
    unsqueeze_400: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_302, -1)
    unsqueeze_401: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    mul_706: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_700, unsqueeze_401);  mul_700 = unsqueeze_401 = None
    unsqueeze_402: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_303, -1);  primals_303 = None
    unsqueeze_403: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    add_543: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_706, unsqueeze_403);  mul_706 = unsqueeze_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_544: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(add_543, add_528);  add_543 = add_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_96: "f32[8, 1024, 7, 7]" = torch.ops.aten.relu.default(add_544);  add_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_101: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_96, primals_304, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    add_545: "i64[]" = torch.ops.aten.add.Tensor(primals_623, 1)
    var_mean_101 = torch.ops.aten.var_mean.correction(convolution_101, [0, 2, 3], correction = 0, keepdim = True)
    getitem_220: "f32[1, 512, 1, 1]" = var_mean_101[0]
    getitem_221: "f32[1, 512, 1, 1]" = var_mean_101[1];  var_mean_101 = None
    add_546: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-05)
    rsqrt_101: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_546);  add_546 = None
    sub_101: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_101, getitem_221)
    mul_707: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = None
    squeeze_303: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_221, [0, 2, 3]);  getitem_221 = None
    squeeze_304: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_101, [0, 2, 3]);  rsqrt_101 = None
    mul_708: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_303, 0.1)
    mul_709: "f32[512]" = torch.ops.aten.mul.Tensor(primals_621, 0.9)
    add_547: "f32[512]" = torch.ops.aten.add.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
    squeeze_305: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_220, [0, 2, 3]);  getitem_220 = None
    mul_710: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_305, 1.0025575447570332);  squeeze_305 = None
    mul_711: "f32[512]" = torch.ops.aten.mul.Tensor(mul_710, 0.1);  mul_710 = None
    mul_712: "f32[512]" = torch.ops.aten.mul.Tensor(primals_622, 0.9)
    add_548: "f32[512]" = torch.ops.aten.add.Tensor(mul_711, mul_712);  mul_711 = mul_712 = None
    unsqueeze_404: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_305, -1)
    unsqueeze_405: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_713: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_707, unsqueeze_405);  mul_707 = unsqueeze_405 = None
    unsqueeze_406: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_306, -1);  primals_306 = None
    unsqueeze_407: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_549: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_713, unsqueeze_407);  mul_713 = unsqueeze_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    relu_97: "f32[8, 512, 7, 7]" = torch.ops.aten.relu.default(add_549);  add_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_102: "f32[8, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_97, primals_307, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    add_550: "i64[]" = torch.ops.aten.add.Tensor(primals_626, 1)
    var_mean_102 = torch.ops.aten.var_mean.correction(convolution_102, [0, 2, 3], correction = 0, keepdim = True)
    getitem_222: "f32[1, 512, 1, 1]" = var_mean_102[0]
    getitem_223: "f32[1, 512, 1, 1]" = var_mean_102[1];  var_mean_102 = None
    add_551: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-05)
    rsqrt_102: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_551);  add_551 = None
    sub_102: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, getitem_223)
    mul_714: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = None
    squeeze_306: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_223, [0, 2, 3]);  getitem_223 = None
    squeeze_307: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_102, [0, 2, 3]);  rsqrt_102 = None
    mul_715: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_306, 0.1)
    mul_716: "f32[512]" = torch.ops.aten.mul.Tensor(primals_624, 0.9)
    add_552: "f32[512]" = torch.ops.aten.add.Tensor(mul_715, mul_716);  mul_715 = mul_716 = None
    squeeze_308: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_222, [0, 2, 3]);  getitem_222 = None
    mul_717: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_308, 1.0025575447570332);  squeeze_308 = None
    mul_718: "f32[512]" = torch.ops.aten.mul.Tensor(mul_717, 0.1);  mul_717 = None
    mul_719: "f32[512]" = torch.ops.aten.mul.Tensor(primals_625, 0.9)
    add_553: "f32[512]" = torch.ops.aten.add.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
    unsqueeze_408: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_308, -1)
    unsqueeze_409: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    mul_720: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_714, unsqueeze_409);  mul_714 = unsqueeze_409 = None
    unsqueeze_410: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_309, -1);  primals_309 = None
    unsqueeze_411: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    add_554: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_720, unsqueeze_411);  mul_720 = unsqueeze_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    relu_98: "f32[8, 512, 7, 7]" = torch.ops.aten.relu.default(add_554);  add_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_103: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(relu_98, primals_310, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    add_555: "i64[]" = torch.ops.aten.add.Tensor(primals_629, 1)
    var_mean_103 = torch.ops.aten.var_mean.correction(convolution_103, [0, 2, 3], correction = 0, keepdim = True)
    getitem_224: "f32[1, 1024, 1, 1]" = var_mean_103[0]
    getitem_225: "f32[1, 1024, 1, 1]" = var_mean_103[1];  var_mean_103 = None
    add_556: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-05)
    rsqrt_103: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_556);  add_556 = None
    sub_103: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_103, getitem_225)
    mul_721: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = None
    squeeze_309: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_225, [0, 2, 3]);  getitem_225 = None
    squeeze_310: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_103, [0, 2, 3]);  rsqrt_103 = None
    mul_722: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_309, 0.1)
    mul_723: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_627, 0.9)
    add_557: "f32[1024]" = torch.ops.aten.add.Tensor(mul_722, mul_723);  mul_722 = mul_723 = None
    squeeze_311: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_224, [0, 2, 3]);  getitem_224 = None
    mul_724: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_311, 1.0025575447570332);  squeeze_311 = None
    mul_725: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_724, 0.1);  mul_724 = None
    mul_726: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_628, 0.9)
    add_558: "f32[1024]" = torch.ops.aten.add.Tensor(mul_725, mul_726);  mul_725 = mul_726 = None
    unsqueeze_412: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_311, -1)
    unsqueeze_413: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_727: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_721, unsqueeze_413);  mul_721 = unsqueeze_413 = None
    unsqueeze_414: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_312, -1);  primals_312 = None
    unsqueeze_415: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_559: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_727, unsqueeze_415);  mul_727 = unsqueeze_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_560: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(add_559, relu_96);  add_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    relu_99: "f32[8, 1024, 7, 7]" = torch.ops.aten.relu.default(add_560);  add_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    cat_13: "f32[8, 2560, 7, 7]" = torch.ops.aten.cat.default([relu_99, relu_96, getitem_210], 1)
    convolution_104: "f32[8, 1024, 7, 7]" = torch.ops.aten.convolution.default(cat_13, primals_313, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    add_561: "i64[]" = torch.ops.aten.add.Tensor(primals_632, 1)
    var_mean_104 = torch.ops.aten.var_mean.correction(convolution_104, [0, 2, 3], correction = 0, keepdim = True)
    getitem_226: "f32[1, 1024, 1, 1]" = var_mean_104[0]
    getitem_227: "f32[1, 1024, 1, 1]" = var_mean_104[1];  var_mean_104 = None
    add_562: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-05)
    rsqrt_104: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_562);  add_562 = None
    sub_104: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_104, getitem_227)
    mul_728: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_104);  sub_104 = None
    squeeze_312: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_227, [0, 2, 3]);  getitem_227 = None
    squeeze_313: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_104, [0, 2, 3]);  rsqrt_104 = None
    mul_729: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_312, 0.1)
    mul_730: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_630, 0.9)
    add_563: "f32[1024]" = torch.ops.aten.add.Tensor(mul_729, mul_730);  mul_729 = mul_730 = None
    squeeze_314: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_226, [0, 2, 3]);  getitem_226 = None
    mul_731: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_314, 1.0025575447570332);  squeeze_314 = None
    mul_732: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_731, 0.1);  mul_731 = None
    mul_733: "f32[1024]" = torch.ops.aten.mul.Tensor(primals_631, 0.9)
    add_564: "f32[1024]" = torch.ops.aten.add.Tensor(mul_732, mul_733);  mul_732 = mul_733 = None
    unsqueeze_416: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_314, -1)
    unsqueeze_417: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    mul_734: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_728, unsqueeze_417);  mul_728 = unsqueeze_417 = None
    unsqueeze_418: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_315, -1);  primals_315 = None
    unsqueeze_419: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    add_565: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_734, unsqueeze_419);  mul_734 = unsqueeze_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:179, code: x += x_children[0]
    add_566: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(add_565, relu_99);  add_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    relu_100: "f32[8, 1024, 7, 7]" = torch.ops.aten.relu.default(add_566);  add_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean: "f32[8, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu_100, [-1, -2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:374, code: x = self.head_drop(x)
    clone: "f32[8, 1024, 1, 1]" = torch.ops.aten.clone.default(mean);  mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:377, code: x = self.fc(x)
    convolution_105: "f32[8, 1000, 1, 1]" = torch.ops.aten.convolution.default(clone, primals_316, primals_317, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:378, code: return self.flatten(x)
    view: "f32[8, 1000]" = torch.ops.aten.view.default(convolution_105, [8, 1000]);  convolution_105 = None
    view_1: "f32[8, 1000, 1, 1]" = torch.ops.aten.view.default(tangents_1, [8, 1000, 1, 1]);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:377, code: x = self.fc(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(view_1, clone, primals_316, [1000], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_1 = clone = primals_316 = None
    getitem_228: "f32[8, 1024, 1, 1]" = convolution_backward[0]
    getitem_229: "f32[1000, 1024, 1, 1]" = convolution_backward[1]
    getitem_230: "f32[1000]" = convolution_backward[2];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1024, 7, 7]" = torch.ops.aten.expand.default(getitem_228, [8, 1024, 7, 7]);  getitem_228 = None
    div: "f32[8, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_102: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(relu_100);  relu_100 = None
    alias_103: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    le: "b8[8, 1024, 7, 7]" = torch.ops.aten.le.Scalar(alias_103, 0);  alias_103 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le, scalar_tensor, div);  le = scalar_tensor = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_420: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_312, 0);  squeeze_312 = None
    unsqueeze_421: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
    unsqueeze_422: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
    sum_1: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_105: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_422)
    mul_735: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_105);  sub_105 = None
    sum_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_735, [0, 2, 3]);  mul_735 = None
    mul_736: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_1, 0.002551020408163265)
    unsqueeze_423: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
    unsqueeze_424: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    unsqueeze_425: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
    mul_737: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    mul_738: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_313, squeeze_313)
    mul_739: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_737, mul_738);  mul_737 = mul_738 = None
    unsqueeze_426: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    unsqueeze_427: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
    unsqueeze_428: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
    mul_740: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_313, primals_314);  primals_314 = None
    unsqueeze_429: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    unsqueeze_430: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    sub_106: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_422);  convolution_104 = unsqueeze_422 = None
    mul_741: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_428);  sub_106 = unsqueeze_428 = None
    sub_107: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_741);  mul_741 = None
    sub_108: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_425);  sub_107 = unsqueeze_425 = None
    mul_742: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_431);  sub_108 = unsqueeze_431 = None
    mul_743: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_2, squeeze_313);  sum_2 = squeeze_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_742, cat_13, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_742 = cat_13 = primals_313 = None
    getitem_231: "f32[8, 2560, 7, 7]" = convolution_backward_1[0]
    getitem_232: "f32[1024, 2560, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    slice_1: "f32[8, 1024, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_231, 1, 0, 1024)
    slice_2: "f32[8, 1024, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_231, 1, 1024, 2048)
    slice_3: "f32[8, 512, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_231, 1, 2048, 2560);  getitem_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_567: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(where, slice_1);  where = slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_105: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(relu_99);  relu_99 = None
    alias_106: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(alias_105);  alias_105 = None
    le_1: "b8[8, 1024, 7, 7]" = torch.ops.aten.le.Scalar(alias_106, 0);  alias_106 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le_1, scalar_tensor_1, add_567);  le_1 = scalar_tensor_1 = add_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_568: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(slice_2, where_1);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_432: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_309, 0);  squeeze_309 = None
    unsqueeze_433: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
    unsqueeze_434: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
    sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_109: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_434)
    mul_744: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_109);  sub_109 = None
    sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_744, [0, 2, 3]);  mul_744 = None
    mul_745: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    unsqueeze_435: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_745, 0);  mul_745 = None
    unsqueeze_436: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
    unsqueeze_437: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 3);  unsqueeze_436 = None
    mul_746: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    mul_747: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_310, squeeze_310)
    mul_748: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_746, mul_747);  mul_746 = mul_747 = None
    unsqueeze_438: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
    unsqueeze_439: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
    unsqueeze_440: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
    mul_749: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_310, primals_311);  primals_311 = None
    unsqueeze_441: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
    unsqueeze_442: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
    unsqueeze_443: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
    sub_110: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_434);  convolution_103 = unsqueeze_434 = None
    mul_750: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_440);  sub_110 = unsqueeze_440 = None
    sub_111: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_750);  where_1 = mul_750 = None
    sub_112: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_437);  sub_111 = unsqueeze_437 = None
    mul_751: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_443);  sub_112 = unsqueeze_443 = None
    mul_752: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_4, squeeze_310);  sum_4 = squeeze_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_751, relu_98, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_751 = primals_310 = None
    getitem_234: "f32[8, 512, 7, 7]" = convolution_backward_2[0]
    getitem_235: "f32[1024, 512, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_108: "f32[8, 512, 7, 7]" = torch.ops.aten.alias.default(relu_98);  relu_98 = None
    alias_109: "f32[8, 512, 7, 7]" = torch.ops.aten.alias.default(alias_108);  alias_108 = None
    le_2: "b8[8, 512, 7, 7]" = torch.ops.aten.le.Scalar(alias_109, 0);  alias_109 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[8, 512, 7, 7]" = torch.ops.aten.where.self(le_2, scalar_tensor_2, getitem_234);  le_2 = scalar_tensor_2 = getitem_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_444: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_306, 0);  squeeze_306 = None
    unsqueeze_445: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 2);  unsqueeze_444 = None
    unsqueeze_446: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 3);  unsqueeze_445 = None
    sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_113: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_446)
    mul_753: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_113);  sub_113 = None
    sum_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_753, [0, 2, 3]);  mul_753 = None
    mul_754: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    unsqueeze_447: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
    unsqueeze_448: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
    unsqueeze_449: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 3);  unsqueeze_448 = None
    mul_755: "f32[512]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    mul_756: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_307, squeeze_307)
    mul_757: "f32[512]" = torch.ops.aten.mul.Tensor(mul_755, mul_756);  mul_755 = mul_756 = None
    unsqueeze_450: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_757, 0);  mul_757 = None
    unsqueeze_451: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 2);  unsqueeze_450 = None
    unsqueeze_452: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 3);  unsqueeze_451 = None
    mul_758: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_307, primals_308);  primals_308 = None
    unsqueeze_453: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_454: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
    unsqueeze_455: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
    sub_114: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_446);  convolution_102 = unsqueeze_446 = None
    mul_759: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_452);  sub_114 = unsqueeze_452 = None
    sub_115: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_759);  where_2 = mul_759 = None
    sub_116: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_449);  sub_115 = unsqueeze_449 = None
    mul_760: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_455);  sub_116 = unsqueeze_455 = None
    mul_761: "f32[512]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_307);  sum_6 = squeeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_760, relu_97, primals_307, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_760 = primals_307 = None
    getitem_237: "f32[8, 512, 7, 7]" = convolution_backward_3[0]
    getitem_238: "f32[512, 512, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_111: "f32[8, 512, 7, 7]" = torch.ops.aten.alias.default(relu_97);  relu_97 = None
    alias_112: "f32[8, 512, 7, 7]" = torch.ops.aten.alias.default(alias_111);  alias_111 = None
    le_3: "b8[8, 512, 7, 7]" = torch.ops.aten.le.Scalar(alias_112, 0);  alias_112 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[8, 512, 7, 7]" = torch.ops.aten.where.self(le_3, scalar_tensor_3, getitem_237);  le_3 = scalar_tensor_3 = getitem_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_456: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_303, 0);  squeeze_303 = None
    unsqueeze_457: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 2);  unsqueeze_456 = None
    unsqueeze_458: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 3);  unsqueeze_457 = None
    sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_117: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_458)
    mul_762: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_117);  sub_117 = None
    sum_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_762, [0, 2, 3]);  mul_762 = None
    mul_763: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    unsqueeze_459: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
    unsqueeze_460: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
    unsqueeze_461: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 3);  unsqueeze_460 = None
    mul_764: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    mul_765: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_304, squeeze_304)
    mul_766: "f32[512]" = torch.ops.aten.mul.Tensor(mul_764, mul_765);  mul_764 = mul_765 = None
    unsqueeze_462: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
    unsqueeze_463: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 2);  unsqueeze_462 = None
    unsqueeze_464: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 3);  unsqueeze_463 = None
    mul_767: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_304, primals_305);  primals_305 = None
    unsqueeze_465: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
    unsqueeze_466: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
    unsqueeze_467: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
    sub_118: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_458);  convolution_101 = unsqueeze_458 = None
    mul_768: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_464);  sub_118 = unsqueeze_464 = None
    sub_119: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_768);  where_3 = mul_768 = None
    sub_120: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_461);  sub_119 = unsqueeze_461 = None
    mul_769: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_467);  sub_120 = unsqueeze_467 = None
    mul_770: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_304);  sum_8 = squeeze_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_769, relu_96, primals_304, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_769 = primals_304 = None
    getitem_240: "f32[8, 1024, 7, 7]" = convolution_backward_4[0]
    getitem_241: "f32[512, 1024, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_569: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(add_568, getitem_240);  add_568 = getitem_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_114: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(relu_96);  relu_96 = None
    alias_115: "f32[8, 1024, 7, 7]" = torch.ops.aten.alias.default(alias_114);  alias_114 = None
    le_4: "b8[8, 1024, 7, 7]" = torch.ops.aten.le.Scalar(alias_115, 0);  alias_115 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le_4, scalar_tensor_4, add_569);  le_4 = scalar_tensor_4 = add_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_468: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_300, 0);  squeeze_300 = None
    unsqueeze_469: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 2);  unsqueeze_468 = None
    unsqueeze_470: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 3);  unsqueeze_469 = None
    sum_9: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_121: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_470)
    mul_771: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_121);  sub_121 = None
    sum_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3]);  mul_771 = None
    mul_772: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    unsqueeze_471: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
    unsqueeze_472: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
    unsqueeze_473: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 3);  unsqueeze_472 = None
    mul_773: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    mul_774: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_301, squeeze_301)
    mul_775: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_773, mul_774);  mul_773 = mul_774 = None
    unsqueeze_474: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
    unsqueeze_475: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 2);  unsqueeze_474 = None
    unsqueeze_476: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 3);  unsqueeze_475 = None
    mul_776: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_301, primals_302);  primals_302 = None
    unsqueeze_477: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_478: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
    unsqueeze_479: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 3);  unsqueeze_478 = None
    sub_122: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_470);  convolution_100 = unsqueeze_470 = None
    mul_777: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_476);  sub_122 = unsqueeze_476 = None
    sub_123: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_777);  mul_777 = None
    sub_124: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_473);  sub_123 = unsqueeze_473 = None
    mul_778: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_479);  sub_124 = unsqueeze_479 = None
    mul_779: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_301);  sum_10 = squeeze_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_778, relu_95, primals_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_778 = primals_301 = None
    getitem_243: "f32[8, 512, 7, 7]" = convolution_backward_5[0]
    getitem_244: "f32[1024, 512, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_117: "f32[8, 512, 7, 7]" = torch.ops.aten.alias.default(relu_95);  relu_95 = None
    alias_118: "f32[8, 512, 7, 7]" = torch.ops.aten.alias.default(alias_117);  alias_117 = None
    le_5: "b8[8, 512, 7, 7]" = torch.ops.aten.le.Scalar(alias_118, 0);  alias_118 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[8, 512, 7, 7]" = torch.ops.aten.where.self(le_5, scalar_tensor_5, getitem_243);  le_5 = scalar_tensor_5 = getitem_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_480: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_297, 0);  squeeze_297 = None
    unsqueeze_481: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 2);  unsqueeze_480 = None
    unsqueeze_482: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 3);  unsqueeze_481 = None
    sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_125: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_482)
    mul_780: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_125);  sub_125 = None
    sum_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 2, 3]);  mul_780 = None
    mul_781: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    unsqueeze_483: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
    unsqueeze_484: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
    unsqueeze_485: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 3);  unsqueeze_484 = None
    mul_782: "f32[512]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    mul_783: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_298, squeeze_298)
    mul_784: "f32[512]" = torch.ops.aten.mul.Tensor(mul_782, mul_783);  mul_782 = mul_783 = None
    unsqueeze_486: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
    unsqueeze_487: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 2);  unsqueeze_486 = None
    unsqueeze_488: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 3);  unsqueeze_487 = None
    mul_785: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_298, primals_299);  primals_299 = None
    unsqueeze_489: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
    unsqueeze_490: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
    unsqueeze_491: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 3);  unsqueeze_490 = None
    sub_126: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_482);  convolution_99 = unsqueeze_482 = None
    mul_786: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_488);  sub_126 = unsqueeze_488 = None
    sub_127: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_786);  where_5 = mul_786 = None
    sub_128: "f32[8, 512, 7, 7]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_485);  sub_127 = unsqueeze_485 = None
    mul_787: "f32[8, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_491);  sub_128 = unsqueeze_491 = None
    mul_788: "f32[512]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_298);  sum_12 = squeeze_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_787, relu_94, primals_298, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_787 = primals_298 = None
    getitem_246: "f32[8, 512, 14, 14]" = convolution_backward_6[0]
    getitem_247: "f32[512, 512, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_120: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_94);  relu_94 = None
    alias_121: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_120);  alias_120 = None
    le_6: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_121, 0);  alias_121 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_6, scalar_tensor_6, getitem_246);  le_6 = scalar_tensor_6 = getitem_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_492: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_294, 0);  squeeze_294 = None
    unsqueeze_493: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 2);  unsqueeze_492 = None
    unsqueeze_494: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 3);  unsqueeze_493 = None
    sum_13: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_129: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_494)
    mul_789: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_129);  sub_129 = None
    sum_14: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_789, [0, 2, 3]);  mul_789 = None
    mul_790: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, 0.0006377551020408163)
    unsqueeze_495: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
    unsqueeze_496: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
    unsqueeze_497: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 3);  unsqueeze_496 = None
    mul_791: "f32[512]" = torch.ops.aten.mul.Tensor(sum_14, 0.0006377551020408163)
    mul_792: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_295, squeeze_295)
    mul_793: "f32[512]" = torch.ops.aten.mul.Tensor(mul_791, mul_792);  mul_791 = mul_792 = None
    unsqueeze_498: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_793, 0);  mul_793 = None
    unsqueeze_499: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 2);  unsqueeze_498 = None
    unsqueeze_500: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 3);  unsqueeze_499 = None
    mul_794: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_295, primals_296);  primals_296 = None
    unsqueeze_501: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
    unsqueeze_502: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
    unsqueeze_503: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 3);  unsqueeze_502 = None
    sub_130: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_494);  convolution_98 = unsqueeze_494 = None
    mul_795: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_500);  sub_130 = unsqueeze_500 = None
    sub_131: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_6, mul_795);  where_6 = mul_795 = None
    sub_132: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_497);  sub_131 = unsqueeze_497 = None
    mul_796: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_503);  sub_132 = unsqueeze_503 = None
    mul_797: "f32[512]" = torch.ops.aten.mul.Tensor(sum_14, squeeze_295);  sum_14 = squeeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_796, relu_93, primals_295, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_796 = primals_295 = None
    getitem_249: "f32[8, 512, 14, 14]" = convolution_backward_7[0]
    getitem_250: "f32[512, 512, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    unsqueeze_504: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_291, 0);  squeeze_291 = None
    unsqueeze_505: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 2);  unsqueeze_504 = None
    unsqueeze_506: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 3);  unsqueeze_505 = None
    sum_15: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_133: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_506)
    mul_798: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_133);  sub_133 = None
    sum_16: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_798, [0, 2, 3]);  mul_798 = None
    mul_799: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    unsqueeze_507: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
    unsqueeze_508: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
    unsqueeze_509: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 3);  unsqueeze_508 = None
    mul_800: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    mul_801: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_292, squeeze_292)
    mul_802: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_800, mul_801);  mul_800 = mul_801 = None
    unsqueeze_510: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
    unsqueeze_511: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 2);  unsqueeze_510 = None
    unsqueeze_512: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 3);  unsqueeze_511 = None
    mul_803: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_292, primals_293);  primals_293 = None
    unsqueeze_513: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    unsqueeze_514: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
    unsqueeze_515: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
    sub_134: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_506);  convolution_97 = unsqueeze_506 = None
    mul_804: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_512);  sub_134 = unsqueeze_512 = None
    sub_135: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_804);  where_4 = mul_804 = None
    sub_136: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_509);  sub_135 = unsqueeze_509 = None
    mul_805: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_515);  sub_136 = unsqueeze_515 = None
    mul_806: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_16, squeeze_292);  sum_16 = squeeze_292 = None
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_805, getitem_210, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_805 = getitem_210 = primals_292 = None
    getitem_252: "f32[8, 512, 7, 7]" = convolution_backward_8[0]
    getitem_253: "f32[1024, 512, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    add_570: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(slice_3, getitem_252);  slice_3 = getitem_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_backward: "f32[8, 512, 14, 14]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_570, relu_93, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_211);  add_570 = getitem_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    add_571: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(getitem_249, max_pool2d_with_indices_backward);  getitem_249 = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_123: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_93);  relu_93 = None
    alias_124: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_123);  alias_123 = None
    le_7: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_124, 0);  alias_124 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_7: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_7, scalar_tensor_7, add_571);  le_7 = scalar_tensor_7 = add_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_516: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_288, 0);  squeeze_288 = None
    unsqueeze_517: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 2);  unsqueeze_516 = None
    unsqueeze_518: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 3);  unsqueeze_517 = None
    sum_17: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_137: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_518)
    mul_807: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_137);  sub_137 = None
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_807, [0, 2, 3]);  mul_807 = None
    mul_808: "f32[512]" = torch.ops.aten.mul.Tensor(sum_17, 0.0006377551020408163)
    unsqueeze_519: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
    unsqueeze_520: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 2);  unsqueeze_519 = None
    unsqueeze_521: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 3);  unsqueeze_520 = None
    mul_809: "f32[512]" = torch.ops.aten.mul.Tensor(sum_18, 0.0006377551020408163)
    mul_810: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_289, squeeze_289)
    mul_811: "f32[512]" = torch.ops.aten.mul.Tensor(mul_809, mul_810);  mul_809 = mul_810 = None
    unsqueeze_522: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_811, 0);  mul_811 = None
    unsqueeze_523: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 2);  unsqueeze_522 = None
    unsqueeze_524: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 3);  unsqueeze_523 = None
    mul_812: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_289, primals_290);  primals_290 = None
    unsqueeze_525: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    unsqueeze_526: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 2);  unsqueeze_525 = None
    unsqueeze_527: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 3);  unsqueeze_526 = None
    sub_138: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_518);  convolution_96 = unsqueeze_518 = None
    mul_813: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_524);  sub_138 = unsqueeze_524 = None
    sub_139: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_7, mul_813);  mul_813 = None
    sub_140: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_521);  sub_139 = unsqueeze_521 = None
    mul_814: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_527);  sub_140 = unsqueeze_527 = None
    mul_815: "f32[512]" = torch.ops.aten.mul.Tensor(sum_18, squeeze_289);  sum_18 = squeeze_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_814, cat_12, primals_289, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_814 = cat_12 = primals_289 = None
    getitem_255: "f32[8, 2816, 14, 14]" = convolution_backward_9[0]
    getitem_256: "f32[512, 2816, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    slice_4: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_255, 1, 0, 512)
    slice_5: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_255, 1, 512, 1024)
    slice_6: "f32[8, 256, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_255, 1, 1024, 1280)
    slice_7: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_255, 1, 1280, 1792)
    slice_8: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_255, 1, 1792, 2304)
    slice_9: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_255, 1, 2304, 2816);  getitem_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_572: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_7, slice_4);  where_7 = slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_126: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_92);  relu_92 = None
    alias_127: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_126);  alias_126 = None
    le_8: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_127, 0);  alias_127 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_8: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_8, scalar_tensor_8, add_572);  le_8 = scalar_tensor_8 = add_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_573: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_5, where_8);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_528: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_285, 0);  squeeze_285 = None
    unsqueeze_529: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 2);  unsqueeze_528 = None
    unsqueeze_530: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 3);  unsqueeze_529 = None
    sum_19: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_141: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_530)
    mul_816: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_141);  sub_141 = None
    sum_20: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_816, [0, 2, 3]);  mul_816 = None
    mul_817: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006377551020408163)
    unsqueeze_531: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_817, 0);  mul_817 = None
    unsqueeze_532: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 2);  unsqueeze_531 = None
    unsqueeze_533: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 3);  unsqueeze_532 = None
    mul_818: "f32[512]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
    mul_819: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
    mul_820: "f32[512]" = torch.ops.aten.mul.Tensor(mul_818, mul_819);  mul_818 = mul_819 = None
    unsqueeze_534: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_820, 0);  mul_820 = None
    unsqueeze_535: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 2);  unsqueeze_534 = None
    unsqueeze_536: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 3);  unsqueeze_535 = None
    mul_821: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_286, primals_287);  primals_287 = None
    unsqueeze_537: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    unsqueeze_538: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 2);  unsqueeze_537 = None
    unsqueeze_539: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 3);  unsqueeze_538 = None
    sub_142: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_530);  convolution_95 = unsqueeze_530 = None
    mul_822: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_536);  sub_142 = unsqueeze_536 = None
    sub_143: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_822);  where_8 = mul_822 = None
    sub_144: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_533);  sub_143 = unsqueeze_533 = None
    mul_823: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_539);  sub_144 = unsqueeze_539 = None
    mul_824: "f32[512]" = torch.ops.aten.mul.Tensor(sum_20, squeeze_286);  sum_20 = squeeze_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_823, relu_91, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_823 = primals_286 = None
    getitem_258: "f32[8, 256, 14, 14]" = convolution_backward_10[0]
    getitem_259: "f32[512, 256, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_129: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_91);  relu_91 = None
    alias_130: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_129);  alias_129 = None
    le_9: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_130, 0);  alias_130 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_9: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_9, scalar_tensor_9, getitem_258);  le_9 = scalar_tensor_9 = getitem_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_540: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_282, 0);  squeeze_282 = None
    unsqueeze_541: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 2);  unsqueeze_540 = None
    unsqueeze_542: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 3);  unsqueeze_541 = None
    sum_21: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_145: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_542)
    mul_825: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_145);  sub_145 = None
    sum_22: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_825, [0, 2, 3]);  mul_825 = None
    mul_826: "f32[256]" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
    unsqueeze_543: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_826, 0);  mul_826 = None
    unsqueeze_544: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 2);  unsqueeze_543 = None
    unsqueeze_545: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 3);  unsqueeze_544 = None
    mul_827: "f32[256]" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
    mul_828: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
    mul_829: "f32[256]" = torch.ops.aten.mul.Tensor(mul_827, mul_828);  mul_827 = mul_828 = None
    unsqueeze_546: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
    unsqueeze_547: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 2);  unsqueeze_546 = None
    unsqueeze_548: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 3);  unsqueeze_547 = None
    mul_830: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_283, primals_284);  primals_284 = None
    unsqueeze_549: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_830, 0);  mul_830 = None
    unsqueeze_550: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 2);  unsqueeze_549 = None
    unsqueeze_551: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 3);  unsqueeze_550 = None
    sub_146: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_542);  convolution_94 = unsqueeze_542 = None
    mul_831: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_548);  sub_146 = unsqueeze_548 = None
    sub_147: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_9, mul_831);  where_9 = mul_831 = None
    sub_148: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_545);  sub_147 = unsqueeze_545 = None
    mul_832: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_551);  sub_148 = unsqueeze_551 = None
    mul_833: "f32[256]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_283);  sum_22 = squeeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_832, relu_90, primals_283, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_832 = primals_283 = None
    getitem_261: "f32[8, 256, 14, 14]" = convolution_backward_11[0]
    getitem_262: "f32[256, 256, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_132: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_90);  relu_90 = None
    alias_133: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_132);  alias_132 = None
    le_10: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_133, 0);  alias_133 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_10, scalar_tensor_10, getitem_261);  le_10 = scalar_tensor_10 = getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_552: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_279, 0);  squeeze_279 = None
    unsqueeze_553: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 2);  unsqueeze_552 = None
    unsqueeze_554: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 3);  unsqueeze_553 = None
    sum_23: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_149: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_554)
    mul_834: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_149);  sub_149 = None
    sum_24: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_834, [0, 2, 3]);  mul_834 = None
    mul_835: "f32[256]" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
    unsqueeze_555: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_835, 0);  mul_835 = None
    unsqueeze_556: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 2);  unsqueeze_555 = None
    unsqueeze_557: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 3);  unsqueeze_556 = None
    mul_836: "f32[256]" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
    mul_837: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
    mul_838: "f32[256]" = torch.ops.aten.mul.Tensor(mul_836, mul_837);  mul_836 = mul_837 = None
    unsqueeze_558: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_838, 0);  mul_838 = None
    unsqueeze_559: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 2);  unsqueeze_558 = None
    unsqueeze_560: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 3);  unsqueeze_559 = None
    mul_839: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_280, primals_281);  primals_281 = None
    unsqueeze_561: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_839, 0);  mul_839 = None
    unsqueeze_562: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 2);  unsqueeze_561 = None
    unsqueeze_563: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 3);  unsqueeze_562 = None
    sub_150: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_554);  convolution_93 = unsqueeze_554 = None
    mul_840: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_560);  sub_150 = unsqueeze_560 = None
    sub_151: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_840);  where_10 = mul_840 = None
    sub_152: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_557);  sub_151 = unsqueeze_557 = None
    mul_841: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_563);  sub_152 = unsqueeze_563 = None
    mul_842: "f32[256]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_280);  sum_24 = squeeze_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_841, relu_89, primals_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_841 = primals_280 = None
    getitem_264: "f32[8, 512, 14, 14]" = convolution_backward_12[0]
    getitem_265: "f32[256, 512, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_574: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_573, getitem_264);  add_573 = getitem_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_135: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_89);  relu_89 = None
    alias_136: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_135);  alias_135 = None
    le_11: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_136, 0);  alias_136 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_11: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_11, scalar_tensor_11, add_574);  le_11 = scalar_tensor_11 = add_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_575: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_9, where_11);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_564: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_276, 0);  squeeze_276 = None
    unsqueeze_565: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 2);  unsqueeze_564 = None
    unsqueeze_566: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 3);  unsqueeze_565 = None
    sum_25: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_153: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_566)
    mul_843: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_153);  sub_153 = None
    sum_26: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_843, [0, 2, 3]);  mul_843 = None
    mul_844: "f32[512]" = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
    unsqueeze_567: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
    unsqueeze_568: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 2);  unsqueeze_567 = None
    unsqueeze_569: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 3);  unsqueeze_568 = None
    mul_845: "f32[512]" = torch.ops.aten.mul.Tensor(sum_26, 0.0006377551020408163)
    mul_846: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
    mul_847: "f32[512]" = torch.ops.aten.mul.Tensor(mul_845, mul_846);  mul_845 = mul_846 = None
    unsqueeze_570: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_847, 0);  mul_847 = None
    unsqueeze_571: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 2);  unsqueeze_570 = None
    unsqueeze_572: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 3);  unsqueeze_571 = None
    mul_848: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_277, primals_278);  primals_278 = None
    unsqueeze_573: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
    unsqueeze_574: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 2);  unsqueeze_573 = None
    unsqueeze_575: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 3);  unsqueeze_574 = None
    sub_154: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_566);  convolution_92 = unsqueeze_566 = None
    mul_849: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_572);  sub_154 = unsqueeze_572 = None
    sub_155: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_849);  where_11 = mul_849 = None
    sub_156: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_569);  sub_155 = unsqueeze_569 = None
    mul_850: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_575);  sub_156 = unsqueeze_575 = None
    mul_851: "f32[512]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_277);  sum_26 = squeeze_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_850, relu_88, primals_277, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_850 = primals_277 = None
    getitem_267: "f32[8, 256, 14, 14]" = convolution_backward_13[0]
    getitem_268: "f32[512, 256, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_138: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_88);  relu_88 = None
    alias_139: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_138);  alias_138 = None
    le_12: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_139, 0);  alias_139 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_12: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_12, scalar_tensor_12, getitem_267);  le_12 = scalar_tensor_12 = getitem_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_576: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_273, 0);  squeeze_273 = None
    unsqueeze_577: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 2);  unsqueeze_576 = None
    unsqueeze_578: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 3);  unsqueeze_577 = None
    sum_27: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_157: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_578)
    mul_852: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_157);  sub_157 = None
    sum_28: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_852, [0, 2, 3]);  mul_852 = None
    mul_853: "f32[256]" = torch.ops.aten.mul.Tensor(sum_27, 0.0006377551020408163)
    unsqueeze_579: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_853, 0);  mul_853 = None
    unsqueeze_580: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 2);  unsqueeze_579 = None
    unsqueeze_581: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 3);  unsqueeze_580 = None
    mul_854: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, 0.0006377551020408163)
    mul_855: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
    mul_856: "f32[256]" = torch.ops.aten.mul.Tensor(mul_854, mul_855);  mul_854 = mul_855 = None
    unsqueeze_582: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_856, 0);  mul_856 = None
    unsqueeze_583: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 2);  unsqueeze_582 = None
    unsqueeze_584: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 3);  unsqueeze_583 = None
    mul_857: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_274, primals_275);  primals_275 = None
    unsqueeze_585: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_857, 0);  mul_857 = None
    unsqueeze_586: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
    unsqueeze_587: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
    sub_158: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_578);  convolution_91 = unsqueeze_578 = None
    mul_858: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_584);  sub_158 = unsqueeze_584 = None
    sub_159: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_858);  where_12 = mul_858 = None
    sub_160: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_581);  sub_159 = unsqueeze_581 = None
    mul_859: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_587);  sub_160 = unsqueeze_587 = None
    mul_860: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_274);  sum_28 = squeeze_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_859, relu_87, primals_274, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_859 = primals_274 = None
    getitem_270: "f32[8, 256, 14, 14]" = convolution_backward_14[0]
    getitem_271: "f32[256, 256, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_141: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_87);  relu_87 = None
    alias_142: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_141);  alias_141 = None
    le_13: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_142, 0);  alias_142 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_13, scalar_tensor_13, getitem_270);  le_13 = scalar_tensor_13 = getitem_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_588: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_270, 0);  squeeze_270 = None
    unsqueeze_589: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 2);  unsqueeze_588 = None
    unsqueeze_590: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 3);  unsqueeze_589 = None
    sum_29: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_161: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_590)
    mul_861: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_161);  sub_161 = None
    sum_30: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_861, [0, 2, 3]);  mul_861 = None
    mul_862: "f32[256]" = torch.ops.aten.mul.Tensor(sum_29, 0.0006377551020408163)
    unsqueeze_591: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
    unsqueeze_592: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 2);  unsqueeze_591 = None
    unsqueeze_593: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 3);  unsqueeze_592 = None
    mul_863: "f32[256]" = torch.ops.aten.mul.Tensor(sum_30, 0.0006377551020408163)
    mul_864: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
    mul_865: "f32[256]" = torch.ops.aten.mul.Tensor(mul_863, mul_864);  mul_863 = mul_864 = None
    unsqueeze_594: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_865, 0);  mul_865 = None
    unsqueeze_595: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 2);  unsqueeze_594 = None
    unsqueeze_596: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 3);  unsqueeze_595 = None
    mul_866: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_271, primals_272);  primals_272 = None
    unsqueeze_597: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_866, 0);  mul_866 = None
    unsqueeze_598: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
    unsqueeze_599: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
    sub_162: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_590);  convolution_90 = unsqueeze_590 = None
    mul_867: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_596);  sub_162 = unsqueeze_596 = None
    sub_163: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_867);  where_13 = mul_867 = None
    sub_164: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_593);  sub_163 = unsqueeze_593 = None
    mul_868: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_599);  sub_164 = unsqueeze_599 = None
    mul_869: "f32[256]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_271);  sum_30 = squeeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_868, relu_86, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_868 = primals_271 = None
    getitem_273: "f32[8, 512, 14, 14]" = convolution_backward_15[0]
    getitem_274: "f32[256, 512, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_576: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_575, getitem_273);  add_575 = getitem_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_144: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_86);  relu_86 = None
    alias_145: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_144);  alias_144 = None
    le_14: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_145, 0);  alias_145 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_14: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_14, scalar_tensor_14, add_576);  le_14 = scalar_tensor_14 = add_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_600: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_267, 0);  squeeze_267 = None
    unsqueeze_601: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 2);  unsqueeze_600 = None
    unsqueeze_602: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 3);  unsqueeze_601 = None
    sum_31: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_165: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_602)
    mul_870: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_165);  sub_165 = None
    sum_32: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_870, [0, 2, 3]);  mul_870 = None
    mul_871: "f32[512]" = torch.ops.aten.mul.Tensor(sum_31, 0.0006377551020408163)
    unsqueeze_603: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
    unsqueeze_604: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 2);  unsqueeze_603 = None
    unsqueeze_605: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 3);  unsqueeze_604 = None
    mul_872: "f32[512]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    mul_873: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
    mul_874: "f32[512]" = torch.ops.aten.mul.Tensor(mul_872, mul_873);  mul_872 = mul_873 = None
    unsqueeze_606: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_874, 0);  mul_874 = None
    unsqueeze_607: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 2);  unsqueeze_606 = None
    unsqueeze_608: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 3);  unsqueeze_607 = None
    mul_875: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_268, primals_269);  primals_269 = None
    unsqueeze_609: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
    unsqueeze_610: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
    unsqueeze_611: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
    sub_166: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_602);  convolution_89 = unsqueeze_602 = None
    mul_876: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_608);  sub_166 = unsqueeze_608 = None
    sub_167: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_876);  mul_876 = None
    sub_168: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_605);  sub_167 = unsqueeze_605 = None
    mul_877: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_611);  sub_168 = unsqueeze_611 = None
    mul_878: "f32[512]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_268);  sum_32 = squeeze_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_877, cat_11, primals_268, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_877 = cat_11 = primals_268 = None
    getitem_276: "f32[8, 1024, 14, 14]" = convolution_backward_16[0]
    getitem_277: "f32[512, 1024, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    slice_10: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_276, 1, 0, 512)
    slice_11: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_276, 1, 512, 1024);  getitem_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_577: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_14, slice_10);  where_14 = slice_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_147: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_85);  relu_85 = None
    alias_148: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_147);  alias_147 = None
    le_15: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_148, 0);  alias_148 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_15, scalar_tensor_15, add_577);  le_15 = scalar_tensor_15 = add_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_578: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_11, where_15);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_612: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_264, 0);  squeeze_264 = None
    unsqueeze_613: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 2);  unsqueeze_612 = None
    unsqueeze_614: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 3);  unsqueeze_613 = None
    sum_33: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_169: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_614)
    mul_879: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_169);  sub_169 = None
    sum_34: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_879, [0, 2, 3]);  mul_879 = None
    mul_880: "f32[512]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    unsqueeze_615: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_880, 0);  mul_880 = None
    unsqueeze_616: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 2);  unsqueeze_615 = None
    unsqueeze_617: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 3);  unsqueeze_616 = None
    mul_881: "f32[512]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    mul_882: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
    mul_883: "f32[512]" = torch.ops.aten.mul.Tensor(mul_881, mul_882);  mul_881 = mul_882 = None
    unsqueeze_618: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_883, 0);  mul_883 = None
    unsqueeze_619: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 2);  unsqueeze_618 = None
    unsqueeze_620: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 3);  unsqueeze_619 = None
    mul_884: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_265, primals_266);  primals_266 = None
    unsqueeze_621: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_622: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 2);  unsqueeze_621 = None
    unsqueeze_623: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 3);  unsqueeze_622 = None
    sub_170: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_614);  convolution_88 = unsqueeze_614 = None
    mul_885: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_620);  sub_170 = unsqueeze_620 = None
    sub_171: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_885);  where_15 = mul_885 = None
    sub_172: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_617);  sub_171 = unsqueeze_617 = None
    mul_886: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_623);  sub_172 = unsqueeze_623 = None
    mul_887: "f32[512]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_265);  sum_34 = squeeze_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_886, relu_84, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_886 = primals_265 = None
    getitem_279: "f32[8, 256, 14, 14]" = convolution_backward_17[0]
    getitem_280: "f32[512, 256, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_150: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_84);  relu_84 = None
    alias_151: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_150);  alias_150 = None
    le_16: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_151, 0);  alias_151 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_16: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_16, scalar_tensor_16, getitem_279);  le_16 = scalar_tensor_16 = getitem_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_624: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_261, 0);  squeeze_261 = None
    unsqueeze_625: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 2);  unsqueeze_624 = None
    unsqueeze_626: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 3);  unsqueeze_625 = None
    sum_35: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_173: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_626)
    mul_888: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_173);  sub_173 = None
    sum_36: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_888, [0, 2, 3]);  mul_888 = None
    mul_889: "f32[256]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    unsqueeze_627: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_889, 0);  mul_889 = None
    unsqueeze_628: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 2);  unsqueeze_627 = None
    unsqueeze_629: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 3);  unsqueeze_628 = None
    mul_890: "f32[256]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    mul_891: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
    mul_892: "f32[256]" = torch.ops.aten.mul.Tensor(mul_890, mul_891);  mul_890 = mul_891 = None
    unsqueeze_630: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
    unsqueeze_631: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 2);  unsqueeze_630 = None
    unsqueeze_632: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 3);  unsqueeze_631 = None
    mul_893: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_262, primals_263);  primals_263 = None
    unsqueeze_633: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_893, 0);  mul_893 = None
    unsqueeze_634: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 2);  unsqueeze_633 = None
    unsqueeze_635: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 3);  unsqueeze_634 = None
    sub_174: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_626);  convolution_87 = unsqueeze_626 = None
    mul_894: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_632);  sub_174 = unsqueeze_632 = None
    sub_175: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_894);  where_16 = mul_894 = None
    sub_176: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_629);  sub_175 = unsqueeze_629 = None
    mul_895: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_635);  sub_176 = unsqueeze_635 = None
    mul_896: "f32[256]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_262);  sum_36 = squeeze_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_895, relu_83, primals_262, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_895 = primals_262 = None
    getitem_282: "f32[8, 256, 14, 14]" = convolution_backward_18[0]
    getitem_283: "f32[256, 256, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_153: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_83);  relu_83 = None
    alias_154: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_153);  alias_153 = None
    le_17: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_154, 0);  alias_154 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_17, scalar_tensor_17, getitem_282);  le_17 = scalar_tensor_17 = getitem_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_636: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_258, 0);  squeeze_258 = None
    unsqueeze_637: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 2);  unsqueeze_636 = None
    unsqueeze_638: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 3);  unsqueeze_637 = None
    sum_37: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_177: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_638)
    mul_897: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_177);  sub_177 = None
    sum_38: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 2, 3]);  mul_897 = None
    mul_898: "f32[256]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    unsqueeze_639: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_898, 0);  mul_898 = None
    unsqueeze_640: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 2);  unsqueeze_639 = None
    unsqueeze_641: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 3);  unsqueeze_640 = None
    mul_899: "f32[256]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    mul_900: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
    mul_901: "f32[256]" = torch.ops.aten.mul.Tensor(mul_899, mul_900);  mul_899 = mul_900 = None
    unsqueeze_642: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
    unsqueeze_643: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 2);  unsqueeze_642 = None
    unsqueeze_644: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 3);  unsqueeze_643 = None
    mul_902: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_259, primals_260);  primals_260 = None
    unsqueeze_645: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
    unsqueeze_646: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 2);  unsqueeze_645 = None
    unsqueeze_647: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 3);  unsqueeze_646 = None
    sub_178: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_638);  convolution_86 = unsqueeze_638 = None
    mul_903: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_644);  sub_178 = unsqueeze_644 = None
    sub_179: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_903);  where_17 = mul_903 = None
    sub_180: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_641);  sub_179 = unsqueeze_641 = None
    mul_904: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_647);  sub_180 = unsqueeze_647 = None
    mul_905: "f32[256]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_259);  sum_38 = squeeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_904, relu_82, primals_259, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_904 = primals_259 = None
    getitem_285: "f32[8, 512, 14, 14]" = convolution_backward_19[0]
    getitem_286: "f32[256, 512, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_579: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_578, getitem_285);  add_578 = getitem_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_156: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_82);  relu_82 = None
    alias_157: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_156);  alias_156 = None
    le_18: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_157, 0);  alias_157 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_18: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_18, scalar_tensor_18, add_579);  le_18 = scalar_tensor_18 = add_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_580: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_8, where_18);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_648: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_255, 0);  squeeze_255 = None
    unsqueeze_649: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 2);  unsqueeze_648 = None
    unsqueeze_650: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 3);  unsqueeze_649 = None
    sum_39: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_181: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_650)
    mul_906: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_181);  sub_181 = None
    sum_40: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_906, [0, 2, 3]);  mul_906 = None
    mul_907: "f32[512]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    unsqueeze_651: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_907, 0);  mul_907 = None
    unsqueeze_652: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 2);  unsqueeze_651 = None
    unsqueeze_653: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 3);  unsqueeze_652 = None
    mul_908: "f32[512]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    mul_909: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
    mul_910: "f32[512]" = torch.ops.aten.mul.Tensor(mul_908, mul_909);  mul_908 = mul_909 = None
    unsqueeze_654: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_910, 0);  mul_910 = None
    unsqueeze_655: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 2);  unsqueeze_654 = None
    unsqueeze_656: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 3);  unsqueeze_655 = None
    mul_911: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_256, primals_257);  primals_257 = None
    unsqueeze_657: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
    unsqueeze_658: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 2);  unsqueeze_657 = None
    unsqueeze_659: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 3);  unsqueeze_658 = None
    sub_182: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_650);  convolution_85 = unsqueeze_650 = None
    mul_912: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_656);  sub_182 = unsqueeze_656 = None
    sub_183: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_912);  where_18 = mul_912 = None
    sub_184: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_653);  sub_183 = unsqueeze_653 = None
    mul_913: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_659);  sub_184 = unsqueeze_659 = None
    mul_914: "f32[512]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_256);  sum_40 = squeeze_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_913, relu_81, primals_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_913 = primals_256 = None
    getitem_288: "f32[8, 256, 14, 14]" = convolution_backward_20[0]
    getitem_289: "f32[512, 256, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_159: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_81);  relu_81 = None
    alias_160: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_159);  alias_159 = None
    le_19: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_160, 0);  alias_160 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_19: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_19, scalar_tensor_19, getitem_288);  le_19 = scalar_tensor_19 = getitem_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_660: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
    unsqueeze_661: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 2);  unsqueeze_660 = None
    unsqueeze_662: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 3);  unsqueeze_661 = None
    sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_185: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_662)
    mul_915: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_185);  sub_185 = None
    sum_42: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_915, [0, 2, 3]);  mul_915 = None
    mul_916: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    unsqueeze_663: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_916, 0);  mul_916 = None
    unsqueeze_664: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 2);  unsqueeze_663 = None
    unsqueeze_665: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 3);  unsqueeze_664 = None
    mul_917: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    mul_918: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
    mul_919: "f32[256]" = torch.ops.aten.mul.Tensor(mul_917, mul_918);  mul_917 = mul_918 = None
    unsqueeze_666: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_919, 0);  mul_919 = None
    unsqueeze_667: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 2);  unsqueeze_666 = None
    unsqueeze_668: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 3);  unsqueeze_667 = None
    mul_920: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_254);  primals_254 = None
    unsqueeze_669: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
    unsqueeze_670: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 2);  unsqueeze_669 = None
    unsqueeze_671: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 3);  unsqueeze_670 = None
    sub_186: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_662);  convolution_84 = unsqueeze_662 = None
    mul_921: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_668);  sub_186 = unsqueeze_668 = None
    sub_187: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_921);  where_19 = mul_921 = None
    sub_188: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_665);  sub_187 = unsqueeze_665 = None
    mul_922: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_671);  sub_188 = unsqueeze_671 = None
    mul_923: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, squeeze_253);  sum_42 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_922, relu_80, primals_253, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_922 = primals_253 = None
    getitem_291: "f32[8, 256, 14, 14]" = convolution_backward_21[0]
    getitem_292: "f32[256, 256, 3, 3]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_162: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_80);  relu_80 = None
    alias_163: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_162);  alias_162 = None
    le_20: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_163, 0);  alias_163 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_20: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_20, scalar_tensor_20, getitem_291);  le_20 = scalar_tensor_20 = getitem_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_672: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
    unsqueeze_673: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 2);  unsqueeze_672 = None
    unsqueeze_674: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 3);  unsqueeze_673 = None
    sum_43: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_189: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_674)
    mul_924: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_189);  sub_189 = None
    sum_44: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_924, [0, 2, 3]);  mul_924 = None
    mul_925: "f32[256]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    unsqueeze_675: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_925, 0);  mul_925 = None
    unsqueeze_676: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 2);  unsqueeze_675 = None
    unsqueeze_677: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 3);  unsqueeze_676 = None
    mul_926: "f32[256]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    mul_927: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
    mul_928: "f32[256]" = torch.ops.aten.mul.Tensor(mul_926, mul_927);  mul_926 = mul_927 = None
    unsqueeze_678: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_928, 0);  mul_928 = None
    unsqueeze_679: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 2);  unsqueeze_678 = None
    unsqueeze_680: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 3);  unsqueeze_679 = None
    mul_929: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_251);  primals_251 = None
    unsqueeze_681: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_929, 0);  mul_929 = None
    unsqueeze_682: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 2);  unsqueeze_681 = None
    unsqueeze_683: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 3);  unsqueeze_682 = None
    sub_190: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_674);  convolution_83 = unsqueeze_674 = None
    mul_930: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_680);  sub_190 = unsqueeze_680 = None
    sub_191: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_930);  where_20 = mul_930 = None
    sub_192: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_677);  sub_191 = unsqueeze_677 = None
    mul_931: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_683);  sub_192 = unsqueeze_683 = None
    mul_932: "f32[256]" = torch.ops.aten.mul.Tensor(sum_44, squeeze_250);  sum_44 = squeeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_931, relu_79, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_931 = primals_250 = None
    getitem_294: "f32[8, 512, 14, 14]" = convolution_backward_22[0]
    getitem_295: "f32[256, 512, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_581: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_580, getitem_294);  add_580 = getitem_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_165: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_79);  relu_79 = None
    alias_166: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_165);  alias_165 = None
    le_21: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_166, 0);  alias_166 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_21: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_21, scalar_tensor_21, add_581);  le_21 = scalar_tensor_21 = add_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_684: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
    unsqueeze_685: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 2);  unsqueeze_684 = None
    unsqueeze_686: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 3);  unsqueeze_685 = None
    sum_45: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_193: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_686)
    mul_933: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_193);  sub_193 = None
    sum_46: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_933, [0, 2, 3]);  mul_933 = None
    mul_934: "f32[512]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    unsqueeze_687: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_934, 0);  mul_934 = None
    unsqueeze_688: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 2);  unsqueeze_687 = None
    unsqueeze_689: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 3);  unsqueeze_688 = None
    mul_935: "f32[512]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    mul_936: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
    mul_937: "f32[512]" = torch.ops.aten.mul.Tensor(mul_935, mul_936);  mul_935 = mul_936 = None
    unsqueeze_690: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_937, 0);  mul_937 = None
    unsqueeze_691: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 2);  unsqueeze_690 = None
    unsqueeze_692: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 3);  unsqueeze_691 = None
    mul_938: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_248);  primals_248 = None
    unsqueeze_693: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
    unsqueeze_694: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 2);  unsqueeze_693 = None
    unsqueeze_695: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 3);  unsqueeze_694 = None
    sub_194: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_686);  convolution_82 = unsqueeze_686 = None
    mul_939: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_692);  sub_194 = unsqueeze_692 = None
    sub_195: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_939);  mul_939 = None
    sub_196: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_689);  sub_195 = unsqueeze_689 = None
    mul_940: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_695);  sub_196 = unsqueeze_695 = None
    mul_941: "f32[512]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_247);  sum_46 = squeeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_940, cat_10, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_940 = cat_10 = primals_247 = None
    getitem_297: "f32[8, 1536, 14, 14]" = convolution_backward_23[0]
    getitem_298: "f32[512, 1536, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    slice_12: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_297, 1, 0, 512)
    slice_13: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_297, 1, 512, 1024)
    slice_14: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_297, 1, 1024, 1536);  getitem_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_582: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_21, slice_12);  where_21 = slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_168: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_78);  relu_78 = None
    alias_169: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_168);  alias_168 = None
    le_22: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_169, 0);  alias_169 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_22: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_22, scalar_tensor_22, add_582);  le_22 = scalar_tensor_22 = add_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_583: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_13, where_22);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_696: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
    unsqueeze_697: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 2);  unsqueeze_696 = None
    unsqueeze_698: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 3);  unsqueeze_697 = None
    sum_47: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_197: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_698)
    mul_942: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_197);  sub_197 = None
    sum_48: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_942, [0, 2, 3]);  mul_942 = None
    mul_943: "f32[512]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    unsqueeze_699: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_943, 0);  mul_943 = None
    unsqueeze_700: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 2);  unsqueeze_699 = None
    unsqueeze_701: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 3);  unsqueeze_700 = None
    mul_944: "f32[512]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    mul_945: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
    mul_946: "f32[512]" = torch.ops.aten.mul.Tensor(mul_944, mul_945);  mul_944 = mul_945 = None
    unsqueeze_702: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_946, 0);  mul_946 = None
    unsqueeze_703: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 2);  unsqueeze_702 = None
    unsqueeze_704: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 3);  unsqueeze_703 = None
    mul_947: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_245);  primals_245 = None
    unsqueeze_705: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_947, 0);  mul_947 = None
    unsqueeze_706: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 2);  unsqueeze_705 = None
    unsqueeze_707: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 3);  unsqueeze_706 = None
    sub_198: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_698);  convolution_81 = unsqueeze_698 = None
    mul_948: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_704);  sub_198 = unsqueeze_704 = None
    sub_199: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_22, mul_948);  where_22 = mul_948 = None
    sub_200: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_701);  sub_199 = unsqueeze_701 = None
    mul_949: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_707);  sub_200 = unsqueeze_707 = None
    mul_950: "f32[512]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_244);  sum_48 = squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_949, relu_77, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_949 = primals_244 = None
    getitem_300: "f32[8, 256, 14, 14]" = convolution_backward_24[0]
    getitem_301: "f32[512, 256, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_171: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_77);  relu_77 = None
    alias_172: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_171);  alias_171 = None
    le_23: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_172, 0);  alias_172 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_23: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_23, scalar_tensor_23, getitem_300);  le_23 = scalar_tensor_23 = getitem_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_708: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
    unsqueeze_709: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 2);  unsqueeze_708 = None
    unsqueeze_710: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 3);  unsqueeze_709 = None
    sum_49: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_201: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_710)
    mul_951: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_201);  sub_201 = None
    sum_50: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_951, [0, 2, 3]);  mul_951 = None
    mul_952: "f32[256]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    unsqueeze_711: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_952, 0);  mul_952 = None
    unsqueeze_712: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 2);  unsqueeze_711 = None
    unsqueeze_713: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 3);  unsqueeze_712 = None
    mul_953: "f32[256]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    mul_954: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
    mul_955: "f32[256]" = torch.ops.aten.mul.Tensor(mul_953, mul_954);  mul_953 = mul_954 = None
    unsqueeze_714: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_955, 0);  mul_955 = None
    unsqueeze_715: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 2);  unsqueeze_714 = None
    unsqueeze_716: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 3);  unsqueeze_715 = None
    mul_956: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_242);  primals_242 = None
    unsqueeze_717: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_956, 0);  mul_956 = None
    unsqueeze_718: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 2);  unsqueeze_717 = None
    unsqueeze_719: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 3);  unsqueeze_718 = None
    sub_202: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_710);  convolution_80 = unsqueeze_710 = None
    mul_957: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_716);  sub_202 = unsqueeze_716 = None
    sub_203: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_957);  where_23 = mul_957 = None
    sub_204: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_713);  sub_203 = unsqueeze_713 = None
    mul_958: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_719);  sub_204 = unsqueeze_719 = None
    mul_959: "f32[256]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_241);  sum_50 = squeeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_958, relu_76, primals_241, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_958 = primals_241 = None
    getitem_303: "f32[8, 256, 14, 14]" = convolution_backward_25[0]
    getitem_304: "f32[256, 256, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_174: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_76);  relu_76 = None
    alias_175: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_174);  alias_174 = None
    le_24: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_175, 0);  alias_175 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_24: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_24, scalar_tensor_24, getitem_303);  le_24 = scalar_tensor_24 = getitem_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_720: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
    unsqueeze_721: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 2);  unsqueeze_720 = None
    unsqueeze_722: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 3);  unsqueeze_721 = None
    sum_51: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_205: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_722)
    mul_960: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_205);  sub_205 = None
    sum_52: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_960, [0, 2, 3]);  mul_960 = None
    mul_961: "f32[256]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    unsqueeze_723: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_961, 0);  mul_961 = None
    unsqueeze_724: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 2);  unsqueeze_723 = None
    unsqueeze_725: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 3);  unsqueeze_724 = None
    mul_962: "f32[256]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    mul_963: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_964: "f32[256]" = torch.ops.aten.mul.Tensor(mul_962, mul_963);  mul_962 = mul_963 = None
    unsqueeze_726: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_964, 0);  mul_964 = None
    unsqueeze_727: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 2);  unsqueeze_726 = None
    unsqueeze_728: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 3);  unsqueeze_727 = None
    mul_965: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_239);  primals_239 = None
    unsqueeze_729: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_965, 0);  mul_965 = None
    unsqueeze_730: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 2);  unsqueeze_729 = None
    unsqueeze_731: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 3);  unsqueeze_730 = None
    sub_206: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_722);  convolution_79 = unsqueeze_722 = None
    mul_966: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_728);  sub_206 = unsqueeze_728 = None
    sub_207: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_966);  where_24 = mul_966 = None
    sub_208: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_725);  sub_207 = unsqueeze_725 = None
    mul_967: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_731);  sub_208 = unsqueeze_731 = None
    mul_968: "f32[256]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_238);  sum_52 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_967, relu_75, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_967 = primals_238 = None
    getitem_306: "f32[8, 512, 14, 14]" = convolution_backward_26[0]
    getitem_307: "f32[256, 512, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_584: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_583, getitem_306);  add_583 = getitem_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_177: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_75);  relu_75 = None
    alias_178: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_177);  alias_177 = None
    le_25: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_178, 0);  alias_178 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_25: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_25, scalar_tensor_25, add_584);  le_25 = scalar_tensor_25 = add_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_585: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_14, where_25);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_732: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
    unsqueeze_733: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 2);  unsqueeze_732 = None
    unsqueeze_734: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 3);  unsqueeze_733 = None
    sum_53: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_209: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_734)
    mul_969: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_209);  sub_209 = None
    sum_54: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_969, [0, 2, 3]);  mul_969 = None
    mul_970: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    unsqueeze_735: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_970, 0);  mul_970 = None
    unsqueeze_736: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 2);  unsqueeze_735 = None
    unsqueeze_737: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 3);  unsqueeze_736 = None
    mul_971: "f32[512]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    mul_972: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_973: "f32[512]" = torch.ops.aten.mul.Tensor(mul_971, mul_972);  mul_971 = mul_972 = None
    unsqueeze_738: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_973, 0);  mul_973 = None
    unsqueeze_739: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 2);  unsqueeze_738 = None
    unsqueeze_740: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 3);  unsqueeze_739 = None
    mul_974: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_236);  primals_236 = None
    unsqueeze_741: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
    unsqueeze_742: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 2);  unsqueeze_741 = None
    unsqueeze_743: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 3);  unsqueeze_742 = None
    sub_210: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_734);  convolution_78 = unsqueeze_734 = None
    mul_975: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_740);  sub_210 = unsqueeze_740 = None
    sub_211: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_975);  where_25 = mul_975 = None
    sub_212: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_737);  sub_211 = unsqueeze_737 = None
    mul_976: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_743);  sub_212 = unsqueeze_743 = None
    mul_977: "f32[512]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_235);  sum_54 = squeeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_976, relu_74, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_976 = primals_235 = None
    getitem_309: "f32[8, 256, 14, 14]" = convolution_backward_27[0]
    getitem_310: "f32[512, 256, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_180: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_74);  relu_74 = None
    alias_181: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_180);  alias_180 = None
    le_26: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_181, 0);  alias_181 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_26: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_26, scalar_tensor_26, getitem_309);  le_26 = scalar_tensor_26 = getitem_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_744: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
    unsqueeze_745: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 2);  unsqueeze_744 = None
    unsqueeze_746: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 3);  unsqueeze_745 = None
    sum_55: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_213: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_746)
    mul_978: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_213);  sub_213 = None
    sum_56: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_978, [0, 2, 3]);  mul_978 = None
    mul_979: "f32[256]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    unsqueeze_747: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_979, 0);  mul_979 = None
    unsqueeze_748: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 2);  unsqueeze_747 = None
    unsqueeze_749: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 3);  unsqueeze_748 = None
    mul_980: "f32[256]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    mul_981: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_982: "f32[256]" = torch.ops.aten.mul.Tensor(mul_980, mul_981);  mul_980 = mul_981 = None
    unsqueeze_750: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_982, 0);  mul_982 = None
    unsqueeze_751: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 2);  unsqueeze_750 = None
    unsqueeze_752: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 3);  unsqueeze_751 = None
    mul_983: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_233);  primals_233 = None
    unsqueeze_753: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_754: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 2);  unsqueeze_753 = None
    unsqueeze_755: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 3);  unsqueeze_754 = None
    sub_214: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_746);  convolution_77 = unsqueeze_746 = None
    mul_984: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_752);  sub_214 = unsqueeze_752 = None
    sub_215: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_26, mul_984);  where_26 = mul_984 = None
    sub_216: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_749);  sub_215 = unsqueeze_749 = None
    mul_985: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_755);  sub_216 = unsqueeze_755 = None
    mul_986: "f32[256]" = torch.ops.aten.mul.Tensor(sum_56, squeeze_232);  sum_56 = squeeze_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_985, relu_73, primals_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_985 = primals_232 = None
    getitem_312: "f32[8, 256, 14, 14]" = convolution_backward_28[0]
    getitem_313: "f32[256, 256, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_183: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_73);  relu_73 = None
    alias_184: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_183);  alias_183 = None
    le_27: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_184, 0);  alias_184 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_27: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_27, scalar_tensor_27, getitem_312);  le_27 = scalar_tensor_27 = getitem_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_756: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
    unsqueeze_757: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 2);  unsqueeze_756 = None
    unsqueeze_758: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 3);  unsqueeze_757 = None
    sum_57: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_217: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_758)
    mul_987: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_217);  sub_217 = None
    sum_58: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_987, [0, 2, 3]);  mul_987 = None
    mul_988: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    unsqueeze_759: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_988, 0);  mul_988 = None
    unsqueeze_760: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 2);  unsqueeze_759 = None
    unsqueeze_761: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 3);  unsqueeze_760 = None
    mul_989: "f32[256]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    mul_990: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_991: "f32[256]" = torch.ops.aten.mul.Tensor(mul_989, mul_990);  mul_989 = mul_990 = None
    unsqueeze_762: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_991, 0);  mul_991 = None
    unsqueeze_763: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 2);  unsqueeze_762 = None
    unsqueeze_764: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 3);  unsqueeze_763 = None
    mul_992: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_230);  primals_230 = None
    unsqueeze_765: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_992, 0);  mul_992 = None
    unsqueeze_766: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 2);  unsqueeze_765 = None
    unsqueeze_767: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 3);  unsqueeze_766 = None
    sub_218: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_758);  convolution_76 = unsqueeze_758 = None
    mul_993: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_764);  sub_218 = unsqueeze_764 = None
    sub_219: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_993);  where_27 = mul_993 = None
    sub_220: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_761);  sub_219 = unsqueeze_761 = None
    mul_994: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_767);  sub_220 = unsqueeze_767 = None
    mul_995: "f32[256]" = torch.ops.aten.mul.Tensor(sum_58, squeeze_229);  sum_58 = squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_994, relu_72, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_994 = primals_229 = None
    getitem_315: "f32[8, 512, 14, 14]" = convolution_backward_29[0]
    getitem_316: "f32[256, 512, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_586: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_585, getitem_315);  add_585 = getitem_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_186: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_72);  relu_72 = None
    alias_187: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_186);  alias_186 = None
    le_28: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_187, 0);  alias_187 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_28: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_28, scalar_tensor_28, add_586);  le_28 = scalar_tensor_28 = add_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_768: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
    unsqueeze_769: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 2);  unsqueeze_768 = None
    unsqueeze_770: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 3);  unsqueeze_769 = None
    sum_59: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_221: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_770)
    mul_996: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_221);  sub_221 = None
    sum_60: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_996, [0, 2, 3]);  mul_996 = None
    mul_997: "f32[512]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    unsqueeze_771: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_997, 0);  mul_997 = None
    unsqueeze_772: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 2);  unsqueeze_771 = None
    unsqueeze_773: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 3);  unsqueeze_772 = None
    mul_998: "f32[512]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    mul_999: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_1000: "f32[512]" = torch.ops.aten.mul.Tensor(mul_998, mul_999);  mul_998 = mul_999 = None
    unsqueeze_774: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
    unsqueeze_775: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 2);  unsqueeze_774 = None
    unsqueeze_776: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 3);  unsqueeze_775 = None
    mul_1001: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_227);  primals_227 = None
    unsqueeze_777: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1001, 0);  mul_1001 = None
    unsqueeze_778: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 2);  unsqueeze_777 = None
    unsqueeze_779: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 3);  unsqueeze_778 = None
    sub_222: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_770);  convolution_75 = unsqueeze_770 = None
    mul_1002: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_776);  sub_222 = unsqueeze_776 = None
    sub_223: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_28, mul_1002);  mul_1002 = None
    sub_224: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_773);  sub_223 = unsqueeze_773 = None
    mul_1003: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_779);  sub_224 = unsqueeze_779 = None
    mul_1004: "f32[512]" = torch.ops.aten.mul.Tensor(sum_60, squeeze_226);  sum_60 = squeeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_1003, cat_9, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1003 = cat_9 = primals_226 = None
    getitem_318: "f32[8, 1024, 14, 14]" = convolution_backward_30[0]
    getitem_319: "f32[512, 1024, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    slice_15: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_318, 1, 0, 512)
    slice_16: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_318, 1, 512, 1024);  getitem_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_587: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_28, slice_15);  where_28 = slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_189: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_71);  relu_71 = None
    alias_190: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_189);  alias_189 = None
    le_29: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_190, 0);  alias_190 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_29: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_29, scalar_tensor_29, add_587);  le_29 = scalar_tensor_29 = add_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_588: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_16, where_29);  slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_780: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
    unsqueeze_781: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 2);  unsqueeze_780 = None
    unsqueeze_782: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 3);  unsqueeze_781 = None
    sum_61: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_225: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_782)
    mul_1005: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_29, sub_225);  sub_225 = None
    sum_62: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1005, [0, 2, 3]);  mul_1005 = None
    mul_1006: "f32[512]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    unsqueeze_783: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1006, 0);  mul_1006 = None
    unsqueeze_784: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 2);  unsqueeze_783 = None
    unsqueeze_785: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 3);  unsqueeze_784 = None
    mul_1007: "f32[512]" = torch.ops.aten.mul.Tensor(sum_62, 0.0006377551020408163)
    mul_1008: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_1009: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1007, mul_1008);  mul_1007 = mul_1008 = None
    unsqueeze_786: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1009, 0);  mul_1009 = None
    unsqueeze_787: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 2);  unsqueeze_786 = None
    unsqueeze_788: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 3);  unsqueeze_787 = None
    mul_1010: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_224);  primals_224 = None
    unsqueeze_789: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1010, 0);  mul_1010 = None
    unsqueeze_790: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 2);  unsqueeze_789 = None
    unsqueeze_791: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 3);  unsqueeze_790 = None
    sub_226: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_782);  convolution_74 = unsqueeze_782 = None
    mul_1011: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_788);  sub_226 = unsqueeze_788 = None
    sub_227: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_29, mul_1011);  where_29 = mul_1011 = None
    sub_228: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_785);  sub_227 = unsqueeze_785 = None
    mul_1012: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_791);  sub_228 = unsqueeze_791 = None
    mul_1013: "f32[512]" = torch.ops.aten.mul.Tensor(sum_62, squeeze_223);  sum_62 = squeeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_1012, relu_70, primals_223, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1012 = primals_223 = None
    getitem_321: "f32[8, 256, 14, 14]" = convolution_backward_31[0]
    getitem_322: "f32[512, 256, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_192: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_70);  relu_70 = None
    alias_193: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_192);  alias_192 = None
    le_30: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_193, 0);  alias_193 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_30: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_30, scalar_tensor_30, getitem_321);  le_30 = scalar_tensor_30 = getitem_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_792: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
    unsqueeze_793: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 2);  unsqueeze_792 = None
    unsqueeze_794: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 3);  unsqueeze_793 = None
    sum_63: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_229: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_794)
    mul_1014: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_229);  sub_229 = None
    sum_64: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1014, [0, 2, 3]);  mul_1014 = None
    mul_1015: "f32[256]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    unsqueeze_795: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1015, 0);  mul_1015 = None
    unsqueeze_796: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 2);  unsqueeze_795 = None
    unsqueeze_797: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 3);  unsqueeze_796 = None
    mul_1016: "f32[256]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    mul_1017: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_1018: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1016, mul_1017);  mul_1016 = mul_1017 = None
    unsqueeze_798: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1018, 0);  mul_1018 = None
    unsqueeze_799: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 2);  unsqueeze_798 = None
    unsqueeze_800: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 3);  unsqueeze_799 = None
    mul_1019: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_221);  primals_221 = None
    unsqueeze_801: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
    unsqueeze_802: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 2);  unsqueeze_801 = None
    unsqueeze_803: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 3);  unsqueeze_802 = None
    sub_230: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_794);  convolution_73 = unsqueeze_794 = None
    mul_1020: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_800);  sub_230 = unsqueeze_800 = None
    sub_231: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_30, mul_1020);  where_30 = mul_1020 = None
    sub_232: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_797);  sub_231 = unsqueeze_797 = None
    mul_1021: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_803);  sub_232 = unsqueeze_803 = None
    mul_1022: "f32[256]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_220);  sum_64 = squeeze_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_1021, relu_69, primals_220, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1021 = primals_220 = None
    getitem_324: "f32[8, 256, 14, 14]" = convolution_backward_32[0]
    getitem_325: "f32[256, 256, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_195: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_69);  relu_69 = None
    alias_196: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_195);  alias_195 = None
    le_31: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_196, 0);  alias_196 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_31: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_31, scalar_tensor_31, getitem_324);  le_31 = scalar_tensor_31 = getitem_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_804: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
    unsqueeze_805: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 2);  unsqueeze_804 = None
    unsqueeze_806: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 3);  unsqueeze_805 = None
    sum_65: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_233: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_806)
    mul_1023: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_233);  sub_233 = None
    sum_66: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1023, [0, 2, 3]);  mul_1023 = None
    mul_1024: "f32[256]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    unsqueeze_807: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1024, 0);  mul_1024 = None
    unsqueeze_808: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 2);  unsqueeze_807 = None
    unsqueeze_809: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 3);  unsqueeze_808 = None
    mul_1025: "f32[256]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    mul_1026: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_1027: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1025, mul_1026);  mul_1025 = mul_1026 = None
    unsqueeze_810: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1027, 0);  mul_1027 = None
    unsqueeze_811: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 2);  unsqueeze_810 = None
    unsqueeze_812: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 3);  unsqueeze_811 = None
    mul_1028: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_218);  primals_218 = None
    unsqueeze_813: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1028, 0);  mul_1028 = None
    unsqueeze_814: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 2);  unsqueeze_813 = None
    unsqueeze_815: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 3);  unsqueeze_814 = None
    sub_234: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_806);  convolution_72 = unsqueeze_806 = None
    mul_1029: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_812);  sub_234 = unsqueeze_812 = None
    sub_235: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_1029);  where_31 = mul_1029 = None
    sub_236: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_809);  sub_235 = unsqueeze_809 = None
    mul_1030: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_815);  sub_236 = unsqueeze_815 = None
    mul_1031: "f32[256]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_217);  sum_66 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_1030, relu_68, primals_217, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1030 = primals_217 = None
    getitem_327: "f32[8, 512, 14, 14]" = convolution_backward_33[0]
    getitem_328: "f32[256, 512, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_589: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_588, getitem_327);  add_588 = getitem_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_198: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_68);  relu_68 = None
    alias_199: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_198);  alias_198 = None
    le_32: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_199, 0);  alias_199 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_32: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_32, scalar_tensor_32, add_589);  le_32 = scalar_tensor_32 = add_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_590: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_7, where_32);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_816: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
    unsqueeze_817: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 2);  unsqueeze_816 = None
    unsqueeze_818: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 3);  unsqueeze_817 = None
    sum_67: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_237: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_818)
    mul_1032: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_237);  sub_237 = None
    sum_68: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1032, [0, 2, 3]);  mul_1032 = None
    mul_1033: "f32[512]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    unsqueeze_819: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1033, 0);  mul_1033 = None
    unsqueeze_820: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 2);  unsqueeze_819 = None
    unsqueeze_821: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 3);  unsqueeze_820 = None
    mul_1034: "f32[512]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    mul_1035: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_1036: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1034, mul_1035);  mul_1034 = mul_1035 = None
    unsqueeze_822: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1036, 0);  mul_1036 = None
    unsqueeze_823: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 2);  unsqueeze_822 = None
    unsqueeze_824: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 3);  unsqueeze_823 = None
    mul_1037: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_215);  primals_215 = None
    unsqueeze_825: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1037, 0);  mul_1037 = None
    unsqueeze_826: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 2);  unsqueeze_825 = None
    unsqueeze_827: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 3);  unsqueeze_826 = None
    sub_238: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_818);  convolution_71 = unsqueeze_818 = None
    mul_1038: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_824);  sub_238 = unsqueeze_824 = None
    sub_239: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_32, mul_1038);  where_32 = mul_1038 = None
    sub_240: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_821);  sub_239 = unsqueeze_821 = None
    mul_1039: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_827);  sub_240 = unsqueeze_827 = None
    mul_1040: "f32[512]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_214);  sum_68 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_1039, relu_67, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1039 = primals_214 = None
    getitem_330: "f32[8, 256, 14, 14]" = convolution_backward_34[0]
    getitem_331: "f32[512, 256, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_201: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_67);  relu_67 = None
    alias_202: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_201);  alias_201 = None
    le_33: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_202, 0);  alias_202 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_33: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_33, scalar_tensor_33, getitem_330);  le_33 = scalar_tensor_33 = getitem_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_828: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
    unsqueeze_829: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 2);  unsqueeze_828 = None
    unsqueeze_830: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 3);  unsqueeze_829 = None
    sum_69: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_241: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_830)
    mul_1041: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_241);  sub_241 = None
    sum_70: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1041, [0, 2, 3]);  mul_1041 = None
    mul_1042: "f32[256]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    unsqueeze_831: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1042, 0);  mul_1042 = None
    unsqueeze_832: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 2);  unsqueeze_831 = None
    unsqueeze_833: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 3);  unsqueeze_832 = None
    mul_1043: "f32[256]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    mul_1044: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_1045: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1043, mul_1044);  mul_1043 = mul_1044 = None
    unsqueeze_834: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1045, 0);  mul_1045 = None
    unsqueeze_835: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 2);  unsqueeze_834 = None
    unsqueeze_836: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 3);  unsqueeze_835 = None
    mul_1046: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_212);  primals_212 = None
    unsqueeze_837: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
    unsqueeze_838: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 2);  unsqueeze_837 = None
    unsqueeze_839: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 3);  unsqueeze_838 = None
    sub_242: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_830);  convolution_70 = unsqueeze_830 = None
    mul_1047: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_836);  sub_242 = unsqueeze_836 = None
    sub_243: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_33, mul_1047);  where_33 = mul_1047 = None
    sub_244: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_833);  sub_243 = unsqueeze_833 = None
    mul_1048: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_839);  sub_244 = unsqueeze_839 = None
    mul_1049: "f32[256]" = torch.ops.aten.mul.Tensor(sum_70, squeeze_211);  sum_70 = squeeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_1048, relu_66, primals_211, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1048 = primals_211 = None
    getitem_333: "f32[8, 256, 14, 14]" = convolution_backward_35[0]
    getitem_334: "f32[256, 256, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_204: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_66);  relu_66 = None
    alias_205: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_204);  alias_204 = None
    le_34: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_205, 0);  alias_205 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_34: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_34, scalar_tensor_34, getitem_333);  le_34 = scalar_tensor_34 = getitem_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_840: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
    unsqueeze_841: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 2);  unsqueeze_840 = None
    unsqueeze_842: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 3);  unsqueeze_841 = None
    sum_71: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_245: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_842)
    mul_1050: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_245);  sub_245 = None
    sum_72: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1050, [0, 2, 3]);  mul_1050 = None
    mul_1051: "f32[256]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    unsqueeze_843: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1051, 0);  mul_1051 = None
    unsqueeze_844: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 2);  unsqueeze_843 = None
    unsqueeze_845: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 3);  unsqueeze_844 = None
    mul_1052: "f32[256]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    mul_1053: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_1054: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1052, mul_1053);  mul_1052 = mul_1053 = None
    unsqueeze_846: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1054, 0);  mul_1054 = None
    unsqueeze_847: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 2);  unsqueeze_846 = None
    unsqueeze_848: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 3);  unsqueeze_847 = None
    mul_1055: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_209);  primals_209 = None
    unsqueeze_849: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1055, 0);  mul_1055 = None
    unsqueeze_850: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 2);  unsqueeze_849 = None
    unsqueeze_851: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 3);  unsqueeze_850 = None
    sub_246: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_842);  convolution_69 = unsqueeze_842 = None
    mul_1056: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_848);  sub_246 = unsqueeze_848 = None
    sub_247: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_34, mul_1056);  where_34 = mul_1056 = None
    sub_248: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_845);  sub_247 = unsqueeze_845 = None
    mul_1057: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_851);  sub_248 = unsqueeze_851 = None
    mul_1058: "f32[256]" = torch.ops.aten.mul.Tensor(sum_72, squeeze_208);  sum_72 = squeeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_1057, relu_65, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1057 = primals_208 = None
    getitem_336: "f32[8, 512, 14, 14]" = convolution_backward_36[0]
    getitem_337: "f32[256, 512, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_591: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_590, getitem_336);  add_590 = getitem_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_207: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_65);  relu_65 = None
    alias_208: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_207);  alias_207 = None
    le_35: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_208, 0);  alias_208 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_35: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_35, scalar_tensor_35, add_591);  le_35 = scalar_tensor_35 = add_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_852: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
    unsqueeze_853: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 2);  unsqueeze_852 = None
    unsqueeze_854: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 3);  unsqueeze_853 = None
    sum_73: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_249: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_854)
    mul_1059: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_249);  sub_249 = None
    sum_74: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1059, [0, 2, 3]);  mul_1059 = None
    mul_1060: "f32[512]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    unsqueeze_855: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1060, 0);  mul_1060 = None
    unsqueeze_856: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 2);  unsqueeze_855 = None
    unsqueeze_857: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 3);  unsqueeze_856 = None
    mul_1061: "f32[512]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    mul_1062: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_1063: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1061, mul_1062);  mul_1061 = mul_1062 = None
    unsqueeze_858: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1063, 0);  mul_1063 = None
    unsqueeze_859: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 2);  unsqueeze_858 = None
    unsqueeze_860: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 3);  unsqueeze_859 = None
    mul_1064: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_206);  primals_206 = None
    unsqueeze_861: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
    unsqueeze_862: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 2);  unsqueeze_861 = None
    unsqueeze_863: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 3);  unsqueeze_862 = None
    sub_250: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_854);  convolution_68 = unsqueeze_854 = None
    mul_1065: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_860);  sub_250 = unsqueeze_860 = None
    sub_251: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_1065);  mul_1065 = None
    sub_252: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_857);  sub_251 = unsqueeze_857 = None
    mul_1066: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_863);  sub_252 = unsqueeze_863 = None
    mul_1067: "f32[512]" = torch.ops.aten.mul.Tensor(sum_74, squeeze_205);  sum_74 = squeeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_1066, cat_8, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1066 = cat_8 = primals_205 = None
    getitem_339: "f32[8, 2048, 14, 14]" = convolution_backward_37[0]
    getitem_340: "f32[512, 2048, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    slice_17: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_339, 1, 0, 512)
    slice_18: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_339, 1, 512, 1024)
    slice_19: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_339, 1, 1024, 1536)
    slice_20: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_339, 1, 1536, 2048);  getitem_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_592: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_35, slice_17);  where_35 = slice_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_210: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_64);  relu_64 = None
    alias_211: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_210);  alias_210 = None
    le_36: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_211, 0);  alias_211 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_36: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_36, scalar_tensor_36, add_592);  le_36 = scalar_tensor_36 = add_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_593: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_18, where_36);  slice_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_864: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
    unsqueeze_865: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 2);  unsqueeze_864 = None
    unsqueeze_866: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 3);  unsqueeze_865 = None
    sum_75: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_253: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_866)
    mul_1068: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_253);  sub_253 = None
    sum_76: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1068, [0, 2, 3]);  mul_1068 = None
    mul_1069: "f32[512]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    unsqueeze_867: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1069, 0);  mul_1069 = None
    unsqueeze_868: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 2);  unsqueeze_867 = None
    unsqueeze_869: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 3);  unsqueeze_868 = None
    mul_1070: "f32[512]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    mul_1071: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_1072: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1070, mul_1071);  mul_1070 = mul_1071 = None
    unsqueeze_870: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1072, 0);  mul_1072 = None
    unsqueeze_871: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 2);  unsqueeze_870 = None
    unsqueeze_872: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 3);  unsqueeze_871 = None
    mul_1073: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_203);  primals_203 = None
    unsqueeze_873: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
    unsqueeze_874: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 2);  unsqueeze_873 = None
    unsqueeze_875: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 3);  unsqueeze_874 = None
    sub_254: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_866);  convolution_67 = unsqueeze_866 = None
    mul_1074: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_872);  sub_254 = unsqueeze_872 = None
    sub_255: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_36, mul_1074);  where_36 = mul_1074 = None
    sub_256: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_869);  sub_255 = unsqueeze_869 = None
    mul_1075: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_875);  sub_256 = unsqueeze_875 = None
    mul_1076: "f32[512]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_202);  sum_76 = squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_1075, relu_63, primals_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1075 = primals_202 = None
    getitem_342: "f32[8, 256, 14, 14]" = convolution_backward_38[0]
    getitem_343: "f32[512, 256, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_213: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_63);  relu_63 = None
    alias_214: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_213);  alias_213 = None
    le_37: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_214, 0);  alias_214 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_37: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_37, scalar_tensor_37, getitem_342);  le_37 = scalar_tensor_37 = getitem_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_876: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
    unsqueeze_877: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 2);  unsqueeze_876 = None
    unsqueeze_878: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 3);  unsqueeze_877 = None
    sum_77: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_257: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_878)
    mul_1077: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_257);  sub_257 = None
    sum_78: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1077, [0, 2, 3]);  mul_1077 = None
    mul_1078: "f32[256]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    unsqueeze_879: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1078, 0);  mul_1078 = None
    unsqueeze_880: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 2);  unsqueeze_879 = None
    unsqueeze_881: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 3);  unsqueeze_880 = None
    mul_1079: "f32[256]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    mul_1080: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_1081: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1079, mul_1080);  mul_1079 = mul_1080 = None
    unsqueeze_882: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1081, 0);  mul_1081 = None
    unsqueeze_883: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 2);  unsqueeze_882 = None
    unsqueeze_884: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 3);  unsqueeze_883 = None
    mul_1082: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_200);  primals_200 = None
    unsqueeze_885: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
    unsqueeze_886: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 2);  unsqueeze_885 = None
    unsqueeze_887: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 3);  unsqueeze_886 = None
    sub_258: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_878);  convolution_66 = unsqueeze_878 = None
    mul_1083: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_884);  sub_258 = unsqueeze_884 = None
    sub_259: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_37, mul_1083);  where_37 = mul_1083 = None
    sub_260: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_881);  sub_259 = unsqueeze_881 = None
    mul_1084: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_887);  sub_260 = unsqueeze_887 = None
    mul_1085: "f32[256]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_199);  sum_78 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_1084, relu_62, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1084 = primals_199 = None
    getitem_345: "f32[8, 256, 14, 14]" = convolution_backward_39[0]
    getitem_346: "f32[256, 256, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_216: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_62);  relu_62 = None
    alias_217: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_216);  alias_216 = None
    le_38: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_217, 0);  alias_217 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_38: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_38, scalar_tensor_38, getitem_345);  le_38 = scalar_tensor_38 = getitem_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_888: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
    unsqueeze_889: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 2);  unsqueeze_888 = None
    unsqueeze_890: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 3);  unsqueeze_889 = None
    sum_79: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_261: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_890)
    mul_1086: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_261);  sub_261 = None
    sum_80: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1086, [0, 2, 3]);  mul_1086 = None
    mul_1087: "f32[256]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    unsqueeze_891: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1087, 0);  mul_1087 = None
    unsqueeze_892: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 2);  unsqueeze_891 = None
    unsqueeze_893: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 3);  unsqueeze_892 = None
    mul_1088: "f32[256]" = torch.ops.aten.mul.Tensor(sum_80, 0.0006377551020408163)
    mul_1089: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_1090: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1088, mul_1089);  mul_1088 = mul_1089 = None
    unsqueeze_894: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1090, 0);  mul_1090 = None
    unsqueeze_895: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 2);  unsqueeze_894 = None
    unsqueeze_896: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 3);  unsqueeze_895 = None
    mul_1091: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_197);  primals_197 = None
    unsqueeze_897: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1091, 0);  mul_1091 = None
    unsqueeze_898: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 2);  unsqueeze_897 = None
    unsqueeze_899: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 3);  unsqueeze_898 = None
    sub_262: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_890);  convolution_65 = unsqueeze_890 = None
    mul_1092: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_896);  sub_262 = unsqueeze_896 = None
    sub_263: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_38, mul_1092);  where_38 = mul_1092 = None
    sub_264: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_893);  sub_263 = unsqueeze_893 = None
    mul_1093: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_899);  sub_264 = unsqueeze_899 = None
    mul_1094: "f32[256]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_196);  sum_80 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_1093, relu_61, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1093 = primals_196 = None
    getitem_348: "f32[8, 512, 14, 14]" = convolution_backward_40[0]
    getitem_349: "f32[256, 512, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_594: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_593, getitem_348);  add_593 = getitem_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_219: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_61);  relu_61 = None
    alias_220: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_219);  alias_219 = None
    le_39: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_220, 0);  alias_220 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_39: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_39, scalar_tensor_39, add_594);  le_39 = scalar_tensor_39 = add_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_595: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_20, where_39);  slice_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_900: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_901: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 2);  unsqueeze_900 = None
    unsqueeze_902: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 3);  unsqueeze_901 = None
    sum_81: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_265: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_902)
    mul_1095: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_265);  sub_265 = None
    sum_82: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1095, [0, 2, 3]);  mul_1095 = None
    mul_1096: "f32[512]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    unsqueeze_903: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1096, 0);  mul_1096 = None
    unsqueeze_904: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 2);  unsqueeze_903 = None
    unsqueeze_905: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 3);  unsqueeze_904 = None
    mul_1097: "f32[512]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    mul_1098: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_1099: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1097, mul_1098);  mul_1097 = mul_1098 = None
    unsqueeze_906: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1099, 0);  mul_1099 = None
    unsqueeze_907: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 2);  unsqueeze_906 = None
    unsqueeze_908: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 3);  unsqueeze_907 = None
    mul_1100: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_194);  primals_194 = None
    unsqueeze_909: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1100, 0);  mul_1100 = None
    unsqueeze_910: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 2);  unsqueeze_909 = None
    unsqueeze_911: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 3);  unsqueeze_910 = None
    sub_266: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_902);  convolution_64 = unsqueeze_902 = None
    mul_1101: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_908);  sub_266 = unsqueeze_908 = None
    sub_267: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_39, mul_1101);  where_39 = mul_1101 = None
    sub_268: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_905);  sub_267 = unsqueeze_905 = None
    mul_1102: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_911);  sub_268 = unsqueeze_911 = None
    mul_1103: "f32[512]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_193);  sum_82 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_1102, relu_60, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1102 = primals_193 = None
    getitem_351: "f32[8, 256, 14, 14]" = convolution_backward_41[0]
    getitem_352: "f32[512, 256, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_222: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_60);  relu_60 = None
    alias_223: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_222);  alias_222 = None
    le_40: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_223, 0);  alias_223 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_40: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_40, scalar_tensor_40, getitem_351);  le_40 = scalar_tensor_40 = getitem_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_912: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_913: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 2);  unsqueeze_912 = None
    unsqueeze_914: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 3);  unsqueeze_913 = None
    sum_83: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_269: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_914)
    mul_1104: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_269);  sub_269 = None
    sum_84: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1104, [0, 2, 3]);  mul_1104 = None
    mul_1105: "f32[256]" = torch.ops.aten.mul.Tensor(sum_83, 0.0006377551020408163)
    unsqueeze_915: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1105, 0);  mul_1105 = None
    unsqueeze_916: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 2);  unsqueeze_915 = None
    unsqueeze_917: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 3);  unsqueeze_916 = None
    mul_1106: "f32[256]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    mul_1107: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_1108: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1106, mul_1107);  mul_1106 = mul_1107 = None
    unsqueeze_918: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1108, 0);  mul_1108 = None
    unsqueeze_919: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 2);  unsqueeze_918 = None
    unsqueeze_920: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 3);  unsqueeze_919 = None
    mul_1109: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_191);  primals_191 = None
    unsqueeze_921: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1109, 0);  mul_1109 = None
    unsqueeze_922: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 2);  unsqueeze_921 = None
    unsqueeze_923: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 3);  unsqueeze_922 = None
    sub_270: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_914);  convolution_63 = unsqueeze_914 = None
    mul_1110: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_920);  sub_270 = unsqueeze_920 = None
    sub_271: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_1110);  where_40 = mul_1110 = None
    sub_272: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_917);  sub_271 = unsqueeze_917 = None
    mul_1111: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_923);  sub_272 = unsqueeze_923 = None
    mul_1112: "f32[256]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_190);  sum_84 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_1111, relu_59, primals_190, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1111 = primals_190 = None
    getitem_354: "f32[8, 256, 14, 14]" = convolution_backward_42[0]
    getitem_355: "f32[256, 256, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_225: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_59);  relu_59 = None
    alias_226: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_225);  alias_225 = None
    le_41: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_226, 0);  alias_226 = None
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_41: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_41, scalar_tensor_41, getitem_354);  le_41 = scalar_tensor_41 = getitem_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_924: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_925: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 2);  unsqueeze_924 = None
    unsqueeze_926: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 3);  unsqueeze_925 = None
    sum_85: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_273: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_926)
    mul_1113: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, sub_273);  sub_273 = None
    sum_86: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1113, [0, 2, 3]);  mul_1113 = None
    mul_1114: "f32[256]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    unsqueeze_927: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1114, 0);  mul_1114 = None
    unsqueeze_928: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 2);  unsqueeze_927 = None
    unsqueeze_929: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 3);  unsqueeze_928 = None
    mul_1115: "f32[256]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006377551020408163)
    mul_1116: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_1117: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1115, mul_1116);  mul_1115 = mul_1116 = None
    unsqueeze_930: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1117, 0);  mul_1117 = None
    unsqueeze_931: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 2);  unsqueeze_930 = None
    unsqueeze_932: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 3);  unsqueeze_931 = None
    mul_1118: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_188);  primals_188 = None
    unsqueeze_933: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1118, 0);  mul_1118 = None
    unsqueeze_934: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 2);  unsqueeze_933 = None
    unsqueeze_935: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 3);  unsqueeze_934 = None
    sub_274: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_926);  convolution_62 = unsqueeze_926 = None
    mul_1119: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_932);  sub_274 = unsqueeze_932 = None
    sub_275: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_41, mul_1119);  where_41 = mul_1119 = None
    sub_276: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_929);  sub_275 = unsqueeze_929 = None
    mul_1120: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_935);  sub_276 = unsqueeze_935 = None
    mul_1121: "f32[256]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_187);  sum_86 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_1120, relu_58, primals_187, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1120 = primals_187 = None
    getitem_357: "f32[8, 512, 14, 14]" = convolution_backward_43[0]
    getitem_358: "f32[256, 512, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_596: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_595, getitem_357);  add_595 = getitem_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_228: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_58);  relu_58 = None
    alias_229: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_228);  alias_228 = None
    le_42: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_229, 0);  alias_229 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_42: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_42, scalar_tensor_42, add_596);  le_42 = scalar_tensor_42 = add_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_936: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_937: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 2);  unsqueeze_936 = None
    unsqueeze_938: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 3);  unsqueeze_937 = None
    sum_87: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_277: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_938)
    mul_1122: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_277);  sub_277 = None
    sum_88: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1122, [0, 2, 3]);  mul_1122 = None
    mul_1123: "f32[512]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006377551020408163)
    unsqueeze_939: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1123, 0);  mul_1123 = None
    unsqueeze_940: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 2);  unsqueeze_939 = None
    unsqueeze_941: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 3);  unsqueeze_940 = None
    mul_1124: "f32[512]" = torch.ops.aten.mul.Tensor(sum_88, 0.0006377551020408163)
    mul_1125: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_1126: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1124, mul_1125);  mul_1124 = mul_1125 = None
    unsqueeze_942: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1126, 0);  mul_1126 = None
    unsqueeze_943: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 2);  unsqueeze_942 = None
    unsqueeze_944: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 3);  unsqueeze_943 = None
    mul_1127: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_185);  primals_185 = None
    unsqueeze_945: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1127, 0);  mul_1127 = None
    unsqueeze_946: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 2);  unsqueeze_945 = None
    unsqueeze_947: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 3);  unsqueeze_946 = None
    sub_278: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_938);  convolution_61 = unsqueeze_938 = None
    mul_1128: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_944);  sub_278 = unsqueeze_944 = None
    sub_279: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_42, mul_1128);  mul_1128 = None
    sub_280: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_941);  sub_279 = unsqueeze_941 = None
    mul_1129: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_947);  sub_280 = unsqueeze_947 = None
    mul_1130: "f32[512]" = torch.ops.aten.mul.Tensor(sum_88, squeeze_184);  sum_88 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_1129, cat_7, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1129 = cat_7 = primals_184 = None
    getitem_360: "f32[8, 1024, 14, 14]" = convolution_backward_44[0]
    getitem_361: "f32[512, 1024, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    slice_21: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_360, 1, 0, 512)
    slice_22: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_360, 1, 512, 1024);  getitem_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_597: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_42, slice_21);  where_42 = slice_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_231: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_57);  relu_57 = None
    alias_232: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_231);  alias_231 = None
    le_43: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_232, 0);  alias_232 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_43: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_43, scalar_tensor_43, add_597);  le_43 = scalar_tensor_43 = add_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_598: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_22, where_43);  slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_948: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_949: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 2);  unsqueeze_948 = None
    unsqueeze_950: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 3);  unsqueeze_949 = None
    sum_89: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_281: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_950)
    mul_1131: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_281);  sub_281 = None
    sum_90: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1131, [0, 2, 3]);  mul_1131 = None
    mul_1132: "f32[512]" = torch.ops.aten.mul.Tensor(sum_89, 0.0006377551020408163)
    unsqueeze_951: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1132, 0);  mul_1132 = None
    unsqueeze_952: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 2);  unsqueeze_951 = None
    unsqueeze_953: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 3);  unsqueeze_952 = None
    mul_1133: "f32[512]" = torch.ops.aten.mul.Tensor(sum_90, 0.0006377551020408163)
    mul_1134: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_1135: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1133, mul_1134);  mul_1133 = mul_1134 = None
    unsqueeze_954: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1135, 0);  mul_1135 = None
    unsqueeze_955: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 2);  unsqueeze_954 = None
    unsqueeze_956: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 3);  unsqueeze_955 = None
    mul_1136: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_182);  primals_182 = None
    unsqueeze_957: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1136, 0);  mul_1136 = None
    unsqueeze_958: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 2);  unsqueeze_957 = None
    unsqueeze_959: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 3);  unsqueeze_958 = None
    sub_282: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_950);  convolution_60 = unsqueeze_950 = None
    mul_1137: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_956);  sub_282 = unsqueeze_956 = None
    sub_283: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_43, mul_1137);  where_43 = mul_1137 = None
    sub_284: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_953);  sub_283 = unsqueeze_953 = None
    mul_1138: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_959);  sub_284 = unsqueeze_959 = None
    mul_1139: "f32[512]" = torch.ops.aten.mul.Tensor(sum_90, squeeze_181);  sum_90 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1138, relu_56, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1138 = primals_181 = None
    getitem_363: "f32[8, 256, 14, 14]" = convolution_backward_45[0]
    getitem_364: "f32[512, 256, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_234: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_56);  relu_56 = None
    alias_235: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_234);  alias_234 = None
    le_44: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_235, 0);  alias_235 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_44: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_44, scalar_tensor_44, getitem_363);  le_44 = scalar_tensor_44 = getitem_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_960: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_961: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 2);  unsqueeze_960 = None
    unsqueeze_962: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 3);  unsqueeze_961 = None
    sum_91: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_285: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_962)
    mul_1140: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_44, sub_285);  sub_285 = None
    sum_92: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1140, [0, 2, 3]);  mul_1140 = None
    mul_1141: "f32[256]" = torch.ops.aten.mul.Tensor(sum_91, 0.0006377551020408163)
    unsqueeze_963: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1141, 0);  mul_1141 = None
    unsqueeze_964: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 2);  unsqueeze_963 = None
    unsqueeze_965: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 3);  unsqueeze_964 = None
    mul_1142: "f32[256]" = torch.ops.aten.mul.Tensor(sum_92, 0.0006377551020408163)
    mul_1143: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_1144: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1142, mul_1143);  mul_1142 = mul_1143 = None
    unsqueeze_966: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1144, 0);  mul_1144 = None
    unsqueeze_967: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 2);  unsqueeze_966 = None
    unsqueeze_968: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 3);  unsqueeze_967 = None
    mul_1145: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_179);  primals_179 = None
    unsqueeze_969: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1145, 0);  mul_1145 = None
    unsqueeze_970: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 2);  unsqueeze_969 = None
    unsqueeze_971: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 3);  unsqueeze_970 = None
    sub_286: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_962);  convolution_59 = unsqueeze_962 = None
    mul_1146: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_968);  sub_286 = unsqueeze_968 = None
    sub_287: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_44, mul_1146);  where_44 = mul_1146 = None
    sub_288: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_965);  sub_287 = unsqueeze_965 = None
    mul_1147: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_971);  sub_288 = unsqueeze_971 = None
    mul_1148: "f32[256]" = torch.ops.aten.mul.Tensor(sum_92, squeeze_178);  sum_92 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1147, relu_55, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1147 = primals_178 = None
    getitem_366: "f32[8, 256, 14, 14]" = convolution_backward_46[0]
    getitem_367: "f32[256, 256, 3, 3]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_237: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_55);  relu_55 = None
    alias_238: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_237);  alias_237 = None
    le_45: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_238, 0);  alias_238 = None
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_45: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_45, scalar_tensor_45, getitem_366);  le_45 = scalar_tensor_45 = getitem_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_972: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_973: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 2);  unsqueeze_972 = None
    unsqueeze_974: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 3);  unsqueeze_973 = None
    sum_93: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_289: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_974)
    mul_1149: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, sub_289);  sub_289 = None
    sum_94: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1149, [0, 2, 3]);  mul_1149 = None
    mul_1150: "f32[256]" = torch.ops.aten.mul.Tensor(sum_93, 0.0006377551020408163)
    unsqueeze_975: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1150, 0);  mul_1150 = None
    unsqueeze_976: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 2);  unsqueeze_975 = None
    unsqueeze_977: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 3);  unsqueeze_976 = None
    mul_1151: "f32[256]" = torch.ops.aten.mul.Tensor(sum_94, 0.0006377551020408163)
    mul_1152: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_1153: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1151, mul_1152);  mul_1151 = mul_1152 = None
    unsqueeze_978: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1153, 0);  mul_1153 = None
    unsqueeze_979: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 2);  unsqueeze_978 = None
    unsqueeze_980: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 3);  unsqueeze_979 = None
    mul_1154: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_176);  primals_176 = None
    unsqueeze_981: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1154, 0);  mul_1154 = None
    unsqueeze_982: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 2);  unsqueeze_981 = None
    unsqueeze_983: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 3);  unsqueeze_982 = None
    sub_290: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_974);  convolution_58 = unsqueeze_974 = None
    mul_1155: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_980);  sub_290 = unsqueeze_980 = None
    sub_291: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_45, mul_1155);  where_45 = mul_1155 = None
    sub_292: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_977);  sub_291 = unsqueeze_977 = None
    mul_1156: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_983);  sub_292 = unsqueeze_983 = None
    mul_1157: "f32[256]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_175);  sum_94 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1156, relu_54, primals_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1156 = primals_175 = None
    getitem_369: "f32[8, 512, 14, 14]" = convolution_backward_47[0]
    getitem_370: "f32[256, 512, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_599: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_598, getitem_369);  add_598 = getitem_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_240: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_54);  relu_54 = None
    alias_241: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_240);  alias_240 = None
    le_46: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_241, 0);  alias_241 = None
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_46: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_46, scalar_tensor_46, add_599);  le_46 = scalar_tensor_46 = add_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_600: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_19, where_46);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_984: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_985: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 2);  unsqueeze_984 = None
    unsqueeze_986: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 3);  unsqueeze_985 = None
    sum_95: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_293: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_986)
    mul_1158: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_46, sub_293);  sub_293 = None
    sum_96: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1158, [0, 2, 3]);  mul_1158 = None
    mul_1159: "f32[512]" = torch.ops.aten.mul.Tensor(sum_95, 0.0006377551020408163)
    unsqueeze_987: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1159, 0);  mul_1159 = None
    unsqueeze_988: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 2);  unsqueeze_987 = None
    unsqueeze_989: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 3);  unsqueeze_988 = None
    mul_1160: "f32[512]" = torch.ops.aten.mul.Tensor(sum_96, 0.0006377551020408163)
    mul_1161: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_1162: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1160, mul_1161);  mul_1160 = mul_1161 = None
    unsqueeze_990: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1162, 0);  mul_1162 = None
    unsqueeze_991: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 2);  unsqueeze_990 = None
    unsqueeze_992: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 3);  unsqueeze_991 = None
    mul_1163: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_173);  primals_173 = None
    unsqueeze_993: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1163, 0);  mul_1163 = None
    unsqueeze_994: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 2);  unsqueeze_993 = None
    unsqueeze_995: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 3);  unsqueeze_994 = None
    sub_294: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_986);  convolution_57 = unsqueeze_986 = None
    mul_1164: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_992);  sub_294 = unsqueeze_992 = None
    sub_295: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_46, mul_1164);  where_46 = mul_1164 = None
    sub_296: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_989);  sub_295 = unsqueeze_989 = None
    mul_1165: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_995);  sub_296 = unsqueeze_995 = None
    mul_1166: "f32[512]" = torch.ops.aten.mul.Tensor(sum_96, squeeze_172);  sum_96 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1165, relu_53, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1165 = primals_172 = None
    getitem_372: "f32[8, 256, 14, 14]" = convolution_backward_48[0]
    getitem_373: "f32[512, 256, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_243: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_53);  relu_53 = None
    alias_244: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_243);  alias_243 = None
    le_47: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_244, 0);  alias_244 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_47: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_47, scalar_tensor_47, getitem_372);  le_47 = scalar_tensor_47 = getitem_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_996: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_997: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 2);  unsqueeze_996 = None
    unsqueeze_998: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 3);  unsqueeze_997 = None
    sum_97: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_297: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_998)
    mul_1167: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, sub_297);  sub_297 = None
    sum_98: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1167, [0, 2, 3]);  mul_1167 = None
    mul_1168: "f32[256]" = torch.ops.aten.mul.Tensor(sum_97, 0.0006377551020408163)
    unsqueeze_999: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1168, 0);  mul_1168 = None
    unsqueeze_1000: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 2);  unsqueeze_999 = None
    unsqueeze_1001: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 3);  unsqueeze_1000 = None
    mul_1169: "f32[256]" = torch.ops.aten.mul.Tensor(sum_98, 0.0006377551020408163)
    mul_1170: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_1171: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1169, mul_1170);  mul_1169 = mul_1170 = None
    unsqueeze_1002: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1171, 0);  mul_1171 = None
    unsqueeze_1003: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 2);  unsqueeze_1002 = None
    unsqueeze_1004: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 3);  unsqueeze_1003 = None
    mul_1172: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_170);  primals_170 = None
    unsqueeze_1005: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1172, 0);  mul_1172 = None
    unsqueeze_1006: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 2);  unsqueeze_1005 = None
    unsqueeze_1007: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 3);  unsqueeze_1006 = None
    sub_298: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_998);  convolution_56 = unsqueeze_998 = None
    mul_1173: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_1004);  sub_298 = unsqueeze_1004 = None
    sub_299: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_47, mul_1173);  where_47 = mul_1173 = None
    sub_300: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_299, unsqueeze_1001);  sub_299 = unsqueeze_1001 = None
    mul_1174: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_1007);  sub_300 = unsqueeze_1007 = None
    mul_1175: "f32[256]" = torch.ops.aten.mul.Tensor(sum_98, squeeze_169);  sum_98 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1174, relu_52, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1174 = primals_169 = None
    getitem_375: "f32[8, 256, 14, 14]" = convolution_backward_49[0]
    getitem_376: "f32[256, 256, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_246: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_52);  relu_52 = None
    alias_247: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_246);  alias_246 = None
    le_48: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_247, 0);  alias_247 = None
    scalar_tensor_48: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_48: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_48, scalar_tensor_48, getitem_375);  le_48 = scalar_tensor_48 = getitem_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1008: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_1009: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 2);  unsqueeze_1008 = None
    unsqueeze_1010: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 3);  unsqueeze_1009 = None
    sum_99: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_301: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_1010)
    mul_1176: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_48, sub_301);  sub_301 = None
    sum_100: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1176, [0, 2, 3]);  mul_1176 = None
    mul_1177: "f32[256]" = torch.ops.aten.mul.Tensor(sum_99, 0.0006377551020408163)
    unsqueeze_1011: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1177, 0);  mul_1177 = None
    unsqueeze_1012: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 2);  unsqueeze_1011 = None
    unsqueeze_1013: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 3);  unsqueeze_1012 = None
    mul_1178: "f32[256]" = torch.ops.aten.mul.Tensor(sum_100, 0.0006377551020408163)
    mul_1179: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_1180: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1178, mul_1179);  mul_1178 = mul_1179 = None
    unsqueeze_1014: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1180, 0);  mul_1180 = None
    unsqueeze_1015: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 2);  unsqueeze_1014 = None
    unsqueeze_1016: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 3);  unsqueeze_1015 = None
    mul_1181: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_167);  primals_167 = None
    unsqueeze_1017: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1181, 0);  mul_1181 = None
    unsqueeze_1018: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 2);  unsqueeze_1017 = None
    unsqueeze_1019: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 3);  unsqueeze_1018 = None
    sub_302: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_1010);  convolution_55 = unsqueeze_1010 = None
    mul_1182: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_1016);  sub_302 = unsqueeze_1016 = None
    sub_303: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_48, mul_1182);  where_48 = mul_1182 = None
    sub_304: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_1013);  sub_303 = unsqueeze_1013 = None
    mul_1183: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_1019);  sub_304 = unsqueeze_1019 = None
    mul_1184: "f32[256]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_166);  sum_100 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1183, relu_51, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1183 = primals_166 = None
    getitem_378: "f32[8, 512, 14, 14]" = convolution_backward_50[0]
    getitem_379: "f32[256, 512, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_601: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_600, getitem_378);  add_600 = getitem_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_249: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_51);  relu_51 = None
    alias_250: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_249);  alias_249 = None
    le_49: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_250, 0);  alias_250 = None
    scalar_tensor_49: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_49: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_49, scalar_tensor_49, add_601);  le_49 = scalar_tensor_49 = add_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_1020: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_1021: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 2);  unsqueeze_1020 = None
    unsqueeze_1022: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 3);  unsqueeze_1021 = None
    sum_101: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_305: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_1022)
    mul_1185: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_49, sub_305);  sub_305 = None
    sum_102: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1185, [0, 2, 3]);  mul_1185 = None
    mul_1186: "f32[512]" = torch.ops.aten.mul.Tensor(sum_101, 0.0006377551020408163)
    unsqueeze_1023: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1186, 0);  mul_1186 = None
    unsqueeze_1024: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 2);  unsqueeze_1023 = None
    unsqueeze_1025: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 3);  unsqueeze_1024 = None
    mul_1187: "f32[512]" = torch.ops.aten.mul.Tensor(sum_102, 0.0006377551020408163)
    mul_1188: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_1189: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1187, mul_1188);  mul_1187 = mul_1188 = None
    unsqueeze_1026: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1189, 0);  mul_1189 = None
    unsqueeze_1027: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 2);  unsqueeze_1026 = None
    unsqueeze_1028: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 3);  unsqueeze_1027 = None
    mul_1190: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_164);  primals_164 = None
    unsqueeze_1029: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1190, 0);  mul_1190 = None
    unsqueeze_1030: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 2);  unsqueeze_1029 = None
    unsqueeze_1031: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 3);  unsqueeze_1030 = None
    sub_306: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_1022);  convolution_54 = unsqueeze_1022 = None
    mul_1191: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_1028);  sub_306 = unsqueeze_1028 = None
    sub_307: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_49, mul_1191);  mul_1191 = None
    sub_308: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_307, unsqueeze_1025);  sub_307 = unsqueeze_1025 = None
    mul_1192: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1031);  sub_308 = unsqueeze_1031 = None
    mul_1193: "f32[512]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_163);  sum_102 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1192, cat_6, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1192 = cat_6 = primals_163 = None
    getitem_381: "f32[8, 1536, 14, 14]" = convolution_backward_51[0]
    getitem_382: "f32[512, 1536, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    slice_23: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_381, 1, 0, 512)
    slice_24: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_381, 1, 512, 1024)
    slice_25: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_381, 1, 1024, 1536);  getitem_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_602: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_49, slice_23);  where_49 = slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_252: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_50);  relu_50 = None
    alias_253: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_252);  alias_252 = None
    le_50: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_253, 0);  alias_253 = None
    scalar_tensor_50: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_50: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_50, scalar_tensor_50, add_602);  le_50 = scalar_tensor_50 = add_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_603: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_24, where_50);  slice_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1032: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_1033: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 2);  unsqueeze_1032 = None
    unsqueeze_1034: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 3);  unsqueeze_1033 = None
    sum_103: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_309: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_1034)
    mul_1194: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_50, sub_309);  sub_309 = None
    sum_104: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1194, [0, 2, 3]);  mul_1194 = None
    mul_1195: "f32[512]" = torch.ops.aten.mul.Tensor(sum_103, 0.0006377551020408163)
    unsqueeze_1035: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1195, 0);  mul_1195 = None
    unsqueeze_1036: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 2);  unsqueeze_1035 = None
    unsqueeze_1037: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 3);  unsqueeze_1036 = None
    mul_1196: "f32[512]" = torch.ops.aten.mul.Tensor(sum_104, 0.0006377551020408163)
    mul_1197: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_1198: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1196, mul_1197);  mul_1196 = mul_1197 = None
    unsqueeze_1038: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1198, 0);  mul_1198 = None
    unsqueeze_1039: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 2);  unsqueeze_1038 = None
    unsqueeze_1040: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 3);  unsqueeze_1039 = None
    mul_1199: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_161);  primals_161 = None
    unsqueeze_1041: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1199, 0);  mul_1199 = None
    unsqueeze_1042: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 2);  unsqueeze_1041 = None
    unsqueeze_1043: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 3);  unsqueeze_1042 = None
    sub_310: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_1034);  convolution_53 = unsqueeze_1034 = None
    mul_1200: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_1040);  sub_310 = unsqueeze_1040 = None
    sub_311: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_50, mul_1200);  where_50 = mul_1200 = None
    sub_312: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_311, unsqueeze_1037);  sub_311 = unsqueeze_1037 = None
    mul_1201: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1043);  sub_312 = unsqueeze_1043 = None
    mul_1202: "f32[512]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_160);  sum_104 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1201, relu_49, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1201 = primals_160 = None
    getitem_384: "f32[8, 256, 14, 14]" = convolution_backward_52[0]
    getitem_385: "f32[512, 256, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_255: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_49);  relu_49 = None
    alias_256: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_255);  alias_255 = None
    le_51: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_256, 0);  alias_256 = None
    scalar_tensor_51: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_51: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_51, scalar_tensor_51, getitem_384);  le_51 = scalar_tensor_51 = getitem_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1044: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_1045: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, 2);  unsqueeze_1044 = None
    unsqueeze_1046: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 3);  unsqueeze_1045 = None
    sum_105: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_313: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_1046)
    mul_1203: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_51, sub_313);  sub_313 = None
    sum_106: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1203, [0, 2, 3]);  mul_1203 = None
    mul_1204: "f32[256]" = torch.ops.aten.mul.Tensor(sum_105, 0.0006377551020408163)
    unsqueeze_1047: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1204, 0);  mul_1204 = None
    unsqueeze_1048: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 2);  unsqueeze_1047 = None
    unsqueeze_1049: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 3);  unsqueeze_1048 = None
    mul_1205: "f32[256]" = torch.ops.aten.mul.Tensor(sum_106, 0.0006377551020408163)
    mul_1206: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_1207: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1205, mul_1206);  mul_1205 = mul_1206 = None
    unsqueeze_1050: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1207, 0);  mul_1207 = None
    unsqueeze_1051: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 2);  unsqueeze_1050 = None
    unsqueeze_1052: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 3);  unsqueeze_1051 = None
    mul_1208: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_158);  primals_158 = None
    unsqueeze_1053: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1208, 0);  mul_1208 = None
    unsqueeze_1054: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 2);  unsqueeze_1053 = None
    unsqueeze_1055: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 3);  unsqueeze_1054 = None
    sub_314: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_1046);  convolution_52 = unsqueeze_1046 = None
    mul_1209: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_1052);  sub_314 = unsqueeze_1052 = None
    sub_315: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_51, mul_1209);  where_51 = mul_1209 = None
    sub_316: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_315, unsqueeze_1049);  sub_315 = unsqueeze_1049 = None
    mul_1210: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1055);  sub_316 = unsqueeze_1055 = None
    mul_1211: "f32[256]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_157);  sum_106 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1210, relu_48, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1210 = primals_157 = None
    getitem_387: "f32[8, 256, 14, 14]" = convolution_backward_53[0]
    getitem_388: "f32[256, 256, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_258: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_48);  relu_48 = None
    alias_259: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_258);  alias_258 = None
    le_52: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_259, 0);  alias_259 = None
    scalar_tensor_52: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_52: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_52, scalar_tensor_52, getitem_387);  le_52 = scalar_tensor_52 = getitem_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1056: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_1057: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, 2);  unsqueeze_1056 = None
    unsqueeze_1058: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 3);  unsqueeze_1057 = None
    sum_107: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_317: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_1058)
    mul_1212: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_52, sub_317);  sub_317 = None
    sum_108: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1212, [0, 2, 3]);  mul_1212 = None
    mul_1213: "f32[256]" = torch.ops.aten.mul.Tensor(sum_107, 0.0006377551020408163)
    unsqueeze_1059: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1213, 0);  mul_1213 = None
    unsqueeze_1060: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 2);  unsqueeze_1059 = None
    unsqueeze_1061: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 3);  unsqueeze_1060 = None
    mul_1214: "f32[256]" = torch.ops.aten.mul.Tensor(sum_108, 0.0006377551020408163)
    mul_1215: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_1216: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1214, mul_1215);  mul_1214 = mul_1215 = None
    unsqueeze_1062: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1216, 0);  mul_1216 = None
    unsqueeze_1063: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 2);  unsqueeze_1062 = None
    unsqueeze_1064: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 3);  unsqueeze_1063 = None
    mul_1217: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_155);  primals_155 = None
    unsqueeze_1065: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1217, 0);  mul_1217 = None
    unsqueeze_1066: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 2);  unsqueeze_1065 = None
    unsqueeze_1067: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 3);  unsqueeze_1066 = None
    sub_318: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_1058);  convolution_51 = unsqueeze_1058 = None
    mul_1218: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_1064);  sub_318 = unsqueeze_1064 = None
    sub_319: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_52, mul_1218);  where_52 = mul_1218 = None
    sub_320: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_319, unsqueeze_1061);  sub_319 = unsqueeze_1061 = None
    mul_1219: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1067);  sub_320 = unsqueeze_1067 = None
    mul_1220: "f32[256]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_154);  sum_108 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1219, relu_47, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1219 = primals_154 = None
    getitem_390: "f32[8, 512, 14, 14]" = convolution_backward_54[0]
    getitem_391: "f32[256, 512, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_604: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_603, getitem_390);  add_603 = getitem_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_261: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_47);  relu_47 = None
    alias_262: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_261);  alias_261 = None
    le_53: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_262, 0);  alias_262 = None
    scalar_tensor_53: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_53: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_53, scalar_tensor_53, add_604);  le_53 = scalar_tensor_53 = add_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_605: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_25, where_53);  slice_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1068: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_1069: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, 2);  unsqueeze_1068 = None
    unsqueeze_1070: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 3);  unsqueeze_1069 = None
    sum_109: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_321: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_1070)
    mul_1221: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_53, sub_321);  sub_321 = None
    sum_110: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1221, [0, 2, 3]);  mul_1221 = None
    mul_1222: "f32[512]" = torch.ops.aten.mul.Tensor(sum_109, 0.0006377551020408163)
    unsqueeze_1071: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1222, 0);  mul_1222 = None
    unsqueeze_1072: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 2);  unsqueeze_1071 = None
    unsqueeze_1073: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 3);  unsqueeze_1072 = None
    mul_1223: "f32[512]" = torch.ops.aten.mul.Tensor(sum_110, 0.0006377551020408163)
    mul_1224: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_1225: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1223, mul_1224);  mul_1223 = mul_1224 = None
    unsqueeze_1074: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1225, 0);  mul_1225 = None
    unsqueeze_1075: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 2);  unsqueeze_1074 = None
    unsqueeze_1076: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 3);  unsqueeze_1075 = None
    mul_1226: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_152);  primals_152 = None
    unsqueeze_1077: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1226, 0);  mul_1226 = None
    unsqueeze_1078: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 2);  unsqueeze_1077 = None
    unsqueeze_1079: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 3);  unsqueeze_1078 = None
    sub_322: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_1070);  convolution_50 = unsqueeze_1070 = None
    mul_1227: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_1076);  sub_322 = unsqueeze_1076 = None
    sub_323: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_53, mul_1227);  where_53 = mul_1227 = None
    sub_324: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_1073);  sub_323 = unsqueeze_1073 = None
    mul_1228: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1079);  sub_324 = unsqueeze_1079 = None
    mul_1229: "f32[512]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_151);  sum_110 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1228, relu_46, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1228 = primals_151 = None
    getitem_393: "f32[8, 256, 14, 14]" = convolution_backward_55[0]
    getitem_394: "f32[512, 256, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_264: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_46);  relu_46 = None
    alias_265: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_264);  alias_264 = None
    le_54: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_265, 0);  alias_265 = None
    scalar_tensor_54: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_54: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_54, scalar_tensor_54, getitem_393);  le_54 = scalar_tensor_54 = getitem_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1080: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_1081: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, 2);  unsqueeze_1080 = None
    unsqueeze_1082: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1081, 3);  unsqueeze_1081 = None
    sum_111: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    sub_325: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_1082)
    mul_1230: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_54, sub_325);  sub_325 = None
    sum_112: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1230, [0, 2, 3]);  mul_1230 = None
    mul_1231: "f32[256]" = torch.ops.aten.mul.Tensor(sum_111, 0.0006377551020408163)
    unsqueeze_1083: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1231, 0);  mul_1231 = None
    unsqueeze_1084: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 2);  unsqueeze_1083 = None
    unsqueeze_1085: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 3);  unsqueeze_1084 = None
    mul_1232: "f32[256]" = torch.ops.aten.mul.Tensor(sum_112, 0.0006377551020408163)
    mul_1233: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_1234: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1232, mul_1233);  mul_1232 = mul_1233 = None
    unsqueeze_1086: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1234, 0);  mul_1234 = None
    unsqueeze_1087: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 2);  unsqueeze_1086 = None
    unsqueeze_1088: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 3);  unsqueeze_1087 = None
    mul_1235: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_149);  primals_149 = None
    unsqueeze_1089: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1235, 0);  mul_1235 = None
    unsqueeze_1090: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 2);  unsqueeze_1089 = None
    unsqueeze_1091: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 3);  unsqueeze_1090 = None
    sub_326: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_1082);  convolution_49 = unsqueeze_1082 = None
    mul_1236: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_1088);  sub_326 = unsqueeze_1088 = None
    sub_327: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_54, mul_1236);  where_54 = mul_1236 = None
    sub_328: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_327, unsqueeze_1085);  sub_327 = unsqueeze_1085 = None
    mul_1237: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1091);  sub_328 = unsqueeze_1091 = None
    mul_1238: "f32[256]" = torch.ops.aten.mul.Tensor(sum_112, squeeze_148);  sum_112 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1237, relu_45, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1237 = primals_148 = None
    getitem_396: "f32[8, 256, 14, 14]" = convolution_backward_56[0]
    getitem_397: "f32[256, 256, 3, 3]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_267: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_45);  relu_45 = None
    alias_268: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_267);  alias_267 = None
    le_55: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_268, 0);  alias_268 = None
    scalar_tensor_55: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_55: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_55, scalar_tensor_55, getitem_396);  le_55 = scalar_tensor_55 = getitem_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1092: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_1093: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, 2);  unsqueeze_1092 = None
    unsqueeze_1094: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1093, 3);  unsqueeze_1093 = None
    sum_113: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    sub_329: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_1094)
    mul_1239: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_55, sub_329);  sub_329 = None
    sum_114: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1239, [0, 2, 3]);  mul_1239 = None
    mul_1240: "f32[256]" = torch.ops.aten.mul.Tensor(sum_113, 0.0006377551020408163)
    unsqueeze_1095: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1240, 0);  mul_1240 = None
    unsqueeze_1096: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 2);  unsqueeze_1095 = None
    unsqueeze_1097: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 3);  unsqueeze_1096 = None
    mul_1241: "f32[256]" = torch.ops.aten.mul.Tensor(sum_114, 0.0006377551020408163)
    mul_1242: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_1243: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1241, mul_1242);  mul_1241 = mul_1242 = None
    unsqueeze_1098: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1243, 0);  mul_1243 = None
    unsqueeze_1099: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 2);  unsqueeze_1098 = None
    unsqueeze_1100: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 3);  unsqueeze_1099 = None
    mul_1244: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_146);  primals_146 = None
    unsqueeze_1101: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1244, 0);  mul_1244 = None
    unsqueeze_1102: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 2);  unsqueeze_1101 = None
    unsqueeze_1103: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, 3);  unsqueeze_1102 = None
    sub_330: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_1094);  convolution_48 = unsqueeze_1094 = None
    mul_1245: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_1100);  sub_330 = unsqueeze_1100 = None
    sub_331: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_55, mul_1245);  where_55 = mul_1245 = None
    sub_332: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_331, unsqueeze_1097);  sub_331 = unsqueeze_1097 = None
    mul_1246: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1103);  sub_332 = unsqueeze_1103 = None
    mul_1247: "f32[256]" = torch.ops.aten.mul.Tensor(sum_114, squeeze_145);  sum_114 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1246, relu_44, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1246 = primals_145 = None
    getitem_399: "f32[8, 512, 14, 14]" = convolution_backward_57[0]
    getitem_400: "f32[256, 512, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_606: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_605, getitem_399);  add_605 = getitem_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_270: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_44);  relu_44 = None
    alias_271: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_270);  alias_270 = None
    le_56: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_271, 0);  alias_271 = None
    scalar_tensor_56: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_56: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_56, scalar_tensor_56, add_606);  le_56 = scalar_tensor_56 = add_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_1104: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_1105: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, 2);  unsqueeze_1104 = None
    unsqueeze_1106: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1105, 3);  unsqueeze_1105 = None
    sum_115: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_333: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_1106)
    mul_1248: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_56, sub_333);  sub_333 = None
    sum_116: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1248, [0, 2, 3]);  mul_1248 = None
    mul_1249: "f32[512]" = torch.ops.aten.mul.Tensor(sum_115, 0.0006377551020408163)
    unsqueeze_1107: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1249, 0);  mul_1249 = None
    unsqueeze_1108: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 2);  unsqueeze_1107 = None
    unsqueeze_1109: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 3);  unsqueeze_1108 = None
    mul_1250: "f32[512]" = torch.ops.aten.mul.Tensor(sum_116, 0.0006377551020408163)
    mul_1251: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_1252: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1250, mul_1251);  mul_1250 = mul_1251 = None
    unsqueeze_1110: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1252, 0);  mul_1252 = None
    unsqueeze_1111: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 2);  unsqueeze_1110 = None
    unsqueeze_1112: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 3);  unsqueeze_1111 = None
    mul_1253: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_143);  primals_143 = None
    unsqueeze_1113: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1253, 0);  mul_1253 = None
    unsqueeze_1114: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 2);  unsqueeze_1113 = None
    unsqueeze_1115: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, 3);  unsqueeze_1114 = None
    sub_334: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_1106);  convolution_47 = unsqueeze_1106 = None
    mul_1254: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_1112);  sub_334 = unsqueeze_1112 = None
    sub_335: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_56, mul_1254);  mul_1254 = None
    sub_336: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_335, unsqueeze_1109);  sub_335 = unsqueeze_1109 = None
    mul_1255: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_1115);  sub_336 = unsqueeze_1115 = None
    mul_1256: "f32[512]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_142);  sum_116 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1255, cat_5, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1255 = cat_5 = primals_142 = None
    getitem_402: "f32[8, 1024, 14, 14]" = convolution_backward_58[0]
    getitem_403: "f32[512, 1024, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    slice_26: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_402, 1, 0, 512)
    slice_27: "f32[8, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_402, 1, 512, 1024);  getitem_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_607: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(where_56, slice_26);  where_56 = slice_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_273: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_43);  relu_43 = None
    alias_274: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_273);  alias_273 = None
    le_57: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_274, 0);  alias_274 = None
    scalar_tensor_57: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_57: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_57, scalar_tensor_57, add_607);  le_57 = scalar_tensor_57 = add_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_608: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_27, where_57);  slice_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1116: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_1117: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, 2);  unsqueeze_1116 = None
    unsqueeze_1118: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1117, 3);  unsqueeze_1117 = None
    sum_117: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_337: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_1118)
    mul_1257: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_57, sub_337);  sub_337 = None
    sum_118: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1257, [0, 2, 3]);  mul_1257 = None
    mul_1258: "f32[512]" = torch.ops.aten.mul.Tensor(sum_117, 0.0006377551020408163)
    unsqueeze_1119: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1258, 0);  mul_1258 = None
    unsqueeze_1120: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 2);  unsqueeze_1119 = None
    unsqueeze_1121: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 3);  unsqueeze_1120 = None
    mul_1259: "f32[512]" = torch.ops.aten.mul.Tensor(sum_118, 0.0006377551020408163)
    mul_1260: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_1261: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1259, mul_1260);  mul_1259 = mul_1260 = None
    unsqueeze_1122: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1261, 0);  mul_1261 = None
    unsqueeze_1123: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 2);  unsqueeze_1122 = None
    unsqueeze_1124: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1123, 3);  unsqueeze_1123 = None
    mul_1262: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_140);  primals_140 = None
    unsqueeze_1125: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1262, 0);  mul_1262 = None
    unsqueeze_1126: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 2);  unsqueeze_1125 = None
    unsqueeze_1127: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, 3);  unsqueeze_1126 = None
    sub_338: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_1118);  convolution_46 = unsqueeze_1118 = None
    mul_1263: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_1124);  sub_338 = unsqueeze_1124 = None
    sub_339: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_57, mul_1263);  where_57 = mul_1263 = None
    sub_340: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_339, unsqueeze_1121);  sub_339 = unsqueeze_1121 = None
    mul_1264: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_1127);  sub_340 = unsqueeze_1127 = None
    mul_1265: "f32[512]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_139);  sum_118 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1264, relu_42, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1264 = primals_139 = None
    getitem_405: "f32[8, 256, 14, 14]" = convolution_backward_59[0]
    getitem_406: "f32[512, 256, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_276: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
    alias_277: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_276);  alias_276 = None
    le_58: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_277, 0);  alias_277 = None
    scalar_tensor_58: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_58: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_58, scalar_tensor_58, getitem_405);  le_58 = scalar_tensor_58 = getitem_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1128: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_1129: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, 2);  unsqueeze_1128 = None
    unsqueeze_1130: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1129, 3);  unsqueeze_1129 = None
    sum_119: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    sub_341: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_1130)
    mul_1266: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_58, sub_341);  sub_341 = None
    sum_120: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1266, [0, 2, 3]);  mul_1266 = None
    mul_1267: "f32[256]" = torch.ops.aten.mul.Tensor(sum_119, 0.0006377551020408163)
    unsqueeze_1131: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1267, 0);  mul_1267 = None
    unsqueeze_1132: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 2);  unsqueeze_1131 = None
    unsqueeze_1133: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 3);  unsqueeze_1132 = None
    mul_1268: "f32[256]" = torch.ops.aten.mul.Tensor(sum_120, 0.0006377551020408163)
    mul_1269: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_1270: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1268, mul_1269);  mul_1268 = mul_1269 = None
    unsqueeze_1134: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1270, 0);  mul_1270 = None
    unsqueeze_1135: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 2);  unsqueeze_1134 = None
    unsqueeze_1136: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 3);  unsqueeze_1135 = None
    mul_1271: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_137);  primals_137 = None
    unsqueeze_1137: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1271, 0);  mul_1271 = None
    unsqueeze_1138: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 2);  unsqueeze_1137 = None
    unsqueeze_1139: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 3);  unsqueeze_1138 = None
    sub_342: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_1130);  convolution_45 = unsqueeze_1130 = None
    mul_1272: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_1136);  sub_342 = unsqueeze_1136 = None
    sub_343: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_58, mul_1272);  where_58 = mul_1272 = None
    sub_344: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_343, unsqueeze_1133);  sub_343 = unsqueeze_1133 = None
    mul_1273: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_1139);  sub_344 = unsqueeze_1139 = None
    mul_1274: "f32[256]" = torch.ops.aten.mul.Tensor(sum_120, squeeze_136);  sum_120 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1273, relu_41, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1273 = primals_136 = None
    getitem_408: "f32[8, 256, 14, 14]" = convolution_backward_60[0]
    getitem_409: "f32[256, 256, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_279: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_280: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_279);  alias_279 = None
    le_59: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_280, 0);  alias_280 = None
    scalar_tensor_59: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_59: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_59, scalar_tensor_59, getitem_408);  le_59 = scalar_tensor_59 = getitem_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1140: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_1141: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, 2);  unsqueeze_1140 = None
    unsqueeze_1142: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1141, 3);  unsqueeze_1141 = None
    sum_121: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_345: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_1142)
    mul_1275: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_59, sub_345);  sub_345 = None
    sum_122: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1275, [0, 2, 3]);  mul_1275 = None
    mul_1276: "f32[256]" = torch.ops.aten.mul.Tensor(sum_121, 0.0006377551020408163)
    unsqueeze_1143: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1276, 0);  mul_1276 = None
    unsqueeze_1144: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 2);  unsqueeze_1143 = None
    unsqueeze_1145: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 3);  unsqueeze_1144 = None
    mul_1277: "f32[256]" = torch.ops.aten.mul.Tensor(sum_122, 0.0006377551020408163)
    mul_1278: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_1279: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1277, mul_1278);  mul_1277 = mul_1278 = None
    unsqueeze_1146: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1279, 0);  mul_1279 = None
    unsqueeze_1147: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 2);  unsqueeze_1146 = None
    unsqueeze_1148: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 3);  unsqueeze_1147 = None
    mul_1280: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_134);  primals_134 = None
    unsqueeze_1149: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1280, 0);  mul_1280 = None
    unsqueeze_1150: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 2);  unsqueeze_1149 = None
    unsqueeze_1151: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 3);  unsqueeze_1150 = None
    sub_346: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_1142);  convolution_44 = unsqueeze_1142 = None
    mul_1281: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_1148);  sub_346 = unsqueeze_1148 = None
    sub_347: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_59, mul_1281);  where_59 = mul_1281 = None
    sub_348: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_347, unsqueeze_1145);  sub_347 = unsqueeze_1145 = None
    mul_1282: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_1151);  sub_348 = unsqueeze_1151 = None
    mul_1283: "f32[256]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_133);  sum_122 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1282, relu_40, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1282 = primals_133 = None
    getitem_411: "f32[8, 512, 14, 14]" = convolution_backward_61[0]
    getitem_412: "f32[256, 512, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_609: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_608, getitem_411);  add_608 = getitem_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_282: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(relu_40);  relu_40 = None
    alias_283: "f32[8, 512, 14, 14]" = torch.ops.aten.alias.default(alias_282);  alias_282 = None
    le_60: "b8[8, 512, 14, 14]" = torch.ops.aten.le.Scalar(alias_283, 0);  alias_283 = None
    scalar_tensor_60: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_60: "f32[8, 512, 14, 14]" = torch.ops.aten.where.self(le_60, scalar_tensor_60, add_609);  le_60 = scalar_tensor_60 = add_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1152: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_1153: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, 2);  unsqueeze_1152 = None
    unsqueeze_1154: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1153, 3);  unsqueeze_1153 = None
    sum_123: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_349: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_1154)
    mul_1284: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_60, sub_349);  sub_349 = None
    sum_124: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1284, [0, 2, 3]);  mul_1284 = None
    mul_1285: "f32[512]" = torch.ops.aten.mul.Tensor(sum_123, 0.0006377551020408163)
    unsqueeze_1155: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1285, 0);  mul_1285 = None
    unsqueeze_1156: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 2);  unsqueeze_1155 = None
    unsqueeze_1157: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 3);  unsqueeze_1156 = None
    mul_1286: "f32[512]" = torch.ops.aten.mul.Tensor(sum_124, 0.0006377551020408163)
    mul_1287: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_1288: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1286, mul_1287);  mul_1286 = mul_1287 = None
    unsqueeze_1158: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1288, 0);  mul_1288 = None
    unsqueeze_1159: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 2);  unsqueeze_1158 = None
    unsqueeze_1160: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 3);  unsqueeze_1159 = None
    mul_1289: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_131);  primals_131 = None
    unsqueeze_1161: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1289, 0);  mul_1289 = None
    unsqueeze_1162: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 2);  unsqueeze_1161 = None
    unsqueeze_1163: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 3);  unsqueeze_1162 = None
    sub_350: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_1154);  convolution_43 = unsqueeze_1154 = None
    mul_1290: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_350, unsqueeze_1160);  sub_350 = unsqueeze_1160 = None
    sub_351: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_60, mul_1290);  mul_1290 = None
    sub_352: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_351, unsqueeze_1157);  sub_351 = unsqueeze_1157 = None
    mul_1291: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_1163);  sub_352 = unsqueeze_1163 = None
    mul_1292: "f32[512]" = torch.ops.aten.mul.Tensor(sum_124, squeeze_130);  sum_124 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1291, relu_39, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1291 = primals_130 = None
    getitem_414: "f32[8, 256, 14, 14]" = convolution_backward_62[0]
    getitem_415: "f32[512, 256, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_285: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_286: "f32[8, 256, 14, 14]" = torch.ops.aten.alias.default(alias_285);  alias_285 = None
    le_61: "b8[8, 256, 14, 14]" = torch.ops.aten.le.Scalar(alias_286, 0);  alias_286 = None
    scalar_tensor_61: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_61: "f32[8, 256, 14, 14]" = torch.ops.aten.where.self(le_61, scalar_tensor_61, getitem_414);  le_61 = scalar_tensor_61 = getitem_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1164: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_1165: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, 2);  unsqueeze_1164 = None
    unsqueeze_1166: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1165, 3);  unsqueeze_1165 = None
    sum_125: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_353: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_1166)
    mul_1293: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(where_61, sub_353);  sub_353 = None
    sum_126: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1293, [0, 2, 3]);  mul_1293 = None
    mul_1294: "f32[256]" = torch.ops.aten.mul.Tensor(sum_125, 0.0006377551020408163)
    unsqueeze_1167: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1294, 0);  mul_1294 = None
    unsqueeze_1168: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 2);  unsqueeze_1167 = None
    unsqueeze_1169: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 3);  unsqueeze_1168 = None
    mul_1295: "f32[256]" = torch.ops.aten.mul.Tensor(sum_126, 0.0006377551020408163)
    mul_1296: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_1297: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1295, mul_1296);  mul_1295 = mul_1296 = None
    unsqueeze_1170: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1297, 0);  mul_1297 = None
    unsqueeze_1171: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 2);  unsqueeze_1170 = None
    unsqueeze_1172: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 3);  unsqueeze_1171 = None
    mul_1298: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_128);  primals_128 = None
    unsqueeze_1173: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1298, 0);  mul_1298 = None
    unsqueeze_1174: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 2);  unsqueeze_1173 = None
    unsqueeze_1175: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 3);  unsqueeze_1174 = None
    sub_354: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_1166);  convolution_42 = unsqueeze_1166 = None
    mul_1299: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_354, unsqueeze_1172);  sub_354 = unsqueeze_1172 = None
    sub_355: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(where_61, mul_1299);  where_61 = mul_1299 = None
    sub_356: "f32[8, 256, 14, 14]" = torch.ops.aten.sub.Tensor(sub_355, unsqueeze_1169);  sub_355 = unsqueeze_1169 = None
    mul_1300: "f32[8, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_1175);  sub_356 = unsqueeze_1175 = None
    mul_1301: "f32[256]" = torch.ops.aten.mul.Tensor(sum_126, squeeze_127);  sum_126 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1300, relu_38, primals_127, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1300 = primals_127 = None
    getitem_417: "f32[8, 256, 28, 28]" = convolution_backward_63[0]
    getitem_418: "f32[256, 256, 3, 3]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_288: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_289: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_288);  alias_288 = None
    le_62: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_289, 0);  alias_289 = None
    scalar_tensor_62: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_62: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_62, scalar_tensor_62, getitem_417);  le_62 = scalar_tensor_62 = getitem_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1176: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_1177: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, 2);  unsqueeze_1176 = None
    unsqueeze_1178: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1177, 3);  unsqueeze_1177 = None
    sum_127: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_357: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1178)
    mul_1302: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_62, sub_357);  sub_357 = None
    sum_128: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1302, [0, 2, 3]);  mul_1302 = None
    mul_1303: "f32[256]" = torch.ops.aten.mul.Tensor(sum_127, 0.00015943877551020407)
    unsqueeze_1179: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1303, 0);  mul_1303 = None
    unsqueeze_1180: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 2);  unsqueeze_1179 = None
    unsqueeze_1181: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 3);  unsqueeze_1180 = None
    mul_1304: "f32[256]" = torch.ops.aten.mul.Tensor(sum_128, 0.00015943877551020407)
    mul_1305: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_1306: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1304, mul_1305);  mul_1304 = mul_1305 = None
    unsqueeze_1182: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1306, 0);  mul_1306 = None
    unsqueeze_1183: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 2);  unsqueeze_1182 = None
    unsqueeze_1184: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1183, 3);  unsqueeze_1183 = None
    mul_1307: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_125);  primals_125 = None
    unsqueeze_1185: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1307, 0);  mul_1307 = None
    unsqueeze_1186: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 2);  unsqueeze_1185 = None
    unsqueeze_1187: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, 3);  unsqueeze_1186 = None
    sub_358: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1178);  convolution_41 = unsqueeze_1178 = None
    mul_1308: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_358, unsqueeze_1184);  sub_358 = unsqueeze_1184 = None
    sub_359: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_62, mul_1308);  where_62 = mul_1308 = None
    sub_360: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_359, unsqueeze_1181);  sub_359 = unsqueeze_1181 = None
    mul_1309: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_1187);  sub_360 = unsqueeze_1187 = None
    mul_1310: "f32[256]" = torch.ops.aten.mul.Tensor(sum_128, squeeze_124);  sum_128 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1309, relu_37, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1309 = primals_124 = None
    getitem_420: "f32[8, 256, 28, 28]" = convolution_backward_64[0]
    getitem_421: "f32[256, 256, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    unsqueeze_1188: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_1189: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, 2);  unsqueeze_1188 = None
    unsqueeze_1190: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1189, 3);  unsqueeze_1189 = None
    sum_129: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_361: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_1190)
    mul_1311: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(where_60, sub_361);  sub_361 = None
    sum_130: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1311, [0, 2, 3]);  mul_1311 = None
    mul_1312: "f32[512]" = torch.ops.aten.mul.Tensor(sum_129, 0.0006377551020408163)
    unsqueeze_1191: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1312, 0);  mul_1312 = None
    unsqueeze_1192: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 2);  unsqueeze_1191 = None
    unsqueeze_1193: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, 3);  unsqueeze_1192 = None
    mul_1313: "f32[512]" = torch.ops.aten.mul.Tensor(sum_130, 0.0006377551020408163)
    mul_1314: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_1315: "f32[512]" = torch.ops.aten.mul.Tensor(mul_1313, mul_1314);  mul_1313 = mul_1314 = None
    unsqueeze_1194: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1315, 0);  mul_1315 = None
    unsqueeze_1195: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 2);  unsqueeze_1194 = None
    unsqueeze_1196: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 3);  unsqueeze_1195 = None
    mul_1316: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_122);  primals_122 = None
    unsqueeze_1197: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_1316, 0);  mul_1316 = None
    unsqueeze_1198: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 2);  unsqueeze_1197 = None
    unsqueeze_1199: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 3);  unsqueeze_1198 = None
    sub_362: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_1190);  convolution_40 = unsqueeze_1190 = None
    mul_1317: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_362, unsqueeze_1196);  sub_362 = unsqueeze_1196 = None
    sub_363: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(where_60, mul_1317);  where_60 = mul_1317 = None
    sub_364: "f32[8, 512, 14, 14]" = torch.ops.aten.sub.Tensor(sub_363, unsqueeze_1193);  sub_363 = unsqueeze_1193 = None
    mul_1318: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_1199);  sub_364 = unsqueeze_1199 = None
    mul_1319: "f32[512]" = torch.ops.aten.mul.Tensor(sum_130, squeeze_121);  sum_130 = squeeze_121 = None
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1318, getitem_94, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1318 = getitem_94 = primals_121 = None
    getitem_423: "f32[8, 256, 14, 14]" = convolution_backward_65[0]
    getitem_424: "f32[512, 256, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_backward_1: "f32[8, 256, 28, 28]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_423, relu_37, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_95);  getitem_423 = getitem_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    add_610: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(getitem_420, max_pool2d_with_indices_backward_1);  getitem_420 = max_pool2d_with_indices_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_backward_2: "f32[8, 256, 28, 28]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_6, relu_37, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_89);  slice_6 = getitem_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    add_611: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_610, max_pool2d_with_indices_backward_2);  add_610 = max_pool2d_with_indices_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_291: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_292: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_291);  alias_291 = None
    le_63: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_292, 0);  alias_292 = None
    scalar_tensor_63: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_63: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_63, scalar_tensor_63, add_611);  le_63 = scalar_tensor_63 = add_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_1200: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_1201: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, 2);  unsqueeze_1200 = None
    unsqueeze_1202: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1201, 3);  unsqueeze_1201 = None
    sum_131: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_365: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_1202)
    mul_1320: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_63, sub_365);  sub_365 = None
    sum_132: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1320, [0, 2, 3]);  mul_1320 = None
    mul_1321: "f32[256]" = torch.ops.aten.mul.Tensor(sum_131, 0.00015943877551020407)
    unsqueeze_1203: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1321, 0);  mul_1321 = None
    unsqueeze_1204: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 2);  unsqueeze_1203 = None
    unsqueeze_1205: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 3);  unsqueeze_1204 = None
    mul_1322: "f32[256]" = torch.ops.aten.mul.Tensor(sum_132, 0.00015943877551020407)
    mul_1323: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_1324: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1322, mul_1323);  mul_1322 = mul_1323 = None
    unsqueeze_1206: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1324, 0);  mul_1324 = None
    unsqueeze_1207: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 2);  unsqueeze_1206 = None
    unsqueeze_1208: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 3);  unsqueeze_1207 = None
    mul_1325: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_119);  primals_119 = None
    unsqueeze_1209: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1325, 0);  mul_1325 = None
    unsqueeze_1210: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 2);  unsqueeze_1209 = None
    unsqueeze_1211: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 3);  unsqueeze_1210 = None
    sub_366: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_1202);  convolution_39 = unsqueeze_1202 = None
    mul_1326: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_366, unsqueeze_1208);  sub_366 = unsqueeze_1208 = None
    sub_367: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_63, mul_1326);  mul_1326 = None
    sub_368: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_367, unsqueeze_1205);  sub_367 = unsqueeze_1205 = None
    mul_1327: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_1211);  sub_368 = unsqueeze_1211 = None
    mul_1328: "f32[256]" = torch.ops.aten.mul.Tensor(sum_132, squeeze_118);  sum_132 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1327, cat_4, primals_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1327 = cat_4 = primals_118 = None
    getitem_426: "f32[8, 1152, 28, 28]" = convolution_backward_66[0]
    getitem_427: "f32[256, 1152, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    slice_28: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_426, 1, 0, 256)
    slice_29: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_426, 1, 256, 512)
    slice_30: "f32[8, 128, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_426, 1, 512, 640)
    slice_31: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_426, 1, 640, 896)
    slice_32: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_426, 1, 896, 1152);  getitem_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_612: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(where_63, slice_28);  where_63 = slice_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_294: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_295: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_294);  alias_294 = None
    le_64: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_295, 0);  alias_295 = None
    scalar_tensor_64: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_64: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_64, scalar_tensor_64, add_612);  le_64 = scalar_tensor_64 = add_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_613: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_29, where_64);  slice_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1212: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_1213: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, 2);  unsqueeze_1212 = None
    unsqueeze_1214: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1213, 3);  unsqueeze_1213 = None
    sum_133: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_369: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_1214)
    mul_1329: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_64, sub_369);  sub_369 = None
    sum_134: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1329, [0, 2, 3]);  mul_1329 = None
    mul_1330: "f32[256]" = torch.ops.aten.mul.Tensor(sum_133, 0.00015943877551020407)
    unsqueeze_1215: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1330, 0);  mul_1330 = None
    unsqueeze_1216: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 2);  unsqueeze_1215 = None
    unsqueeze_1217: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 3);  unsqueeze_1216 = None
    mul_1331: "f32[256]" = torch.ops.aten.mul.Tensor(sum_134, 0.00015943877551020407)
    mul_1332: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_1333: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1331, mul_1332);  mul_1331 = mul_1332 = None
    unsqueeze_1218: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1333, 0);  mul_1333 = None
    unsqueeze_1219: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 2);  unsqueeze_1218 = None
    unsqueeze_1220: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 3);  unsqueeze_1219 = None
    mul_1334: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_116);  primals_116 = None
    unsqueeze_1221: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1334, 0);  mul_1334 = None
    unsqueeze_1222: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 2);  unsqueeze_1221 = None
    unsqueeze_1223: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 3);  unsqueeze_1222 = None
    sub_370: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_1214);  convolution_38 = unsqueeze_1214 = None
    mul_1335: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_370, unsqueeze_1220);  sub_370 = unsqueeze_1220 = None
    sub_371: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_64, mul_1335);  where_64 = mul_1335 = None
    sub_372: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_371, unsqueeze_1217);  sub_371 = unsqueeze_1217 = None
    mul_1336: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_1223);  sub_372 = unsqueeze_1223 = None
    mul_1337: "f32[256]" = torch.ops.aten.mul.Tensor(sum_134, squeeze_115);  sum_134 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1336, relu_35, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1336 = primals_115 = None
    getitem_429: "f32[8, 128, 28, 28]" = convolution_backward_67[0]
    getitem_430: "f32[256, 128, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_297: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_298: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_297);  alias_297 = None
    le_65: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_298, 0);  alias_298 = None
    scalar_tensor_65: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_65: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_65, scalar_tensor_65, getitem_429);  le_65 = scalar_tensor_65 = getitem_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1224: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_1225: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, 2);  unsqueeze_1224 = None
    unsqueeze_1226: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1225, 3);  unsqueeze_1225 = None
    sum_135: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_373: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1226)
    mul_1338: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_65, sub_373);  sub_373 = None
    sum_136: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1338, [0, 2, 3]);  mul_1338 = None
    mul_1339: "f32[128]" = torch.ops.aten.mul.Tensor(sum_135, 0.00015943877551020407)
    unsqueeze_1227: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1339, 0);  mul_1339 = None
    unsqueeze_1228: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 2);  unsqueeze_1227 = None
    unsqueeze_1229: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 3);  unsqueeze_1228 = None
    mul_1340: "f32[128]" = torch.ops.aten.mul.Tensor(sum_136, 0.00015943877551020407)
    mul_1341: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_1342: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1340, mul_1341);  mul_1340 = mul_1341 = None
    unsqueeze_1230: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1342, 0);  mul_1342 = None
    unsqueeze_1231: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 2);  unsqueeze_1230 = None
    unsqueeze_1232: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 3);  unsqueeze_1231 = None
    mul_1343: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_113);  primals_113 = None
    unsqueeze_1233: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1343, 0);  mul_1343 = None
    unsqueeze_1234: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 2);  unsqueeze_1233 = None
    unsqueeze_1235: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 3);  unsqueeze_1234 = None
    sub_374: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1226);  convolution_37 = unsqueeze_1226 = None
    mul_1344: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_374, unsqueeze_1232);  sub_374 = unsqueeze_1232 = None
    sub_375: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_65, mul_1344);  where_65 = mul_1344 = None
    sub_376: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_375, unsqueeze_1229);  sub_375 = unsqueeze_1229 = None
    mul_1345: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_1235);  sub_376 = unsqueeze_1235 = None
    mul_1346: "f32[128]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_112);  sum_136 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1345, relu_34, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1345 = primals_112 = None
    getitem_432: "f32[8, 128, 28, 28]" = convolution_backward_68[0]
    getitem_433: "f32[128, 128, 3, 3]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_300: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_301: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_300);  alias_300 = None
    le_66: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_301, 0);  alias_301 = None
    scalar_tensor_66: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_66: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_66, scalar_tensor_66, getitem_432);  le_66 = scalar_tensor_66 = getitem_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1236: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_1237: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, 2);  unsqueeze_1236 = None
    unsqueeze_1238: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1237, 3);  unsqueeze_1237 = None
    sum_137: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_377: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1238)
    mul_1347: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_66, sub_377);  sub_377 = None
    sum_138: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1347, [0, 2, 3]);  mul_1347 = None
    mul_1348: "f32[128]" = torch.ops.aten.mul.Tensor(sum_137, 0.00015943877551020407)
    unsqueeze_1239: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1348, 0);  mul_1348 = None
    unsqueeze_1240: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 2);  unsqueeze_1239 = None
    unsqueeze_1241: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, 3);  unsqueeze_1240 = None
    mul_1349: "f32[128]" = torch.ops.aten.mul.Tensor(sum_138, 0.00015943877551020407)
    mul_1350: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_1351: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1349, mul_1350);  mul_1349 = mul_1350 = None
    unsqueeze_1242: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1351, 0);  mul_1351 = None
    unsqueeze_1243: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 2);  unsqueeze_1242 = None
    unsqueeze_1244: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1243, 3);  unsqueeze_1243 = None
    mul_1352: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_110);  primals_110 = None
    unsqueeze_1245: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1352, 0);  mul_1352 = None
    unsqueeze_1246: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 2);  unsqueeze_1245 = None
    unsqueeze_1247: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, 3);  unsqueeze_1246 = None
    sub_378: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1238);  convolution_36 = unsqueeze_1238 = None
    mul_1353: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_378, unsqueeze_1244);  sub_378 = unsqueeze_1244 = None
    sub_379: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_66, mul_1353);  where_66 = mul_1353 = None
    sub_380: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_379, unsqueeze_1241);  sub_379 = unsqueeze_1241 = None
    mul_1354: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_1247);  sub_380 = unsqueeze_1247 = None
    mul_1355: "f32[128]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_109);  sum_138 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1354, relu_33, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1354 = primals_109 = None
    getitem_435: "f32[8, 256, 28, 28]" = convolution_backward_69[0]
    getitem_436: "f32[128, 256, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_614: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_613, getitem_435);  add_613 = getitem_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_303: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_304: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_303);  alias_303 = None
    le_67: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_304, 0);  alias_304 = None
    scalar_tensor_67: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_67: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_67, scalar_tensor_67, add_614);  le_67 = scalar_tensor_67 = add_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_615: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_32, where_67);  slice_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1248: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_1249: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, 2);  unsqueeze_1248 = None
    unsqueeze_1250: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1249, 3);  unsqueeze_1249 = None
    sum_139: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_381: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_1250)
    mul_1356: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_67, sub_381);  sub_381 = None
    sum_140: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1356, [0, 2, 3]);  mul_1356 = None
    mul_1357: "f32[256]" = torch.ops.aten.mul.Tensor(sum_139, 0.00015943877551020407)
    unsqueeze_1251: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1357, 0);  mul_1357 = None
    unsqueeze_1252: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 2);  unsqueeze_1251 = None
    unsqueeze_1253: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, 3);  unsqueeze_1252 = None
    mul_1358: "f32[256]" = torch.ops.aten.mul.Tensor(sum_140, 0.00015943877551020407)
    mul_1359: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_1360: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1358, mul_1359);  mul_1358 = mul_1359 = None
    unsqueeze_1254: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1360, 0);  mul_1360 = None
    unsqueeze_1255: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 2);  unsqueeze_1254 = None
    unsqueeze_1256: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1255, 3);  unsqueeze_1255 = None
    mul_1361: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_107);  primals_107 = None
    unsqueeze_1257: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1361, 0);  mul_1361 = None
    unsqueeze_1258: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 2);  unsqueeze_1257 = None
    unsqueeze_1259: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, 3);  unsqueeze_1258 = None
    sub_382: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_1250);  convolution_35 = unsqueeze_1250 = None
    mul_1362: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_382, unsqueeze_1256);  sub_382 = unsqueeze_1256 = None
    sub_383: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_67, mul_1362);  where_67 = mul_1362 = None
    sub_384: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_383, unsqueeze_1253);  sub_383 = unsqueeze_1253 = None
    mul_1363: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_1259);  sub_384 = unsqueeze_1259 = None
    mul_1364: "f32[256]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_106);  sum_140 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1363, relu_32, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1363 = primals_106 = None
    getitem_438: "f32[8, 128, 28, 28]" = convolution_backward_70[0]
    getitem_439: "f32[256, 128, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_306: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_307: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_306);  alias_306 = None
    le_68: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_307, 0);  alias_307 = None
    scalar_tensor_68: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_68: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_68, scalar_tensor_68, getitem_438);  le_68 = scalar_tensor_68 = getitem_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1260: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_1261: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, 2);  unsqueeze_1260 = None
    unsqueeze_1262: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1261, 3);  unsqueeze_1261 = None
    sum_141: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_385: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_1262)
    mul_1365: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_68, sub_385);  sub_385 = None
    sum_142: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1365, [0, 2, 3]);  mul_1365 = None
    mul_1366: "f32[128]" = torch.ops.aten.mul.Tensor(sum_141, 0.00015943877551020407)
    unsqueeze_1263: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1366, 0);  mul_1366 = None
    unsqueeze_1264: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 2);  unsqueeze_1263 = None
    unsqueeze_1265: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, 3);  unsqueeze_1264 = None
    mul_1367: "f32[128]" = torch.ops.aten.mul.Tensor(sum_142, 0.00015943877551020407)
    mul_1368: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_1369: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1367, mul_1368);  mul_1367 = mul_1368 = None
    unsqueeze_1266: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1369, 0);  mul_1369 = None
    unsqueeze_1267: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 2);  unsqueeze_1266 = None
    unsqueeze_1268: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1267, 3);  unsqueeze_1267 = None
    mul_1370: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_104);  primals_104 = None
    unsqueeze_1269: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1370, 0);  mul_1370 = None
    unsqueeze_1270: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 2);  unsqueeze_1269 = None
    unsqueeze_1271: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, 3);  unsqueeze_1270 = None
    sub_386: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_1262);  convolution_34 = unsqueeze_1262 = None
    mul_1371: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_386, unsqueeze_1268);  sub_386 = unsqueeze_1268 = None
    sub_387: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_68, mul_1371);  where_68 = mul_1371 = None
    sub_388: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_387, unsqueeze_1265);  sub_387 = unsqueeze_1265 = None
    mul_1372: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_1271);  sub_388 = unsqueeze_1271 = None
    mul_1373: "f32[128]" = torch.ops.aten.mul.Tensor(sum_142, squeeze_103);  sum_142 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1372, relu_31, primals_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1372 = primals_103 = None
    getitem_441: "f32[8, 128, 28, 28]" = convolution_backward_71[0]
    getitem_442: "f32[128, 128, 3, 3]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_309: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_310: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_309);  alias_309 = None
    le_69: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_310, 0);  alias_310 = None
    scalar_tensor_69: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_69: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_69, scalar_tensor_69, getitem_441);  le_69 = scalar_tensor_69 = getitem_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1272: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_1273: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, 2);  unsqueeze_1272 = None
    unsqueeze_1274: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1273, 3);  unsqueeze_1273 = None
    sum_143: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_389: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1274)
    mul_1374: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_69, sub_389);  sub_389 = None
    sum_144: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1374, [0, 2, 3]);  mul_1374 = None
    mul_1375: "f32[128]" = torch.ops.aten.mul.Tensor(sum_143, 0.00015943877551020407)
    unsqueeze_1275: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1375, 0);  mul_1375 = None
    unsqueeze_1276: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 2);  unsqueeze_1275 = None
    unsqueeze_1277: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, 3);  unsqueeze_1276 = None
    mul_1376: "f32[128]" = torch.ops.aten.mul.Tensor(sum_144, 0.00015943877551020407)
    mul_1377: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_1378: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1376, mul_1377);  mul_1376 = mul_1377 = None
    unsqueeze_1278: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1378, 0);  mul_1378 = None
    unsqueeze_1279: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 2);  unsqueeze_1278 = None
    unsqueeze_1280: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1279, 3);  unsqueeze_1279 = None
    mul_1379: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_101);  primals_101 = None
    unsqueeze_1281: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1379, 0);  mul_1379 = None
    unsqueeze_1282: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 2);  unsqueeze_1281 = None
    unsqueeze_1283: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 3);  unsqueeze_1282 = None
    sub_390: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1274);  convolution_33 = unsqueeze_1274 = None
    mul_1380: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_390, unsqueeze_1280);  sub_390 = unsqueeze_1280 = None
    sub_391: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_69, mul_1380);  where_69 = mul_1380 = None
    sub_392: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_391, unsqueeze_1277);  sub_391 = unsqueeze_1277 = None
    mul_1381: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_1283);  sub_392 = unsqueeze_1283 = None
    mul_1382: "f32[128]" = torch.ops.aten.mul.Tensor(sum_144, squeeze_100);  sum_144 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1381, relu_30, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1381 = primals_100 = None
    getitem_444: "f32[8, 256, 28, 28]" = convolution_backward_72[0]
    getitem_445: "f32[128, 256, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_616: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_615, getitem_444);  add_615 = getitem_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_312: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_313: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_312);  alias_312 = None
    le_70: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_313, 0);  alias_313 = None
    scalar_tensor_70: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_70: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_70, scalar_tensor_70, add_616);  le_70 = scalar_tensor_70 = add_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_1284: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_1285: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, 2);  unsqueeze_1284 = None
    unsqueeze_1286: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1285, 3);  unsqueeze_1285 = None
    sum_145: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_393: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1286)
    mul_1383: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_70, sub_393);  sub_393 = None
    sum_146: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1383, [0, 2, 3]);  mul_1383 = None
    mul_1384: "f32[256]" = torch.ops.aten.mul.Tensor(sum_145, 0.00015943877551020407)
    unsqueeze_1287: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1384, 0);  mul_1384 = None
    unsqueeze_1288: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1287, 2);  unsqueeze_1287 = None
    unsqueeze_1289: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, 3);  unsqueeze_1288 = None
    mul_1385: "f32[256]" = torch.ops.aten.mul.Tensor(sum_146, 0.00015943877551020407)
    mul_1386: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_1387: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1385, mul_1386);  mul_1385 = mul_1386 = None
    unsqueeze_1290: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1387, 0);  mul_1387 = None
    unsqueeze_1291: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 2);  unsqueeze_1290 = None
    unsqueeze_1292: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 3);  unsqueeze_1291 = None
    mul_1388: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_98);  primals_98 = None
    unsqueeze_1293: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1388, 0);  mul_1388 = None
    unsqueeze_1294: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1293, 2);  unsqueeze_1293 = None
    unsqueeze_1295: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 3);  unsqueeze_1294 = None
    sub_394: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1286);  convolution_32 = unsqueeze_1286 = None
    mul_1389: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_394, unsqueeze_1292);  sub_394 = unsqueeze_1292 = None
    sub_395: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_70, mul_1389);  mul_1389 = None
    sub_396: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_395, unsqueeze_1289);  sub_395 = unsqueeze_1289 = None
    mul_1390: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_1295);  sub_396 = unsqueeze_1295 = None
    mul_1391: "f32[256]" = torch.ops.aten.mul.Tensor(sum_146, squeeze_97);  sum_146 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1390, cat_3, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1390 = cat_3 = primals_97 = None
    getitem_447: "f32[8, 512, 28, 28]" = convolution_backward_73[0]
    getitem_448: "f32[256, 512, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
    slice_33: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_447, 1, 0, 256)
    slice_34: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_447, 1, 256, 512);  getitem_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_617: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(where_70, slice_33);  where_70 = slice_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_315: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_316: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_315);  alias_315 = None
    le_71: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_316, 0);  alias_316 = None
    scalar_tensor_71: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_71: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_71, scalar_tensor_71, add_617);  le_71 = scalar_tensor_71 = add_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_618: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_34, where_71);  slice_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1296: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_1297: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, 2);  unsqueeze_1296 = None
    unsqueeze_1298: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1297, 3);  unsqueeze_1297 = None
    sum_147: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_397: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1298)
    mul_1392: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_71, sub_397);  sub_397 = None
    sum_148: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1392, [0, 2, 3]);  mul_1392 = None
    mul_1393: "f32[256]" = torch.ops.aten.mul.Tensor(sum_147, 0.00015943877551020407)
    unsqueeze_1299: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1393, 0);  mul_1393 = None
    unsqueeze_1300: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1299, 2);  unsqueeze_1299 = None
    unsqueeze_1301: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, 3);  unsqueeze_1300 = None
    mul_1394: "f32[256]" = torch.ops.aten.mul.Tensor(sum_148, 0.00015943877551020407)
    mul_1395: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_1396: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1394, mul_1395);  mul_1394 = mul_1395 = None
    unsqueeze_1302: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1396, 0);  mul_1396 = None
    unsqueeze_1303: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 2);  unsqueeze_1302 = None
    unsqueeze_1304: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 3);  unsqueeze_1303 = None
    mul_1397: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_95);  primals_95 = None
    unsqueeze_1305: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1397, 0);  mul_1397 = None
    unsqueeze_1306: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1305, 2);  unsqueeze_1305 = None
    unsqueeze_1307: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 3);  unsqueeze_1306 = None
    sub_398: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1298);  convolution_31 = unsqueeze_1298 = None
    mul_1398: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_398, unsqueeze_1304);  sub_398 = unsqueeze_1304 = None
    sub_399: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_71, mul_1398);  where_71 = mul_1398 = None
    sub_400: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_399, unsqueeze_1301);  sub_399 = unsqueeze_1301 = None
    mul_1399: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_1307);  sub_400 = unsqueeze_1307 = None
    mul_1400: "f32[256]" = torch.ops.aten.mul.Tensor(sum_148, squeeze_94);  sum_148 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1399, relu_28, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1399 = primals_94 = None
    getitem_450: "f32[8, 128, 28, 28]" = convolution_backward_74[0]
    getitem_451: "f32[256, 128, 1, 1]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_318: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_319: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_318);  alias_318 = None
    le_72: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_319, 0);  alias_319 = None
    scalar_tensor_72: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_72: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_72, scalar_tensor_72, getitem_450);  le_72 = scalar_tensor_72 = getitem_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1308: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_1309: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, 2);  unsqueeze_1308 = None
    unsqueeze_1310: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1309, 3);  unsqueeze_1309 = None
    sum_149: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_401: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_1310)
    mul_1401: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_72, sub_401);  sub_401 = None
    sum_150: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1401, [0, 2, 3]);  mul_1401 = None
    mul_1402: "f32[128]" = torch.ops.aten.mul.Tensor(sum_149, 0.00015943877551020407)
    unsqueeze_1311: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1402, 0);  mul_1402 = None
    unsqueeze_1312: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1311, 2);  unsqueeze_1311 = None
    unsqueeze_1313: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, 3);  unsqueeze_1312 = None
    mul_1403: "f32[128]" = torch.ops.aten.mul.Tensor(sum_150, 0.00015943877551020407)
    mul_1404: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_1405: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1403, mul_1404);  mul_1403 = mul_1404 = None
    unsqueeze_1314: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1405, 0);  mul_1405 = None
    unsqueeze_1315: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 2);  unsqueeze_1314 = None
    unsqueeze_1316: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 3);  unsqueeze_1315 = None
    mul_1406: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_92);  primals_92 = None
    unsqueeze_1317: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1406, 0);  mul_1406 = None
    unsqueeze_1318: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1317, 2);  unsqueeze_1317 = None
    unsqueeze_1319: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 3);  unsqueeze_1318 = None
    sub_402: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_1310);  convolution_30 = unsqueeze_1310 = None
    mul_1407: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_402, unsqueeze_1316);  sub_402 = unsqueeze_1316 = None
    sub_403: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_72, mul_1407);  where_72 = mul_1407 = None
    sub_404: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_403, unsqueeze_1313);  sub_403 = unsqueeze_1313 = None
    mul_1408: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_404, unsqueeze_1319);  sub_404 = unsqueeze_1319 = None
    mul_1409: "f32[128]" = torch.ops.aten.mul.Tensor(sum_150, squeeze_91);  sum_150 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1408, relu_27, primals_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1408 = primals_91 = None
    getitem_453: "f32[8, 128, 28, 28]" = convolution_backward_75[0]
    getitem_454: "f32[128, 128, 3, 3]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_321: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_322: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_321);  alias_321 = None
    le_73: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_322, 0);  alias_322 = None
    scalar_tensor_73: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_73: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_73, scalar_tensor_73, getitem_453);  le_73 = scalar_tensor_73 = getitem_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1320: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_1321: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, 2);  unsqueeze_1320 = None
    unsqueeze_1322: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1321, 3);  unsqueeze_1321 = None
    sum_151: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_405: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1322)
    mul_1410: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_73, sub_405);  sub_405 = None
    sum_152: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1410, [0, 2, 3]);  mul_1410 = None
    mul_1411: "f32[128]" = torch.ops.aten.mul.Tensor(sum_151, 0.00015943877551020407)
    unsqueeze_1323: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1411, 0);  mul_1411 = None
    unsqueeze_1324: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1323, 2);  unsqueeze_1323 = None
    unsqueeze_1325: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, 3);  unsqueeze_1324 = None
    mul_1412: "f32[128]" = torch.ops.aten.mul.Tensor(sum_152, 0.00015943877551020407)
    mul_1413: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_1414: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1412, mul_1413);  mul_1412 = mul_1413 = None
    unsqueeze_1326: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1414, 0);  mul_1414 = None
    unsqueeze_1327: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 2);  unsqueeze_1326 = None
    unsqueeze_1328: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 3);  unsqueeze_1327 = None
    mul_1415: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_89);  primals_89 = None
    unsqueeze_1329: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1415, 0);  mul_1415 = None
    unsqueeze_1330: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1329, 2);  unsqueeze_1329 = None
    unsqueeze_1331: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, 3);  unsqueeze_1330 = None
    sub_406: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_1322);  convolution_29 = unsqueeze_1322 = None
    mul_1416: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_406, unsqueeze_1328);  sub_406 = unsqueeze_1328 = None
    sub_407: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_73, mul_1416);  where_73 = mul_1416 = None
    sub_408: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_407, unsqueeze_1325);  sub_407 = unsqueeze_1325 = None
    mul_1417: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_408, unsqueeze_1331);  sub_408 = unsqueeze_1331 = None
    mul_1418: "f32[128]" = torch.ops.aten.mul.Tensor(sum_152, squeeze_88);  sum_152 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1417, relu_26, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1417 = primals_88 = None
    getitem_456: "f32[8, 256, 28, 28]" = convolution_backward_76[0]
    getitem_457: "f32[128, 256, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_619: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_618, getitem_456);  add_618 = getitem_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_324: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_325: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_324);  alias_324 = None
    le_74: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_325, 0);  alias_325 = None
    scalar_tensor_74: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_74: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_74, scalar_tensor_74, add_619);  le_74 = scalar_tensor_74 = add_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_620: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_31, where_74);  slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1332: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_1333: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, 2);  unsqueeze_1332 = None
    unsqueeze_1334: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1333, 3);  unsqueeze_1333 = None
    sum_153: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_409: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1334)
    mul_1419: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_74, sub_409);  sub_409 = None
    sum_154: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1419, [0, 2, 3]);  mul_1419 = None
    mul_1420: "f32[256]" = torch.ops.aten.mul.Tensor(sum_153, 0.00015943877551020407)
    unsqueeze_1335: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1420, 0);  mul_1420 = None
    unsqueeze_1336: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1335, 2);  unsqueeze_1335 = None
    unsqueeze_1337: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, 3);  unsqueeze_1336 = None
    mul_1421: "f32[256]" = torch.ops.aten.mul.Tensor(sum_154, 0.00015943877551020407)
    mul_1422: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_1423: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1421, mul_1422);  mul_1421 = mul_1422 = None
    unsqueeze_1338: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1423, 0);  mul_1423 = None
    unsqueeze_1339: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 2);  unsqueeze_1338 = None
    unsqueeze_1340: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1339, 3);  unsqueeze_1339 = None
    mul_1424: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_86);  primals_86 = None
    unsqueeze_1341: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1424, 0);  mul_1424 = None
    unsqueeze_1342: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 2);  unsqueeze_1341 = None
    unsqueeze_1343: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, 3);  unsqueeze_1342 = None
    sub_410: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1334);  convolution_28 = unsqueeze_1334 = None
    mul_1425: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_410, unsqueeze_1340);  sub_410 = unsqueeze_1340 = None
    sub_411: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_74, mul_1425);  where_74 = mul_1425 = None
    sub_412: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_411, unsqueeze_1337);  sub_411 = unsqueeze_1337 = None
    mul_1426: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_412, unsqueeze_1343);  sub_412 = unsqueeze_1343 = None
    mul_1427: "f32[256]" = torch.ops.aten.mul.Tensor(sum_154, squeeze_85);  sum_154 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1426, relu_25, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1426 = primals_85 = None
    getitem_459: "f32[8, 128, 28, 28]" = convolution_backward_77[0]
    getitem_460: "f32[256, 128, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_327: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_328: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_327);  alias_327 = None
    le_75: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_328, 0);  alias_328 = None
    scalar_tensor_75: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_75: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_75, scalar_tensor_75, getitem_459);  le_75 = scalar_tensor_75 = getitem_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1344: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_1345: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, 2);  unsqueeze_1344 = None
    unsqueeze_1346: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1345, 3);  unsqueeze_1345 = None
    sum_155: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_413: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1346)
    mul_1428: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_75, sub_413);  sub_413 = None
    sum_156: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1428, [0, 2, 3]);  mul_1428 = None
    mul_1429: "f32[128]" = torch.ops.aten.mul.Tensor(sum_155, 0.00015943877551020407)
    unsqueeze_1347: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1429, 0);  mul_1429 = None
    unsqueeze_1348: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1347, 2);  unsqueeze_1347 = None
    unsqueeze_1349: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, 3);  unsqueeze_1348 = None
    mul_1430: "f32[128]" = torch.ops.aten.mul.Tensor(sum_156, 0.00015943877551020407)
    mul_1431: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_1432: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1430, mul_1431);  mul_1430 = mul_1431 = None
    unsqueeze_1350: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1432, 0);  mul_1432 = None
    unsqueeze_1351: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 2);  unsqueeze_1350 = None
    unsqueeze_1352: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 3);  unsqueeze_1351 = None
    mul_1433: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_83);  primals_83 = None
    unsqueeze_1353: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1433, 0);  mul_1433 = None
    unsqueeze_1354: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 2);  unsqueeze_1353 = None
    unsqueeze_1355: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 3);  unsqueeze_1354 = None
    sub_414: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1346);  convolution_27 = unsqueeze_1346 = None
    mul_1434: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_414, unsqueeze_1352);  sub_414 = unsqueeze_1352 = None
    sub_415: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_75, mul_1434);  where_75 = mul_1434 = None
    sub_416: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_415, unsqueeze_1349);  sub_415 = unsqueeze_1349 = None
    mul_1435: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_416, unsqueeze_1355);  sub_416 = unsqueeze_1355 = None
    mul_1436: "f32[128]" = torch.ops.aten.mul.Tensor(sum_156, squeeze_82);  sum_156 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1435, relu_24, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1435 = primals_82 = None
    getitem_462: "f32[8, 128, 28, 28]" = convolution_backward_78[0]
    getitem_463: "f32[128, 128, 3, 3]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_330: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_331: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_330);  alias_330 = None
    le_76: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_331, 0);  alias_331 = None
    scalar_tensor_76: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_76: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_76, scalar_tensor_76, getitem_462);  le_76 = scalar_tensor_76 = getitem_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1356: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_1357: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, 2);  unsqueeze_1356 = None
    unsqueeze_1358: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1357, 3);  unsqueeze_1357 = None
    sum_157: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_417: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1358)
    mul_1437: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_76, sub_417);  sub_417 = None
    sum_158: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1437, [0, 2, 3]);  mul_1437 = None
    mul_1438: "f32[128]" = torch.ops.aten.mul.Tensor(sum_157, 0.00015943877551020407)
    unsqueeze_1359: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1438, 0);  mul_1438 = None
    unsqueeze_1360: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1359, 2);  unsqueeze_1359 = None
    unsqueeze_1361: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, 3);  unsqueeze_1360 = None
    mul_1439: "f32[128]" = torch.ops.aten.mul.Tensor(sum_158, 0.00015943877551020407)
    mul_1440: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_1441: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1439, mul_1440);  mul_1439 = mul_1440 = None
    unsqueeze_1362: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1441, 0);  mul_1441 = None
    unsqueeze_1363: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 2);  unsqueeze_1362 = None
    unsqueeze_1364: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1363, 3);  unsqueeze_1363 = None
    mul_1442: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_80);  primals_80 = None
    unsqueeze_1365: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1442, 0);  mul_1442 = None
    unsqueeze_1366: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 2);  unsqueeze_1365 = None
    unsqueeze_1367: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, 3);  unsqueeze_1366 = None
    sub_418: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1358);  convolution_26 = unsqueeze_1358 = None
    mul_1443: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_418, unsqueeze_1364);  sub_418 = unsqueeze_1364 = None
    sub_419: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_76, mul_1443);  where_76 = mul_1443 = None
    sub_420: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_419, unsqueeze_1361);  sub_419 = unsqueeze_1361 = None
    mul_1444: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_420, unsqueeze_1367);  sub_420 = unsqueeze_1367 = None
    mul_1445: "f32[128]" = torch.ops.aten.mul.Tensor(sum_158, squeeze_79);  sum_158 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1444, relu_23, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1444 = primals_79 = None
    getitem_465: "f32[8, 256, 28, 28]" = convolution_backward_79[0]
    getitem_466: "f32[128, 256, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_621: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_620, getitem_465);  add_620 = getitem_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_333: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_334: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_333);  alias_333 = None
    le_77: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_334, 0);  alias_334 = None
    scalar_tensor_77: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_77: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_77, scalar_tensor_77, add_621);  le_77 = scalar_tensor_77 = add_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_1368: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_1369: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, 2);  unsqueeze_1368 = None
    unsqueeze_1370: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1369, 3);  unsqueeze_1369 = None
    sum_159: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_421: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_1370)
    mul_1446: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_77, sub_421);  sub_421 = None
    sum_160: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1446, [0, 2, 3]);  mul_1446 = None
    mul_1447: "f32[256]" = torch.ops.aten.mul.Tensor(sum_159, 0.00015943877551020407)
    unsqueeze_1371: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1447, 0);  mul_1447 = None
    unsqueeze_1372: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1371, 2);  unsqueeze_1371 = None
    unsqueeze_1373: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, 3);  unsqueeze_1372 = None
    mul_1448: "f32[256]" = torch.ops.aten.mul.Tensor(sum_160, 0.00015943877551020407)
    mul_1449: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_1450: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1448, mul_1449);  mul_1448 = mul_1449 = None
    unsqueeze_1374: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1450, 0);  mul_1450 = None
    unsqueeze_1375: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 2);  unsqueeze_1374 = None
    unsqueeze_1376: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1375, 3);  unsqueeze_1375 = None
    mul_1451: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_77);  primals_77 = None
    unsqueeze_1377: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1451, 0);  mul_1451 = None
    unsqueeze_1378: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 2);  unsqueeze_1377 = None
    unsqueeze_1379: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, 3);  unsqueeze_1378 = None
    sub_422: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_1370);  convolution_25 = unsqueeze_1370 = None
    mul_1452: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_422, unsqueeze_1376);  sub_422 = unsqueeze_1376 = None
    sub_423: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_77, mul_1452);  mul_1452 = None
    sub_424: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_423, unsqueeze_1373);  sub_423 = unsqueeze_1373 = None
    mul_1453: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_424, unsqueeze_1379);  sub_424 = unsqueeze_1379 = None
    mul_1454: "f32[256]" = torch.ops.aten.mul.Tensor(sum_160, squeeze_76);  sum_160 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1453, cat_2, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1453 = cat_2 = primals_76 = None
    getitem_468: "f32[8, 768, 28, 28]" = convolution_backward_80[0]
    getitem_469: "f32[256, 768, 1, 1]" = convolution_backward_80[1];  convolution_backward_80 = None
    slice_35: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_468, 1, 0, 256)
    slice_36: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_468, 1, 256, 512)
    slice_37: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_468, 1, 512, 768);  getitem_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_622: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(where_77, slice_35);  where_77 = slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_336: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_337: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_336);  alias_336 = None
    le_78: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_337, 0);  alias_337 = None
    scalar_tensor_78: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_78: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_78, scalar_tensor_78, add_622);  le_78 = scalar_tensor_78 = add_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_623: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_36, where_78);  slice_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1380: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_1381: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, 2);  unsqueeze_1380 = None
    unsqueeze_1382: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1381, 3);  unsqueeze_1381 = None
    sum_161: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_425: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1382)
    mul_1455: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_78, sub_425);  sub_425 = None
    sum_162: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1455, [0, 2, 3]);  mul_1455 = None
    mul_1456: "f32[256]" = torch.ops.aten.mul.Tensor(sum_161, 0.00015943877551020407)
    unsqueeze_1383: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1456, 0);  mul_1456 = None
    unsqueeze_1384: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1383, 2);  unsqueeze_1383 = None
    unsqueeze_1385: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, 3);  unsqueeze_1384 = None
    mul_1457: "f32[256]" = torch.ops.aten.mul.Tensor(sum_162, 0.00015943877551020407)
    mul_1458: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_1459: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1457, mul_1458);  mul_1457 = mul_1458 = None
    unsqueeze_1386: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1459, 0);  mul_1459 = None
    unsqueeze_1387: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 2);  unsqueeze_1386 = None
    unsqueeze_1388: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1387, 3);  unsqueeze_1387 = None
    mul_1460: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_74);  primals_74 = None
    unsqueeze_1389: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1460, 0);  mul_1460 = None
    unsqueeze_1390: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1389, 2);  unsqueeze_1389 = None
    unsqueeze_1391: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, 3);  unsqueeze_1390 = None
    sub_426: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_1382);  convolution_24 = unsqueeze_1382 = None
    mul_1461: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_426, unsqueeze_1388);  sub_426 = unsqueeze_1388 = None
    sub_427: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_78, mul_1461);  where_78 = mul_1461 = None
    sub_428: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_427, unsqueeze_1385);  sub_427 = unsqueeze_1385 = None
    mul_1462: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_428, unsqueeze_1391);  sub_428 = unsqueeze_1391 = None
    mul_1463: "f32[256]" = torch.ops.aten.mul.Tensor(sum_162, squeeze_73);  sum_162 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1462, relu_21, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1462 = primals_73 = None
    getitem_471: "f32[8, 128, 28, 28]" = convolution_backward_81[0]
    getitem_472: "f32[256, 128, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_339: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_340: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_339);  alias_339 = None
    le_79: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_340, 0);  alias_340 = None
    scalar_tensor_79: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_79: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_79, scalar_tensor_79, getitem_471);  le_79 = scalar_tensor_79 = getitem_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1392: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_1393: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1392, 2);  unsqueeze_1392 = None
    unsqueeze_1394: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1393, 3);  unsqueeze_1393 = None
    sum_163: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_429: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1394)
    mul_1464: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_79, sub_429);  sub_429 = None
    sum_164: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1464, [0, 2, 3]);  mul_1464 = None
    mul_1465: "f32[128]" = torch.ops.aten.mul.Tensor(sum_163, 0.00015943877551020407)
    unsqueeze_1395: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1465, 0);  mul_1465 = None
    unsqueeze_1396: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1395, 2);  unsqueeze_1395 = None
    unsqueeze_1397: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1396, 3);  unsqueeze_1396 = None
    mul_1466: "f32[128]" = torch.ops.aten.mul.Tensor(sum_164, 0.00015943877551020407)
    mul_1467: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_1468: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1466, mul_1467);  mul_1466 = mul_1467 = None
    unsqueeze_1398: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1468, 0);  mul_1468 = None
    unsqueeze_1399: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, 2);  unsqueeze_1398 = None
    unsqueeze_1400: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1399, 3);  unsqueeze_1399 = None
    mul_1469: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_71);  primals_71 = None
    unsqueeze_1401: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1469, 0);  mul_1469 = None
    unsqueeze_1402: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1401, 2);  unsqueeze_1401 = None
    unsqueeze_1403: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1402, 3);  unsqueeze_1402 = None
    sub_430: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1394);  convolution_23 = unsqueeze_1394 = None
    mul_1470: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_430, unsqueeze_1400);  sub_430 = unsqueeze_1400 = None
    sub_431: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_79, mul_1470);  where_79 = mul_1470 = None
    sub_432: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_431, unsqueeze_1397);  sub_431 = unsqueeze_1397 = None
    mul_1471: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_432, unsqueeze_1403);  sub_432 = unsqueeze_1403 = None
    mul_1472: "f32[128]" = torch.ops.aten.mul.Tensor(sum_164, squeeze_70);  sum_164 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1471, relu_20, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1471 = primals_70 = None
    getitem_474: "f32[8, 128, 28, 28]" = convolution_backward_82[0]
    getitem_475: "f32[128, 128, 3, 3]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_342: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_343: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_342);  alias_342 = None
    le_80: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_343, 0);  alias_343 = None
    scalar_tensor_80: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_80: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_80, scalar_tensor_80, getitem_474);  le_80 = scalar_tensor_80 = getitem_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1404: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_1405: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1404, 2);  unsqueeze_1404 = None
    unsqueeze_1406: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1405, 3);  unsqueeze_1405 = None
    sum_165: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_433: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1406)
    mul_1473: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_80, sub_433);  sub_433 = None
    sum_166: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1473, [0, 2, 3]);  mul_1473 = None
    mul_1474: "f32[128]" = torch.ops.aten.mul.Tensor(sum_165, 0.00015943877551020407)
    unsqueeze_1407: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1474, 0);  mul_1474 = None
    unsqueeze_1408: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1407, 2);  unsqueeze_1407 = None
    unsqueeze_1409: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1408, 3);  unsqueeze_1408 = None
    mul_1475: "f32[128]" = torch.ops.aten.mul.Tensor(sum_166, 0.00015943877551020407)
    mul_1476: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_1477: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1475, mul_1476);  mul_1475 = mul_1476 = None
    unsqueeze_1410: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1477, 0);  mul_1477 = None
    unsqueeze_1411: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, 2);  unsqueeze_1410 = None
    unsqueeze_1412: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1411, 3);  unsqueeze_1411 = None
    mul_1478: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_68);  primals_68 = None
    unsqueeze_1413: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1478, 0);  mul_1478 = None
    unsqueeze_1414: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1413, 2);  unsqueeze_1413 = None
    unsqueeze_1415: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1414, 3);  unsqueeze_1414 = None
    sub_434: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1406);  convolution_22 = unsqueeze_1406 = None
    mul_1479: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_434, unsqueeze_1412);  sub_434 = unsqueeze_1412 = None
    sub_435: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_80, mul_1479);  where_80 = mul_1479 = None
    sub_436: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_435, unsqueeze_1409);  sub_435 = unsqueeze_1409 = None
    mul_1480: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_436, unsqueeze_1415);  sub_436 = unsqueeze_1415 = None
    mul_1481: "f32[128]" = torch.ops.aten.mul.Tensor(sum_166, squeeze_67);  sum_166 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(mul_1480, relu_19, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1480 = primals_67 = None
    getitem_477: "f32[8, 256, 28, 28]" = convolution_backward_83[0]
    getitem_478: "f32[128, 256, 1, 1]" = convolution_backward_83[1];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_624: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_623, getitem_477);  add_623 = getitem_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_345: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_346: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_345);  alias_345 = None
    le_81: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_346, 0);  alias_346 = None
    scalar_tensor_81: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_81: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_81, scalar_tensor_81, add_624);  le_81 = scalar_tensor_81 = add_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_625: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_37, where_81);  slice_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1416: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_1417: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1416, 2);  unsqueeze_1416 = None
    unsqueeze_1418: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1417, 3);  unsqueeze_1417 = None
    sum_167: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
    sub_437: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1418)
    mul_1482: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_81, sub_437);  sub_437 = None
    sum_168: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1482, [0, 2, 3]);  mul_1482 = None
    mul_1483: "f32[256]" = torch.ops.aten.mul.Tensor(sum_167, 0.00015943877551020407)
    unsqueeze_1419: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1483, 0);  mul_1483 = None
    unsqueeze_1420: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1419, 2);  unsqueeze_1419 = None
    unsqueeze_1421: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1420, 3);  unsqueeze_1420 = None
    mul_1484: "f32[256]" = torch.ops.aten.mul.Tensor(sum_168, 0.00015943877551020407)
    mul_1485: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_1486: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1484, mul_1485);  mul_1484 = mul_1485 = None
    unsqueeze_1422: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1486, 0);  mul_1486 = None
    unsqueeze_1423: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, 2);  unsqueeze_1422 = None
    unsqueeze_1424: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1423, 3);  unsqueeze_1423 = None
    mul_1487: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_65);  primals_65 = None
    unsqueeze_1425: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1487, 0);  mul_1487 = None
    unsqueeze_1426: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1425, 2);  unsqueeze_1425 = None
    unsqueeze_1427: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1426, 3);  unsqueeze_1426 = None
    sub_438: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1418);  convolution_21 = unsqueeze_1418 = None
    mul_1488: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_438, unsqueeze_1424);  sub_438 = unsqueeze_1424 = None
    sub_439: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_81, mul_1488);  where_81 = mul_1488 = None
    sub_440: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_439, unsqueeze_1421);  sub_439 = unsqueeze_1421 = None
    mul_1489: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_440, unsqueeze_1427);  sub_440 = unsqueeze_1427 = None
    mul_1490: "f32[256]" = torch.ops.aten.mul.Tensor(sum_168, squeeze_64);  sum_168 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(mul_1489, relu_18, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1489 = primals_64 = None
    getitem_480: "f32[8, 128, 28, 28]" = convolution_backward_84[0]
    getitem_481: "f32[256, 128, 1, 1]" = convolution_backward_84[1];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_348: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_349: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_348);  alias_348 = None
    le_82: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_349, 0);  alias_349 = None
    scalar_tensor_82: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_82: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_82, scalar_tensor_82, getitem_480);  le_82 = scalar_tensor_82 = getitem_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1428: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_1429: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1428, 2);  unsqueeze_1428 = None
    unsqueeze_1430: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1429, 3);  unsqueeze_1429 = None
    sum_169: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
    sub_441: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_1430)
    mul_1491: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_82, sub_441);  sub_441 = None
    sum_170: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1491, [0, 2, 3]);  mul_1491 = None
    mul_1492: "f32[128]" = torch.ops.aten.mul.Tensor(sum_169, 0.00015943877551020407)
    unsqueeze_1431: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1492, 0);  mul_1492 = None
    unsqueeze_1432: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1431, 2);  unsqueeze_1431 = None
    unsqueeze_1433: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, 3);  unsqueeze_1432 = None
    mul_1493: "f32[128]" = torch.ops.aten.mul.Tensor(sum_170, 0.00015943877551020407)
    mul_1494: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_1495: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1493, mul_1494);  mul_1493 = mul_1494 = None
    unsqueeze_1434: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1495, 0);  mul_1495 = None
    unsqueeze_1435: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, 2);  unsqueeze_1434 = None
    unsqueeze_1436: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1435, 3);  unsqueeze_1435 = None
    mul_1496: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_62);  primals_62 = None
    unsqueeze_1437: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1496, 0);  mul_1496 = None
    unsqueeze_1438: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1437, 2);  unsqueeze_1437 = None
    unsqueeze_1439: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, 3);  unsqueeze_1438 = None
    sub_442: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_1430);  convolution_20 = unsqueeze_1430 = None
    mul_1497: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_442, unsqueeze_1436);  sub_442 = unsqueeze_1436 = None
    sub_443: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_82, mul_1497);  where_82 = mul_1497 = None
    sub_444: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_443, unsqueeze_1433);  sub_443 = unsqueeze_1433 = None
    mul_1498: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_444, unsqueeze_1439);  sub_444 = unsqueeze_1439 = None
    mul_1499: "f32[128]" = torch.ops.aten.mul.Tensor(sum_170, squeeze_61);  sum_170 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1498, relu_17, primals_61, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1498 = primals_61 = None
    getitem_483: "f32[8, 128, 28, 28]" = convolution_backward_85[0]
    getitem_484: "f32[128, 128, 3, 3]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_351: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_352: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_351);  alias_351 = None
    le_83: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_352, 0);  alias_352 = None
    scalar_tensor_83: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_83: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_83, scalar_tensor_83, getitem_483);  le_83 = scalar_tensor_83 = getitem_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1440: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_1441: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1440, 2);  unsqueeze_1440 = None
    unsqueeze_1442: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1441, 3);  unsqueeze_1441 = None
    sum_171: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
    sub_445: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1442)
    mul_1500: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_83, sub_445);  sub_445 = None
    sum_172: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1500, [0, 2, 3]);  mul_1500 = None
    mul_1501: "f32[128]" = torch.ops.aten.mul.Tensor(sum_171, 0.00015943877551020407)
    unsqueeze_1443: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1501, 0);  mul_1501 = None
    unsqueeze_1444: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1443, 2);  unsqueeze_1443 = None
    unsqueeze_1445: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, 3);  unsqueeze_1444 = None
    mul_1502: "f32[128]" = torch.ops.aten.mul.Tensor(sum_172, 0.00015943877551020407)
    mul_1503: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_1504: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1502, mul_1503);  mul_1502 = mul_1503 = None
    unsqueeze_1446: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1504, 0);  mul_1504 = None
    unsqueeze_1447: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, 2);  unsqueeze_1446 = None
    unsqueeze_1448: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1447, 3);  unsqueeze_1447 = None
    mul_1505: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_59);  primals_59 = None
    unsqueeze_1449: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1505, 0);  mul_1505 = None
    unsqueeze_1450: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1449, 2);  unsqueeze_1449 = None
    unsqueeze_1451: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, 3);  unsqueeze_1450 = None
    sub_446: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_1442);  convolution_19 = unsqueeze_1442 = None
    mul_1506: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_446, unsqueeze_1448);  sub_446 = unsqueeze_1448 = None
    sub_447: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_83, mul_1506);  where_83 = mul_1506 = None
    sub_448: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_447, unsqueeze_1445);  sub_447 = unsqueeze_1445 = None
    mul_1507: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_448, unsqueeze_1451);  sub_448 = unsqueeze_1451 = None
    mul_1508: "f32[128]" = torch.ops.aten.mul.Tensor(sum_172, squeeze_58);  sum_172 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1507, relu_16, primals_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1507 = primals_58 = None
    getitem_486: "f32[8, 256, 28, 28]" = convolution_backward_86[0]
    getitem_487: "f32[128, 256, 1, 1]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_626: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_625, getitem_486);  add_625 = getitem_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_354: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_355: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_354);  alias_354 = None
    le_84: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_355, 0);  alias_355 = None
    scalar_tensor_84: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_84: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_84, scalar_tensor_84, add_626);  le_84 = scalar_tensor_84 = add_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_1452: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_1453: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1452, 2);  unsqueeze_1452 = None
    unsqueeze_1454: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1453, 3);  unsqueeze_1453 = None
    sum_173: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
    sub_449: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1454)
    mul_1509: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_84, sub_449);  sub_449 = None
    sum_174: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1509, [0, 2, 3]);  mul_1509 = None
    mul_1510: "f32[256]" = torch.ops.aten.mul.Tensor(sum_173, 0.00015943877551020407)
    unsqueeze_1455: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1510, 0);  mul_1510 = None
    unsqueeze_1456: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1455, 2);  unsqueeze_1455 = None
    unsqueeze_1457: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, 3);  unsqueeze_1456 = None
    mul_1511: "f32[256]" = torch.ops.aten.mul.Tensor(sum_174, 0.00015943877551020407)
    mul_1512: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1513: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1511, mul_1512);  mul_1511 = mul_1512 = None
    unsqueeze_1458: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1513, 0);  mul_1513 = None
    unsqueeze_1459: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, 2);  unsqueeze_1458 = None
    unsqueeze_1460: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1459, 3);  unsqueeze_1459 = None
    mul_1514: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_56);  primals_56 = None
    unsqueeze_1461: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1514, 0);  mul_1514 = None
    unsqueeze_1462: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1461, 2);  unsqueeze_1461 = None
    unsqueeze_1463: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, 3);  unsqueeze_1462 = None
    sub_450: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1454);  convolution_18 = unsqueeze_1454 = None
    mul_1515: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_450, unsqueeze_1460);  sub_450 = unsqueeze_1460 = None
    sub_451: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_84, mul_1515);  mul_1515 = None
    sub_452: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_451, unsqueeze_1457);  sub_451 = unsqueeze_1457 = None
    mul_1516: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_452, unsqueeze_1463);  sub_452 = unsqueeze_1463 = None
    mul_1517: "f32[256]" = torch.ops.aten.mul.Tensor(sum_174, squeeze_55);  sum_174 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1516, cat_1, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1516 = cat_1 = primals_55 = None
    getitem_489: "f32[8, 512, 28, 28]" = convolution_backward_87[0]
    getitem_490: "f32[256, 512, 1, 1]" = convolution_backward_87[1];  convolution_backward_87 = None
    slice_38: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_489, 1, 0, 256)
    slice_39: "f32[8, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_489, 1, 256, 512);  getitem_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_627: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(where_84, slice_38);  where_84 = slice_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_357: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_358: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_357);  alias_357 = None
    le_85: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_358, 0);  alias_358 = None
    scalar_tensor_85: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_85: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_85, scalar_tensor_85, add_627);  le_85 = scalar_tensor_85 = add_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_628: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_39, where_85);  slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1464: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_1465: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1464, 2);  unsqueeze_1464 = None
    unsqueeze_1466: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1465, 3);  unsqueeze_1465 = None
    sum_175: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_453: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1466)
    mul_1518: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_85, sub_453);  sub_453 = None
    sum_176: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1518, [0, 2, 3]);  mul_1518 = None
    mul_1519: "f32[256]" = torch.ops.aten.mul.Tensor(sum_175, 0.00015943877551020407)
    unsqueeze_1467: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1519, 0);  mul_1519 = None
    unsqueeze_1468: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1467, 2);  unsqueeze_1467 = None
    unsqueeze_1469: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, 3);  unsqueeze_1468 = None
    mul_1520: "f32[256]" = torch.ops.aten.mul.Tensor(sum_176, 0.00015943877551020407)
    mul_1521: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1522: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1520, mul_1521);  mul_1520 = mul_1521 = None
    unsqueeze_1470: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1522, 0);  mul_1522 = None
    unsqueeze_1471: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, 2);  unsqueeze_1470 = None
    unsqueeze_1472: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1471, 3);  unsqueeze_1471 = None
    mul_1523: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_53);  primals_53 = None
    unsqueeze_1473: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1523, 0);  mul_1523 = None
    unsqueeze_1474: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1473, 2);  unsqueeze_1473 = None
    unsqueeze_1475: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, 3);  unsqueeze_1474 = None
    sub_454: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1466);  convolution_17 = unsqueeze_1466 = None
    mul_1524: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_454, unsqueeze_1472);  sub_454 = unsqueeze_1472 = None
    sub_455: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_85, mul_1524);  where_85 = mul_1524 = None
    sub_456: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_455, unsqueeze_1469);  sub_455 = unsqueeze_1469 = None
    mul_1525: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_456, unsqueeze_1475);  sub_456 = unsqueeze_1475 = None
    mul_1526: "f32[256]" = torch.ops.aten.mul.Tensor(sum_176, squeeze_52);  sum_176 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(mul_1525, relu_14, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1525 = primals_52 = None
    getitem_492: "f32[8, 128, 28, 28]" = convolution_backward_88[0]
    getitem_493: "f32[256, 128, 1, 1]" = convolution_backward_88[1];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_360: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_361: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_360);  alias_360 = None
    le_86: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_361, 0);  alias_361 = None
    scalar_tensor_86: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_86: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_86, scalar_tensor_86, getitem_492);  le_86 = scalar_tensor_86 = getitem_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1476: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_1477: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1476, 2);  unsqueeze_1476 = None
    unsqueeze_1478: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1477, 3);  unsqueeze_1477 = None
    sum_177: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
    sub_457: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1478)
    mul_1527: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_86, sub_457);  sub_457 = None
    sum_178: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1527, [0, 2, 3]);  mul_1527 = None
    mul_1528: "f32[128]" = torch.ops.aten.mul.Tensor(sum_177, 0.00015943877551020407)
    unsqueeze_1479: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1528, 0);  mul_1528 = None
    unsqueeze_1480: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1479, 2);  unsqueeze_1479 = None
    unsqueeze_1481: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1480, 3);  unsqueeze_1480 = None
    mul_1529: "f32[128]" = torch.ops.aten.mul.Tensor(sum_178, 0.00015943877551020407)
    mul_1530: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1531: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1529, mul_1530);  mul_1529 = mul_1530 = None
    unsqueeze_1482: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1531, 0);  mul_1531 = None
    unsqueeze_1483: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, 2);  unsqueeze_1482 = None
    unsqueeze_1484: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1483, 3);  unsqueeze_1483 = None
    mul_1532: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_50);  primals_50 = None
    unsqueeze_1485: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1532, 0);  mul_1532 = None
    unsqueeze_1486: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1485, 2);  unsqueeze_1485 = None
    unsqueeze_1487: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1486, 3);  unsqueeze_1486 = None
    sub_458: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1478);  convolution_16 = unsqueeze_1478 = None
    mul_1533: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_458, unsqueeze_1484);  sub_458 = unsqueeze_1484 = None
    sub_459: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_86, mul_1533);  where_86 = mul_1533 = None
    sub_460: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_459, unsqueeze_1481);  sub_459 = unsqueeze_1481 = None
    mul_1534: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_460, unsqueeze_1487);  sub_460 = unsqueeze_1487 = None
    mul_1535: "f32[128]" = torch.ops.aten.mul.Tensor(sum_178, squeeze_49);  sum_178 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(mul_1534, relu_13, primals_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1534 = primals_49 = None
    getitem_495: "f32[8, 128, 28, 28]" = convolution_backward_89[0]
    getitem_496: "f32[128, 128, 3, 3]" = convolution_backward_89[1];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_363: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_364: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_363);  alias_363 = None
    le_87: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_364, 0);  alias_364 = None
    scalar_tensor_87: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_87: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_87, scalar_tensor_87, getitem_495);  le_87 = scalar_tensor_87 = getitem_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1488: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_1489: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1488, 2);  unsqueeze_1488 = None
    unsqueeze_1490: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1489, 3);  unsqueeze_1489 = None
    sum_179: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_461: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1490)
    mul_1536: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_87, sub_461);  sub_461 = None
    sum_180: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1536, [0, 2, 3]);  mul_1536 = None
    mul_1537: "f32[128]" = torch.ops.aten.mul.Tensor(sum_179, 0.00015943877551020407)
    unsqueeze_1491: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1537, 0);  mul_1537 = None
    unsqueeze_1492: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1491, 2);  unsqueeze_1491 = None
    unsqueeze_1493: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1492, 3);  unsqueeze_1492 = None
    mul_1538: "f32[128]" = torch.ops.aten.mul.Tensor(sum_180, 0.00015943877551020407)
    mul_1539: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1540: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1538, mul_1539);  mul_1538 = mul_1539 = None
    unsqueeze_1494: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1540, 0);  mul_1540 = None
    unsqueeze_1495: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, 2);  unsqueeze_1494 = None
    unsqueeze_1496: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1495, 3);  unsqueeze_1495 = None
    mul_1541: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_47);  primals_47 = None
    unsqueeze_1497: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1541, 0);  mul_1541 = None
    unsqueeze_1498: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1497, 2);  unsqueeze_1497 = None
    unsqueeze_1499: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, 3);  unsqueeze_1498 = None
    sub_462: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1490);  convolution_15 = unsqueeze_1490 = None
    mul_1542: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_462, unsqueeze_1496);  sub_462 = unsqueeze_1496 = None
    sub_463: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_87, mul_1542);  where_87 = mul_1542 = None
    sub_464: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_463, unsqueeze_1493);  sub_463 = unsqueeze_1493 = None
    mul_1543: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_464, unsqueeze_1499);  sub_464 = unsqueeze_1499 = None
    mul_1544: "f32[128]" = torch.ops.aten.mul.Tensor(sum_180, squeeze_46);  sum_180 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1543, relu_12, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1543 = primals_46 = None
    getitem_498: "f32[8, 256, 28, 28]" = convolution_backward_90[0]
    getitem_499: "f32[128, 256, 1, 1]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_629: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_628, getitem_498);  add_628 = getitem_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_366: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_367: "f32[8, 256, 28, 28]" = torch.ops.aten.alias.default(alias_366);  alias_366 = None
    le_88: "b8[8, 256, 28, 28]" = torch.ops.aten.le.Scalar(alias_367, 0);  alias_367 = None
    scalar_tensor_88: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_88: "f32[8, 256, 28, 28]" = torch.ops.aten.where.self(le_88, scalar_tensor_88, add_629);  le_88 = scalar_tensor_88 = add_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1500: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_1501: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1500, 2);  unsqueeze_1500 = None
    unsqueeze_1502: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1501, 3);  unsqueeze_1501 = None
    sum_181: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_465: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1502)
    mul_1545: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_88, sub_465);  sub_465 = None
    sum_182: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1545, [0, 2, 3]);  mul_1545 = None
    mul_1546: "f32[256]" = torch.ops.aten.mul.Tensor(sum_181, 0.00015943877551020407)
    unsqueeze_1503: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1546, 0);  mul_1546 = None
    unsqueeze_1504: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1503, 2);  unsqueeze_1503 = None
    unsqueeze_1505: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1504, 3);  unsqueeze_1504 = None
    mul_1547: "f32[256]" = torch.ops.aten.mul.Tensor(sum_182, 0.00015943877551020407)
    mul_1548: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1549: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1547, mul_1548);  mul_1547 = mul_1548 = None
    unsqueeze_1506: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1549, 0);  mul_1549 = None
    unsqueeze_1507: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, 2);  unsqueeze_1506 = None
    unsqueeze_1508: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1507, 3);  unsqueeze_1507 = None
    mul_1550: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_44);  primals_44 = None
    unsqueeze_1509: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1550, 0);  mul_1550 = None
    unsqueeze_1510: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1509, 2);  unsqueeze_1509 = None
    unsqueeze_1511: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1510, 3);  unsqueeze_1510 = None
    sub_466: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1502);  convolution_14 = unsqueeze_1502 = None
    mul_1551: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_466, unsqueeze_1508);  sub_466 = unsqueeze_1508 = None
    sub_467: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_88, mul_1551);  mul_1551 = None
    sub_468: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_467, unsqueeze_1505);  sub_467 = unsqueeze_1505 = None
    mul_1552: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_468, unsqueeze_1511);  sub_468 = unsqueeze_1511 = None
    mul_1553: "f32[256]" = torch.ops.aten.mul.Tensor(sum_182, squeeze_43);  sum_182 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1552, relu_11, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1552 = primals_43 = None
    getitem_501: "f32[8, 128, 28, 28]" = convolution_backward_91[0]
    getitem_502: "f32[256, 128, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_369: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_370: "f32[8, 128, 28, 28]" = torch.ops.aten.alias.default(alias_369);  alias_369 = None
    le_89: "b8[8, 128, 28, 28]" = torch.ops.aten.le.Scalar(alias_370, 0);  alias_370 = None
    scalar_tensor_89: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_89: "f32[8, 128, 28, 28]" = torch.ops.aten.where.self(le_89, scalar_tensor_89, getitem_501);  le_89 = scalar_tensor_89 = getitem_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1512: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_1513: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1512, 2);  unsqueeze_1512 = None
    unsqueeze_1514: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1513, 3);  unsqueeze_1513 = None
    sum_183: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
    sub_469: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1514)
    mul_1554: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(where_89, sub_469);  sub_469 = None
    sum_184: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1554, [0, 2, 3]);  mul_1554 = None
    mul_1555: "f32[128]" = torch.ops.aten.mul.Tensor(sum_183, 0.00015943877551020407)
    unsqueeze_1515: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1555, 0);  mul_1555 = None
    unsqueeze_1516: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1515, 2);  unsqueeze_1515 = None
    unsqueeze_1517: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1516, 3);  unsqueeze_1516 = None
    mul_1556: "f32[128]" = torch.ops.aten.mul.Tensor(sum_184, 0.00015943877551020407)
    mul_1557: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1558: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1556, mul_1557);  mul_1556 = mul_1557 = None
    unsqueeze_1518: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1558, 0);  mul_1558 = None
    unsqueeze_1519: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, 2);  unsqueeze_1518 = None
    unsqueeze_1520: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1519, 3);  unsqueeze_1519 = None
    mul_1559: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_41);  primals_41 = None
    unsqueeze_1521: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1559, 0);  mul_1559 = None
    unsqueeze_1522: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1521, 2);  unsqueeze_1521 = None
    unsqueeze_1523: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1522, 3);  unsqueeze_1522 = None
    sub_470: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1514);  convolution_13 = unsqueeze_1514 = None
    mul_1560: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_470, unsqueeze_1520);  sub_470 = unsqueeze_1520 = None
    sub_471: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(where_89, mul_1560);  where_89 = mul_1560 = None
    sub_472: "f32[8, 128, 28, 28]" = torch.ops.aten.sub.Tensor(sub_471, unsqueeze_1517);  sub_471 = unsqueeze_1517 = None
    mul_1561: "f32[8, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_472, unsqueeze_1523);  sub_472 = unsqueeze_1523 = None
    mul_1562: "f32[128]" = torch.ops.aten.mul.Tensor(sum_184, squeeze_40);  sum_184 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1561, relu_10, primals_40, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1561 = primals_40 = None
    getitem_504: "f32[8, 128, 56, 56]" = convolution_backward_92[0]
    getitem_505: "f32[128, 128, 3, 3]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_372: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_373: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(alias_372);  alias_372 = None
    le_90: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_373, 0);  alias_373 = None
    scalar_tensor_90: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_90: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_90, scalar_tensor_90, getitem_504);  le_90 = scalar_tensor_90 = getitem_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1524: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_1525: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1524, 2);  unsqueeze_1524 = None
    unsqueeze_1526: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1525, 3);  unsqueeze_1525 = None
    sum_185: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
    sub_473: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1526)
    mul_1563: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_90, sub_473);  sub_473 = None
    sum_186: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1563, [0, 2, 3]);  mul_1563 = None
    mul_1564: "f32[128]" = torch.ops.aten.mul.Tensor(sum_185, 3.985969387755102e-05)
    unsqueeze_1527: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1564, 0);  mul_1564 = None
    unsqueeze_1528: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1527, 2);  unsqueeze_1527 = None
    unsqueeze_1529: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1528, 3);  unsqueeze_1528 = None
    mul_1565: "f32[128]" = torch.ops.aten.mul.Tensor(sum_186, 3.985969387755102e-05)
    mul_1566: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1567: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1565, mul_1566);  mul_1565 = mul_1566 = None
    unsqueeze_1530: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1567, 0);  mul_1567 = None
    unsqueeze_1531: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, 2);  unsqueeze_1530 = None
    unsqueeze_1532: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1531, 3);  unsqueeze_1531 = None
    mul_1568: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_38);  primals_38 = None
    unsqueeze_1533: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1568, 0);  mul_1568 = None
    unsqueeze_1534: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1533, 2);  unsqueeze_1533 = None
    unsqueeze_1535: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1534, 3);  unsqueeze_1534 = None
    sub_474: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1526);  convolution_12 = unsqueeze_1526 = None
    mul_1569: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_474, unsqueeze_1532);  sub_474 = unsqueeze_1532 = None
    sub_475: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_90, mul_1569);  where_90 = mul_1569 = None
    sub_476: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_475, unsqueeze_1529);  sub_475 = unsqueeze_1529 = None
    mul_1570: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_476, unsqueeze_1535);  sub_476 = unsqueeze_1535 = None
    mul_1571: "f32[128]" = torch.ops.aten.mul.Tensor(sum_186, squeeze_37);  sum_186 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(mul_1570, relu_9, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1570 = primals_37 = None
    getitem_507: "f32[8, 128, 56, 56]" = convolution_backward_93[0]
    getitem_508: "f32[128, 128, 1, 1]" = convolution_backward_93[1];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    unsqueeze_1536: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_1537: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1536, 2);  unsqueeze_1536 = None
    unsqueeze_1538: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1537, 3);  unsqueeze_1537 = None
    sum_187: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_477: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1538)
    mul_1572: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(where_88, sub_477);  sub_477 = None
    sum_188: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1572, [0, 2, 3]);  mul_1572 = None
    mul_1573: "f32[256]" = torch.ops.aten.mul.Tensor(sum_187, 0.00015943877551020407)
    unsqueeze_1539: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1573, 0);  mul_1573 = None
    unsqueeze_1540: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1539, 2);  unsqueeze_1539 = None
    unsqueeze_1541: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1540, 3);  unsqueeze_1540 = None
    mul_1574: "f32[256]" = torch.ops.aten.mul.Tensor(sum_188, 0.00015943877551020407)
    mul_1575: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1576: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1574, mul_1575);  mul_1574 = mul_1575 = None
    unsqueeze_1542: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1576, 0);  mul_1576 = None
    unsqueeze_1543: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, 2);  unsqueeze_1542 = None
    unsqueeze_1544: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1543, 3);  unsqueeze_1543 = None
    mul_1577: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_35);  primals_35 = None
    unsqueeze_1545: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1577, 0);  mul_1577 = None
    unsqueeze_1546: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1545, 2);  unsqueeze_1545 = None
    unsqueeze_1547: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1546, 3);  unsqueeze_1546 = None
    sub_478: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1538);  convolution_11 = unsqueeze_1538 = None
    mul_1578: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_478, unsqueeze_1544);  sub_478 = unsqueeze_1544 = None
    sub_479: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(where_88, mul_1578);  where_88 = mul_1578 = None
    sub_480: "f32[8, 256, 28, 28]" = torch.ops.aten.sub.Tensor(sub_479, unsqueeze_1541);  sub_479 = unsqueeze_1541 = None
    mul_1579: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_480, unsqueeze_1547);  sub_480 = unsqueeze_1547 = None
    mul_1580: "f32[256]" = torch.ops.aten.mul.Tensor(sum_188, squeeze_34);  sum_188 = squeeze_34 = None
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(mul_1579, getitem_28, primals_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1579 = getitem_28 = primals_34 = None
    getitem_510: "f32[8, 128, 28, 28]" = convolution_backward_94[0]
    getitem_511: "f32[256, 128, 1, 1]" = convolution_backward_94[1];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_backward_3: "f32[8, 128, 56, 56]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_510, relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_29);  getitem_510 = getitem_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    add_630: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(getitem_507, max_pool2d_with_indices_backward_3);  getitem_507 = max_pool2d_with_indices_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_backward_4: "f32[8, 128, 56, 56]" = torch.ops.aten.max_pool2d_with_indices_backward.default(slice_30, relu_9, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_25);  slice_30 = getitem_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    add_631: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(add_630, max_pool2d_with_indices_backward_4);  add_630 = max_pool2d_with_indices_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:180, code: x = self.relu(x)
    alias_375: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_376: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(alias_375);  alias_375 = None
    le_91: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_376, 0);  alias_376 = None
    scalar_tensor_91: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_91: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_91, scalar_tensor_91, add_631);  le_91 = scalar_tensor_91 = add_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:177, code: x = self.bn(x)
    unsqueeze_1548: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_1549: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1548, 2);  unsqueeze_1548 = None
    unsqueeze_1550: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1549, 3);  unsqueeze_1549 = None
    sum_189: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
    sub_481: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1550)
    mul_1581: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_91, sub_481);  sub_481 = None
    sum_190: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1581, [0, 2, 3]);  mul_1581 = None
    mul_1582: "f32[128]" = torch.ops.aten.mul.Tensor(sum_189, 3.985969387755102e-05)
    unsqueeze_1551: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1582, 0);  mul_1582 = None
    unsqueeze_1552: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1551, 2);  unsqueeze_1551 = None
    unsqueeze_1553: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1552, 3);  unsqueeze_1552 = None
    mul_1583: "f32[128]" = torch.ops.aten.mul.Tensor(sum_190, 3.985969387755102e-05)
    mul_1584: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1585: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1583, mul_1584);  mul_1583 = mul_1584 = None
    unsqueeze_1554: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1585, 0);  mul_1585 = None
    unsqueeze_1555: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, 2);  unsqueeze_1554 = None
    unsqueeze_1556: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1555, 3);  unsqueeze_1555 = None
    mul_1586: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_32);  primals_32 = None
    unsqueeze_1557: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1586, 0);  mul_1586 = None
    unsqueeze_1558: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1557, 2);  unsqueeze_1557 = None
    unsqueeze_1559: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1558, 3);  unsqueeze_1558 = None
    sub_482: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1550);  convolution_10 = unsqueeze_1550 = None
    mul_1587: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_482, unsqueeze_1556);  sub_482 = unsqueeze_1556 = None
    sub_483: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_91, mul_1587);  mul_1587 = None
    sub_484: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_483, unsqueeze_1553);  sub_483 = unsqueeze_1553 = None
    mul_1588: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_484, unsqueeze_1559);  sub_484 = unsqueeze_1559 = None
    mul_1589: "f32[128]" = torch.ops.aten.mul.Tensor(sum_190, squeeze_31);  sum_190 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1588, cat, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1588 = cat = primals_31 = None
    getitem_513: "f32[8, 256, 56, 56]" = convolution_backward_95[0]
    getitem_514: "f32[128, 256, 1, 1]" = convolution_backward_95[1];  convolution_backward_95 = None
    slice_40: "f32[8, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_513, 1, 0, 128)
    slice_41: "f32[8, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_513, 1, 128, 256);  getitem_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:176, code: x = self.conv(torch.cat(x_children, 1))
    add_632: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(where_91, slice_40);  where_91 = slice_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_378: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_379: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(alias_378);  alias_378 = None
    le_92: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_379, 0);  alias_379 = None
    scalar_tensor_92: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_92: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_92, scalar_tensor_92, add_632);  le_92 = scalar_tensor_92 = add_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:91, code: out += shortcut
    add_633: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_41, where_92);  slice_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1560: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_1561: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1560, 2);  unsqueeze_1560 = None
    unsqueeze_1562: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1561, 3);  unsqueeze_1561 = None
    sum_191: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_485: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1562)
    mul_1590: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_92, sub_485);  sub_485 = None
    sum_192: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1590, [0, 2, 3]);  mul_1590 = None
    mul_1591: "f32[128]" = torch.ops.aten.mul.Tensor(sum_191, 3.985969387755102e-05)
    unsqueeze_1563: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1591, 0);  mul_1591 = None
    unsqueeze_1564: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1563, 2);  unsqueeze_1563 = None
    unsqueeze_1565: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1564, 3);  unsqueeze_1564 = None
    mul_1592: "f32[128]" = torch.ops.aten.mul.Tensor(sum_192, 3.985969387755102e-05)
    mul_1593: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1594: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1592, mul_1593);  mul_1592 = mul_1593 = None
    unsqueeze_1566: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1594, 0);  mul_1594 = None
    unsqueeze_1567: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, 2);  unsqueeze_1566 = None
    unsqueeze_1568: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1567, 3);  unsqueeze_1567 = None
    mul_1595: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_29);  primals_29 = None
    unsqueeze_1569: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1595, 0);  mul_1595 = None
    unsqueeze_1570: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1569, 2);  unsqueeze_1569 = None
    unsqueeze_1571: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1570, 3);  unsqueeze_1570 = None
    sub_486: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1562);  convolution_9 = unsqueeze_1562 = None
    mul_1596: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_486, unsqueeze_1568);  sub_486 = unsqueeze_1568 = None
    sub_487: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_92, mul_1596);  where_92 = mul_1596 = None
    sub_488: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_487, unsqueeze_1565);  sub_487 = unsqueeze_1565 = None
    mul_1597: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_488, unsqueeze_1571);  sub_488 = unsqueeze_1571 = None
    mul_1598: "f32[128]" = torch.ops.aten.mul.Tensor(sum_192, squeeze_28);  sum_192 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(mul_1597, relu_7, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1597 = primals_28 = None
    getitem_516: "f32[8, 64, 56, 56]" = convolution_backward_96[0]
    getitem_517: "f32[128, 64, 1, 1]" = convolution_backward_96[1];  convolution_backward_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_381: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_382: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(alias_381);  alias_381 = None
    le_93: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_382, 0);  alias_382 = None
    scalar_tensor_93: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_93: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_93, scalar_tensor_93, getitem_516);  le_93 = scalar_tensor_93 = getitem_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1572: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_1573: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1572, 2);  unsqueeze_1572 = None
    unsqueeze_1574: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1573, 3);  unsqueeze_1573 = None
    sum_193: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
    sub_489: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1574)
    mul_1599: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_93, sub_489);  sub_489 = None
    sum_194: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1599, [0, 2, 3]);  mul_1599 = None
    mul_1600: "f32[64]" = torch.ops.aten.mul.Tensor(sum_193, 3.985969387755102e-05)
    unsqueeze_1575: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1600, 0);  mul_1600 = None
    unsqueeze_1576: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1575, 2);  unsqueeze_1575 = None
    unsqueeze_1577: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1576, 3);  unsqueeze_1576 = None
    mul_1601: "f32[64]" = torch.ops.aten.mul.Tensor(sum_194, 3.985969387755102e-05)
    mul_1602: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1603: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1601, mul_1602);  mul_1601 = mul_1602 = None
    unsqueeze_1578: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1603, 0);  mul_1603 = None
    unsqueeze_1579: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, 2);  unsqueeze_1578 = None
    unsqueeze_1580: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1579, 3);  unsqueeze_1579 = None
    mul_1604: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_26);  primals_26 = None
    unsqueeze_1581: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1604, 0);  mul_1604 = None
    unsqueeze_1582: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1581, 2);  unsqueeze_1581 = None
    unsqueeze_1583: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1582, 3);  unsqueeze_1582 = None
    sub_490: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1574);  convolution_8 = unsqueeze_1574 = None
    mul_1605: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_490, unsqueeze_1580);  sub_490 = unsqueeze_1580 = None
    sub_491: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_93, mul_1605);  where_93 = mul_1605 = None
    sub_492: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_491, unsqueeze_1577);  sub_491 = unsqueeze_1577 = None
    mul_1606: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_492, unsqueeze_1583);  sub_492 = unsqueeze_1583 = None
    mul_1607: "f32[64]" = torch.ops.aten.mul.Tensor(sum_194, squeeze_25);  sum_194 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(mul_1606, relu_6, primals_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1606 = primals_25 = None
    getitem_519: "f32[8, 64, 56, 56]" = convolution_backward_97[0]
    getitem_520: "f32[64, 64, 3, 3]" = convolution_backward_97[1];  convolution_backward_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_384: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_385: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(alias_384);  alias_384 = None
    le_94: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_385, 0);  alias_385 = None
    scalar_tensor_94: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_94: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_94, scalar_tensor_94, getitem_519);  le_94 = scalar_tensor_94 = getitem_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1584: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_1585: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1584, 2);  unsqueeze_1584 = None
    unsqueeze_1586: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1585, 3);  unsqueeze_1585 = None
    sum_195: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_493: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1586)
    mul_1608: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_94, sub_493);  sub_493 = None
    sum_196: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1608, [0, 2, 3]);  mul_1608 = None
    mul_1609: "f32[64]" = torch.ops.aten.mul.Tensor(sum_195, 3.985969387755102e-05)
    unsqueeze_1587: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1609, 0);  mul_1609 = None
    unsqueeze_1588: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1587, 2);  unsqueeze_1587 = None
    unsqueeze_1589: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1588, 3);  unsqueeze_1588 = None
    mul_1610: "f32[64]" = torch.ops.aten.mul.Tensor(sum_196, 3.985969387755102e-05)
    mul_1611: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1612: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1610, mul_1611);  mul_1610 = mul_1611 = None
    unsqueeze_1590: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1612, 0);  mul_1612 = None
    unsqueeze_1591: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, 2);  unsqueeze_1590 = None
    unsqueeze_1592: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1591, 3);  unsqueeze_1591 = None
    mul_1613: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_23);  primals_23 = None
    unsqueeze_1593: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1613, 0);  mul_1613 = None
    unsqueeze_1594: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1593, 2);  unsqueeze_1593 = None
    unsqueeze_1595: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1594, 3);  unsqueeze_1594 = None
    sub_494: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1586);  convolution_7 = unsqueeze_1586 = None
    mul_1614: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_494, unsqueeze_1592);  sub_494 = unsqueeze_1592 = None
    sub_495: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_94, mul_1614);  where_94 = mul_1614 = None
    sub_496: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_495, unsqueeze_1589);  sub_495 = unsqueeze_1589 = None
    mul_1615: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_496, unsqueeze_1595);  sub_496 = unsqueeze_1595 = None
    mul_1616: "f32[64]" = torch.ops.aten.mul.Tensor(sum_196, squeeze_22);  sum_196 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(mul_1615, relu_5, primals_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1615 = primals_22 = None
    getitem_522: "f32[8, 128, 56, 56]" = convolution_backward_98[0]
    getitem_523: "f32[64, 128, 1, 1]" = convolution_backward_98[1];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    add_634: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(add_633, getitem_522);  add_633 = getitem_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:92, code: out = self.relu(out)
    alias_387: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_388: "f32[8, 128, 56, 56]" = torch.ops.aten.alias.default(alias_387);  alias_387 = None
    le_95: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(alias_388, 0);  alias_388 = None
    scalar_tensor_95: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_95: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_95, scalar_tensor_95, add_634);  le_95 = scalar_tensor_95 = add_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:89, code: out = self.bn3(out)
    unsqueeze_1596: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_1597: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1596, 2);  unsqueeze_1596 = None
    unsqueeze_1598: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1597, 3);  unsqueeze_1597 = None
    sum_197: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
    sub_497: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1598)
    mul_1617: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_95, sub_497);  sub_497 = None
    sum_198: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1617, [0, 2, 3]);  mul_1617 = None
    mul_1618: "f32[128]" = torch.ops.aten.mul.Tensor(sum_197, 3.985969387755102e-05)
    unsqueeze_1599: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1618, 0);  mul_1618 = None
    unsqueeze_1600: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1599, 2);  unsqueeze_1599 = None
    unsqueeze_1601: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1600, 3);  unsqueeze_1600 = None
    mul_1619: "f32[128]" = torch.ops.aten.mul.Tensor(sum_198, 3.985969387755102e-05)
    mul_1620: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1621: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1619, mul_1620);  mul_1619 = mul_1620 = None
    unsqueeze_1602: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1621, 0);  mul_1621 = None
    unsqueeze_1603: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, 2);  unsqueeze_1602 = None
    unsqueeze_1604: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1603, 3);  unsqueeze_1603 = None
    mul_1622: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_20);  primals_20 = None
    unsqueeze_1605: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1622, 0);  mul_1622 = None
    unsqueeze_1606: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1605, 2);  unsqueeze_1605 = None
    unsqueeze_1607: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1606, 3);  unsqueeze_1606 = None
    sub_498: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1598);  convolution_6 = unsqueeze_1598 = None
    mul_1623: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_498, unsqueeze_1604);  sub_498 = unsqueeze_1604 = None
    sub_499: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_95, mul_1623);  mul_1623 = None
    sub_500: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_499, unsqueeze_1601);  sub_499 = unsqueeze_1601 = None
    mul_1624: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_500, unsqueeze_1607);  sub_500 = unsqueeze_1607 = None
    mul_1625: "f32[128]" = torch.ops.aten.mul.Tensor(sum_198, squeeze_19);  sum_198 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:88, code: out = self.conv3(out)
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(mul_1624, relu_4, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1624 = primals_19 = None
    getitem_525: "f32[8, 64, 56, 56]" = convolution_backward_99[0]
    getitem_526: "f32[128, 64, 1, 1]" = convolution_backward_99[1];  convolution_backward_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:86, code: out = self.relu(out)
    alias_390: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_391: "f32[8, 64, 56, 56]" = torch.ops.aten.alias.default(alias_390);  alias_390 = None
    le_96: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(alias_391, 0);  alias_391 = None
    scalar_tensor_96: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_96: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_96, scalar_tensor_96, getitem_525);  le_96 = scalar_tensor_96 = getitem_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:85, code: out = self.bn2(out)
    unsqueeze_1608: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_1609: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1608, 2);  unsqueeze_1608 = None
    unsqueeze_1610: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1609, 3);  unsqueeze_1609 = None
    sum_199: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_501: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1610)
    mul_1626: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_96, sub_501);  sub_501 = None
    sum_200: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1626, [0, 2, 3]);  mul_1626 = None
    mul_1627: "f32[64]" = torch.ops.aten.mul.Tensor(sum_199, 3.985969387755102e-05)
    unsqueeze_1611: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1627, 0);  mul_1627 = None
    unsqueeze_1612: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1611, 2);  unsqueeze_1611 = None
    unsqueeze_1613: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1612, 3);  unsqueeze_1612 = None
    mul_1628: "f32[64]" = torch.ops.aten.mul.Tensor(sum_200, 3.985969387755102e-05)
    mul_1629: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1630: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1628, mul_1629);  mul_1628 = mul_1629 = None
    unsqueeze_1614: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1630, 0);  mul_1630 = None
    unsqueeze_1615: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, 2);  unsqueeze_1614 = None
    unsqueeze_1616: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1615, 3);  unsqueeze_1615 = None
    mul_1631: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_17);  primals_17 = None
    unsqueeze_1617: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1631, 0);  mul_1631 = None
    unsqueeze_1618: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1617, 2);  unsqueeze_1617 = None
    unsqueeze_1619: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1618, 3);  unsqueeze_1618 = None
    sub_502: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1610);  convolution_5 = unsqueeze_1610 = None
    mul_1632: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_502, unsqueeze_1616);  sub_502 = unsqueeze_1616 = None
    sub_503: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_96, mul_1632);  where_96 = mul_1632 = None
    sub_504: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_503, unsqueeze_1613);  sub_503 = unsqueeze_1613 = None
    mul_1633: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_504, unsqueeze_1619);  sub_504 = unsqueeze_1619 = None
    mul_1634: "f32[64]" = torch.ops.aten.mul.Tensor(sum_200, squeeze_16);  sum_200 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:84, code: out = self.conv2(out)
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(mul_1633, relu_3, primals_16, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1633 = primals_16 = None
    getitem_528: "f32[8, 64, 112, 112]" = convolution_backward_100[0]
    getitem_529: "f32[64, 64, 3, 3]" = convolution_backward_100[1];  convolution_backward_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:82, code: out = self.relu(out)
    alias_393: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_394: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_393);  alias_393 = None
    le_97: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_394, 0);  alias_394 = None
    scalar_tensor_97: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_97: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_97, scalar_tensor_97, getitem_528);  le_97 = scalar_tensor_97 = getitem_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:81, code: out = self.bn1(out)
    unsqueeze_1620: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_1621: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1620, 2);  unsqueeze_1620 = None
    unsqueeze_1622: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1621, 3);  unsqueeze_1621 = None
    sum_201: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_97, [0, 2, 3])
    sub_505: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1622)
    mul_1635: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_97, sub_505);  sub_505 = None
    sum_202: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1635, [0, 2, 3]);  mul_1635 = None
    mul_1636: "f32[64]" = torch.ops.aten.mul.Tensor(sum_201, 9.964923469387754e-06)
    unsqueeze_1623: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1636, 0);  mul_1636 = None
    unsqueeze_1624: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1623, 2);  unsqueeze_1623 = None
    unsqueeze_1625: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1624, 3);  unsqueeze_1624 = None
    mul_1637: "f32[64]" = torch.ops.aten.mul.Tensor(sum_202, 9.964923469387754e-06)
    mul_1638: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1639: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1637, mul_1638);  mul_1637 = mul_1638 = None
    unsqueeze_1626: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1639, 0);  mul_1639 = None
    unsqueeze_1627: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, 2);  unsqueeze_1626 = None
    unsqueeze_1628: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1627, 3);  unsqueeze_1627 = None
    mul_1640: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
    unsqueeze_1629: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1640, 0);  mul_1640 = None
    unsqueeze_1630: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1629, 2);  unsqueeze_1629 = None
    unsqueeze_1631: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1630, 3);  unsqueeze_1630 = None
    sub_506: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1622);  convolution_4 = unsqueeze_1622 = None
    mul_1641: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_506, unsqueeze_1628);  sub_506 = unsqueeze_1628 = None
    sub_507: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_97, mul_1641);  where_97 = mul_1641 = None
    sub_508: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_507, unsqueeze_1625);  sub_507 = unsqueeze_1625 = None
    mul_1642: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_508, unsqueeze_1631);  sub_508 = unsqueeze_1631 = None
    mul_1643: "f32[64]" = torch.ops.aten.mul.Tensor(sum_202, squeeze_13);  sum_202 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:80, code: out = self.conv1(x)
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(mul_1642, relu_2, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1642 = primals_13 = None
    getitem_531: "f32[8, 32, 112, 112]" = convolution_backward_101[0]
    getitem_532: "f32[64, 32, 1, 1]" = convolution_backward_101[1];  convolution_backward_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:248, code: shortcut = self.project(bottom)
    unsqueeze_1632: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_1633: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1632, 2);  unsqueeze_1632 = None
    unsqueeze_1634: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1633, 3);  unsqueeze_1633 = None
    sum_203: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
    sub_509: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1634)
    mul_1644: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_95, sub_509);  sub_509 = None
    sum_204: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1644, [0, 2, 3]);  mul_1644 = None
    mul_1645: "f32[128]" = torch.ops.aten.mul.Tensor(sum_203, 3.985969387755102e-05)
    unsqueeze_1635: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1645, 0);  mul_1645 = None
    unsqueeze_1636: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1635, 2);  unsqueeze_1635 = None
    unsqueeze_1637: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1636, 3);  unsqueeze_1636 = None
    mul_1646: "f32[128]" = torch.ops.aten.mul.Tensor(sum_204, 3.985969387755102e-05)
    mul_1647: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1648: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1646, mul_1647);  mul_1646 = mul_1647 = None
    unsqueeze_1638: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1648, 0);  mul_1648 = None
    unsqueeze_1639: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, 2);  unsqueeze_1638 = None
    unsqueeze_1640: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1639, 3);  unsqueeze_1639 = None
    mul_1649: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
    unsqueeze_1641: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1649, 0);  mul_1649 = None
    unsqueeze_1642: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1641, 2);  unsqueeze_1641 = None
    unsqueeze_1643: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1642, 3);  unsqueeze_1642 = None
    sub_510: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1634);  convolution_3 = unsqueeze_1634 = None
    mul_1650: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_510, unsqueeze_1640);  sub_510 = unsqueeze_1640 = None
    sub_511: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_95, mul_1650);  where_95 = mul_1650 = None
    sub_512: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_511, unsqueeze_1637);  sub_511 = unsqueeze_1637 = None
    mul_1651: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_512, unsqueeze_1643);  sub_512 = unsqueeze_1643 = None
    mul_1652: "f32[128]" = torch.ops.aten.mul.Tensor(sum_204, squeeze_10);  sum_204 = squeeze_10 = None
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(mul_1651, getitem_6, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1651 = getitem_6 = primals_10 = None
    getitem_534: "f32[8, 32, 56, 56]" = convolution_backward_102[0]
    getitem_535: "f32[128, 32, 1, 1]" = convolution_backward_102[1];  convolution_backward_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    max_pool2d_with_indices_backward_5: "f32[8, 32, 112, 112]" = torch.ops.aten.max_pool2d_with_indices_backward.default(getitem_534, relu_2, [2, 2], [2, 2], [0, 0], [1, 1], False, getitem_7);  getitem_534 = getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:247, code: bottom = self.downsample(x)
    add_635: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(getitem_531, max_pool2d_with_indices_backward_5);  getitem_531 = max_pool2d_with_indices_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:365, code: x = self.level1(x)
    alias_396: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_397: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(alias_396);  alias_396 = None
    le_98: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_397, 0);  alias_397 = None
    scalar_tensor_98: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_98: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_98, scalar_tensor_98, add_635);  le_98 = scalar_tensor_98 = add_635 = None
    unsqueeze_1644: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_1645: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1644, 2);  unsqueeze_1644 = None
    unsqueeze_1646: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1645, 3);  unsqueeze_1645 = None
    sum_205: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_98, [0, 2, 3])
    sub_513: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1646)
    mul_1653: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_98, sub_513);  sub_513 = None
    sum_206: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1653, [0, 2, 3]);  mul_1653 = None
    mul_1654: "f32[32]" = torch.ops.aten.mul.Tensor(sum_205, 9.964923469387754e-06)
    unsqueeze_1647: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1654, 0);  mul_1654 = None
    unsqueeze_1648: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1647, 2);  unsqueeze_1647 = None
    unsqueeze_1649: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1648, 3);  unsqueeze_1648 = None
    mul_1655: "f32[32]" = torch.ops.aten.mul.Tensor(sum_206, 9.964923469387754e-06)
    mul_1656: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1657: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1655, mul_1656);  mul_1655 = mul_1656 = None
    unsqueeze_1650: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1657, 0);  mul_1657 = None
    unsqueeze_1651: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, 2);  unsqueeze_1650 = None
    unsqueeze_1652: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1651, 3);  unsqueeze_1651 = None
    mul_1658: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
    unsqueeze_1653: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1658, 0);  mul_1658 = None
    unsqueeze_1654: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1653, 2);  unsqueeze_1653 = None
    unsqueeze_1655: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1654, 3);  unsqueeze_1654 = None
    sub_514: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1646);  convolution_2 = unsqueeze_1646 = None
    mul_1659: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_514, unsqueeze_1652);  sub_514 = unsqueeze_1652 = None
    sub_515: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_98, mul_1659);  where_98 = mul_1659 = None
    sub_516: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_515, unsqueeze_1649);  sub_515 = unsqueeze_1649 = None
    mul_1660: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_516, unsqueeze_1655);  sub_516 = unsqueeze_1655 = None
    mul_1661: "f32[32]" = torch.ops.aten.mul.Tensor(sum_206, squeeze_7);  sum_206 = squeeze_7 = None
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(mul_1660, relu_1, primals_7, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1660 = primals_7 = None
    getitem_537: "f32[8, 16, 224, 224]" = convolution_backward_103[0]
    getitem_538: "f32[32, 16, 3, 3]" = convolution_backward_103[1];  convolution_backward_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:364, code: x = self.level0(x)
    alias_399: "f32[8, 16, 224, 224]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_400: "f32[8, 16, 224, 224]" = torch.ops.aten.alias.default(alias_399);  alias_399 = None
    le_99: "b8[8, 16, 224, 224]" = torch.ops.aten.le.Scalar(alias_400, 0);  alias_400 = None
    scalar_tensor_99: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_99: "f32[8, 16, 224, 224]" = torch.ops.aten.where.self(le_99, scalar_tensor_99, getitem_537);  le_99 = scalar_tensor_99 = getitem_537 = None
    unsqueeze_1656: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_1657: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1656, 2);  unsqueeze_1656 = None
    unsqueeze_1658: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1657, 3);  unsqueeze_1657 = None
    sum_207: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_99, [0, 2, 3])
    sub_517: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1658)
    mul_1662: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(where_99, sub_517);  sub_517 = None
    sum_208: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1662, [0, 2, 3]);  mul_1662 = None
    mul_1663: "f32[16]" = torch.ops.aten.mul.Tensor(sum_207, 2.4912308673469386e-06)
    unsqueeze_1659: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1663, 0);  mul_1663 = None
    unsqueeze_1660: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1659, 2);  unsqueeze_1659 = None
    unsqueeze_1661: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1660, 3);  unsqueeze_1660 = None
    mul_1664: "f32[16]" = torch.ops.aten.mul.Tensor(sum_208, 2.4912308673469386e-06)
    mul_1665: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1666: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1664, mul_1665);  mul_1664 = mul_1665 = None
    unsqueeze_1662: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1666, 0);  mul_1666 = None
    unsqueeze_1663: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, 2);  unsqueeze_1662 = None
    unsqueeze_1664: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1663, 3);  unsqueeze_1663 = None
    mul_1667: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
    unsqueeze_1665: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1667, 0);  mul_1667 = None
    unsqueeze_1666: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1665, 2);  unsqueeze_1665 = None
    unsqueeze_1667: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1666, 3);  unsqueeze_1666 = None
    sub_518: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1658);  convolution_1 = unsqueeze_1658 = None
    mul_1668: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(sub_518, unsqueeze_1664);  sub_518 = unsqueeze_1664 = None
    sub_519: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(where_99, mul_1668);  where_99 = mul_1668 = None
    sub_520: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(sub_519, unsqueeze_1661);  sub_519 = unsqueeze_1661 = None
    mul_1669: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(sub_520, unsqueeze_1667);  sub_520 = unsqueeze_1667 = None
    mul_1670: "f32[16]" = torch.ops.aten.mul.Tensor(sum_208, squeeze_4);  sum_208 = squeeze_4 = None
    convolution_backward_104 = torch.ops.aten.convolution_backward.default(mul_1669, relu, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1669 = primals_4 = None
    getitem_540: "f32[8, 16, 224, 224]" = convolution_backward_104[0]
    getitem_541: "f32[16, 16, 3, 3]" = convolution_backward_104[1];  convolution_backward_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/dla.py:363, code: x = self.base_layer(x)
    alias_402: "f32[8, 16, 224, 224]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_403: "f32[8, 16, 224, 224]" = torch.ops.aten.alias.default(alias_402);  alias_402 = None
    le_100: "b8[8, 16, 224, 224]" = torch.ops.aten.le.Scalar(alias_403, 0);  alias_403 = None
    scalar_tensor_100: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_100: "f32[8, 16, 224, 224]" = torch.ops.aten.where.self(le_100, scalar_tensor_100, getitem_540);  le_100 = scalar_tensor_100 = getitem_540 = None
    unsqueeze_1668: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_1669: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1668, 2);  unsqueeze_1668 = None
    unsqueeze_1670: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1669, 3);  unsqueeze_1669 = None
    sum_209: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_100, [0, 2, 3])
    sub_521: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1670)
    mul_1671: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(where_100, sub_521);  sub_521 = None
    sum_210: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1671, [0, 2, 3]);  mul_1671 = None
    mul_1672: "f32[16]" = torch.ops.aten.mul.Tensor(sum_209, 2.4912308673469386e-06)
    unsqueeze_1671: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1672, 0);  mul_1672 = None
    unsqueeze_1672: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1671, 2);  unsqueeze_1671 = None
    unsqueeze_1673: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1672, 3);  unsqueeze_1672 = None
    mul_1673: "f32[16]" = torch.ops.aten.mul.Tensor(sum_210, 2.4912308673469386e-06)
    mul_1674: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1675: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1673, mul_1674);  mul_1673 = mul_1674 = None
    unsqueeze_1674: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1675, 0);  mul_1675 = None
    unsqueeze_1675: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1674, 2);  unsqueeze_1674 = None
    unsqueeze_1676: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1675, 3);  unsqueeze_1675 = None
    mul_1676: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
    unsqueeze_1677: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1676, 0);  mul_1676 = None
    unsqueeze_1678: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1677, 2);  unsqueeze_1677 = None
    unsqueeze_1679: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1678, 3);  unsqueeze_1678 = None
    sub_522: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1670);  convolution = unsqueeze_1670 = None
    mul_1677: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(sub_522, unsqueeze_1676);  sub_522 = unsqueeze_1676 = None
    sub_523: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(where_100, mul_1677);  where_100 = mul_1677 = None
    sub_524: "f32[8, 16, 224, 224]" = torch.ops.aten.sub.Tensor(sub_523, unsqueeze_1673);  sub_523 = unsqueeze_1673 = None
    mul_1678: "f32[8, 16, 224, 224]" = torch.ops.aten.mul.Tensor(sub_524, unsqueeze_1679);  sub_524 = unsqueeze_1679 = None
    mul_1679: "f32[16]" = torch.ops.aten.mul.Tensor(sum_210, squeeze_1);  sum_210 = squeeze_1 = None
    convolution_backward_105 = torch.ops.aten.convolution_backward.default(mul_1678, primals_633, primals_1, [0], [1, 1], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1678 = primals_633 = primals_1 = None
    getitem_544: "f32[16, 3, 7, 7]" = convolution_backward_105[1];  convolution_backward_105 = None
    
    # No stacktrace found for following nodes
    copy_: "f32[16]" = torch.ops.aten.copy_.default(primals_318, add_2);  primals_318 = add_2 = None
    copy__1: "f32[16]" = torch.ops.aten.copy_.default(primals_319, add_3);  primals_319 = add_3 = None
    copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_320, add);  primals_320 = add = None
    copy__3: "f32[16]" = torch.ops.aten.copy_.default(primals_321, add_7);  primals_321 = add_7 = None
    copy__4: "f32[16]" = torch.ops.aten.copy_.default(primals_322, add_8);  primals_322 = add_8 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_323, add_5);  primals_323 = add_5 = None
    copy__6: "f32[32]" = torch.ops.aten.copy_.default(primals_324, add_12);  primals_324 = add_12 = None
    copy__7: "f32[32]" = torch.ops.aten.copy_.default(primals_325, add_13);  primals_325 = add_13 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_326, add_10);  primals_326 = add_10 = None
    copy__9: "f32[128]" = torch.ops.aten.copy_.default(primals_327, add_17);  primals_327 = add_17 = None
    copy__10: "f32[128]" = torch.ops.aten.copy_.default(primals_328, add_18);  primals_328 = add_18 = None
    copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_329, add_15);  primals_329 = add_15 = None
    copy__12: "f32[64]" = torch.ops.aten.copy_.default(primals_330, add_22);  primals_330 = add_22 = None
    copy__13: "f32[64]" = torch.ops.aten.copy_.default(primals_331, add_23);  primals_331 = add_23 = None
    copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_332, add_20);  primals_332 = add_20 = None
    copy__15: "f32[64]" = torch.ops.aten.copy_.default(primals_333, add_27);  primals_333 = add_27 = None
    copy__16: "f32[64]" = torch.ops.aten.copy_.default(primals_334, add_28);  primals_334 = add_28 = None
    copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_335, add_25);  primals_335 = add_25 = None
    copy__18: "f32[128]" = torch.ops.aten.copy_.default(primals_336, add_32);  primals_336 = add_32 = None
    copy__19: "f32[128]" = torch.ops.aten.copy_.default(primals_337, add_33);  primals_337 = add_33 = None
    copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_338, add_30);  primals_338 = add_30 = None
    copy__21: "f32[64]" = torch.ops.aten.copy_.default(primals_339, add_38);  primals_339 = add_38 = None
    copy__22: "f32[64]" = torch.ops.aten.copy_.default(primals_340, add_39);  primals_340 = add_39 = None
    copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_341, add_36);  primals_341 = add_36 = None
    copy__24: "f32[64]" = torch.ops.aten.copy_.default(primals_342, add_43);  primals_342 = add_43 = None
    copy__25: "f32[64]" = torch.ops.aten.copy_.default(primals_343, add_44);  primals_343 = add_44 = None
    copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_344, add_41);  primals_344 = add_41 = None
    copy__27: "f32[128]" = torch.ops.aten.copy_.default(primals_345, add_48);  primals_345 = add_48 = None
    copy__28: "f32[128]" = torch.ops.aten.copy_.default(primals_346, add_49);  primals_346 = add_49 = None
    copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_347, add_46);  primals_347 = add_46 = None
    copy__30: "f32[128]" = torch.ops.aten.copy_.default(primals_348, add_54);  primals_348 = add_54 = None
    copy__31: "f32[128]" = torch.ops.aten.copy_.default(primals_349, add_55);  primals_349 = add_55 = None
    copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_350, add_52);  primals_350 = add_52 = None
    copy__33: "f32[256]" = torch.ops.aten.copy_.default(primals_351, add_60);  primals_351 = add_60 = None
    copy__34: "f32[256]" = torch.ops.aten.copy_.default(primals_352, add_61);  primals_352 = add_61 = None
    copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_353, add_58);  primals_353 = add_58 = None
    copy__36: "f32[128]" = torch.ops.aten.copy_.default(primals_354, add_65);  primals_354 = add_65 = None
    copy__37: "f32[128]" = torch.ops.aten.copy_.default(primals_355, add_66);  primals_355 = add_66 = None
    copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_356, add_63);  primals_356 = add_63 = None
    copy__39: "f32[128]" = torch.ops.aten.copy_.default(primals_357, add_70);  primals_357 = add_70 = None
    copy__40: "f32[128]" = torch.ops.aten.copy_.default(primals_358, add_71);  primals_358 = add_71 = None
    copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_359, add_68);  primals_359 = add_68 = None
    copy__42: "f32[256]" = torch.ops.aten.copy_.default(primals_360, add_75);  primals_360 = add_75 = None
    copy__43: "f32[256]" = torch.ops.aten.copy_.default(primals_361, add_76);  primals_361 = add_76 = None
    copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_362, add_73);  primals_362 = add_73 = None
    copy__45: "f32[128]" = torch.ops.aten.copy_.default(primals_363, add_81);  primals_363 = add_81 = None
    copy__46: "f32[128]" = torch.ops.aten.copy_.default(primals_364, add_82);  primals_364 = add_82 = None
    copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_365, add_79);  primals_365 = add_79 = None
    copy__48: "f32[128]" = torch.ops.aten.copy_.default(primals_366, add_86);  primals_366 = add_86 = None
    copy__49: "f32[128]" = torch.ops.aten.copy_.default(primals_367, add_87);  primals_367 = add_87 = None
    copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_368, add_84);  primals_368 = add_84 = None
    copy__51: "f32[256]" = torch.ops.aten.copy_.default(primals_369, add_91);  primals_369 = add_91 = None
    copy__52: "f32[256]" = torch.ops.aten.copy_.default(primals_370, add_92);  primals_370 = add_92 = None
    copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_371, add_89);  primals_371 = add_89 = None
    copy__54: "f32[256]" = torch.ops.aten.copy_.default(primals_372, add_97);  primals_372 = add_97 = None
    copy__55: "f32[256]" = torch.ops.aten.copy_.default(primals_373, add_98);  primals_373 = add_98 = None
    copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_374, add_95);  primals_374 = add_95 = None
    copy__57: "f32[128]" = torch.ops.aten.copy_.default(primals_375, add_103);  primals_375 = add_103 = None
    copy__58: "f32[128]" = torch.ops.aten.copy_.default(primals_376, add_104);  primals_376 = add_104 = None
    copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_377, add_101);  primals_377 = add_101 = None
    copy__60: "f32[128]" = torch.ops.aten.copy_.default(primals_378, add_108);  primals_378 = add_108 = None
    copy__61: "f32[128]" = torch.ops.aten.copy_.default(primals_379, add_109);  primals_379 = add_109 = None
    copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_380, add_106);  primals_380 = add_106 = None
    copy__63: "f32[256]" = torch.ops.aten.copy_.default(primals_381, add_113);  primals_381 = add_113 = None
    copy__64: "f32[256]" = torch.ops.aten.copy_.default(primals_382, add_114);  primals_382 = add_114 = None
    copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_383, add_111);  primals_383 = add_111 = None
    copy__66: "f32[128]" = torch.ops.aten.copy_.default(primals_384, add_119);  primals_384 = add_119 = None
    copy__67: "f32[128]" = torch.ops.aten.copy_.default(primals_385, add_120);  primals_385 = add_120 = None
    copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_386, add_117);  primals_386 = add_117 = None
    copy__69: "f32[128]" = torch.ops.aten.copy_.default(primals_387, add_124);  primals_387 = add_124 = None
    copy__70: "f32[128]" = torch.ops.aten.copy_.default(primals_388, add_125);  primals_388 = add_125 = None
    copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_389, add_122);  primals_389 = add_122 = None
    copy__72: "f32[256]" = torch.ops.aten.copy_.default(primals_390, add_129);  primals_390 = add_129 = None
    copy__73: "f32[256]" = torch.ops.aten.copy_.default(primals_391, add_130);  primals_391 = add_130 = None
    copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_392, add_127);  primals_392 = add_127 = None
    copy__75: "f32[256]" = torch.ops.aten.copy_.default(primals_393, add_135);  primals_393 = add_135 = None
    copy__76: "f32[256]" = torch.ops.aten.copy_.default(primals_394, add_136);  primals_394 = add_136 = None
    copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_395, add_133);  primals_395 = add_133 = None
    copy__78: "f32[128]" = torch.ops.aten.copy_.default(primals_396, add_141);  primals_396 = add_141 = None
    copy__79: "f32[128]" = torch.ops.aten.copy_.default(primals_397, add_142);  primals_397 = add_142 = None
    copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_398, add_139);  primals_398 = add_139 = None
    copy__81: "f32[128]" = torch.ops.aten.copy_.default(primals_399, add_146);  primals_399 = add_146 = None
    copy__82: "f32[128]" = torch.ops.aten.copy_.default(primals_400, add_147);  primals_400 = add_147 = None
    copy__83: "i64[]" = torch.ops.aten.copy_.default(primals_401, add_144);  primals_401 = add_144 = None
    copy__84: "f32[256]" = torch.ops.aten.copy_.default(primals_402, add_151);  primals_402 = add_151 = None
    copy__85: "f32[256]" = torch.ops.aten.copy_.default(primals_403, add_152);  primals_403 = add_152 = None
    copy__86: "i64[]" = torch.ops.aten.copy_.default(primals_404, add_149);  primals_404 = add_149 = None
    copy__87: "f32[128]" = torch.ops.aten.copy_.default(primals_405, add_157);  primals_405 = add_157 = None
    copy__88: "f32[128]" = torch.ops.aten.copy_.default(primals_406, add_158);  primals_406 = add_158 = None
    copy__89: "i64[]" = torch.ops.aten.copy_.default(primals_407, add_155);  primals_407 = add_155 = None
    copy__90: "f32[128]" = torch.ops.aten.copy_.default(primals_408, add_162);  primals_408 = add_162 = None
    copy__91: "f32[128]" = torch.ops.aten.copy_.default(primals_409, add_163);  primals_409 = add_163 = None
    copy__92: "i64[]" = torch.ops.aten.copy_.default(primals_410, add_160);  primals_410 = add_160 = None
    copy__93: "f32[256]" = torch.ops.aten.copy_.default(primals_411, add_167);  primals_411 = add_167 = None
    copy__94: "f32[256]" = torch.ops.aten.copy_.default(primals_412, add_168);  primals_412 = add_168 = None
    copy__95: "i64[]" = torch.ops.aten.copy_.default(primals_413, add_165);  primals_413 = add_165 = None
    copy__96: "f32[256]" = torch.ops.aten.copy_.default(primals_414, add_173);  primals_414 = add_173 = None
    copy__97: "f32[256]" = torch.ops.aten.copy_.default(primals_415, add_174);  primals_415 = add_174 = None
    copy__98: "i64[]" = torch.ops.aten.copy_.default(primals_416, add_171);  primals_416 = add_171 = None
    copy__99: "f32[128]" = torch.ops.aten.copy_.default(primals_417, add_179);  primals_417 = add_179 = None
    copy__100: "f32[128]" = torch.ops.aten.copy_.default(primals_418, add_180);  primals_418 = add_180 = None
    copy__101: "i64[]" = torch.ops.aten.copy_.default(primals_419, add_177);  primals_419 = add_177 = None
    copy__102: "f32[128]" = torch.ops.aten.copy_.default(primals_420, add_184);  primals_420 = add_184 = None
    copy__103: "f32[128]" = torch.ops.aten.copy_.default(primals_421, add_185);  primals_421 = add_185 = None
    copy__104: "i64[]" = torch.ops.aten.copy_.default(primals_422, add_182);  primals_422 = add_182 = None
    copy__105: "f32[256]" = torch.ops.aten.copy_.default(primals_423, add_189);  primals_423 = add_189 = None
    copy__106: "f32[256]" = torch.ops.aten.copy_.default(primals_424, add_190);  primals_424 = add_190 = None
    copy__107: "i64[]" = torch.ops.aten.copy_.default(primals_425, add_187);  primals_425 = add_187 = None
    copy__108: "f32[128]" = torch.ops.aten.copy_.default(primals_426, add_195);  primals_426 = add_195 = None
    copy__109: "f32[128]" = torch.ops.aten.copy_.default(primals_427, add_196);  primals_427 = add_196 = None
    copy__110: "i64[]" = torch.ops.aten.copy_.default(primals_428, add_193);  primals_428 = add_193 = None
    copy__111: "f32[128]" = torch.ops.aten.copy_.default(primals_429, add_200);  primals_429 = add_200 = None
    copy__112: "f32[128]" = torch.ops.aten.copy_.default(primals_430, add_201);  primals_430 = add_201 = None
    copy__113: "i64[]" = torch.ops.aten.copy_.default(primals_431, add_198);  primals_431 = add_198 = None
    copy__114: "f32[256]" = torch.ops.aten.copy_.default(primals_432, add_205);  primals_432 = add_205 = None
    copy__115: "f32[256]" = torch.ops.aten.copy_.default(primals_433, add_206);  primals_433 = add_206 = None
    copy__116: "i64[]" = torch.ops.aten.copy_.default(primals_434, add_203);  primals_434 = add_203 = None
    copy__117: "f32[256]" = torch.ops.aten.copy_.default(primals_435, add_211);  primals_435 = add_211 = None
    copy__118: "f32[256]" = torch.ops.aten.copy_.default(primals_436, add_212);  primals_436 = add_212 = None
    copy__119: "i64[]" = torch.ops.aten.copy_.default(primals_437, add_209);  primals_437 = add_209 = None
    copy__120: "f32[512]" = torch.ops.aten.copy_.default(primals_438, add_217);  primals_438 = add_217 = None
    copy__121: "f32[512]" = torch.ops.aten.copy_.default(primals_439, add_218);  primals_439 = add_218 = None
    copy__122: "i64[]" = torch.ops.aten.copy_.default(primals_440, add_215);  primals_440 = add_215 = None
    copy__123: "f32[256]" = torch.ops.aten.copy_.default(primals_441, add_222);  primals_441 = add_222 = None
    copy__124: "f32[256]" = torch.ops.aten.copy_.default(primals_442, add_223);  primals_442 = add_223 = None
    copy__125: "i64[]" = torch.ops.aten.copy_.default(primals_443, add_220);  primals_443 = add_220 = None
    copy__126: "f32[256]" = torch.ops.aten.copy_.default(primals_444, add_227);  primals_444 = add_227 = None
    copy__127: "f32[256]" = torch.ops.aten.copy_.default(primals_445, add_228);  primals_445 = add_228 = None
    copy__128: "i64[]" = torch.ops.aten.copy_.default(primals_446, add_225);  primals_446 = add_225 = None
    copy__129: "f32[512]" = torch.ops.aten.copy_.default(primals_447, add_232);  primals_447 = add_232 = None
    copy__130: "f32[512]" = torch.ops.aten.copy_.default(primals_448, add_233);  primals_448 = add_233 = None
    copy__131: "i64[]" = torch.ops.aten.copy_.default(primals_449, add_230);  primals_449 = add_230 = None
    copy__132: "f32[256]" = torch.ops.aten.copy_.default(primals_450, add_238);  primals_450 = add_238 = None
    copy__133: "f32[256]" = torch.ops.aten.copy_.default(primals_451, add_239);  primals_451 = add_239 = None
    copy__134: "i64[]" = torch.ops.aten.copy_.default(primals_452, add_236);  primals_452 = add_236 = None
    copy__135: "f32[256]" = torch.ops.aten.copy_.default(primals_453, add_243);  primals_453 = add_243 = None
    copy__136: "f32[256]" = torch.ops.aten.copy_.default(primals_454, add_244);  primals_454 = add_244 = None
    copy__137: "i64[]" = torch.ops.aten.copy_.default(primals_455, add_241);  primals_455 = add_241 = None
    copy__138: "f32[512]" = torch.ops.aten.copy_.default(primals_456, add_248);  primals_456 = add_248 = None
    copy__139: "f32[512]" = torch.ops.aten.copy_.default(primals_457, add_249);  primals_457 = add_249 = None
    copy__140: "i64[]" = torch.ops.aten.copy_.default(primals_458, add_246);  primals_458 = add_246 = None
    copy__141: "f32[512]" = torch.ops.aten.copy_.default(primals_459, add_254);  primals_459 = add_254 = None
    copy__142: "f32[512]" = torch.ops.aten.copy_.default(primals_460, add_255);  primals_460 = add_255 = None
    copy__143: "i64[]" = torch.ops.aten.copy_.default(primals_461, add_252);  primals_461 = add_252 = None
    copy__144: "f32[256]" = torch.ops.aten.copy_.default(primals_462, add_260);  primals_462 = add_260 = None
    copy__145: "f32[256]" = torch.ops.aten.copy_.default(primals_463, add_261);  primals_463 = add_261 = None
    copy__146: "i64[]" = torch.ops.aten.copy_.default(primals_464, add_258);  primals_464 = add_258 = None
    copy__147: "f32[256]" = torch.ops.aten.copy_.default(primals_465, add_265);  primals_465 = add_265 = None
    copy__148: "f32[256]" = torch.ops.aten.copy_.default(primals_466, add_266);  primals_466 = add_266 = None
    copy__149: "i64[]" = torch.ops.aten.copy_.default(primals_467, add_263);  primals_467 = add_263 = None
    copy__150: "f32[512]" = torch.ops.aten.copy_.default(primals_468, add_270);  primals_468 = add_270 = None
    copy__151: "f32[512]" = torch.ops.aten.copy_.default(primals_469, add_271);  primals_469 = add_271 = None
    copy__152: "i64[]" = torch.ops.aten.copy_.default(primals_470, add_268);  primals_470 = add_268 = None
    copy__153: "f32[256]" = torch.ops.aten.copy_.default(primals_471, add_276);  primals_471 = add_276 = None
    copy__154: "f32[256]" = torch.ops.aten.copy_.default(primals_472, add_277);  primals_472 = add_277 = None
    copy__155: "i64[]" = torch.ops.aten.copy_.default(primals_473, add_274);  primals_473 = add_274 = None
    copy__156: "f32[256]" = torch.ops.aten.copy_.default(primals_474, add_281);  primals_474 = add_281 = None
    copy__157: "f32[256]" = torch.ops.aten.copy_.default(primals_475, add_282);  primals_475 = add_282 = None
    copy__158: "i64[]" = torch.ops.aten.copy_.default(primals_476, add_279);  primals_476 = add_279 = None
    copy__159: "f32[512]" = torch.ops.aten.copy_.default(primals_477, add_286);  primals_477 = add_286 = None
    copy__160: "f32[512]" = torch.ops.aten.copy_.default(primals_478, add_287);  primals_478 = add_287 = None
    copy__161: "i64[]" = torch.ops.aten.copy_.default(primals_479, add_284);  primals_479 = add_284 = None
    copy__162: "f32[512]" = torch.ops.aten.copy_.default(primals_480, add_292);  primals_480 = add_292 = None
    copy__163: "f32[512]" = torch.ops.aten.copy_.default(primals_481, add_293);  primals_481 = add_293 = None
    copy__164: "i64[]" = torch.ops.aten.copy_.default(primals_482, add_290);  primals_482 = add_290 = None
    copy__165: "f32[256]" = torch.ops.aten.copy_.default(primals_483, add_298);  primals_483 = add_298 = None
    copy__166: "f32[256]" = torch.ops.aten.copy_.default(primals_484, add_299);  primals_484 = add_299 = None
    copy__167: "i64[]" = torch.ops.aten.copy_.default(primals_485, add_296);  primals_485 = add_296 = None
    copy__168: "f32[256]" = torch.ops.aten.copy_.default(primals_486, add_303);  primals_486 = add_303 = None
    copy__169: "f32[256]" = torch.ops.aten.copy_.default(primals_487, add_304);  primals_487 = add_304 = None
    copy__170: "i64[]" = torch.ops.aten.copy_.default(primals_488, add_301);  primals_488 = add_301 = None
    copy__171: "f32[512]" = torch.ops.aten.copy_.default(primals_489, add_308);  primals_489 = add_308 = None
    copy__172: "f32[512]" = torch.ops.aten.copy_.default(primals_490, add_309);  primals_490 = add_309 = None
    copy__173: "i64[]" = torch.ops.aten.copy_.default(primals_491, add_306);  primals_491 = add_306 = None
    copy__174: "f32[256]" = torch.ops.aten.copy_.default(primals_492, add_314);  primals_492 = add_314 = None
    copy__175: "f32[256]" = torch.ops.aten.copy_.default(primals_493, add_315);  primals_493 = add_315 = None
    copy__176: "i64[]" = torch.ops.aten.copy_.default(primals_494, add_312);  primals_494 = add_312 = None
    copy__177: "f32[256]" = torch.ops.aten.copy_.default(primals_495, add_319);  primals_495 = add_319 = None
    copy__178: "f32[256]" = torch.ops.aten.copy_.default(primals_496, add_320);  primals_496 = add_320 = None
    copy__179: "i64[]" = torch.ops.aten.copy_.default(primals_497, add_317);  primals_497 = add_317 = None
    copy__180: "f32[512]" = torch.ops.aten.copy_.default(primals_498, add_324);  primals_498 = add_324 = None
    copy__181: "f32[512]" = torch.ops.aten.copy_.default(primals_499, add_325);  primals_499 = add_325 = None
    copy__182: "i64[]" = torch.ops.aten.copy_.default(primals_500, add_322);  primals_500 = add_322 = None
    copy__183: "f32[512]" = torch.ops.aten.copy_.default(primals_501, add_330);  primals_501 = add_330 = None
    copy__184: "f32[512]" = torch.ops.aten.copy_.default(primals_502, add_331);  primals_502 = add_331 = None
    copy__185: "i64[]" = torch.ops.aten.copy_.default(primals_503, add_328);  primals_503 = add_328 = None
    copy__186: "f32[256]" = torch.ops.aten.copy_.default(primals_504, add_336);  primals_504 = add_336 = None
    copy__187: "f32[256]" = torch.ops.aten.copy_.default(primals_505, add_337);  primals_505 = add_337 = None
    copy__188: "i64[]" = torch.ops.aten.copy_.default(primals_506, add_334);  primals_506 = add_334 = None
    copy__189: "f32[256]" = torch.ops.aten.copy_.default(primals_507, add_341);  primals_507 = add_341 = None
    copy__190: "f32[256]" = torch.ops.aten.copy_.default(primals_508, add_342);  primals_508 = add_342 = None
    copy__191: "i64[]" = torch.ops.aten.copy_.default(primals_509, add_339);  primals_509 = add_339 = None
    copy__192: "f32[512]" = torch.ops.aten.copy_.default(primals_510, add_346);  primals_510 = add_346 = None
    copy__193: "f32[512]" = torch.ops.aten.copy_.default(primals_511, add_347);  primals_511 = add_347 = None
    copy__194: "i64[]" = torch.ops.aten.copy_.default(primals_512, add_344);  primals_512 = add_344 = None
    copy__195: "f32[256]" = torch.ops.aten.copy_.default(primals_513, add_352);  primals_513 = add_352 = None
    copy__196: "f32[256]" = torch.ops.aten.copy_.default(primals_514, add_353);  primals_514 = add_353 = None
    copy__197: "i64[]" = torch.ops.aten.copy_.default(primals_515, add_350);  primals_515 = add_350 = None
    copy__198: "f32[256]" = torch.ops.aten.copy_.default(primals_516, add_357);  primals_516 = add_357 = None
    copy__199: "f32[256]" = torch.ops.aten.copy_.default(primals_517, add_358);  primals_517 = add_358 = None
    copy__200: "i64[]" = torch.ops.aten.copy_.default(primals_518, add_355);  primals_518 = add_355 = None
    copy__201: "f32[512]" = torch.ops.aten.copy_.default(primals_519, add_362);  primals_519 = add_362 = None
    copy__202: "f32[512]" = torch.ops.aten.copy_.default(primals_520, add_363);  primals_520 = add_363 = None
    copy__203: "i64[]" = torch.ops.aten.copy_.default(primals_521, add_360);  primals_521 = add_360 = None
    copy__204: "f32[512]" = torch.ops.aten.copy_.default(primals_522, add_368);  primals_522 = add_368 = None
    copy__205: "f32[512]" = torch.ops.aten.copy_.default(primals_523, add_369);  primals_523 = add_369 = None
    copy__206: "i64[]" = torch.ops.aten.copy_.default(primals_524, add_366);  primals_524 = add_366 = None
    copy__207: "f32[256]" = torch.ops.aten.copy_.default(primals_525, add_374);  primals_525 = add_374 = None
    copy__208: "f32[256]" = torch.ops.aten.copy_.default(primals_526, add_375);  primals_526 = add_375 = None
    copy__209: "i64[]" = torch.ops.aten.copy_.default(primals_527, add_372);  primals_527 = add_372 = None
    copy__210: "f32[256]" = torch.ops.aten.copy_.default(primals_528, add_379);  primals_528 = add_379 = None
    copy__211: "f32[256]" = torch.ops.aten.copy_.default(primals_529, add_380);  primals_529 = add_380 = None
    copy__212: "i64[]" = torch.ops.aten.copy_.default(primals_530, add_377);  primals_530 = add_377 = None
    copy__213: "f32[512]" = torch.ops.aten.copy_.default(primals_531, add_384);  primals_531 = add_384 = None
    copy__214: "f32[512]" = torch.ops.aten.copy_.default(primals_532, add_385);  primals_532 = add_385 = None
    copy__215: "i64[]" = torch.ops.aten.copy_.default(primals_533, add_382);  primals_533 = add_382 = None
    copy__216: "f32[256]" = torch.ops.aten.copy_.default(primals_534, add_390);  primals_534 = add_390 = None
    copy__217: "f32[256]" = torch.ops.aten.copy_.default(primals_535, add_391);  primals_535 = add_391 = None
    copy__218: "i64[]" = torch.ops.aten.copy_.default(primals_536, add_388);  primals_536 = add_388 = None
    copy__219: "f32[256]" = torch.ops.aten.copy_.default(primals_537, add_395);  primals_537 = add_395 = None
    copy__220: "f32[256]" = torch.ops.aten.copy_.default(primals_538, add_396);  primals_538 = add_396 = None
    copy__221: "i64[]" = torch.ops.aten.copy_.default(primals_539, add_393);  primals_539 = add_393 = None
    copy__222: "f32[512]" = torch.ops.aten.copy_.default(primals_540, add_400);  primals_540 = add_400 = None
    copy__223: "f32[512]" = torch.ops.aten.copy_.default(primals_541, add_401);  primals_541 = add_401 = None
    copy__224: "i64[]" = torch.ops.aten.copy_.default(primals_542, add_398);  primals_542 = add_398 = None
    copy__225: "f32[512]" = torch.ops.aten.copy_.default(primals_543, add_406);  primals_543 = add_406 = None
    copy__226: "f32[512]" = torch.ops.aten.copy_.default(primals_544, add_407);  primals_544 = add_407 = None
    copy__227: "i64[]" = torch.ops.aten.copy_.default(primals_545, add_404);  primals_545 = add_404 = None
    copy__228: "f32[256]" = torch.ops.aten.copy_.default(primals_546, add_412);  primals_546 = add_412 = None
    copy__229: "f32[256]" = torch.ops.aten.copy_.default(primals_547, add_413);  primals_547 = add_413 = None
    copy__230: "i64[]" = torch.ops.aten.copy_.default(primals_548, add_410);  primals_548 = add_410 = None
    copy__231: "f32[256]" = torch.ops.aten.copy_.default(primals_549, add_417);  primals_549 = add_417 = None
    copy__232: "f32[256]" = torch.ops.aten.copy_.default(primals_550, add_418);  primals_550 = add_418 = None
    copy__233: "i64[]" = torch.ops.aten.copy_.default(primals_551, add_415);  primals_551 = add_415 = None
    copy__234: "f32[512]" = torch.ops.aten.copy_.default(primals_552, add_422);  primals_552 = add_422 = None
    copy__235: "f32[512]" = torch.ops.aten.copy_.default(primals_553, add_423);  primals_553 = add_423 = None
    copy__236: "i64[]" = torch.ops.aten.copy_.default(primals_554, add_420);  primals_554 = add_420 = None
    copy__237: "f32[256]" = torch.ops.aten.copy_.default(primals_555, add_428);  primals_555 = add_428 = None
    copy__238: "f32[256]" = torch.ops.aten.copy_.default(primals_556, add_429);  primals_556 = add_429 = None
    copy__239: "i64[]" = torch.ops.aten.copy_.default(primals_557, add_426);  primals_557 = add_426 = None
    copy__240: "f32[256]" = torch.ops.aten.copy_.default(primals_558, add_433);  primals_558 = add_433 = None
    copy__241: "f32[256]" = torch.ops.aten.copy_.default(primals_559, add_434);  primals_559 = add_434 = None
    copy__242: "i64[]" = torch.ops.aten.copy_.default(primals_560, add_431);  primals_560 = add_431 = None
    copy__243: "f32[512]" = torch.ops.aten.copy_.default(primals_561, add_438);  primals_561 = add_438 = None
    copy__244: "f32[512]" = torch.ops.aten.copy_.default(primals_562, add_439);  primals_562 = add_439 = None
    copy__245: "i64[]" = torch.ops.aten.copy_.default(primals_563, add_436);  primals_563 = add_436 = None
    copy__246: "f32[512]" = torch.ops.aten.copy_.default(primals_564, add_444);  primals_564 = add_444 = None
    copy__247: "f32[512]" = torch.ops.aten.copy_.default(primals_565, add_445);  primals_565 = add_445 = None
    copy__248: "i64[]" = torch.ops.aten.copy_.default(primals_566, add_442);  primals_566 = add_442 = None
    copy__249: "f32[256]" = torch.ops.aten.copy_.default(primals_567, add_450);  primals_567 = add_450 = None
    copy__250: "f32[256]" = torch.ops.aten.copy_.default(primals_568, add_451);  primals_568 = add_451 = None
    copy__251: "i64[]" = torch.ops.aten.copy_.default(primals_569, add_448);  primals_569 = add_448 = None
    copy__252: "f32[256]" = torch.ops.aten.copy_.default(primals_570, add_455);  primals_570 = add_455 = None
    copy__253: "f32[256]" = torch.ops.aten.copy_.default(primals_571, add_456);  primals_571 = add_456 = None
    copy__254: "i64[]" = torch.ops.aten.copy_.default(primals_572, add_453);  primals_572 = add_453 = None
    copy__255: "f32[512]" = torch.ops.aten.copy_.default(primals_573, add_460);  primals_573 = add_460 = None
    copy__256: "f32[512]" = torch.ops.aten.copy_.default(primals_574, add_461);  primals_574 = add_461 = None
    copy__257: "i64[]" = torch.ops.aten.copy_.default(primals_575, add_458);  primals_575 = add_458 = None
    copy__258: "f32[256]" = torch.ops.aten.copy_.default(primals_576, add_466);  primals_576 = add_466 = None
    copy__259: "f32[256]" = torch.ops.aten.copy_.default(primals_577, add_467);  primals_577 = add_467 = None
    copy__260: "i64[]" = torch.ops.aten.copy_.default(primals_578, add_464);  primals_578 = add_464 = None
    copy__261: "f32[256]" = torch.ops.aten.copy_.default(primals_579, add_471);  primals_579 = add_471 = None
    copy__262: "f32[256]" = torch.ops.aten.copy_.default(primals_580, add_472);  primals_580 = add_472 = None
    copy__263: "i64[]" = torch.ops.aten.copy_.default(primals_581, add_469);  primals_581 = add_469 = None
    copy__264: "f32[512]" = torch.ops.aten.copy_.default(primals_582, add_476);  primals_582 = add_476 = None
    copy__265: "f32[512]" = torch.ops.aten.copy_.default(primals_583, add_477);  primals_583 = add_477 = None
    copy__266: "i64[]" = torch.ops.aten.copy_.default(primals_584, add_474);  primals_584 = add_474 = None
    copy__267: "f32[512]" = torch.ops.aten.copy_.default(primals_585, add_482);  primals_585 = add_482 = None
    copy__268: "f32[512]" = torch.ops.aten.copy_.default(primals_586, add_483);  primals_586 = add_483 = None
    copy__269: "i64[]" = torch.ops.aten.copy_.default(primals_587, add_480);  primals_587 = add_480 = None
    copy__270: "f32[256]" = torch.ops.aten.copy_.default(primals_588, add_488);  primals_588 = add_488 = None
    copy__271: "f32[256]" = torch.ops.aten.copy_.default(primals_589, add_489);  primals_589 = add_489 = None
    copy__272: "i64[]" = torch.ops.aten.copy_.default(primals_590, add_486);  primals_590 = add_486 = None
    copy__273: "f32[256]" = torch.ops.aten.copy_.default(primals_591, add_493);  primals_591 = add_493 = None
    copy__274: "f32[256]" = torch.ops.aten.copy_.default(primals_592, add_494);  primals_592 = add_494 = None
    copy__275: "i64[]" = torch.ops.aten.copy_.default(primals_593, add_491);  primals_593 = add_491 = None
    copy__276: "f32[512]" = torch.ops.aten.copy_.default(primals_594, add_498);  primals_594 = add_498 = None
    copy__277: "f32[512]" = torch.ops.aten.copy_.default(primals_595, add_499);  primals_595 = add_499 = None
    copy__278: "i64[]" = torch.ops.aten.copy_.default(primals_596, add_496);  primals_596 = add_496 = None
    copy__279: "f32[256]" = torch.ops.aten.copy_.default(primals_597, add_504);  primals_597 = add_504 = None
    copy__280: "f32[256]" = torch.ops.aten.copy_.default(primals_598, add_505);  primals_598 = add_505 = None
    copy__281: "i64[]" = torch.ops.aten.copy_.default(primals_599, add_502);  primals_599 = add_502 = None
    copy__282: "f32[256]" = torch.ops.aten.copy_.default(primals_600, add_509);  primals_600 = add_509 = None
    copy__283: "f32[256]" = torch.ops.aten.copy_.default(primals_601, add_510);  primals_601 = add_510 = None
    copy__284: "i64[]" = torch.ops.aten.copy_.default(primals_602, add_507);  primals_602 = add_507 = None
    copy__285: "f32[512]" = torch.ops.aten.copy_.default(primals_603, add_514);  primals_603 = add_514 = None
    copy__286: "f32[512]" = torch.ops.aten.copy_.default(primals_604, add_515);  primals_604 = add_515 = None
    copy__287: "i64[]" = torch.ops.aten.copy_.default(primals_605, add_512);  primals_605 = add_512 = None
    copy__288: "f32[512]" = torch.ops.aten.copy_.default(primals_606, add_520);  primals_606 = add_520 = None
    copy__289: "f32[512]" = torch.ops.aten.copy_.default(primals_607, add_521);  primals_607 = add_521 = None
    copy__290: "i64[]" = torch.ops.aten.copy_.default(primals_608, add_518);  primals_608 = add_518 = None
    copy__291: "f32[1024]" = torch.ops.aten.copy_.default(primals_609, add_526);  primals_609 = add_526 = None
    copy__292: "f32[1024]" = torch.ops.aten.copy_.default(primals_610, add_527);  primals_610 = add_527 = None
    copy__293: "i64[]" = torch.ops.aten.copy_.default(primals_611, add_524);  primals_611 = add_524 = None
    copy__294: "f32[512]" = torch.ops.aten.copy_.default(primals_612, add_531);  primals_612 = add_531 = None
    copy__295: "f32[512]" = torch.ops.aten.copy_.default(primals_613, add_532);  primals_613 = add_532 = None
    copy__296: "i64[]" = torch.ops.aten.copy_.default(primals_614, add_529);  primals_614 = add_529 = None
    copy__297: "f32[512]" = torch.ops.aten.copy_.default(primals_615, add_536);  primals_615 = add_536 = None
    copy__298: "f32[512]" = torch.ops.aten.copy_.default(primals_616, add_537);  primals_616 = add_537 = None
    copy__299: "i64[]" = torch.ops.aten.copy_.default(primals_617, add_534);  primals_617 = add_534 = None
    copy__300: "f32[1024]" = torch.ops.aten.copy_.default(primals_618, add_541);  primals_618 = add_541 = None
    copy__301: "f32[1024]" = torch.ops.aten.copy_.default(primals_619, add_542);  primals_619 = add_542 = None
    copy__302: "i64[]" = torch.ops.aten.copy_.default(primals_620, add_539);  primals_620 = add_539 = None
    copy__303: "f32[512]" = torch.ops.aten.copy_.default(primals_621, add_547);  primals_621 = add_547 = None
    copy__304: "f32[512]" = torch.ops.aten.copy_.default(primals_622, add_548);  primals_622 = add_548 = None
    copy__305: "i64[]" = torch.ops.aten.copy_.default(primals_623, add_545);  primals_623 = add_545 = None
    copy__306: "f32[512]" = torch.ops.aten.copy_.default(primals_624, add_552);  primals_624 = add_552 = None
    copy__307: "f32[512]" = torch.ops.aten.copy_.default(primals_625, add_553);  primals_625 = add_553 = None
    copy__308: "i64[]" = torch.ops.aten.copy_.default(primals_626, add_550);  primals_626 = add_550 = None
    copy__309: "f32[1024]" = torch.ops.aten.copy_.default(primals_627, add_557);  primals_627 = add_557 = None
    copy__310: "f32[1024]" = torch.ops.aten.copy_.default(primals_628, add_558);  primals_628 = add_558 = None
    copy__311: "i64[]" = torch.ops.aten.copy_.default(primals_629, add_555);  primals_629 = add_555 = None
    copy__312: "f32[1024]" = torch.ops.aten.copy_.default(primals_630, add_563);  primals_630 = add_563 = None
    copy__313: "f32[1024]" = torch.ops.aten.copy_.default(primals_631, add_564);  primals_631 = add_564 = None
    copy__314: "i64[]" = torch.ops.aten.copy_.default(primals_632, add_561);  primals_632 = add_561 = None
    return pytree.tree_unflatten([view, getitem_544, mul_1679, sum_209, getitem_541, mul_1670, sum_207, getitem_538, mul_1661, sum_205, getitem_535, mul_1652, sum_203, getitem_532, mul_1643, sum_201, getitem_529, mul_1634, sum_199, getitem_526, mul_1625, sum_197, getitem_523, mul_1616, sum_195, getitem_520, mul_1607, sum_193, getitem_517, mul_1598, sum_191, getitem_514, mul_1589, sum_189, getitem_511, mul_1580, sum_187, getitem_508, mul_1571, sum_185, getitem_505, mul_1562, sum_183, getitem_502, mul_1553, sum_181, getitem_499, mul_1544, sum_179, getitem_496, mul_1535, sum_177, getitem_493, mul_1526, sum_175, getitem_490, mul_1517, sum_173, getitem_487, mul_1508, sum_171, getitem_484, mul_1499, sum_169, getitem_481, mul_1490, sum_167, getitem_478, mul_1481, sum_165, getitem_475, mul_1472, sum_163, getitem_472, mul_1463, sum_161, getitem_469, mul_1454, sum_159, getitem_466, mul_1445, sum_157, getitem_463, mul_1436, sum_155, getitem_460, mul_1427, sum_153, getitem_457, mul_1418, sum_151, getitem_454, mul_1409, sum_149, getitem_451, mul_1400, sum_147, getitem_448, mul_1391, sum_145, getitem_445, mul_1382, sum_143, getitem_442, mul_1373, sum_141, getitem_439, mul_1364, sum_139, getitem_436, mul_1355, sum_137, getitem_433, mul_1346, sum_135, getitem_430, mul_1337, sum_133, getitem_427, mul_1328, sum_131, getitem_424, mul_1319, sum_129, getitem_421, mul_1310, sum_127, getitem_418, mul_1301, sum_125, getitem_415, mul_1292, sum_123, getitem_412, mul_1283, sum_121, getitem_409, mul_1274, sum_119, getitem_406, mul_1265, sum_117, getitem_403, mul_1256, sum_115, getitem_400, mul_1247, sum_113, getitem_397, mul_1238, sum_111, getitem_394, mul_1229, sum_109, getitem_391, mul_1220, sum_107, getitem_388, mul_1211, sum_105, getitem_385, mul_1202, sum_103, getitem_382, mul_1193, sum_101, getitem_379, mul_1184, sum_99, getitem_376, mul_1175, sum_97, getitem_373, mul_1166, sum_95, getitem_370, mul_1157, sum_93, getitem_367, mul_1148, sum_91, getitem_364, mul_1139, sum_89, getitem_361, mul_1130, sum_87, getitem_358, mul_1121, sum_85, getitem_355, mul_1112, sum_83, getitem_352, mul_1103, sum_81, getitem_349, mul_1094, sum_79, getitem_346, mul_1085, sum_77, getitem_343, mul_1076, sum_75, getitem_340, mul_1067, sum_73, getitem_337, mul_1058, sum_71, getitem_334, mul_1049, sum_69, getitem_331, mul_1040, sum_67, getitem_328, mul_1031, sum_65, getitem_325, mul_1022, sum_63, getitem_322, mul_1013, sum_61, getitem_319, mul_1004, sum_59, getitem_316, mul_995, sum_57, getitem_313, mul_986, sum_55, getitem_310, mul_977, sum_53, getitem_307, mul_968, sum_51, getitem_304, mul_959, sum_49, getitem_301, mul_950, sum_47, getitem_298, mul_941, sum_45, getitem_295, mul_932, sum_43, getitem_292, mul_923, sum_41, getitem_289, mul_914, sum_39, getitem_286, mul_905, sum_37, getitem_283, mul_896, sum_35, getitem_280, mul_887, sum_33, getitem_277, mul_878, sum_31, getitem_274, mul_869, sum_29, getitem_271, mul_860, sum_27, getitem_268, mul_851, sum_25, getitem_265, mul_842, sum_23, getitem_262, mul_833, sum_21, getitem_259, mul_824, sum_19, getitem_256, mul_815, sum_17, getitem_253, mul_806, sum_15, getitem_250, mul_797, sum_13, getitem_247, mul_788, sum_11, getitem_244, mul_779, sum_9, getitem_241, mul_770, sum_7, getitem_238, mul_761, sum_5, getitem_235, mul_752, sum_3, getitem_232, mul_743, sum_1, getitem_229, getitem_230, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    