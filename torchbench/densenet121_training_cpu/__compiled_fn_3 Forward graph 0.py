from __future__ import annotations



def forward(self, primals_1: "f32[64, 3, 7, 7]", primals_2: "f32[64]", primals_3: "f32[64]", primals_4: "f32[64]", primals_5: "f32[64]", primals_6: "f32[128, 64, 1, 1]", primals_7: "f32[128]", primals_8: "f32[128]", primals_9: "f32[32, 128, 3, 3]", primals_10: "f32[96]", primals_11: "f32[96]", primals_12: "f32[128, 96, 1, 1]", primals_13: "f32[128]", primals_14: "f32[128]", primals_15: "f32[32, 128, 3, 3]", primals_16: "f32[128]", primals_17: "f32[128]", primals_18: "f32[128, 128, 1, 1]", primals_19: "f32[128]", primals_20: "f32[128]", primals_21: "f32[32, 128, 3, 3]", primals_22: "f32[160]", primals_23: "f32[160]", primals_24: "f32[128, 160, 1, 1]", primals_25: "f32[128]", primals_26: "f32[128]", primals_27: "f32[32, 128, 3, 3]", primals_28: "f32[192]", primals_29: "f32[192]", primals_30: "f32[128, 192, 1, 1]", primals_31: "f32[128]", primals_32: "f32[128]", primals_33: "f32[32, 128, 3, 3]", primals_34: "f32[224]", primals_35: "f32[224]", primals_36: "f32[128, 224, 1, 1]", primals_37: "f32[128]", primals_38: "f32[128]", primals_39: "f32[32, 128, 3, 3]", primals_40: "f32[256]", primals_41: "f32[256]", primals_42: "f32[128, 256, 1, 1]", primals_43: "f32[128]", primals_44: "f32[128]", primals_45: "f32[128, 128, 1, 1]", primals_46: "f32[128]", primals_47: "f32[128]", primals_48: "f32[32, 128, 3, 3]", primals_49: "f32[160]", primals_50: "f32[160]", primals_51: "f32[128, 160, 1, 1]", primals_52: "f32[128]", primals_53: "f32[128]", primals_54: "f32[32, 128, 3, 3]", primals_55: "f32[192]", primals_56: "f32[192]", primals_57: "f32[128, 192, 1, 1]", primals_58: "f32[128]", primals_59: "f32[128]", primals_60: "f32[32, 128, 3, 3]", primals_61: "f32[224]", primals_62: "f32[224]", primals_63: "f32[128, 224, 1, 1]", primals_64: "f32[128]", primals_65: "f32[128]", primals_66: "f32[32, 128, 3, 3]", primals_67: "f32[256]", primals_68: "f32[256]", primals_69: "f32[128, 256, 1, 1]", primals_70: "f32[128]", primals_71: "f32[128]", primals_72: "f32[32, 128, 3, 3]", primals_73: "f32[288]", primals_74: "f32[288]", primals_75: "f32[128, 288, 1, 1]", primals_76: "f32[128]", primals_77: "f32[128]", primals_78: "f32[32, 128, 3, 3]", primals_79: "f32[320]", primals_80: "f32[320]", primals_81: "f32[128, 320, 1, 1]", primals_82: "f32[128]", primals_83: "f32[128]", primals_84: "f32[32, 128, 3, 3]", primals_85: "f32[352]", primals_86: "f32[352]", primals_87: "f32[128, 352, 1, 1]", primals_88: "f32[128]", primals_89: "f32[128]", primals_90: "f32[32, 128, 3, 3]", primals_91: "f32[384]", primals_92: "f32[384]", primals_93: "f32[128, 384, 1, 1]", primals_94: "f32[128]", primals_95: "f32[128]", primals_96: "f32[32, 128, 3, 3]", primals_97: "f32[416]", primals_98: "f32[416]", primals_99: "f32[128, 416, 1, 1]", primals_100: "f32[128]", primals_101: "f32[128]", primals_102: "f32[32, 128, 3, 3]", primals_103: "f32[448]", primals_104: "f32[448]", primals_105: "f32[128, 448, 1, 1]", primals_106: "f32[128]", primals_107: "f32[128]", primals_108: "f32[32, 128, 3, 3]", primals_109: "f32[480]", primals_110: "f32[480]", primals_111: "f32[128, 480, 1, 1]", primals_112: "f32[128]", primals_113: "f32[128]", primals_114: "f32[32, 128, 3, 3]", primals_115: "f32[512]", primals_116: "f32[512]", primals_117: "f32[256, 512, 1, 1]", primals_118: "f32[256]", primals_119: "f32[256]", primals_120: "f32[128, 256, 1, 1]", primals_121: "f32[128]", primals_122: "f32[128]", primals_123: "f32[32, 128, 3, 3]", primals_124: "f32[288]", primals_125: "f32[288]", primals_126: "f32[128, 288, 1, 1]", primals_127: "f32[128]", primals_128: "f32[128]", primals_129: "f32[32, 128, 3, 3]", primals_130: "f32[320]", primals_131: "f32[320]", primals_132: "f32[128, 320, 1, 1]", primals_133: "f32[128]", primals_134: "f32[128]", primals_135: "f32[32, 128, 3, 3]", primals_136: "f32[352]", primals_137: "f32[352]", primals_138: "f32[128, 352, 1, 1]", primals_139: "f32[128]", primals_140: "f32[128]", primals_141: "f32[32, 128, 3, 3]", primals_142: "f32[384]", primals_143: "f32[384]", primals_144: "f32[128, 384, 1, 1]", primals_145: "f32[128]", primals_146: "f32[128]", primals_147: "f32[32, 128, 3, 3]", primals_148: "f32[416]", primals_149: "f32[416]", primals_150: "f32[128, 416, 1, 1]", primals_151: "f32[128]", primals_152: "f32[128]", primals_153: "f32[32, 128, 3, 3]", primals_154: "f32[448]", primals_155: "f32[448]", primals_156: "f32[128, 448, 1, 1]", primals_157: "f32[128]", primals_158: "f32[128]", primals_159: "f32[32, 128, 3, 3]", primals_160: "f32[480]", primals_161: "f32[480]", primals_162: "f32[128, 480, 1, 1]", primals_163: "f32[128]", primals_164: "f32[128]", primals_165: "f32[32, 128, 3, 3]", primals_166: "f32[512]", primals_167: "f32[512]", primals_168: "f32[128, 512, 1, 1]", primals_169: "f32[128]", primals_170: "f32[128]", primals_171: "f32[32, 128, 3, 3]", primals_172: "f32[544]", primals_173: "f32[544]", primals_174: "f32[128, 544, 1, 1]", primals_175: "f32[128]", primals_176: "f32[128]", primals_177: "f32[32, 128, 3, 3]", primals_178: "f32[576]", primals_179: "f32[576]", primals_180: "f32[128, 576, 1, 1]", primals_181: "f32[128]", primals_182: "f32[128]", primals_183: "f32[32, 128, 3, 3]", primals_184: "f32[608]", primals_185: "f32[608]", primals_186: "f32[128, 608, 1, 1]", primals_187: "f32[128]", primals_188: "f32[128]", primals_189: "f32[32, 128, 3, 3]", primals_190: "f32[640]", primals_191: "f32[640]", primals_192: "f32[128, 640, 1, 1]", primals_193: "f32[128]", primals_194: "f32[128]", primals_195: "f32[32, 128, 3, 3]", primals_196: "f32[672]", primals_197: "f32[672]", primals_198: "f32[128, 672, 1, 1]", primals_199: "f32[128]", primals_200: "f32[128]", primals_201: "f32[32, 128, 3, 3]", primals_202: "f32[704]", primals_203: "f32[704]", primals_204: "f32[128, 704, 1, 1]", primals_205: "f32[128]", primals_206: "f32[128]", primals_207: "f32[32, 128, 3, 3]", primals_208: "f32[736]", primals_209: "f32[736]", primals_210: "f32[128, 736, 1, 1]", primals_211: "f32[128]", primals_212: "f32[128]", primals_213: "f32[32, 128, 3, 3]", primals_214: "f32[768]", primals_215: "f32[768]", primals_216: "f32[128, 768, 1, 1]", primals_217: "f32[128]", primals_218: "f32[128]", primals_219: "f32[32, 128, 3, 3]", primals_220: "f32[800]", primals_221: "f32[800]", primals_222: "f32[128, 800, 1, 1]", primals_223: "f32[128]", primals_224: "f32[128]", primals_225: "f32[32, 128, 3, 3]", primals_226: "f32[832]", primals_227: "f32[832]", primals_228: "f32[128, 832, 1, 1]", primals_229: "f32[128]", primals_230: "f32[128]", primals_231: "f32[32, 128, 3, 3]", primals_232: "f32[864]", primals_233: "f32[864]", primals_234: "f32[128, 864, 1, 1]", primals_235: "f32[128]", primals_236: "f32[128]", primals_237: "f32[32, 128, 3, 3]", primals_238: "f32[896]", primals_239: "f32[896]", primals_240: "f32[128, 896, 1, 1]", primals_241: "f32[128]", primals_242: "f32[128]", primals_243: "f32[32, 128, 3, 3]", primals_244: "f32[928]", primals_245: "f32[928]", primals_246: "f32[128, 928, 1, 1]", primals_247: "f32[128]", primals_248: "f32[128]", primals_249: "f32[32, 128, 3, 3]", primals_250: "f32[960]", primals_251: "f32[960]", primals_252: "f32[128, 960, 1, 1]", primals_253: "f32[128]", primals_254: "f32[128]", primals_255: "f32[32, 128, 3, 3]", primals_256: "f32[992]", primals_257: "f32[992]", primals_258: "f32[128, 992, 1, 1]", primals_259: "f32[128]", primals_260: "f32[128]", primals_261: "f32[32, 128, 3, 3]", primals_262: "f32[1024]", primals_263: "f32[1024]", primals_264: "f32[512, 1024, 1, 1]", primals_265: "f32[512]", primals_266: "f32[512]", primals_267: "f32[128, 512, 1, 1]", primals_268: "f32[128]", primals_269: "f32[128]", primals_270: "f32[32, 128, 3, 3]", primals_271: "f32[544]", primals_272: "f32[544]", primals_273: "f32[128, 544, 1, 1]", primals_274: "f32[128]", primals_275: "f32[128]", primals_276: "f32[32, 128, 3, 3]", primals_277: "f32[576]", primals_278: "f32[576]", primals_279: "f32[128, 576, 1, 1]", primals_280: "f32[128]", primals_281: "f32[128]", primals_282: "f32[32, 128, 3, 3]", primals_283: "f32[608]", primals_284: "f32[608]", primals_285: "f32[128, 608, 1, 1]", primals_286: "f32[128]", primals_287: "f32[128]", primals_288: "f32[32, 128, 3, 3]", primals_289: "f32[640]", primals_290: "f32[640]", primals_291: "f32[128, 640, 1, 1]", primals_292: "f32[128]", primals_293: "f32[128]", primals_294: "f32[32, 128, 3, 3]", primals_295: "f32[672]", primals_296: "f32[672]", primals_297: "f32[128, 672, 1, 1]", primals_298: "f32[128]", primals_299: "f32[128]", primals_300: "f32[32, 128, 3, 3]", primals_301: "f32[704]", primals_302: "f32[704]", primals_303: "f32[128, 704, 1, 1]", primals_304: "f32[128]", primals_305: "f32[128]", primals_306: "f32[32, 128, 3, 3]", primals_307: "f32[736]", primals_308: "f32[736]", primals_309: "f32[128, 736, 1, 1]", primals_310: "f32[128]", primals_311: "f32[128]", primals_312: "f32[32, 128, 3, 3]", primals_313: "f32[768]", primals_314: "f32[768]", primals_315: "f32[128, 768, 1, 1]", primals_316: "f32[128]", primals_317: "f32[128]", primals_318: "f32[32, 128, 3, 3]", primals_319: "f32[800]", primals_320: "f32[800]", primals_321: "f32[128, 800, 1, 1]", primals_322: "f32[128]", primals_323: "f32[128]", primals_324: "f32[32, 128, 3, 3]", primals_325: "f32[832]", primals_326: "f32[832]", primals_327: "f32[128, 832, 1, 1]", primals_328: "f32[128]", primals_329: "f32[128]", primals_330: "f32[32, 128, 3, 3]", primals_331: "f32[864]", primals_332: "f32[864]", primals_333: "f32[128, 864, 1, 1]", primals_334: "f32[128]", primals_335: "f32[128]", primals_336: "f32[32, 128, 3, 3]", primals_337: "f32[896]", primals_338: "f32[896]", primals_339: "f32[128, 896, 1, 1]", primals_340: "f32[128]", primals_341: "f32[128]", primals_342: "f32[32, 128, 3, 3]", primals_343: "f32[928]", primals_344: "f32[928]", primals_345: "f32[128, 928, 1, 1]", primals_346: "f32[128]", primals_347: "f32[128]", primals_348: "f32[32, 128, 3, 3]", primals_349: "f32[960]", primals_350: "f32[960]", primals_351: "f32[128, 960, 1, 1]", primals_352: "f32[128]", primals_353: "f32[128]", primals_354: "f32[32, 128, 3, 3]", primals_355: "f32[992]", primals_356: "f32[992]", primals_357: "f32[128, 992, 1, 1]", primals_358: "f32[128]", primals_359: "f32[128]", primals_360: "f32[32, 128, 3, 3]", primals_361: "f32[1024]", primals_362: "f32[1024]", primals_363: "f32[1000, 1024]", primals_364: "f32[1000]", primals_365: "f32[64]", primals_366: "f32[64]", primals_367: "i64[]", primals_368: "f32[64]", primals_369: "f32[64]", primals_370: "i64[]", primals_371: "f32[128]", primals_372: "f32[128]", primals_373: "i64[]", primals_374: "f32[96]", primals_375: "f32[96]", primals_376: "i64[]", primals_377: "f32[128]", primals_378: "f32[128]", primals_379: "i64[]", primals_380: "f32[128]", primals_381: "f32[128]", primals_382: "i64[]", primals_383: "f32[128]", primals_384: "f32[128]", primals_385: "i64[]", primals_386: "f32[160]", primals_387: "f32[160]", primals_388: "i64[]", primals_389: "f32[128]", primals_390: "f32[128]", primals_391: "i64[]", primals_392: "f32[192]", primals_393: "f32[192]", primals_394: "i64[]", primals_395: "f32[128]", primals_396: "f32[128]", primals_397: "i64[]", primals_398: "f32[224]", primals_399: "f32[224]", primals_400: "i64[]", primals_401: "f32[128]", primals_402: "f32[128]", primals_403: "i64[]", primals_404: "f32[256]", primals_405: "f32[256]", primals_406: "i64[]", primals_407: "f32[128]", primals_408: "f32[128]", primals_409: "i64[]", primals_410: "f32[128]", primals_411: "f32[128]", primals_412: "i64[]", primals_413: "f32[160]", primals_414: "f32[160]", primals_415: "i64[]", primals_416: "f32[128]", primals_417: "f32[128]", primals_418: "i64[]", primals_419: "f32[192]", primals_420: "f32[192]", primals_421: "i64[]", primals_422: "f32[128]", primals_423: "f32[128]", primals_424: "i64[]", primals_425: "f32[224]", primals_426: "f32[224]", primals_427: "i64[]", primals_428: "f32[128]", primals_429: "f32[128]", primals_430: "i64[]", primals_431: "f32[256]", primals_432: "f32[256]", primals_433: "i64[]", primals_434: "f32[128]", primals_435: "f32[128]", primals_436: "i64[]", primals_437: "f32[288]", primals_438: "f32[288]", primals_439: "i64[]", primals_440: "f32[128]", primals_441: "f32[128]", primals_442: "i64[]", primals_443: "f32[320]", primals_444: "f32[320]", primals_445: "i64[]", primals_446: "f32[128]", primals_447: "f32[128]", primals_448: "i64[]", primals_449: "f32[352]", primals_450: "f32[352]", primals_451: "i64[]", primals_452: "f32[128]", primals_453: "f32[128]", primals_454: "i64[]", primals_455: "f32[384]", primals_456: "f32[384]", primals_457: "i64[]", primals_458: "f32[128]", primals_459: "f32[128]", primals_460: "i64[]", primals_461: "f32[416]", primals_462: "f32[416]", primals_463: "i64[]", primals_464: "f32[128]", primals_465: "f32[128]", primals_466: "i64[]", primals_467: "f32[448]", primals_468: "f32[448]", primals_469: "i64[]", primals_470: "f32[128]", primals_471: "f32[128]", primals_472: "i64[]", primals_473: "f32[480]", primals_474: "f32[480]", primals_475: "i64[]", primals_476: "f32[128]", primals_477: "f32[128]", primals_478: "i64[]", primals_479: "f32[512]", primals_480: "f32[512]", primals_481: "i64[]", primals_482: "f32[256]", primals_483: "f32[256]", primals_484: "i64[]", primals_485: "f32[128]", primals_486: "f32[128]", primals_487: "i64[]", primals_488: "f32[288]", primals_489: "f32[288]", primals_490: "i64[]", primals_491: "f32[128]", primals_492: "f32[128]", primals_493: "i64[]", primals_494: "f32[320]", primals_495: "f32[320]", primals_496: "i64[]", primals_497: "f32[128]", primals_498: "f32[128]", primals_499: "i64[]", primals_500: "f32[352]", primals_501: "f32[352]", primals_502: "i64[]", primals_503: "f32[128]", primals_504: "f32[128]", primals_505: "i64[]", primals_506: "f32[384]", primals_507: "f32[384]", primals_508: "i64[]", primals_509: "f32[128]", primals_510: "f32[128]", primals_511: "i64[]", primals_512: "f32[416]", primals_513: "f32[416]", primals_514: "i64[]", primals_515: "f32[128]", primals_516: "f32[128]", primals_517: "i64[]", primals_518: "f32[448]", primals_519: "f32[448]", primals_520: "i64[]", primals_521: "f32[128]", primals_522: "f32[128]", primals_523: "i64[]", primals_524: "f32[480]", primals_525: "f32[480]", primals_526: "i64[]", primals_527: "f32[128]", primals_528: "f32[128]", primals_529: "i64[]", primals_530: "f32[512]", primals_531: "f32[512]", primals_532: "i64[]", primals_533: "f32[128]", primals_534: "f32[128]", primals_535: "i64[]", primals_536: "f32[544]", primals_537: "f32[544]", primals_538: "i64[]", primals_539: "f32[128]", primals_540: "f32[128]", primals_541: "i64[]", primals_542: "f32[576]", primals_543: "f32[576]", primals_544: "i64[]", primals_545: "f32[128]", primals_546: "f32[128]", primals_547: "i64[]", primals_548: "f32[608]", primals_549: "f32[608]", primals_550: "i64[]", primals_551: "f32[128]", primals_552: "f32[128]", primals_553: "i64[]", primals_554: "f32[640]", primals_555: "f32[640]", primals_556: "i64[]", primals_557: "f32[128]", primals_558: "f32[128]", primals_559: "i64[]", primals_560: "f32[672]", primals_561: "f32[672]", primals_562: "i64[]", primals_563: "f32[128]", primals_564: "f32[128]", primals_565: "i64[]", primals_566: "f32[704]", primals_567: "f32[704]", primals_568: "i64[]", primals_569: "f32[128]", primals_570: "f32[128]", primals_571: "i64[]", primals_572: "f32[736]", primals_573: "f32[736]", primals_574: "i64[]", primals_575: "f32[128]", primals_576: "f32[128]", primals_577: "i64[]", primals_578: "f32[768]", primals_579: "f32[768]", primals_580: "i64[]", primals_581: "f32[128]", primals_582: "f32[128]", primals_583: "i64[]", primals_584: "f32[800]", primals_585: "f32[800]", primals_586: "i64[]", primals_587: "f32[128]", primals_588: "f32[128]", primals_589: "i64[]", primals_590: "f32[832]", primals_591: "f32[832]", primals_592: "i64[]", primals_593: "f32[128]", primals_594: "f32[128]", primals_595: "i64[]", primals_596: "f32[864]", primals_597: "f32[864]", primals_598: "i64[]", primals_599: "f32[128]", primals_600: "f32[128]", primals_601: "i64[]", primals_602: "f32[896]", primals_603: "f32[896]", primals_604: "i64[]", primals_605: "f32[128]", primals_606: "f32[128]", primals_607: "i64[]", primals_608: "f32[928]", primals_609: "f32[928]", primals_610: "i64[]", primals_611: "f32[128]", primals_612: "f32[128]", primals_613: "i64[]", primals_614: "f32[960]", primals_615: "f32[960]", primals_616: "i64[]", primals_617: "f32[128]", primals_618: "f32[128]", primals_619: "i64[]", primals_620: "f32[992]", primals_621: "f32[992]", primals_622: "i64[]", primals_623: "f32[128]", primals_624: "f32[128]", primals_625: "i64[]", primals_626: "f32[1024]", primals_627: "f32[1024]", primals_628: "i64[]", primals_629: "f32[512]", primals_630: "f32[512]", primals_631: "i64[]", primals_632: "f32[128]", primals_633: "f32[128]", primals_634: "i64[]", primals_635: "f32[544]", primals_636: "f32[544]", primals_637: "i64[]", primals_638: "f32[128]", primals_639: "f32[128]", primals_640: "i64[]", primals_641: "f32[576]", primals_642: "f32[576]", primals_643: "i64[]", primals_644: "f32[128]", primals_645: "f32[128]", primals_646: "i64[]", primals_647: "f32[608]", primals_648: "f32[608]", primals_649: "i64[]", primals_650: "f32[128]", primals_651: "f32[128]", primals_652: "i64[]", primals_653: "f32[640]", primals_654: "f32[640]", primals_655: "i64[]", primals_656: "f32[128]", primals_657: "f32[128]", primals_658: "i64[]", primals_659: "f32[672]", primals_660: "f32[672]", primals_661: "i64[]", primals_662: "f32[128]", primals_663: "f32[128]", primals_664: "i64[]", primals_665: "f32[704]", primals_666: "f32[704]", primals_667: "i64[]", primals_668: "f32[128]", primals_669: "f32[128]", primals_670: "i64[]", primals_671: "f32[736]", primals_672: "f32[736]", primals_673: "i64[]", primals_674: "f32[128]", primals_675: "f32[128]", primals_676: "i64[]", primals_677: "f32[768]", primals_678: "f32[768]", primals_679: "i64[]", primals_680: "f32[128]", primals_681: "f32[128]", primals_682: "i64[]", primals_683: "f32[800]", primals_684: "f32[800]", primals_685: "i64[]", primals_686: "f32[128]", primals_687: "f32[128]", primals_688: "i64[]", primals_689: "f32[832]", primals_690: "f32[832]", primals_691: "i64[]", primals_692: "f32[128]", primals_693: "f32[128]", primals_694: "i64[]", primals_695: "f32[864]", primals_696: "f32[864]", primals_697: "i64[]", primals_698: "f32[128]", primals_699: "f32[128]", primals_700: "i64[]", primals_701: "f32[896]", primals_702: "f32[896]", primals_703: "i64[]", primals_704: "f32[128]", primals_705: "f32[128]", primals_706: "i64[]", primals_707: "f32[928]", primals_708: "f32[928]", primals_709: "i64[]", primals_710: "f32[128]", primals_711: "f32[128]", primals_712: "i64[]", primals_713: "f32[960]", primals_714: "f32[960]", primals_715: "i64[]", primals_716: "f32[128]", primals_717: "f32[128]", primals_718: "i64[]", primals_719: "f32[992]", primals_720: "f32[992]", primals_721: "i64[]", primals_722: "f32[128]", primals_723: "f32[128]", primals_724: "i64[]", primals_725: "f32[1024]", primals_726: "f32[1024]", primals_727: "i64[]", primals_728: "f32[4, 3, 224, 224]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    convolution: "f32[4, 64, 112, 112]" = torch.ops.aten.convolution.default(primals_728, primals_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
    convert_element_type: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_365, torch.float32)
    convert_element_type_1: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_366, torch.float32)
    add: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
    sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add);  add = None
    reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
    mul: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
    unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
    unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
    unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    sub: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  unsqueeze_1 = None
    mul_1: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
    unsqueeze_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1)
    unsqueeze_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_2: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
    unsqueeze_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
    unsqueeze_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_1: "f32[4, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
    relu: "f32[4, 64, 112, 112]" = torch.ops.aten.relu.default(add_1);  add_1 = None
    max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices.default(relu, [3, 3], [2, 2], [1, 1])
    getitem: "f32[4, 64, 56, 56]" = max_pool2d_with_indices[0]
    getitem_1: "i64[4, 64, 56, 56]" = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    clone: "f32[4, 64, 56, 56]" = torch.ops.aten.clone.default(getitem)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_2: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_368, torch.float32)
    convert_element_type_3: "f32[64]" = torch.ops.prims.convert_element_type.default(primals_369, torch.float32)
    add_2: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1e-05);  convert_element_type_3 = None
    sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
    reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
    mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
    unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
    unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
    unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    sub_1: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(clone, unsqueeze_9);  unsqueeze_9 = None
    mul_4: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
    unsqueeze_12: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1)
    unsqueeze_13: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_5: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
    unsqueeze_14: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1);  primals_5 = None
    unsqueeze_15: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_3: "f32[4, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
    relu_1: "f32[4, 64, 56, 56]" = torch.ops.aten.relu.default(add_3);  add_3 = None
    convolution_1: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_6, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_4: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_371, torch.float32)
    convert_element_type_5: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_372, torch.float32)
    add_4: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_5, 1e-05);  convert_element_type_5 = None
    sqrt_2: "f32[128]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
    reciprocal_2: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
    mul_6: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
    unsqueeze_16: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, -1);  convert_element_type_4 = None
    unsqueeze_17: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    unsqueeze_18: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
    unsqueeze_19: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    sub_2: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_17);  unsqueeze_17 = None
    mul_7: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
    unsqueeze_20: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_21: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_8: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
    unsqueeze_22: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_23: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_5: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
    relu_2: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_5);  add_5 = None
    convolution_2: "f32[4, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_2, primals_9, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat: "f32[4, 96, 56, 56]" = torch.ops.aten.cat.default([getitem, convolution_2], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_6: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_374, torch.float32)
    convert_element_type_7: "f32[96]" = torch.ops.prims.convert_element_type.default(primals_375, torch.float32)
    add_6: "f32[96]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
    sqrt_3: "f32[96]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
    reciprocal_3: "f32[96]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
    mul_9: "f32[96]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
    unsqueeze_24: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
    unsqueeze_25: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    unsqueeze_26: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
    unsqueeze_27: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    sub_3: "f32[4, 96, 56, 56]" = torch.ops.aten.sub.Tensor(cat, unsqueeze_25);  unsqueeze_25 = None
    mul_10: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
    unsqueeze_28: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1)
    unsqueeze_29: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_11: "f32[4, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
    unsqueeze_30: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1);  primals_11 = None
    unsqueeze_31: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_7: "f32[4, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
    relu_3: "f32[4, 96, 56, 56]" = torch.ops.aten.relu.default(add_7);  add_7 = None
    convolution_3: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_3, primals_12, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_8: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_377, torch.float32)
    convert_element_type_9: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_378, torch.float32)
    add_8: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_9, 1e-05);  convert_element_type_9 = None
    sqrt_4: "f32[128]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
    reciprocal_4: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
    mul_12: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
    unsqueeze_32: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, -1);  convert_element_type_8 = None
    unsqueeze_33: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    unsqueeze_34: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
    unsqueeze_35: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    sub_4: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_33);  unsqueeze_33 = None
    mul_13: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
    unsqueeze_36: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_37: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_14: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
    unsqueeze_38: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_39: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_9: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
    relu_4: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_9);  add_9 = None
    convolution_4: "f32[4, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_15, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_1: "f32[4, 128, 56, 56]" = torch.ops.aten.cat.default([getitem, convolution_2, convolution_4], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_10: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_380, torch.float32)
    convert_element_type_11: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_381, torch.float32)
    add_10: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_11, 1e-05);  convert_element_type_11 = None
    sqrt_5: "f32[128]" = torch.ops.aten.sqrt.default(add_10);  add_10 = None
    reciprocal_5: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
    mul_15: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
    unsqueeze_40: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, -1);  convert_element_type_10 = None
    unsqueeze_41: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    unsqueeze_42: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
    unsqueeze_43: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    sub_5: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(cat_1, unsqueeze_41);  unsqueeze_41 = None
    mul_16: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
    unsqueeze_44: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1)
    unsqueeze_45: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_17: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
    unsqueeze_46: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1);  primals_17 = None
    unsqueeze_47: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_11: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
    relu_5: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_11);  add_11 = None
    convolution_5: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_18, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_12: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_383, torch.float32)
    convert_element_type_13: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_384, torch.float32)
    add_12: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
    sqrt_6: "f32[128]" = torch.ops.aten.sqrt.default(add_12);  add_12 = None
    reciprocal_6: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
    mul_18: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
    unsqueeze_48: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
    unsqueeze_49: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    unsqueeze_50: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
    unsqueeze_51: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    sub_6: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_49);  unsqueeze_49 = None
    mul_19: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
    unsqueeze_52: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_53: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_20: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
    unsqueeze_54: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_55: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_13: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
    relu_6: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_13);  add_13 = None
    convolution_6: "f32[4, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_6, primals_21, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_2: "f32[4, 160, 56, 56]" = torch.ops.aten.cat.default([getitem, convolution_2, convolution_4, convolution_6], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_14: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_386, torch.float32)
    convert_element_type_15: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_387, torch.float32)
    add_14: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_15, 1e-05);  convert_element_type_15 = None
    sqrt_7: "f32[160]" = torch.ops.aten.sqrt.default(add_14);  add_14 = None
    reciprocal_7: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
    mul_21: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
    unsqueeze_56: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_14, -1);  convert_element_type_14 = None
    unsqueeze_57: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    unsqueeze_58: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
    unsqueeze_59: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    sub_7: "f32[4, 160, 56, 56]" = torch.ops.aten.sub.Tensor(cat_2, unsqueeze_57);  unsqueeze_57 = None
    mul_22: "f32[4, 160, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
    unsqueeze_60: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1)
    unsqueeze_61: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_23: "f32[4, 160, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
    unsqueeze_62: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1);  primals_23 = None
    unsqueeze_63: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_15: "f32[4, 160, 56, 56]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
    relu_7: "f32[4, 160, 56, 56]" = torch.ops.aten.relu.default(add_15);  add_15 = None
    convolution_7: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_7, primals_24, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_16: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_389, torch.float32)
    convert_element_type_17: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_390, torch.float32)
    add_16: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_17, 1e-05);  convert_element_type_17 = None
    sqrt_8: "f32[128]" = torch.ops.aten.sqrt.default(add_16);  add_16 = None
    reciprocal_8: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
    mul_24: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
    unsqueeze_64: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, -1);  convert_element_type_16 = None
    unsqueeze_65: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    unsqueeze_66: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
    unsqueeze_67: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    sub_8: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_65);  unsqueeze_65 = None
    mul_25: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
    unsqueeze_68: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_69: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_26: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
    unsqueeze_70: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_71: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_17: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
    relu_8: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_17);  add_17 = None
    convolution_8: "f32[4, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_8, primals_27, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_3: "f32[4, 192, 56, 56]" = torch.ops.aten.cat.default([getitem, convolution_2, convolution_4, convolution_6, convolution_8], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_18: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_392, torch.float32)
    convert_element_type_19: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_393, torch.float32)
    add_18: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
    sqrt_9: "f32[192]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
    reciprocal_9: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
    mul_27: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
    unsqueeze_72: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
    unsqueeze_73: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    unsqueeze_74: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
    unsqueeze_75: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    sub_9: "f32[4, 192, 56, 56]" = torch.ops.aten.sub.Tensor(cat_3, unsqueeze_73);  unsqueeze_73 = None
    mul_28: "f32[4, 192, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
    unsqueeze_76: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1)
    unsqueeze_77: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_29: "f32[4, 192, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
    unsqueeze_78: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1);  primals_29 = None
    unsqueeze_79: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_19: "f32[4, 192, 56, 56]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
    relu_9: "f32[4, 192, 56, 56]" = torch.ops.aten.relu.default(add_19);  add_19 = None
    convolution_9: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_9, primals_30, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_20: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_395, torch.float32)
    convert_element_type_21: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_396, torch.float32)
    add_20: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_21, 1e-05);  convert_element_type_21 = None
    sqrt_10: "f32[128]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
    reciprocal_10: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
    mul_30: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
    unsqueeze_80: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_20, -1);  convert_element_type_20 = None
    unsqueeze_81: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    unsqueeze_82: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
    unsqueeze_83: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    sub_10: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_81);  unsqueeze_81 = None
    mul_31: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
    unsqueeze_84: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_85: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_32: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
    unsqueeze_86: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_87: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_21: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
    relu_10: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_21);  add_21 = None
    convolution_10: "f32[4, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_10, primals_33, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_4: "f32[4, 224, 56, 56]" = torch.ops.aten.cat.default([getitem, convolution_2, convolution_4, convolution_6, convolution_8, convolution_10], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_22: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_398, torch.float32)
    convert_element_type_23: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_399, torch.float32)
    add_22: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_23, 1e-05);  convert_element_type_23 = None
    sqrt_11: "f32[224]" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
    reciprocal_11: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
    mul_33: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
    unsqueeze_88: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, -1);  convert_element_type_22 = None
    unsqueeze_89: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    unsqueeze_90: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
    unsqueeze_91: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    sub_11: "f32[4, 224, 56, 56]" = torch.ops.aten.sub.Tensor(cat_4, unsqueeze_89);  unsqueeze_89 = None
    mul_34: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
    unsqueeze_92: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1)
    unsqueeze_93: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_35: "f32[4, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
    unsqueeze_94: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1);  primals_35 = None
    unsqueeze_95: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_23: "f32[4, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
    relu_11: "f32[4, 224, 56, 56]" = torch.ops.aten.relu.default(add_23);  add_23 = None
    convolution_11: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_11, primals_36, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_24: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_401, torch.float32)
    convert_element_type_25: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_402, torch.float32)
    add_24: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
    sqrt_12: "f32[128]" = torch.ops.aten.sqrt.default(add_24);  add_24 = None
    reciprocal_12: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
    mul_36: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
    unsqueeze_96: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
    unsqueeze_97: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    unsqueeze_98: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
    unsqueeze_99: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    sub_12: "f32[4, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_97);  unsqueeze_97 = None
    mul_37: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
    unsqueeze_100: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_101: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_38: "f32[4, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
    unsqueeze_102: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_103: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_25: "f32[4, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
    relu_12: "f32[4, 128, 56, 56]" = torch.ops.aten.relu.default(add_25);  add_25 = None
    convolution_12: "f32[4, 32, 56, 56]" = torch.ops.aten.convolution.default(relu_12, primals_39, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:124, code: return torch.cat(features, 1)
    cat_5: "f32[4, 256, 56, 56]" = torch.ops.aten.cat.default([getitem, convolution_2, convolution_4, convolution_6, convolution_8, convolution_10, convolution_12], 1);  getitem = convolution_2 = convolution_4 = convolution_6 = convolution_8 = convolution_10 = convolution_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    convert_element_type_26: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_404, torch.float32)
    convert_element_type_27: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_405, torch.float32)
    add_26: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_27, 1e-05);  convert_element_type_27 = None
    sqrt_13: "f32[256]" = torch.ops.aten.sqrt.default(add_26);  add_26 = None
    reciprocal_13: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
    mul_39: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
    unsqueeze_104: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_26, -1);  convert_element_type_26 = None
    unsqueeze_105: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    unsqueeze_106: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
    unsqueeze_107: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    sub_13: "f32[4, 256, 56, 56]" = torch.ops.aten.sub.Tensor(cat_5, unsqueeze_105);  unsqueeze_105 = None
    mul_40: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
    unsqueeze_108: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1)
    unsqueeze_109: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_41: "f32[4, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
    unsqueeze_110: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1);  primals_41 = None
    unsqueeze_111: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_27: "f32[4, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
    relu_13: "f32[4, 256, 56, 56]" = torch.ops.aten.relu.default(add_27);  add_27 = None
    convolution_13: "f32[4, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_13, primals_42, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    avg_pool2d: "f32[4, 128, 28, 28]" = torch.ops.aten.avg_pool2d.default(convolution_13, [2, 2], [2, 2])
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    clone_1: "f32[4, 128, 28, 28]" = torch.ops.aten.clone.default(avg_pool2d)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_28: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_407, torch.float32)
    convert_element_type_29: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_408, torch.float32)
    add_28: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_29, 1e-05);  convert_element_type_29 = None
    sqrt_14: "f32[128]" = torch.ops.aten.sqrt.default(add_28);  add_28 = None
    reciprocal_14: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
    mul_42: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
    unsqueeze_112: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, -1);  convert_element_type_28 = None
    unsqueeze_113: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    unsqueeze_114: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
    unsqueeze_115: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    sub_14: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(clone_1, unsqueeze_113);  clone_1 = unsqueeze_113 = None
    mul_43: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
    unsqueeze_116: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_117: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_44: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
    unsqueeze_118: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_119: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_29: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
    relu_14: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_29);  add_29 = None
    convolution_14: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_14, primals_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_30: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_410, torch.float32)
    convert_element_type_31: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_411, torch.float32)
    add_30: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
    sqrt_15: "f32[128]" = torch.ops.aten.sqrt.default(add_30);  add_30 = None
    reciprocal_15: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
    mul_45: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
    unsqueeze_120: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
    unsqueeze_121: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    unsqueeze_122: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
    unsqueeze_123: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    sub_15: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_121);  unsqueeze_121 = None
    mul_46: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
    unsqueeze_124: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1)
    unsqueeze_125: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_47: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
    unsqueeze_126: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1);  primals_47 = None
    unsqueeze_127: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_31: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
    relu_15: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_31);  add_31 = None
    convolution_15: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_15, primals_48, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_6: "f32[4, 160, 28, 28]" = torch.ops.aten.cat.default([avg_pool2d, convolution_15], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_32: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_413, torch.float32)
    convert_element_type_33: "f32[160]" = torch.ops.prims.convert_element_type.default(primals_414, torch.float32)
    add_32: "f32[160]" = torch.ops.aten.add.Tensor(convert_element_type_33, 1e-05);  convert_element_type_33 = None
    sqrt_16: "f32[160]" = torch.ops.aten.sqrt.default(add_32);  add_32 = None
    reciprocal_16: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
    mul_48: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
    unsqueeze_128: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, -1);  convert_element_type_32 = None
    unsqueeze_129: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    unsqueeze_130: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
    unsqueeze_131: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    sub_16: "f32[4, 160, 28, 28]" = torch.ops.aten.sub.Tensor(cat_6, unsqueeze_129);  unsqueeze_129 = None
    mul_49: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
    unsqueeze_132: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_133: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_50: "f32[4, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
    unsqueeze_134: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_135: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_33: "f32[4, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
    relu_16: "f32[4, 160, 28, 28]" = torch.ops.aten.relu.default(add_33);  add_33 = None
    convolution_16: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_16, primals_51, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_34: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_416, torch.float32)
    convert_element_type_35: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_417, torch.float32)
    add_34: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_35, 1e-05);  convert_element_type_35 = None
    sqrt_17: "f32[128]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
    reciprocal_17: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
    mul_51: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
    unsqueeze_136: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_34, -1);  convert_element_type_34 = None
    unsqueeze_137: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    unsqueeze_138: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
    unsqueeze_139: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    sub_17: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_137);  unsqueeze_137 = None
    mul_52: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
    unsqueeze_140: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1)
    unsqueeze_141: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_53: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
    unsqueeze_142: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1);  primals_53 = None
    unsqueeze_143: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_35: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
    relu_17: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_35);  add_35 = None
    convolution_17: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_17, primals_54, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_7: "f32[4, 192, 28, 28]" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_36: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_419, torch.float32)
    convert_element_type_37: "f32[192]" = torch.ops.prims.convert_element_type.default(primals_420, torch.float32)
    add_36: "f32[192]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
    sqrt_18: "f32[192]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
    reciprocal_18: "f32[192]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
    mul_54: "f32[192]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
    unsqueeze_144: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
    unsqueeze_145: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    unsqueeze_146: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
    unsqueeze_147: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    sub_18: "f32[4, 192, 28, 28]" = torch.ops.aten.sub.Tensor(cat_7, unsqueeze_145);  unsqueeze_145 = None
    mul_55: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
    unsqueeze_148: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_149: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_56: "f32[4, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
    unsqueeze_150: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_151: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_37: "f32[4, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
    relu_18: "f32[4, 192, 28, 28]" = torch.ops.aten.relu.default(add_37);  add_37 = None
    convolution_18: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_18, primals_57, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_38: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_422, torch.float32)
    convert_element_type_39: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_423, torch.float32)
    add_38: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_39, 1e-05);  convert_element_type_39 = None
    sqrt_19: "f32[128]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
    reciprocal_19: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
    mul_57: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
    unsqueeze_152: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_38, -1);  convert_element_type_38 = None
    unsqueeze_153: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    unsqueeze_154: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
    unsqueeze_155: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    sub_19: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_153);  unsqueeze_153 = None
    mul_58: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
    unsqueeze_156: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1)
    unsqueeze_157: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_59: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
    unsqueeze_158: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1);  primals_59 = None
    unsqueeze_159: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_39: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
    relu_19: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_39);  add_39 = None
    convolution_19: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_19, primals_60, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_8: "f32[4, 224, 28, 28]" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_40: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_425, torch.float32)
    convert_element_type_41: "f32[224]" = torch.ops.prims.convert_element_type.default(primals_426, torch.float32)
    add_40: "f32[224]" = torch.ops.aten.add.Tensor(convert_element_type_41, 1e-05);  convert_element_type_41 = None
    sqrt_20: "f32[224]" = torch.ops.aten.sqrt.default(add_40);  add_40 = None
    reciprocal_20: "f32[224]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
    mul_60: "f32[224]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
    unsqueeze_160: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_40, -1);  convert_element_type_40 = None
    unsqueeze_161: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    unsqueeze_162: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
    unsqueeze_163: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    sub_20: "f32[4, 224, 28, 28]" = torch.ops.aten.sub.Tensor(cat_8, unsqueeze_161);  unsqueeze_161 = None
    mul_61: "f32[4, 224, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
    unsqueeze_164: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_165: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_62: "f32[4, 224, 28, 28]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
    unsqueeze_166: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_167: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_41: "f32[4, 224, 28, 28]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
    relu_20: "f32[4, 224, 28, 28]" = torch.ops.aten.relu.default(add_41);  add_41 = None
    convolution_20: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_20, primals_63, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_42: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_428, torch.float32)
    convert_element_type_43: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_429, torch.float32)
    add_42: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
    sqrt_21: "f32[128]" = torch.ops.aten.sqrt.default(add_42);  add_42 = None
    reciprocal_21: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
    mul_63: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
    unsqueeze_168: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
    unsqueeze_169: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    unsqueeze_170: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
    unsqueeze_171: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    sub_21: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_169);  unsqueeze_169 = None
    mul_64: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
    unsqueeze_172: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1)
    unsqueeze_173: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_65: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
    unsqueeze_174: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1);  primals_65 = None
    unsqueeze_175: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_43: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
    relu_21: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_43);  add_43 = None
    convolution_21: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_21, primals_66, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_9: "f32[4, 256, 28, 28]" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_44: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_431, torch.float32)
    convert_element_type_45: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_432, torch.float32)
    add_44: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_45, 1e-05);  convert_element_type_45 = None
    sqrt_22: "f32[256]" = torch.ops.aten.sqrt.default(add_44);  add_44 = None
    reciprocal_22: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
    mul_66: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
    unsqueeze_176: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_44, -1);  convert_element_type_44 = None
    unsqueeze_177: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    unsqueeze_178: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
    unsqueeze_179: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    sub_22: "f32[4, 256, 28, 28]" = torch.ops.aten.sub.Tensor(cat_9, unsqueeze_177);  unsqueeze_177 = None
    mul_67: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
    unsqueeze_180: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_181: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_68: "f32[4, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
    unsqueeze_182: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_183: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_45: "f32[4, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
    relu_22: "f32[4, 256, 28, 28]" = torch.ops.aten.relu.default(add_45);  add_45 = None
    convolution_22: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_22, primals_69, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_46: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_434, torch.float32)
    convert_element_type_47: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_435, torch.float32)
    add_46: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_47, 1e-05);  convert_element_type_47 = None
    sqrt_23: "f32[128]" = torch.ops.aten.sqrt.default(add_46);  add_46 = None
    reciprocal_23: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
    mul_69: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
    unsqueeze_184: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_46, -1);  convert_element_type_46 = None
    unsqueeze_185: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    unsqueeze_186: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
    unsqueeze_187: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    sub_23: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_185);  unsqueeze_185 = None
    mul_70: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
    unsqueeze_188: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1)
    unsqueeze_189: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_71: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
    unsqueeze_190: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1);  primals_71 = None
    unsqueeze_191: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_47: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
    relu_23: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_47);  add_47 = None
    convolution_23: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_23, primals_72, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_10: "f32[4, 288, 28, 28]" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_48: "f32[288]" = torch.ops.prims.convert_element_type.default(primals_437, torch.float32)
    convert_element_type_49: "f32[288]" = torch.ops.prims.convert_element_type.default(primals_438, torch.float32)
    add_48: "f32[288]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
    sqrt_24: "f32[288]" = torch.ops.aten.sqrt.default(add_48);  add_48 = None
    reciprocal_24: "f32[288]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
    mul_72: "f32[288]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
    unsqueeze_192: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
    unsqueeze_193: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    unsqueeze_194: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
    unsqueeze_195: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    sub_24: "f32[4, 288, 28, 28]" = torch.ops.aten.sub.Tensor(cat_10, unsqueeze_193);  unsqueeze_193 = None
    mul_73: "f32[4, 288, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
    unsqueeze_196: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_197: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_74: "f32[4, 288, 28, 28]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
    unsqueeze_198: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_199: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_49: "f32[4, 288, 28, 28]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
    relu_24: "f32[4, 288, 28, 28]" = torch.ops.aten.relu.default(add_49);  add_49 = None
    convolution_24: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_24, primals_75, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_50: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_440, torch.float32)
    convert_element_type_51: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_441, torch.float32)
    add_50: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_51, 1e-05);  convert_element_type_51 = None
    sqrt_25: "f32[128]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
    reciprocal_25: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
    mul_75: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
    unsqueeze_200: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, -1);  convert_element_type_50 = None
    unsqueeze_201: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    unsqueeze_202: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
    unsqueeze_203: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    sub_25: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_201);  unsqueeze_201 = None
    mul_76: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
    unsqueeze_204: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1)
    unsqueeze_205: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_77: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
    unsqueeze_206: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1);  primals_77 = None
    unsqueeze_207: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_51: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
    relu_25: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_51);  add_51 = None
    convolution_25: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_25, primals_78, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_11: "f32[4, 320, 28, 28]" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_52: "f32[320]" = torch.ops.prims.convert_element_type.default(primals_443, torch.float32)
    convert_element_type_53: "f32[320]" = torch.ops.prims.convert_element_type.default(primals_444, torch.float32)
    add_52: "f32[320]" = torch.ops.aten.add.Tensor(convert_element_type_53, 1e-05);  convert_element_type_53 = None
    sqrt_26: "f32[320]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
    reciprocal_26: "f32[320]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
    mul_78: "f32[320]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
    unsqueeze_208: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, -1);  convert_element_type_52 = None
    unsqueeze_209: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    unsqueeze_210: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
    unsqueeze_211: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    sub_26: "f32[4, 320, 28, 28]" = torch.ops.aten.sub.Tensor(cat_11, unsqueeze_209);  unsqueeze_209 = None
    mul_79: "f32[4, 320, 28, 28]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
    unsqueeze_212: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_213: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_80: "f32[4, 320, 28, 28]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
    unsqueeze_214: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_215: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_53: "f32[4, 320, 28, 28]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
    relu_26: "f32[4, 320, 28, 28]" = torch.ops.aten.relu.default(add_53);  add_53 = None
    convolution_26: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_26, primals_81, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_54: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_446, torch.float32)
    convert_element_type_55: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_447, torch.float32)
    add_54: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
    sqrt_27: "f32[128]" = torch.ops.aten.sqrt.default(add_54);  add_54 = None
    reciprocal_27: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
    mul_81: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
    unsqueeze_216: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
    unsqueeze_217: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    unsqueeze_218: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
    unsqueeze_219: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    sub_27: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_217);  unsqueeze_217 = None
    mul_82: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
    unsqueeze_220: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1)
    unsqueeze_221: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_83: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_221);  mul_82 = unsqueeze_221 = None
    unsqueeze_222: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1);  primals_83 = None
    unsqueeze_223: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_55: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_223);  mul_83 = unsqueeze_223 = None
    relu_27: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_55);  add_55 = None
    convolution_27: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_27, primals_84, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_12: "f32[4, 352, 28, 28]" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_56: "f32[352]" = torch.ops.prims.convert_element_type.default(primals_449, torch.float32)
    convert_element_type_57: "f32[352]" = torch.ops.prims.convert_element_type.default(primals_450, torch.float32)
    add_56: "f32[352]" = torch.ops.aten.add.Tensor(convert_element_type_57, 1e-05);  convert_element_type_57 = None
    sqrt_28: "f32[352]" = torch.ops.aten.sqrt.default(add_56);  add_56 = None
    reciprocal_28: "f32[352]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
    mul_84: "f32[352]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
    unsqueeze_224: "f32[352, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_56, -1);  convert_element_type_56 = None
    unsqueeze_225: "f32[352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    unsqueeze_226: "f32[352, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
    unsqueeze_227: "f32[352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    sub_28: "f32[4, 352, 28, 28]" = torch.ops.aten.sub.Tensor(cat_12, unsqueeze_225);  unsqueeze_225 = None
    mul_85: "f32[4, 352, 28, 28]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
    unsqueeze_228: "f32[352, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_229: "f32[352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_86: "f32[4, 352, 28, 28]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_229);  mul_85 = unsqueeze_229 = None
    unsqueeze_230: "f32[352, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_231: "f32[352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_57: "f32[4, 352, 28, 28]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_231);  mul_86 = unsqueeze_231 = None
    relu_28: "f32[4, 352, 28, 28]" = torch.ops.aten.relu.default(add_57);  add_57 = None
    convolution_28: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_28, primals_87, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_58: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_452, torch.float32)
    convert_element_type_59: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_453, torch.float32)
    add_58: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_59, 1e-05);  convert_element_type_59 = None
    sqrt_29: "f32[128]" = torch.ops.aten.sqrt.default(add_58);  add_58 = None
    reciprocal_29: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
    mul_87: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
    unsqueeze_232: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, -1);  convert_element_type_58 = None
    unsqueeze_233: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    unsqueeze_234: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
    unsqueeze_235: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    sub_29: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_233);  unsqueeze_233 = None
    mul_88: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
    unsqueeze_236: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1)
    unsqueeze_237: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_89: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_237);  mul_88 = unsqueeze_237 = None
    unsqueeze_238: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1);  primals_89 = None
    unsqueeze_239: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_59: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_239);  mul_89 = unsqueeze_239 = None
    relu_29: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_59);  add_59 = None
    convolution_29: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_29, primals_90, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_13: "f32[4, 384, 28, 28]" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_60: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_455, torch.float32)
    convert_element_type_61: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_456, torch.float32)
    add_60: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
    sqrt_30: "f32[384]" = torch.ops.aten.sqrt.default(add_60);  add_60 = None
    reciprocal_30: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
    mul_90: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
    unsqueeze_240: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
    unsqueeze_241: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    unsqueeze_242: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
    unsqueeze_243: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    sub_30: "f32[4, 384, 28, 28]" = torch.ops.aten.sub.Tensor(cat_13, unsqueeze_241);  unsqueeze_241 = None
    mul_91: "f32[4, 384, 28, 28]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
    unsqueeze_244: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_245: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_92: "f32[4, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_245);  mul_91 = unsqueeze_245 = None
    unsqueeze_246: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_247: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_61: "f32[4, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_247);  mul_92 = unsqueeze_247 = None
    relu_30: "f32[4, 384, 28, 28]" = torch.ops.aten.relu.default(add_61);  add_61 = None
    convolution_30: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_30, primals_93, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_62: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_458, torch.float32)
    convert_element_type_63: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_459, torch.float32)
    add_62: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_63, 1e-05);  convert_element_type_63 = None
    sqrt_31: "f32[128]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
    reciprocal_31: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
    mul_93: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
    unsqueeze_248: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, -1);  convert_element_type_62 = None
    unsqueeze_249: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    unsqueeze_250: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
    unsqueeze_251: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    sub_31: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_249);  unsqueeze_249 = None
    mul_94: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
    unsqueeze_252: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1)
    unsqueeze_253: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_95: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_253);  mul_94 = unsqueeze_253 = None
    unsqueeze_254: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1);  primals_95 = None
    unsqueeze_255: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_63: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_255);  mul_95 = unsqueeze_255 = None
    relu_31: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_63);  add_63 = None
    convolution_31: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_31, primals_96, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_14: "f32[4, 416, 28, 28]" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29, convolution_31], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_64: "f32[416]" = torch.ops.prims.convert_element_type.default(primals_461, torch.float32)
    convert_element_type_65: "f32[416]" = torch.ops.prims.convert_element_type.default(primals_462, torch.float32)
    add_64: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_65, 1e-05);  convert_element_type_65 = None
    sqrt_32: "f32[416]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
    reciprocal_32: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
    mul_96: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
    unsqueeze_256: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_64, -1);  convert_element_type_64 = None
    unsqueeze_257: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    unsqueeze_258: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
    unsqueeze_259: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    sub_32: "f32[4, 416, 28, 28]" = torch.ops.aten.sub.Tensor(cat_14, unsqueeze_257);  unsqueeze_257 = None
    mul_97: "f32[4, 416, 28, 28]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
    unsqueeze_260: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_261: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_98: "f32[4, 416, 28, 28]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_261);  mul_97 = unsqueeze_261 = None
    unsqueeze_262: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_263: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_65: "f32[4, 416, 28, 28]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_263);  mul_98 = unsqueeze_263 = None
    relu_32: "f32[4, 416, 28, 28]" = torch.ops.aten.relu.default(add_65);  add_65 = None
    convolution_32: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_32, primals_99, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_66: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_464, torch.float32)
    convert_element_type_67: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_465, torch.float32)
    add_66: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
    sqrt_33: "f32[128]" = torch.ops.aten.sqrt.default(add_66);  add_66 = None
    reciprocal_33: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
    mul_99: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
    unsqueeze_264: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
    unsqueeze_265: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    unsqueeze_266: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
    unsqueeze_267: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    sub_33: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_265);  unsqueeze_265 = None
    mul_100: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
    unsqueeze_268: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1)
    unsqueeze_269: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_101: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_269);  mul_100 = unsqueeze_269 = None
    unsqueeze_270: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1);  primals_101 = None
    unsqueeze_271: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_67: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_271);  mul_101 = unsqueeze_271 = None
    relu_33: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_67);  add_67 = None
    convolution_33: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_33, primals_102, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_15: "f32[4, 448, 28, 28]" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29, convolution_31, convolution_33], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_68: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_467, torch.float32)
    convert_element_type_69: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_468, torch.float32)
    add_68: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_69, 1e-05);  convert_element_type_69 = None
    sqrt_34: "f32[448]" = torch.ops.aten.sqrt.default(add_68);  add_68 = None
    reciprocal_34: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
    mul_102: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
    unsqueeze_272: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_68, -1);  convert_element_type_68 = None
    unsqueeze_273: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    unsqueeze_274: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
    unsqueeze_275: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    sub_34: "f32[4, 448, 28, 28]" = torch.ops.aten.sub.Tensor(cat_15, unsqueeze_273);  unsqueeze_273 = None
    mul_103: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
    unsqueeze_276: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
    unsqueeze_277: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_104: "f32[4, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_277);  mul_103 = unsqueeze_277 = None
    unsqueeze_278: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
    unsqueeze_279: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_69: "f32[4, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_279);  mul_104 = unsqueeze_279 = None
    relu_34: "f32[4, 448, 28, 28]" = torch.ops.aten.relu.default(add_69);  add_69 = None
    convolution_34: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_34, primals_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_70: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_470, torch.float32)
    convert_element_type_71: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_471, torch.float32)
    add_70: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_71, 1e-05);  convert_element_type_71 = None
    sqrt_35: "f32[128]" = torch.ops.aten.sqrt.default(add_70);  add_70 = None
    reciprocal_35: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
    mul_105: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
    unsqueeze_280: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, -1);  convert_element_type_70 = None
    unsqueeze_281: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    unsqueeze_282: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
    unsqueeze_283: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    sub_35: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_281);  unsqueeze_281 = None
    mul_106: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
    unsqueeze_284: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1)
    unsqueeze_285: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_107: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_285);  mul_106 = unsqueeze_285 = None
    unsqueeze_286: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1);  primals_107 = None
    unsqueeze_287: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_71: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_287);  mul_107 = unsqueeze_287 = None
    relu_35: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_71);  add_71 = None
    convolution_35: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_35, primals_108, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_16: "f32[4, 480, 28, 28]" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29, convolution_31, convolution_33, convolution_35], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_72: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_473, torch.float32)
    convert_element_type_73: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_474, torch.float32)
    add_72: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
    sqrt_36: "f32[480]" = torch.ops.aten.sqrt.default(add_72);  add_72 = None
    reciprocal_36: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
    mul_108: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
    unsqueeze_288: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
    unsqueeze_289: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    unsqueeze_290: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
    unsqueeze_291: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    sub_36: "f32[4, 480, 28, 28]" = torch.ops.aten.sub.Tensor(cat_16, unsqueeze_289);  unsqueeze_289 = None
    mul_109: "f32[4, 480, 28, 28]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
    unsqueeze_292: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1)
    unsqueeze_293: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_110: "f32[4, 480, 28, 28]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_293);  mul_109 = unsqueeze_293 = None
    unsqueeze_294: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
    unsqueeze_295: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_73: "f32[4, 480, 28, 28]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_295);  mul_110 = unsqueeze_295 = None
    relu_36: "f32[4, 480, 28, 28]" = torch.ops.aten.relu.default(add_73);  add_73 = None
    convolution_36: "f32[4, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_36, primals_111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_74: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_476, torch.float32)
    convert_element_type_75: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_477, torch.float32)
    add_74: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_75, 1e-05);  convert_element_type_75 = None
    sqrt_37: "f32[128]" = torch.ops.aten.sqrt.default(add_74);  add_74 = None
    reciprocal_37: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
    mul_111: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
    unsqueeze_296: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, -1);  convert_element_type_74 = None
    unsqueeze_297: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    unsqueeze_298: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
    unsqueeze_299: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    sub_37: "f32[4, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_297);  unsqueeze_297 = None
    mul_112: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
    unsqueeze_300: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1)
    unsqueeze_301: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_113: "f32[4, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_301);  mul_112 = unsqueeze_301 = None
    unsqueeze_302: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1);  primals_113 = None
    unsqueeze_303: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_75: "f32[4, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_303);  mul_113 = unsqueeze_303 = None
    relu_37: "f32[4, 128, 28, 28]" = torch.ops.aten.relu.default(add_75);  add_75 = None
    convolution_37: "f32[4, 32, 28, 28]" = torch.ops.aten.convolution.default(relu_37, primals_114, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:124, code: return torch.cat(features, 1)
    cat_17: "f32[4, 512, 28, 28]" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29, convolution_31, convolution_33, convolution_35, convolution_37], 1);  convolution_15 = convolution_17 = convolution_19 = convolution_21 = convolution_23 = convolution_25 = convolution_27 = convolution_29 = convolution_31 = convolution_33 = convolution_35 = convolution_37 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    convert_element_type_76: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_479, torch.float32)
    convert_element_type_77: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_480, torch.float32)
    add_76: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_77, 1e-05);  convert_element_type_77 = None
    sqrt_38: "f32[512]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
    reciprocal_38: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
    mul_114: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
    unsqueeze_304: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_76, -1);  convert_element_type_76 = None
    unsqueeze_305: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    unsqueeze_306: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
    unsqueeze_307: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    sub_38: "f32[4, 512, 28, 28]" = torch.ops.aten.sub.Tensor(cat_17, unsqueeze_305);  unsqueeze_305 = None
    mul_115: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
    unsqueeze_308: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1)
    unsqueeze_309: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_116: "f32[4, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_309);  mul_115 = unsqueeze_309 = None
    unsqueeze_310: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1);  primals_116 = None
    unsqueeze_311: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_77: "f32[4, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_311);  mul_116 = unsqueeze_311 = None
    relu_38: "f32[4, 512, 28, 28]" = torch.ops.aten.relu.default(add_77);  add_77 = None
    convolution_38: "f32[4, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_38, primals_117, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    avg_pool2d_1: "f32[4, 256, 14, 14]" = torch.ops.aten.avg_pool2d.default(convolution_38, [2, 2], [2, 2])
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    clone_2: "f32[4, 256, 14, 14]" = torch.ops.aten.clone.default(avg_pool2d_1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_78: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_482, torch.float32)
    convert_element_type_79: "f32[256]" = torch.ops.prims.convert_element_type.default(primals_483, torch.float32)
    add_78: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
    sqrt_39: "f32[256]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
    reciprocal_39: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
    mul_117: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
    unsqueeze_312: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
    unsqueeze_313: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    unsqueeze_314: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
    unsqueeze_315: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    sub_39: "f32[4, 256, 14, 14]" = torch.ops.aten.sub.Tensor(clone_2, unsqueeze_313);  clone_2 = unsqueeze_313 = None
    mul_118: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
    unsqueeze_316: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1)
    unsqueeze_317: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_119: "f32[4, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_317);  mul_118 = unsqueeze_317 = None
    unsqueeze_318: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1);  primals_119 = None
    unsqueeze_319: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_79: "f32[4, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_319);  mul_119 = unsqueeze_319 = None
    relu_39: "f32[4, 256, 14, 14]" = torch.ops.aten.relu.default(add_79);  add_79 = None
    convolution_39: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_39, primals_120, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_80: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_485, torch.float32)
    convert_element_type_81: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_486, torch.float32)
    add_80: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_81, 1e-05);  convert_element_type_81 = None
    sqrt_40: "f32[128]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
    reciprocal_40: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
    mul_120: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
    unsqueeze_320: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_80, -1);  convert_element_type_80 = None
    unsqueeze_321: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    unsqueeze_322: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
    unsqueeze_323: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    sub_40: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_321);  unsqueeze_321 = None
    mul_121: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
    unsqueeze_324: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1)
    unsqueeze_325: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_122: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_325);  mul_121 = unsqueeze_325 = None
    unsqueeze_326: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1);  primals_122 = None
    unsqueeze_327: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_81: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_327);  mul_122 = unsqueeze_327 = None
    relu_40: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_81);  add_81 = None
    convolution_40: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_40, primals_123, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_18: "f32[4, 288, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_82: "f32[288]" = torch.ops.prims.convert_element_type.default(primals_488, torch.float32)
    convert_element_type_83: "f32[288]" = torch.ops.prims.convert_element_type.default(primals_489, torch.float32)
    add_82: "f32[288]" = torch.ops.aten.add.Tensor(convert_element_type_83, 1e-05);  convert_element_type_83 = None
    sqrt_41: "f32[288]" = torch.ops.aten.sqrt.default(add_82);  add_82 = None
    reciprocal_41: "f32[288]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
    mul_123: "f32[288]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
    unsqueeze_328: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, -1);  convert_element_type_82 = None
    unsqueeze_329: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    unsqueeze_330: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
    unsqueeze_331: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    sub_41: "f32[4, 288, 14, 14]" = torch.ops.aten.sub.Tensor(cat_18, unsqueeze_329);  unsqueeze_329 = None
    mul_124: "f32[4, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
    unsqueeze_332: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1)
    unsqueeze_333: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_125: "f32[4, 288, 14, 14]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
    unsqueeze_334: "f32[288, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1);  primals_125 = None
    unsqueeze_335: "f32[288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_83: "f32[4, 288, 14, 14]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
    relu_41: "f32[4, 288, 14, 14]" = torch.ops.aten.relu.default(add_83);  add_83 = None
    convolution_41: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_41, primals_126, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_84: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_491, torch.float32)
    convert_element_type_85: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_492, torch.float32)
    add_84: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
    sqrt_42: "f32[128]" = torch.ops.aten.sqrt.default(add_84);  add_84 = None
    reciprocal_42: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
    mul_126: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
    unsqueeze_336: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
    unsqueeze_337: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    unsqueeze_338: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
    unsqueeze_339: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    sub_42: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_337);  unsqueeze_337 = None
    mul_127: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
    unsqueeze_340: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1)
    unsqueeze_341: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_128: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
    unsqueeze_342: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1);  primals_128 = None
    unsqueeze_343: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_85: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
    relu_42: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_85);  add_85 = None
    convolution_42: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_42, primals_129, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_19: "f32[4, 320, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_86: "f32[320]" = torch.ops.prims.convert_element_type.default(primals_494, torch.float32)
    convert_element_type_87: "f32[320]" = torch.ops.prims.convert_element_type.default(primals_495, torch.float32)
    add_86: "f32[320]" = torch.ops.aten.add.Tensor(convert_element_type_87, 1e-05);  convert_element_type_87 = None
    sqrt_43: "f32[320]" = torch.ops.aten.sqrt.default(add_86);  add_86 = None
    reciprocal_43: "f32[320]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
    mul_129: "f32[320]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
    unsqueeze_344: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_86, -1);  convert_element_type_86 = None
    unsqueeze_345: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    unsqueeze_346: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
    unsqueeze_347: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    sub_43: "f32[4, 320, 14, 14]" = torch.ops.aten.sub.Tensor(cat_19, unsqueeze_345);  unsqueeze_345 = None
    mul_130: "f32[4, 320, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
    unsqueeze_348: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1)
    unsqueeze_349: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
    mul_131: "f32[4, 320, 14, 14]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
    unsqueeze_350: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1);  primals_131 = None
    unsqueeze_351: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
    add_87: "f32[4, 320, 14, 14]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
    relu_43: "f32[4, 320, 14, 14]" = torch.ops.aten.relu.default(add_87);  add_87 = None
    convolution_43: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_43, primals_132, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_88: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_497, torch.float32)
    convert_element_type_89: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_498, torch.float32)
    add_88: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_89, 1e-05);  convert_element_type_89 = None
    sqrt_44: "f32[128]" = torch.ops.aten.sqrt.default(add_88);  add_88 = None
    reciprocal_44: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
    mul_132: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
    unsqueeze_352: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_88, -1);  convert_element_type_88 = None
    unsqueeze_353: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
    unsqueeze_354: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
    unsqueeze_355: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
    sub_44: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_353);  unsqueeze_353 = None
    mul_133: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
    unsqueeze_356: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_133, -1)
    unsqueeze_357: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
    mul_134: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
    unsqueeze_358: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1);  primals_134 = None
    unsqueeze_359: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
    add_89: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
    relu_44: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_89);  add_89 = None
    convolution_44: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_44, primals_135, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_20: "f32[4, 352, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_90: "f32[352]" = torch.ops.prims.convert_element_type.default(primals_500, torch.float32)
    convert_element_type_91: "f32[352]" = torch.ops.prims.convert_element_type.default(primals_501, torch.float32)
    add_90: "f32[352]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
    sqrt_45: "f32[352]" = torch.ops.aten.sqrt.default(add_90);  add_90 = None
    reciprocal_45: "f32[352]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
    mul_135: "f32[352]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
    unsqueeze_360: "f32[352, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
    unsqueeze_361: "f32[352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
    unsqueeze_362: "f32[352, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
    unsqueeze_363: "f32[352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
    sub_45: "f32[4, 352, 14, 14]" = torch.ops.aten.sub.Tensor(cat_20, unsqueeze_361);  unsqueeze_361 = None
    mul_136: "f32[4, 352, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
    unsqueeze_364: "f32[352, 1]" = torch.ops.aten.unsqueeze.default(primals_136, -1)
    unsqueeze_365: "f32[352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
    mul_137: "f32[4, 352, 14, 14]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
    unsqueeze_366: "f32[352, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1);  primals_137 = None
    unsqueeze_367: "f32[352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
    add_91: "f32[4, 352, 14, 14]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
    relu_45: "f32[4, 352, 14, 14]" = torch.ops.aten.relu.default(add_91);  add_91 = None
    convolution_45: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_45, primals_138, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_92: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_503, torch.float32)
    convert_element_type_93: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_504, torch.float32)
    add_92: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_93, 1e-05);  convert_element_type_93 = None
    sqrt_46: "f32[128]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
    reciprocal_46: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
    mul_138: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
    unsqueeze_368: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, -1);  convert_element_type_92 = None
    unsqueeze_369: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
    unsqueeze_370: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
    unsqueeze_371: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
    sub_46: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_369);  unsqueeze_369 = None
    mul_139: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
    unsqueeze_372: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_139, -1)
    unsqueeze_373: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
    mul_140: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
    unsqueeze_374: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1);  primals_140 = None
    unsqueeze_375: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
    add_93: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
    relu_46: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_93);  add_93 = None
    convolution_46: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_46, primals_141, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_21: "f32[4, 384, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_94: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_506, torch.float32)
    convert_element_type_95: "f32[384]" = torch.ops.prims.convert_element_type.default(primals_507, torch.float32)
    add_94: "f32[384]" = torch.ops.aten.add.Tensor(convert_element_type_95, 1e-05);  convert_element_type_95 = None
    sqrt_47: "f32[384]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
    reciprocal_47: "f32[384]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
    mul_141: "f32[384]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
    unsqueeze_376: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, -1);  convert_element_type_94 = None
    unsqueeze_377: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
    unsqueeze_378: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
    unsqueeze_379: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
    sub_47: "f32[4, 384, 14, 14]" = torch.ops.aten.sub.Tensor(cat_21, unsqueeze_377);  unsqueeze_377 = None
    mul_142: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
    unsqueeze_380: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_142, -1)
    unsqueeze_381: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
    mul_143: "f32[4, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
    unsqueeze_382: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1);  primals_143 = None
    unsqueeze_383: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
    add_95: "f32[4, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
    relu_47: "f32[4, 384, 14, 14]" = torch.ops.aten.relu.default(add_95);  add_95 = None
    convolution_47: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_47, primals_144, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_96: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_509, torch.float32)
    convert_element_type_97: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_510, torch.float32)
    add_96: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
    sqrt_48: "f32[128]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
    reciprocal_48: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
    mul_144: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
    unsqueeze_384: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
    unsqueeze_385: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
    unsqueeze_386: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
    unsqueeze_387: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
    sub_48: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_385);  unsqueeze_385 = None
    mul_145: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
    unsqueeze_388: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_145, -1)
    unsqueeze_389: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
    mul_146: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
    unsqueeze_390: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1);  primals_146 = None
    unsqueeze_391: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
    add_97: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
    relu_48: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_97);  add_97 = None
    convolution_48: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_48, primals_147, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_22: "f32[4, 416, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_98: "f32[416]" = torch.ops.prims.convert_element_type.default(primals_512, torch.float32)
    convert_element_type_99: "f32[416]" = torch.ops.prims.convert_element_type.default(primals_513, torch.float32)
    add_98: "f32[416]" = torch.ops.aten.add.Tensor(convert_element_type_99, 1e-05);  convert_element_type_99 = None
    sqrt_49: "f32[416]" = torch.ops.aten.sqrt.default(add_98);  add_98 = None
    reciprocal_49: "f32[416]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
    mul_147: "f32[416]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
    unsqueeze_392: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_98, -1);  convert_element_type_98 = None
    unsqueeze_393: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
    unsqueeze_394: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
    unsqueeze_395: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
    sub_49: "f32[4, 416, 14, 14]" = torch.ops.aten.sub.Tensor(cat_22, unsqueeze_393);  unsqueeze_393 = None
    mul_148: "f32[4, 416, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
    unsqueeze_396: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_148, -1)
    unsqueeze_397: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
    mul_149: "f32[4, 416, 14, 14]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
    unsqueeze_398: "f32[416, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1);  primals_149 = None
    unsqueeze_399: "f32[416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
    add_99: "f32[4, 416, 14, 14]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
    relu_49: "f32[4, 416, 14, 14]" = torch.ops.aten.relu.default(add_99);  add_99 = None
    convolution_49: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_49, primals_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_100: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_515, torch.float32)
    convert_element_type_101: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_516, torch.float32)
    add_100: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_101, 1e-05);  convert_element_type_101 = None
    sqrt_50: "f32[128]" = torch.ops.aten.sqrt.default(add_100);  add_100 = None
    reciprocal_50: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
    mul_150: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
    unsqueeze_400: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_100, -1);  convert_element_type_100 = None
    unsqueeze_401: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
    unsqueeze_402: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
    unsqueeze_403: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
    sub_50: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_401);  unsqueeze_401 = None
    mul_151: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
    unsqueeze_404: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_151, -1)
    unsqueeze_405: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
    mul_152: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
    unsqueeze_406: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1);  primals_152 = None
    unsqueeze_407: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
    add_101: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
    relu_50: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_101);  add_101 = None
    convolution_50: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_50, primals_153, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_23: "f32[4, 448, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_102: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_518, torch.float32)
    convert_element_type_103: "f32[448]" = torch.ops.prims.convert_element_type.default(primals_519, torch.float32)
    add_102: "f32[448]" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
    sqrt_51: "f32[448]" = torch.ops.aten.sqrt.default(add_102);  add_102 = None
    reciprocal_51: "f32[448]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
    mul_153: "f32[448]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
    unsqueeze_408: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
    unsqueeze_409: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
    unsqueeze_410: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
    unsqueeze_411: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
    sub_51: "f32[4, 448, 14, 14]" = torch.ops.aten.sub.Tensor(cat_23, unsqueeze_409);  unsqueeze_409 = None
    mul_154: "f32[4, 448, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
    unsqueeze_412: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_154, -1)
    unsqueeze_413: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
    mul_155: "f32[4, 448, 14, 14]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
    unsqueeze_414: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1);  primals_155 = None
    unsqueeze_415: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
    add_103: "f32[4, 448, 14, 14]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
    relu_51: "f32[4, 448, 14, 14]" = torch.ops.aten.relu.default(add_103);  add_103 = None
    convolution_51: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_51, primals_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_104: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_521, torch.float32)
    convert_element_type_105: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_522, torch.float32)
    add_104: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_105, 1e-05);  convert_element_type_105 = None
    sqrt_52: "f32[128]" = torch.ops.aten.sqrt.default(add_104);  add_104 = None
    reciprocal_52: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
    mul_156: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
    unsqueeze_416: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_104, -1);  convert_element_type_104 = None
    unsqueeze_417: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
    unsqueeze_418: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
    unsqueeze_419: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
    sub_52: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_417);  unsqueeze_417 = None
    mul_157: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
    unsqueeze_420: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_157, -1)
    unsqueeze_421: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
    mul_158: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_421);  mul_157 = unsqueeze_421 = None
    unsqueeze_422: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_158, -1);  primals_158 = None
    unsqueeze_423: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
    add_105: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_423);  mul_158 = unsqueeze_423 = None
    relu_52: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_105);  add_105 = None
    convolution_52: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_52, primals_159, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_24: "f32[4, 480, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_106: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_524, torch.float32)
    convert_element_type_107: "f32[480]" = torch.ops.prims.convert_element_type.default(primals_525, torch.float32)
    add_106: "f32[480]" = torch.ops.aten.add.Tensor(convert_element_type_107, 1e-05);  convert_element_type_107 = None
    sqrt_53: "f32[480]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
    reciprocal_53: "f32[480]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
    mul_159: "f32[480]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
    unsqueeze_424: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_106, -1);  convert_element_type_106 = None
    unsqueeze_425: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
    unsqueeze_426: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
    unsqueeze_427: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
    sub_53: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(cat_24, unsqueeze_425);  unsqueeze_425 = None
    mul_160: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
    unsqueeze_428: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_160, -1)
    unsqueeze_429: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
    mul_161: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_429);  mul_160 = unsqueeze_429 = None
    unsqueeze_430: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_161, -1);  primals_161 = None
    unsqueeze_431: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
    add_107: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_431);  mul_161 = unsqueeze_431 = None
    relu_53: "f32[4, 480, 14, 14]" = torch.ops.aten.relu.default(add_107);  add_107 = None
    convolution_53: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_53, primals_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_108: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_527, torch.float32)
    convert_element_type_109: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_528, torch.float32)
    add_108: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_109, 1e-05);  convert_element_type_109 = None
    sqrt_54: "f32[128]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
    reciprocal_54: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
    mul_162: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
    unsqueeze_432: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_108, -1);  convert_element_type_108 = None
    unsqueeze_433: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
    unsqueeze_434: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
    unsqueeze_435: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
    sub_54: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_433);  unsqueeze_433 = None
    mul_163: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
    unsqueeze_436: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1)
    unsqueeze_437: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
    mul_164: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_437);  mul_163 = unsqueeze_437 = None
    unsqueeze_438: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_164, -1);  primals_164 = None
    unsqueeze_439: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
    add_109: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_439);  mul_164 = unsqueeze_439 = None
    relu_54: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_109);  add_109 = None
    convolution_54: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_54, primals_165, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_25: "f32[4, 512, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_110: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_530, torch.float32)
    convert_element_type_111: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_531, torch.float32)
    add_110: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_111, 1e-05);  convert_element_type_111 = None
    sqrt_55: "f32[512]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
    reciprocal_55: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
    mul_165: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
    unsqueeze_440: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_110, -1);  convert_element_type_110 = None
    unsqueeze_441: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
    unsqueeze_442: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
    unsqueeze_443: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
    sub_55: "f32[4, 512, 14, 14]" = torch.ops.aten.sub.Tensor(cat_25, unsqueeze_441);  unsqueeze_441 = None
    mul_166: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
    unsqueeze_444: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_166, -1)
    unsqueeze_445: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
    mul_167: "f32[4, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_445);  mul_166 = unsqueeze_445 = None
    unsqueeze_446: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1);  primals_167 = None
    unsqueeze_447: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
    add_111: "f32[4, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_447);  mul_167 = unsqueeze_447 = None
    relu_55: "f32[4, 512, 14, 14]" = torch.ops.aten.relu.default(add_111);  add_111 = None
    convolution_55: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_55, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_112: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_533, torch.float32)
    convert_element_type_113: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_534, torch.float32)
    add_112: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_113, 1e-05);  convert_element_type_113 = None
    sqrt_56: "f32[128]" = torch.ops.aten.sqrt.default(add_112);  add_112 = None
    reciprocal_56: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
    mul_168: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
    unsqueeze_448: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_112, -1);  convert_element_type_112 = None
    unsqueeze_449: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
    unsqueeze_450: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
    unsqueeze_451: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
    sub_56: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_449);  unsqueeze_449 = None
    mul_169: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
    unsqueeze_452: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_169, -1)
    unsqueeze_453: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
    mul_170: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_453);  mul_169 = unsqueeze_453 = None
    unsqueeze_454: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_170, -1);  primals_170 = None
    unsqueeze_455: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
    add_113: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_455);  mul_170 = unsqueeze_455 = None
    relu_56: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_113);  add_113 = None
    convolution_56: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_56, primals_171, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_26: "f32[4, 544, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_114: "f32[544]" = torch.ops.prims.convert_element_type.default(primals_536, torch.float32)
    convert_element_type_115: "f32[544]" = torch.ops.prims.convert_element_type.default(primals_537, torch.float32)
    add_114: "f32[544]" = torch.ops.aten.add.Tensor(convert_element_type_115, 1e-05);  convert_element_type_115 = None
    sqrt_57: "f32[544]" = torch.ops.aten.sqrt.default(add_114);  add_114 = None
    reciprocal_57: "f32[544]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
    mul_171: "f32[544]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
    unsqueeze_456: "f32[544, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_114, -1);  convert_element_type_114 = None
    unsqueeze_457: "f32[544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
    unsqueeze_458: "f32[544, 1]" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
    unsqueeze_459: "f32[544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
    sub_57: "f32[4, 544, 14, 14]" = torch.ops.aten.sub.Tensor(cat_26, unsqueeze_457);  unsqueeze_457 = None
    mul_172: "f32[4, 544, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
    unsqueeze_460: "f32[544, 1]" = torch.ops.aten.unsqueeze.default(primals_172, -1)
    unsqueeze_461: "f32[544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
    mul_173: "f32[4, 544, 14, 14]" = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_461);  mul_172 = unsqueeze_461 = None
    unsqueeze_462: "f32[544, 1]" = torch.ops.aten.unsqueeze.default(primals_173, -1);  primals_173 = None
    unsqueeze_463: "f32[544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
    add_115: "f32[4, 544, 14, 14]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_463);  mul_173 = unsqueeze_463 = None
    relu_57: "f32[4, 544, 14, 14]" = torch.ops.aten.relu.default(add_115);  add_115 = None
    convolution_57: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_57, primals_174, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_116: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_539, torch.float32)
    convert_element_type_117: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_540, torch.float32)
    add_116: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_117, 1e-05);  convert_element_type_117 = None
    sqrt_58: "f32[128]" = torch.ops.aten.sqrt.default(add_116);  add_116 = None
    reciprocal_58: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
    mul_174: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
    unsqueeze_464: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_116, -1);  convert_element_type_116 = None
    unsqueeze_465: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
    unsqueeze_466: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
    unsqueeze_467: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
    sub_58: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_465);  unsqueeze_465 = None
    mul_175: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
    unsqueeze_468: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_175, -1)
    unsqueeze_469: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
    mul_176: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_469);  mul_175 = unsqueeze_469 = None
    unsqueeze_470: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_176, -1);  primals_176 = None
    unsqueeze_471: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
    add_117: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_176, unsqueeze_471);  mul_176 = unsqueeze_471 = None
    relu_58: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_117);  add_117 = None
    convolution_58: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_58, primals_177, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_27: "f32[4, 576, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_118: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_542, torch.float32)
    convert_element_type_119: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_543, torch.float32)
    add_118: "f32[576]" = torch.ops.aten.add.Tensor(convert_element_type_119, 1e-05);  convert_element_type_119 = None
    sqrt_59: "f32[576]" = torch.ops.aten.sqrt.default(add_118);  add_118 = None
    reciprocal_59: "f32[576]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
    mul_177: "f32[576]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
    unsqueeze_472: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_118, -1);  convert_element_type_118 = None
    unsqueeze_473: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
    unsqueeze_474: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(mul_177, -1);  mul_177 = None
    unsqueeze_475: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
    sub_59: "f32[4, 576, 14, 14]" = torch.ops.aten.sub.Tensor(cat_27, unsqueeze_473);  unsqueeze_473 = None
    mul_178: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
    unsqueeze_476: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_178, -1)
    unsqueeze_477: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
    mul_179: "f32[4, 576, 14, 14]" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_477);  mul_178 = unsqueeze_477 = None
    unsqueeze_478: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_179, -1);  primals_179 = None
    unsqueeze_479: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
    add_119: "f32[4, 576, 14, 14]" = torch.ops.aten.add.Tensor(mul_179, unsqueeze_479);  mul_179 = unsqueeze_479 = None
    relu_59: "f32[4, 576, 14, 14]" = torch.ops.aten.relu.default(add_119);  add_119 = None
    convolution_59: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_59, primals_180, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_120: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_545, torch.float32)
    convert_element_type_121: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_546, torch.float32)
    add_120: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_121, 1e-05);  convert_element_type_121 = None
    sqrt_60: "f32[128]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
    reciprocal_60: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
    mul_180: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
    unsqueeze_480: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_120, -1);  convert_element_type_120 = None
    unsqueeze_481: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
    unsqueeze_482: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
    unsqueeze_483: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
    sub_60: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_481);  unsqueeze_481 = None
    mul_181: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
    unsqueeze_484: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_181, -1)
    unsqueeze_485: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
    mul_182: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_485);  mul_181 = unsqueeze_485 = None
    unsqueeze_486: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_182, -1);  primals_182 = None
    unsqueeze_487: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
    add_121: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_487);  mul_182 = unsqueeze_487 = None
    relu_60: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_121);  add_121 = None
    convolution_60: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_60, primals_183, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_28: "f32[4, 608, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_122: "f32[608]" = torch.ops.prims.convert_element_type.default(primals_548, torch.float32)
    convert_element_type_123: "f32[608]" = torch.ops.prims.convert_element_type.default(primals_549, torch.float32)
    add_122: "f32[608]" = torch.ops.aten.add.Tensor(convert_element_type_123, 1e-05);  convert_element_type_123 = None
    sqrt_61: "f32[608]" = torch.ops.aten.sqrt.default(add_122);  add_122 = None
    reciprocal_61: "f32[608]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
    mul_183: "f32[608]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
    unsqueeze_488: "f32[608, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_122, -1);  convert_element_type_122 = None
    unsqueeze_489: "f32[608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
    unsqueeze_490: "f32[608, 1]" = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
    unsqueeze_491: "f32[608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
    sub_61: "f32[4, 608, 14, 14]" = torch.ops.aten.sub.Tensor(cat_28, unsqueeze_489);  unsqueeze_489 = None
    mul_184: "f32[4, 608, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
    unsqueeze_492: "f32[608, 1]" = torch.ops.aten.unsqueeze.default(primals_184, -1)
    unsqueeze_493: "f32[608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
    mul_185: "f32[4, 608, 14, 14]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_493);  mul_184 = unsqueeze_493 = None
    unsqueeze_494: "f32[608, 1]" = torch.ops.aten.unsqueeze.default(primals_185, -1);  primals_185 = None
    unsqueeze_495: "f32[608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
    add_123: "f32[4, 608, 14, 14]" = torch.ops.aten.add.Tensor(mul_185, unsqueeze_495);  mul_185 = unsqueeze_495 = None
    relu_61: "f32[4, 608, 14, 14]" = torch.ops.aten.relu.default(add_123);  add_123 = None
    convolution_61: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_61, primals_186, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_124: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_551, torch.float32)
    convert_element_type_125: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_552, torch.float32)
    add_124: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_125, 1e-05);  convert_element_type_125 = None
    sqrt_62: "f32[128]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
    reciprocal_62: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
    mul_186: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
    unsqueeze_496: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_124, -1);  convert_element_type_124 = None
    unsqueeze_497: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
    unsqueeze_498: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
    unsqueeze_499: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
    sub_62: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_497);  unsqueeze_497 = None
    mul_187: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
    unsqueeze_500: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_187, -1)
    unsqueeze_501: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
    mul_188: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_501);  mul_187 = unsqueeze_501 = None
    unsqueeze_502: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_188, -1);  primals_188 = None
    unsqueeze_503: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
    add_125: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_503);  mul_188 = unsqueeze_503 = None
    relu_62: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_125);  add_125 = None
    convolution_62: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_62, primals_189, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_29: "f32[4, 640, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_126: "f32[640]" = torch.ops.prims.convert_element_type.default(primals_554, torch.float32)
    convert_element_type_127: "f32[640]" = torch.ops.prims.convert_element_type.default(primals_555, torch.float32)
    add_126: "f32[640]" = torch.ops.aten.add.Tensor(convert_element_type_127, 1e-05);  convert_element_type_127 = None
    sqrt_63: "f32[640]" = torch.ops.aten.sqrt.default(add_126);  add_126 = None
    reciprocal_63: "f32[640]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
    mul_189: "f32[640]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
    unsqueeze_504: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_126, -1);  convert_element_type_126 = None
    unsqueeze_505: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
    unsqueeze_506: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
    unsqueeze_507: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
    sub_63: "f32[4, 640, 14, 14]" = torch.ops.aten.sub.Tensor(cat_29, unsqueeze_505);  unsqueeze_505 = None
    mul_190: "f32[4, 640, 14, 14]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
    unsqueeze_508: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_190, -1)
    unsqueeze_509: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
    mul_191: "f32[4, 640, 14, 14]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_509);  mul_190 = unsqueeze_509 = None
    unsqueeze_510: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_191, -1);  primals_191 = None
    unsqueeze_511: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
    add_127: "f32[4, 640, 14, 14]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_511);  mul_191 = unsqueeze_511 = None
    relu_63: "f32[4, 640, 14, 14]" = torch.ops.aten.relu.default(add_127);  add_127 = None
    convolution_63: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_63, primals_192, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_128: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_557, torch.float32)
    convert_element_type_129: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_558, torch.float32)
    add_128: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_129, 1e-05);  convert_element_type_129 = None
    sqrt_64: "f32[128]" = torch.ops.aten.sqrt.default(add_128);  add_128 = None
    reciprocal_64: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
    mul_192: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
    unsqueeze_512: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_128, -1);  convert_element_type_128 = None
    unsqueeze_513: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
    unsqueeze_514: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
    unsqueeze_515: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
    sub_64: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_513);  unsqueeze_513 = None
    mul_193: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
    unsqueeze_516: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_193, -1)
    unsqueeze_517: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
    mul_194: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_517);  mul_193 = unsqueeze_517 = None
    unsqueeze_518: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_194, -1);  primals_194 = None
    unsqueeze_519: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
    add_129: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_519);  mul_194 = unsqueeze_519 = None
    relu_64: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_129);  add_129 = None
    convolution_64: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_64, primals_195, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_30: "f32[4, 672, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_130: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_560, torch.float32)
    convert_element_type_131: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_561, torch.float32)
    add_130: "f32[672]" = torch.ops.aten.add.Tensor(convert_element_type_131, 1e-05);  convert_element_type_131 = None
    sqrt_65: "f32[672]" = torch.ops.aten.sqrt.default(add_130);  add_130 = None
    reciprocal_65: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
    mul_195: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
    unsqueeze_520: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_130, -1);  convert_element_type_130 = None
    unsqueeze_521: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
    unsqueeze_522: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
    unsqueeze_523: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
    sub_65: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(cat_30, unsqueeze_521);  unsqueeze_521 = None
    mul_196: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
    unsqueeze_524: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_196, -1)
    unsqueeze_525: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
    mul_197: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_525);  mul_196 = unsqueeze_525 = None
    unsqueeze_526: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_197, -1);  primals_197 = None
    unsqueeze_527: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
    add_131: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_527);  mul_197 = unsqueeze_527 = None
    relu_65: "f32[4, 672, 14, 14]" = torch.ops.aten.relu.default(add_131);  add_131 = None
    convolution_65: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_65, primals_198, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_132: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_563, torch.float32)
    convert_element_type_133: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_564, torch.float32)
    add_132: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_133, 1e-05);  convert_element_type_133 = None
    sqrt_66: "f32[128]" = torch.ops.aten.sqrt.default(add_132);  add_132 = None
    reciprocal_66: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
    mul_198: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
    unsqueeze_528: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_132, -1);  convert_element_type_132 = None
    unsqueeze_529: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
    unsqueeze_530: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
    unsqueeze_531: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
    sub_66: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_529);  unsqueeze_529 = None
    mul_199: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
    unsqueeze_532: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_199, -1)
    unsqueeze_533: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
    mul_200: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_533);  mul_199 = unsqueeze_533 = None
    unsqueeze_534: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_200, -1);  primals_200 = None
    unsqueeze_535: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
    add_133: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_200, unsqueeze_535);  mul_200 = unsqueeze_535 = None
    relu_66: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_133);  add_133 = None
    convolution_66: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_66, primals_201, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_31: "f32[4, 704, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_134: "f32[704]" = torch.ops.prims.convert_element_type.default(primals_566, torch.float32)
    convert_element_type_135: "f32[704]" = torch.ops.prims.convert_element_type.default(primals_567, torch.float32)
    add_134: "f32[704]" = torch.ops.aten.add.Tensor(convert_element_type_135, 1e-05);  convert_element_type_135 = None
    sqrt_67: "f32[704]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
    reciprocal_67: "f32[704]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
    mul_201: "f32[704]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
    unsqueeze_536: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_134, -1);  convert_element_type_134 = None
    unsqueeze_537: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
    unsqueeze_538: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(mul_201, -1);  mul_201 = None
    unsqueeze_539: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
    sub_67: "f32[4, 704, 14, 14]" = torch.ops.aten.sub.Tensor(cat_31, unsqueeze_537);  unsqueeze_537 = None
    mul_202: "f32[4, 704, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
    unsqueeze_540: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(primals_202, -1)
    unsqueeze_541: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
    mul_203: "f32[4, 704, 14, 14]" = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_541);  mul_202 = unsqueeze_541 = None
    unsqueeze_542: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1);  primals_203 = None
    unsqueeze_543: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
    add_135: "f32[4, 704, 14, 14]" = torch.ops.aten.add.Tensor(mul_203, unsqueeze_543);  mul_203 = unsqueeze_543 = None
    relu_67: "f32[4, 704, 14, 14]" = torch.ops.aten.relu.default(add_135);  add_135 = None
    convolution_67: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_67, primals_204, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_136: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_569, torch.float32)
    convert_element_type_137: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_570, torch.float32)
    add_136: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_137, 1e-05);  convert_element_type_137 = None
    sqrt_68: "f32[128]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
    reciprocal_68: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
    mul_204: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
    unsqueeze_544: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_136, -1);  convert_element_type_136 = None
    unsqueeze_545: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
    unsqueeze_546: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_204, -1);  mul_204 = None
    unsqueeze_547: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
    sub_68: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_545);  unsqueeze_545 = None
    mul_205: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
    unsqueeze_548: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_205, -1)
    unsqueeze_549: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
    mul_206: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_549);  mul_205 = unsqueeze_549 = None
    unsqueeze_550: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_206, -1);  primals_206 = None
    unsqueeze_551: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
    add_137: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_551);  mul_206 = unsqueeze_551 = None
    relu_68: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_137);  add_137 = None
    convolution_68: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_68, primals_207, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_32: "f32[4, 736, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_138: "f32[736]" = torch.ops.prims.convert_element_type.default(primals_572, torch.float32)
    convert_element_type_139: "f32[736]" = torch.ops.prims.convert_element_type.default(primals_573, torch.float32)
    add_138: "f32[736]" = torch.ops.aten.add.Tensor(convert_element_type_139, 1e-05);  convert_element_type_139 = None
    sqrt_69: "f32[736]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
    reciprocal_69: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
    mul_207: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
    unsqueeze_552: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_138, -1);  convert_element_type_138 = None
    unsqueeze_553: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
    unsqueeze_554: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
    unsqueeze_555: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
    sub_69: "f32[4, 736, 14, 14]" = torch.ops.aten.sub.Tensor(cat_32, unsqueeze_553);  unsqueeze_553 = None
    mul_208: "f32[4, 736, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
    unsqueeze_556: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_208, -1)
    unsqueeze_557: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
    mul_209: "f32[4, 736, 14, 14]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_557);  mul_208 = unsqueeze_557 = None
    unsqueeze_558: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_209, -1);  primals_209 = None
    unsqueeze_559: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
    add_139: "f32[4, 736, 14, 14]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_559);  mul_209 = unsqueeze_559 = None
    relu_69: "f32[4, 736, 14, 14]" = torch.ops.aten.relu.default(add_139);  add_139 = None
    convolution_69: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_69, primals_210, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_140: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_575, torch.float32)
    convert_element_type_141: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_576, torch.float32)
    add_140: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_141, 1e-05);  convert_element_type_141 = None
    sqrt_70: "f32[128]" = torch.ops.aten.sqrt.default(add_140);  add_140 = None
    reciprocal_70: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
    mul_210: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
    unsqueeze_560: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_140, -1);  convert_element_type_140 = None
    unsqueeze_561: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
    unsqueeze_562: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_210, -1);  mul_210 = None
    unsqueeze_563: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
    sub_70: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_561);  unsqueeze_561 = None
    mul_211: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
    unsqueeze_564: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_211, -1)
    unsqueeze_565: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
    mul_212: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_565);  mul_211 = unsqueeze_565 = None
    unsqueeze_566: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_212, -1);  primals_212 = None
    unsqueeze_567: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
    add_141: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_567);  mul_212 = unsqueeze_567 = None
    relu_70: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_141);  add_141 = None
    convolution_70: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_70, primals_213, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_33: "f32[4, 768, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_142: "f32[768]" = torch.ops.prims.convert_element_type.default(primals_578, torch.float32)
    convert_element_type_143: "f32[768]" = torch.ops.prims.convert_element_type.default(primals_579, torch.float32)
    add_142: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_143, 1e-05);  convert_element_type_143 = None
    sqrt_71: "f32[768]" = torch.ops.aten.sqrt.default(add_142);  add_142 = None
    reciprocal_71: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
    mul_213: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
    unsqueeze_568: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_142, -1);  convert_element_type_142 = None
    unsqueeze_569: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
    unsqueeze_570: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
    unsqueeze_571: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
    sub_71: "f32[4, 768, 14, 14]" = torch.ops.aten.sub.Tensor(cat_33, unsqueeze_569);  unsqueeze_569 = None
    mul_214: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
    unsqueeze_572: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_214, -1)
    unsqueeze_573: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
    mul_215: "f32[4, 768, 14, 14]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_573);  mul_214 = unsqueeze_573 = None
    unsqueeze_574: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_215, -1);  primals_215 = None
    unsqueeze_575: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
    add_143: "f32[4, 768, 14, 14]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_575);  mul_215 = unsqueeze_575 = None
    relu_71: "f32[4, 768, 14, 14]" = torch.ops.aten.relu.default(add_143);  add_143 = None
    convolution_71: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_71, primals_216, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_144: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_581, torch.float32)
    convert_element_type_145: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_582, torch.float32)
    add_144: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_145, 1e-05);  convert_element_type_145 = None
    sqrt_72: "f32[128]" = torch.ops.aten.sqrt.default(add_144);  add_144 = None
    reciprocal_72: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
    mul_216: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
    unsqueeze_576: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_144, -1);  convert_element_type_144 = None
    unsqueeze_577: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
    unsqueeze_578: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
    unsqueeze_579: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
    sub_72: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_577);  unsqueeze_577 = None
    mul_217: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
    unsqueeze_580: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_217, -1)
    unsqueeze_581: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
    mul_218: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_581);  mul_217 = unsqueeze_581 = None
    unsqueeze_582: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_218, -1);  primals_218 = None
    unsqueeze_583: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
    add_145: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_583);  mul_218 = unsqueeze_583 = None
    relu_72: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_145);  add_145 = None
    convolution_72: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_72, primals_219, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_34: "f32[4, 800, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_146: "f32[800]" = torch.ops.prims.convert_element_type.default(primals_584, torch.float32)
    convert_element_type_147: "f32[800]" = torch.ops.prims.convert_element_type.default(primals_585, torch.float32)
    add_146: "f32[800]" = torch.ops.aten.add.Tensor(convert_element_type_147, 1e-05);  convert_element_type_147 = None
    sqrt_73: "f32[800]" = torch.ops.aten.sqrt.default(add_146);  add_146 = None
    reciprocal_73: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
    mul_219: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
    unsqueeze_584: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_146, -1);  convert_element_type_146 = None
    unsqueeze_585: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
    unsqueeze_586: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
    unsqueeze_587: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
    sub_73: "f32[4, 800, 14, 14]" = torch.ops.aten.sub.Tensor(cat_34, unsqueeze_585);  unsqueeze_585 = None
    mul_220: "f32[4, 800, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
    unsqueeze_588: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(primals_220, -1)
    unsqueeze_589: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
    mul_221: "f32[4, 800, 14, 14]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_589);  mul_220 = unsqueeze_589 = None
    unsqueeze_590: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(primals_221, -1);  primals_221 = None
    unsqueeze_591: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
    add_147: "f32[4, 800, 14, 14]" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_591);  mul_221 = unsqueeze_591 = None
    relu_73: "f32[4, 800, 14, 14]" = torch.ops.aten.relu.default(add_147);  add_147 = None
    convolution_73: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_73, primals_222, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_148: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_587, torch.float32)
    convert_element_type_149: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_588, torch.float32)
    add_148: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_149, 1e-05);  convert_element_type_149 = None
    sqrt_74: "f32[128]" = torch.ops.aten.sqrt.default(add_148);  add_148 = None
    reciprocal_74: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
    mul_222: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
    unsqueeze_592: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_148, -1);  convert_element_type_148 = None
    unsqueeze_593: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
    unsqueeze_594: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
    unsqueeze_595: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
    sub_74: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_593);  unsqueeze_593 = None
    mul_223: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
    unsqueeze_596: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_223, -1)
    unsqueeze_597: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
    mul_224: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_597);  mul_223 = unsqueeze_597 = None
    unsqueeze_598: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_224, -1);  primals_224 = None
    unsqueeze_599: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
    add_149: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_599);  mul_224 = unsqueeze_599 = None
    relu_74: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_149);  add_149 = None
    convolution_74: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_74, primals_225, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_35: "f32[4, 832, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_150: "f32[832]" = torch.ops.prims.convert_element_type.default(primals_590, torch.float32)
    convert_element_type_151: "f32[832]" = torch.ops.prims.convert_element_type.default(primals_591, torch.float32)
    add_150: "f32[832]" = torch.ops.aten.add.Tensor(convert_element_type_151, 1e-05);  convert_element_type_151 = None
    sqrt_75: "f32[832]" = torch.ops.aten.sqrt.default(add_150);  add_150 = None
    reciprocal_75: "f32[832]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
    mul_225: "f32[832]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
    unsqueeze_600: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_150, -1);  convert_element_type_150 = None
    unsqueeze_601: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
    unsqueeze_602: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
    unsqueeze_603: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
    sub_75: "f32[4, 832, 14, 14]" = torch.ops.aten.sub.Tensor(cat_35, unsqueeze_601);  unsqueeze_601 = None
    mul_226: "f32[4, 832, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
    unsqueeze_604: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(primals_226, -1)
    unsqueeze_605: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
    mul_227: "f32[4, 832, 14, 14]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_605);  mul_226 = unsqueeze_605 = None
    unsqueeze_606: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(primals_227, -1);  primals_227 = None
    unsqueeze_607: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
    add_151: "f32[4, 832, 14, 14]" = torch.ops.aten.add.Tensor(mul_227, unsqueeze_607);  mul_227 = unsqueeze_607 = None
    relu_75: "f32[4, 832, 14, 14]" = torch.ops.aten.relu.default(add_151);  add_151 = None
    convolution_75: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_75, primals_228, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_152: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_593, torch.float32)
    convert_element_type_153: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_594, torch.float32)
    add_152: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_153, 1e-05);  convert_element_type_153 = None
    sqrt_76: "f32[128]" = torch.ops.aten.sqrt.default(add_152);  add_152 = None
    reciprocal_76: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
    mul_228: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
    unsqueeze_608: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_152, -1);  convert_element_type_152 = None
    unsqueeze_609: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
    unsqueeze_610: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
    unsqueeze_611: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
    sub_76: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_609);  unsqueeze_609 = None
    mul_229: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
    unsqueeze_612: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_229, -1)
    unsqueeze_613: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
    mul_230: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_613);  mul_229 = unsqueeze_613 = None
    unsqueeze_614: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_230, -1);  primals_230 = None
    unsqueeze_615: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
    add_153: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_615);  mul_230 = unsqueeze_615 = None
    relu_76: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_153);  add_153 = None
    convolution_76: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_76, primals_231, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_36: "f32[4, 864, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_154: "f32[864]" = torch.ops.prims.convert_element_type.default(primals_596, torch.float32)
    convert_element_type_155: "f32[864]" = torch.ops.prims.convert_element_type.default(primals_597, torch.float32)
    add_154: "f32[864]" = torch.ops.aten.add.Tensor(convert_element_type_155, 1e-05);  convert_element_type_155 = None
    sqrt_77: "f32[864]" = torch.ops.aten.sqrt.default(add_154);  add_154 = None
    reciprocal_77: "f32[864]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
    mul_231: "f32[864]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
    unsqueeze_616: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_154, -1);  convert_element_type_154 = None
    unsqueeze_617: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
    unsqueeze_618: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
    unsqueeze_619: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
    sub_77: "f32[4, 864, 14, 14]" = torch.ops.aten.sub.Tensor(cat_36, unsqueeze_617);  unsqueeze_617 = None
    mul_232: "f32[4, 864, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
    unsqueeze_620: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_232, -1)
    unsqueeze_621: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
    mul_233: "f32[4, 864, 14, 14]" = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_621);  mul_232 = unsqueeze_621 = None
    unsqueeze_622: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_233, -1);  primals_233 = None
    unsqueeze_623: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
    add_155: "f32[4, 864, 14, 14]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_623);  mul_233 = unsqueeze_623 = None
    relu_77: "f32[4, 864, 14, 14]" = torch.ops.aten.relu.default(add_155);  add_155 = None
    convolution_77: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_77, primals_234, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_156: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_599, torch.float32)
    convert_element_type_157: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_600, torch.float32)
    add_156: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_157, 1e-05);  convert_element_type_157 = None
    sqrt_78: "f32[128]" = torch.ops.aten.sqrt.default(add_156);  add_156 = None
    reciprocal_78: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
    mul_234: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
    unsqueeze_624: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_156, -1);  convert_element_type_156 = None
    unsqueeze_625: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
    unsqueeze_626: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
    unsqueeze_627: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
    sub_78: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_625);  unsqueeze_625 = None
    mul_235: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
    unsqueeze_628: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_235, -1)
    unsqueeze_629: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
    mul_236: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_629);  mul_235 = unsqueeze_629 = None
    unsqueeze_630: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_236, -1);  primals_236 = None
    unsqueeze_631: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
    add_157: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_631);  mul_236 = unsqueeze_631 = None
    relu_78: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_157);  add_157 = None
    convolution_78: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_78, primals_237, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_37: "f32[4, 896, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_158: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_602, torch.float32)
    convert_element_type_159: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_603, torch.float32)
    add_158: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_159, 1e-05);  convert_element_type_159 = None
    sqrt_79: "f32[896]" = torch.ops.aten.sqrt.default(add_158);  add_158 = None
    reciprocal_79: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
    mul_237: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
    unsqueeze_632: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_158, -1);  convert_element_type_158 = None
    unsqueeze_633: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
    unsqueeze_634: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
    unsqueeze_635: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
    sub_79: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(cat_37, unsqueeze_633);  unsqueeze_633 = None
    mul_238: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
    unsqueeze_636: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_238, -1)
    unsqueeze_637: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
    mul_239: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_637);  mul_238 = unsqueeze_637 = None
    unsqueeze_638: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_239, -1);  primals_239 = None
    unsqueeze_639: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
    add_159: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_639);  mul_239 = unsqueeze_639 = None
    relu_79: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_159);  add_159 = None
    convolution_79: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_79, primals_240, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_160: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_605, torch.float32)
    convert_element_type_161: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_606, torch.float32)
    add_160: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_161, 1e-05);  convert_element_type_161 = None
    sqrt_80: "f32[128]" = torch.ops.aten.sqrt.default(add_160);  add_160 = None
    reciprocal_80: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
    mul_240: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
    unsqueeze_640: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_160, -1);  convert_element_type_160 = None
    unsqueeze_641: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
    unsqueeze_642: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
    unsqueeze_643: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
    sub_80: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_641);  unsqueeze_641 = None
    mul_241: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
    unsqueeze_644: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_241, -1)
    unsqueeze_645: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
    mul_242: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_645);  mul_241 = unsqueeze_645 = None
    unsqueeze_646: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_242, -1);  primals_242 = None
    unsqueeze_647: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
    add_161: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_647);  mul_242 = unsqueeze_647 = None
    relu_80: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_161);  add_161 = None
    convolution_80: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_80, primals_243, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_38: "f32[4, 928, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78, convolution_80], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_162: "f32[928]" = torch.ops.prims.convert_element_type.default(primals_608, torch.float32)
    convert_element_type_163: "f32[928]" = torch.ops.prims.convert_element_type.default(primals_609, torch.float32)
    add_162: "f32[928]" = torch.ops.aten.add.Tensor(convert_element_type_163, 1e-05);  convert_element_type_163 = None
    sqrt_81: "f32[928]" = torch.ops.aten.sqrt.default(add_162);  add_162 = None
    reciprocal_81: "f32[928]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
    mul_243: "f32[928]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
    unsqueeze_648: "f32[928, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_162, -1);  convert_element_type_162 = None
    unsqueeze_649: "f32[928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
    unsqueeze_650: "f32[928, 1]" = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
    unsqueeze_651: "f32[928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
    sub_81: "f32[4, 928, 14, 14]" = torch.ops.aten.sub.Tensor(cat_38, unsqueeze_649);  unsqueeze_649 = None
    mul_244: "f32[4, 928, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
    unsqueeze_652: "f32[928, 1]" = torch.ops.aten.unsqueeze.default(primals_244, -1)
    unsqueeze_653: "f32[928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
    mul_245: "f32[4, 928, 14, 14]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_653);  mul_244 = unsqueeze_653 = None
    unsqueeze_654: "f32[928, 1]" = torch.ops.aten.unsqueeze.default(primals_245, -1);  primals_245 = None
    unsqueeze_655: "f32[928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
    add_163: "f32[4, 928, 14, 14]" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_655);  mul_245 = unsqueeze_655 = None
    relu_81: "f32[4, 928, 14, 14]" = torch.ops.aten.relu.default(add_163);  add_163 = None
    convolution_81: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_81, primals_246, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_164: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_611, torch.float32)
    convert_element_type_165: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_612, torch.float32)
    add_164: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_165, 1e-05);  convert_element_type_165 = None
    sqrt_82: "f32[128]" = torch.ops.aten.sqrt.default(add_164);  add_164 = None
    reciprocal_82: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
    mul_246: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
    unsqueeze_656: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_164, -1);  convert_element_type_164 = None
    unsqueeze_657: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
    unsqueeze_658: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_246, -1);  mul_246 = None
    unsqueeze_659: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
    sub_82: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_657);  unsqueeze_657 = None
    mul_247: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
    unsqueeze_660: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_247, -1)
    unsqueeze_661: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
    mul_248: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_247, unsqueeze_661);  mul_247 = unsqueeze_661 = None
    unsqueeze_662: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_248, -1);  primals_248 = None
    unsqueeze_663: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
    add_165: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_248, unsqueeze_663);  mul_248 = unsqueeze_663 = None
    relu_82: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_165);  add_165 = None
    convolution_82: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_82, primals_249, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_39: "f32[4, 960, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78, convolution_80, convolution_82], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_166: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_614, torch.float32)
    convert_element_type_167: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_615, torch.float32)
    add_166: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_167, 1e-05);  convert_element_type_167 = None
    sqrt_83: "f32[960]" = torch.ops.aten.sqrt.default(add_166);  add_166 = None
    reciprocal_83: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
    mul_249: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
    unsqueeze_664: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_166, -1);  convert_element_type_166 = None
    unsqueeze_665: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
    unsqueeze_666: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_249, -1);  mul_249 = None
    unsqueeze_667: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
    sub_83: "f32[4, 960, 14, 14]" = torch.ops.aten.sub.Tensor(cat_39, unsqueeze_665);  unsqueeze_665 = None
    mul_250: "f32[4, 960, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
    unsqueeze_668: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_250, -1)
    unsqueeze_669: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
    mul_251: "f32[4, 960, 14, 14]" = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_669);  mul_250 = unsqueeze_669 = None
    unsqueeze_670: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_251, -1);  primals_251 = None
    unsqueeze_671: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
    add_167: "f32[4, 960, 14, 14]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_671);  mul_251 = unsqueeze_671 = None
    relu_83: "f32[4, 960, 14, 14]" = torch.ops.aten.relu.default(add_167);  add_167 = None
    convolution_83: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_83, primals_252, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_168: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_617, torch.float32)
    convert_element_type_169: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_618, torch.float32)
    add_168: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_169, 1e-05);  convert_element_type_169 = None
    sqrt_84: "f32[128]" = torch.ops.aten.sqrt.default(add_168);  add_168 = None
    reciprocal_84: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
    mul_252: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
    unsqueeze_672: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_168, -1);  convert_element_type_168 = None
    unsqueeze_673: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
    unsqueeze_674: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_252, -1);  mul_252 = None
    unsqueeze_675: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
    sub_84: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_673);  unsqueeze_673 = None
    mul_253: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
    unsqueeze_676: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_253, -1)
    unsqueeze_677: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
    mul_254: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_253, unsqueeze_677);  mul_253 = unsqueeze_677 = None
    unsqueeze_678: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_254, -1);  primals_254 = None
    unsqueeze_679: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
    add_169: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_679);  mul_254 = unsqueeze_679 = None
    relu_84: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_169);  add_169 = None
    convolution_84: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_84, primals_255, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_40: "f32[4, 992, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78, convolution_80, convolution_82, convolution_84], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_170: "f32[992]" = torch.ops.prims.convert_element_type.default(primals_620, torch.float32)
    convert_element_type_171: "f32[992]" = torch.ops.prims.convert_element_type.default(primals_621, torch.float32)
    add_170: "f32[992]" = torch.ops.aten.add.Tensor(convert_element_type_171, 1e-05);  convert_element_type_171 = None
    sqrt_85: "f32[992]" = torch.ops.aten.sqrt.default(add_170);  add_170 = None
    reciprocal_85: "f32[992]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
    mul_255: "f32[992]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
    unsqueeze_680: "f32[992, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_170, -1);  convert_element_type_170 = None
    unsqueeze_681: "f32[992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
    unsqueeze_682: "f32[992, 1]" = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
    unsqueeze_683: "f32[992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
    sub_85: "f32[4, 992, 14, 14]" = torch.ops.aten.sub.Tensor(cat_40, unsqueeze_681);  unsqueeze_681 = None
    mul_256: "f32[4, 992, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
    unsqueeze_684: "f32[992, 1]" = torch.ops.aten.unsqueeze.default(primals_256, -1)
    unsqueeze_685: "f32[992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
    mul_257: "f32[4, 992, 14, 14]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_685);  mul_256 = unsqueeze_685 = None
    unsqueeze_686: "f32[992, 1]" = torch.ops.aten.unsqueeze.default(primals_257, -1);  primals_257 = None
    unsqueeze_687: "f32[992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
    add_171: "f32[4, 992, 14, 14]" = torch.ops.aten.add.Tensor(mul_257, unsqueeze_687);  mul_257 = unsqueeze_687 = None
    relu_85: "f32[4, 992, 14, 14]" = torch.ops.aten.relu.default(add_171);  add_171 = None
    convolution_85: "f32[4, 128, 14, 14]" = torch.ops.aten.convolution.default(relu_85, primals_258, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_172: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_623, torch.float32)
    convert_element_type_173: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_624, torch.float32)
    add_172: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_173, 1e-05);  convert_element_type_173 = None
    sqrt_86: "f32[128]" = torch.ops.aten.sqrt.default(add_172);  add_172 = None
    reciprocal_86: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
    mul_258: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
    unsqueeze_688: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_172, -1);  convert_element_type_172 = None
    unsqueeze_689: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
    unsqueeze_690: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_258, -1);  mul_258 = None
    unsqueeze_691: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
    sub_86: "f32[4, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_689);  unsqueeze_689 = None
    mul_259: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
    unsqueeze_692: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_259, -1)
    unsqueeze_693: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
    mul_260: "f32[4, 128, 14, 14]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_693);  mul_259 = unsqueeze_693 = None
    unsqueeze_694: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_260, -1);  primals_260 = None
    unsqueeze_695: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
    add_173: "f32[4, 128, 14, 14]" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_695);  mul_260 = unsqueeze_695 = None
    relu_86: "f32[4, 128, 14, 14]" = torch.ops.aten.relu.default(add_173);  add_173 = None
    convolution_86: "f32[4, 32, 14, 14]" = torch.ops.aten.convolution.default(relu_86, primals_261, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:124, code: return torch.cat(features, 1)
    cat_41: "f32[4, 1024, 14, 14]" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78, convolution_80, convolution_82, convolution_84, convolution_86], 1);  convolution_40 = convolution_42 = convolution_44 = convolution_46 = convolution_48 = convolution_50 = convolution_52 = convolution_54 = convolution_56 = convolution_58 = convolution_60 = convolution_62 = convolution_64 = convolution_66 = convolution_68 = convolution_70 = convolution_72 = convolution_74 = convolution_76 = convolution_78 = convolution_80 = convolution_82 = convolution_84 = convolution_86 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    convert_element_type_174: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_626, torch.float32)
    convert_element_type_175: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_627, torch.float32)
    add_174: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_175, 1e-05);  convert_element_type_175 = None
    sqrt_87: "f32[1024]" = torch.ops.aten.sqrt.default(add_174);  add_174 = None
    reciprocal_87: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
    mul_261: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
    unsqueeze_696: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_174, -1);  convert_element_type_174 = None
    unsqueeze_697: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
    unsqueeze_698: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
    unsqueeze_699: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
    sub_87: "f32[4, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(cat_41, unsqueeze_697);  unsqueeze_697 = None
    mul_262: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
    unsqueeze_700: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_262, -1)
    unsqueeze_701: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
    mul_263: "f32[4, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_701);  mul_262 = unsqueeze_701 = None
    unsqueeze_702: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_263, -1);  primals_263 = None
    unsqueeze_703: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
    add_175: "f32[4, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_263, unsqueeze_703);  mul_263 = unsqueeze_703 = None
    relu_87: "f32[4, 1024, 14, 14]" = torch.ops.aten.relu.default(add_175);  add_175 = None
    convolution_87: "f32[4, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_87, primals_264, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    avg_pool2d_2: "f32[4, 512, 7, 7]" = torch.ops.aten.avg_pool2d.default(convolution_87, [2, 2], [2, 2])
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    clone_3: "f32[4, 512, 7, 7]" = torch.ops.aten.clone.default(avg_pool2d_2)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_176: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_629, torch.float32)
    convert_element_type_177: "f32[512]" = torch.ops.prims.convert_element_type.default(primals_630, torch.float32)
    add_176: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_177, 1e-05);  convert_element_type_177 = None
    sqrt_88: "f32[512]" = torch.ops.aten.sqrt.default(add_176);  add_176 = None
    reciprocal_88: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
    mul_264: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
    unsqueeze_704: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_176, -1);  convert_element_type_176 = None
    unsqueeze_705: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
    unsqueeze_706: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
    unsqueeze_707: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
    sub_88: "f32[4, 512, 7, 7]" = torch.ops.aten.sub.Tensor(clone_3, unsqueeze_705);  clone_3 = unsqueeze_705 = None
    mul_265: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
    unsqueeze_708: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_265, -1)
    unsqueeze_709: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
    mul_266: "f32[4, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_709);  mul_265 = unsqueeze_709 = None
    unsqueeze_710: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_266, -1);  primals_266 = None
    unsqueeze_711: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
    add_177: "f32[4, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_266, unsqueeze_711);  mul_266 = unsqueeze_711 = None
    relu_88: "f32[4, 512, 7, 7]" = torch.ops.aten.relu.default(add_177);  add_177 = None
    convolution_88: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_88, primals_267, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_178: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_632, torch.float32)
    convert_element_type_179: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_633, torch.float32)
    add_178: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_179, 1e-05);  convert_element_type_179 = None
    sqrt_89: "f32[128]" = torch.ops.aten.sqrt.default(add_178);  add_178 = None
    reciprocal_89: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
    mul_267: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
    unsqueeze_712: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_178, -1);  convert_element_type_178 = None
    unsqueeze_713: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
    unsqueeze_714: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
    unsqueeze_715: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
    sub_89: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_713);  unsqueeze_713 = None
    mul_268: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
    unsqueeze_716: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_268, -1)
    unsqueeze_717: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
    mul_269: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_717);  mul_268 = unsqueeze_717 = None
    unsqueeze_718: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_269, -1);  primals_269 = None
    unsqueeze_719: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
    add_179: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_719);  mul_269 = unsqueeze_719 = None
    relu_89: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_179);  add_179 = None
    convolution_89: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_89, primals_270, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_42: "f32[4, 544, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_180: "f32[544]" = torch.ops.prims.convert_element_type.default(primals_635, torch.float32)
    convert_element_type_181: "f32[544]" = torch.ops.prims.convert_element_type.default(primals_636, torch.float32)
    add_180: "f32[544]" = torch.ops.aten.add.Tensor(convert_element_type_181, 1e-05);  convert_element_type_181 = None
    sqrt_90: "f32[544]" = torch.ops.aten.sqrt.default(add_180);  add_180 = None
    reciprocal_90: "f32[544]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
    mul_270: "f32[544]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
    unsqueeze_720: "f32[544, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_180, -1);  convert_element_type_180 = None
    unsqueeze_721: "f32[544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
    unsqueeze_722: "f32[544, 1]" = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
    unsqueeze_723: "f32[544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
    sub_90: "f32[4, 544, 7, 7]" = torch.ops.aten.sub.Tensor(cat_42, unsqueeze_721);  unsqueeze_721 = None
    mul_271: "f32[4, 544, 7, 7]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
    unsqueeze_724: "f32[544, 1]" = torch.ops.aten.unsqueeze.default(primals_271, -1)
    unsqueeze_725: "f32[544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
    mul_272: "f32[4, 544, 7, 7]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_725);  mul_271 = unsqueeze_725 = None
    unsqueeze_726: "f32[544, 1]" = torch.ops.aten.unsqueeze.default(primals_272, -1);  primals_272 = None
    unsqueeze_727: "f32[544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
    add_181: "f32[4, 544, 7, 7]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_727);  mul_272 = unsqueeze_727 = None
    relu_90: "f32[4, 544, 7, 7]" = torch.ops.aten.relu.default(add_181);  add_181 = None
    convolution_90: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_90, primals_273, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_182: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_638, torch.float32)
    convert_element_type_183: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_639, torch.float32)
    add_182: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_183, 1e-05);  convert_element_type_183 = None
    sqrt_91: "f32[128]" = torch.ops.aten.sqrt.default(add_182);  add_182 = None
    reciprocal_91: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
    mul_273: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
    unsqueeze_728: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_182, -1);  convert_element_type_182 = None
    unsqueeze_729: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
    unsqueeze_730: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
    unsqueeze_731: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
    sub_91: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_729);  unsqueeze_729 = None
    mul_274: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
    unsqueeze_732: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_274, -1)
    unsqueeze_733: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
    mul_275: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_733);  mul_274 = unsqueeze_733 = None
    unsqueeze_734: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_275, -1);  primals_275 = None
    unsqueeze_735: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
    add_183: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_735);  mul_275 = unsqueeze_735 = None
    relu_91: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_183);  add_183 = None
    convolution_91: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_91, primals_276, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_43: "f32[4, 576, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_184: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_641, torch.float32)
    convert_element_type_185: "f32[576]" = torch.ops.prims.convert_element_type.default(primals_642, torch.float32)
    add_184: "f32[576]" = torch.ops.aten.add.Tensor(convert_element_type_185, 1e-05);  convert_element_type_185 = None
    sqrt_92: "f32[576]" = torch.ops.aten.sqrt.default(add_184);  add_184 = None
    reciprocal_92: "f32[576]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
    mul_276: "f32[576]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
    unsqueeze_736: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_184, -1);  convert_element_type_184 = None
    unsqueeze_737: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
    unsqueeze_738: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(mul_276, -1);  mul_276 = None
    unsqueeze_739: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
    sub_92: "f32[4, 576, 7, 7]" = torch.ops.aten.sub.Tensor(cat_43, unsqueeze_737);  unsqueeze_737 = None
    mul_277: "f32[4, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
    unsqueeze_740: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_277, -1)
    unsqueeze_741: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
    mul_278: "f32[4, 576, 7, 7]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_741);  mul_277 = unsqueeze_741 = None
    unsqueeze_742: "f32[576, 1]" = torch.ops.aten.unsqueeze.default(primals_278, -1);  primals_278 = None
    unsqueeze_743: "f32[576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
    add_185: "f32[4, 576, 7, 7]" = torch.ops.aten.add.Tensor(mul_278, unsqueeze_743);  mul_278 = unsqueeze_743 = None
    relu_92: "f32[4, 576, 7, 7]" = torch.ops.aten.relu.default(add_185);  add_185 = None
    convolution_92: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_92, primals_279, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_186: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_644, torch.float32)
    convert_element_type_187: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_645, torch.float32)
    add_186: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_187, 1e-05);  convert_element_type_187 = None
    sqrt_93: "f32[128]" = torch.ops.aten.sqrt.default(add_186);  add_186 = None
    reciprocal_93: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
    mul_279: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
    unsqueeze_744: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_186, -1);  convert_element_type_186 = None
    unsqueeze_745: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
    unsqueeze_746: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_279, -1);  mul_279 = None
    unsqueeze_747: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
    sub_93: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_745);  unsqueeze_745 = None
    mul_280: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
    unsqueeze_748: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_280, -1)
    unsqueeze_749: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
    mul_281: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_749);  mul_280 = unsqueeze_749 = None
    unsqueeze_750: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_281, -1);  primals_281 = None
    unsqueeze_751: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
    add_187: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_281, unsqueeze_751);  mul_281 = unsqueeze_751 = None
    relu_93: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_187);  add_187 = None
    convolution_93: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_93, primals_282, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_44: "f32[4, 608, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_188: "f32[608]" = torch.ops.prims.convert_element_type.default(primals_647, torch.float32)
    convert_element_type_189: "f32[608]" = torch.ops.prims.convert_element_type.default(primals_648, torch.float32)
    add_188: "f32[608]" = torch.ops.aten.add.Tensor(convert_element_type_189, 1e-05);  convert_element_type_189 = None
    sqrt_94: "f32[608]" = torch.ops.aten.sqrt.default(add_188);  add_188 = None
    reciprocal_94: "f32[608]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
    mul_282: "f32[608]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
    unsqueeze_752: "f32[608, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_188, -1);  convert_element_type_188 = None
    unsqueeze_753: "f32[608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
    unsqueeze_754: "f32[608, 1]" = torch.ops.aten.unsqueeze.default(mul_282, -1);  mul_282 = None
    unsqueeze_755: "f32[608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
    sub_94: "f32[4, 608, 7, 7]" = torch.ops.aten.sub.Tensor(cat_44, unsqueeze_753);  unsqueeze_753 = None
    mul_283: "f32[4, 608, 7, 7]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
    unsqueeze_756: "f32[608, 1]" = torch.ops.aten.unsqueeze.default(primals_283, -1)
    unsqueeze_757: "f32[608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
    mul_284: "f32[4, 608, 7, 7]" = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_757);  mul_283 = unsqueeze_757 = None
    unsqueeze_758: "f32[608, 1]" = torch.ops.aten.unsqueeze.default(primals_284, -1);  primals_284 = None
    unsqueeze_759: "f32[608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
    add_189: "f32[4, 608, 7, 7]" = torch.ops.aten.add.Tensor(mul_284, unsqueeze_759);  mul_284 = unsqueeze_759 = None
    relu_94: "f32[4, 608, 7, 7]" = torch.ops.aten.relu.default(add_189);  add_189 = None
    convolution_94: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_94, primals_285, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_190: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_650, torch.float32)
    convert_element_type_191: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_651, torch.float32)
    add_190: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_191, 1e-05);  convert_element_type_191 = None
    sqrt_95: "f32[128]" = torch.ops.aten.sqrt.default(add_190);  add_190 = None
    reciprocal_95: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
    mul_285: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
    unsqueeze_760: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_190, -1);  convert_element_type_190 = None
    unsqueeze_761: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
    unsqueeze_762: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_285, -1);  mul_285 = None
    unsqueeze_763: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
    sub_95: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_761);  unsqueeze_761 = None
    mul_286: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
    unsqueeze_764: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_286, -1)
    unsqueeze_765: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
    mul_287: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_765);  mul_286 = unsqueeze_765 = None
    unsqueeze_766: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_287, -1);  primals_287 = None
    unsqueeze_767: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
    add_191: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_287, unsqueeze_767);  mul_287 = unsqueeze_767 = None
    relu_95: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_191);  add_191 = None
    convolution_95: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_95, primals_288, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_45: "f32[4, 640, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_192: "f32[640]" = torch.ops.prims.convert_element_type.default(primals_653, torch.float32)
    convert_element_type_193: "f32[640]" = torch.ops.prims.convert_element_type.default(primals_654, torch.float32)
    add_192: "f32[640]" = torch.ops.aten.add.Tensor(convert_element_type_193, 1e-05);  convert_element_type_193 = None
    sqrt_96: "f32[640]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
    reciprocal_96: "f32[640]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
    mul_288: "f32[640]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
    unsqueeze_768: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_192, -1);  convert_element_type_192 = None
    unsqueeze_769: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
    unsqueeze_770: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
    unsqueeze_771: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
    sub_96: "f32[4, 640, 7, 7]" = torch.ops.aten.sub.Tensor(cat_45, unsqueeze_769);  unsqueeze_769 = None
    mul_289: "f32[4, 640, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
    unsqueeze_772: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_289, -1)
    unsqueeze_773: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
    mul_290: "f32[4, 640, 7, 7]" = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_773);  mul_289 = unsqueeze_773 = None
    unsqueeze_774: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(primals_290, -1);  primals_290 = None
    unsqueeze_775: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
    add_193: "f32[4, 640, 7, 7]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_775);  mul_290 = unsqueeze_775 = None
    relu_96: "f32[4, 640, 7, 7]" = torch.ops.aten.relu.default(add_193);  add_193 = None
    convolution_96: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_96, primals_291, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_194: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_656, torch.float32)
    convert_element_type_195: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_657, torch.float32)
    add_194: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_195, 1e-05);  convert_element_type_195 = None
    sqrt_97: "f32[128]" = torch.ops.aten.sqrt.default(add_194);  add_194 = None
    reciprocal_97: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
    mul_291: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
    unsqueeze_776: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_194, -1);  convert_element_type_194 = None
    unsqueeze_777: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
    unsqueeze_778: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_291, -1);  mul_291 = None
    unsqueeze_779: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
    sub_97: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_777);  unsqueeze_777 = None
    mul_292: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
    unsqueeze_780: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_292, -1)
    unsqueeze_781: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
    mul_293: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_781);  mul_292 = unsqueeze_781 = None
    unsqueeze_782: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_293, -1);  primals_293 = None
    unsqueeze_783: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
    add_195: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_783);  mul_293 = unsqueeze_783 = None
    relu_97: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_195);  add_195 = None
    convolution_97: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_97, primals_294, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_46: "f32[4, 672, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_196: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_659, torch.float32)
    convert_element_type_197: "f32[672]" = torch.ops.prims.convert_element_type.default(primals_660, torch.float32)
    add_196: "f32[672]" = torch.ops.aten.add.Tensor(convert_element_type_197, 1e-05);  convert_element_type_197 = None
    sqrt_98: "f32[672]" = torch.ops.aten.sqrt.default(add_196);  add_196 = None
    reciprocal_98: "f32[672]" = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
    mul_294: "f32[672]" = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
    unsqueeze_784: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_196, -1);  convert_element_type_196 = None
    unsqueeze_785: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
    unsqueeze_786: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(mul_294, -1);  mul_294 = None
    unsqueeze_787: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
    sub_98: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(cat_46, unsqueeze_785);  unsqueeze_785 = None
    mul_295: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
    unsqueeze_788: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_295, -1)
    unsqueeze_789: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
    mul_296: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_295, unsqueeze_789);  mul_295 = unsqueeze_789 = None
    unsqueeze_790: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_296, -1);  primals_296 = None
    unsqueeze_791: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
    add_197: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_296, unsqueeze_791);  mul_296 = unsqueeze_791 = None
    relu_98: "f32[4, 672, 7, 7]" = torch.ops.aten.relu.default(add_197);  add_197 = None
    convolution_98: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_98, primals_297, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_198: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_662, torch.float32)
    convert_element_type_199: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_663, torch.float32)
    add_198: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_199, 1e-05);  convert_element_type_199 = None
    sqrt_99: "f32[128]" = torch.ops.aten.sqrt.default(add_198);  add_198 = None
    reciprocal_99: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
    mul_297: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
    unsqueeze_792: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_198, -1);  convert_element_type_198 = None
    unsqueeze_793: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
    unsqueeze_794: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_297, -1);  mul_297 = None
    unsqueeze_795: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
    sub_99: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_793);  unsqueeze_793 = None
    mul_298: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
    unsqueeze_796: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_298, -1)
    unsqueeze_797: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
    mul_299: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_298, unsqueeze_797);  mul_298 = unsqueeze_797 = None
    unsqueeze_798: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_299, -1);  primals_299 = None
    unsqueeze_799: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
    add_199: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_299, unsqueeze_799);  mul_299 = unsqueeze_799 = None
    relu_99: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_199);  add_199 = None
    convolution_99: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_99, primals_300, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_47: "f32[4, 704, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_200: "f32[704]" = torch.ops.prims.convert_element_type.default(primals_665, torch.float32)
    convert_element_type_201: "f32[704]" = torch.ops.prims.convert_element_type.default(primals_666, torch.float32)
    add_200: "f32[704]" = torch.ops.aten.add.Tensor(convert_element_type_201, 1e-05);  convert_element_type_201 = None
    sqrt_100: "f32[704]" = torch.ops.aten.sqrt.default(add_200);  add_200 = None
    reciprocal_100: "f32[704]" = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
    mul_300: "f32[704]" = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
    unsqueeze_800: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_200, -1);  convert_element_type_200 = None
    unsqueeze_801: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
    unsqueeze_802: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
    unsqueeze_803: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
    sub_100: "f32[4, 704, 7, 7]" = torch.ops.aten.sub.Tensor(cat_47, unsqueeze_801);  unsqueeze_801 = None
    mul_301: "f32[4, 704, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
    unsqueeze_804: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(primals_301, -1)
    unsqueeze_805: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
    mul_302: "f32[4, 704, 7, 7]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_805);  mul_301 = unsqueeze_805 = None
    unsqueeze_806: "f32[704, 1]" = torch.ops.aten.unsqueeze.default(primals_302, -1);  primals_302 = None
    unsqueeze_807: "f32[704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
    add_201: "f32[4, 704, 7, 7]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_807);  mul_302 = unsqueeze_807 = None
    relu_100: "f32[4, 704, 7, 7]" = torch.ops.aten.relu.default(add_201);  add_201 = None
    convolution_100: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_100, primals_303, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_202: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_668, torch.float32)
    convert_element_type_203: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_669, torch.float32)
    add_202: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_203, 1e-05);  convert_element_type_203 = None
    sqrt_101: "f32[128]" = torch.ops.aten.sqrt.default(add_202);  add_202 = None
    reciprocal_101: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
    mul_303: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
    unsqueeze_808: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_202, -1);  convert_element_type_202 = None
    unsqueeze_809: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
    unsqueeze_810: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
    unsqueeze_811: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
    sub_101: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_809);  unsqueeze_809 = None
    mul_304: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
    unsqueeze_812: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_304, -1)
    unsqueeze_813: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
    mul_305: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_813);  mul_304 = unsqueeze_813 = None
    unsqueeze_814: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_305, -1);  primals_305 = None
    unsqueeze_815: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
    add_203: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_815);  mul_305 = unsqueeze_815 = None
    relu_101: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_203);  add_203 = None
    convolution_101: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_101, primals_306, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_48: "f32[4, 736, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_204: "f32[736]" = torch.ops.prims.convert_element_type.default(primals_671, torch.float32)
    convert_element_type_205: "f32[736]" = torch.ops.prims.convert_element_type.default(primals_672, torch.float32)
    add_204: "f32[736]" = torch.ops.aten.add.Tensor(convert_element_type_205, 1e-05);  convert_element_type_205 = None
    sqrt_102: "f32[736]" = torch.ops.aten.sqrt.default(add_204);  add_204 = None
    reciprocal_102: "f32[736]" = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
    mul_306: "f32[736]" = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
    unsqueeze_816: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_204, -1);  convert_element_type_204 = None
    unsqueeze_817: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
    unsqueeze_818: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(mul_306, -1);  mul_306 = None
    unsqueeze_819: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
    sub_102: "f32[4, 736, 7, 7]" = torch.ops.aten.sub.Tensor(cat_48, unsqueeze_817);  unsqueeze_817 = None
    mul_307: "f32[4, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
    unsqueeze_820: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_307, -1)
    unsqueeze_821: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
    mul_308: "f32[4, 736, 7, 7]" = torch.ops.aten.mul.Tensor(mul_307, unsqueeze_821);  mul_307 = unsqueeze_821 = None
    unsqueeze_822: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_308, -1);  primals_308 = None
    unsqueeze_823: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
    add_205: "f32[4, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_308, unsqueeze_823);  mul_308 = unsqueeze_823 = None
    relu_102: "f32[4, 736, 7, 7]" = torch.ops.aten.relu.default(add_205);  add_205 = None
    convolution_102: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_102, primals_309, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_206: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_674, torch.float32)
    convert_element_type_207: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_675, torch.float32)
    add_206: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_207, 1e-05);  convert_element_type_207 = None
    sqrt_103: "f32[128]" = torch.ops.aten.sqrt.default(add_206);  add_206 = None
    reciprocal_103: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
    mul_309: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
    unsqueeze_824: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_206, -1);  convert_element_type_206 = None
    unsqueeze_825: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
    unsqueeze_826: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
    unsqueeze_827: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
    sub_103: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_825);  unsqueeze_825 = None
    mul_310: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
    unsqueeze_828: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_310, -1)
    unsqueeze_829: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
    mul_311: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_829);  mul_310 = unsqueeze_829 = None
    unsqueeze_830: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_311, -1);  primals_311 = None
    unsqueeze_831: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
    add_207: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_311, unsqueeze_831);  mul_311 = unsqueeze_831 = None
    relu_103: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_207);  add_207 = None
    convolution_103: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_103, primals_312, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_49: "f32[4, 768, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_208: "f32[768]" = torch.ops.prims.convert_element_type.default(primals_677, torch.float32)
    convert_element_type_209: "f32[768]" = torch.ops.prims.convert_element_type.default(primals_678, torch.float32)
    add_208: "f32[768]" = torch.ops.aten.add.Tensor(convert_element_type_209, 1e-05);  convert_element_type_209 = None
    sqrt_104: "f32[768]" = torch.ops.aten.sqrt.default(add_208);  add_208 = None
    reciprocal_104: "f32[768]" = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
    mul_312: "f32[768]" = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
    unsqueeze_832: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_208, -1);  convert_element_type_208 = None
    unsqueeze_833: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
    unsqueeze_834: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(mul_312, -1);  mul_312 = None
    unsqueeze_835: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
    sub_104: "f32[4, 768, 7, 7]" = torch.ops.aten.sub.Tensor(cat_49, unsqueeze_833);  unsqueeze_833 = None
    mul_313: "f32[4, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
    unsqueeze_836: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_313, -1)
    unsqueeze_837: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
    mul_314: "f32[4, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_837);  mul_313 = unsqueeze_837 = None
    unsqueeze_838: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_314, -1);  primals_314 = None
    unsqueeze_839: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
    add_209: "f32[4, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_839);  mul_314 = unsqueeze_839 = None
    relu_104: "f32[4, 768, 7, 7]" = torch.ops.aten.relu.default(add_209);  add_209 = None
    convolution_104: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_104, primals_315, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_210: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_680, torch.float32)
    convert_element_type_211: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_681, torch.float32)
    add_210: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_211, 1e-05);  convert_element_type_211 = None
    sqrt_105: "f32[128]" = torch.ops.aten.sqrt.default(add_210);  add_210 = None
    reciprocal_105: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
    mul_315: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
    unsqueeze_840: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_210, -1);  convert_element_type_210 = None
    unsqueeze_841: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
    unsqueeze_842: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_315, -1);  mul_315 = None
    unsqueeze_843: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
    sub_105: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_841);  unsqueeze_841 = None
    mul_316: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
    unsqueeze_844: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_316, -1)
    unsqueeze_845: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
    mul_317: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_316, unsqueeze_845);  mul_316 = unsqueeze_845 = None
    unsqueeze_846: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_317, -1);  primals_317 = None
    unsqueeze_847: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
    add_211: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_317, unsqueeze_847);  mul_317 = unsqueeze_847 = None
    relu_105: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_211);  add_211 = None
    convolution_105: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_105, primals_318, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_50: "f32[4, 800, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_212: "f32[800]" = torch.ops.prims.convert_element_type.default(primals_683, torch.float32)
    convert_element_type_213: "f32[800]" = torch.ops.prims.convert_element_type.default(primals_684, torch.float32)
    add_212: "f32[800]" = torch.ops.aten.add.Tensor(convert_element_type_213, 1e-05);  convert_element_type_213 = None
    sqrt_106: "f32[800]" = torch.ops.aten.sqrt.default(add_212);  add_212 = None
    reciprocal_106: "f32[800]" = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
    mul_318: "f32[800]" = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
    unsqueeze_848: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_212, -1);  convert_element_type_212 = None
    unsqueeze_849: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
    unsqueeze_850: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(mul_318, -1);  mul_318 = None
    unsqueeze_851: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
    sub_106: "f32[4, 800, 7, 7]" = torch.ops.aten.sub.Tensor(cat_50, unsqueeze_849);  unsqueeze_849 = None
    mul_319: "f32[4, 800, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
    unsqueeze_852: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(primals_319, -1)
    unsqueeze_853: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
    mul_320: "f32[4, 800, 7, 7]" = torch.ops.aten.mul.Tensor(mul_319, unsqueeze_853);  mul_319 = unsqueeze_853 = None
    unsqueeze_854: "f32[800, 1]" = torch.ops.aten.unsqueeze.default(primals_320, -1);  primals_320 = None
    unsqueeze_855: "f32[800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
    add_213: "f32[4, 800, 7, 7]" = torch.ops.aten.add.Tensor(mul_320, unsqueeze_855);  mul_320 = unsqueeze_855 = None
    relu_106: "f32[4, 800, 7, 7]" = torch.ops.aten.relu.default(add_213);  add_213 = None
    convolution_106: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_106, primals_321, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_214: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_686, torch.float32)
    convert_element_type_215: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_687, torch.float32)
    add_214: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_215, 1e-05);  convert_element_type_215 = None
    sqrt_107: "f32[128]" = torch.ops.aten.sqrt.default(add_214);  add_214 = None
    reciprocal_107: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
    mul_321: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
    unsqueeze_856: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_214, -1);  convert_element_type_214 = None
    unsqueeze_857: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
    unsqueeze_858: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_321, -1);  mul_321 = None
    unsqueeze_859: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
    sub_107: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_857);  unsqueeze_857 = None
    mul_322: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
    unsqueeze_860: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_322, -1)
    unsqueeze_861: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
    mul_323: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_861);  mul_322 = unsqueeze_861 = None
    unsqueeze_862: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_323, -1);  primals_323 = None
    unsqueeze_863: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
    add_215: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_323, unsqueeze_863);  mul_323 = unsqueeze_863 = None
    relu_107: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_215);  add_215 = None
    convolution_107: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_107, primals_324, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_51: "f32[4, 832, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_216: "f32[832]" = torch.ops.prims.convert_element_type.default(primals_689, torch.float32)
    convert_element_type_217: "f32[832]" = torch.ops.prims.convert_element_type.default(primals_690, torch.float32)
    add_216: "f32[832]" = torch.ops.aten.add.Tensor(convert_element_type_217, 1e-05);  convert_element_type_217 = None
    sqrt_108: "f32[832]" = torch.ops.aten.sqrt.default(add_216);  add_216 = None
    reciprocal_108: "f32[832]" = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
    mul_324: "f32[832]" = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
    unsqueeze_864: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_216, -1);  convert_element_type_216 = None
    unsqueeze_865: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
    unsqueeze_866: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(mul_324, -1);  mul_324 = None
    unsqueeze_867: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
    sub_108: "f32[4, 832, 7, 7]" = torch.ops.aten.sub.Tensor(cat_51, unsqueeze_865);  unsqueeze_865 = None
    mul_325: "f32[4, 832, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
    unsqueeze_868: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(primals_325, -1)
    unsqueeze_869: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
    mul_326: "f32[4, 832, 7, 7]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_869);  mul_325 = unsqueeze_869 = None
    unsqueeze_870: "f32[832, 1]" = torch.ops.aten.unsqueeze.default(primals_326, -1);  primals_326 = None
    unsqueeze_871: "f32[832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
    add_217: "f32[4, 832, 7, 7]" = torch.ops.aten.add.Tensor(mul_326, unsqueeze_871);  mul_326 = unsqueeze_871 = None
    relu_108: "f32[4, 832, 7, 7]" = torch.ops.aten.relu.default(add_217);  add_217 = None
    convolution_108: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_108, primals_327, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_218: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_692, torch.float32)
    convert_element_type_219: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_693, torch.float32)
    add_218: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_219, 1e-05);  convert_element_type_219 = None
    sqrt_109: "f32[128]" = torch.ops.aten.sqrt.default(add_218);  add_218 = None
    reciprocal_109: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
    mul_327: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
    unsqueeze_872: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_218, -1);  convert_element_type_218 = None
    unsqueeze_873: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
    unsqueeze_874: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
    unsqueeze_875: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
    sub_109: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_873);  unsqueeze_873 = None
    mul_328: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
    unsqueeze_876: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_328, -1)
    unsqueeze_877: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
    mul_329: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_877);  mul_328 = unsqueeze_877 = None
    unsqueeze_878: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_329, -1);  primals_329 = None
    unsqueeze_879: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
    add_219: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_329, unsqueeze_879);  mul_329 = unsqueeze_879 = None
    relu_109: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_219);  add_219 = None
    convolution_109: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_109, primals_330, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_52: "f32[4, 864, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_220: "f32[864]" = torch.ops.prims.convert_element_type.default(primals_695, torch.float32)
    convert_element_type_221: "f32[864]" = torch.ops.prims.convert_element_type.default(primals_696, torch.float32)
    add_220: "f32[864]" = torch.ops.aten.add.Tensor(convert_element_type_221, 1e-05);  convert_element_type_221 = None
    sqrt_110: "f32[864]" = torch.ops.aten.sqrt.default(add_220);  add_220 = None
    reciprocal_110: "f32[864]" = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
    mul_330: "f32[864]" = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
    unsqueeze_880: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_220, -1);  convert_element_type_220 = None
    unsqueeze_881: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
    unsqueeze_882: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
    unsqueeze_883: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
    sub_110: "f32[4, 864, 7, 7]" = torch.ops.aten.sub.Tensor(cat_52, unsqueeze_881);  unsqueeze_881 = None
    mul_331: "f32[4, 864, 7, 7]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
    unsqueeze_884: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_331, -1)
    unsqueeze_885: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
    mul_332: "f32[4, 864, 7, 7]" = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_885);  mul_331 = unsqueeze_885 = None
    unsqueeze_886: "f32[864, 1]" = torch.ops.aten.unsqueeze.default(primals_332, -1);  primals_332 = None
    unsqueeze_887: "f32[864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
    add_221: "f32[4, 864, 7, 7]" = torch.ops.aten.add.Tensor(mul_332, unsqueeze_887);  mul_332 = unsqueeze_887 = None
    relu_110: "f32[4, 864, 7, 7]" = torch.ops.aten.relu.default(add_221);  add_221 = None
    convolution_110: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_110, primals_333, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_222: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_698, torch.float32)
    convert_element_type_223: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_699, torch.float32)
    add_222: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_223, 1e-05);  convert_element_type_223 = None
    sqrt_111: "f32[128]" = torch.ops.aten.sqrt.default(add_222);  add_222 = None
    reciprocal_111: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
    mul_333: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
    unsqueeze_888: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_222, -1);  convert_element_type_222 = None
    unsqueeze_889: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
    unsqueeze_890: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
    unsqueeze_891: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
    sub_111: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_889);  unsqueeze_889 = None
    mul_334: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
    unsqueeze_892: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_334, -1)
    unsqueeze_893: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
    mul_335: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_893);  mul_334 = unsqueeze_893 = None
    unsqueeze_894: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_335, -1);  primals_335 = None
    unsqueeze_895: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
    add_223: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_895);  mul_335 = unsqueeze_895 = None
    relu_111: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_223);  add_223 = None
    convolution_111: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_111, primals_336, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_53: "f32[4, 896, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_224: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_701, torch.float32)
    convert_element_type_225: "f32[896]" = torch.ops.prims.convert_element_type.default(primals_702, torch.float32)
    add_224: "f32[896]" = torch.ops.aten.add.Tensor(convert_element_type_225, 1e-05);  convert_element_type_225 = None
    sqrt_112: "f32[896]" = torch.ops.aten.sqrt.default(add_224);  add_224 = None
    reciprocal_112: "f32[896]" = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
    mul_336: "f32[896]" = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
    unsqueeze_896: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_224, -1);  convert_element_type_224 = None
    unsqueeze_897: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
    unsqueeze_898: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
    unsqueeze_899: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
    sub_112: "f32[4, 896, 7, 7]" = torch.ops.aten.sub.Tensor(cat_53, unsqueeze_897);  unsqueeze_897 = None
    mul_337: "f32[4, 896, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
    unsqueeze_900: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_337, -1)
    unsqueeze_901: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
    mul_338: "f32[4, 896, 7, 7]" = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_901);  mul_337 = unsqueeze_901 = None
    unsqueeze_902: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_338, -1);  primals_338 = None
    unsqueeze_903: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
    add_225: "f32[4, 896, 7, 7]" = torch.ops.aten.add.Tensor(mul_338, unsqueeze_903);  mul_338 = unsqueeze_903 = None
    relu_112: "f32[4, 896, 7, 7]" = torch.ops.aten.relu.default(add_225);  add_225 = None
    convolution_112: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_112, primals_339, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_226: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_704, torch.float32)
    convert_element_type_227: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_705, torch.float32)
    add_226: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_227, 1e-05);  convert_element_type_227 = None
    sqrt_113: "f32[128]" = torch.ops.aten.sqrt.default(add_226);  add_226 = None
    reciprocal_113: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
    mul_339: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
    unsqueeze_904: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_226, -1);  convert_element_type_226 = None
    unsqueeze_905: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
    unsqueeze_906: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
    unsqueeze_907: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
    sub_113: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_905);  unsqueeze_905 = None
    mul_340: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
    unsqueeze_908: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_340, -1)
    unsqueeze_909: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
    mul_341: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_909);  mul_340 = unsqueeze_909 = None
    unsqueeze_910: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_341, -1);  primals_341 = None
    unsqueeze_911: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
    add_227: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_341, unsqueeze_911);  mul_341 = unsqueeze_911 = None
    relu_113: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_227);  add_227 = None
    convolution_113: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_113, primals_342, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_54: "f32[4, 928, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111, convolution_113], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_228: "f32[928]" = torch.ops.prims.convert_element_type.default(primals_707, torch.float32)
    convert_element_type_229: "f32[928]" = torch.ops.prims.convert_element_type.default(primals_708, torch.float32)
    add_228: "f32[928]" = torch.ops.aten.add.Tensor(convert_element_type_229, 1e-05);  convert_element_type_229 = None
    sqrt_114: "f32[928]" = torch.ops.aten.sqrt.default(add_228);  add_228 = None
    reciprocal_114: "f32[928]" = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
    mul_342: "f32[928]" = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
    unsqueeze_912: "f32[928, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_228, -1);  convert_element_type_228 = None
    unsqueeze_913: "f32[928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
    unsqueeze_914: "f32[928, 1]" = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
    unsqueeze_915: "f32[928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
    sub_114: "f32[4, 928, 7, 7]" = torch.ops.aten.sub.Tensor(cat_54, unsqueeze_913);  unsqueeze_913 = None
    mul_343: "f32[4, 928, 7, 7]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
    unsqueeze_916: "f32[928, 1]" = torch.ops.aten.unsqueeze.default(primals_343, -1)
    unsqueeze_917: "f32[928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
    mul_344: "f32[4, 928, 7, 7]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_917);  mul_343 = unsqueeze_917 = None
    unsqueeze_918: "f32[928, 1]" = torch.ops.aten.unsqueeze.default(primals_344, -1);  primals_344 = None
    unsqueeze_919: "f32[928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
    add_229: "f32[4, 928, 7, 7]" = torch.ops.aten.add.Tensor(mul_344, unsqueeze_919);  mul_344 = unsqueeze_919 = None
    relu_114: "f32[4, 928, 7, 7]" = torch.ops.aten.relu.default(add_229);  add_229 = None
    convolution_114: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_114, primals_345, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_230: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_710, torch.float32)
    convert_element_type_231: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_711, torch.float32)
    add_230: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_231, 1e-05);  convert_element_type_231 = None
    sqrt_115: "f32[128]" = torch.ops.aten.sqrt.default(add_230);  add_230 = None
    reciprocal_115: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
    mul_345: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
    unsqueeze_920: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_230, -1);  convert_element_type_230 = None
    unsqueeze_921: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
    unsqueeze_922: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_345, -1);  mul_345 = None
    unsqueeze_923: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
    sub_115: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_921);  unsqueeze_921 = None
    mul_346: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
    unsqueeze_924: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_346, -1)
    unsqueeze_925: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
    mul_347: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_925);  mul_346 = unsqueeze_925 = None
    unsqueeze_926: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_347, -1);  primals_347 = None
    unsqueeze_927: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
    add_231: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_347, unsqueeze_927);  mul_347 = unsqueeze_927 = None
    relu_115: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_231);  add_231 = None
    convolution_115: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_115, primals_348, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_55: "f32[4, 960, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111, convolution_113, convolution_115], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_232: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_713, torch.float32)
    convert_element_type_233: "f32[960]" = torch.ops.prims.convert_element_type.default(primals_714, torch.float32)
    add_232: "f32[960]" = torch.ops.aten.add.Tensor(convert_element_type_233, 1e-05);  convert_element_type_233 = None
    sqrt_116: "f32[960]" = torch.ops.aten.sqrt.default(add_232);  add_232 = None
    reciprocal_116: "f32[960]" = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
    mul_348: "f32[960]" = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
    unsqueeze_928: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_232, -1);  convert_element_type_232 = None
    unsqueeze_929: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
    unsqueeze_930: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
    unsqueeze_931: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
    sub_116: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(cat_55, unsqueeze_929);  unsqueeze_929 = None
    mul_349: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
    unsqueeze_932: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_349, -1)
    unsqueeze_933: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
    mul_350: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_933);  mul_349 = unsqueeze_933 = None
    unsqueeze_934: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(primals_350, -1);  primals_350 = None
    unsqueeze_935: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
    add_233: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_350, unsqueeze_935);  mul_350 = unsqueeze_935 = None
    relu_116: "f32[4, 960, 7, 7]" = torch.ops.aten.relu.default(add_233);  add_233 = None
    convolution_116: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_116, primals_351, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_234: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_716, torch.float32)
    convert_element_type_235: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_717, torch.float32)
    add_234: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_235, 1e-05);  convert_element_type_235 = None
    sqrt_117: "f32[128]" = torch.ops.aten.sqrt.default(add_234);  add_234 = None
    reciprocal_117: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
    mul_351: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
    unsqueeze_936: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_234, -1);  convert_element_type_234 = None
    unsqueeze_937: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
    unsqueeze_938: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
    unsqueeze_939: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
    sub_117: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_937);  unsqueeze_937 = None
    mul_352: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
    unsqueeze_940: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_352, -1)
    unsqueeze_941: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
    mul_353: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_941);  mul_352 = unsqueeze_941 = None
    unsqueeze_942: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_353, -1);  primals_353 = None
    unsqueeze_943: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
    add_235: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_353, unsqueeze_943);  mul_353 = unsqueeze_943 = None
    relu_117: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_235);  add_235 = None
    convolution_117: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_117, primals_354, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:48, code: concated_features = torch.cat(inputs, 1)
    cat_56: "f32[4, 992, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111, convolution_113, convolution_115, convolution_117], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    convert_element_type_236: "f32[992]" = torch.ops.prims.convert_element_type.default(primals_719, torch.float32)
    convert_element_type_237: "f32[992]" = torch.ops.prims.convert_element_type.default(primals_720, torch.float32)
    add_236: "f32[992]" = torch.ops.aten.add.Tensor(convert_element_type_237, 1e-05);  convert_element_type_237 = None
    sqrt_118: "f32[992]" = torch.ops.aten.sqrt.default(add_236);  add_236 = None
    reciprocal_118: "f32[992]" = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
    mul_354: "f32[992]" = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
    unsqueeze_944: "f32[992, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_236, -1);  convert_element_type_236 = None
    unsqueeze_945: "f32[992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
    unsqueeze_946: "f32[992, 1]" = torch.ops.aten.unsqueeze.default(mul_354, -1);  mul_354 = None
    unsqueeze_947: "f32[992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
    sub_118: "f32[4, 992, 7, 7]" = torch.ops.aten.sub.Tensor(cat_56, unsqueeze_945);  unsqueeze_945 = None
    mul_355: "f32[4, 992, 7, 7]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
    unsqueeze_948: "f32[992, 1]" = torch.ops.aten.unsqueeze.default(primals_355, -1)
    unsqueeze_949: "f32[992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
    mul_356: "f32[4, 992, 7, 7]" = torch.ops.aten.mul.Tensor(mul_355, unsqueeze_949);  mul_355 = unsqueeze_949 = None
    unsqueeze_950: "f32[992, 1]" = torch.ops.aten.unsqueeze.default(primals_356, -1);  primals_356 = None
    unsqueeze_951: "f32[992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
    add_237: "f32[4, 992, 7, 7]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_951);  mul_356 = unsqueeze_951 = None
    relu_118: "f32[4, 992, 7, 7]" = torch.ops.aten.relu.default(add_237);  add_237 = None
    convolution_118: "f32[4, 128, 7, 7]" = torch.ops.aten.convolution.default(relu_118, primals_357, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:90, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
    convert_element_type_238: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_722, torch.float32)
    convert_element_type_239: "f32[128]" = torch.ops.prims.convert_element_type.default(primals_723, torch.float32)
    add_238: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_239, 1e-05);  convert_element_type_239 = None
    sqrt_119: "f32[128]" = torch.ops.aten.sqrt.default(add_238);  add_238 = None
    reciprocal_119: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
    mul_357: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
    unsqueeze_952: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_238, -1);  convert_element_type_238 = None
    unsqueeze_953: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
    unsqueeze_954: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_357, -1);  mul_357 = None
    unsqueeze_955: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
    sub_119: "f32[4, 128, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_953);  unsqueeze_953 = None
    mul_358: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
    unsqueeze_956: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_358, -1)
    unsqueeze_957: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
    mul_359: "f32[4, 128, 7, 7]" = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_957);  mul_358 = unsqueeze_957 = None
    unsqueeze_958: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_359, -1);  primals_359 = None
    unsqueeze_959: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
    add_239: "f32[4, 128, 7, 7]" = torch.ops.aten.add.Tensor(mul_359, unsqueeze_959);  mul_359 = unsqueeze_959 = None
    relu_119: "f32[4, 128, 7, 7]" = torch.ops.aten.relu.default(add_239);  add_239 = None
    convolution_119: "f32[4, 32, 7, 7]" = torch.ops.aten.convolution.default(relu_119, primals_360, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:124, code: return torch.cat(features, 1)
    cat_57: "f32[4, 1024, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111, convolution_113, convolution_115, convolution_117, convolution_119], 1);  convolution_89 = convolution_91 = convolution_93 = convolution_95 = convolution_97 = convolution_99 = convolution_101 = convolution_103 = convolution_105 = convolution_107 = convolution_109 = convolution_111 = convolution_113 = convolution_115 = convolution_117 = convolution_119 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:213, code: features = self.features(x)
    convert_element_type_240: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_725, torch.float32)
    convert_element_type_241: "f32[1024]" = torch.ops.prims.convert_element_type.default(primals_726, torch.float32)
    add_240: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_241, 1e-05);  convert_element_type_241 = None
    sqrt_120: "f32[1024]" = torch.ops.aten.sqrt.default(add_240);  add_240 = None
    reciprocal_120: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
    mul_360: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
    unsqueeze_960: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_240, -1);  convert_element_type_240 = None
    unsqueeze_961: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
    unsqueeze_962: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
    unsqueeze_963: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
    sub_120: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(cat_57, unsqueeze_961);  unsqueeze_961 = None
    mul_361: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
    unsqueeze_964: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_361, -1)
    unsqueeze_965: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
    mul_362: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_965);  mul_361 = unsqueeze_965 = None
    unsqueeze_966: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_362, -1);  primals_362 = None
    unsqueeze_967: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
    add_241: "f32[4, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_362, unsqueeze_967);  mul_362 = unsqueeze_967 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:214, code: out = F.relu(features, inplace=True)
    relu_120: "f32[4, 1024, 7, 7]" = torch.ops.aten.relu.default(add_241);  add_241 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:215, code: out = F.adaptive_avg_pool2d(out, (1, 1))
    mean: "f32[4, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu_120, [-1, -2], True)
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:216, code: out = torch.flatten(out, 1)
    view: "f32[4, 1024]" = torch.ops.aten.view.default(mean, [4, 1024]);  mean = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:217, code: out = self.classifier(out)
    permute: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_363, [1, 0]);  primals_363 = None
    addmm: "f32[4, 1000]" = torch.ops.aten.addmm.default(primals_364, view, permute);  primals_364 = None
    permute_1: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:214, code: out = F.relu(features, inplace=True)
    alias_122: "f32[4, 1024, 7, 7]" = torch.ops.aten.alias.default(relu_120);  relu_120 = None
    alias_123: "f32[4, 1024, 7, 7]" = torch.ops.aten.alias.default(alias_122);  alias_122 = None
    le: "b8[4, 1024, 7, 7]" = torch.ops.aten.le.Scalar(alias_123, 0);  alias_123 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/densenet.py:49, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
    unsqueeze_2396: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_368, 0);  primals_368 = None
    unsqueeze_2397: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2396, 2);  unsqueeze_2396 = None
    unsqueeze_2398: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2397, 3);  unsqueeze_2397 = None
    sub_240: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(clone, unsqueeze_2398);  clone = unsqueeze_2398 = None
    return [addmm, primals_1, primals_2, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_148, primals_150, primals_151, primals_153, primals_154, primals_156, primals_157, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, primals_316, primals_318, primals_319, primals_321, primals_322, primals_324, primals_325, primals_327, primals_328, primals_330, primals_331, primals_333, primals_334, primals_336, primals_337, primals_339, primals_340, primals_342, primals_343, primals_345, primals_346, primals_348, primals_349, primals_351, primals_352, primals_354, primals_355, primals_357, primals_358, primals_360, primals_361, primals_365, primals_366, primals_369, primals_371, primals_372, primals_374, primals_375, primals_377, primals_378, primals_380, primals_381, primals_383, primals_384, primals_386, primals_387, primals_389, primals_390, primals_392, primals_393, primals_395, primals_396, primals_398, primals_399, primals_401, primals_402, primals_404, primals_405, primals_407, primals_408, primals_410, primals_411, primals_413, primals_414, primals_416, primals_417, primals_419, primals_420, primals_422, primals_423, primals_425, primals_426, primals_428, primals_429, primals_431, primals_432, primals_434, primals_435, primals_437, primals_438, primals_440, primals_441, primals_443, primals_444, primals_446, primals_447, primals_449, primals_450, primals_452, primals_453, primals_455, primals_456, primals_458, primals_459, primals_461, primals_462, primals_464, primals_465, primals_467, primals_468, primals_470, primals_471, primals_473, primals_474, primals_476, primals_477, primals_479, primals_480, primals_482, primals_483, primals_485, primals_486, primals_488, primals_489, primals_491, primals_492, primals_494, primals_495, primals_497, primals_498, primals_500, primals_501, primals_503, primals_504, primals_506, primals_507, primals_509, primals_510, primals_512, primals_513, primals_515, primals_516, primals_518, primals_519, primals_521, primals_522, primals_524, primals_525, primals_527, primals_528, primals_530, primals_531, primals_533, primals_534, primals_536, primals_537, primals_539, primals_540, primals_542, primals_543, primals_545, primals_546, primals_548, primals_549, primals_551, primals_552, primals_554, primals_555, primals_557, primals_558, primals_560, primals_561, primals_563, primals_564, primals_566, primals_567, primals_569, primals_570, primals_572, primals_573, primals_575, primals_576, primals_578, primals_579, primals_581, primals_582, primals_584, primals_585, primals_587, primals_588, primals_590, primals_591, primals_593, primals_594, primals_596, primals_597, primals_599, primals_600, primals_602, primals_603, primals_605, primals_606, primals_608, primals_609, primals_611, primals_612, primals_614, primals_615, primals_617, primals_618, primals_620, primals_621, primals_623, primals_624, primals_626, primals_627, primals_629, primals_630, primals_632, primals_633, primals_635, primals_636, primals_638, primals_639, primals_641, primals_642, primals_644, primals_645, primals_647, primals_648, primals_650, primals_651, primals_653, primals_654, primals_656, primals_657, primals_659, primals_660, primals_662, primals_663, primals_665, primals_666, primals_668, primals_669, primals_671, primals_672, primals_674, primals_675, primals_677, primals_678, primals_680, primals_681, primals_683, primals_684, primals_686, primals_687, primals_689, primals_690, primals_692, primals_693, primals_695, primals_696, primals_698, primals_699, primals_701, primals_702, primals_704, primals_705, primals_707, primals_708, primals_710, primals_711, primals_713, primals_714, primals_716, primals_717, primals_719, primals_720, primals_722, primals_723, primals_725, primals_726, primals_728, convolution, relu, getitem_1, relu_1, convolution_1, relu_2, cat, relu_3, convolution_3, relu_4, cat_1, relu_5, convolution_5, relu_6, cat_2, relu_7, convolution_7, relu_8, cat_3, relu_9, convolution_9, relu_10, cat_4, relu_11, convolution_11, relu_12, cat_5, relu_13, convolution_13, avg_pool2d, relu_14, convolution_14, relu_15, cat_6, relu_16, convolution_16, relu_17, cat_7, relu_18, convolution_18, relu_19, cat_8, relu_20, convolution_20, relu_21, cat_9, relu_22, convolution_22, relu_23, cat_10, relu_24, convolution_24, relu_25, cat_11, relu_26, convolution_26, relu_27, cat_12, relu_28, convolution_28, relu_29, cat_13, relu_30, convolution_30, relu_31, cat_14, relu_32, convolution_32, relu_33, cat_15, relu_34, convolution_34, relu_35, cat_16, relu_36, convolution_36, relu_37, cat_17, relu_38, convolution_38, avg_pool2d_1, relu_39, convolution_39, relu_40, cat_18, relu_41, convolution_41, relu_42, cat_19, relu_43, convolution_43, relu_44, cat_20, relu_45, convolution_45, relu_46, cat_21, relu_47, convolution_47, relu_48, cat_22, relu_49, convolution_49, relu_50, cat_23, relu_51, convolution_51, relu_52, cat_24, relu_53, convolution_53, relu_54, cat_25, relu_55, convolution_55, relu_56, cat_26, relu_57, convolution_57, relu_58, cat_27, relu_59, convolution_59, relu_60, cat_28, relu_61, convolution_61, relu_62, cat_29, relu_63, convolution_63, relu_64, cat_30, relu_65, convolution_65, relu_66, cat_31, relu_67, convolution_67, relu_68, cat_32, relu_69, convolution_69, relu_70, cat_33, relu_71, convolution_71, relu_72, cat_34, relu_73, convolution_73, relu_74, cat_35, relu_75, convolution_75, relu_76, cat_36, relu_77, convolution_77, relu_78, cat_37, relu_79, convolution_79, relu_80, cat_38, relu_81, convolution_81, relu_82, cat_39, relu_83, convolution_83, relu_84, cat_40, relu_85, convolution_85, relu_86, cat_41, relu_87, convolution_87, avg_pool2d_2, relu_88, convolution_88, relu_89, cat_42, relu_90, convolution_90, relu_91, cat_43, relu_92, convolution_92, relu_93, cat_44, relu_94, convolution_94, relu_95, cat_45, relu_96, convolution_96, relu_97, cat_46, relu_98, convolution_98, relu_99, cat_47, relu_100, convolution_100, relu_101, cat_48, relu_102, convolution_102, relu_103, cat_49, relu_104, convolution_104, relu_105, cat_50, relu_106, convolution_106, relu_107, cat_51, relu_108, convolution_108, relu_109, cat_52, relu_110, convolution_110, relu_111, cat_53, relu_112, convolution_112, relu_113, cat_54, relu_114, convolution_114, relu_115, cat_55, relu_116, convolution_116, relu_117, cat_56, relu_118, convolution_118, relu_119, cat_57, view, permute_1, le, sub_240]
    