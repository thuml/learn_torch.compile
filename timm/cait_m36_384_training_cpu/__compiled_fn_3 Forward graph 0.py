from __future__ import annotations



def forward(self, primals_1: "f32[1, 576, 768]", primals_2: "f32[768]", primals_3: "f32[768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[768]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[768]", primals_27: "f32[768]", primals_28: "f32[768]", primals_29: "f32[768]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[768]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[768]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[768]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[768]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[768]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[1, 1, 768]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[768]", primals_78: "f32[768]", primals_79: "f32[768, 3, 16, 16]", primals_80: "f32[768]", primals_81: "f32[768]", primals_82: "f32[768]", primals_83: "f32[2304, 768]", primals_84: "f32[2304]", primals_85: "f32[16, 16]", primals_86: "f32[16]", primals_87: "f32[16, 16]", primals_88: "f32[16]", primals_89: "f32[768, 768]", primals_90: "f32[768]", primals_91: "f32[768]", primals_92: "f32[768]", primals_93: "f32[3072, 768]", primals_94: "f32[3072]", primals_95: "f32[768, 3072]", primals_96: "f32[768]", primals_97: "f32[768]", primals_98: "f32[768]", primals_99: "f32[2304, 768]", primals_100: "f32[2304]", primals_101: "f32[16, 16]", primals_102: "f32[16]", primals_103: "f32[16, 16]", primals_104: "f32[16]", primals_105: "f32[768, 768]", primals_106: "f32[768]", primals_107: "f32[768]", primals_108: "f32[768]", primals_109: "f32[3072, 768]", primals_110: "f32[3072]", primals_111: "f32[768, 3072]", primals_112: "f32[768]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[2304, 768]", primals_116: "f32[2304]", primals_117: "f32[16, 16]", primals_118: "f32[16]", primals_119: "f32[16, 16]", primals_120: "f32[16]", primals_121: "f32[768, 768]", primals_122: "f32[768]", primals_123: "f32[768]", primals_124: "f32[768]", primals_125: "f32[3072, 768]", primals_126: "f32[3072]", primals_127: "f32[768, 3072]", primals_128: "f32[768]", primals_129: "f32[768]", primals_130: "f32[768]", primals_131: "f32[2304, 768]", primals_132: "f32[2304]", primals_133: "f32[16, 16]", primals_134: "f32[16]", primals_135: "f32[16, 16]", primals_136: "f32[16]", primals_137: "f32[768, 768]", primals_138: "f32[768]", primals_139: "f32[768]", primals_140: "f32[768]", primals_141: "f32[3072, 768]", primals_142: "f32[3072]", primals_143: "f32[768, 3072]", primals_144: "f32[768]", primals_145: "f32[768]", primals_146: "f32[768]", primals_147: "f32[2304, 768]", primals_148: "f32[2304]", primals_149: "f32[16, 16]", primals_150: "f32[16]", primals_151: "f32[16, 16]", primals_152: "f32[16]", primals_153: "f32[768, 768]", primals_154: "f32[768]", primals_155: "f32[768]", primals_156: "f32[768]", primals_157: "f32[3072, 768]", primals_158: "f32[3072]", primals_159: "f32[768, 3072]", primals_160: "f32[768]", primals_161: "f32[768]", primals_162: "f32[768]", primals_163: "f32[2304, 768]", primals_164: "f32[2304]", primals_165: "f32[16, 16]", primals_166: "f32[16]", primals_167: "f32[16, 16]", primals_168: "f32[16]", primals_169: "f32[768, 768]", primals_170: "f32[768]", primals_171: "f32[768]", primals_172: "f32[768]", primals_173: "f32[3072, 768]", primals_174: "f32[3072]", primals_175: "f32[768, 3072]", primals_176: "f32[768]", primals_177: "f32[768]", primals_178: "f32[768]", primals_179: "f32[2304, 768]", primals_180: "f32[2304]", primals_181: "f32[16, 16]", primals_182: "f32[16]", primals_183: "f32[16, 16]", primals_184: "f32[16]", primals_185: "f32[768, 768]", primals_186: "f32[768]", primals_187: "f32[768]", primals_188: "f32[768]", primals_189: "f32[3072, 768]", primals_190: "f32[3072]", primals_191: "f32[768, 3072]", primals_192: "f32[768]", primals_193: "f32[768]", primals_194: "f32[768]", primals_195: "f32[2304, 768]", primals_196: "f32[2304]", primals_197: "f32[16, 16]", primals_198: "f32[16]", primals_199: "f32[16, 16]", primals_200: "f32[16]", primals_201: "f32[768, 768]", primals_202: "f32[768]", primals_203: "f32[768]", primals_204: "f32[768]", primals_205: "f32[3072, 768]", primals_206: "f32[3072]", primals_207: "f32[768, 3072]", primals_208: "f32[768]", primals_209: "f32[768]", primals_210: "f32[768]", primals_211: "f32[2304, 768]", primals_212: "f32[2304]", primals_213: "f32[16, 16]", primals_214: "f32[16]", primals_215: "f32[16, 16]", primals_216: "f32[16]", primals_217: "f32[768, 768]", primals_218: "f32[768]", primals_219: "f32[768]", primals_220: "f32[768]", primals_221: "f32[3072, 768]", primals_222: "f32[3072]", primals_223: "f32[768, 3072]", primals_224: "f32[768]", primals_225: "f32[768]", primals_226: "f32[768]", primals_227: "f32[2304, 768]", primals_228: "f32[2304]", primals_229: "f32[16, 16]", primals_230: "f32[16]", primals_231: "f32[16, 16]", primals_232: "f32[16]", primals_233: "f32[768, 768]", primals_234: "f32[768]", primals_235: "f32[768]", primals_236: "f32[768]", primals_237: "f32[3072, 768]", primals_238: "f32[3072]", primals_239: "f32[768, 3072]", primals_240: "f32[768]", primals_241: "f32[768]", primals_242: "f32[768]", primals_243: "f32[2304, 768]", primals_244: "f32[2304]", primals_245: "f32[16, 16]", primals_246: "f32[16]", primals_247: "f32[16, 16]", primals_248: "f32[16]", primals_249: "f32[768, 768]", primals_250: "f32[768]", primals_251: "f32[768]", primals_252: "f32[768]", primals_253: "f32[3072, 768]", primals_254: "f32[3072]", primals_255: "f32[768, 3072]", primals_256: "f32[768]", primals_257: "f32[768]", primals_258: "f32[768]", primals_259: "f32[2304, 768]", primals_260: "f32[2304]", primals_261: "f32[16, 16]", primals_262: "f32[16]", primals_263: "f32[16, 16]", primals_264: "f32[16]", primals_265: "f32[768, 768]", primals_266: "f32[768]", primals_267: "f32[768]", primals_268: "f32[768]", primals_269: "f32[3072, 768]", primals_270: "f32[3072]", primals_271: "f32[768, 3072]", primals_272: "f32[768]", primals_273: "f32[768]", primals_274: "f32[768]", primals_275: "f32[2304, 768]", primals_276: "f32[2304]", primals_277: "f32[16, 16]", primals_278: "f32[16]", primals_279: "f32[16, 16]", primals_280: "f32[16]", primals_281: "f32[768, 768]", primals_282: "f32[768]", primals_283: "f32[768]", primals_284: "f32[768]", primals_285: "f32[3072, 768]", primals_286: "f32[3072]", primals_287: "f32[768, 3072]", primals_288: "f32[768]", primals_289: "f32[768]", primals_290: "f32[768]", primals_291: "f32[2304, 768]", primals_292: "f32[2304]", primals_293: "f32[16, 16]", primals_294: "f32[16]", primals_295: "f32[16, 16]", primals_296: "f32[16]", primals_297: "f32[768, 768]", primals_298: "f32[768]", primals_299: "f32[768]", primals_300: "f32[768]", primals_301: "f32[3072, 768]", primals_302: "f32[3072]", primals_303: "f32[768, 3072]", primals_304: "f32[768]", primals_305: "f32[768]", primals_306: "f32[768]", primals_307: "f32[2304, 768]", primals_308: "f32[2304]", primals_309: "f32[16, 16]", primals_310: "f32[16]", primals_311: "f32[16, 16]", primals_312: "f32[16]", primals_313: "f32[768, 768]", primals_314: "f32[768]", primals_315: "f32[768]", primals_316: "f32[768]", primals_317: "f32[3072, 768]", primals_318: "f32[3072]", primals_319: "f32[768, 3072]", primals_320: "f32[768]", primals_321: "f32[768]", primals_322: "f32[768]", primals_323: "f32[2304, 768]", primals_324: "f32[2304]", primals_325: "f32[16, 16]", primals_326: "f32[16]", primals_327: "f32[16, 16]", primals_328: "f32[16]", primals_329: "f32[768, 768]", primals_330: "f32[768]", primals_331: "f32[768]", primals_332: "f32[768]", primals_333: "f32[3072, 768]", primals_334: "f32[3072]", primals_335: "f32[768, 3072]", primals_336: "f32[768]", primals_337: "f32[768]", primals_338: "f32[768]", primals_339: "f32[2304, 768]", primals_340: "f32[2304]", primals_341: "f32[16, 16]", primals_342: "f32[16]", primals_343: "f32[16, 16]", primals_344: "f32[16]", primals_345: "f32[768, 768]", primals_346: "f32[768]", primals_347: "f32[768]", primals_348: "f32[768]", primals_349: "f32[3072, 768]", primals_350: "f32[3072]", primals_351: "f32[768, 3072]", primals_352: "f32[768]", primals_353: "f32[768]", primals_354: "f32[768]", primals_355: "f32[2304, 768]", primals_356: "f32[2304]", primals_357: "f32[16, 16]", primals_358: "f32[16]", primals_359: "f32[16, 16]", primals_360: "f32[16]", primals_361: "f32[768, 768]", primals_362: "f32[768]", primals_363: "f32[768]", primals_364: "f32[768]", primals_365: "f32[3072, 768]", primals_366: "f32[3072]", primals_367: "f32[768, 3072]", primals_368: "f32[768]", primals_369: "f32[768]", primals_370: "f32[768]", primals_371: "f32[2304, 768]", primals_372: "f32[2304]", primals_373: "f32[16, 16]", primals_374: "f32[16]", primals_375: "f32[16, 16]", primals_376: "f32[16]", primals_377: "f32[768, 768]", primals_378: "f32[768]", primals_379: "f32[768]", primals_380: "f32[768]", primals_381: "f32[3072, 768]", primals_382: "f32[3072]", primals_383: "f32[768, 3072]", primals_384: "f32[768]", primals_385: "f32[768]", primals_386: "f32[768]", primals_387: "f32[2304, 768]", primals_388: "f32[2304]", primals_389: "f32[16, 16]", primals_390: "f32[16]", primals_391: "f32[16, 16]", primals_392: "f32[16]", primals_393: "f32[768, 768]", primals_394: "f32[768]", primals_395: "f32[768]", primals_396: "f32[768]", primals_397: "f32[3072, 768]", primals_398: "f32[3072]", primals_399: "f32[768, 3072]", primals_400: "f32[768]", primals_401: "f32[768]", primals_402: "f32[768]", primals_403: "f32[2304, 768]", primals_404: "f32[2304]", primals_405: "f32[16, 16]", primals_406: "f32[16]", primals_407: "f32[16, 16]", primals_408: "f32[16]", primals_409: "f32[768, 768]", primals_410: "f32[768]", primals_411: "f32[768]", primals_412: "f32[768]", primals_413: "f32[3072, 768]", primals_414: "f32[3072]", primals_415: "f32[768, 3072]", primals_416: "f32[768]", primals_417: "f32[768]", primals_418: "f32[768]", primals_419: "f32[2304, 768]", primals_420: "f32[2304]", primals_421: "f32[16, 16]", primals_422: "f32[16]", primals_423: "f32[16, 16]", primals_424: "f32[16]", primals_425: "f32[768, 768]", primals_426: "f32[768]", primals_427: "f32[768]", primals_428: "f32[768]", primals_429: "f32[3072, 768]", primals_430: "f32[3072]", primals_431: "f32[768, 3072]", primals_432: "f32[768]", primals_433: "f32[768]", primals_434: "f32[768]", primals_435: "f32[2304, 768]", primals_436: "f32[2304]", primals_437: "f32[16, 16]", primals_438: "f32[16]", primals_439: "f32[16, 16]", primals_440: "f32[16]", primals_441: "f32[768, 768]", primals_442: "f32[768]", primals_443: "f32[768]", primals_444: "f32[768]", primals_445: "f32[3072, 768]", primals_446: "f32[3072]", primals_447: "f32[768, 3072]", primals_448: "f32[768]", primals_449: "f32[768]", primals_450: "f32[768]", primals_451: "f32[2304, 768]", primals_452: "f32[2304]", primals_453: "f32[16, 16]", primals_454: "f32[16]", primals_455: "f32[16, 16]", primals_456: "f32[16]", primals_457: "f32[768, 768]", primals_458: "f32[768]", primals_459: "f32[768]", primals_460: "f32[768]", primals_461: "f32[3072, 768]", primals_462: "f32[3072]", primals_463: "f32[768, 3072]", primals_464: "f32[768]", primals_465: "f32[768]", primals_466: "f32[768]", primals_467: "f32[2304, 768]", primals_468: "f32[2304]", primals_469: "f32[16, 16]", primals_470: "f32[16]", primals_471: "f32[16, 16]", primals_472: "f32[16]", primals_473: "f32[768, 768]", primals_474: "f32[768]", primals_475: "f32[768]", primals_476: "f32[768]", primals_477: "f32[3072, 768]", primals_478: "f32[3072]", primals_479: "f32[768, 3072]", primals_480: "f32[768]", primals_481: "f32[768]", primals_482: "f32[768]", primals_483: "f32[2304, 768]", primals_484: "f32[2304]", primals_485: "f32[16, 16]", primals_486: "f32[16]", primals_487: "f32[16, 16]", primals_488: "f32[16]", primals_489: "f32[768, 768]", primals_490: "f32[768]", primals_491: "f32[768]", primals_492: "f32[768]", primals_493: "f32[3072, 768]", primals_494: "f32[3072]", primals_495: "f32[768, 3072]", primals_496: "f32[768]", primals_497: "f32[768]", primals_498: "f32[768]", primals_499: "f32[2304, 768]", primals_500: "f32[2304]", primals_501: "f32[16, 16]", primals_502: "f32[16]", primals_503: "f32[16, 16]", primals_504: "f32[16]", primals_505: "f32[768, 768]", primals_506: "f32[768]", primals_507: "f32[768]", primals_508: "f32[768]", primals_509: "f32[3072, 768]", primals_510: "f32[3072]", primals_511: "f32[768, 3072]", primals_512: "f32[768]", primals_513: "f32[768]", primals_514: "f32[768]", primals_515: "f32[2304, 768]", primals_516: "f32[2304]", primals_517: "f32[16, 16]", primals_518: "f32[16]", primals_519: "f32[16, 16]", primals_520: "f32[16]", primals_521: "f32[768, 768]", primals_522: "f32[768]", primals_523: "f32[768]", primals_524: "f32[768]", primals_525: "f32[3072, 768]", primals_526: "f32[3072]", primals_527: "f32[768, 3072]", primals_528: "f32[768]", primals_529: "f32[768]", primals_530: "f32[768]", primals_531: "f32[2304, 768]", primals_532: "f32[2304]", primals_533: "f32[16, 16]", primals_534: "f32[16]", primals_535: "f32[16, 16]", primals_536: "f32[16]", primals_537: "f32[768, 768]", primals_538: "f32[768]", primals_539: "f32[768]", primals_540: "f32[768]", primals_541: "f32[3072, 768]", primals_542: "f32[3072]", primals_543: "f32[768, 3072]", primals_544: "f32[768]", primals_545: "f32[768]", primals_546: "f32[768]", primals_547: "f32[2304, 768]", primals_548: "f32[2304]", primals_549: "f32[16, 16]", primals_550: "f32[16]", primals_551: "f32[16, 16]", primals_552: "f32[16]", primals_553: "f32[768, 768]", primals_554: "f32[768]", primals_555: "f32[768]", primals_556: "f32[768]", primals_557: "f32[3072, 768]", primals_558: "f32[3072]", primals_559: "f32[768, 3072]", primals_560: "f32[768]", primals_561: "f32[768]", primals_562: "f32[768]", primals_563: "f32[2304, 768]", primals_564: "f32[2304]", primals_565: "f32[16, 16]", primals_566: "f32[16]", primals_567: "f32[16, 16]", primals_568: "f32[16]", primals_569: "f32[768, 768]", primals_570: "f32[768]", primals_571: "f32[768]", primals_572: "f32[768]", primals_573: "f32[3072, 768]", primals_574: "f32[3072]", primals_575: "f32[768, 3072]", primals_576: "f32[768]", primals_577: "f32[768]", primals_578: "f32[768]", primals_579: "f32[2304, 768]", primals_580: "f32[2304]", primals_581: "f32[16, 16]", primals_582: "f32[16]", primals_583: "f32[16, 16]", primals_584: "f32[16]", primals_585: "f32[768, 768]", primals_586: "f32[768]", primals_587: "f32[768]", primals_588: "f32[768]", primals_589: "f32[3072, 768]", primals_590: "f32[3072]", primals_591: "f32[768, 3072]", primals_592: "f32[768]", primals_593: "f32[768]", primals_594: "f32[768]", primals_595: "f32[2304, 768]", primals_596: "f32[2304]", primals_597: "f32[16, 16]", primals_598: "f32[16]", primals_599: "f32[16, 16]", primals_600: "f32[16]", primals_601: "f32[768, 768]", primals_602: "f32[768]", primals_603: "f32[768]", primals_604: "f32[768]", primals_605: "f32[3072, 768]", primals_606: "f32[3072]", primals_607: "f32[768, 3072]", primals_608: "f32[768]", primals_609: "f32[768]", primals_610: "f32[768]", primals_611: "f32[2304, 768]", primals_612: "f32[2304]", primals_613: "f32[16, 16]", primals_614: "f32[16]", primals_615: "f32[16, 16]", primals_616: "f32[16]", primals_617: "f32[768, 768]", primals_618: "f32[768]", primals_619: "f32[768]", primals_620: "f32[768]", primals_621: "f32[3072, 768]", primals_622: "f32[3072]", primals_623: "f32[768, 3072]", primals_624: "f32[768]", primals_625: "f32[768]", primals_626: "f32[768]", primals_627: "f32[2304, 768]", primals_628: "f32[2304]", primals_629: "f32[16, 16]", primals_630: "f32[16]", primals_631: "f32[16, 16]", primals_632: "f32[16]", primals_633: "f32[768, 768]", primals_634: "f32[768]", primals_635: "f32[768]", primals_636: "f32[768]", primals_637: "f32[3072, 768]", primals_638: "f32[3072]", primals_639: "f32[768, 3072]", primals_640: "f32[768]", primals_641: "f32[768]", primals_642: "f32[768]", primals_643: "f32[2304, 768]", primals_644: "f32[2304]", primals_645: "f32[16, 16]", primals_646: "f32[16]", primals_647: "f32[16, 16]", primals_648: "f32[16]", primals_649: "f32[768, 768]", primals_650: "f32[768]", primals_651: "f32[768]", primals_652: "f32[768]", primals_653: "f32[3072, 768]", primals_654: "f32[3072]", primals_655: "f32[768, 3072]", primals_656: "f32[768]", primals_657: "f32[768]", primals_658: "f32[768]", primals_659: "f32[768, 768]", primals_660: "f32[768]", primals_661: "f32[768, 768]", primals_662: "f32[768]", primals_663: "f32[768, 768]", primals_664: "f32[768]", primals_665: "f32[768, 768]", primals_666: "f32[768]", primals_667: "f32[768]", primals_668: "f32[768]", primals_669: "f32[3072, 768]", primals_670: "f32[3072]", primals_671: "f32[768, 3072]", primals_672: "f32[768]", primals_673: "f32[768]", primals_674: "f32[768]", primals_675: "f32[768, 768]", primals_676: "f32[768]", primals_677: "f32[768, 768]", primals_678: "f32[768]", primals_679: "f32[768, 768]", primals_680: "f32[768]", primals_681: "f32[768, 768]", primals_682: "f32[768]", primals_683: "f32[768]", primals_684: "f32[768]", primals_685: "f32[3072, 768]", primals_686: "f32[3072]", primals_687: "f32[768, 3072]", primals_688: "f32[768]", primals_689: "f32[768]", primals_690: "f32[768]", primals_691: "f32[1000, 768]", primals_692: "f32[1000]", primals_693: "f32[8, 3, 384, 384]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 24, 24]" = torch.ops.aten.convolution.default(primals_693, primals_79, primals_80, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 576]" = torch.ops.aten.view.default(convolution, [8, 768, 576]);  convolution = None
    permute: "f32[8, 576, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:341, code: x = x + self.pos_embed
    add: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(permute, primals_1);  permute = primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:342, code: x = self.pos_drop(x)
    clone: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_1: "f32[8, 576, 768]" = torch.ops.aten.clone.default(clone, memory_format = torch.contiguous_format)
    var_mean = torch.ops.aten.var_mean.correction(clone_1, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 576, 1]" = var_mean[0]
    getitem_1: "f32[8, 576, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_1, getitem_1);  clone_1 = getitem_1 = None
    mul: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul, primals_81)
    add_2: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_82);  mul_1 = primals_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_1: "f32[4608, 768]" = torch.ops.aten.view.default(add_2, [4608, 768]);  add_2 = None
    permute_1: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    addmm: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_84, view_1, permute_1);  primals_84 = None
    view_2: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm, [8, 576, 2304]);  addmm = None
    view_3: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_2, [8, 576, 3, 16, 48]);  view_2 = None
    permute_2: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_2, 0, 0)
    mul_2: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select, 0.14433756729740643);  select = None
    select_1: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_2, 0, 1)
    select_2: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_2, 0, 2);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_3: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_1, [0, 1, 3, 2]);  select_1 = None
    expand: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_2, [8, 16, 576, 48]);  mul_2 = None
    clone_2: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_4: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_2, [128, 576, 48]);  clone_2 = None
    expand_1: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_3, [8, 16, 48, 576]);  permute_3 = None
    clone_3: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_5: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_3, [128, 48, 576]);  clone_3 = None
    bmm: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_4, view_5)
    view_6: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm, [8, 16, 576, 576]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_4: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_6, [0, 2, 3, 1]);  view_6 = None
    permute_5: "f32[16, 16]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    clone_4: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    view_7: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_4, [2654208, 16]);  clone_4 = None
    mm: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_7, permute_5)
    view_8: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm, [8, 576, 576, 16]);  mm = None
    add_3: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_8, primals_86);  view_8 = primals_86 = None
    permute_6: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_3, [0, 3, 1, 2]);  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_5: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    amax: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_5, [-1], True)
    sub_1: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_5, amax);  clone_5 = amax = None
    exp: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_7: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div, [0, 2, 3, 1]);  div = None
    permute_8: "f32[16, 16]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    clone_6: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_9: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_6, [2654208, 16]);  clone_6 = None
    mm_1: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_9, permute_8)
    view_10: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_1, [8, 576, 576, 16]);  mm_1 = None
    add_4: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_10, primals_88);  view_10 = primals_88 = None
    permute_9: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_4, [0, 3, 1, 2]);  add_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_7: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_9);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_2: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_7, [8, 16, 576, 576]);  clone_7 = None
    clone_8: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_11: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_8, [128, 576, 576]);  clone_8 = None
    expand_3: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_2, [8, 16, 576, 48]);  select_2 = None
    clone_9: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_12: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_9, [128, 576, 48]);  clone_9 = None
    bmm_1: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_11, view_12)
    view_13: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_1, [8, 16, 576, 48]);  bmm_1 = None
    permute_10: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
    clone_10: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
    view_14: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_10, [8, 576, 768]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_15: "f32[4608, 768]" = torch.ops.aten.view.default(view_14, [4608, 768]);  view_14 = None
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    addmm_1: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_90, view_15, permute_11);  primals_90 = None
    view_16: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_1, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_11: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_16);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_3: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_2, clone_11);  clone_11 = None
    add_5: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(clone, mul_3);  clone = mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_12: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_5, memory_format = torch.contiguous_format)
    var_mean_1 = torch.ops.aten.var_mean.correction(clone_12, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 576, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 576, 1]" = var_mean_1[1];  var_mean_1 = None
    add_6: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-06);  getitem_2 = None
    rsqrt_1: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
    sub_2: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_12, getitem_3);  clone_12 = getitem_3 = None
    mul_4: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_5: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_4, primals_91)
    add_7: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_5, primals_92);  mul_5 = primals_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_17: "f32[4608, 768]" = torch.ops.aten.view.default(add_7, [4608, 768]);  add_7 = None
    permute_12: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    addmm_2: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_94, view_17, permute_12);  primals_94 = None
    view_18: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_2, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_6: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_7: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
    erf: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_8: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_6, add_8);  mul_6 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_13: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_19: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_13, [4608, 3072]);  clone_13 = None
    permute_13: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_3: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_96, view_19, permute_13);  primals_96 = None
    view_20: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_3, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_14: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_9: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_3, clone_14);  clone_14 = None
    add_9: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_5, mul_9);  add_5 = mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_15: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_9, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_15, [2], correction = 0, keepdim = True)
    getitem_4: "f32[8, 576, 1]" = var_mean_2[0]
    getitem_5: "f32[8, 576, 1]" = var_mean_2[1];  var_mean_2 = None
    add_10: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-06);  getitem_4 = None
    rsqrt_2: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_3: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_15, getitem_5);  clone_15 = getitem_5 = None
    mul_10: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_11: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_10, primals_97)
    add_11: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_11, primals_98);  mul_11 = primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_21: "f32[4608, 768]" = torch.ops.aten.view.default(add_11, [4608, 768]);  add_11 = None
    permute_14: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_4: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_100, view_21, permute_14);  primals_100 = None
    view_22: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_4, [8, 576, 2304]);  addmm_4 = None
    view_23: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_22, [8, 576, 3, 16, 48]);  view_22 = None
    permute_15: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_23, [2, 0, 3, 1, 4]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_3: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_15, 0, 0)
    mul_12: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_3, 0.14433756729740643);  select_3 = None
    select_4: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_15, 0, 1)
    select_5: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_15, 0, 2);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_16: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_4, [0, 1, 3, 2]);  select_4 = None
    expand_4: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_12, [8, 16, 576, 48]);  mul_12 = None
    clone_16: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_24: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_16, [128, 576, 48]);  clone_16 = None
    expand_5: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_16, [8, 16, 48, 576]);  permute_16 = None
    clone_17: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_25: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_17, [128, 48, 576]);  clone_17 = None
    bmm_2: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_24, view_25)
    view_26: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_2, [8, 16, 576, 576]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_17: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_26, [0, 2, 3, 1]);  view_26 = None
    permute_18: "f32[16, 16]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    clone_18: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    view_27: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_18, [2654208, 16]);  clone_18 = None
    mm_2: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_27, permute_18)
    view_28: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_2, [8, 576, 576, 16]);  mm_2 = None
    add_12: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_28, primals_102);  view_28 = primals_102 = None
    permute_19: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_12, [0, 3, 1, 2]);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_19: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    amax_1: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_19, [-1], True)
    sub_4: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_19, amax_1);  clone_19 = amax_1 = None
    exp_1: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_20: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_1, [0, 2, 3, 1]);  div_1 = None
    permute_21: "f32[16, 16]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    clone_20: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_29: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_20, [2654208, 16]);  clone_20 = None
    mm_3: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_29, permute_21)
    view_30: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_3, [8, 576, 576, 16]);  mm_3 = None
    add_13: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_30, primals_104);  view_30 = primals_104 = None
    permute_22: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_13, [0, 3, 1, 2]);  add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_21: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_22);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_6: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_21, [8, 16, 576, 576]);  clone_21 = None
    clone_22: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_31: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_22, [128, 576, 576]);  clone_22 = None
    expand_7: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_5, [8, 16, 576, 48]);  select_5 = None
    clone_23: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_32: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_23, [128, 576, 48]);  clone_23 = None
    bmm_3: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_31, view_32)
    view_33: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_3, [8, 16, 576, 48]);  bmm_3 = None
    permute_23: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    clone_24: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_34: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_24, [8, 576, 768]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_35: "f32[4608, 768]" = torch.ops.aten.view.default(view_34, [4608, 768]);  view_34 = None
    permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_5: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_106, view_35, permute_24);  primals_106 = None
    view_36: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_5, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_25: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_13: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_4, clone_25);  clone_25 = None
    add_14: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_9, mul_13);  add_9 = mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_26: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_14, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_26, [2], correction = 0, keepdim = True)
    getitem_6: "f32[8, 576, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 576, 1]" = var_mean_3[1];  var_mean_3 = None
    add_15: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_3: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_5: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_26, getitem_7);  clone_26 = getitem_7 = None
    mul_14: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_15: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_107)
    add_16: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_15, primals_108);  mul_15 = primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_37: "f32[4608, 768]" = torch.ops.aten.view.default(add_16, [4608, 768]);  add_16 = None
    permute_25: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    addmm_6: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_110, view_37, permute_25);  primals_110 = None
    view_38: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_6, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_16: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_17: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_1: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
    add_17: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_18: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_16, add_17);  mul_16 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_27: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_18);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_39: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_27, [4608, 3072]);  clone_27 = None
    permute_26: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_7: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_112, view_39, permute_26);  primals_112 = None
    view_40: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_7, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_28: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_19: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_5, clone_28);  clone_28 = None
    add_18: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_14, mul_19);  add_14 = mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_29: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_18, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_29, [2], correction = 0, keepdim = True)
    getitem_8: "f32[8, 576, 1]" = var_mean_4[0]
    getitem_9: "f32[8, 576, 1]" = var_mean_4[1];  var_mean_4 = None
    add_19: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-06);  getitem_8 = None
    rsqrt_4: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_6: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_29, getitem_9);  clone_29 = getitem_9 = None
    mul_20: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_21: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_20, primals_113)
    add_20: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_21, primals_114);  mul_21 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_41: "f32[4608, 768]" = torch.ops.aten.view.default(add_20, [4608, 768]);  add_20 = None
    permute_27: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_8: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_116, view_41, permute_27);  primals_116 = None
    view_42: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_8, [8, 576, 2304]);  addmm_8 = None
    view_43: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_42, [8, 576, 3, 16, 48]);  view_42 = None
    permute_28: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_43, [2, 0, 3, 1, 4]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_6: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_28, 0, 0)
    mul_22: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_6, 0.14433756729740643);  select_6 = None
    select_7: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_28, 0, 1)
    select_8: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_28, 0, 2);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_29: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_7, [0, 1, 3, 2]);  select_7 = None
    expand_8: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_22, [8, 16, 576, 48]);  mul_22 = None
    clone_30: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_44: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_30, [128, 576, 48]);  clone_30 = None
    expand_9: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_29, [8, 16, 48, 576]);  permute_29 = None
    clone_31: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_45: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_31, [128, 48, 576]);  clone_31 = None
    bmm_4: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_44, view_45)
    view_46: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_4, [8, 16, 576, 576]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_30: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_46, [0, 2, 3, 1]);  view_46 = None
    permute_31: "f32[16, 16]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    clone_32: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    view_47: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_32, [2654208, 16]);  clone_32 = None
    mm_4: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_47, permute_31)
    view_48: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_4, [8, 576, 576, 16]);  mm_4 = None
    add_21: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_48, primals_118);  view_48 = primals_118 = None
    permute_32: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_21, [0, 3, 1, 2]);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_33: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    amax_2: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_33, [-1], True)
    sub_7: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_33, amax_2);  clone_33 = amax_2 = None
    exp_2: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_33: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_2, [0, 2, 3, 1]);  div_2 = None
    permute_34: "f32[16, 16]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    clone_34: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
    view_49: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_34, [2654208, 16]);  clone_34 = None
    mm_5: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_49, permute_34)
    view_50: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_5, [8, 576, 576, 16]);  mm_5 = None
    add_22: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_50, primals_120);  view_50 = primals_120 = None
    permute_35: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_22, [0, 3, 1, 2]);  add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_35: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_35);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_10: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_35, [8, 16, 576, 576]);  clone_35 = None
    clone_36: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_51: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_36, [128, 576, 576]);  clone_36 = None
    expand_11: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_8, [8, 16, 576, 48]);  select_8 = None
    clone_37: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_52: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_37, [128, 576, 48]);  clone_37 = None
    bmm_5: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_51, view_52)
    view_53: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_5, [8, 16, 576, 48]);  bmm_5 = None
    permute_36: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    clone_38: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
    view_54: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_38, [8, 576, 768]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_55: "f32[4608, 768]" = torch.ops.aten.view.default(view_54, [4608, 768]);  view_54 = None
    permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_9: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_122, view_55, permute_37);  primals_122 = None
    view_56: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_9, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_39: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_56);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_23: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_6, clone_39);  clone_39 = None
    add_23: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_18, mul_23);  add_18 = mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_40: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_23, memory_format = torch.contiguous_format)
    var_mean_5 = torch.ops.aten.var_mean.correction(clone_40, [2], correction = 0, keepdim = True)
    getitem_10: "f32[8, 576, 1]" = var_mean_5[0]
    getitem_11: "f32[8, 576, 1]" = var_mean_5[1];  var_mean_5 = None
    add_24: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-06);  getitem_10 = None
    rsqrt_5: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_8: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_40, getitem_11);  clone_40 = getitem_11 = None
    mul_24: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_25: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_24, primals_123)
    add_25: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_25, primals_124);  mul_25 = primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_57: "f32[4608, 768]" = torch.ops.aten.view.default(add_25, [4608, 768]);  add_25 = None
    permute_38: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_10: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_126, view_57, permute_38);  primals_126 = None
    view_58: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_26: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_27: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
    erf_2: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_26: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_28: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_26, add_26);  mul_26 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_41: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_28);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_59: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_41, [4608, 3072]);  clone_41 = None
    permute_39: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    addmm_11: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_128, view_59, permute_39);  primals_128 = None
    view_60: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_11, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_42: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_29: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_7, clone_42);  clone_42 = None
    add_27: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_23, mul_29);  add_23 = mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_43: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_27, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_43, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 576, 1]" = var_mean_6[0]
    getitem_13: "f32[8, 576, 1]" = var_mean_6[1];  var_mean_6 = None
    add_28: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_6: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_9: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_43, getitem_13);  clone_43 = getitem_13 = None
    mul_30: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_31: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_129)
    add_29: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_130);  mul_31 = primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_61: "f32[4608, 768]" = torch.ops.aten.view.default(add_29, [4608, 768]);  add_29 = None
    permute_40: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_12: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_132, view_61, permute_40);  primals_132 = None
    view_62: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_12, [8, 576, 2304]);  addmm_12 = None
    view_63: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_62, [8, 576, 3, 16, 48]);  view_62 = None
    permute_41: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_9: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_41, 0, 0)
    mul_32: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_9, 0.14433756729740643);  select_9 = None
    select_10: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_41, 0, 1)
    select_11: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_41, 0, 2);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_42: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_10, [0, 1, 3, 2]);  select_10 = None
    expand_12: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_32, [8, 16, 576, 48]);  mul_32 = None
    clone_44: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_64: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_44, [128, 576, 48]);  clone_44 = None
    expand_13: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_42, [8, 16, 48, 576]);  permute_42 = None
    clone_45: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_65: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_45, [128, 48, 576]);  clone_45 = None
    bmm_6: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_64, view_65)
    view_66: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_6, [8, 16, 576, 576]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_43: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_66, [0, 2, 3, 1]);  view_66 = None
    permute_44: "f32[16, 16]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    clone_46: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    view_67: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_46, [2654208, 16]);  clone_46 = None
    mm_6: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_67, permute_44)
    view_68: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_6, [8, 576, 576, 16]);  mm_6 = None
    add_30: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_68, primals_134);  view_68 = primals_134 = None
    permute_45: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_30, [0, 3, 1, 2]);  add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_47: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    amax_3: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_47, [-1], True)
    sub_10: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_47, amax_3);  clone_47 = amax_3 = None
    exp_3: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_46: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_3, [0, 2, 3, 1]);  div_3 = None
    permute_47: "f32[16, 16]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    clone_48: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    view_69: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_48, [2654208, 16]);  clone_48 = None
    mm_7: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_69, permute_47)
    view_70: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_7, [8, 576, 576, 16]);  mm_7 = None
    add_31: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_70, primals_136);  view_70 = primals_136 = None
    permute_48: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_31, [0, 3, 1, 2]);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_49: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_48);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_14: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_49, [8, 16, 576, 576]);  clone_49 = None
    clone_50: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_71: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_50, [128, 576, 576]);  clone_50 = None
    expand_15: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_11, [8, 16, 576, 48]);  select_11 = None
    clone_51: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_72: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_51, [128, 576, 48]);  clone_51 = None
    bmm_7: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_71, view_72)
    view_73: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_7, [8, 16, 576, 48]);  bmm_7 = None
    permute_49: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    clone_52: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    view_74: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_52, [8, 576, 768]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_75: "f32[4608, 768]" = torch.ops.aten.view.default(view_74, [4608, 768]);  view_74 = None
    permute_50: "f32[768, 768]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_13: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_138, view_75, permute_50);  primals_138 = None
    view_76: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_13, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_53: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_76);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_33: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_8, clone_53);  clone_53 = None
    add_32: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_27, mul_33);  add_27 = mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_54: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_32, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_54, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 576, 1]" = var_mean_7[0]
    getitem_15: "f32[8, 576, 1]" = var_mean_7[1];  var_mean_7 = None
    add_33: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_7: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_11: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_54, getitem_15);  clone_54 = getitem_15 = None
    mul_34: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_35: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_34, primals_139)
    add_34: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_35, primals_140);  mul_35 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_77: "f32[4608, 768]" = torch.ops.aten.view.default(add_34, [4608, 768]);  add_34 = None
    permute_51: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_14: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_142, view_77, permute_51);  primals_142 = None
    view_78: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_14, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_36: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_37: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
    erf_3: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_37);  mul_37 = None
    add_35: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_38: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_36, add_35);  mul_36 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_55: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_38);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_79: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_55, [4608, 3072]);  clone_55 = None
    permute_52: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_15: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_144, view_79, permute_52);  primals_144 = None
    view_80: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_15, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_56: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_39: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_9, clone_56);  clone_56 = None
    add_36: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_32, mul_39);  add_32 = mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_57: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_36, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_57, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 576, 1]" = var_mean_8[0]
    getitem_17: "f32[8, 576, 1]" = var_mean_8[1];  var_mean_8 = None
    add_37: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_8: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_12: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_57, getitem_17);  clone_57 = getitem_17 = None
    mul_40: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_41: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_40, primals_145)
    add_38: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_41, primals_146);  mul_41 = primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_81: "f32[4608, 768]" = torch.ops.aten.view.default(add_38, [4608, 768]);  add_38 = None
    permute_53: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm_16: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_148, view_81, permute_53);  primals_148 = None
    view_82: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_16, [8, 576, 2304]);  addmm_16 = None
    view_83: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_82, [8, 576, 3, 16, 48]);  view_82 = None
    permute_54: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_83, [2, 0, 3, 1, 4]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_12: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_54, 0, 0)
    mul_42: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_12, 0.14433756729740643);  select_12 = None
    select_13: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_54, 0, 1)
    select_14: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_54, 0, 2);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_55: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_13, [0, 1, 3, 2]);  select_13 = None
    expand_16: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_42, [8, 16, 576, 48]);  mul_42 = None
    clone_58: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_84: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_58, [128, 576, 48]);  clone_58 = None
    expand_17: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_55, [8, 16, 48, 576]);  permute_55 = None
    clone_59: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_85: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_59, [128, 48, 576]);  clone_59 = None
    bmm_8: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_84, view_85)
    view_86: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_8, [8, 16, 576, 576]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_56: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_86, [0, 2, 3, 1]);  view_86 = None
    permute_57: "f32[16, 16]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    clone_60: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
    view_87: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_60, [2654208, 16]);  clone_60 = None
    mm_8: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_87, permute_57)
    view_88: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_8, [8, 576, 576, 16]);  mm_8 = None
    add_39: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_88, primals_150);  view_88 = primals_150 = None
    permute_58: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_39, [0, 3, 1, 2]);  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_61: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
    amax_4: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_61, [-1], True)
    sub_13: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_61, amax_4);  clone_61 = amax_4 = None
    exp_4: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_59: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_4, [0, 2, 3, 1]);  div_4 = None
    permute_60: "f32[16, 16]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    clone_62: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    view_89: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_62, [2654208, 16]);  clone_62 = None
    mm_9: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_89, permute_60)
    view_90: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_9, [8, 576, 576, 16]);  mm_9 = None
    add_40: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_90, primals_152);  view_90 = primals_152 = None
    permute_61: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_40, [0, 3, 1, 2]);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_63: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_61);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_18: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_63, [8, 16, 576, 576]);  clone_63 = None
    clone_64: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_91: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_64, [128, 576, 576]);  clone_64 = None
    expand_19: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_14, [8, 16, 576, 48]);  select_14 = None
    clone_65: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_92: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_65, [128, 576, 48]);  clone_65 = None
    bmm_9: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_91, view_92)
    view_93: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_9, [8, 16, 576, 48]);  bmm_9 = None
    permute_62: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    clone_66: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_94: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_66, [8, 576, 768]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_95: "f32[4608, 768]" = torch.ops.aten.view.default(view_94, [4608, 768]);  view_94 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    addmm_17: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_154, view_95, permute_63);  primals_154 = None
    view_96: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_17, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_67: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_43: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_10, clone_67);  clone_67 = None
    add_41: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_36, mul_43);  add_36 = mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_68: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_41, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_68, [2], correction = 0, keepdim = True)
    getitem_18: "f32[8, 576, 1]" = var_mean_9[0]
    getitem_19: "f32[8, 576, 1]" = var_mean_9[1];  var_mean_9 = None
    add_42: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-06);  getitem_18 = None
    rsqrt_9: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_14: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_68, getitem_19);  clone_68 = getitem_19 = None
    mul_44: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_45: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_155)
    add_43: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_156);  mul_45 = primals_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_97: "f32[4608, 768]" = torch.ops.aten.view.default(add_43, [4608, 768]);  add_43 = None
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm_18: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_158, view_97, permute_64);  primals_158 = None
    view_98: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_18, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_46: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_47: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
    erf_4: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_44: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_48: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_44);  mul_46 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_69: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_99: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_69, [4608, 3072]);  clone_69 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    addmm_19: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_160, view_99, permute_65);  primals_160 = None
    view_100: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_19, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_70: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_49: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_11, clone_70);  clone_70 = None
    add_45: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_41, mul_49);  add_41 = mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_71: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_45, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_71, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 576, 1]" = var_mean_10[0]
    getitem_21: "f32[8, 576, 1]" = var_mean_10[1];  var_mean_10 = None
    add_46: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_10: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_15: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_71, getitem_21);  clone_71 = getitem_21 = None
    mul_50: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_51: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_161)
    add_47: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_162);  mul_51 = primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_101: "f32[4608, 768]" = torch.ops.aten.view.default(add_47, [4608, 768]);  add_47 = None
    permute_66: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    addmm_20: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_164, view_101, permute_66);  primals_164 = None
    view_102: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_20, [8, 576, 2304]);  addmm_20 = None
    view_103: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_102, [8, 576, 3, 16, 48]);  view_102 = None
    permute_67: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_103, [2, 0, 3, 1, 4]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_15: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_67, 0, 0)
    mul_52: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_15, 0.14433756729740643);  select_15 = None
    select_16: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_67, 0, 1)
    select_17: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_67, 0, 2);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_68: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_16, [0, 1, 3, 2]);  select_16 = None
    expand_20: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_52, [8, 16, 576, 48]);  mul_52 = None
    clone_72: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_104: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_72, [128, 576, 48]);  clone_72 = None
    expand_21: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_68, [8, 16, 48, 576]);  permute_68 = None
    clone_73: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_105: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_73, [128, 48, 576]);  clone_73 = None
    bmm_10: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_104, view_105)
    view_106: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_10, [8, 16, 576, 576]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_69: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_106, [0, 2, 3, 1]);  view_106 = None
    permute_70: "f32[16, 16]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    clone_74: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    view_107: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_74, [2654208, 16]);  clone_74 = None
    mm_10: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_107, permute_70)
    view_108: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_10, [8, 576, 576, 16]);  mm_10 = None
    add_48: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_108, primals_166);  view_108 = primals_166 = None
    permute_71: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_48, [0, 3, 1, 2]);  add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_75: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    amax_5: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_75, [-1], True)
    sub_16: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_75, amax_5);  clone_75 = amax_5 = None
    exp_5: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_72: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_5, [0, 2, 3, 1]);  div_5 = None
    permute_73: "f32[16, 16]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
    clone_76: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
    view_109: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_76, [2654208, 16]);  clone_76 = None
    mm_11: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_109, permute_73)
    view_110: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_11, [8, 576, 576, 16]);  mm_11 = None
    add_49: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_110, primals_168);  view_110 = primals_168 = None
    permute_74: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_49, [0, 3, 1, 2]);  add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_77: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_74);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_22: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_77, [8, 16, 576, 576]);  clone_77 = None
    clone_78: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    view_111: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_78, [128, 576, 576]);  clone_78 = None
    expand_23: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_17, [8, 16, 576, 48]);  select_17 = None
    clone_79: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_112: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_79, [128, 576, 48]);  clone_79 = None
    bmm_11: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_111, view_112)
    view_113: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_11, [8, 16, 576, 48]);  bmm_11 = None
    permute_75: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    clone_80: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
    view_114: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_80, [8, 576, 768]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_115: "f32[4608, 768]" = torch.ops.aten.view.default(view_114, [4608, 768]);  view_114 = None
    permute_76: "f32[768, 768]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    addmm_21: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_170, view_115, permute_76);  primals_170 = None
    view_116: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_21, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_81: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_116);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_53: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_12, clone_81);  clone_81 = None
    add_50: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_45, mul_53);  add_45 = mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_82: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_50, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_82, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 576, 1]" = var_mean_11[0]
    getitem_23: "f32[8, 576, 1]" = var_mean_11[1];  var_mean_11 = None
    add_51: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_11: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_17: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_82, getitem_23);  clone_82 = getitem_23 = None
    mul_54: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_55: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_54, primals_171)
    add_52: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_55, primals_172);  mul_55 = primals_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_117: "f32[4608, 768]" = torch.ops.aten.view.default(add_52, [4608, 768]);  add_52 = None
    permute_77: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
    addmm_22: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_174, view_117, permute_77);  primals_174 = None
    view_118: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_56: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_57: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
    erf_5: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
    add_53: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_58: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_56, add_53);  mul_56 = add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_83: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_58);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_119: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_83, [4608, 3072]);  clone_83 = None
    permute_78: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_23: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_176, view_119, permute_78);  primals_176 = None
    view_120: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_23, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_84: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_59: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_13, clone_84);  clone_84 = None
    add_54: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_50, mul_59);  add_50 = mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_85: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_54, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
    getitem_24: "f32[8, 576, 1]" = var_mean_12[0]
    getitem_25: "f32[8, 576, 1]" = var_mean_12[1];  var_mean_12 = None
    add_55: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-06);  getitem_24 = None
    rsqrt_12: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_18: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_85, getitem_25);  clone_85 = getitem_25 = None
    mul_60: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_61: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_60, primals_177)
    add_56: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_178);  mul_61 = primals_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_121: "f32[4608, 768]" = torch.ops.aten.view.default(add_56, [4608, 768]);  add_56 = None
    permute_79: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    addmm_24: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_180, view_121, permute_79);  primals_180 = None
    view_122: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_24, [8, 576, 2304]);  addmm_24 = None
    view_123: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_122, [8, 576, 3, 16, 48]);  view_122 = None
    permute_80: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_123, [2, 0, 3, 1, 4]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_18: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_80, 0, 0)
    mul_62: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_18, 0.14433756729740643);  select_18 = None
    select_19: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_80, 0, 1)
    select_20: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_80, 0, 2);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_81: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_19, [0, 1, 3, 2]);  select_19 = None
    expand_24: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_62, [8, 16, 576, 48]);  mul_62 = None
    clone_86: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_124: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_86, [128, 576, 48]);  clone_86 = None
    expand_25: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_81, [8, 16, 48, 576]);  permute_81 = None
    clone_87: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_125: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_87, [128, 48, 576]);  clone_87 = None
    bmm_12: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_124, view_125)
    view_126: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_12, [8, 16, 576, 576]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_82: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_126, [0, 2, 3, 1]);  view_126 = None
    permute_83: "f32[16, 16]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    clone_88: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_127: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_88, [2654208, 16]);  clone_88 = None
    mm_12: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_127, permute_83)
    view_128: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_12, [8, 576, 576, 16]);  mm_12 = None
    add_57: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_128, primals_182);  view_128 = primals_182 = None
    permute_84: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_57, [0, 3, 1, 2]);  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_89: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    amax_6: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_89, [-1], True)
    sub_19: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_89, amax_6);  clone_89 = amax_6 = None
    exp_6: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_85: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_6, [0, 2, 3, 1]);  div_6 = None
    permute_86: "f32[16, 16]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    clone_90: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_129: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_90, [2654208, 16]);  clone_90 = None
    mm_13: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_129, permute_86)
    view_130: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_13, [8, 576, 576, 16]);  mm_13 = None
    add_58: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_130, primals_184);  view_130 = primals_184 = None
    permute_87: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_58, [0, 3, 1, 2]);  add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_91: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_87);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_26: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_91, [8, 16, 576, 576]);  clone_91 = None
    clone_92: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_131: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_92, [128, 576, 576]);  clone_92 = None
    expand_27: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_20, [8, 16, 576, 48]);  select_20 = None
    clone_93: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_132: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_93, [128, 576, 48]);  clone_93 = None
    bmm_13: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_131, view_132)
    view_133: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_13, [8, 16, 576, 48]);  bmm_13 = None
    permute_88: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    clone_94: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
    view_134: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_94, [8, 576, 768]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_135: "f32[4608, 768]" = torch.ops.aten.view.default(view_134, [4608, 768]);  view_134 = None
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    addmm_25: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_186, view_135, permute_89);  primals_186 = None
    view_136: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_25, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_95: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_136);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_63: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_14, clone_95);  clone_95 = None
    add_59: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_54, mul_63);  add_54 = mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_96: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_59, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_96, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 576, 1]" = var_mean_13[0]
    getitem_27: "f32[8, 576, 1]" = var_mean_13[1];  var_mean_13 = None
    add_60: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_13: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_20: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_96, getitem_27);  clone_96 = getitem_27 = None
    mul_64: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
    mul_65: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_64, primals_187)
    add_61: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_65, primals_188);  mul_65 = primals_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_137: "f32[4608, 768]" = torch.ops.aten.view.default(add_61, [4608, 768]);  add_61 = None
    permute_90: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    addmm_26: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_190, view_137, permute_90);  primals_190 = None
    view_138: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_26, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_66: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_138, 0.5)
    mul_67: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_138, 0.7071067811865476);  view_138 = None
    erf_6: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_67);  mul_67 = None
    add_62: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_68: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_66, add_62);  mul_66 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_97: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_68);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_139: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_97, [4608, 3072]);  clone_97 = None
    permute_91: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
    addmm_27: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_192, view_139, permute_91);  primals_192 = None
    view_140: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_27, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_98: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_69: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_15, clone_98);  clone_98 = None
    add_63: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_59, mul_69);  add_59 = mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_99: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_63, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_99, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 576, 1]" = var_mean_14[0]
    getitem_29: "f32[8, 576, 1]" = var_mean_14[1];  var_mean_14 = None
    add_64: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_14: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_21: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_99, getitem_29);  clone_99 = getitem_29 = None
    mul_70: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
    mul_71: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_70, primals_193)
    add_65: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_71, primals_194);  mul_71 = primals_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_141: "f32[4608, 768]" = torch.ops.aten.view.default(add_65, [4608, 768]);  add_65 = None
    permute_92: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    addmm_28: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_196, view_141, permute_92);  primals_196 = None
    view_142: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_28, [8, 576, 2304]);  addmm_28 = None
    view_143: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_142, [8, 576, 3, 16, 48]);  view_142 = None
    permute_93: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_143, [2, 0, 3, 1, 4]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_21: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_93, 0, 0)
    mul_72: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_21, 0.14433756729740643);  select_21 = None
    select_22: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_93, 0, 1)
    select_23: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_93, 0, 2);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_94: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_22, [0, 1, 3, 2]);  select_22 = None
    expand_28: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_72, [8, 16, 576, 48]);  mul_72 = None
    clone_100: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_144: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_100, [128, 576, 48]);  clone_100 = None
    expand_29: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_94, [8, 16, 48, 576]);  permute_94 = None
    clone_101: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_145: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_101, [128, 48, 576]);  clone_101 = None
    bmm_14: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_144, view_145)
    view_146: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_14, [8, 16, 576, 576]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_95: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_146, [0, 2, 3, 1]);  view_146 = None
    permute_96: "f32[16, 16]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
    clone_102: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_147: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_102, [2654208, 16]);  clone_102 = None
    mm_14: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_147, permute_96)
    view_148: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_14, [8, 576, 576, 16]);  mm_14 = None
    add_66: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_148, primals_198);  view_148 = primals_198 = None
    permute_97: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_66, [0, 3, 1, 2]);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_103: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
    amax_7: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_103, [-1], True)
    sub_22: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_103, amax_7);  clone_103 = amax_7 = None
    exp_7: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_98: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_7, [0, 2, 3, 1]);  div_7 = None
    permute_99: "f32[16, 16]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    clone_104: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_98, memory_format = torch.contiguous_format);  permute_98 = None
    view_149: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_104, [2654208, 16]);  clone_104 = None
    mm_15: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_149, permute_99)
    view_150: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_15, [8, 576, 576, 16]);  mm_15 = None
    add_67: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_150, primals_200);  view_150 = primals_200 = None
    permute_100: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_67, [0, 3, 1, 2]);  add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_105: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_100);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_30: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_105, [8, 16, 576, 576]);  clone_105 = None
    clone_106: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_151: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_106, [128, 576, 576]);  clone_106 = None
    expand_31: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_23, [8, 16, 576, 48]);  select_23 = None
    clone_107: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_152: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_107, [128, 576, 48]);  clone_107 = None
    bmm_15: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_151, view_152)
    view_153: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_15, [8, 16, 576, 48]);  bmm_15 = None
    permute_101: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
    clone_108: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    view_154: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_108, [8, 576, 768]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_155: "f32[4608, 768]" = torch.ops.aten.view.default(view_154, [4608, 768]);  view_154 = None
    permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
    addmm_29: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_202, view_155, permute_102);  primals_202 = None
    view_156: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_29, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_109: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_156);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_73: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_16, clone_109);  clone_109 = None
    add_68: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_63, mul_73);  add_63 = mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_110: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_68, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_110, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 576, 1]" = var_mean_15[0]
    getitem_31: "f32[8, 576, 1]" = var_mean_15[1];  var_mean_15 = None
    add_69: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_15: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_23: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_110, getitem_31);  clone_110 = getitem_31 = None
    mul_74: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
    mul_75: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_74, primals_203)
    add_70: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_75, primals_204);  mul_75 = primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_157: "f32[4608, 768]" = torch.ops.aten.view.default(add_70, [4608, 768]);  add_70 = None
    permute_103: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    addmm_30: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_206, view_157, permute_103);  primals_206 = None
    view_158: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_30, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_76: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
    mul_77: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
    erf_7: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
    add_71: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_78: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_76, add_71);  mul_76 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_111: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_78);  mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_159: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_111, [4608, 3072]);  clone_111 = None
    permute_104: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
    addmm_31: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_208, view_159, permute_104);  primals_208 = None
    view_160: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_31, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_112: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_160);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_79: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_17, clone_112);  clone_112 = None
    add_72: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_68, mul_79);  add_68 = mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_113: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_72, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_113, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 576, 1]" = var_mean_16[0]
    getitem_33: "f32[8, 576, 1]" = var_mean_16[1];  var_mean_16 = None
    add_73: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_16: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_24: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_113, getitem_33);  clone_113 = getitem_33 = None
    mul_80: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
    mul_81: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_80, primals_209)
    add_74: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_81, primals_210);  mul_81 = primals_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_161: "f32[4608, 768]" = torch.ops.aten.view.default(add_74, [4608, 768]);  add_74 = None
    permute_105: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
    addmm_32: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_212, view_161, permute_105);  primals_212 = None
    view_162: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_32, [8, 576, 2304]);  addmm_32 = None
    view_163: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_162, [8, 576, 3, 16, 48]);  view_162 = None
    permute_106: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_163, [2, 0, 3, 1, 4]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_24: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_106, 0, 0)
    mul_82: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_24, 0.14433756729740643);  select_24 = None
    select_25: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_106, 0, 1)
    select_26: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_106, 0, 2);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_107: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_25, [0, 1, 3, 2]);  select_25 = None
    expand_32: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_82, [8, 16, 576, 48]);  mul_82 = None
    clone_114: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_164: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_114, [128, 576, 48]);  clone_114 = None
    expand_33: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_107, [8, 16, 48, 576]);  permute_107 = None
    clone_115: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_165: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_115, [128, 48, 576]);  clone_115 = None
    bmm_16: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_164, view_165)
    view_166: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_16, [8, 16, 576, 576]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_108: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_166, [0, 2, 3, 1]);  view_166 = None
    permute_109: "f32[16, 16]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
    clone_116: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
    view_167: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_116, [2654208, 16]);  clone_116 = None
    mm_16: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_167, permute_109)
    view_168: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_16, [8, 576, 576, 16]);  mm_16 = None
    add_75: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_168, primals_214);  view_168 = primals_214 = None
    permute_110: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_75, [0, 3, 1, 2]);  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_117: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
    amax_8: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_117, [-1], True)
    sub_25: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_117, amax_8);  clone_117 = amax_8 = None
    exp_8: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_111: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_8, [0, 2, 3, 1]);  div_8 = None
    permute_112: "f32[16, 16]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    clone_118: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
    view_169: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_118, [2654208, 16]);  clone_118 = None
    mm_17: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_169, permute_112)
    view_170: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_17, [8, 576, 576, 16]);  mm_17 = None
    add_76: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_170, primals_216);  view_170 = primals_216 = None
    permute_113: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_76, [0, 3, 1, 2]);  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_119: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_113);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_34: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_119, [8, 16, 576, 576]);  clone_119 = None
    clone_120: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_171: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_120, [128, 576, 576]);  clone_120 = None
    expand_35: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_26, [8, 16, 576, 48]);  select_26 = None
    clone_121: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_172: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_121, [128, 576, 48]);  clone_121 = None
    bmm_17: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_171, view_172)
    view_173: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_17, [8, 16, 576, 48]);  bmm_17 = None
    permute_114: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
    clone_122: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_174: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_122, [8, 576, 768]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_175: "f32[4608, 768]" = torch.ops.aten.view.default(view_174, [4608, 768]);  view_174 = None
    permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
    addmm_33: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_218, view_175, permute_115);  primals_218 = None
    view_176: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_33, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_123: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_176);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_83: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_18, clone_123);  clone_123 = None
    add_77: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_72, mul_83);  add_72 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_124: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_77, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_124, [2], correction = 0, keepdim = True)
    getitem_34: "f32[8, 576, 1]" = var_mean_17[0]
    getitem_35: "f32[8, 576, 1]" = var_mean_17[1];  var_mean_17 = None
    add_78: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-06);  getitem_34 = None
    rsqrt_17: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_26: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_124, getitem_35);  clone_124 = getitem_35 = None
    mul_84: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
    mul_85: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_84, primals_219)
    add_79: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_85, primals_220);  mul_85 = primals_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_177: "f32[4608, 768]" = torch.ops.aten.view.default(add_79, [4608, 768]);  add_79 = None
    permute_116: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
    addmm_34: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_222, view_177, permute_116);  primals_222 = None
    view_178: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
    mul_87: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.7071067811865476);  view_178 = None
    erf_8: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_80: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_88: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_86, add_80);  mul_86 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_125: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_179: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_125, [4608, 3072]);  clone_125 = None
    permute_117: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
    addmm_35: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_224, view_179, permute_117);  primals_224 = None
    view_180: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_35, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_126: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_89: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_19, clone_126);  clone_126 = None
    add_81: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_77, mul_89);  add_77 = mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_127: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_81, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_127, [2], correction = 0, keepdim = True)
    getitem_36: "f32[8, 576, 1]" = var_mean_18[0]
    getitem_37: "f32[8, 576, 1]" = var_mean_18[1];  var_mean_18 = None
    add_82: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-06);  getitem_36 = None
    rsqrt_18: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_27: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_127, getitem_37);  clone_127 = getitem_37 = None
    mul_90: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
    mul_91: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_90, primals_225)
    add_83: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_91, primals_226);  mul_91 = primals_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_181: "f32[4608, 768]" = torch.ops.aten.view.default(add_83, [4608, 768]);  add_83 = None
    permute_118: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
    addmm_36: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_228, view_181, permute_118);  primals_228 = None
    view_182: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_36, [8, 576, 2304]);  addmm_36 = None
    view_183: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_182, [8, 576, 3, 16, 48]);  view_182 = None
    permute_119: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_183, [2, 0, 3, 1, 4]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_27: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_119, 0, 0)
    mul_92: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_27, 0.14433756729740643);  select_27 = None
    select_28: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_119, 0, 1)
    select_29: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_119, 0, 2);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_120: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_28, [0, 1, 3, 2]);  select_28 = None
    expand_36: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_92, [8, 16, 576, 48]);  mul_92 = None
    clone_128: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_184: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_128, [128, 576, 48]);  clone_128 = None
    expand_37: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_120, [8, 16, 48, 576]);  permute_120 = None
    clone_129: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_185: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_129, [128, 48, 576]);  clone_129 = None
    bmm_18: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_184, view_185)
    view_186: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_18, [8, 16, 576, 576]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_121: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_186, [0, 2, 3, 1]);  view_186 = None
    permute_122: "f32[16, 16]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
    clone_130: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_187: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_130, [2654208, 16]);  clone_130 = None
    mm_18: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_187, permute_122)
    view_188: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_18, [8, 576, 576, 16]);  mm_18 = None
    add_84: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_188, primals_230);  view_188 = primals_230 = None
    permute_123: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_84, [0, 3, 1, 2]);  add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_131: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    amax_9: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_131, [-1], True)
    sub_28: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_131, amax_9);  clone_131 = amax_9 = None
    exp_9: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_124: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_9, [0, 2, 3, 1]);  div_9 = None
    permute_125: "f32[16, 16]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
    clone_132: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    view_189: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_132, [2654208, 16]);  clone_132 = None
    mm_19: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_189, permute_125)
    view_190: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_19, [8, 576, 576, 16]);  mm_19 = None
    add_85: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_190, primals_232);  view_190 = primals_232 = None
    permute_126: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_85, [0, 3, 1, 2]);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_133: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_126);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_38: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_133, [8, 16, 576, 576]);  clone_133 = None
    clone_134: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_191: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_134, [128, 576, 576]);  clone_134 = None
    expand_39: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_29, [8, 16, 576, 48]);  select_29 = None
    clone_135: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_192: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_135, [128, 576, 48]);  clone_135 = None
    bmm_19: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_191, view_192)
    view_193: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_19, [8, 16, 576, 48]);  bmm_19 = None
    permute_127: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
    clone_136: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    view_194: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_136, [8, 576, 768]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_195: "f32[4608, 768]" = torch.ops.aten.view.default(view_194, [4608, 768]);  view_194 = None
    permute_128: "f32[768, 768]" = torch.ops.aten.permute.default(primals_233, [1, 0]);  primals_233 = None
    addmm_37: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_234, view_195, permute_128);  primals_234 = None
    view_196: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_37, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_137: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_196);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_93: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_20, clone_137);  clone_137 = None
    add_86: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_81, mul_93);  add_81 = mul_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_138: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_86, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_138, [2], correction = 0, keepdim = True)
    getitem_38: "f32[8, 576, 1]" = var_mean_19[0]
    getitem_39: "f32[8, 576, 1]" = var_mean_19[1];  var_mean_19 = None
    add_87: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-06);  getitem_38 = None
    rsqrt_19: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_29: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_138, getitem_39);  clone_138 = getitem_39 = None
    mul_94: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
    mul_95: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_94, primals_235)
    add_88: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_95, primals_236);  mul_95 = primals_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_197: "f32[4608, 768]" = torch.ops.aten.view.default(add_88, [4608, 768]);  add_88 = None
    permute_129: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
    addmm_38: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_238, view_197, permute_129);  primals_238 = None
    view_198: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_38, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_96: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_198, 0.5)
    mul_97: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_198, 0.7071067811865476);  view_198 = None
    erf_9: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
    add_89: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_98: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_96, add_89);  mul_96 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_139: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_98);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_199: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_139, [4608, 3072]);  clone_139 = None
    permute_130: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
    addmm_39: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_240, view_199, permute_130);  primals_240 = None
    view_200: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_39, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_140: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_200);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_99: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_21, clone_140);  clone_140 = None
    add_90: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_86, mul_99);  add_86 = mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_141: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_141, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 576, 1]" = var_mean_20[0]
    getitem_41: "f32[8, 576, 1]" = var_mean_20[1];  var_mean_20 = None
    add_91: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_20: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_30: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_141, getitem_41);  clone_141 = getitem_41 = None
    mul_100: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
    mul_101: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_100, primals_241)
    add_92: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_101, primals_242);  mul_101 = primals_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_201: "f32[4608, 768]" = torch.ops.aten.view.default(add_92, [4608, 768]);  add_92 = None
    permute_131: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_243, [1, 0]);  primals_243 = None
    addmm_40: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_244, view_201, permute_131);  primals_244 = None
    view_202: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_40, [8, 576, 2304]);  addmm_40 = None
    view_203: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_202, [8, 576, 3, 16, 48]);  view_202 = None
    permute_132: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_203, [2, 0, 3, 1, 4]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_30: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_132, 0, 0)
    mul_102: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_30, 0.14433756729740643);  select_30 = None
    select_31: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_132, 0, 1)
    select_32: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_132, 0, 2);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_133: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_31, [0, 1, 3, 2]);  select_31 = None
    expand_40: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_102, [8, 16, 576, 48]);  mul_102 = None
    clone_142: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_204: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_142, [128, 576, 48]);  clone_142 = None
    expand_41: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_133, [8, 16, 48, 576]);  permute_133 = None
    clone_143: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_205: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_143, [128, 48, 576]);  clone_143 = None
    bmm_20: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_204, view_205)
    view_206: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_20, [8, 16, 576, 576]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_134: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_206, [0, 2, 3, 1]);  view_206 = None
    permute_135: "f32[16, 16]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
    clone_144: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
    view_207: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_144, [2654208, 16]);  clone_144 = None
    mm_20: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_207, permute_135)
    view_208: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_20, [8, 576, 576, 16]);  mm_20 = None
    add_93: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_208, primals_246);  view_208 = primals_246 = None
    permute_136: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_93, [0, 3, 1, 2]);  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_145: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    amax_10: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_145, [-1], True)
    sub_31: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_145, amax_10);  clone_145 = amax_10 = None
    exp_10: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_137: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_10, [0, 2, 3, 1]);  div_10 = None
    permute_138: "f32[16, 16]" = torch.ops.aten.permute.default(primals_247, [1, 0]);  primals_247 = None
    clone_146: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    view_209: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_146, [2654208, 16]);  clone_146 = None
    mm_21: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_209, permute_138)
    view_210: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_21, [8, 576, 576, 16]);  mm_21 = None
    add_94: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_210, primals_248);  view_210 = primals_248 = None
    permute_139: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_94, [0, 3, 1, 2]);  add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_147: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_139);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_42: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_147, [8, 16, 576, 576]);  clone_147 = None
    clone_148: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_211: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_148, [128, 576, 576]);  clone_148 = None
    expand_43: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_32, [8, 16, 576, 48]);  select_32 = None
    clone_149: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_212: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_149, [128, 576, 48]);  clone_149 = None
    bmm_21: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_211, view_212)
    view_213: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_21, [8, 16, 576, 48]);  bmm_21 = None
    permute_140: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
    clone_150: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    view_214: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_150, [8, 576, 768]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_215: "f32[4608, 768]" = torch.ops.aten.view.default(view_214, [4608, 768]);  view_214 = None
    permute_141: "f32[768, 768]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
    addmm_41: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_250, view_215, permute_141);  primals_250 = None
    view_216: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_41, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_151: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_216);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_103: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_22, clone_151);  clone_151 = None
    add_95: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_90, mul_103);  add_90 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_152: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_95, memory_format = torch.contiguous_format)
    var_mean_21 = torch.ops.aten.var_mean.correction(clone_152, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 576, 1]" = var_mean_21[0]
    getitem_43: "f32[8, 576, 1]" = var_mean_21[1];  var_mean_21 = None
    add_96: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_21: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_32: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_152, getitem_43);  clone_152 = getitem_43 = None
    mul_104: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
    mul_105: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_104, primals_251)
    add_97: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_105, primals_252);  mul_105 = primals_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_217: "f32[4608, 768]" = torch.ops.aten.view.default(add_97, [4608, 768]);  add_97 = None
    permute_142: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
    addmm_42: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_254, view_217, permute_142);  primals_254 = None
    view_218: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_42, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_106: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_218, 0.5)
    mul_107: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_218, 0.7071067811865476);  view_218 = None
    erf_10: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
    add_98: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_108: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_106, add_98);  mul_106 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_153: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_108);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_219: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_153, [4608, 3072]);  clone_153 = None
    permute_143: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_43: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_256, view_219, permute_143);  primals_256 = None
    view_220: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_43, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_154: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_220);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_109: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_23, clone_154);  clone_154 = None
    add_99: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_95, mul_109);  add_95 = mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_155: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_99, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_155, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 576, 1]" = var_mean_22[0]
    getitem_45: "f32[8, 576, 1]" = var_mean_22[1];  var_mean_22 = None
    add_100: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_22: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    sub_33: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_155, getitem_45);  clone_155 = getitem_45 = None
    mul_110: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
    mul_111: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_110, primals_257)
    add_101: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_111, primals_258);  mul_111 = primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_221: "f32[4608, 768]" = torch.ops.aten.view.default(add_101, [4608, 768]);  add_101 = None
    permute_144: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
    addmm_44: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_260, view_221, permute_144);  primals_260 = None
    view_222: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_44, [8, 576, 2304]);  addmm_44 = None
    view_223: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_222, [8, 576, 3, 16, 48]);  view_222 = None
    permute_145: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_223, [2, 0, 3, 1, 4]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_33: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_145, 0, 0)
    mul_112: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_33, 0.14433756729740643);  select_33 = None
    select_34: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_145, 0, 1)
    select_35: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_145, 0, 2);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_146: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_34, [0, 1, 3, 2]);  select_34 = None
    expand_44: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_112, [8, 16, 576, 48]);  mul_112 = None
    clone_156: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_224: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_156, [128, 576, 48]);  clone_156 = None
    expand_45: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_146, [8, 16, 48, 576]);  permute_146 = None
    clone_157: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_225: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_157, [128, 48, 576]);  clone_157 = None
    bmm_22: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_224, view_225)
    view_226: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_22, [8, 16, 576, 576]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_147: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_226, [0, 2, 3, 1]);  view_226 = None
    permute_148: "f32[16, 16]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
    clone_158: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    view_227: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_158, [2654208, 16]);  clone_158 = None
    mm_22: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_227, permute_148)
    view_228: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_22, [8, 576, 576, 16]);  mm_22 = None
    add_102: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_228, primals_262);  view_228 = primals_262 = None
    permute_149: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_102, [0, 3, 1, 2]);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_159: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
    amax_11: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_159, [-1], True)
    sub_34: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_159, amax_11);  clone_159 = amax_11 = None
    exp_11: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_150: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_11, [0, 2, 3, 1]);  div_11 = None
    permute_151: "f32[16, 16]" = torch.ops.aten.permute.default(primals_263, [1, 0]);  primals_263 = None
    clone_160: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    view_229: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_160, [2654208, 16]);  clone_160 = None
    mm_23: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_229, permute_151)
    view_230: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_23, [8, 576, 576, 16]);  mm_23 = None
    add_103: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_230, primals_264);  view_230 = primals_264 = None
    permute_152: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_103, [0, 3, 1, 2]);  add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_161: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_152);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_46: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_161, [8, 16, 576, 576]);  clone_161 = None
    clone_162: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_231: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_162, [128, 576, 576]);  clone_162 = None
    expand_47: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_35, [8, 16, 576, 48]);  select_35 = None
    clone_163: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_232: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_163, [128, 576, 48]);  clone_163 = None
    bmm_23: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_231, view_232)
    view_233: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_23, [8, 16, 576, 48]);  bmm_23 = None
    permute_153: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
    clone_164: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
    view_234: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_164, [8, 576, 768]);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_235: "f32[4608, 768]" = torch.ops.aten.view.default(view_234, [4608, 768]);  view_234 = None
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
    addmm_45: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_266, view_235, permute_154);  primals_266 = None
    view_236: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_45, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_165: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_236);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_113: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_24, clone_165);  clone_165 = None
    add_104: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_99, mul_113);  add_99 = mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_166: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_104, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_166, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 576, 1]" = var_mean_23[0]
    getitem_47: "f32[8, 576, 1]" = var_mean_23[1];  var_mean_23 = None
    add_105: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_23: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    sub_35: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_166, getitem_47);  clone_166 = getitem_47 = None
    mul_114: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
    mul_115: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_114, primals_267)
    add_106: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_115, primals_268);  mul_115 = primals_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_237: "f32[4608, 768]" = torch.ops.aten.view.default(add_106, [4608, 768]);  add_106 = None
    permute_155: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
    addmm_46: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_270, view_237, permute_155);  primals_270 = None
    view_238: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_116: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_238, 0.5)
    mul_117: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_238, 0.7071067811865476);  view_238 = None
    erf_11: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_117);  mul_117 = None
    add_107: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_118: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_116, add_107);  mul_116 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_167: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_118);  mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_239: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_167, [4608, 3072]);  clone_167 = None
    permute_156: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
    addmm_47: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_272, view_239, permute_156);  primals_272 = None
    view_240: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_47, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_168: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_240);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_119: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_25, clone_168);  clone_168 = None
    add_108: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_104, mul_119);  add_104 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_169: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_108, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_169, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 576, 1]" = var_mean_24[0]
    getitem_49: "f32[8, 576, 1]" = var_mean_24[1];  var_mean_24 = None
    add_109: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_24: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    sub_36: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_169, getitem_49);  clone_169 = getitem_49 = None
    mul_120: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
    mul_121: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_120, primals_273)
    add_110: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_121, primals_274);  mul_121 = primals_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_241: "f32[4608, 768]" = torch.ops.aten.view.default(add_110, [4608, 768]);  add_110 = None
    permute_157: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_275, [1, 0]);  primals_275 = None
    addmm_48: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_276, view_241, permute_157);  primals_276 = None
    view_242: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_48, [8, 576, 2304]);  addmm_48 = None
    view_243: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_242, [8, 576, 3, 16, 48]);  view_242 = None
    permute_158: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_243, [2, 0, 3, 1, 4]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_36: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_158, 0, 0)
    mul_122: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_36, 0.14433756729740643);  select_36 = None
    select_37: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_158, 0, 1)
    select_38: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_158, 0, 2);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_159: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_37, [0, 1, 3, 2]);  select_37 = None
    expand_48: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_122, [8, 16, 576, 48]);  mul_122 = None
    clone_170: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_244: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_170, [128, 576, 48]);  clone_170 = None
    expand_49: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_159, [8, 16, 48, 576]);  permute_159 = None
    clone_171: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_245: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_171, [128, 48, 576]);  clone_171 = None
    bmm_24: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_244, view_245)
    view_246: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_24, [8, 16, 576, 576]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_160: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_246, [0, 2, 3, 1]);  view_246 = None
    permute_161: "f32[16, 16]" = torch.ops.aten.permute.default(primals_277, [1, 0]);  primals_277 = None
    clone_172: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    view_247: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_172, [2654208, 16]);  clone_172 = None
    mm_24: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_247, permute_161)
    view_248: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_24, [8, 576, 576, 16]);  mm_24 = None
    add_111: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_248, primals_278);  view_248 = primals_278 = None
    permute_162: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_111, [0, 3, 1, 2]);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_173: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    amax_12: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_173, [-1], True)
    sub_37: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_173, amax_12);  clone_173 = amax_12 = None
    exp_12: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_13: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_12: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_163: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_12, [0, 2, 3, 1]);  div_12 = None
    permute_164: "f32[16, 16]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
    clone_174: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    view_249: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_174, [2654208, 16]);  clone_174 = None
    mm_25: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_249, permute_164)
    view_250: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_25, [8, 576, 576, 16]);  mm_25 = None
    add_112: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_250, primals_280);  view_250 = primals_280 = None
    permute_165: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_112, [0, 3, 1, 2]);  add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_175: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_165);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_50: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_175, [8, 16, 576, 576]);  clone_175 = None
    clone_176: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
    view_251: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_176, [128, 576, 576]);  clone_176 = None
    expand_51: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_38, [8, 16, 576, 48]);  select_38 = None
    clone_177: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_252: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_177, [128, 576, 48]);  clone_177 = None
    bmm_25: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_251, view_252)
    view_253: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_25, [8, 16, 576, 48]);  bmm_25 = None
    permute_166: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
    clone_178: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
    view_254: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_178, [8, 576, 768]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_255: "f32[4608, 768]" = torch.ops.aten.view.default(view_254, [4608, 768]);  view_254 = None
    permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
    addmm_49: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_282, view_255, permute_167);  primals_282 = None
    view_256: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_49, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_179: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_256);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_123: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_26, clone_179);  clone_179 = None
    add_113: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_108, mul_123);  add_108 = mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_180: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_113, memory_format = torch.contiguous_format)
    var_mean_25 = torch.ops.aten.var_mean.correction(clone_180, [2], correction = 0, keepdim = True)
    getitem_50: "f32[8, 576, 1]" = var_mean_25[0]
    getitem_51: "f32[8, 576, 1]" = var_mean_25[1];  var_mean_25 = None
    add_114: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
    rsqrt_25: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
    sub_38: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_180, getitem_51);  clone_180 = getitem_51 = None
    mul_124: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
    mul_125: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_124, primals_283)
    add_115: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_125, primals_284);  mul_125 = primals_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_257: "f32[4608, 768]" = torch.ops.aten.view.default(add_115, [4608, 768]);  add_115 = None
    permute_168: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
    addmm_50: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_286, view_257, permute_168);  primals_286 = None
    view_258: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_50, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_126: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.5)
    mul_127: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.7071067811865476);  view_258 = None
    erf_12: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_127);  mul_127 = None
    add_116: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_128: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_126, add_116);  mul_126 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_181: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_128);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_259: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_181, [4608, 3072]);  clone_181 = None
    permute_169: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_287, [1, 0]);  primals_287 = None
    addmm_51: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_288, view_259, permute_169);  primals_288 = None
    view_260: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_51, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_182: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_260);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_129: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_27, clone_182);  clone_182 = None
    add_117: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_113, mul_129);  add_113 = mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_183: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_117, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_183, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 576, 1]" = var_mean_26[0]
    getitem_53: "f32[8, 576, 1]" = var_mean_26[1];  var_mean_26 = None
    add_118: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_26: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    sub_39: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_183, getitem_53);  clone_183 = getitem_53 = None
    mul_130: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = None
    mul_131: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_130, primals_289)
    add_119: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_131, primals_290);  mul_131 = primals_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_261: "f32[4608, 768]" = torch.ops.aten.view.default(add_119, [4608, 768]);  add_119 = None
    permute_170: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
    addmm_52: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_292, view_261, permute_170);  primals_292 = None
    view_262: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_52, [8, 576, 2304]);  addmm_52 = None
    view_263: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_262, [8, 576, 3, 16, 48]);  view_262 = None
    permute_171: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_263, [2, 0, 3, 1, 4]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_39: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_171, 0, 0)
    mul_132: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_39, 0.14433756729740643);  select_39 = None
    select_40: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_171, 0, 1)
    select_41: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_171, 0, 2);  permute_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_172: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_40, [0, 1, 3, 2]);  select_40 = None
    expand_52: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_132, [8, 16, 576, 48]);  mul_132 = None
    clone_184: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_264: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_184, [128, 576, 48]);  clone_184 = None
    expand_53: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_172, [8, 16, 48, 576]);  permute_172 = None
    clone_185: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_265: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_185, [128, 48, 576]);  clone_185 = None
    bmm_26: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_264, view_265)
    view_266: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_26, [8, 16, 576, 576]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_173: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_266, [0, 2, 3, 1]);  view_266 = None
    permute_174: "f32[16, 16]" = torch.ops.aten.permute.default(primals_293, [1, 0]);  primals_293 = None
    clone_186: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
    view_267: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_186, [2654208, 16]);  clone_186 = None
    mm_26: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_267, permute_174)
    view_268: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_26, [8, 576, 576, 16]);  mm_26 = None
    add_120: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_268, primals_294);  view_268 = primals_294 = None
    permute_175: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_120, [0, 3, 1, 2]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_187: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
    amax_13: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_187, [-1], True)
    sub_40: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_187, amax_13);  clone_187 = amax_13 = None
    exp_13: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_14: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_13: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_176: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_13, [0, 2, 3, 1]);  div_13 = None
    permute_177: "f32[16, 16]" = torch.ops.aten.permute.default(primals_295, [1, 0]);  primals_295 = None
    clone_188: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
    view_269: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_188, [2654208, 16]);  clone_188 = None
    mm_27: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_269, permute_177)
    view_270: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_27, [8, 576, 576, 16]);  mm_27 = None
    add_121: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_270, primals_296);  view_270 = primals_296 = None
    permute_178: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_121, [0, 3, 1, 2]);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_189: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_178);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_54: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_189, [8, 16, 576, 576]);  clone_189 = None
    clone_190: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
    view_271: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_190, [128, 576, 576]);  clone_190 = None
    expand_55: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_41, [8, 16, 576, 48]);  select_41 = None
    clone_191: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    view_272: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_191, [128, 576, 48]);  clone_191 = None
    bmm_27: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_271, view_272)
    view_273: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_27, [8, 16, 576, 48]);  bmm_27 = None
    permute_179: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
    clone_192: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    view_274: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_192, [8, 576, 768]);  clone_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_275: "f32[4608, 768]" = torch.ops.aten.view.default(view_274, [4608, 768]);  view_274 = None
    permute_180: "f32[768, 768]" = torch.ops.aten.permute.default(primals_297, [1, 0]);  primals_297 = None
    addmm_53: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_298, view_275, permute_180);  primals_298 = None
    view_276: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_53, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_193: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_276);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_133: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_28, clone_193);  clone_193 = None
    add_122: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_117, mul_133);  add_117 = mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_194: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_122, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_194, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 576, 1]" = var_mean_27[0]
    getitem_55: "f32[8, 576, 1]" = var_mean_27[1];  var_mean_27 = None
    add_123: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_27: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_41: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_194, getitem_55);  clone_194 = getitem_55 = None
    mul_134: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = None
    mul_135: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_134, primals_299)
    add_124: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_135, primals_300);  mul_135 = primals_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_277: "f32[4608, 768]" = torch.ops.aten.view.default(add_124, [4608, 768]);  add_124 = None
    permute_181: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_301, [1, 0]);  primals_301 = None
    addmm_54: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_302, view_277, permute_181);  primals_302 = None
    view_278: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_54, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_136: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_278, 0.5)
    mul_137: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_278, 0.7071067811865476);  view_278 = None
    erf_13: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_137);  mul_137 = None
    add_125: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_138: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_136, add_125);  mul_136 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_195: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_138);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_279: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_195, [4608, 3072]);  clone_195 = None
    permute_182: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_303, [1, 0]);  primals_303 = None
    addmm_55: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_304, view_279, permute_182);  primals_304 = None
    view_280: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_55, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_196: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_280);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_139: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_29, clone_196);  clone_196 = None
    add_126: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_122, mul_139);  add_122 = mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_197: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_126, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_197, [2], correction = 0, keepdim = True)
    getitem_56: "f32[8, 576, 1]" = var_mean_28[0]
    getitem_57: "f32[8, 576, 1]" = var_mean_28[1];  var_mean_28 = None
    add_127: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_28: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_42: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_197, getitem_57);  clone_197 = getitem_57 = None
    mul_140: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = None
    mul_141: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_140, primals_305)
    add_128: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_141, primals_306);  mul_141 = primals_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_281: "f32[4608, 768]" = torch.ops.aten.view.default(add_128, [4608, 768]);  add_128 = None
    permute_183: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_307, [1, 0]);  primals_307 = None
    addmm_56: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_308, view_281, permute_183);  primals_308 = None
    view_282: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_56, [8, 576, 2304]);  addmm_56 = None
    view_283: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_282, [8, 576, 3, 16, 48]);  view_282 = None
    permute_184: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_283, [2, 0, 3, 1, 4]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_42: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_184, 0, 0)
    mul_142: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_42, 0.14433756729740643);  select_42 = None
    select_43: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_184, 0, 1)
    select_44: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_184, 0, 2);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_185: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_43, [0, 1, 3, 2]);  select_43 = None
    expand_56: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_142, [8, 16, 576, 48]);  mul_142 = None
    clone_198: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_284: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_198, [128, 576, 48]);  clone_198 = None
    expand_57: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_185, [8, 16, 48, 576]);  permute_185 = None
    clone_199: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_285: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_199, [128, 48, 576]);  clone_199 = None
    bmm_28: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_284, view_285)
    view_286: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_28, [8, 16, 576, 576]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_186: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_286, [0, 2, 3, 1]);  view_286 = None
    permute_187: "f32[16, 16]" = torch.ops.aten.permute.default(primals_309, [1, 0]);  primals_309 = None
    clone_200: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    view_287: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_200, [2654208, 16]);  clone_200 = None
    mm_28: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_287, permute_187)
    view_288: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_28, [8, 576, 576, 16]);  mm_28 = None
    add_129: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_288, primals_310);  view_288 = primals_310 = None
    permute_188: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_129, [0, 3, 1, 2]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_201: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    amax_14: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_201, [-1], True)
    sub_43: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_201, amax_14);  clone_201 = amax_14 = None
    exp_14: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_15: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_14: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_189: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_14, [0, 2, 3, 1]);  div_14 = None
    permute_190: "f32[16, 16]" = torch.ops.aten.permute.default(primals_311, [1, 0]);  primals_311 = None
    clone_202: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    view_289: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_202, [2654208, 16]);  clone_202 = None
    mm_29: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_289, permute_190)
    view_290: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_29, [8, 576, 576, 16]);  mm_29 = None
    add_130: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_290, primals_312);  view_290 = primals_312 = None
    permute_191: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_130, [0, 3, 1, 2]);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_203: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_191);  permute_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_58: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_203, [8, 16, 576, 576]);  clone_203 = None
    clone_204: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
    view_291: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_204, [128, 576, 576]);  clone_204 = None
    expand_59: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_44, [8, 16, 576, 48]);  select_44 = None
    clone_205: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_292: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_205, [128, 576, 48]);  clone_205 = None
    bmm_29: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_291, view_292)
    view_293: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_29, [8, 16, 576, 48]);  bmm_29 = None
    permute_192: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
    clone_206: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
    view_294: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_206, [8, 576, 768]);  clone_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_295: "f32[4608, 768]" = torch.ops.aten.view.default(view_294, [4608, 768]);  view_294 = None
    permute_193: "f32[768, 768]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
    addmm_57: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_314, view_295, permute_193);  primals_314 = None
    view_296: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_57, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_207: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_296);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_143: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_30, clone_207);  clone_207 = None
    add_131: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_126, mul_143);  add_126 = mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_208: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_131, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_208, [2], correction = 0, keepdim = True)
    getitem_58: "f32[8, 576, 1]" = var_mean_29[0]
    getitem_59: "f32[8, 576, 1]" = var_mean_29[1];  var_mean_29 = None
    add_132: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-06);  getitem_58 = None
    rsqrt_29: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    sub_44: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_208, getitem_59);  clone_208 = getitem_59 = None
    mul_144: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = None
    mul_145: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_144, primals_315)
    add_133: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_145, primals_316);  mul_145 = primals_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_297: "f32[4608, 768]" = torch.ops.aten.view.default(add_133, [4608, 768]);  add_133 = None
    permute_194: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_317, [1, 0]);  primals_317 = None
    addmm_58: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_318, view_297, permute_194);  primals_318 = None
    view_298: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_58, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_146: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_298, 0.5)
    mul_147: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_298, 0.7071067811865476);  view_298 = None
    erf_14: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_134: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_148: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_146, add_134);  mul_146 = add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_209: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_148);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_299: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_209, [4608, 3072]);  clone_209 = None
    permute_195: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_319, [1, 0]);  primals_319 = None
    addmm_59: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_320, view_299, permute_195);  primals_320 = None
    view_300: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_59, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_210: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_300);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_149: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_31, clone_210);  clone_210 = None
    add_135: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_131, mul_149);  add_131 = mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_211: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_135, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_211, [2], correction = 0, keepdim = True)
    getitem_60: "f32[8, 576, 1]" = var_mean_30[0]
    getitem_61: "f32[8, 576, 1]" = var_mean_30[1];  var_mean_30 = None
    add_136: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-06);  getitem_60 = None
    rsqrt_30: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    sub_45: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_211, getitem_61);  clone_211 = getitem_61 = None
    mul_150: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = None
    mul_151: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_150, primals_321)
    add_137: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_151, primals_322);  mul_151 = primals_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_301: "f32[4608, 768]" = torch.ops.aten.view.default(add_137, [4608, 768]);  add_137 = None
    permute_196: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_323, [1, 0]);  primals_323 = None
    addmm_60: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_324, view_301, permute_196);  primals_324 = None
    view_302: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_60, [8, 576, 2304]);  addmm_60 = None
    view_303: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_302, [8, 576, 3, 16, 48]);  view_302 = None
    permute_197: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_303, [2, 0, 3, 1, 4]);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_45: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_197, 0, 0)
    mul_152: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_45, 0.14433756729740643);  select_45 = None
    select_46: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_197, 0, 1)
    select_47: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_197, 0, 2);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_198: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_46, [0, 1, 3, 2]);  select_46 = None
    expand_60: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_152, [8, 16, 576, 48]);  mul_152 = None
    clone_212: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_304: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_212, [128, 576, 48]);  clone_212 = None
    expand_61: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_198, [8, 16, 48, 576]);  permute_198 = None
    clone_213: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_305: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_213, [128, 48, 576]);  clone_213 = None
    bmm_30: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_304, view_305)
    view_306: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_30, [8, 16, 576, 576]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_199: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_306, [0, 2, 3, 1]);  view_306 = None
    permute_200: "f32[16, 16]" = torch.ops.aten.permute.default(primals_325, [1, 0]);  primals_325 = None
    clone_214: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_307: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_214, [2654208, 16]);  clone_214 = None
    mm_30: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_307, permute_200)
    view_308: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_30, [8, 576, 576, 16]);  mm_30 = None
    add_138: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_308, primals_326);  view_308 = primals_326 = None
    permute_201: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_138, [0, 3, 1, 2]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_215: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
    amax_15: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_215, [-1], True)
    sub_46: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_215, amax_15);  clone_215 = amax_15 = None
    exp_15: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_16: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_15: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_202: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_15, [0, 2, 3, 1]);  div_15 = None
    permute_203: "f32[16, 16]" = torch.ops.aten.permute.default(primals_327, [1, 0]);  primals_327 = None
    clone_216: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
    view_309: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_216, [2654208, 16]);  clone_216 = None
    mm_31: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_309, permute_203)
    view_310: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_31, [8, 576, 576, 16]);  mm_31 = None
    add_139: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_310, primals_328);  view_310 = primals_328 = None
    permute_204: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_139, [0, 3, 1, 2]);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_217: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_204);  permute_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_62: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_217, [8, 16, 576, 576]);  clone_217 = None
    clone_218: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
    view_311: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_218, [128, 576, 576]);  clone_218 = None
    expand_63: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_47, [8, 16, 576, 48]);  select_47 = None
    clone_219: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_312: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_219, [128, 576, 48]);  clone_219 = None
    bmm_31: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_311, view_312)
    view_313: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_31, [8, 16, 576, 48]);  bmm_31 = None
    permute_205: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
    clone_220: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    view_314: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_220, [8, 576, 768]);  clone_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_315: "f32[4608, 768]" = torch.ops.aten.view.default(view_314, [4608, 768]);  view_314 = None
    permute_206: "f32[768, 768]" = torch.ops.aten.permute.default(primals_329, [1, 0]);  primals_329 = None
    addmm_61: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_330, view_315, permute_206);  primals_330 = None
    view_316: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_61, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_221: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_316);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_153: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_32, clone_221);  clone_221 = None
    add_140: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_135, mul_153);  add_135 = mul_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_222: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_140, memory_format = torch.contiguous_format)
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_222, [2], correction = 0, keepdim = True)
    getitem_62: "f32[8, 576, 1]" = var_mean_31[0]
    getitem_63: "f32[8, 576, 1]" = var_mean_31[1];  var_mean_31 = None
    add_141: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
    rsqrt_31: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_47: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_222, getitem_63);  clone_222 = getitem_63 = None
    mul_154: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = None
    mul_155: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_154, primals_331)
    add_142: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_155, primals_332);  mul_155 = primals_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_317: "f32[4608, 768]" = torch.ops.aten.view.default(add_142, [4608, 768]);  add_142 = None
    permute_207: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_333, [1, 0]);  primals_333 = None
    addmm_62: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_334, view_317, permute_207);  primals_334 = None
    view_318: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_62, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_156: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_318, 0.5)
    mul_157: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_318, 0.7071067811865476);  view_318 = None
    erf_15: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_157);  mul_157 = None
    add_143: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_158: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_156, add_143);  mul_156 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_223: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_158);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_319: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_223, [4608, 3072]);  clone_223 = None
    permute_208: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_335, [1, 0]);  primals_335 = None
    addmm_63: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_336, view_319, permute_208);  primals_336 = None
    view_320: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_63, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_224: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_320);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_159: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_33, clone_224);  clone_224 = None
    add_144: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_140, mul_159);  add_140 = mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_225: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_144, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_225, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 576, 1]" = var_mean_32[0]
    getitem_65: "f32[8, 576, 1]" = var_mean_32[1];  var_mean_32 = None
    add_145: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_32: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    sub_48: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_225, getitem_65);  clone_225 = getitem_65 = None
    mul_160: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = None
    mul_161: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_160, primals_337)
    add_146: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_161, primals_338);  mul_161 = primals_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_321: "f32[4608, 768]" = torch.ops.aten.view.default(add_146, [4608, 768]);  add_146 = None
    permute_209: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_339, [1, 0]);  primals_339 = None
    addmm_64: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_340, view_321, permute_209);  primals_340 = None
    view_322: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_64, [8, 576, 2304]);  addmm_64 = None
    view_323: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_322, [8, 576, 3, 16, 48]);  view_322 = None
    permute_210: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_323, [2, 0, 3, 1, 4]);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_48: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_210, 0, 0)
    mul_162: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_48, 0.14433756729740643);  select_48 = None
    select_49: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_210, 0, 1)
    select_50: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_210, 0, 2);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_211: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_49, [0, 1, 3, 2]);  select_49 = None
    expand_64: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_162, [8, 16, 576, 48]);  mul_162 = None
    clone_226: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_324: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_226, [128, 576, 48]);  clone_226 = None
    expand_65: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_211, [8, 16, 48, 576]);  permute_211 = None
    clone_227: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_325: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_227, [128, 48, 576]);  clone_227 = None
    bmm_32: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_324, view_325)
    view_326: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_32, [8, 16, 576, 576]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_212: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_326, [0, 2, 3, 1]);  view_326 = None
    permute_213: "f32[16, 16]" = torch.ops.aten.permute.default(primals_341, [1, 0]);  primals_341 = None
    clone_228: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
    view_327: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_228, [2654208, 16]);  clone_228 = None
    mm_32: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_327, permute_213)
    view_328: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_32, [8, 576, 576, 16]);  mm_32 = None
    add_147: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_328, primals_342);  view_328 = primals_342 = None
    permute_214: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_147, [0, 3, 1, 2]);  add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_229: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    amax_16: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_229, [-1], True)
    sub_49: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_229, amax_16);  clone_229 = amax_16 = None
    exp_16: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_17: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_16: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_215: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_16, [0, 2, 3, 1]);  div_16 = None
    permute_216: "f32[16, 16]" = torch.ops.aten.permute.default(primals_343, [1, 0]);  primals_343 = None
    clone_230: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
    view_329: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_230, [2654208, 16]);  clone_230 = None
    mm_33: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_329, permute_216)
    view_330: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_33, [8, 576, 576, 16]);  mm_33 = None
    add_148: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_330, primals_344);  view_330 = primals_344 = None
    permute_217: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_148, [0, 3, 1, 2]);  add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_231: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_217);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_66: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_231, [8, 16, 576, 576]);  clone_231 = None
    clone_232: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
    view_331: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_232, [128, 576, 576]);  clone_232 = None
    expand_67: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_50, [8, 16, 576, 48]);  select_50 = None
    clone_233: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
    view_332: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_233, [128, 576, 48]);  clone_233 = None
    bmm_33: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_331, view_332)
    view_333: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_33, [8, 16, 576, 48]);  bmm_33 = None
    permute_218: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None
    clone_234: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_218, memory_format = torch.contiguous_format);  permute_218 = None
    view_334: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_234, [8, 576, 768]);  clone_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_335: "f32[4608, 768]" = torch.ops.aten.view.default(view_334, [4608, 768]);  view_334 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(primals_345, [1, 0]);  primals_345 = None
    addmm_65: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_346, view_335, permute_219);  primals_346 = None
    view_336: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_65, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_235: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_336);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_163: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_34, clone_235);  clone_235 = None
    add_149: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_144, mul_163);  add_144 = mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_236: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_149, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_236, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 576, 1]" = var_mean_33[0]
    getitem_67: "f32[8, 576, 1]" = var_mean_33[1];  var_mean_33 = None
    add_150: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_33: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_50: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_236, getitem_67);  clone_236 = getitem_67 = None
    mul_164: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = None
    mul_165: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_164, primals_347)
    add_151: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_165, primals_348);  mul_165 = primals_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_337: "f32[4608, 768]" = torch.ops.aten.view.default(add_151, [4608, 768]);  add_151 = None
    permute_220: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_349, [1, 0]);  primals_349 = None
    addmm_66: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_350, view_337, permute_220);  primals_350 = None
    view_338: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_66, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_166: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_338, 0.5)
    mul_167: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_338, 0.7071067811865476);  view_338 = None
    erf_16: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
    add_152: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_168: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_166, add_152);  mul_166 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_237: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_168);  mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_339: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_237, [4608, 3072]);  clone_237 = None
    permute_221: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_351, [1, 0]);  primals_351 = None
    addmm_67: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_352, view_339, permute_221);  primals_352 = None
    view_340: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_67, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_238: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_340);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_169: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_35, clone_238);  clone_238 = None
    add_153: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_149, mul_169);  add_149 = mul_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_239: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_153, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_239, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 576, 1]" = var_mean_34[0]
    getitem_69: "f32[8, 576, 1]" = var_mean_34[1];  var_mean_34 = None
    add_154: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_34: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_51: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_239, getitem_69);  clone_239 = getitem_69 = None
    mul_170: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = None
    mul_171: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_170, primals_353)
    add_155: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_171, primals_354);  mul_171 = primals_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_341: "f32[4608, 768]" = torch.ops.aten.view.default(add_155, [4608, 768]);  add_155 = None
    permute_222: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_355, [1, 0]);  primals_355 = None
    addmm_68: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_356, view_341, permute_222);  primals_356 = None
    view_342: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_68, [8, 576, 2304]);  addmm_68 = None
    view_343: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_342, [8, 576, 3, 16, 48]);  view_342 = None
    permute_223: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_343, [2, 0, 3, 1, 4]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_51: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_223, 0, 0)
    mul_172: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_51, 0.14433756729740643);  select_51 = None
    select_52: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_223, 0, 1)
    select_53: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_223, 0, 2);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_224: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_52, [0, 1, 3, 2]);  select_52 = None
    expand_68: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_172, [8, 16, 576, 48]);  mul_172 = None
    clone_240: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_344: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_240, [128, 576, 48]);  clone_240 = None
    expand_69: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_224, [8, 16, 48, 576]);  permute_224 = None
    clone_241: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    view_345: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_241, [128, 48, 576]);  clone_241 = None
    bmm_34: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_344, view_345)
    view_346: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_34, [8, 16, 576, 576]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_225: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_346, [0, 2, 3, 1]);  view_346 = None
    permute_226: "f32[16, 16]" = torch.ops.aten.permute.default(primals_357, [1, 0]);  primals_357 = None
    clone_242: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
    view_347: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_242, [2654208, 16]);  clone_242 = None
    mm_34: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_347, permute_226)
    view_348: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_34, [8, 576, 576, 16]);  mm_34 = None
    add_156: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_348, primals_358);  view_348 = primals_358 = None
    permute_227: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_156, [0, 3, 1, 2]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_243: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    amax_17: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_243, [-1], True)
    sub_52: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_243, amax_17);  clone_243 = amax_17 = None
    exp_17: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_18: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_17: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_228: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_17, [0, 2, 3, 1]);  div_17 = None
    permute_229: "f32[16, 16]" = torch.ops.aten.permute.default(primals_359, [1, 0]);  primals_359 = None
    clone_244: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_349: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_244, [2654208, 16]);  clone_244 = None
    mm_35: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_349, permute_229)
    view_350: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_35, [8, 576, 576, 16]);  mm_35 = None
    add_157: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_350, primals_360);  view_350 = primals_360 = None
    permute_230: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_157, [0, 3, 1, 2]);  add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_245: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_230);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_70: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_245, [8, 16, 576, 576]);  clone_245 = None
    clone_246: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
    view_351: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_246, [128, 576, 576]);  clone_246 = None
    expand_71: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_53, [8, 16, 576, 48]);  select_53 = None
    clone_247: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    view_352: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_247, [128, 576, 48]);  clone_247 = None
    bmm_35: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_351, view_352)
    view_353: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_35, [8, 16, 576, 48]);  bmm_35 = None
    permute_231: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
    clone_248: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
    view_354: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_248, [8, 576, 768]);  clone_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_355: "f32[4608, 768]" = torch.ops.aten.view.default(view_354, [4608, 768]);  view_354 = None
    permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(primals_361, [1, 0]);  primals_361 = None
    addmm_69: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_362, view_355, permute_232);  primals_362 = None
    view_356: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_69, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_249: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_356);  view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_173: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_36, clone_249);  clone_249 = None
    add_158: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_153, mul_173);  add_153 = mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_250: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_158, memory_format = torch.contiguous_format)
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_250, [2], correction = 0, keepdim = True)
    getitem_70: "f32[8, 576, 1]" = var_mean_35[0]
    getitem_71: "f32[8, 576, 1]" = var_mean_35[1];  var_mean_35 = None
    add_159: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
    rsqrt_35: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    sub_53: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_250, getitem_71);  clone_250 = getitem_71 = None
    mul_174: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = None
    mul_175: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_174, primals_363)
    add_160: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_175, primals_364);  mul_175 = primals_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_357: "f32[4608, 768]" = torch.ops.aten.view.default(add_160, [4608, 768]);  add_160 = None
    permute_233: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_365, [1, 0]);  primals_365 = None
    addmm_70: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_366, view_357, permute_233);  primals_366 = None
    view_358: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_70, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_176: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_358, 0.5)
    mul_177: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_358, 0.7071067811865476);  view_358 = None
    erf_17: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_177);  mul_177 = None
    add_161: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_178: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_176, add_161);  mul_176 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_251: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_178);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_359: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_251, [4608, 3072]);  clone_251 = None
    permute_234: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_367, [1, 0]);  primals_367 = None
    addmm_71: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_368, view_359, permute_234);  primals_368 = None
    view_360: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_71, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_252: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_360);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_179: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_37, clone_252);  clone_252 = None
    add_162: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_158, mul_179);  add_158 = mul_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_253: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_162, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_253, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 576, 1]" = var_mean_36[0]
    getitem_73: "f32[8, 576, 1]" = var_mean_36[1];  var_mean_36 = None
    add_163: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_36: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_54: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_253, getitem_73);  clone_253 = getitem_73 = None
    mul_180: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = None
    mul_181: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_180, primals_369)
    add_164: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_181, primals_370);  mul_181 = primals_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_361: "f32[4608, 768]" = torch.ops.aten.view.default(add_164, [4608, 768]);  add_164 = None
    permute_235: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_371, [1, 0]);  primals_371 = None
    addmm_72: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_372, view_361, permute_235);  primals_372 = None
    view_362: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_72, [8, 576, 2304]);  addmm_72 = None
    view_363: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_362, [8, 576, 3, 16, 48]);  view_362 = None
    permute_236: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_363, [2, 0, 3, 1, 4]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_54: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_236, 0, 0)
    mul_182: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_54, 0.14433756729740643);  select_54 = None
    select_55: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_236, 0, 1)
    select_56: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_236, 0, 2);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_237: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_55, [0, 1, 3, 2]);  select_55 = None
    expand_72: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_182, [8, 16, 576, 48]);  mul_182 = None
    clone_254: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_364: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_254, [128, 576, 48]);  clone_254 = None
    expand_73: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_237, [8, 16, 48, 576]);  permute_237 = None
    clone_255: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
    view_365: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_255, [128, 48, 576]);  clone_255 = None
    bmm_36: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_364, view_365)
    view_366: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_36, [8, 16, 576, 576]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_238: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_366, [0, 2, 3, 1]);  view_366 = None
    permute_239: "f32[16, 16]" = torch.ops.aten.permute.default(primals_373, [1, 0]);  primals_373 = None
    clone_256: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    view_367: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_256, [2654208, 16]);  clone_256 = None
    mm_36: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_367, permute_239)
    view_368: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_36, [8, 576, 576, 16]);  mm_36 = None
    add_165: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_368, primals_374);  view_368 = primals_374 = None
    permute_240: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_165, [0, 3, 1, 2]);  add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_257: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
    amax_18: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_257, [-1], True)
    sub_55: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_257, amax_18);  clone_257 = amax_18 = None
    exp_18: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_19: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_18: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_241: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_18, [0, 2, 3, 1]);  div_18 = None
    permute_242: "f32[16, 16]" = torch.ops.aten.permute.default(primals_375, [1, 0]);  primals_375 = None
    clone_258: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
    view_369: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_258, [2654208, 16]);  clone_258 = None
    mm_37: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_369, permute_242)
    view_370: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_37, [8, 576, 576, 16]);  mm_37 = None
    add_166: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_370, primals_376);  view_370 = primals_376 = None
    permute_243: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_166, [0, 3, 1, 2]);  add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_259: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_243);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_74: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_259, [8, 16, 576, 576]);  clone_259 = None
    clone_260: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    view_371: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_260, [128, 576, 576]);  clone_260 = None
    expand_75: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_56, [8, 16, 576, 48]);  select_56 = None
    clone_261: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    view_372: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_261, [128, 576, 48]);  clone_261 = None
    bmm_37: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_371, view_372)
    view_373: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_37, [8, 16, 576, 48]);  bmm_37 = None
    permute_244: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    clone_262: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
    view_374: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_262, [8, 576, 768]);  clone_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_375: "f32[4608, 768]" = torch.ops.aten.view.default(view_374, [4608, 768]);  view_374 = None
    permute_245: "f32[768, 768]" = torch.ops.aten.permute.default(primals_377, [1, 0]);  primals_377 = None
    addmm_73: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_378, view_375, permute_245);  primals_378 = None
    view_376: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_73, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_263: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_376);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_183: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_38, clone_263);  clone_263 = None
    add_167: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_162, mul_183);  add_162 = mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_264: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format)
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_264, [2], correction = 0, keepdim = True)
    getitem_74: "f32[8, 576, 1]" = var_mean_37[0]
    getitem_75: "f32[8, 576, 1]" = var_mean_37[1];  var_mean_37 = None
    add_168: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-06);  getitem_74 = None
    rsqrt_37: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_56: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_264, getitem_75);  clone_264 = getitem_75 = None
    mul_184: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = None
    mul_185: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_184, primals_379)
    add_169: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_185, primals_380);  mul_185 = primals_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_377: "f32[4608, 768]" = torch.ops.aten.view.default(add_169, [4608, 768]);  add_169 = None
    permute_246: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_381, [1, 0]);  primals_381 = None
    addmm_74: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_382, view_377, permute_246);  primals_382 = None
    view_378: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_74, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_186: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_378, 0.5)
    mul_187: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_378, 0.7071067811865476);  view_378 = None
    erf_18: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_187);  mul_187 = None
    add_170: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_188: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_186, add_170);  mul_186 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_265: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_188);  mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_379: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_265, [4608, 3072]);  clone_265 = None
    permute_247: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_383, [1, 0]);  primals_383 = None
    addmm_75: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_384, view_379, permute_247);  primals_384 = None
    view_380: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_75, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_266: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_380);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_189: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_39, clone_266);  clone_266 = None
    add_171: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_167, mul_189);  add_167 = mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_267: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_171, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_267, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 576, 1]" = var_mean_38[0]
    getitem_77: "f32[8, 576, 1]" = var_mean_38[1];  var_mean_38 = None
    add_172: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_38: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_57: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_267, getitem_77);  clone_267 = getitem_77 = None
    mul_190: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = None
    mul_191: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_190, primals_385)
    add_173: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_191, primals_386);  mul_191 = primals_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_381: "f32[4608, 768]" = torch.ops.aten.view.default(add_173, [4608, 768]);  add_173 = None
    permute_248: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_387, [1, 0]);  primals_387 = None
    addmm_76: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_388, view_381, permute_248);  primals_388 = None
    view_382: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_76, [8, 576, 2304]);  addmm_76 = None
    view_383: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_382, [8, 576, 3, 16, 48]);  view_382 = None
    permute_249: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_383, [2, 0, 3, 1, 4]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_57: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_249, 0, 0)
    mul_192: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_57, 0.14433756729740643);  select_57 = None
    select_58: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_249, 0, 1)
    select_59: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_249, 0, 2);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_250: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_58, [0, 1, 3, 2]);  select_58 = None
    expand_76: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_192, [8, 16, 576, 48]);  mul_192 = None
    clone_268: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
    view_384: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_268, [128, 576, 48]);  clone_268 = None
    expand_77: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_250, [8, 16, 48, 576]);  permute_250 = None
    clone_269: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
    view_385: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_269, [128, 48, 576]);  clone_269 = None
    bmm_38: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_384, view_385)
    view_386: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_38, [8, 16, 576, 576]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_251: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_386, [0, 2, 3, 1]);  view_386 = None
    permute_252: "f32[16, 16]" = torch.ops.aten.permute.default(primals_389, [1, 0]);  primals_389 = None
    clone_270: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_251, memory_format = torch.contiguous_format);  permute_251 = None
    view_387: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_270, [2654208, 16]);  clone_270 = None
    mm_38: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_387, permute_252)
    view_388: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_38, [8, 576, 576, 16]);  mm_38 = None
    add_174: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_388, primals_390);  view_388 = primals_390 = None
    permute_253: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_174, [0, 3, 1, 2]);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_271: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
    amax_19: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_271, [-1], True)
    sub_58: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_271, amax_19);  clone_271 = amax_19 = None
    exp_19: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_20: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    alias_19: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_254: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_19, [0, 2, 3, 1]);  div_19 = None
    permute_255: "f32[16, 16]" = torch.ops.aten.permute.default(primals_391, [1, 0]);  primals_391 = None
    clone_272: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
    view_389: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_272, [2654208, 16]);  clone_272 = None
    mm_39: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_389, permute_255)
    view_390: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_39, [8, 576, 576, 16]);  mm_39 = None
    add_175: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_390, primals_392);  view_390 = primals_392 = None
    permute_256: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_175, [0, 3, 1, 2]);  add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_273: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_256);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_78: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_273, [8, 16, 576, 576]);  clone_273 = None
    clone_274: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
    view_391: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_274, [128, 576, 576]);  clone_274 = None
    expand_79: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_59, [8, 16, 576, 48]);  select_59 = None
    clone_275: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
    view_392: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_275, [128, 576, 48]);  clone_275 = None
    bmm_39: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_391, view_392)
    view_393: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_39, [8, 16, 576, 48]);  bmm_39 = None
    permute_257: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
    clone_276: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
    view_394: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_276, [8, 576, 768]);  clone_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_395: "f32[4608, 768]" = torch.ops.aten.view.default(view_394, [4608, 768]);  view_394 = None
    permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(primals_393, [1, 0]);  primals_393 = None
    addmm_77: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_394, view_395, permute_258);  primals_394 = None
    view_396: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_77, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_277: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_396);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_193: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_40, clone_277);  clone_277 = None
    add_176: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_171, mul_193);  add_171 = mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_278: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_176, memory_format = torch.contiguous_format)
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_278, [2], correction = 0, keepdim = True)
    getitem_78: "f32[8, 576, 1]" = var_mean_39[0]
    getitem_79: "f32[8, 576, 1]" = var_mean_39[1];  var_mean_39 = None
    add_177: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_39: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    sub_59: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_278, getitem_79);  clone_278 = getitem_79 = None
    mul_194: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = None
    mul_195: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_194, primals_395)
    add_178: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_195, primals_396);  mul_195 = primals_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_397: "f32[4608, 768]" = torch.ops.aten.view.default(add_178, [4608, 768]);  add_178 = None
    permute_259: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_397, [1, 0]);  primals_397 = None
    addmm_78: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_398, view_397, permute_259);  primals_398 = None
    view_398: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_78, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_196: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_398, 0.5)
    mul_197: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_398, 0.7071067811865476);  view_398 = None
    erf_19: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_197);  mul_197 = None
    add_179: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_198: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_196, add_179);  mul_196 = add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_279: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_198);  mul_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_399: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_279, [4608, 3072]);  clone_279 = None
    permute_260: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_399, [1, 0]);  primals_399 = None
    addmm_79: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_400, view_399, permute_260);  primals_400 = None
    view_400: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_79, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_280: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_400);  view_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_199: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_41, clone_280);  clone_280 = None
    add_180: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_176, mul_199);  add_176 = mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_281: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_180, memory_format = torch.contiguous_format)
    var_mean_40 = torch.ops.aten.var_mean.correction(clone_281, [2], correction = 0, keepdim = True)
    getitem_80: "f32[8, 576, 1]" = var_mean_40[0]
    getitem_81: "f32[8, 576, 1]" = var_mean_40[1];  var_mean_40 = None
    add_181: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
    rsqrt_40: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    sub_60: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_281, getitem_81);  clone_281 = getitem_81 = None
    mul_200: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = None
    mul_201: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_200, primals_401)
    add_182: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_201, primals_402);  mul_201 = primals_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_401: "f32[4608, 768]" = torch.ops.aten.view.default(add_182, [4608, 768]);  add_182 = None
    permute_261: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_403, [1, 0]);  primals_403 = None
    addmm_80: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_404, view_401, permute_261);  primals_404 = None
    view_402: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_80, [8, 576, 2304]);  addmm_80 = None
    view_403: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_402, [8, 576, 3, 16, 48]);  view_402 = None
    permute_262: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_403, [2, 0, 3, 1, 4]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_60: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_262, 0, 0)
    mul_202: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_60, 0.14433756729740643);  select_60 = None
    select_61: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_262, 0, 1)
    select_62: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_262, 0, 2);  permute_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_263: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_61, [0, 1, 3, 2]);  select_61 = None
    expand_80: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_202, [8, 16, 576, 48]);  mul_202 = None
    clone_282: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    view_404: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_282, [128, 576, 48]);  clone_282 = None
    expand_81: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_263, [8, 16, 48, 576]);  permute_263 = None
    clone_283: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    view_405: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_283, [128, 48, 576]);  clone_283 = None
    bmm_40: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_404, view_405)
    view_406: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_40, [8, 16, 576, 576]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_264: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_406, [0, 2, 3, 1]);  view_406 = None
    permute_265: "f32[16, 16]" = torch.ops.aten.permute.default(primals_405, [1, 0]);  primals_405 = None
    clone_284: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_264, memory_format = torch.contiguous_format);  permute_264 = None
    view_407: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_284, [2654208, 16]);  clone_284 = None
    mm_40: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_407, permute_265)
    view_408: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_40, [8, 576, 576, 16]);  mm_40 = None
    add_183: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_408, primals_406);  view_408 = primals_406 = None
    permute_266: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_183, [0, 3, 1, 2]);  add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_285: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    amax_20: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_285, [-1], True)
    sub_61: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_285, amax_20);  clone_285 = amax_20 = None
    exp_20: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_21: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    alias_20: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_267: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_20, [0, 2, 3, 1]);  div_20 = None
    permute_268: "f32[16, 16]" = torch.ops.aten.permute.default(primals_407, [1, 0]);  primals_407 = None
    clone_286: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
    view_409: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_286, [2654208, 16]);  clone_286 = None
    mm_41: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_409, permute_268)
    view_410: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_41, [8, 576, 576, 16]);  mm_41 = None
    add_184: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_410, primals_408);  view_410 = primals_408 = None
    permute_269: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_184, [0, 3, 1, 2]);  add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_287: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_269);  permute_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_82: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_287, [8, 16, 576, 576]);  clone_287 = None
    clone_288: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
    view_411: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_288, [128, 576, 576]);  clone_288 = None
    expand_83: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_62, [8, 16, 576, 48]);  select_62 = None
    clone_289: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
    view_412: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_289, [128, 576, 48]);  clone_289 = None
    bmm_41: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_411, view_412)
    view_413: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_41, [8, 16, 576, 48]);  bmm_41 = None
    permute_270: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
    clone_290: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
    view_414: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_290, [8, 576, 768]);  clone_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_415: "f32[4608, 768]" = torch.ops.aten.view.default(view_414, [4608, 768]);  view_414 = None
    permute_271: "f32[768, 768]" = torch.ops.aten.permute.default(primals_409, [1, 0]);  primals_409 = None
    addmm_81: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_410, view_415, permute_271);  primals_410 = None
    view_416: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_81, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_291: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_416);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_203: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_42, clone_291);  clone_291 = None
    add_185: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_180, mul_203);  add_180 = mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_292: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_185, memory_format = torch.contiguous_format)
    var_mean_41 = torch.ops.aten.var_mean.correction(clone_292, [2], correction = 0, keepdim = True)
    getitem_82: "f32[8, 576, 1]" = var_mean_41[0]
    getitem_83: "f32[8, 576, 1]" = var_mean_41[1];  var_mean_41 = None
    add_186: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-06);  getitem_82 = None
    rsqrt_41: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
    sub_62: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_292, getitem_83);  clone_292 = getitem_83 = None
    mul_204: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = None
    mul_205: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_204, primals_411)
    add_187: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_205, primals_412);  mul_205 = primals_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_417: "f32[4608, 768]" = torch.ops.aten.view.default(add_187, [4608, 768]);  add_187 = None
    permute_272: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_413, [1, 0]);  primals_413 = None
    addmm_82: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_414, view_417, permute_272);  primals_414 = None
    view_418: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_82, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_206: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_418, 0.5)
    mul_207: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_418, 0.7071067811865476);  view_418 = None
    erf_20: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
    add_188: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_208: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_206, add_188);  mul_206 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_293: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_208);  mul_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_419: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_293, [4608, 3072]);  clone_293 = None
    permute_273: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_415, [1, 0]);  primals_415 = None
    addmm_83: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_416, view_419, permute_273);  primals_416 = None
    view_420: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_83, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_294: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_420);  view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_209: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_43, clone_294);  clone_294 = None
    add_189: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_185, mul_209);  add_185 = mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_295: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_189, memory_format = torch.contiguous_format)
    var_mean_42 = torch.ops.aten.var_mean.correction(clone_295, [2], correction = 0, keepdim = True)
    getitem_84: "f32[8, 576, 1]" = var_mean_42[0]
    getitem_85: "f32[8, 576, 1]" = var_mean_42[1];  var_mean_42 = None
    add_190: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
    rsqrt_42: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
    sub_63: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_295, getitem_85);  clone_295 = getitem_85 = None
    mul_210: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = None
    mul_211: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_210, primals_417)
    add_191: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_211, primals_418);  mul_211 = primals_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_421: "f32[4608, 768]" = torch.ops.aten.view.default(add_191, [4608, 768]);  add_191 = None
    permute_274: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_419, [1, 0]);  primals_419 = None
    addmm_84: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_420, view_421, permute_274);  primals_420 = None
    view_422: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_84, [8, 576, 2304]);  addmm_84 = None
    view_423: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_422, [8, 576, 3, 16, 48]);  view_422 = None
    permute_275: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_423, [2, 0, 3, 1, 4]);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_63: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_275, 0, 0)
    mul_212: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_63, 0.14433756729740643);  select_63 = None
    select_64: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_275, 0, 1)
    select_65: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_275, 0, 2);  permute_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_276: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_64, [0, 1, 3, 2]);  select_64 = None
    expand_84: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_212, [8, 16, 576, 48]);  mul_212 = None
    clone_296: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
    view_424: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_296, [128, 576, 48]);  clone_296 = None
    expand_85: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_276, [8, 16, 48, 576]);  permute_276 = None
    clone_297: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
    view_425: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_297, [128, 48, 576]);  clone_297 = None
    bmm_42: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_424, view_425)
    view_426: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_42, [8, 16, 576, 576]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_277: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_426, [0, 2, 3, 1]);  view_426 = None
    permute_278: "f32[16, 16]" = torch.ops.aten.permute.default(primals_421, [1, 0]);  primals_421 = None
    clone_298: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_277, memory_format = torch.contiguous_format);  permute_277 = None
    view_427: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_298, [2654208, 16]);  clone_298 = None
    mm_42: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_427, permute_278)
    view_428: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_42, [8, 576, 576, 16]);  mm_42 = None
    add_192: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_428, primals_422);  view_428 = primals_422 = None
    permute_279: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_192, [0, 3, 1, 2]);  add_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_299: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
    amax_21: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_299, [-1], True)
    sub_64: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_299, amax_21);  clone_299 = amax_21 = None
    exp_21: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_22: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_21: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    alias_21: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_280: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_21, [0, 2, 3, 1]);  div_21 = None
    permute_281: "f32[16, 16]" = torch.ops.aten.permute.default(primals_423, [1, 0]);  primals_423 = None
    clone_300: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_280, memory_format = torch.contiguous_format);  permute_280 = None
    view_429: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_300, [2654208, 16]);  clone_300 = None
    mm_43: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_429, permute_281)
    view_430: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_43, [8, 576, 576, 16]);  mm_43 = None
    add_193: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_430, primals_424);  view_430 = primals_424 = None
    permute_282: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_193, [0, 3, 1, 2]);  add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_301: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_282);  permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_86: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_301, [8, 16, 576, 576]);  clone_301 = None
    clone_302: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
    view_431: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_302, [128, 576, 576]);  clone_302 = None
    expand_87: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_65, [8, 16, 576, 48]);  select_65 = None
    clone_303: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
    view_432: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_303, [128, 576, 48]);  clone_303 = None
    bmm_43: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_431, view_432)
    view_433: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_43, [8, 16, 576, 48]);  bmm_43 = None
    permute_283: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_433, [0, 2, 1, 3]);  view_433 = None
    clone_304: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_283, memory_format = torch.contiguous_format);  permute_283 = None
    view_434: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_304, [8, 576, 768]);  clone_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_435: "f32[4608, 768]" = torch.ops.aten.view.default(view_434, [4608, 768]);  view_434 = None
    permute_284: "f32[768, 768]" = torch.ops.aten.permute.default(primals_425, [1, 0]);  primals_425 = None
    addmm_85: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_426, view_435, permute_284);  primals_426 = None
    view_436: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_85, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_305: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_436);  view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_213: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_44, clone_305);  clone_305 = None
    add_194: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_189, mul_213);  add_189 = mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_306: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_194, memory_format = torch.contiguous_format)
    var_mean_43 = torch.ops.aten.var_mean.correction(clone_306, [2], correction = 0, keepdim = True)
    getitem_86: "f32[8, 576, 1]" = var_mean_43[0]
    getitem_87: "f32[8, 576, 1]" = var_mean_43[1];  var_mean_43 = None
    add_195: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
    rsqrt_43: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
    sub_65: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_306, getitem_87);  clone_306 = getitem_87 = None
    mul_214: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = None
    mul_215: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_214, primals_427)
    add_196: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_215, primals_428);  mul_215 = primals_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_437: "f32[4608, 768]" = torch.ops.aten.view.default(add_196, [4608, 768]);  add_196 = None
    permute_285: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_429, [1, 0]);  primals_429 = None
    addmm_86: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_430, view_437, permute_285);  primals_430 = None
    view_438: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_86, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_216: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_438, 0.5)
    mul_217: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_438, 0.7071067811865476);  view_438 = None
    erf_21: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_217);  mul_217 = None
    add_197: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_218: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_216, add_197);  mul_216 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_307: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_218);  mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_439: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_307, [4608, 3072]);  clone_307 = None
    permute_286: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_431, [1, 0]);  primals_431 = None
    addmm_87: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_432, view_439, permute_286);  primals_432 = None
    view_440: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_87, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_308: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_440);  view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_219: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_45, clone_308);  clone_308 = None
    add_198: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_194, mul_219);  add_194 = mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_309: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_198, memory_format = torch.contiguous_format)
    var_mean_44 = torch.ops.aten.var_mean.correction(clone_309, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 576, 1]" = var_mean_44[0]
    getitem_89: "f32[8, 576, 1]" = var_mean_44[1];  var_mean_44 = None
    add_199: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_44: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
    sub_66: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_309, getitem_89);  clone_309 = getitem_89 = None
    mul_220: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = None
    mul_221: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_220, primals_433)
    add_200: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_221, primals_434);  mul_221 = primals_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_441: "f32[4608, 768]" = torch.ops.aten.view.default(add_200, [4608, 768]);  add_200 = None
    permute_287: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_435, [1, 0]);  primals_435 = None
    addmm_88: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_436, view_441, permute_287);  primals_436 = None
    view_442: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_88, [8, 576, 2304]);  addmm_88 = None
    view_443: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_442, [8, 576, 3, 16, 48]);  view_442 = None
    permute_288: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_443, [2, 0, 3, 1, 4]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_66: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_288, 0, 0)
    mul_222: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_66, 0.14433756729740643);  select_66 = None
    select_67: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_288, 0, 1)
    select_68: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_288, 0, 2);  permute_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_289: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_67, [0, 1, 3, 2]);  select_67 = None
    expand_88: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_222, [8, 16, 576, 48]);  mul_222 = None
    clone_310: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
    view_444: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_310, [128, 576, 48]);  clone_310 = None
    expand_89: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_289, [8, 16, 48, 576]);  permute_289 = None
    clone_311: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    view_445: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_311, [128, 48, 576]);  clone_311 = None
    bmm_44: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_444, view_445)
    view_446: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_44, [8, 16, 576, 576]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_290: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_446, [0, 2, 3, 1]);  view_446 = None
    permute_291: "f32[16, 16]" = torch.ops.aten.permute.default(primals_437, [1, 0]);  primals_437 = None
    clone_312: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None
    view_447: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_312, [2654208, 16]);  clone_312 = None
    mm_44: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_447, permute_291)
    view_448: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_44, [8, 576, 576, 16]);  mm_44 = None
    add_201: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_448, primals_438);  view_448 = primals_438 = None
    permute_292: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_201, [0, 3, 1, 2]);  add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_313: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    amax_22: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_313, [-1], True)
    sub_67: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_313, amax_22);  clone_313 = amax_22 = None
    exp_22: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
    sum_23: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    alias_22: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_293: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_22, [0, 2, 3, 1]);  div_22 = None
    permute_294: "f32[16, 16]" = torch.ops.aten.permute.default(primals_439, [1, 0]);  primals_439 = None
    clone_314: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_449: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_314, [2654208, 16]);  clone_314 = None
    mm_45: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_449, permute_294)
    view_450: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_45, [8, 576, 576, 16]);  mm_45 = None
    add_202: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_450, primals_440);  view_450 = primals_440 = None
    permute_295: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_202, [0, 3, 1, 2]);  add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_315: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_295);  permute_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_90: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_315, [8, 16, 576, 576]);  clone_315 = None
    clone_316: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
    view_451: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_316, [128, 576, 576]);  clone_316 = None
    expand_91: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_68, [8, 16, 576, 48]);  select_68 = None
    clone_317: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
    view_452: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_317, [128, 576, 48]);  clone_317 = None
    bmm_45: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_451, view_452)
    view_453: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_45, [8, 16, 576, 48]);  bmm_45 = None
    permute_296: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    clone_318: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_296, memory_format = torch.contiguous_format);  permute_296 = None
    view_454: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_318, [8, 576, 768]);  clone_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_455: "f32[4608, 768]" = torch.ops.aten.view.default(view_454, [4608, 768]);  view_454 = None
    permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(primals_441, [1, 0]);  primals_441 = None
    addmm_89: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_442, view_455, permute_297);  primals_442 = None
    view_456: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_89, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_319: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_456);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_223: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_46, clone_319);  clone_319 = None
    add_203: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_198, mul_223);  add_198 = mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_320: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_203, memory_format = torch.contiguous_format)
    var_mean_45 = torch.ops.aten.var_mean.correction(clone_320, [2], correction = 0, keepdim = True)
    getitem_90: "f32[8, 576, 1]" = var_mean_45[0]
    getitem_91: "f32[8, 576, 1]" = var_mean_45[1];  var_mean_45 = None
    add_204: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
    rsqrt_45: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
    sub_68: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_320, getitem_91);  clone_320 = getitem_91 = None
    mul_224: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = None
    mul_225: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_224, primals_443)
    add_205: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_225, primals_444);  mul_225 = primals_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_457: "f32[4608, 768]" = torch.ops.aten.view.default(add_205, [4608, 768]);  add_205 = None
    permute_298: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_445, [1, 0]);  primals_445 = None
    addmm_90: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_446, view_457, permute_298);  primals_446 = None
    view_458: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_90, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_226: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_458, 0.5)
    mul_227: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_458, 0.7071067811865476);  view_458 = None
    erf_22: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_227);  mul_227 = None
    add_206: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_228: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_226, add_206);  mul_226 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_321: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_228);  mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_459: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_321, [4608, 3072]);  clone_321 = None
    permute_299: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_447, [1, 0]);  primals_447 = None
    addmm_91: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_448, view_459, permute_299);  primals_448 = None
    view_460: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_91, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_322: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_460);  view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_229: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_47, clone_322);  clone_322 = None
    add_207: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_203, mul_229);  add_203 = mul_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_323: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_207, memory_format = torch.contiguous_format)
    var_mean_46 = torch.ops.aten.var_mean.correction(clone_323, [2], correction = 0, keepdim = True)
    getitem_92: "f32[8, 576, 1]" = var_mean_46[0]
    getitem_93: "f32[8, 576, 1]" = var_mean_46[1];  var_mean_46 = None
    add_208: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
    rsqrt_46: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    sub_69: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_323, getitem_93);  clone_323 = getitem_93 = None
    mul_230: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = None
    mul_231: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_230, primals_449)
    add_209: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_231, primals_450);  mul_231 = primals_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_461: "f32[4608, 768]" = torch.ops.aten.view.default(add_209, [4608, 768]);  add_209 = None
    permute_300: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_451, [1, 0]);  primals_451 = None
    addmm_92: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_452, view_461, permute_300);  primals_452 = None
    view_462: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_92, [8, 576, 2304]);  addmm_92 = None
    view_463: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_462, [8, 576, 3, 16, 48]);  view_462 = None
    permute_301: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_463, [2, 0, 3, 1, 4]);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_69: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_301, 0, 0)
    mul_232: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_69, 0.14433756729740643);  select_69 = None
    select_70: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_301, 0, 1)
    select_71: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_301, 0, 2);  permute_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_302: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_70, [0, 1, 3, 2]);  select_70 = None
    expand_92: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_232, [8, 16, 576, 48]);  mul_232 = None
    clone_324: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
    view_464: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_324, [128, 576, 48]);  clone_324 = None
    expand_93: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_302, [8, 16, 48, 576]);  permute_302 = None
    clone_325: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
    view_465: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_325, [128, 48, 576]);  clone_325 = None
    bmm_46: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_464, view_465)
    view_466: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_46, [8, 16, 576, 576]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_303: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_466, [0, 2, 3, 1]);  view_466 = None
    permute_304: "f32[16, 16]" = torch.ops.aten.permute.default(primals_453, [1, 0]);  primals_453 = None
    clone_326: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
    view_467: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_326, [2654208, 16]);  clone_326 = None
    mm_46: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_467, permute_304)
    view_468: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_46, [8, 576, 576, 16]);  mm_46 = None
    add_210: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_468, primals_454);  view_468 = primals_454 = None
    permute_305: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_210, [0, 3, 1, 2]);  add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_327: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_305, memory_format = torch.contiguous_format);  permute_305 = None
    amax_23: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_327, [-1], True)
    sub_70: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_327, amax_23);  clone_327 = amax_23 = None
    exp_23: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
    sum_24: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_23: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    alias_23: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_306: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_23, [0, 2, 3, 1]);  div_23 = None
    permute_307: "f32[16, 16]" = torch.ops.aten.permute.default(primals_455, [1, 0]);  primals_455 = None
    clone_328: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_306, memory_format = torch.contiguous_format);  permute_306 = None
    view_469: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_328, [2654208, 16]);  clone_328 = None
    mm_47: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_469, permute_307)
    view_470: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_47, [8, 576, 576, 16]);  mm_47 = None
    add_211: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_470, primals_456);  view_470 = primals_456 = None
    permute_308: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_211, [0, 3, 1, 2]);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_329: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_308);  permute_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_94: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_329, [8, 16, 576, 576]);  clone_329 = None
    clone_330: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
    view_471: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_330, [128, 576, 576]);  clone_330 = None
    expand_95: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_71, [8, 16, 576, 48]);  select_71 = None
    clone_331: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
    view_472: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_331, [128, 576, 48]);  clone_331 = None
    bmm_47: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_471, view_472)
    view_473: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_47, [8, 16, 576, 48]);  bmm_47 = None
    permute_309: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_473, [0, 2, 1, 3]);  view_473 = None
    clone_332: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
    view_474: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_332, [8, 576, 768]);  clone_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_475: "f32[4608, 768]" = torch.ops.aten.view.default(view_474, [4608, 768]);  view_474 = None
    permute_310: "f32[768, 768]" = torch.ops.aten.permute.default(primals_457, [1, 0]);  primals_457 = None
    addmm_93: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_458, view_475, permute_310);  primals_458 = None
    view_476: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_93, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_333: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_476);  view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_233: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_48, clone_333);  clone_333 = None
    add_212: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_207, mul_233);  add_207 = mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_334: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_212, memory_format = torch.contiguous_format)
    var_mean_47 = torch.ops.aten.var_mean.correction(clone_334, [2], correction = 0, keepdim = True)
    getitem_94: "f32[8, 576, 1]" = var_mean_47[0]
    getitem_95: "f32[8, 576, 1]" = var_mean_47[1];  var_mean_47 = None
    add_213: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
    rsqrt_47: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_71: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_334, getitem_95);  clone_334 = getitem_95 = None
    mul_234: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = None
    mul_235: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_234, primals_459)
    add_214: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_235, primals_460);  mul_235 = primals_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_477: "f32[4608, 768]" = torch.ops.aten.view.default(add_214, [4608, 768]);  add_214 = None
    permute_311: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_461, [1, 0]);  primals_461 = None
    addmm_94: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_462, view_477, permute_311);  primals_462 = None
    view_478: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_94, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_236: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_478, 0.5)
    mul_237: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_478, 0.7071067811865476);  view_478 = None
    erf_23: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_237);  mul_237 = None
    add_215: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_238: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_236, add_215);  mul_236 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_335: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_238);  mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_479: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_335, [4608, 3072]);  clone_335 = None
    permute_312: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_463, [1, 0]);  primals_463 = None
    addmm_95: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_464, view_479, permute_312);  primals_464 = None
    view_480: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_95, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_336: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_480);  view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_239: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_49, clone_336);  clone_336 = None
    add_216: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_212, mul_239);  add_212 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_337: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_216, memory_format = torch.contiguous_format)
    var_mean_48 = torch.ops.aten.var_mean.correction(clone_337, [2], correction = 0, keepdim = True)
    getitem_96: "f32[8, 576, 1]" = var_mean_48[0]
    getitem_97: "f32[8, 576, 1]" = var_mean_48[1];  var_mean_48 = None
    add_217: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
    rsqrt_48: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
    sub_72: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_337, getitem_97);  clone_337 = getitem_97 = None
    mul_240: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = None
    mul_241: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_240, primals_465)
    add_218: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_241, primals_466);  mul_241 = primals_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_481: "f32[4608, 768]" = torch.ops.aten.view.default(add_218, [4608, 768]);  add_218 = None
    permute_313: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_467, [1, 0]);  primals_467 = None
    addmm_96: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_468, view_481, permute_313);  primals_468 = None
    view_482: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_96, [8, 576, 2304]);  addmm_96 = None
    view_483: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_482, [8, 576, 3, 16, 48]);  view_482 = None
    permute_314: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_483, [2, 0, 3, 1, 4]);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_72: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_314, 0, 0)
    mul_242: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_72, 0.14433756729740643);  select_72 = None
    select_73: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_314, 0, 1)
    select_74: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_314, 0, 2);  permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_315: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_73, [0, 1, 3, 2]);  select_73 = None
    expand_96: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_242, [8, 16, 576, 48]);  mul_242 = None
    clone_338: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
    view_484: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_338, [128, 576, 48]);  clone_338 = None
    expand_97: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_315, [8, 16, 48, 576]);  permute_315 = None
    clone_339: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
    view_485: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_339, [128, 48, 576]);  clone_339 = None
    bmm_48: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_484, view_485)
    view_486: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_48, [8, 16, 576, 576]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_316: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_486, [0, 2, 3, 1]);  view_486 = None
    permute_317: "f32[16, 16]" = torch.ops.aten.permute.default(primals_469, [1, 0]);  primals_469 = None
    clone_340: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
    view_487: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_340, [2654208, 16]);  clone_340 = None
    mm_48: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_487, permute_317)
    view_488: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_48, [8, 576, 576, 16]);  mm_48 = None
    add_219: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_488, primals_470);  view_488 = primals_470 = None
    permute_318: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_219, [0, 3, 1, 2]);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_341: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
    amax_24: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_341, [-1], True)
    sub_73: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_341, amax_24);  clone_341 = amax_24 = None
    exp_24: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
    sum_25: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
    div_24: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
    alias_24: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_319: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_24, [0, 2, 3, 1]);  div_24 = None
    permute_320: "f32[16, 16]" = torch.ops.aten.permute.default(primals_471, [1, 0]);  primals_471 = None
    clone_342: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_489: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_342, [2654208, 16]);  clone_342 = None
    mm_49: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_489, permute_320)
    view_490: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_49, [8, 576, 576, 16]);  mm_49 = None
    add_220: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_490, primals_472);  view_490 = primals_472 = None
    permute_321: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_220, [0, 3, 1, 2]);  add_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_343: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_321);  permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_98: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_343, [8, 16, 576, 576]);  clone_343 = None
    clone_344: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
    view_491: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_344, [128, 576, 576]);  clone_344 = None
    expand_99: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_74, [8, 16, 576, 48]);  select_74 = None
    clone_345: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
    view_492: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_345, [128, 576, 48]);  clone_345 = None
    bmm_49: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_491, view_492)
    view_493: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_49, [8, 16, 576, 48]);  bmm_49 = None
    permute_322: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    clone_346: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
    view_494: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_346, [8, 576, 768]);  clone_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_495: "f32[4608, 768]" = torch.ops.aten.view.default(view_494, [4608, 768]);  view_494 = None
    permute_323: "f32[768, 768]" = torch.ops.aten.permute.default(primals_473, [1, 0]);  primals_473 = None
    addmm_97: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_474, view_495, permute_323);  primals_474 = None
    view_496: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_97, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_347: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_496);  view_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_243: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_50, clone_347);  clone_347 = None
    add_221: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_216, mul_243);  add_216 = mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_348: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_221, memory_format = torch.contiguous_format)
    var_mean_49 = torch.ops.aten.var_mean.correction(clone_348, [2], correction = 0, keepdim = True)
    getitem_98: "f32[8, 576, 1]" = var_mean_49[0]
    getitem_99: "f32[8, 576, 1]" = var_mean_49[1];  var_mean_49 = None
    add_222: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-06);  getitem_98 = None
    rsqrt_49: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_222);  add_222 = None
    sub_74: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_348, getitem_99);  clone_348 = getitem_99 = None
    mul_244: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_49);  sub_74 = None
    mul_245: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_244, primals_475)
    add_223: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_245, primals_476);  mul_245 = primals_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_497: "f32[4608, 768]" = torch.ops.aten.view.default(add_223, [4608, 768]);  add_223 = None
    permute_324: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_477, [1, 0]);  primals_477 = None
    addmm_98: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_478, view_497, permute_324);  primals_478 = None
    view_498: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_98, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_246: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_498, 0.5)
    mul_247: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_498, 0.7071067811865476);  view_498 = None
    erf_24: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_247);  mul_247 = None
    add_224: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_248: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_246, add_224);  mul_246 = add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_349: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_248);  mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_499: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_349, [4608, 3072]);  clone_349 = None
    permute_325: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_479, [1, 0]);  primals_479 = None
    addmm_99: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_480, view_499, permute_325);  primals_480 = None
    view_500: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_99, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_350: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_500);  view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_249: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_51, clone_350);  clone_350 = None
    add_225: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_221, mul_249);  add_221 = mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_351: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_225, memory_format = torch.contiguous_format)
    var_mean_50 = torch.ops.aten.var_mean.correction(clone_351, [2], correction = 0, keepdim = True)
    getitem_100: "f32[8, 576, 1]" = var_mean_50[0]
    getitem_101: "f32[8, 576, 1]" = var_mean_50[1];  var_mean_50 = None
    add_226: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-06);  getitem_100 = None
    rsqrt_50: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
    sub_75: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_351, getitem_101);  clone_351 = getitem_101 = None
    mul_250: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_50);  sub_75 = None
    mul_251: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_250, primals_481)
    add_227: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_251, primals_482);  mul_251 = primals_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_501: "f32[4608, 768]" = torch.ops.aten.view.default(add_227, [4608, 768]);  add_227 = None
    permute_326: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_483, [1, 0]);  primals_483 = None
    addmm_100: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_484, view_501, permute_326);  primals_484 = None
    view_502: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_100, [8, 576, 2304]);  addmm_100 = None
    view_503: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_502, [8, 576, 3, 16, 48]);  view_502 = None
    permute_327: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_503, [2, 0, 3, 1, 4]);  view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_75: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_327, 0, 0)
    mul_252: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_75, 0.14433756729740643);  select_75 = None
    select_76: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_327, 0, 1)
    select_77: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_327, 0, 2);  permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_328: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_76, [0, 1, 3, 2]);  select_76 = None
    expand_100: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_252, [8, 16, 576, 48]);  mul_252 = None
    clone_352: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
    view_504: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_352, [128, 576, 48]);  clone_352 = None
    expand_101: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_328, [8, 16, 48, 576]);  permute_328 = None
    clone_353: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
    view_505: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_353, [128, 48, 576]);  clone_353 = None
    bmm_50: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_504, view_505)
    view_506: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_50, [8, 16, 576, 576]);  bmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_329: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_506, [0, 2, 3, 1]);  view_506 = None
    permute_330: "f32[16, 16]" = torch.ops.aten.permute.default(primals_485, [1, 0]);  primals_485 = None
    clone_354: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
    view_507: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_354, [2654208, 16]);  clone_354 = None
    mm_50: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_507, permute_330)
    view_508: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_50, [8, 576, 576, 16]);  mm_50 = None
    add_228: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_508, primals_486);  view_508 = primals_486 = None
    permute_331: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_228, [0, 3, 1, 2]);  add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_355: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_331, memory_format = torch.contiguous_format);  permute_331 = None
    amax_25: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_355, [-1], True)
    sub_76: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_355, amax_25);  clone_355 = amax_25 = None
    exp_25: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
    sum_26: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
    div_25: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
    alias_25: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_332: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_25, [0, 2, 3, 1]);  div_25 = None
    permute_333: "f32[16, 16]" = torch.ops.aten.permute.default(primals_487, [1, 0]);  primals_487 = None
    clone_356: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_332, memory_format = torch.contiguous_format);  permute_332 = None
    view_509: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_356, [2654208, 16]);  clone_356 = None
    mm_51: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_509, permute_333)
    view_510: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_51, [8, 576, 576, 16]);  mm_51 = None
    add_229: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_510, primals_488);  view_510 = primals_488 = None
    permute_334: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_229, [0, 3, 1, 2]);  add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_357: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_334);  permute_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_102: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_357, [8, 16, 576, 576]);  clone_357 = None
    clone_358: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_102, memory_format = torch.contiguous_format);  expand_102 = None
    view_511: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_358, [128, 576, 576]);  clone_358 = None
    expand_103: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_77, [8, 16, 576, 48]);  select_77 = None
    clone_359: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
    view_512: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_359, [128, 576, 48]);  clone_359 = None
    bmm_51: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_511, view_512)
    view_513: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_51, [8, 16, 576, 48]);  bmm_51 = None
    permute_335: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None
    clone_360: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
    view_514: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_360, [8, 576, 768]);  clone_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_515: "f32[4608, 768]" = torch.ops.aten.view.default(view_514, [4608, 768]);  view_514 = None
    permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(primals_489, [1, 0]);  primals_489 = None
    addmm_101: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_490, view_515, permute_336);  primals_490 = None
    view_516: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_101, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_361: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_516);  view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_253: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_52, clone_361);  clone_361 = None
    add_230: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_225, mul_253);  add_225 = mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_362: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_230, memory_format = torch.contiguous_format)
    var_mean_51 = torch.ops.aten.var_mean.correction(clone_362, [2], correction = 0, keepdim = True)
    getitem_102: "f32[8, 576, 1]" = var_mean_51[0]
    getitem_103: "f32[8, 576, 1]" = var_mean_51[1];  var_mean_51 = None
    add_231: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-06);  getitem_102 = None
    rsqrt_51: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_77: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_362, getitem_103);  clone_362 = getitem_103 = None
    mul_254: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_51);  sub_77 = None
    mul_255: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_254, primals_491)
    add_232: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_255, primals_492);  mul_255 = primals_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_517: "f32[4608, 768]" = torch.ops.aten.view.default(add_232, [4608, 768]);  add_232 = None
    permute_337: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_493, [1, 0]);  primals_493 = None
    addmm_102: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_494, view_517, permute_337);  primals_494 = None
    view_518: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_102, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_256: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_518, 0.5)
    mul_257: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_518, 0.7071067811865476);  view_518 = None
    erf_25: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_257);  mul_257 = None
    add_233: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_258: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_256, add_233);  mul_256 = add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_363: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_258);  mul_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_519: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_363, [4608, 3072]);  clone_363 = None
    permute_338: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_495, [1, 0]);  primals_495 = None
    addmm_103: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_496, view_519, permute_338);  primals_496 = None
    view_520: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_103, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_364: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_520);  view_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_259: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_53, clone_364);  clone_364 = None
    add_234: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_230, mul_259);  add_230 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_365: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_234, memory_format = torch.contiguous_format)
    var_mean_52 = torch.ops.aten.var_mean.correction(clone_365, [2], correction = 0, keepdim = True)
    getitem_104: "f32[8, 576, 1]" = var_mean_52[0]
    getitem_105: "f32[8, 576, 1]" = var_mean_52[1];  var_mean_52 = None
    add_235: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
    rsqrt_52: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    sub_78: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_365, getitem_105);  clone_365 = getitem_105 = None
    mul_260: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_52);  sub_78 = None
    mul_261: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_260, primals_497)
    add_236: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_261, primals_498);  mul_261 = primals_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_521: "f32[4608, 768]" = torch.ops.aten.view.default(add_236, [4608, 768]);  add_236 = None
    permute_339: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_499, [1, 0]);  primals_499 = None
    addmm_104: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_500, view_521, permute_339);  primals_500 = None
    view_522: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_104, [8, 576, 2304]);  addmm_104 = None
    view_523: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_522, [8, 576, 3, 16, 48]);  view_522 = None
    permute_340: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_523, [2, 0, 3, 1, 4]);  view_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_78: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_340, 0, 0)
    mul_262: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_78, 0.14433756729740643);  select_78 = None
    select_79: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_340, 0, 1)
    select_80: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_340, 0, 2);  permute_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_341: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_79, [0, 1, 3, 2]);  select_79 = None
    expand_104: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_262, [8, 16, 576, 48]);  mul_262 = None
    clone_366: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
    view_524: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_366, [128, 576, 48]);  clone_366 = None
    expand_105: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_341, [8, 16, 48, 576]);  permute_341 = None
    clone_367: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
    view_525: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_367, [128, 48, 576]);  clone_367 = None
    bmm_52: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_524, view_525)
    view_526: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_52, [8, 16, 576, 576]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_342: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_526, [0, 2, 3, 1]);  view_526 = None
    permute_343: "f32[16, 16]" = torch.ops.aten.permute.default(primals_501, [1, 0]);  primals_501 = None
    clone_368: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_342, memory_format = torch.contiguous_format);  permute_342 = None
    view_527: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_368, [2654208, 16]);  clone_368 = None
    mm_52: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_527, permute_343)
    view_528: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_52, [8, 576, 576, 16]);  mm_52 = None
    add_237: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_528, primals_502);  view_528 = primals_502 = None
    permute_344: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_237, [0, 3, 1, 2]);  add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_369: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_344, memory_format = torch.contiguous_format);  permute_344 = None
    amax_26: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_369, [-1], True)
    sub_79: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_369, amax_26);  clone_369 = amax_26 = None
    exp_26: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_79);  sub_79 = None
    sum_27: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
    div_26: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
    alias_26: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_345: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_26, [0, 2, 3, 1]);  div_26 = None
    permute_346: "f32[16, 16]" = torch.ops.aten.permute.default(primals_503, [1, 0]);  primals_503 = None
    clone_370: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_345, memory_format = torch.contiguous_format);  permute_345 = None
    view_529: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_370, [2654208, 16]);  clone_370 = None
    mm_53: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_529, permute_346)
    view_530: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_53, [8, 576, 576, 16]);  mm_53 = None
    add_238: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_530, primals_504);  view_530 = primals_504 = None
    permute_347: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_238, [0, 3, 1, 2]);  add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_371: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_347);  permute_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_106: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_371, [8, 16, 576, 576]);  clone_371 = None
    clone_372: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
    view_531: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_372, [128, 576, 576]);  clone_372 = None
    expand_107: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_80, [8, 16, 576, 48]);  select_80 = None
    clone_373: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
    view_532: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_373, [128, 576, 48]);  clone_373 = None
    bmm_53: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_531, view_532)
    view_533: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_53, [8, 16, 576, 48]);  bmm_53 = None
    permute_348: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_533, [0, 2, 1, 3]);  view_533 = None
    clone_374: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
    view_534: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_374, [8, 576, 768]);  clone_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_535: "f32[4608, 768]" = torch.ops.aten.view.default(view_534, [4608, 768]);  view_534 = None
    permute_349: "f32[768, 768]" = torch.ops.aten.permute.default(primals_505, [1, 0]);  primals_505 = None
    addmm_105: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_506, view_535, permute_349);  primals_506 = None
    view_536: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_105, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_375: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_536);  view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_263: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_54, clone_375);  clone_375 = None
    add_239: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_234, mul_263);  add_234 = mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_376: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_239, memory_format = torch.contiguous_format)
    var_mean_53 = torch.ops.aten.var_mean.correction(clone_376, [2], correction = 0, keepdim = True)
    getitem_106: "f32[8, 576, 1]" = var_mean_53[0]
    getitem_107: "f32[8, 576, 1]" = var_mean_53[1];  var_mean_53 = None
    add_240: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
    rsqrt_53: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
    sub_80: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_376, getitem_107);  clone_376 = getitem_107 = None
    mul_264: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_53);  sub_80 = None
    mul_265: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_264, primals_507)
    add_241: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_265, primals_508);  mul_265 = primals_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_537: "f32[4608, 768]" = torch.ops.aten.view.default(add_241, [4608, 768]);  add_241 = None
    permute_350: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_509, [1, 0]);  primals_509 = None
    addmm_106: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_510, view_537, permute_350);  primals_510 = None
    view_538: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_106, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_266: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_538, 0.5)
    mul_267: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_538, 0.7071067811865476);  view_538 = None
    erf_26: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_267);  mul_267 = None
    add_242: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_268: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_266, add_242);  mul_266 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_377: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_268);  mul_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_539: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_377, [4608, 3072]);  clone_377 = None
    permute_351: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_511, [1, 0]);  primals_511 = None
    addmm_107: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_512, view_539, permute_351);  primals_512 = None
    view_540: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_107, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_378: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_540);  view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_269: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_55, clone_378);  clone_378 = None
    add_243: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_239, mul_269);  add_239 = mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_379: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_243, memory_format = torch.contiguous_format)
    var_mean_54 = torch.ops.aten.var_mean.correction(clone_379, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 576, 1]" = var_mean_54[0]
    getitem_109: "f32[8, 576, 1]" = var_mean_54[1];  var_mean_54 = None
    add_244: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_54: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
    sub_81: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_379, getitem_109);  clone_379 = getitem_109 = None
    mul_270: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_54);  sub_81 = None
    mul_271: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_270, primals_513)
    add_245: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_271, primals_514);  mul_271 = primals_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_541: "f32[4608, 768]" = torch.ops.aten.view.default(add_245, [4608, 768]);  add_245 = None
    permute_352: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_515, [1, 0]);  primals_515 = None
    addmm_108: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_516, view_541, permute_352);  primals_516 = None
    view_542: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_108, [8, 576, 2304]);  addmm_108 = None
    view_543: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_542, [8, 576, 3, 16, 48]);  view_542 = None
    permute_353: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_543, [2, 0, 3, 1, 4]);  view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_81: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_353, 0, 0)
    mul_272: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_81, 0.14433756729740643);  select_81 = None
    select_82: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_353, 0, 1)
    select_83: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_353, 0, 2);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_354: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_82, [0, 1, 3, 2]);  select_82 = None
    expand_108: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_272, [8, 16, 576, 48]);  mul_272 = None
    clone_380: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_108, memory_format = torch.contiguous_format);  expand_108 = None
    view_544: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_380, [128, 576, 48]);  clone_380 = None
    expand_109: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_354, [8, 16, 48, 576]);  permute_354 = None
    clone_381: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_109, memory_format = torch.contiguous_format);  expand_109 = None
    view_545: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_381, [128, 48, 576]);  clone_381 = None
    bmm_54: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_544, view_545)
    view_546: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_54, [8, 16, 576, 576]);  bmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_355: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_546, [0, 2, 3, 1]);  view_546 = None
    permute_356: "f32[16, 16]" = torch.ops.aten.permute.default(primals_517, [1, 0]);  primals_517 = None
    clone_382: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
    view_547: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_382, [2654208, 16]);  clone_382 = None
    mm_54: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_547, permute_356)
    view_548: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_54, [8, 576, 576, 16]);  mm_54 = None
    add_246: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_548, primals_518);  view_548 = primals_518 = None
    permute_357: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_246, [0, 3, 1, 2]);  add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_383: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
    amax_27: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_383, [-1], True)
    sub_82: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_383, amax_27);  clone_383 = amax_27 = None
    exp_27: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_82);  sub_82 = None
    sum_28: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
    div_27: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
    alias_27: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_358: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_27, [0, 2, 3, 1]);  div_27 = None
    permute_359: "f32[16, 16]" = torch.ops.aten.permute.default(primals_519, [1, 0]);  primals_519 = None
    clone_384: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
    view_549: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_384, [2654208, 16]);  clone_384 = None
    mm_55: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_549, permute_359)
    view_550: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_55, [8, 576, 576, 16]);  mm_55 = None
    add_247: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_550, primals_520);  view_550 = primals_520 = None
    permute_360: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_247, [0, 3, 1, 2]);  add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_385: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_360);  permute_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_110: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_385, [8, 16, 576, 576]);  clone_385 = None
    clone_386: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_110, memory_format = torch.contiguous_format);  expand_110 = None
    view_551: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_386, [128, 576, 576]);  clone_386 = None
    expand_111: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_83, [8, 16, 576, 48]);  select_83 = None
    clone_387: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
    view_552: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_387, [128, 576, 48]);  clone_387 = None
    bmm_55: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_551, view_552)
    view_553: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_55, [8, 16, 576, 48]);  bmm_55 = None
    permute_361: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_553, [0, 2, 1, 3]);  view_553 = None
    clone_388: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_361, memory_format = torch.contiguous_format);  permute_361 = None
    view_554: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_388, [8, 576, 768]);  clone_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_555: "f32[4608, 768]" = torch.ops.aten.view.default(view_554, [4608, 768]);  view_554 = None
    permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(primals_521, [1, 0]);  primals_521 = None
    addmm_109: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_522, view_555, permute_362);  primals_522 = None
    view_556: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_109, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_389: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_556);  view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_273: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_56, clone_389);  clone_389 = None
    add_248: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_243, mul_273);  add_243 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_390: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_248, memory_format = torch.contiguous_format)
    var_mean_55 = torch.ops.aten.var_mean.correction(clone_390, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 576, 1]" = var_mean_55[0]
    getitem_111: "f32[8, 576, 1]" = var_mean_55[1];  var_mean_55 = None
    add_249: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
    rsqrt_55: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_249);  add_249 = None
    sub_83: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_390, getitem_111);  clone_390 = getitem_111 = None
    mul_274: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_55);  sub_83 = None
    mul_275: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_274, primals_523)
    add_250: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_275, primals_524);  mul_275 = primals_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_557: "f32[4608, 768]" = torch.ops.aten.view.default(add_250, [4608, 768]);  add_250 = None
    permute_363: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_525, [1, 0]);  primals_525 = None
    addmm_110: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_526, view_557, permute_363);  primals_526 = None
    view_558: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_110, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_276: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_558, 0.5)
    mul_277: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_558, 0.7071067811865476);  view_558 = None
    erf_27: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_277);  mul_277 = None
    add_251: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_278: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_276, add_251);  mul_276 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_391: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_278);  mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_559: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_391, [4608, 3072]);  clone_391 = None
    permute_364: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_527, [1, 0]);  primals_527 = None
    addmm_111: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_528, view_559, permute_364);  primals_528 = None
    view_560: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_111, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_392: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_560);  view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_279: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_57, clone_392);  clone_392 = None
    add_252: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_248, mul_279);  add_248 = mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_393: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_252, memory_format = torch.contiguous_format)
    var_mean_56 = torch.ops.aten.var_mean.correction(clone_393, [2], correction = 0, keepdim = True)
    getitem_112: "f32[8, 576, 1]" = var_mean_56[0]
    getitem_113: "f32[8, 576, 1]" = var_mean_56[1];  var_mean_56 = None
    add_253: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
    rsqrt_56: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_253);  add_253 = None
    sub_84: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_393, getitem_113);  clone_393 = getitem_113 = None
    mul_280: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_56);  sub_84 = None
    mul_281: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_280, primals_529)
    add_254: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_281, primals_530);  mul_281 = primals_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_561: "f32[4608, 768]" = torch.ops.aten.view.default(add_254, [4608, 768]);  add_254 = None
    permute_365: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_531, [1, 0]);  primals_531 = None
    addmm_112: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_532, view_561, permute_365);  primals_532 = None
    view_562: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_112, [8, 576, 2304]);  addmm_112 = None
    view_563: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_562, [8, 576, 3, 16, 48]);  view_562 = None
    permute_366: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_563, [2, 0, 3, 1, 4]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_84: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_366, 0, 0)
    mul_282: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_84, 0.14433756729740643);  select_84 = None
    select_85: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_366, 0, 1)
    select_86: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_366, 0, 2);  permute_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_367: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_85, [0, 1, 3, 2]);  select_85 = None
    expand_112: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_282, [8, 16, 576, 48]);  mul_282 = None
    clone_394: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_112, memory_format = torch.contiguous_format);  expand_112 = None
    view_564: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_394, [128, 576, 48]);  clone_394 = None
    expand_113: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_367, [8, 16, 48, 576]);  permute_367 = None
    clone_395: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
    view_565: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_395, [128, 48, 576]);  clone_395 = None
    bmm_56: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_564, view_565)
    view_566: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_56, [8, 16, 576, 576]);  bmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_368: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_566, [0, 2, 3, 1]);  view_566 = None
    permute_369: "f32[16, 16]" = torch.ops.aten.permute.default(primals_533, [1, 0]);  primals_533 = None
    clone_396: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
    view_567: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_396, [2654208, 16]);  clone_396 = None
    mm_56: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_567, permute_369)
    view_568: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_56, [8, 576, 576, 16]);  mm_56 = None
    add_255: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_568, primals_534);  view_568 = primals_534 = None
    permute_370: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_255, [0, 3, 1, 2]);  add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_397: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    amax_28: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_397, [-1], True)
    sub_85: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_397, amax_28);  clone_397 = amax_28 = None
    exp_28: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_85);  sub_85 = None
    sum_29: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [-1], True)
    div_28: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_28, sum_29);  exp_28 = sum_29 = None
    alias_28: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_371: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_28, [0, 2, 3, 1]);  div_28 = None
    permute_372: "f32[16, 16]" = torch.ops.aten.permute.default(primals_535, [1, 0]);  primals_535 = None
    clone_398: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_371, memory_format = torch.contiguous_format);  permute_371 = None
    view_569: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_398, [2654208, 16]);  clone_398 = None
    mm_57: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_569, permute_372)
    view_570: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_57, [8, 576, 576, 16]);  mm_57 = None
    add_256: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_570, primals_536);  view_570 = primals_536 = None
    permute_373: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_256, [0, 3, 1, 2]);  add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_399: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_373);  permute_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_114: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_399, [8, 16, 576, 576]);  clone_399 = None
    clone_400: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_114, memory_format = torch.contiguous_format);  expand_114 = None
    view_571: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_400, [128, 576, 576]);  clone_400 = None
    expand_115: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_86, [8, 16, 576, 48]);  select_86 = None
    clone_401: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
    view_572: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_401, [128, 576, 48]);  clone_401 = None
    bmm_57: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_571, view_572)
    view_573: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_57, [8, 16, 576, 48]);  bmm_57 = None
    permute_374: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
    clone_402: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
    view_574: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_402, [8, 576, 768]);  clone_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_575: "f32[4608, 768]" = torch.ops.aten.view.default(view_574, [4608, 768]);  view_574 = None
    permute_375: "f32[768, 768]" = torch.ops.aten.permute.default(primals_537, [1, 0]);  primals_537 = None
    addmm_113: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_538, view_575, permute_375);  primals_538 = None
    view_576: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_113, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_403: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_576);  view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_283: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_58, clone_403);  clone_403 = None
    add_257: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_252, mul_283);  add_252 = mul_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_404: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_257, memory_format = torch.contiguous_format)
    var_mean_57 = torch.ops.aten.var_mean.correction(clone_404, [2], correction = 0, keepdim = True)
    getitem_114: "f32[8, 576, 1]" = var_mean_57[0]
    getitem_115: "f32[8, 576, 1]" = var_mean_57[1];  var_mean_57 = None
    add_258: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-06);  getitem_114 = None
    rsqrt_57: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
    sub_86: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_404, getitem_115);  clone_404 = getitem_115 = None
    mul_284: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_57);  sub_86 = None
    mul_285: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_284, primals_539)
    add_259: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_285, primals_540);  mul_285 = primals_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_577: "f32[4608, 768]" = torch.ops.aten.view.default(add_259, [4608, 768]);  add_259 = None
    permute_376: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_541, [1, 0]);  primals_541 = None
    addmm_114: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_542, view_577, permute_376);  primals_542 = None
    view_578: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_114, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_286: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_578, 0.5)
    mul_287: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_578, 0.7071067811865476);  view_578 = None
    erf_28: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_287);  mul_287 = None
    add_260: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_288: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_286, add_260);  mul_286 = add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_405: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_288);  mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_579: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_405, [4608, 3072]);  clone_405 = None
    permute_377: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_543, [1, 0]);  primals_543 = None
    addmm_115: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_544, view_579, permute_377);  primals_544 = None
    view_580: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_115, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_406: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_580);  view_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_289: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_59, clone_406);  clone_406 = None
    add_261: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_257, mul_289);  add_257 = mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_407: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_261, memory_format = torch.contiguous_format)
    var_mean_58 = torch.ops.aten.var_mean.correction(clone_407, [2], correction = 0, keepdim = True)
    getitem_116: "f32[8, 576, 1]" = var_mean_58[0]
    getitem_117: "f32[8, 576, 1]" = var_mean_58[1];  var_mean_58 = None
    add_262: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-06);  getitem_116 = None
    rsqrt_58: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
    sub_87: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_407, getitem_117);  clone_407 = getitem_117 = None
    mul_290: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_58);  sub_87 = None
    mul_291: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_290, primals_545)
    add_263: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_291, primals_546);  mul_291 = primals_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_581: "f32[4608, 768]" = torch.ops.aten.view.default(add_263, [4608, 768]);  add_263 = None
    permute_378: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_547, [1, 0]);  primals_547 = None
    addmm_116: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_548, view_581, permute_378);  primals_548 = None
    view_582: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_116, [8, 576, 2304]);  addmm_116 = None
    view_583: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_582, [8, 576, 3, 16, 48]);  view_582 = None
    permute_379: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_583, [2, 0, 3, 1, 4]);  view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_87: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_379, 0, 0)
    mul_292: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_87, 0.14433756729740643);  select_87 = None
    select_88: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_379, 0, 1)
    select_89: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_379, 0, 2);  permute_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_380: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_88, [0, 1, 3, 2]);  select_88 = None
    expand_116: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_292, [8, 16, 576, 48]);  mul_292 = None
    clone_408: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
    view_584: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_408, [128, 576, 48]);  clone_408 = None
    expand_117: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_380, [8, 16, 48, 576]);  permute_380 = None
    clone_409: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
    view_585: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_409, [128, 48, 576]);  clone_409 = None
    bmm_58: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_584, view_585)
    view_586: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_58, [8, 16, 576, 576]);  bmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_381: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_586, [0, 2, 3, 1]);  view_586 = None
    permute_382: "f32[16, 16]" = torch.ops.aten.permute.default(primals_549, [1, 0]);  primals_549 = None
    clone_410: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_381, memory_format = torch.contiguous_format);  permute_381 = None
    view_587: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_410, [2654208, 16]);  clone_410 = None
    mm_58: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_587, permute_382)
    view_588: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_58, [8, 576, 576, 16]);  mm_58 = None
    add_264: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_588, primals_550);  view_588 = primals_550 = None
    permute_383: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_264, [0, 3, 1, 2]);  add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_411: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_383, memory_format = torch.contiguous_format);  permute_383 = None
    amax_29: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_411, [-1], True)
    sub_88: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_411, amax_29);  clone_411 = amax_29 = None
    exp_29: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_88);  sub_88 = None
    sum_30: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_29, [-1], True)
    div_29: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_29, sum_30);  exp_29 = sum_30 = None
    alias_29: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_384: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_29, [0, 2, 3, 1]);  div_29 = None
    permute_385: "f32[16, 16]" = torch.ops.aten.permute.default(primals_551, [1, 0]);  primals_551 = None
    clone_412: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_384, memory_format = torch.contiguous_format);  permute_384 = None
    view_589: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_412, [2654208, 16]);  clone_412 = None
    mm_59: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_589, permute_385)
    view_590: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_59, [8, 576, 576, 16]);  mm_59 = None
    add_265: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_590, primals_552);  view_590 = primals_552 = None
    permute_386: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_265, [0, 3, 1, 2]);  add_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_413: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_386);  permute_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_118: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_413, [8, 16, 576, 576]);  clone_413 = None
    clone_414: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_118, memory_format = torch.contiguous_format);  expand_118 = None
    view_591: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_414, [128, 576, 576]);  clone_414 = None
    expand_119: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_89, [8, 16, 576, 48]);  select_89 = None
    clone_415: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_119, memory_format = torch.contiguous_format);  expand_119 = None
    view_592: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_415, [128, 576, 48]);  clone_415 = None
    bmm_59: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_591, view_592)
    view_593: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_59, [8, 16, 576, 48]);  bmm_59 = None
    permute_387: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_593, [0, 2, 1, 3]);  view_593 = None
    clone_416: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    view_594: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_416, [8, 576, 768]);  clone_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_595: "f32[4608, 768]" = torch.ops.aten.view.default(view_594, [4608, 768]);  view_594 = None
    permute_388: "f32[768, 768]" = torch.ops.aten.permute.default(primals_553, [1, 0]);  primals_553 = None
    addmm_117: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_554, view_595, permute_388);  primals_554 = None
    view_596: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_117, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_417: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_596);  view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_293: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_60, clone_417);  clone_417 = None
    add_266: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_261, mul_293);  add_261 = mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_418: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_266, memory_format = torch.contiguous_format)
    var_mean_59 = torch.ops.aten.var_mean.correction(clone_418, [2], correction = 0, keepdim = True)
    getitem_118: "f32[8, 576, 1]" = var_mean_59[0]
    getitem_119: "f32[8, 576, 1]" = var_mean_59[1];  var_mean_59 = None
    add_267: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-06);  getitem_118 = None
    rsqrt_59: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
    sub_89: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_418, getitem_119);  clone_418 = getitem_119 = None
    mul_294: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_59);  sub_89 = None
    mul_295: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_294, primals_555)
    add_268: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_295, primals_556);  mul_295 = primals_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_597: "f32[4608, 768]" = torch.ops.aten.view.default(add_268, [4608, 768]);  add_268 = None
    permute_389: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_557, [1, 0]);  primals_557 = None
    addmm_118: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_558, view_597, permute_389);  primals_558 = None
    view_598: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_118, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_296: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_598, 0.5)
    mul_297: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_598, 0.7071067811865476);  view_598 = None
    erf_29: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_297);  mul_297 = None
    add_269: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_298: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_296, add_269);  mul_296 = add_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_419: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_298);  mul_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_599: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_419, [4608, 3072]);  clone_419 = None
    permute_390: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_559, [1, 0]);  primals_559 = None
    addmm_119: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_560, view_599, permute_390);  primals_560 = None
    view_600: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_119, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_420: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_600);  view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_299: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_61, clone_420);  clone_420 = None
    add_270: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_266, mul_299);  add_266 = mul_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_421: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_270, memory_format = torch.contiguous_format)
    var_mean_60 = torch.ops.aten.var_mean.correction(clone_421, [2], correction = 0, keepdim = True)
    getitem_120: "f32[8, 576, 1]" = var_mean_60[0]
    getitem_121: "f32[8, 576, 1]" = var_mean_60[1];  var_mean_60 = None
    add_271: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-06);  getitem_120 = None
    rsqrt_60: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
    sub_90: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_421, getitem_121);  clone_421 = getitem_121 = None
    mul_300: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_60);  sub_90 = None
    mul_301: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_300, primals_561)
    add_272: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_301, primals_562);  mul_301 = primals_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_601: "f32[4608, 768]" = torch.ops.aten.view.default(add_272, [4608, 768]);  add_272 = None
    permute_391: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_563, [1, 0]);  primals_563 = None
    addmm_120: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_564, view_601, permute_391);  primals_564 = None
    view_602: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_120, [8, 576, 2304]);  addmm_120 = None
    view_603: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_602, [8, 576, 3, 16, 48]);  view_602 = None
    permute_392: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_603, [2, 0, 3, 1, 4]);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_90: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_392, 0, 0)
    mul_302: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_90, 0.14433756729740643);  select_90 = None
    select_91: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_392, 0, 1)
    select_92: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_392, 0, 2);  permute_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_393: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_91, [0, 1, 3, 2]);  select_91 = None
    expand_120: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_302, [8, 16, 576, 48]);  mul_302 = None
    clone_422: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_120, memory_format = torch.contiguous_format);  expand_120 = None
    view_604: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_422, [128, 576, 48]);  clone_422 = None
    expand_121: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_393, [8, 16, 48, 576]);  permute_393 = None
    clone_423: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
    view_605: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_423, [128, 48, 576]);  clone_423 = None
    bmm_60: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_604, view_605)
    view_606: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_60, [8, 16, 576, 576]);  bmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_394: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_606, [0, 2, 3, 1]);  view_606 = None
    permute_395: "f32[16, 16]" = torch.ops.aten.permute.default(primals_565, [1, 0]);  primals_565 = None
    clone_424: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_394, memory_format = torch.contiguous_format);  permute_394 = None
    view_607: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_424, [2654208, 16]);  clone_424 = None
    mm_60: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_607, permute_395)
    view_608: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_60, [8, 576, 576, 16]);  mm_60 = None
    add_273: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_608, primals_566);  view_608 = primals_566 = None
    permute_396: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_273, [0, 3, 1, 2]);  add_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_425: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_396, memory_format = torch.contiguous_format);  permute_396 = None
    amax_30: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_425, [-1], True)
    sub_91: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_425, amax_30);  clone_425 = amax_30 = None
    exp_30: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_91);  sub_91 = None
    sum_31: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_30, [-1], True)
    div_30: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_30, sum_31);  exp_30 = sum_31 = None
    alias_30: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_397: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_30, [0, 2, 3, 1]);  div_30 = None
    permute_398: "f32[16, 16]" = torch.ops.aten.permute.default(primals_567, [1, 0]);  primals_567 = None
    clone_426: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_397, memory_format = torch.contiguous_format);  permute_397 = None
    view_609: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_426, [2654208, 16]);  clone_426 = None
    mm_61: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_609, permute_398)
    view_610: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_61, [8, 576, 576, 16]);  mm_61 = None
    add_274: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_610, primals_568);  view_610 = primals_568 = None
    permute_399: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_274, [0, 3, 1, 2]);  add_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_427: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_399);  permute_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_122: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_427, [8, 16, 576, 576]);  clone_427 = None
    clone_428: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
    view_611: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_428, [128, 576, 576]);  clone_428 = None
    expand_123: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_92, [8, 16, 576, 48]);  select_92 = None
    clone_429: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
    view_612: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_429, [128, 576, 48]);  clone_429 = None
    bmm_61: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_611, view_612)
    view_613: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_61, [8, 16, 576, 48]);  bmm_61 = None
    permute_400: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_613, [0, 2, 1, 3]);  view_613 = None
    clone_430: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_400, memory_format = torch.contiguous_format);  permute_400 = None
    view_614: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_430, [8, 576, 768]);  clone_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_615: "f32[4608, 768]" = torch.ops.aten.view.default(view_614, [4608, 768]);  view_614 = None
    permute_401: "f32[768, 768]" = torch.ops.aten.permute.default(primals_569, [1, 0]);  primals_569 = None
    addmm_121: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_570, view_615, permute_401);  primals_570 = None
    view_616: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_121, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_431: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_616);  view_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_303: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_62, clone_431);  clone_431 = None
    add_275: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_270, mul_303);  add_270 = mul_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_432: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_275, memory_format = torch.contiguous_format)
    var_mean_61 = torch.ops.aten.var_mean.correction(clone_432, [2], correction = 0, keepdim = True)
    getitem_122: "f32[8, 576, 1]" = var_mean_61[0]
    getitem_123: "f32[8, 576, 1]" = var_mean_61[1];  var_mean_61 = None
    add_276: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-06);  getitem_122 = None
    rsqrt_61: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
    sub_92: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_432, getitem_123);  clone_432 = getitem_123 = None
    mul_304: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_61);  sub_92 = None
    mul_305: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_304, primals_571)
    add_277: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_305, primals_572);  mul_305 = primals_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_617: "f32[4608, 768]" = torch.ops.aten.view.default(add_277, [4608, 768]);  add_277 = None
    permute_402: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_573, [1, 0]);  primals_573 = None
    addmm_122: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_574, view_617, permute_402);  primals_574 = None
    view_618: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_122, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_306: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_618, 0.5)
    mul_307: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_618, 0.7071067811865476);  view_618 = None
    erf_30: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_307);  mul_307 = None
    add_278: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_308: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_306, add_278);  mul_306 = add_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_433: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_308);  mul_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_619: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_433, [4608, 3072]);  clone_433 = None
    permute_403: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_575, [1, 0]);  primals_575 = None
    addmm_123: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_576, view_619, permute_403);  primals_576 = None
    view_620: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_123, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_434: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_620);  view_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_309: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_63, clone_434);  clone_434 = None
    add_279: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_275, mul_309);  add_275 = mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_435: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_279, memory_format = torch.contiguous_format)
    var_mean_62 = torch.ops.aten.var_mean.correction(clone_435, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 576, 1]" = var_mean_62[0]
    getitem_125: "f32[8, 576, 1]" = var_mean_62[1];  var_mean_62 = None
    add_280: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-06);  getitem_124 = None
    rsqrt_62: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
    sub_93: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_435, getitem_125);  clone_435 = getitem_125 = None
    mul_310: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_62);  sub_93 = None
    mul_311: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_310, primals_577)
    add_281: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_311, primals_578);  mul_311 = primals_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_621: "f32[4608, 768]" = torch.ops.aten.view.default(add_281, [4608, 768]);  add_281 = None
    permute_404: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_579, [1, 0]);  primals_579 = None
    addmm_124: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_580, view_621, permute_404);  primals_580 = None
    view_622: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_124, [8, 576, 2304]);  addmm_124 = None
    view_623: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_622, [8, 576, 3, 16, 48]);  view_622 = None
    permute_405: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_623, [2, 0, 3, 1, 4]);  view_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_93: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_405, 0, 0)
    mul_312: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_93, 0.14433756729740643);  select_93 = None
    select_94: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_405, 0, 1)
    select_95: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_405, 0, 2);  permute_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_406: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_94, [0, 1, 3, 2]);  select_94 = None
    expand_124: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_312, [8, 16, 576, 48]);  mul_312 = None
    clone_436: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_124, memory_format = torch.contiguous_format);  expand_124 = None
    view_624: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_436, [128, 576, 48]);  clone_436 = None
    expand_125: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_406, [8, 16, 48, 576]);  permute_406 = None
    clone_437: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
    view_625: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_437, [128, 48, 576]);  clone_437 = None
    bmm_62: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_624, view_625)
    view_626: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_62, [8, 16, 576, 576]);  bmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_407: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_626, [0, 2, 3, 1]);  view_626 = None
    permute_408: "f32[16, 16]" = torch.ops.aten.permute.default(primals_581, [1, 0]);  primals_581 = None
    clone_438: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
    view_627: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_438, [2654208, 16]);  clone_438 = None
    mm_62: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_627, permute_408)
    view_628: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_62, [8, 576, 576, 16]);  mm_62 = None
    add_282: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_628, primals_582);  view_628 = primals_582 = None
    permute_409: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_282, [0, 3, 1, 2]);  add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_439: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_409, memory_format = torch.contiguous_format);  permute_409 = None
    amax_31: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_439, [-1], True)
    sub_94: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_439, amax_31);  clone_439 = amax_31 = None
    exp_31: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_94);  sub_94 = None
    sum_32: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_31, [-1], True)
    div_31: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_31, sum_32);  exp_31 = sum_32 = None
    alias_31: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_410: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_31, [0, 2, 3, 1]);  div_31 = None
    permute_411: "f32[16, 16]" = torch.ops.aten.permute.default(primals_583, [1, 0]);  primals_583 = None
    clone_440: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
    view_629: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_440, [2654208, 16]);  clone_440 = None
    mm_63: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_629, permute_411)
    view_630: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_63, [8, 576, 576, 16]);  mm_63 = None
    add_283: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_630, primals_584);  view_630 = primals_584 = None
    permute_412: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_283, [0, 3, 1, 2]);  add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_441: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_412);  permute_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_126: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_441, [8, 16, 576, 576]);  clone_441 = None
    clone_442: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_126, memory_format = torch.contiguous_format);  expand_126 = None
    view_631: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_442, [128, 576, 576]);  clone_442 = None
    expand_127: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_95, [8, 16, 576, 48]);  select_95 = None
    clone_443: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_127, memory_format = torch.contiguous_format);  expand_127 = None
    view_632: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_443, [128, 576, 48]);  clone_443 = None
    bmm_63: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_631, view_632)
    view_633: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_63, [8, 16, 576, 48]);  bmm_63 = None
    permute_413: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_633, [0, 2, 1, 3]);  view_633 = None
    clone_444: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
    view_634: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_444, [8, 576, 768]);  clone_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_635: "f32[4608, 768]" = torch.ops.aten.view.default(view_634, [4608, 768]);  view_634 = None
    permute_414: "f32[768, 768]" = torch.ops.aten.permute.default(primals_585, [1, 0]);  primals_585 = None
    addmm_125: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_586, view_635, permute_414);  primals_586 = None
    view_636: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_125, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_445: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_636);  view_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_313: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_64, clone_445);  clone_445 = None
    add_284: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_279, mul_313);  add_279 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_446: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_284, memory_format = torch.contiguous_format)
    var_mean_63 = torch.ops.aten.var_mean.correction(clone_446, [2], correction = 0, keepdim = True)
    getitem_126: "f32[8, 576, 1]" = var_mean_63[0]
    getitem_127: "f32[8, 576, 1]" = var_mean_63[1];  var_mean_63 = None
    add_285: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
    rsqrt_63: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
    sub_95: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_446, getitem_127);  clone_446 = getitem_127 = None
    mul_314: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_63);  sub_95 = None
    mul_315: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_314, primals_587)
    add_286: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_315, primals_588);  mul_315 = primals_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_637: "f32[4608, 768]" = torch.ops.aten.view.default(add_286, [4608, 768]);  add_286 = None
    permute_415: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_589, [1, 0]);  primals_589 = None
    addmm_126: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_590, view_637, permute_415);  primals_590 = None
    view_638: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_126, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_316: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_638, 0.5)
    mul_317: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_638, 0.7071067811865476);  view_638 = None
    erf_31: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_317);  mul_317 = None
    add_287: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_318: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_316, add_287);  mul_316 = add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_447: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_318);  mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_639: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_447, [4608, 3072]);  clone_447 = None
    permute_416: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_591, [1, 0]);  primals_591 = None
    addmm_127: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_592, view_639, permute_416);  primals_592 = None
    view_640: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_127, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_448: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_640);  view_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_319: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_65, clone_448);  clone_448 = None
    add_288: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_284, mul_319);  add_284 = mul_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_449: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_288, memory_format = torch.contiguous_format)
    var_mean_64 = torch.ops.aten.var_mean.correction(clone_449, [2], correction = 0, keepdim = True)
    getitem_128: "f32[8, 576, 1]" = var_mean_64[0]
    getitem_129: "f32[8, 576, 1]" = var_mean_64[1];  var_mean_64 = None
    add_289: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
    rsqrt_64: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_289);  add_289 = None
    sub_96: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_449, getitem_129);  clone_449 = getitem_129 = None
    mul_320: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_64);  sub_96 = None
    mul_321: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_320, primals_593)
    add_290: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_321, primals_594);  mul_321 = primals_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_641: "f32[4608, 768]" = torch.ops.aten.view.default(add_290, [4608, 768]);  add_290 = None
    permute_417: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_595, [1, 0]);  primals_595 = None
    addmm_128: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_596, view_641, permute_417);  primals_596 = None
    view_642: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_128, [8, 576, 2304]);  addmm_128 = None
    view_643: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_642, [8, 576, 3, 16, 48]);  view_642 = None
    permute_418: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_643, [2, 0, 3, 1, 4]);  view_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_96: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_418, 0, 0)
    mul_322: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_96, 0.14433756729740643);  select_96 = None
    select_97: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_418, 0, 1)
    select_98: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_418, 0, 2);  permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_419: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_97, [0, 1, 3, 2]);  select_97 = None
    expand_128: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_322, [8, 16, 576, 48]);  mul_322 = None
    clone_450: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
    view_644: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_450, [128, 576, 48]);  clone_450 = None
    expand_129: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_419, [8, 16, 48, 576]);  permute_419 = None
    clone_451: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
    view_645: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_451, [128, 48, 576]);  clone_451 = None
    bmm_64: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_644, view_645)
    view_646: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_64, [8, 16, 576, 576]);  bmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_420: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_646, [0, 2, 3, 1]);  view_646 = None
    permute_421: "f32[16, 16]" = torch.ops.aten.permute.default(primals_597, [1, 0]);  primals_597 = None
    clone_452: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_647: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_452, [2654208, 16]);  clone_452 = None
    mm_64: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_647, permute_421)
    view_648: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_64, [8, 576, 576, 16]);  mm_64 = None
    add_291: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_648, primals_598);  view_648 = primals_598 = None
    permute_422: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_291, [0, 3, 1, 2]);  add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_453: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_422, memory_format = torch.contiguous_format);  permute_422 = None
    amax_32: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_453, [-1], True)
    sub_97: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_453, amax_32);  clone_453 = amax_32 = None
    exp_32: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_97);  sub_97 = None
    sum_33: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_32, [-1], True)
    div_32: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_32, sum_33);  exp_32 = sum_33 = None
    alias_32: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_423: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_32, [0, 2, 3, 1]);  div_32 = None
    permute_424: "f32[16, 16]" = torch.ops.aten.permute.default(primals_599, [1, 0]);  primals_599 = None
    clone_454: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_423, memory_format = torch.contiguous_format);  permute_423 = None
    view_649: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_454, [2654208, 16]);  clone_454 = None
    mm_65: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_649, permute_424)
    view_650: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_65, [8, 576, 576, 16]);  mm_65 = None
    add_292: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_650, primals_600);  view_650 = primals_600 = None
    permute_425: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_292, [0, 3, 1, 2]);  add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_455: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_425);  permute_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_130: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_455, [8, 16, 576, 576]);  clone_455 = None
    clone_456: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
    view_651: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_456, [128, 576, 576]);  clone_456 = None
    expand_131: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_98, [8, 16, 576, 48]);  select_98 = None
    clone_457: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
    view_652: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_457, [128, 576, 48]);  clone_457 = None
    bmm_65: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_651, view_652)
    view_653: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_65, [8, 16, 576, 48]);  bmm_65 = None
    permute_426: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
    clone_458: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    view_654: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_458, [8, 576, 768]);  clone_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_655: "f32[4608, 768]" = torch.ops.aten.view.default(view_654, [4608, 768]);  view_654 = None
    permute_427: "f32[768, 768]" = torch.ops.aten.permute.default(primals_601, [1, 0]);  primals_601 = None
    addmm_129: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_602, view_655, permute_427);  primals_602 = None
    view_656: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_129, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_459: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_656);  view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_323: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_66, clone_459);  clone_459 = None
    add_293: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_288, mul_323);  add_288 = mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_460: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_293, memory_format = torch.contiguous_format)
    var_mean_65 = torch.ops.aten.var_mean.correction(clone_460, [2], correction = 0, keepdim = True)
    getitem_130: "f32[8, 576, 1]" = var_mean_65[0]
    getitem_131: "f32[8, 576, 1]" = var_mean_65[1];  var_mean_65 = None
    add_294: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-06);  getitem_130 = None
    rsqrt_65: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
    sub_98: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_460, getitem_131);  clone_460 = getitem_131 = None
    mul_324: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_65);  sub_98 = None
    mul_325: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_324, primals_603)
    add_295: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_325, primals_604);  mul_325 = primals_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_657: "f32[4608, 768]" = torch.ops.aten.view.default(add_295, [4608, 768]);  add_295 = None
    permute_428: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_605, [1, 0]);  primals_605 = None
    addmm_130: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_606, view_657, permute_428);  primals_606 = None
    view_658: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_130, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_326: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_658, 0.5)
    mul_327: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_658, 0.7071067811865476);  view_658 = None
    erf_32: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_327);  mul_327 = None
    add_296: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_328: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_326, add_296);  mul_326 = add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_461: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_328);  mul_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_659: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_461, [4608, 3072]);  clone_461 = None
    permute_429: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_607, [1, 0]);  primals_607 = None
    addmm_131: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_608, view_659, permute_429);  primals_608 = None
    view_660: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_131, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_462: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_660);  view_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_329: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_67, clone_462);  clone_462 = None
    add_297: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_293, mul_329);  add_293 = mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_463: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_297, memory_format = torch.contiguous_format)
    var_mean_66 = torch.ops.aten.var_mean.correction(clone_463, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 576, 1]" = var_mean_66[0]
    getitem_133: "f32[8, 576, 1]" = var_mean_66[1];  var_mean_66 = None
    add_298: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_66: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_298);  add_298 = None
    sub_99: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_463, getitem_133);  clone_463 = getitem_133 = None
    mul_330: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_66);  sub_99 = None
    mul_331: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_330, primals_609)
    add_299: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_331, primals_610);  mul_331 = primals_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_661: "f32[4608, 768]" = torch.ops.aten.view.default(add_299, [4608, 768]);  add_299 = None
    permute_430: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_611, [1, 0]);  primals_611 = None
    addmm_132: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_612, view_661, permute_430);  primals_612 = None
    view_662: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_132, [8, 576, 2304]);  addmm_132 = None
    view_663: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_662, [8, 576, 3, 16, 48]);  view_662 = None
    permute_431: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_663, [2, 0, 3, 1, 4]);  view_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_99: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_431, 0, 0)
    mul_332: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_99, 0.14433756729740643);  select_99 = None
    select_100: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_431, 0, 1)
    select_101: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_431, 0, 2);  permute_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_432: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_100, [0, 1, 3, 2]);  select_100 = None
    expand_132: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_332, [8, 16, 576, 48]);  mul_332 = None
    clone_464: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_132, memory_format = torch.contiguous_format);  expand_132 = None
    view_664: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_464, [128, 576, 48]);  clone_464 = None
    expand_133: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_432, [8, 16, 48, 576]);  permute_432 = None
    clone_465: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_133, memory_format = torch.contiguous_format);  expand_133 = None
    view_665: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_465, [128, 48, 576]);  clone_465 = None
    bmm_66: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_664, view_665)
    view_666: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_66, [8, 16, 576, 576]);  bmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_433: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_666, [0, 2, 3, 1]);  view_666 = None
    permute_434: "f32[16, 16]" = torch.ops.aten.permute.default(primals_613, [1, 0]);  primals_613 = None
    clone_466: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
    view_667: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_466, [2654208, 16]);  clone_466 = None
    mm_66: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_667, permute_434)
    view_668: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_66, [8, 576, 576, 16]);  mm_66 = None
    add_300: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_668, primals_614);  view_668 = primals_614 = None
    permute_435: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_300, [0, 3, 1, 2]);  add_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_467: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_435, memory_format = torch.contiguous_format);  permute_435 = None
    amax_33: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_467, [-1], True)
    sub_100: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_467, amax_33);  clone_467 = amax_33 = None
    exp_33: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_100);  sub_100 = None
    sum_34: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_33, [-1], True)
    div_33: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_33, sum_34);  exp_33 = sum_34 = None
    alias_33: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_436: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_33, [0, 2, 3, 1]);  div_33 = None
    permute_437: "f32[16, 16]" = torch.ops.aten.permute.default(primals_615, [1, 0]);  primals_615 = None
    clone_468: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_436, memory_format = torch.contiguous_format);  permute_436 = None
    view_669: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_468, [2654208, 16]);  clone_468 = None
    mm_67: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_669, permute_437)
    view_670: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_67, [8, 576, 576, 16]);  mm_67 = None
    add_301: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_670, primals_616);  view_670 = primals_616 = None
    permute_438: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_301, [0, 3, 1, 2]);  add_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_469: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_438);  permute_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_134: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_469, [8, 16, 576, 576]);  clone_469 = None
    clone_470: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_134, memory_format = torch.contiguous_format);  expand_134 = None
    view_671: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_470, [128, 576, 576]);  clone_470 = None
    expand_135: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_101, [8, 16, 576, 48]);  select_101 = None
    clone_471: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
    view_672: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_471, [128, 576, 48]);  clone_471 = None
    bmm_67: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_671, view_672)
    view_673: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_67, [8, 16, 576, 48]);  bmm_67 = None
    permute_439: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None
    clone_472: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_439, memory_format = torch.contiguous_format);  permute_439 = None
    view_674: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_472, [8, 576, 768]);  clone_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_675: "f32[4608, 768]" = torch.ops.aten.view.default(view_674, [4608, 768]);  view_674 = None
    permute_440: "f32[768, 768]" = torch.ops.aten.permute.default(primals_617, [1, 0]);  primals_617 = None
    addmm_133: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_618, view_675, permute_440);  primals_618 = None
    view_676: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_133, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_473: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_676);  view_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_333: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_68, clone_473);  clone_473 = None
    add_302: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_297, mul_333);  add_297 = mul_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_474: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_302, memory_format = torch.contiguous_format)
    var_mean_67 = torch.ops.aten.var_mean.correction(clone_474, [2], correction = 0, keepdim = True)
    getitem_134: "f32[8, 576, 1]" = var_mean_67[0]
    getitem_135: "f32[8, 576, 1]" = var_mean_67[1];  var_mean_67 = None
    add_303: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-06);  getitem_134 = None
    rsqrt_67: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
    sub_101: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_474, getitem_135);  clone_474 = getitem_135 = None
    mul_334: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_67);  sub_101 = None
    mul_335: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_334, primals_619)
    add_304: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_335, primals_620);  mul_335 = primals_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_677: "f32[4608, 768]" = torch.ops.aten.view.default(add_304, [4608, 768]);  add_304 = None
    permute_441: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_621, [1, 0]);  primals_621 = None
    addmm_134: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_622, view_677, permute_441);  primals_622 = None
    view_678: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_134, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_336: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_678, 0.5)
    mul_337: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_678, 0.7071067811865476);  view_678 = None
    erf_33: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_337);  mul_337 = None
    add_305: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_338: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_336, add_305);  mul_336 = add_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_475: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_338);  mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_679: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_475, [4608, 3072]);  clone_475 = None
    permute_442: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_623, [1, 0]);  primals_623 = None
    addmm_135: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_624, view_679, permute_442);  primals_624 = None
    view_680: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_135, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_476: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_680);  view_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_339: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_69, clone_476);  clone_476 = None
    add_306: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_302, mul_339);  add_302 = mul_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_477: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_306, memory_format = torch.contiguous_format)
    var_mean_68 = torch.ops.aten.var_mean.correction(clone_477, [2], correction = 0, keepdim = True)
    getitem_136: "f32[8, 576, 1]" = var_mean_68[0]
    getitem_137: "f32[8, 576, 1]" = var_mean_68[1];  var_mean_68 = None
    add_307: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-06);  getitem_136 = None
    rsqrt_68: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_307);  add_307 = None
    sub_102: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_477, getitem_137);  clone_477 = getitem_137 = None
    mul_340: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_68);  sub_102 = None
    mul_341: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_340, primals_625)
    add_308: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_341, primals_626);  mul_341 = primals_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_681: "f32[4608, 768]" = torch.ops.aten.view.default(add_308, [4608, 768]);  add_308 = None
    permute_443: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_627, [1, 0]);  primals_627 = None
    addmm_136: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_628, view_681, permute_443);  primals_628 = None
    view_682: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_136, [8, 576, 2304]);  addmm_136 = None
    view_683: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_682, [8, 576, 3, 16, 48]);  view_682 = None
    permute_444: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_683, [2, 0, 3, 1, 4]);  view_683 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_102: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_444, 0, 0)
    mul_342: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_102, 0.14433756729740643);  select_102 = None
    select_103: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_444, 0, 1)
    select_104: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_444, 0, 2);  permute_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_445: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_103, [0, 1, 3, 2]);  select_103 = None
    expand_136: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_342, [8, 16, 576, 48]);  mul_342 = None
    clone_478: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_136, memory_format = torch.contiguous_format);  expand_136 = None
    view_684: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_478, [128, 576, 48]);  clone_478 = None
    expand_137: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_445, [8, 16, 48, 576]);  permute_445 = None
    clone_479: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
    view_685: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_479, [128, 48, 576]);  clone_479 = None
    bmm_68: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_684, view_685)
    view_686: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_68, [8, 16, 576, 576]);  bmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_446: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_686, [0, 2, 3, 1]);  view_686 = None
    permute_447: "f32[16, 16]" = torch.ops.aten.permute.default(primals_629, [1, 0]);  primals_629 = None
    clone_480: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_446, memory_format = torch.contiguous_format);  permute_446 = None
    view_687: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_480, [2654208, 16]);  clone_480 = None
    mm_68: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_687, permute_447)
    view_688: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_68, [8, 576, 576, 16]);  mm_68 = None
    add_309: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_688, primals_630);  view_688 = primals_630 = None
    permute_448: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_309, [0, 3, 1, 2]);  add_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_481: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_448, memory_format = torch.contiguous_format);  permute_448 = None
    amax_34: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_481, [-1], True)
    sub_103: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_481, amax_34);  clone_481 = amax_34 = None
    exp_34: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_103);  sub_103 = None
    sum_35: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_34, [-1], True)
    div_34: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_34, sum_35);  exp_34 = sum_35 = None
    alias_34: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_449: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_34, [0, 2, 3, 1]);  div_34 = None
    permute_450: "f32[16, 16]" = torch.ops.aten.permute.default(primals_631, [1, 0]);  primals_631 = None
    clone_482: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_449, memory_format = torch.contiguous_format);  permute_449 = None
    view_689: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_482, [2654208, 16]);  clone_482 = None
    mm_69: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_689, permute_450)
    view_690: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_69, [8, 576, 576, 16]);  mm_69 = None
    add_310: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_690, primals_632);  view_690 = primals_632 = None
    permute_451: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_310, [0, 3, 1, 2]);  add_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_483: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_451);  permute_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_138: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_483, [8, 16, 576, 576]);  clone_483 = None
    clone_484: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_138, memory_format = torch.contiguous_format);  expand_138 = None
    view_691: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_484, [128, 576, 576]);  clone_484 = None
    expand_139: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_104, [8, 16, 576, 48]);  select_104 = None
    clone_485: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_139, memory_format = torch.contiguous_format);  expand_139 = None
    view_692: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_485, [128, 576, 48]);  clone_485 = None
    bmm_69: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_691, view_692)
    view_693: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_69, [8, 16, 576, 48]);  bmm_69 = None
    permute_452: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_693, [0, 2, 1, 3]);  view_693 = None
    clone_486: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
    view_694: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_486, [8, 576, 768]);  clone_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_695: "f32[4608, 768]" = torch.ops.aten.view.default(view_694, [4608, 768]);  view_694 = None
    permute_453: "f32[768, 768]" = torch.ops.aten.permute.default(primals_633, [1, 0]);  primals_633 = None
    addmm_137: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_634, view_695, permute_453);  primals_634 = None
    view_696: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_137, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_487: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_696);  view_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_343: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_70, clone_487);  clone_487 = None
    add_311: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_306, mul_343);  add_306 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_488: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_311, memory_format = torch.contiguous_format)
    var_mean_69 = torch.ops.aten.var_mean.correction(clone_488, [2], correction = 0, keepdim = True)
    getitem_138: "f32[8, 576, 1]" = var_mean_69[0]
    getitem_139: "f32[8, 576, 1]" = var_mean_69[1];  var_mean_69 = None
    add_312: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-06);  getitem_138 = None
    rsqrt_69: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_312);  add_312 = None
    sub_104: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_488, getitem_139);  clone_488 = getitem_139 = None
    mul_344: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_69);  sub_104 = None
    mul_345: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_344, primals_635)
    add_313: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_345, primals_636);  mul_345 = primals_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_697: "f32[4608, 768]" = torch.ops.aten.view.default(add_313, [4608, 768]);  add_313 = None
    permute_454: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_637, [1, 0]);  primals_637 = None
    addmm_138: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_638, view_697, permute_454);  primals_638 = None
    view_698: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_138, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_346: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_698, 0.5)
    mul_347: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_698, 0.7071067811865476);  view_698 = None
    erf_34: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_347);  mul_347 = None
    add_314: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_348: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_346, add_314);  mul_346 = add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_489: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_348);  mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_699: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_489, [4608, 3072]);  clone_489 = None
    permute_455: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_639, [1, 0]);  primals_639 = None
    addmm_139: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_640, view_699, permute_455);  primals_640 = None
    view_700: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_139, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_490: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_700);  view_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_349: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_71, clone_490);  clone_490 = None
    add_315: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_311, mul_349);  add_311 = mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    clone_491: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_315, memory_format = torch.contiguous_format)
    var_mean_70 = torch.ops.aten.var_mean.correction(clone_491, [2], correction = 0, keepdim = True)
    getitem_140: "f32[8, 576, 1]" = var_mean_70[0]
    getitem_141: "f32[8, 576, 1]" = var_mean_70[1];  var_mean_70 = None
    add_316: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-06);  getitem_140 = None
    rsqrt_70: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_316);  add_316 = None
    sub_105: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_491, getitem_141);  clone_491 = getitem_141 = None
    mul_350: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_70);  sub_105 = None
    mul_351: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_350, primals_641)
    add_317: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_351, primals_642);  mul_351 = primals_642 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    view_701: "f32[4608, 768]" = torch.ops.aten.view.default(add_317, [4608, 768]);  add_317 = None
    permute_456: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_643, [1, 0]);  primals_643 = None
    addmm_140: "f32[4608, 2304]" = torch.ops.aten.addmm.default(primals_644, view_701, permute_456);  primals_644 = None
    view_702: "f32[8, 576, 2304]" = torch.ops.aten.view.default(addmm_140, [8, 576, 2304]);  addmm_140 = None
    view_703: "f32[8, 576, 3, 16, 48]" = torch.ops.aten.view.default(view_702, [8, 576, 3, 16, 48]);  view_702 = None
    permute_457: "f32[3, 8, 16, 576, 48]" = torch.ops.aten.permute.default(view_703, [2, 0, 3, 1, 4]);  view_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    select_105: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_457, 0, 0)
    mul_352: "f32[8, 16, 576, 48]" = torch.ops.aten.mul.Tensor(select_105, 0.14433756729740643);  select_105 = None
    select_106: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_457, 0, 1)
    select_107: "f32[8, 16, 576, 48]" = torch.ops.aten.select.int(permute_457, 0, 2);  permute_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_458: "f32[8, 16, 48, 576]" = torch.ops.aten.permute.default(select_106, [0, 1, 3, 2]);  select_106 = None
    expand_140: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(mul_352, [8, 16, 576, 48]);  mul_352 = None
    clone_492: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
    view_704: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_492, [128, 576, 48]);  clone_492 = None
    expand_141: "f32[8, 16, 48, 576]" = torch.ops.aten.expand.default(permute_458, [8, 16, 48, 576]);  permute_458 = None
    clone_493: "f32[8, 16, 48, 576]" = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
    view_705: "f32[128, 48, 576]" = torch.ops.aten.view.default(clone_493, [128, 48, 576]);  clone_493 = None
    bmm_70: "f32[128, 576, 576]" = torch.ops.aten.bmm.default(view_704, view_705)
    view_706: "f32[8, 16, 576, 576]" = torch.ops.aten.view.default(bmm_70, [8, 16, 576, 576]);  bmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_459: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(view_706, [0, 2, 3, 1]);  view_706 = None
    permute_460: "f32[16, 16]" = torch.ops.aten.permute.default(primals_645, [1, 0]);  primals_645 = None
    clone_494: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    view_707: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_494, [2654208, 16]);  clone_494 = None
    mm_70: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_707, permute_460)
    view_708: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_70, [8, 576, 576, 16]);  mm_70 = None
    add_318: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_708, primals_646);  view_708 = primals_646 = None
    permute_461: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_318, [0, 3, 1, 2]);  add_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    clone_495: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_461, memory_format = torch.contiguous_format);  permute_461 = None
    amax_35: "f32[8, 16, 576, 1]" = torch.ops.aten.amax.default(clone_495, [-1], True)
    sub_106: "f32[8, 16, 576, 576]" = torch.ops.aten.sub.Tensor(clone_495, amax_35);  clone_495 = amax_35 = None
    exp_35: "f32[8, 16, 576, 576]" = torch.ops.aten.exp.default(sub_106);  sub_106 = None
    sum_36: "f32[8, 16, 576, 1]" = torch.ops.aten.sum.dim_IntList(exp_35, [-1], True)
    div_35: "f32[8, 16, 576, 576]" = torch.ops.aten.div.Tensor(exp_35, sum_36);  exp_35 = sum_36 = None
    alias_35: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(div_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_462: "f32[8, 576, 576, 16]" = torch.ops.aten.permute.default(div_35, [0, 2, 3, 1]);  div_35 = None
    permute_463: "f32[16, 16]" = torch.ops.aten.permute.default(primals_647, [1, 0]);  primals_647 = None
    clone_496: "f32[8, 576, 576, 16]" = torch.ops.aten.clone.default(permute_462, memory_format = torch.contiguous_format);  permute_462 = None
    view_709: "f32[2654208, 16]" = torch.ops.aten.view.default(clone_496, [2654208, 16]);  clone_496 = None
    mm_71: "f32[2654208, 16]" = torch.ops.aten.mm.default(view_709, permute_463)
    view_710: "f32[8, 576, 576, 16]" = torch.ops.aten.view.default(mm_71, [8, 576, 576, 16]);  mm_71 = None
    add_319: "f32[8, 576, 576, 16]" = torch.ops.aten.add.Tensor(view_710, primals_648);  view_710 = primals_648 = None
    permute_464: "f32[8, 16, 576, 576]" = torch.ops.aten.permute.default(add_319, [0, 3, 1, 2]);  add_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    clone_497: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(permute_464);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    expand_142: "f32[8, 16, 576, 576]" = torch.ops.aten.expand.default(clone_497, [8, 16, 576, 576]);  clone_497 = None
    clone_498: "f32[8, 16, 576, 576]" = torch.ops.aten.clone.default(expand_142, memory_format = torch.contiguous_format);  expand_142 = None
    view_711: "f32[128, 576, 576]" = torch.ops.aten.view.default(clone_498, [128, 576, 576]);  clone_498 = None
    expand_143: "f32[8, 16, 576, 48]" = torch.ops.aten.expand.default(select_107, [8, 16, 576, 48]);  select_107 = None
    clone_499: "f32[8, 16, 576, 48]" = torch.ops.aten.clone.default(expand_143, memory_format = torch.contiguous_format);  expand_143 = None
    view_712: "f32[128, 576, 48]" = torch.ops.aten.view.default(clone_499, [128, 576, 48]);  clone_499 = None
    bmm_71: "f32[128, 576, 48]" = torch.ops.aten.bmm.default(view_711, view_712)
    view_713: "f32[8, 16, 576, 48]" = torch.ops.aten.view.default(bmm_71, [8, 16, 576, 48]);  bmm_71 = None
    permute_465: "f32[8, 576, 16, 48]" = torch.ops.aten.permute.default(view_713, [0, 2, 1, 3]);  view_713 = None
    clone_500: "f32[8, 576, 16, 48]" = torch.ops.aten.clone.default(permute_465, memory_format = torch.contiguous_format);  permute_465 = None
    view_714: "f32[8, 576, 768]" = torch.ops.aten.view.default(clone_500, [8, 576, 768]);  clone_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    view_715: "f32[4608, 768]" = torch.ops.aten.view.default(view_714, [4608, 768]);  view_714 = None
    permute_466: "f32[768, 768]" = torch.ops.aten.permute.default(primals_649, [1, 0]);  primals_649 = None
    addmm_141: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_650, view_715, permute_466);  primals_650 = None
    view_716: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_141, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    clone_501: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_716);  view_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_353: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_72, clone_501);  clone_501 = None
    add_320: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_315, mul_353);  add_315 = mul_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    clone_502: "f32[8, 576, 768]" = torch.ops.aten.clone.default(add_320, memory_format = torch.contiguous_format)
    var_mean_71 = torch.ops.aten.var_mean.correction(clone_502, [2], correction = 0, keepdim = True)
    getitem_142: "f32[8, 576, 1]" = var_mean_71[0]
    getitem_143: "f32[8, 576, 1]" = var_mean_71[1];  var_mean_71 = None
    add_321: "f32[8, 576, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
    rsqrt_71: "f32[8, 576, 1]" = torch.ops.aten.rsqrt.default(add_321);  add_321 = None
    sub_107: "f32[8, 576, 768]" = torch.ops.aten.sub.Tensor(clone_502, getitem_143);  clone_502 = getitem_143 = None
    mul_354: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_71);  sub_107 = None
    mul_355: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(mul_354, primals_651)
    add_322: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(mul_355, primals_652);  mul_355 = primals_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_717: "f32[4608, 768]" = torch.ops.aten.view.default(add_322, [4608, 768]);  add_322 = None
    permute_467: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_653, [1, 0]);  primals_653 = None
    addmm_142: "f32[4608, 3072]" = torch.ops.aten.addmm.default(primals_654, view_717, permute_467);  primals_654 = None
    view_718: "f32[8, 576, 3072]" = torch.ops.aten.view.default(addmm_142, [8, 576, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_356: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_718, 0.5)
    mul_357: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(view_718, 0.7071067811865476);  view_718 = None
    erf_35: "f32[8, 576, 3072]" = torch.ops.aten.erf.default(mul_357);  mul_357 = None
    add_323: "f32[8, 576, 3072]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_358: "f32[8, 576, 3072]" = torch.ops.aten.mul.Tensor(mul_356, add_323);  mul_356 = add_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_503: "f32[8, 576, 3072]" = torch.ops.aten.clone.default(mul_358);  mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_719: "f32[4608, 3072]" = torch.ops.aten.view.default(clone_503, [4608, 3072]);  clone_503 = None
    permute_468: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_655, [1, 0]);  primals_655 = None
    addmm_143: "f32[4608, 768]" = torch.ops.aten.addmm.default(primals_656, view_719, permute_468);  primals_656 = None
    view_720: "f32[8, 576, 768]" = torch.ops.aten.view.default(addmm_143, [8, 576, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_504: "f32[8, 576, 768]" = torch.ops.aten.clone.default(view_720);  view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_359: "f32[8, 576, 768]" = torch.ops.aten.mul.Tensor(primals_73, clone_504);  clone_504 = None
    add_324: "f32[8, 576, 768]" = torch.ops.aten.add.Tensor(add_320, mul_359);  add_320 = mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:347, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    expand_144: "f32[8, 1, 768]" = torch.ops.aten.expand.default(primals_74, [8, -1, -1]);  primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:109, code: u = torch.cat((x_cls, x), dim=1)
    cat: "f32[8, 577, 768]" = torch.ops.aten.cat.default([expand_144, add_324], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    var_mean_72 = torch.ops.aten.var_mean.correction(cat, [2], correction = 0, keepdim = True)
    getitem_144: "f32[8, 577, 1]" = var_mean_72[0]
    getitem_145: "f32[8, 577, 1]" = var_mean_72[1];  var_mean_72 = None
    add_325: "f32[8, 577, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_72: "f32[8, 577, 1]" = torch.ops.aten.rsqrt.default(add_325);  add_325 = None
    sub_108: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_145)
    mul_360: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_72);  sub_108 = None
    mul_361: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_360, primals_657);  mul_360 = None
    add_326: "f32[8, 577, 768]" = torch.ops.aten.add.Tensor(mul_361, primals_658);  mul_361 = primals_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_1: "f32[8, 577, 768]" = torch.ops.aten.slice.Tensor(add_326, 0, 0, 9223372036854775807)
    select_108: "f32[8, 768]" = torch.ops.aten.select.int(slice_1, 1, 0);  slice_1 = None
    permute_469: "f32[768, 768]" = torch.ops.aten.permute.default(primals_659, [1, 0]);  primals_659 = None
    addmm_144: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_660, select_108, permute_469);  primals_660 = None
    unsqueeze: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_144, 1);  addmm_144 = None
    view_721: "f32[8, 1, 16, 48]" = torch.ops.aten.view.default(unsqueeze, [8, 1, 16, 48]);  unsqueeze = None
    permute_470: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_721, [0, 2, 1, 3]);  view_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_722: "f32[4616, 768]" = torch.ops.aten.view.default(add_326, [4616, 768]);  add_326 = None
    permute_471: "f32[768, 768]" = torch.ops.aten.permute.default(primals_661, [1, 0]);  primals_661 = None
    addmm_145: "f32[4616, 768]" = torch.ops.aten.addmm.default(primals_662, view_722, permute_471);  primals_662 = None
    view_723: "f32[8, 577, 768]" = torch.ops.aten.view.default(addmm_145, [8, 577, 768]);  addmm_145 = None
    view_724: "f32[8, 577, 16, 48]" = torch.ops.aten.view.default(view_723, [8, 577, 16, 48]);  view_723 = None
    permute_472: "f32[8, 16, 577, 48]" = torch.ops.aten.permute.default(view_724, [0, 2, 1, 3]);  view_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_473: "f32[768, 768]" = torch.ops.aten.permute.default(primals_663, [1, 0]);  primals_663 = None
    addmm_146: "f32[4616, 768]" = torch.ops.aten.addmm.default(primals_664, view_722, permute_473);  primals_664 = None
    view_726: "f32[8, 577, 768]" = torch.ops.aten.view.default(addmm_146, [8, 577, 768]);  addmm_146 = None
    view_727: "f32[8, 577, 16, 48]" = torch.ops.aten.view.default(view_726, [8, 577, 16, 48]);  view_726 = None
    permute_474: "f32[8, 16, 577, 48]" = torch.ops.aten.permute.default(view_727, [0, 2, 1, 3]);  view_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_470, permute_472, permute_474)
    getitem_146: "f32[8, 16, 1, 48]" = _scaled_dot_product_flash_attention[0]
    getitem_147: "f32[8, 16, 1]" = _scaled_dot_product_flash_attention[1]
    getitem_148: "i32[]" = _scaled_dot_product_flash_attention[2]
    getitem_149: "i32[]" = _scaled_dot_product_flash_attention[3]
    getitem_152: "i64[]" = _scaled_dot_product_flash_attention[6]
    getitem_153: "i64[]" = _scaled_dot_product_flash_attention[7];  _scaled_dot_product_flash_attention = None
    alias_36: "f32[8, 16, 1, 48]" = torch.ops.aten.alias.default(getitem_146)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    permute_475: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None
    view_728: "f32[8, 1, 768]" = torch.ops.aten.view.default(permute_475, [8, 1, 768]);  permute_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_729: "f32[8, 768]" = torch.ops.aten.view.default(view_728, [8, 768]);  view_728 = None
    permute_476: "f32[768, 768]" = torch.ops.aten.permute.default(primals_665, [1, 0]);  primals_665 = None
    addmm_147: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_666, view_729, permute_476);  primals_666 = None
    view_730: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_147, [8, 1, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:64, code: x_cls = self.proj_drop(x_cls)
    clone_505: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_730);  view_730 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    mul_362: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(primals_75, clone_505);  clone_505 = None
    add_327: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(expand_144, mul_362);  expand_144 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    var_mean_73 = torch.ops.aten.var_mean.correction(add_327, [2], correction = 0, keepdim = True)
    getitem_155: "f32[8, 1, 1]" = var_mean_73[0]
    getitem_156: "f32[8, 1, 1]" = var_mean_73[1];  var_mean_73 = None
    add_328: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_155, 1e-06);  getitem_155 = None
    rsqrt_73: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_328);  add_328 = None
    sub_109: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(add_327, getitem_156);  getitem_156 = None
    mul_363: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_73);  sub_109 = None
    mul_364: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(mul_363, primals_667)
    add_329: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(mul_364, primals_668);  mul_364 = primals_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_731: "f32[8, 768]" = torch.ops.aten.view.default(add_329, [8, 768]);  add_329 = None
    permute_477: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_669, [1, 0]);  primals_669 = None
    addmm_148: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_670, view_731, permute_477);  primals_670 = None
    view_732: "f32[8, 1, 3072]" = torch.ops.aten.view.default(addmm_148, [8, 1, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_365: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_732, 0.5)
    mul_366: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_732, 0.7071067811865476);  view_732 = None
    erf_36: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_366);  mul_366 = None
    add_330: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_367: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_365, add_330);  mul_365 = add_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_506: "f32[8, 1, 3072]" = torch.ops.aten.clone.default(mul_367);  mul_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_733: "f32[8, 3072]" = torch.ops.aten.view.default(clone_506, [8, 3072]);  clone_506 = None
    permute_478: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_671, [1, 0]);  primals_671 = None
    addmm_149: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_672, view_733, permute_478);  primals_672 = None
    view_734: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_149, [8, 1, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_507: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_734);  view_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    mul_368: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(primals_76, clone_507);  clone_507 = None
    add_331: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(add_327, mul_368);  add_327 = mul_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:109, code: u = torch.cat((x_cls, x), dim=1)
    cat_1: "f32[8, 577, 768]" = torch.ops.aten.cat.default([add_331, add_324], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    var_mean_74 = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
    getitem_157: "f32[8, 577, 1]" = var_mean_74[0]
    getitem_158: "f32[8, 577, 1]" = var_mean_74[1];  var_mean_74 = None
    add_332: "f32[8, 577, 1]" = torch.ops.aten.add.Tensor(getitem_157, 1e-06);  getitem_157 = None
    rsqrt_74: "f32[8, 577, 1]" = torch.ops.aten.rsqrt.default(add_332);  add_332 = None
    sub_110: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(cat_1, getitem_158)
    mul_369: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_74);  sub_110 = None
    mul_370: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_369, primals_673);  mul_369 = None
    add_333: "f32[8, 577, 768]" = torch.ops.aten.add.Tensor(mul_370, primals_674);  mul_370 = primals_674 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_2: "f32[8, 577, 768]" = torch.ops.aten.slice.Tensor(add_333, 0, 0, 9223372036854775807)
    select_109: "f32[8, 768]" = torch.ops.aten.select.int(slice_2, 1, 0);  slice_2 = None
    permute_479: "f32[768, 768]" = torch.ops.aten.permute.default(primals_675, [1, 0]);  primals_675 = None
    addmm_150: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_676, select_109, permute_479);  primals_676 = None
    unsqueeze_1: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_150, 1);  addmm_150 = None
    view_735: "f32[8, 1, 16, 48]" = torch.ops.aten.view.default(unsqueeze_1, [8, 1, 16, 48]);  unsqueeze_1 = None
    permute_480: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_735, [0, 2, 1, 3]);  view_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_736: "f32[4616, 768]" = torch.ops.aten.view.default(add_333, [4616, 768]);  add_333 = None
    permute_481: "f32[768, 768]" = torch.ops.aten.permute.default(primals_677, [1, 0]);  primals_677 = None
    addmm_151: "f32[4616, 768]" = torch.ops.aten.addmm.default(primals_678, view_736, permute_481);  primals_678 = None
    view_737: "f32[8, 577, 768]" = torch.ops.aten.view.default(addmm_151, [8, 577, 768]);  addmm_151 = None
    view_738: "f32[8, 577, 16, 48]" = torch.ops.aten.view.default(view_737, [8, 577, 16, 48]);  view_737 = None
    permute_482: "f32[8, 16, 577, 48]" = torch.ops.aten.permute.default(view_738, [0, 2, 1, 3]);  view_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_483: "f32[768, 768]" = torch.ops.aten.permute.default(primals_679, [1, 0]);  primals_679 = None
    addmm_152: "f32[4616, 768]" = torch.ops.aten.addmm.default(primals_680, view_736, permute_483);  primals_680 = None
    view_740: "f32[8, 577, 768]" = torch.ops.aten.view.default(addmm_152, [8, 577, 768]);  addmm_152 = None
    view_741: "f32[8, 577, 16, 48]" = torch.ops.aten.view.default(view_740, [8, 577, 16, 48]);  view_740 = None
    permute_484: "f32[8, 16, 577, 48]" = torch.ops.aten.permute.default(view_741, [0, 2, 1, 3]);  view_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_480, permute_482, permute_484)
    getitem_159: "f32[8, 16, 1, 48]" = _scaled_dot_product_flash_attention_1[0]
    getitem_160: "f32[8, 16, 1]" = _scaled_dot_product_flash_attention_1[1]
    getitem_161: "i32[]" = _scaled_dot_product_flash_attention_1[2]
    getitem_162: "i32[]" = _scaled_dot_product_flash_attention_1[3]
    getitem_165: "i64[]" = _scaled_dot_product_flash_attention_1[6]
    getitem_166: "i64[]" = _scaled_dot_product_flash_attention_1[7];  _scaled_dot_product_flash_attention_1 = None
    alias_37: "f32[8, 16, 1, 48]" = torch.ops.aten.alias.default(getitem_159)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    permute_485: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3]);  getitem_159 = None
    view_742: "f32[8, 1, 768]" = torch.ops.aten.view.default(permute_485, [8, 1, 768]);  permute_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_743: "f32[8, 768]" = torch.ops.aten.view.default(view_742, [8, 768]);  view_742 = None
    permute_486: "f32[768, 768]" = torch.ops.aten.permute.default(primals_681, [1, 0]);  primals_681 = None
    addmm_153: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_682, view_743, permute_486);  primals_682 = None
    view_744: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_153, [8, 1, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:64, code: x_cls = self.proj_drop(x_cls)
    clone_508: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_744);  view_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    mul_371: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(primals_77, clone_508);  clone_508 = None
    add_334: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(add_331, mul_371);  add_331 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    var_mean_75 = torch.ops.aten.var_mean.correction(add_334, [2], correction = 0, keepdim = True)
    getitem_168: "f32[8, 1, 1]" = var_mean_75[0]
    getitem_169: "f32[8, 1, 1]" = var_mean_75[1];  var_mean_75 = None
    add_335: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
    rsqrt_75: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_335);  add_335 = None
    sub_111: "f32[8, 1, 768]" = torch.ops.aten.sub.Tensor(add_334, getitem_169);  getitem_169 = None
    mul_372: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_75);  sub_111 = None
    mul_373: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(mul_372, primals_683)
    add_336: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(mul_373, primals_684);  mul_373 = primals_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_745: "f32[8, 768]" = torch.ops.aten.view.default(add_336, [8, 768]);  add_336 = None
    permute_487: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_685, [1, 0]);  primals_685 = None
    addmm_154: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_686, view_745, permute_487);  primals_686 = None
    view_746: "f32[8, 1, 3072]" = torch.ops.aten.view.default(addmm_154, [8, 1, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_374: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_746, 0.5)
    mul_375: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(view_746, 0.7071067811865476);  view_746 = None
    erf_37: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_375);  mul_375 = None
    add_337: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_376: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_374, add_337);  mul_374 = add_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_509: "f32[8, 1, 3072]" = torch.ops.aten.clone.default(mul_376);  mul_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_747: "f32[8, 3072]" = torch.ops.aten.view.default(clone_509, [8, 3072]);  clone_509 = None
    permute_488: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_687, [1, 0]);  primals_687 = None
    addmm_155: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_688, view_747, permute_488);  primals_688 = None
    view_748: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_155, [8, 1, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_510: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_748);  view_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    mul_377: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(primals_78, clone_510);  clone_510 = None
    add_338: "f32[8, 1, 768]" = torch.ops.aten.add.Tensor(add_334, mul_377);  add_334 = mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:350, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_2: "f32[8, 577, 768]" = torch.ops.aten.cat.default([add_338, add_324], 1);  add_338 = add_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:351, code: x = self.norm(x)
    var_mean_76 = torch.ops.aten.var_mean.correction(cat_2, [2], correction = 0, keepdim = True)
    getitem_170: "f32[8, 577, 1]" = var_mean_76[0]
    getitem_171: "f32[8, 577, 1]" = var_mean_76[1];  var_mean_76 = None
    add_339: "f32[8, 577, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-06);  getitem_170 = None
    rsqrt_76: "f32[8, 577, 1]" = torch.ops.aten.rsqrt.default(add_339);  add_339 = None
    sub_112: "f32[8, 577, 768]" = torch.ops.aten.sub.Tensor(cat_2, getitem_171)
    mul_378: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_76);  sub_112 = None
    mul_379: "f32[8, 577, 768]" = torch.ops.aten.mul.Tensor(mul_378, primals_689);  mul_378 = None
    add_340: "f32[8, 577, 768]" = torch.ops.aten.add.Tensor(mul_379, primals_690);  mul_379 = primals_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:356, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    slice_3: "f32[8, 577, 768]" = torch.ops.aten.slice.Tensor(add_340, 0, 0, 9223372036854775807);  add_340 = None
    select_110: "f32[8, 768]" = torch.ops.aten.select.int(slice_3, 1, 0);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:357, code: x = self.head_drop(x)
    clone_511: "f32[8, 768]" = torch.ops.aten.clone.default(select_110);  select_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:358, code: return x if pre_logits else self.head(x)
    permute_489: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_691, [1, 0]);  primals_691 = None
    addmm_156: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_692, clone_511, permute_489);  primals_692 = None
    permute_490: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_489, [1, 0]);  permute_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_494: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_488, [1, 0]);  permute_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_498: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_487, [1, 0]);  permute_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    div_37: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_75, 768);  rsqrt_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    permute_502: "f32[768, 768]" = torch.ops.aten.permute.default(permute_486, [1, 0]);  permute_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    alias_38: "f32[8, 16, 1, 48]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_508: "f32[768, 768]" = torch.ops.aten.permute.default(permute_483, [1, 0]);  permute_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_513: "f32[768, 768]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_518: "f32[768, 768]" = torch.ops.aten.permute.default(permute_479, [1, 0]);  permute_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_522: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_526: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    div_39: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_73, 768);  rsqrt_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    permute_530: "f32[768, 768]" = torch.ops.aten.permute.default(permute_476, [1, 0]);  permute_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    alias_39: "f32[8, 16, 1, 48]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_536: "f32[768, 768]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_541: "f32[768, 768]" = torch.ops.aten.permute.default(permute_471, [1, 0]);  permute_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_546: "f32[768, 768]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_550: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_468, [1, 0]);  permute_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_554: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_467, [1, 0]);  permute_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_41: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_71, 768);  rsqrt_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_558: "f32[768, 768]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_563: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_711, [0, 2, 1]);  view_711 = None
    permute_564: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_712, [0, 2, 1]);  view_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_568: "f32[16, 16]" = torch.ops.aten.permute.default(permute_463, [1, 0]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_40: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_574: "f32[16, 16]" = torch.ops.aten.permute.default(permute_460, [1, 0]);  permute_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_577: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_704, [0, 2, 1]);  view_704 = None
    permute_578: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_705, [0, 2, 1]);  view_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_581: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_42: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_70, 768);  rsqrt_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_585: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_455, [1, 0]);  permute_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_589: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_43: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_69, 768);  rsqrt_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_593: "f32[768, 768]" = torch.ops.aten.permute.default(permute_453, [1, 0]);  permute_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_598: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_691, [0, 2, 1]);  view_691 = None
    permute_599: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_692, [0, 2, 1]);  view_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_603: "f32[16, 16]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_41: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_609: "f32[16, 16]" = torch.ops.aten.permute.default(permute_447, [1, 0]);  permute_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_612: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_684, [0, 2, 1]);  view_684 = None
    permute_613: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_685, [0, 2, 1]);  view_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_616: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_443, [1, 0]);  permute_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_44: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_68, 768);  rsqrt_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_620: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_624: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_45: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_67, 768);  rsqrt_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_628: "f32[768, 768]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_633: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_671, [0, 2, 1]);  view_671 = None
    permute_634: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_672, [0, 2, 1]);  view_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_638: "f32[16, 16]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_42: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_644: "f32[16, 16]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_647: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_664, [0, 2, 1]);  view_664 = None
    permute_648: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_665, [0, 2, 1]);  view_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_651: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_430, [1, 0]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_46: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_66, 768);  rsqrt_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_655: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_659: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_47: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_65, 768);  rsqrt_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_663: "f32[768, 768]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_668: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_651, [0, 2, 1]);  view_651 = None
    permute_669: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_652, [0, 2, 1]);  view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_673: "f32[16, 16]" = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_43: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_679: "f32[16, 16]" = torch.ops.aten.permute.default(permute_421, [1, 0]);  permute_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_682: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_644, [0, 2, 1]);  view_644 = None
    permute_683: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_645, [0, 2, 1]);  view_645 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_686: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_48: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_64, 768);  rsqrt_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_690: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_416, [1, 0]);  permute_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_694: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_415, [1, 0]);  permute_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_49: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_63, 768);  rsqrt_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_698: "f32[768, 768]" = torch.ops.aten.permute.default(permute_414, [1, 0]);  permute_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_703: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_631, [0, 2, 1]);  view_631 = None
    permute_704: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_632, [0, 2, 1]);  view_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_708: "f32[16, 16]" = torch.ops.aten.permute.default(permute_411, [1, 0]);  permute_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_44: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_714: "f32[16, 16]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_717: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_624, [0, 2, 1]);  view_624 = None
    permute_718: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_625, [0, 2, 1]);  view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_721: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_50: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_62, 768);  rsqrt_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_725: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_403, [1, 0]);  permute_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_729: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_51: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_61, 768);  rsqrt_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_733: "f32[768, 768]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_738: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_611, [0, 2, 1]);  view_611 = None
    permute_739: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_612, [0, 2, 1]);  view_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_743: "f32[16, 16]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_45: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_749: "f32[16, 16]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_752: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_604, [0, 2, 1]);  view_604 = None
    permute_753: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_605, [0, 2, 1]);  view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_756: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_391, [1, 0]);  permute_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_52: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_60, 768);  rsqrt_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_760: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_390, [1, 0]);  permute_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_764: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_389, [1, 0]);  permute_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_53: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_59, 768);  rsqrt_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_768: "f32[768, 768]" = torch.ops.aten.permute.default(permute_388, [1, 0]);  permute_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_773: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_591, [0, 2, 1]);  view_591 = None
    permute_774: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_592, [0, 2, 1]);  view_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_778: "f32[16, 16]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_46: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_784: "f32[16, 16]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_787: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_584, [0, 2, 1]);  view_584 = None
    permute_788: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_585, [0, 2, 1]);  view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_791: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_54: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_58, 768);  rsqrt_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_795: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_799: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_55: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_57, 768);  rsqrt_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_803: "f32[768, 768]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_808: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_571, [0, 2, 1]);  view_571 = None
    permute_809: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_572, [0, 2, 1]);  view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_813: "f32[16, 16]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_47: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_819: "f32[16, 16]" = torch.ops.aten.permute.default(permute_369, [1, 0]);  permute_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_822: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_564, [0, 2, 1]);  view_564 = None
    permute_823: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_565, [0, 2, 1]);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_826: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_365, [1, 0]);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_56: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_56, 768);  rsqrt_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_830: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_834: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_57: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_55, 768);  rsqrt_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_838: "f32[768, 768]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_843: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_551, [0, 2, 1]);  view_551 = None
    permute_844: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_552, [0, 2, 1]);  view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_848: "f32[16, 16]" = torch.ops.aten.permute.default(permute_359, [1, 0]);  permute_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_48: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_854: "f32[16, 16]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_857: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_544, [0, 2, 1]);  view_544 = None
    permute_858: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_545, [0, 2, 1]);  view_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_861: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_58: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_54, 768);  rsqrt_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_865: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_351, [1, 0]);  permute_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_869: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_59: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_53, 768);  rsqrt_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_873: "f32[768, 768]" = torch.ops.aten.permute.default(permute_349, [1, 0]);  permute_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_878: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_531, [0, 2, 1]);  view_531 = None
    permute_879: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_532, [0, 2, 1]);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_883: "f32[16, 16]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_49: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_889: "f32[16, 16]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_892: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_524, [0, 2, 1]);  view_524 = None
    permute_893: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_525, [0, 2, 1]);  view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_896: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_60: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_52, 768);  rsqrt_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_900: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_904: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_61: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_51, 768);  rsqrt_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_908: "f32[768, 768]" = torch.ops.aten.permute.default(permute_336, [1, 0]);  permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_913: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_511, [0, 2, 1]);  view_511 = None
    permute_914: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_512, [0, 2, 1]);  view_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_918: "f32[16, 16]" = torch.ops.aten.permute.default(permute_333, [1, 0]);  permute_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_50: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_924: "f32[16, 16]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_927: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_504, [0, 2, 1]);  view_504 = None
    permute_928: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_505, [0, 2, 1]);  view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_931: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_62: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 768);  rsqrt_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_935: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_325, [1, 0]);  permute_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_939: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_63: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 768);  rsqrt_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_943: "f32[768, 768]" = torch.ops.aten.permute.default(permute_323, [1, 0]);  permute_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_948: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_491, [0, 2, 1]);  view_491 = None
    permute_949: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_492, [0, 2, 1]);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_953: "f32[16, 16]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_51: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_959: "f32[16, 16]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_962: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_484, [0, 2, 1]);  view_484 = None
    permute_963: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_485, [0, 2, 1]);  view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_966: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_64: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 768);  rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_970: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_974: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_65: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 768);  rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_978: "f32[768, 768]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_983: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_471, [0, 2, 1]);  view_471 = None
    permute_984: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_988: "f32[16, 16]" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_52: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_994: "f32[16, 16]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_997: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_464, [0, 2, 1]);  view_464 = None
    permute_998: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_465, [0, 2, 1]);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1001: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_66: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 768);  rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1005: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1009: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_67: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 768);  rsqrt_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1013: "f32[768, 768]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1018: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_451, [0, 2, 1]);  view_451 = None
    permute_1019: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_452, [0, 2, 1]);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1023: "f32[16, 16]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_53: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1029: "f32[16, 16]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1032: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_444, [0, 2, 1]);  view_444 = None
    permute_1033: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_445, [0, 2, 1]);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1036: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_68: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 768);  rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1040: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1044: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_69: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 768);  rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1048: "f32[768, 768]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1053: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_431, [0, 2, 1]);  view_431 = None
    permute_1054: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_432, [0, 2, 1]);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1058: "f32[16, 16]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_54: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1064: "f32[16, 16]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1067: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_424, [0, 2, 1]);  view_424 = None
    permute_1068: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1071: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_70: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 768);  rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1075: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1079: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_71: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 768);  rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1083: "f32[768, 768]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1088: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None
    permute_1089: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_412, [0, 2, 1]);  view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1093: "f32[16, 16]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_55: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1099: "f32[16, 16]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1102: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_404, [0, 2, 1]);  view_404 = None
    permute_1103: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_405, [0, 2, 1]);  view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1106: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_72: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 768);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1110: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1114: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_259, [1, 0]);  permute_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_73: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 768);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1118: "f32[768, 768]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1123: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_391, [0, 2, 1]);  view_391 = None
    permute_1124: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_392, [0, 2, 1]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1128: "f32[16, 16]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_56: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1134: "f32[16, 16]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1137: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_384, [0, 2, 1]);  view_384 = None
    permute_1138: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1141: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_74: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 768);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1145: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1149: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_75: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 768);  rsqrt_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1153: "f32[768, 768]" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1158: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_371, [0, 2, 1]);  view_371 = None
    permute_1159: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1163: "f32[16, 16]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_57: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1169: "f32[16, 16]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1172: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
    permute_1173: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1176: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_76: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 768);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1180: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1184: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_77: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 768);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1188: "f32[768, 768]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1193: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
    permute_1194: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1198: "f32[16, 16]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_58: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1204: "f32[16, 16]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1207: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_344, [0, 2, 1]);  view_344 = None
    permute_1208: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_345, [0, 2, 1]);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1211: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_78: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 768);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1215: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1219: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_79: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 768);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1223: "f32[768, 768]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1228: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_331, [0, 2, 1]);  view_331 = None
    permute_1229: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_332, [0, 2, 1]);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1233: "f32[16, 16]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_59: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1239: "f32[16, 16]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1242: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None
    permute_1243: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1246: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_80: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 768);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1250: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1254: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_81: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 768);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1258: "f32[768, 768]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1263: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
    permute_1264: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1268: "f32[16, 16]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_60: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1274: "f32[16, 16]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1277: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
    permute_1278: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_305, [0, 2, 1]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1281: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_82: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 768);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1285: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1289: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_83: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 768);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1293: "f32[768, 768]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1298: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_291, [0, 2, 1]);  view_291 = None
    permute_1299: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_292, [0, 2, 1]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1303: "f32[16, 16]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_61: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1309: "f32[16, 16]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1312: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_284, [0, 2, 1]);  view_284 = None
    permute_1313: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1316: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_84: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 768);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1320: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1324: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_85: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 768);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1328: "f32[768, 768]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1333: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None
    permute_1334: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_272, [0, 2, 1]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1338: "f32[16, 16]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_62: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1344: "f32[16, 16]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1347: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
    permute_1348: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_265, [0, 2, 1]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1351: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_86: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 768);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1355: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1359: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_87: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1363: "f32[768, 768]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1368: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    permute_1369: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1373: "f32[16, 16]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_63: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1379: "f32[16, 16]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1382: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
    permute_1383: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1386: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_88: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1390: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1394: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_89: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1398: "f32[768, 768]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1403: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
    permute_1404: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1408: "f32[16, 16]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_64: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1414: "f32[16, 16]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1417: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_224, [0, 2, 1]);  view_224 = None
    permute_1418: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_225, [0, 2, 1]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1421: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_90: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1425: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1429: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_91: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1433: "f32[768, 768]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1438: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
    permute_1439: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1443: "f32[16, 16]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_65: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1449: "f32[16, 16]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1452: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
    permute_1453: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1456: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_92: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1460: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1464: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_93: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1468: "f32[768, 768]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1473: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    permute_1474: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_192, [0, 2, 1]);  view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1478: "f32[16, 16]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_66: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1484: "f32[16, 16]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1487: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_184, [0, 2, 1]);  view_184 = None
    permute_1488: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1491: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_94: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1495: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1499: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_95: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1503: "f32[768, 768]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1508: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    permute_1509: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1513: "f32[16, 16]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_67: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1519: "f32[16, 16]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1522: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None
    permute_1523: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1526: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_96: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1530: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1534: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_97: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1538: "f32[768, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1543: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
    permute_1544: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1548: "f32[16, 16]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_68: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1554: "f32[16, 16]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1557: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    permute_1558: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1561: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_98: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1565: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1569: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_99: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1573: "f32[768, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1578: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None
    permute_1579: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1583: "f32[16, 16]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_69: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1589: "f32[16, 16]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1592: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
    permute_1593: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1596: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_100: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1600: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1604: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_101: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1608: "f32[768, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1613: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
    permute_1614: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1618: "f32[16, 16]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_70: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1624: "f32[16, 16]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1627: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_104, [0, 2, 1]);  view_104 = None
    permute_1628: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_105, [0, 2, 1]);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1631: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_102: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1635: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1639: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_103: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1643: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1648: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_91, [0, 2, 1]);  view_91 = None
    permute_1649: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1653: "f32[16, 16]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_71: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1659: "f32[16, 16]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1662: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_84, [0, 2, 1]);  view_84 = None
    permute_1663: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1666: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_104: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1670: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1674: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_105: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1678: "f32[768, 768]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1683: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
    permute_1684: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1688: "f32[16, 16]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_72: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1694: "f32[16, 16]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1697: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    permute_1698: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1701: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_106: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1705: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1709: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_107: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1713: "f32[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1718: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_51, [0, 2, 1]);  view_51 = None
    permute_1719: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1723: "f32[16, 16]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_73: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1729: "f32[16, 16]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1732: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    permute_1733: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1736: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_108: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1740: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1744: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_109: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1748: "f32[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1753: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
    permute_1754: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1758: "f32[16, 16]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_74: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1764: "f32[16, 16]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1767: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
    permute_1768: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1771: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_110: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_1775: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_1779: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    div_111: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    permute_1783: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    permute_1788: "f32[128, 576, 576]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    permute_1789: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1793: "f32[16, 16]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    alias_75: "f32[8, 16, 576, 576]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1799: "f32[16, 16]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    permute_1802: "f32[128, 48, 576]" = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
    permute_1803: "f32[128, 576, 48]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_1806: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    div_112: "f32[8, 576, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [addmm_156, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_75, primals_76, primals_77, primals_78, primals_79, primals_81, primals_91, primals_97, primals_107, primals_113, primals_123, primals_129, primals_139, primals_145, primals_155, primals_161, primals_171, primals_177, primals_187, primals_193, primals_203, primals_209, primals_219, primals_225, primals_235, primals_241, primals_251, primals_257, primals_267, primals_273, primals_283, primals_289, primals_299, primals_305, primals_315, primals_321, primals_331, primals_337, primals_347, primals_353, primals_363, primals_369, primals_379, primals_385, primals_395, primals_401, primals_411, primals_417, primals_427, primals_433, primals_443, primals_449, primals_459, primals_465, primals_475, primals_481, primals_491, primals_497, primals_507, primals_513, primals_523, primals_529, primals_539, primals_545, primals_555, primals_561, primals_571, primals_577, primals_587, primals_593, primals_603, primals_609, primals_619, primals_625, primals_635, primals_641, primals_651, primals_657, primals_667, primals_673, primals_683, primals_689, primals_693, mul, view_1, view_7, view_9, view_15, addmm_1, mul_4, view_17, addmm_2, view_19, addmm_3, mul_10, view_21, view_27, view_29, view_35, addmm_5, mul_14, view_37, addmm_6, view_39, addmm_7, mul_20, view_41, view_47, view_49, view_55, addmm_9, mul_24, view_57, addmm_10, view_59, addmm_11, mul_30, view_61, view_67, view_69, view_75, addmm_13, mul_34, view_77, addmm_14, view_79, addmm_15, mul_40, view_81, view_87, view_89, view_95, addmm_17, mul_44, view_97, addmm_18, view_99, addmm_19, mul_50, view_101, view_107, view_109, view_115, addmm_21, mul_54, view_117, addmm_22, view_119, addmm_23, mul_60, view_121, view_127, view_129, view_135, addmm_25, mul_64, view_137, addmm_26, view_139, addmm_27, mul_70, view_141, view_147, view_149, view_155, addmm_29, mul_74, view_157, addmm_30, view_159, addmm_31, mul_80, view_161, view_167, view_169, view_175, addmm_33, mul_84, view_177, addmm_34, view_179, addmm_35, mul_90, view_181, view_187, view_189, view_195, addmm_37, mul_94, view_197, addmm_38, view_199, addmm_39, mul_100, view_201, view_207, view_209, view_215, addmm_41, mul_104, view_217, addmm_42, view_219, addmm_43, mul_110, view_221, view_227, view_229, view_235, addmm_45, mul_114, view_237, addmm_46, view_239, addmm_47, mul_120, view_241, view_247, view_249, view_255, addmm_49, mul_124, view_257, addmm_50, view_259, addmm_51, mul_130, view_261, view_267, view_269, view_275, addmm_53, mul_134, view_277, addmm_54, view_279, addmm_55, mul_140, view_281, view_287, view_289, view_295, addmm_57, mul_144, view_297, addmm_58, view_299, addmm_59, mul_150, view_301, view_307, view_309, view_315, addmm_61, mul_154, view_317, addmm_62, view_319, addmm_63, mul_160, view_321, view_327, view_329, view_335, addmm_65, mul_164, view_337, addmm_66, view_339, addmm_67, mul_170, view_341, view_347, view_349, view_355, addmm_69, mul_174, view_357, addmm_70, view_359, addmm_71, mul_180, view_361, view_367, view_369, view_375, addmm_73, mul_184, view_377, addmm_74, view_379, addmm_75, mul_190, view_381, view_387, view_389, view_395, addmm_77, mul_194, view_397, addmm_78, view_399, addmm_79, mul_200, view_401, view_407, view_409, view_415, addmm_81, mul_204, view_417, addmm_82, view_419, addmm_83, mul_210, view_421, view_427, view_429, view_435, addmm_85, mul_214, view_437, addmm_86, view_439, addmm_87, mul_220, view_441, view_447, view_449, view_455, addmm_89, mul_224, view_457, addmm_90, view_459, addmm_91, mul_230, view_461, view_467, view_469, view_475, addmm_93, mul_234, view_477, addmm_94, view_479, addmm_95, mul_240, view_481, view_487, view_489, view_495, addmm_97, mul_244, view_497, addmm_98, view_499, addmm_99, mul_250, view_501, view_507, view_509, view_515, addmm_101, mul_254, view_517, addmm_102, view_519, addmm_103, mul_260, view_521, view_527, view_529, view_535, addmm_105, mul_264, view_537, addmm_106, view_539, addmm_107, mul_270, view_541, view_547, view_549, view_555, addmm_109, mul_274, view_557, addmm_110, view_559, addmm_111, mul_280, view_561, view_567, view_569, view_575, addmm_113, mul_284, view_577, addmm_114, view_579, addmm_115, mul_290, view_581, view_587, view_589, view_595, addmm_117, mul_294, view_597, addmm_118, view_599, addmm_119, mul_300, view_601, view_607, view_609, view_615, addmm_121, mul_304, view_617, addmm_122, view_619, addmm_123, mul_310, view_621, view_627, view_629, view_635, addmm_125, mul_314, view_637, addmm_126, view_639, addmm_127, mul_320, view_641, view_647, view_649, view_655, addmm_129, mul_324, view_657, addmm_130, view_659, addmm_131, mul_330, view_661, view_667, view_669, view_675, addmm_133, mul_334, view_677, addmm_134, view_679, addmm_135, mul_340, view_681, view_687, view_689, view_695, addmm_137, mul_344, view_697, addmm_138, view_699, addmm_139, mul_350, view_701, view_707, view_709, view_715, addmm_141, mul_354, view_717, addmm_142, view_719, addmm_143, cat, getitem_145, rsqrt_72, select_108, permute_470, view_722, permute_472, permute_474, getitem_147, getitem_148, getitem_149, getitem_152, getitem_153, view_729, addmm_147, mul_363, view_731, addmm_148, view_733, addmm_149, cat_1, getitem_158, rsqrt_74, select_109, permute_480, view_736, permute_482, permute_484, getitem_160, getitem_161, getitem_162, getitem_165, getitem_166, view_743, addmm_153, mul_372, view_745, addmm_154, view_747, addmm_155, cat_2, getitem_171, rsqrt_76, clone_511, permute_490, permute_494, permute_498, div_37, permute_502, alias_38, permute_508, permute_513, permute_518, permute_522, permute_526, div_39, permute_530, alias_39, permute_536, permute_541, permute_546, permute_550, permute_554, div_41, permute_558, permute_563, permute_564, permute_568, alias_40, permute_574, permute_577, permute_578, permute_581, div_42, permute_585, permute_589, div_43, permute_593, permute_598, permute_599, permute_603, alias_41, permute_609, permute_612, permute_613, permute_616, div_44, permute_620, permute_624, div_45, permute_628, permute_633, permute_634, permute_638, alias_42, permute_644, permute_647, permute_648, permute_651, div_46, permute_655, permute_659, div_47, permute_663, permute_668, permute_669, permute_673, alias_43, permute_679, permute_682, permute_683, permute_686, div_48, permute_690, permute_694, div_49, permute_698, permute_703, permute_704, permute_708, alias_44, permute_714, permute_717, permute_718, permute_721, div_50, permute_725, permute_729, div_51, permute_733, permute_738, permute_739, permute_743, alias_45, permute_749, permute_752, permute_753, permute_756, div_52, permute_760, permute_764, div_53, permute_768, permute_773, permute_774, permute_778, alias_46, permute_784, permute_787, permute_788, permute_791, div_54, permute_795, permute_799, div_55, permute_803, permute_808, permute_809, permute_813, alias_47, permute_819, permute_822, permute_823, permute_826, div_56, permute_830, permute_834, div_57, permute_838, permute_843, permute_844, permute_848, alias_48, permute_854, permute_857, permute_858, permute_861, div_58, permute_865, permute_869, div_59, permute_873, permute_878, permute_879, permute_883, alias_49, permute_889, permute_892, permute_893, permute_896, div_60, permute_900, permute_904, div_61, permute_908, permute_913, permute_914, permute_918, alias_50, permute_924, permute_927, permute_928, permute_931, div_62, permute_935, permute_939, div_63, permute_943, permute_948, permute_949, permute_953, alias_51, permute_959, permute_962, permute_963, permute_966, div_64, permute_970, permute_974, div_65, permute_978, permute_983, permute_984, permute_988, alias_52, permute_994, permute_997, permute_998, permute_1001, div_66, permute_1005, permute_1009, div_67, permute_1013, permute_1018, permute_1019, permute_1023, alias_53, permute_1029, permute_1032, permute_1033, permute_1036, div_68, permute_1040, permute_1044, div_69, permute_1048, permute_1053, permute_1054, permute_1058, alias_54, permute_1064, permute_1067, permute_1068, permute_1071, div_70, permute_1075, permute_1079, div_71, permute_1083, permute_1088, permute_1089, permute_1093, alias_55, permute_1099, permute_1102, permute_1103, permute_1106, div_72, permute_1110, permute_1114, div_73, permute_1118, permute_1123, permute_1124, permute_1128, alias_56, permute_1134, permute_1137, permute_1138, permute_1141, div_74, permute_1145, permute_1149, div_75, permute_1153, permute_1158, permute_1159, permute_1163, alias_57, permute_1169, permute_1172, permute_1173, permute_1176, div_76, permute_1180, permute_1184, div_77, permute_1188, permute_1193, permute_1194, permute_1198, alias_58, permute_1204, permute_1207, permute_1208, permute_1211, div_78, permute_1215, permute_1219, div_79, permute_1223, permute_1228, permute_1229, permute_1233, alias_59, permute_1239, permute_1242, permute_1243, permute_1246, div_80, permute_1250, permute_1254, div_81, permute_1258, permute_1263, permute_1264, permute_1268, alias_60, permute_1274, permute_1277, permute_1278, permute_1281, div_82, permute_1285, permute_1289, div_83, permute_1293, permute_1298, permute_1299, permute_1303, alias_61, permute_1309, permute_1312, permute_1313, permute_1316, div_84, permute_1320, permute_1324, div_85, permute_1328, permute_1333, permute_1334, permute_1338, alias_62, permute_1344, permute_1347, permute_1348, permute_1351, div_86, permute_1355, permute_1359, div_87, permute_1363, permute_1368, permute_1369, permute_1373, alias_63, permute_1379, permute_1382, permute_1383, permute_1386, div_88, permute_1390, permute_1394, div_89, permute_1398, permute_1403, permute_1404, permute_1408, alias_64, permute_1414, permute_1417, permute_1418, permute_1421, div_90, permute_1425, permute_1429, div_91, permute_1433, permute_1438, permute_1439, permute_1443, alias_65, permute_1449, permute_1452, permute_1453, permute_1456, div_92, permute_1460, permute_1464, div_93, permute_1468, permute_1473, permute_1474, permute_1478, alias_66, permute_1484, permute_1487, permute_1488, permute_1491, div_94, permute_1495, permute_1499, div_95, permute_1503, permute_1508, permute_1509, permute_1513, alias_67, permute_1519, permute_1522, permute_1523, permute_1526, div_96, permute_1530, permute_1534, div_97, permute_1538, permute_1543, permute_1544, permute_1548, alias_68, permute_1554, permute_1557, permute_1558, permute_1561, div_98, permute_1565, permute_1569, div_99, permute_1573, permute_1578, permute_1579, permute_1583, alias_69, permute_1589, permute_1592, permute_1593, permute_1596, div_100, permute_1600, permute_1604, div_101, permute_1608, permute_1613, permute_1614, permute_1618, alias_70, permute_1624, permute_1627, permute_1628, permute_1631, div_102, permute_1635, permute_1639, div_103, permute_1643, permute_1648, permute_1649, permute_1653, alias_71, permute_1659, permute_1662, permute_1663, permute_1666, div_104, permute_1670, permute_1674, div_105, permute_1678, permute_1683, permute_1684, permute_1688, alias_72, permute_1694, permute_1697, permute_1698, permute_1701, div_106, permute_1705, permute_1709, div_107, permute_1713, permute_1718, permute_1719, permute_1723, alias_73, permute_1729, permute_1732, permute_1733, permute_1736, div_108, permute_1740, permute_1744, div_109, permute_1748, permute_1753, permute_1754, permute_1758, alias_74, permute_1764, permute_1767, permute_1768, permute_1771, div_110, permute_1775, permute_1779, div_111, permute_1783, permute_1788, permute_1789, permute_1793, alias_75, permute_1799, permute_1802, permute_1803, permute_1806, div_112]
    