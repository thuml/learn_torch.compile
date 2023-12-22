from __future__ import annotations



def forward(self, primals_1: "f32[512]", primals_2: "f32[512]", primals_3: "f32[128]", primals_4: "f32[128]", primals_5: "f32[128]", primals_6: "f32[128]", primals_7: "f32[128]", primals_8: "f32[128]", primals_9: "f32[128]", primals_10: "f32[128]", primals_11: "f32[128]", primals_12: "f32[128]", primals_13: "f32[128]", primals_14: "f32[128]", primals_15: "f32[128]", primals_16: "f32[128]", primals_17: "f32[512]", primals_18: "f32[512]", primals_19: "f32[128]", primals_20: "f32[128]", primals_21: "f32[128]", primals_22: "f32[128]", primals_23: "f32[128]", primals_24: "f32[128]", primals_25: "f32[128]", primals_26: "f32[128]", primals_27: "f32[128]", primals_28: "f32[128]", primals_29: "f32[128]", primals_30: "f32[128]", primals_31: "f32[128]", primals_32: "f32[128]", primals_33: "f32[512]", primals_34: "f32[512]", primals_35: "f32[128]", primals_36: "f32[128]", primals_37: "f32[128]", primals_38: "f32[128]", primals_39: "f32[128]", primals_40: "f32[128]", primals_41: "f32[128]", primals_42: "f32[128]", primals_43: "f32[128]", primals_44: "f32[128]", primals_45: "f32[128]", primals_46: "f32[128]", primals_47: "f32[128]", primals_48: "f32[128]", primals_49: "f32[512]", primals_50: "f32[512]", primals_51: "f32[128]", primals_52: "f32[128]", primals_53: "f32[128]", primals_54: "f32[128]", primals_55: "f32[128]", primals_56: "f32[128]", primals_57: "f32[128]", primals_58: "f32[128]", primals_59: "f32[128]", primals_60: "f32[128]", primals_61: "f32[128]", primals_62: "f32[128]", primals_63: "f32[128]", primals_64: "f32[128]", primals_65: "f32[512]", primals_66: "f32[512]", primals_67: "f32[128]", primals_68: "f32[128]", primals_69: "f32[128]", primals_70: "f32[128]", primals_71: "f32[128]", primals_72: "f32[128]", primals_73: "f32[128]", primals_74: "f32[128]", primals_75: "f32[128]", primals_76: "f32[128]", primals_77: "f32[128]", primals_78: "f32[128]", primals_79: "f32[128]", primals_80: "f32[128]", primals_81: "f32[512]", primals_82: "f32[512]", primals_83: "f32[128]", primals_84: "f32[128]", primals_85: "f32[128]", primals_86: "f32[128]", primals_87: "f32[128]", primals_88: "f32[128]", primals_89: "f32[128]", primals_90: "f32[128]", primals_91: "f32[128]", primals_92: "f32[128]", primals_93: "f32[128]", primals_94: "f32[128]", primals_95: "f32[128]", primals_96: "f32[128]", primals_97: "f32[512]", primals_98: "f32[512]", primals_99: "f32[128]", primals_100: "f32[128]", primals_101: "f32[128]", primals_102: "f32[128]", primals_103: "f32[128]", primals_104: "f32[128]", primals_105: "f32[128]", primals_106: "f32[128]", primals_107: "f32[128]", primals_108: "f32[128]", primals_109: "f32[128]", primals_110: "f32[128]", primals_111: "f32[128]", primals_112: "f32[128]", primals_113: "f32[512]", primals_114: "f32[512]", primals_115: "f32[128]", primals_116: "f32[128]", primals_117: "f32[128]", primals_118: "f32[128]", primals_119: "f32[128]", primals_120: "f32[128]", primals_121: "f32[128]", primals_122: "f32[128]", primals_123: "f32[128]", primals_124: "f32[128]", primals_125: "f32[128]", primals_126: "f32[128]", primals_127: "f32[128]", primals_128: "f32[128]", primals_129: "f32[512]", primals_130: "f32[512]", primals_131: "f32[128]", primals_132: "f32[128]", primals_133: "f32[128]", primals_134: "f32[128]", primals_135: "f32[128]", primals_136: "f32[128]", primals_137: "f32[128]", primals_138: "f32[128]", primals_139: "f32[128]", primals_140: "f32[128]", primals_141: "f32[128]", primals_142: "f32[128]", primals_143: "f32[128]", primals_144: "f32[128]", primals_145: "f32[512]", primals_146: "f32[512]", primals_147: "f32[128]", primals_148: "f32[128]", primals_149: "f32[128]", primals_150: "f32[128]", primals_151: "f32[128]", primals_152: "f32[128]", primals_153: "f32[128]", primals_154: "f32[128]", primals_155: "f32[128]", primals_156: "f32[128]", primals_157: "f32[128]", primals_158: "f32[128]", primals_159: "f32[128]", primals_160: "f32[128]", primals_161: "f32[512]", primals_162: "f32[512]", primals_163: "f32[128]", primals_164: "f32[128]", primals_165: "f32[128]", primals_166: "f32[128]", primals_167: "f32[128]", primals_168: "f32[128]", primals_169: "f32[128]", primals_170: "f32[128]", primals_171: "f32[128]", primals_172: "f32[128]", primals_173: "f32[128]", primals_174: "f32[128]", primals_175: "f32[128]", primals_176: "f32[128]", primals_177: "f32[512]", primals_178: "f32[512]", primals_179: "f32[128]", primals_180: "f32[128]", primals_181: "f32[128]", primals_182: "f32[128]", primals_183: "f32[128]", primals_184: "f32[128]", primals_185: "f32[128]", primals_186: "f32[128]", primals_187: "f32[128]", primals_188: "f32[128]", primals_189: "f32[128]", primals_190: "f32[128]", primals_191: "f32[128]", primals_192: "f32[128]", primals_193: "f32[512]", primals_194: "f32[512]", primals_195: "f32[128]", primals_196: "f32[128]", primals_197: "f32[128]", primals_198: "f32[128]", primals_199: "f32[128]", primals_200: "f32[128]", primals_201: "f32[128]", primals_202: "f32[128]", primals_203: "f32[128]", primals_204: "f32[128]", primals_205: "f32[128]", primals_206: "f32[128]", primals_207: "f32[128]", primals_208: "f32[128]", primals_209: "f32[512]", primals_210: "f32[512]", primals_211: "f32[128]", primals_212: "f32[128]", primals_213: "f32[128]", primals_214: "f32[128]", primals_215: "f32[128]", primals_216: "f32[128]", primals_217: "f32[128]", primals_218: "f32[128]", primals_219: "f32[128]", primals_220: "f32[128]", primals_221: "f32[128]", primals_222: "f32[128]", primals_223: "f32[128]", primals_224: "f32[128]", primals_225: "f32[512]", primals_226: "f32[512]", primals_227: "f32[128]", primals_228: "f32[128]", primals_229: "f32[128]", primals_230: "f32[128]", primals_231: "f32[128]", primals_232: "f32[128]", primals_233: "f32[128]", primals_234: "f32[128]", primals_235: "f32[128]", primals_236: "f32[128]", primals_237: "f32[128]", primals_238: "f32[128]", primals_239: "f32[128]", primals_240: "f32[128]", primals_241: "f32[512]", primals_242: "f32[512]", primals_243: "f32[128]", primals_244: "f32[128]", primals_245: "f32[128]", primals_246: "f32[128]", primals_247: "f32[128]", primals_248: "f32[128]", primals_249: "f32[128]", primals_250: "f32[128]", primals_251: "f32[128]", primals_252: "f32[128]", primals_253: "f32[128]", primals_254: "f32[128]", primals_255: "f32[128]", primals_256: "f32[128]", primals_257: "f32[512]", primals_258: "f32[512]", primals_259: "f32[128]", primals_260: "f32[128]", primals_261: "f32[128]", primals_262: "f32[128]", primals_263: "f32[128]", primals_264: "f32[128]", primals_265: "f32[128]", primals_266: "f32[128]", primals_267: "f32[128]", primals_268: "f32[128]", primals_269: "f32[128]", primals_270: "f32[128]", primals_271: "f32[128]", primals_272: "f32[128]", primals_273: "f32[512]", primals_274: "f32[512]", primals_275: "f32[128]", primals_276: "f32[128]", primals_277: "f32[128]", primals_278: "f32[128]", primals_279: "f32[128]", primals_280: "f32[128]", primals_281: "f32[128]", primals_282: "f32[128]", primals_283: "f32[128]", primals_284: "f32[128]", primals_285: "f32[128]", primals_286: "f32[128]", primals_287: "f32[128]", primals_288: "f32[128]", primals_289: "f32[512]", primals_290: "f32[512]", primals_291: "f32[128]", primals_292: "f32[128]", primals_293: "f32[128]", primals_294: "f32[128]", primals_295: "f32[128]", primals_296: "f32[128]", primals_297: "f32[128]", primals_298: "f32[128]", primals_299: "f32[128]", primals_300: "f32[128]", primals_301: "f32[128]", primals_302: "f32[128]", primals_303: "f32[128]", primals_304: "f32[128]", primals_305: "f32[512]", primals_306: "f32[512]", primals_307: "f32[128]", primals_308: "f32[128]", primals_309: "f32[128]", primals_310: "f32[128]", primals_311: "f32[128]", primals_312: "f32[128]", primals_313: "f32[128]", primals_314: "f32[128]", primals_315: "f32[128]", primals_316: "f32[128]", primals_317: "f32[128]", primals_318: "f32[128]", primals_319: "f32[128]", primals_320: "f32[128]", primals_321: "f32[512]", primals_322: "f32[512]", primals_323: "f32[128]", primals_324: "f32[128]", primals_325: "f32[128]", primals_326: "f32[128]", primals_327: "f32[128]", primals_328: "f32[128]", primals_329: "f32[128]", primals_330: "f32[128]", primals_331: "f32[128]", primals_332: "f32[128]", primals_333: "f32[128]", primals_334: "f32[128]", primals_335: "f32[128]", primals_336: "f32[128]", primals_337: "f32[512]", primals_338: "f32[512]", primals_339: "f32[128]", primals_340: "f32[128]", primals_341: "f32[128]", primals_342: "f32[128]", primals_343: "f32[128]", primals_344: "f32[128]", primals_345: "f32[128]", primals_346: "f32[128]", primals_347: "f32[128]", primals_348: "f32[128]", primals_349: "f32[128]", primals_350: "f32[128]", primals_351: "f32[128]", primals_352: "f32[128]", primals_353: "f32[512]", primals_354: "f32[512]", primals_355: "f32[128]", primals_356: "f32[128]", primals_357: "f32[128]", primals_358: "f32[128]", primals_359: "f32[128]", primals_360: "f32[128]", primals_361: "f32[128]", primals_362: "f32[128]", primals_363: "f32[128]", primals_364: "f32[128]", primals_365: "f32[128]", primals_366: "f32[128]", primals_367: "f32[128]", primals_368: "f32[128]", primals_369: "f32[512]", primals_370: "f32[512]", primals_371: "f32[128]", primals_372: "f32[128]", primals_373: "f32[128]", primals_374: "f32[128]", primals_375: "f32[128]", primals_376: "f32[128]", primals_377: "f32[128]", primals_378: "f32[128]", primals_379: "f32[128]", primals_380: "f32[128]", primals_381: "f32[128]", primals_382: "f32[128]", primals_383: "f32[128]", primals_384: "f32[128]", primals_385: "f32[512]", primals_386: "f32[512]", primals_387: "f32[30522, 128]", primals_388: "f32[384, 30522]", primals_389: "f32[30522]", primals_390: "f32[30522, 128]", primals_391: "f32[512, 384]", primals_392: "f32[512]", primals_393: "f32[512, 512]", primals_394: "f32[2, 512]", primals_395: "f32[128, 512]", primals_396: "f32[128]", primals_397: "f32[128, 512]", primals_398: "f32[128]", primals_399: "f32[128, 128]", primals_400: "f32[128]", primals_401: "f32[128, 128]", primals_402: "f32[128]", primals_403: "f32[128, 512]", primals_404: "f32[128]", primals_405: "f32[128, 128]", primals_406: "f32[128]", primals_407: "f32[512, 128]", primals_408: "f32[512]", primals_409: "f32[128, 512]", primals_410: "f32[128]", primals_411: "f32[512, 128]", primals_412: "f32[512]", primals_413: "f32[128, 512]", primals_414: "f32[128]", primals_415: "f32[512, 128]", primals_416: "f32[512]", primals_417: "f32[128, 512]", primals_418: "f32[128]", primals_419: "f32[512, 128]", primals_420: "f32[512]", primals_421: "f32[128, 512]", primals_422: "f32[128]", primals_423: "f32[512, 128]", primals_424: "f32[512]", primals_425: "f32[128, 512]", primals_426: "f32[128]", primals_427: "f32[128, 512]", primals_428: "f32[128]", primals_429: "f32[128, 128]", primals_430: "f32[128]", primals_431: "f32[128, 128]", primals_432: "f32[128]", primals_433: "f32[128, 512]", primals_434: "f32[128]", primals_435: "f32[128, 128]", primals_436: "f32[128]", primals_437: "f32[512, 128]", primals_438: "f32[512]", primals_439: "f32[128, 512]", primals_440: "f32[128]", primals_441: "f32[512, 128]", primals_442: "f32[512]", primals_443: "f32[128, 512]", primals_444: "f32[128]", primals_445: "f32[512, 128]", primals_446: "f32[512]", primals_447: "f32[128, 512]", primals_448: "f32[128]", primals_449: "f32[512, 128]", primals_450: "f32[512]", primals_451: "f32[128, 512]", primals_452: "f32[128]", primals_453: "f32[512, 128]", primals_454: "f32[512]", primals_455: "f32[128, 512]", primals_456: "f32[128]", primals_457: "f32[128, 512]", primals_458: "f32[128]", primals_459: "f32[128, 128]", primals_460: "f32[128]", primals_461: "f32[128, 128]", primals_462: "f32[128]", primals_463: "f32[128, 512]", primals_464: "f32[128]", primals_465: "f32[128, 128]", primals_466: "f32[128]", primals_467: "f32[512, 128]", primals_468: "f32[512]", primals_469: "f32[128, 512]", primals_470: "f32[128]", primals_471: "f32[512, 128]", primals_472: "f32[512]", primals_473: "f32[128, 512]", primals_474: "f32[128]", primals_475: "f32[512, 128]", primals_476: "f32[512]", primals_477: "f32[128, 512]", primals_478: "f32[128]", primals_479: "f32[512, 128]", primals_480: "f32[512]", primals_481: "f32[128, 512]", primals_482: "f32[128]", primals_483: "f32[512, 128]", primals_484: "f32[512]", primals_485: "f32[128, 512]", primals_486: "f32[128]", primals_487: "f32[128, 512]", primals_488: "f32[128]", primals_489: "f32[128, 128]", primals_490: "f32[128]", primals_491: "f32[128, 128]", primals_492: "f32[128]", primals_493: "f32[128, 512]", primals_494: "f32[128]", primals_495: "f32[128, 128]", primals_496: "f32[128]", primals_497: "f32[512, 128]", primals_498: "f32[512]", primals_499: "f32[128, 512]", primals_500: "f32[128]", primals_501: "f32[512, 128]", primals_502: "f32[512]", primals_503: "f32[128, 512]", primals_504: "f32[128]", primals_505: "f32[512, 128]", primals_506: "f32[512]", primals_507: "f32[128, 512]", primals_508: "f32[128]", primals_509: "f32[512, 128]", primals_510: "f32[512]", primals_511: "f32[128, 512]", primals_512: "f32[128]", primals_513: "f32[512, 128]", primals_514: "f32[512]", primals_515: "f32[128, 512]", primals_516: "f32[128]", primals_517: "f32[128, 512]", primals_518: "f32[128]", primals_519: "f32[128, 128]", primals_520: "f32[128]", primals_521: "f32[128, 128]", primals_522: "f32[128]", primals_523: "f32[128, 512]", primals_524: "f32[128]", primals_525: "f32[128, 128]", primals_526: "f32[128]", primals_527: "f32[512, 128]", primals_528: "f32[512]", primals_529: "f32[128, 512]", primals_530: "f32[128]", primals_531: "f32[512, 128]", primals_532: "f32[512]", primals_533: "f32[128, 512]", primals_534: "f32[128]", primals_535: "f32[512, 128]", primals_536: "f32[512]", primals_537: "f32[128, 512]", primals_538: "f32[128]", primals_539: "f32[512, 128]", primals_540: "f32[512]", primals_541: "f32[128, 512]", primals_542: "f32[128]", primals_543: "f32[512, 128]", primals_544: "f32[512]", primals_545: "f32[128, 512]", primals_546: "f32[128]", primals_547: "f32[128, 512]", primals_548: "f32[128]", primals_549: "f32[128, 128]", primals_550: "f32[128]", primals_551: "f32[128, 128]", primals_552: "f32[128]", primals_553: "f32[128, 512]", primals_554: "f32[128]", primals_555: "f32[128, 128]", primals_556: "f32[128]", primals_557: "f32[512, 128]", primals_558: "f32[512]", primals_559: "f32[128, 512]", primals_560: "f32[128]", primals_561: "f32[512, 128]", primals_562: "f32[512]", primals_563: "f32[128, 512]", primals_564: "f32[128]", primals_565: "f32[512, 128]", primals_566: "f32[512]", primals_567: "f32[128, 512]", primals_568: "f32[128]", primals_569: "f32[512, 128]", primals_570: "f32[512]", primals_571: "f32[128, 512]", primals_572: "f32[128]", primals_573: "f32[512, 128]", primals_574: "f32[512]", primals_575: "f32[128, 512]", primals_576: "f32[128]", primals_577: "f32[128, 512]", primals_578: "f32[128]", primals_579: "f32[128, 128]", primals_580: "f32[128]", primals_581: "f32[128, 128]", primals_582: "f32[128]", primals_583: "f32[128, 512]", primals_584: "f32[128]", primals_585: "f32[128, 128]", primals_586: "f32[128]", primals_587: "f32[512, 128]", primals_588: "f32[512]", primals_589: "f32[128, 512]", primals_590: "f32[128]", primals_591: "f32[512, 128]", primals_592: "f32[512]", primals_593: "f32[128, 512]", primals_594: "f32[128]", primals_595: "f32[512, 128]", primals_596: "f32[512]", primals_597: "f32[128, 512]", primals_598: "f32[128]", primals_599: "f32[512, 128]", primals_600: "f32[512]", primals_601: "f32[128, 512]", primals_602: "f32[128]", primals_603: "f32[512, 128]", primals_604: "f32[512]", primals_605: "f32[128, 512]", primals_606: "f32[128]", primals_607: "f32[128, 512]", primals_608: "f32[128]", primals_609: "f32[128, 128]", primals_610: "f32[128]", primals_611: "f32[128, 128]", primals_612: "f32[128]", primals_613: "f32[128, 512]", primals_614: "f32[128]", primals_615: "f32[128, 128]", primals_616: "f32[128]", primals_617: "f32[512, 128]", primals_618: "f32[512]", primals_619: "f32[128, 512]", primals_620: "f32[128]", primals_621: "f32[512, 128]", primals_622: "f32[512]", primals_623: "f32[128, 512]", primals_624: "f32[128]", primals_625: "f32[512, 128]", primals_626: "f32[512]", primals_627: "f32[128, 512]", primals_628: "f32[128]", primals_629: "f32[512, 128]", primals_630: "f32[512]", primals_631: "f32[128, 512]", primals_632: "f32[128]", primals_633: "f32[512, 128]", primals_634: "f32[512]", primals_635: "f32[128, 512]", primals_636: "f32[128]", primals_637: "f32[128, 512]", primals_638: "f32[128]", primals_639: "f32[128, 128]", primals_640: "f32[128]", primals_641: "f32[128, 128]", primals_642: "f32[128]", primals_643: "f32[128, 512]", primals_644: "f32[128]", primals_645: "f32[128, 128]", primals_646: "f32[128]", primals_647: "f32[512, 128]", primals_648: "f32[512]", primals_649: "f32[128, 512]", primals_650: "f32[128]", primals_651: "f32[512, 128]", primals_652: "f32[512]", primals_653: "f32[128, 512]", primals_654: "f32[128]", primals_655: "f32[512, 128]", primals_656: "f32[512]", primals_657: "f32[128, 512]", primals_658: "f32[128]", primals_659: "f32[512, 128]", primals_660: "f32[512]", primals_661: "f32[128, 512]", primals_662: "f32[128]", primals_663: "f32[512, 128]", primals_664: "f32[512]", primals_665: "f32[128, 512]", primals_666: "f32[128]", primals_667: "f32[128, 512]", primals_668: "f32[128]", primals_669: "f32[128, 128]", primals_670: "f32[128]", primals_671: "f32[128, 128]", primals_672: "f32[128]", primals_673: "f32[128, 512]", primals_674: "f32[128]", primals_675: "f32[128, 128]", primals_676: "f32[128]", primals_677: "f32[512, 128]", primals_678: "f32[512]", primals_679: "f32[128, 512]", primals_680: "f32[128]", primals_681: "f32[512, 128]", primals_682: "f32[512]", primals_683: "f32[128, 512]", primals_684: "f32[128]", primals_685: "f32[512, 128]", primals_686: "f32[512]", primals_687: "f32[128, 512]", primals_688: "f32[128]", primals_689: "f32[512, 128]", primals_690: "f32[512]", primals_691: "f32[128, 512]", primals_692: "f32[128]", primals_693: "f32[512, 128]", primals_694: "f32[512]", primals_695: "f32[128, 512]", primals_696: "f32[128]", primals_697: "f32[128, 512]", primals_698: "f32[128]", primals_699: "f32[128, 128]", primals_700: "f32[128]", primals_701: "f32[128, 128]", primals_702: "f32[128]", primals_703: "f32[128, 512]", primals_704: "f32[128]", primals_705: "f32[128, 128]", primals_706: "f32[128]", primals_707: "f32[512, 128]", primals_708: "f32[512]", primals_709: "f32[128, 512]", primals_710: "f32[128]", primals_711: "f32[512, 128]", primals_712: "f32[512]", primals_713: "f32[128, 512]", primals_714: "f32[128]", primals_715: "f32[512, 128]", primals_716: "f32[512]", primals_717: "f32[128, 512]", primals_718: "f32[128]", primals_719: "f32[512, 128]", primals_720: "f32[512]", primals_721: "f32[128, 512]", primals_722: "f32[128]", primals_723: "f32[512, 128]", primals_724: "f32[512]", primals_725: "f32[128, 512]", primals_726: "f32[128]", primals_727: "f32[128, 512]", primals_728: "f32[128]", primals_729: "f32[128, 128]", primals_730: "f32[128]", primals_731: "f32[128, 128]", primals_732: "f32[128]", primals_733: "f32[128, 512]", primals_734: "f32[128]", primals_735: "f32[128, 128]", primals_736: "f32[128]", primals_737: "f32[512, 128]", primals_738: "f32[512]", primals_739: "f32[128, 512]", primals_740: "f32[128]", primals_741: "f32[512, 128]", primals_742: "f32[512]", primals_743: "f32[128, 512]", primals_744: "f32[128]", primals_745: "f32[512, 128]", primals_746: "f32[512]", primals_747: "f32[128, 512]", primals_748: "f32[128]", primals_749: "f32[512, 128]", primals_750: "f32[512]", primals_751: "f32[128, 512]", primals_752: "f32[128]", primals_753: "f32[512, 128]", primals_754: "f32[512]", primals_755: "f32[128, 512]", primals_756: "f32[128]", primals_757: "f32[128, 512]", primals_758: "f32[128]", primals_759: "f32[128, 128]", primals_760: "f32[128]", primals_761: "f32[128, 128]", primals_762: "f32[128]", primals_763: "f32[128, 512]", primals_764: "f32[128]", primals_765: "f32[128, 128]", primals_766: "f32[128]", primals_767: "f32[512, 128]", primals_768: "f32[512]", primals_769: "f32[128, 512]", primals_770: "f32[128]", primals_771: "f32[512, 128]", primals_772: "f32[512]", primals_773: "f32[128, 512]", primals_774: "f32[128]", primals_775: "f32[512, 128]", primals_776: "f32[512]", primals_777: "f32[128, 512]", primals_778: "f32[128]", primals_779: "f32[512, 128]", primals_780: "f32[512]", primals_781: "f32[128, 512]", primals_782: "f32[128]", primals_783: "f32[512, 128]", primals_784: "f32[512]", primals_785: "f32[128, 512]", primals_786: "f32[128]", primals_787: "f32[128, 512]", primals_788: "f32[128]", primals_789: "f32[128, 128]", primals_790: "f32[128]", primals_791: "f32[128, 128]", primals_792: "f32[128]", primals_793: "f32[128, 512]", primals_794: "f32[128]", primals_795: "f32[128, 128]", primals_796: "f32[128]", primals_797: "f32[512, 128]", primals_798: "f32[512]", primals_799: "f32[128, 512]", primals_800: "f32[128]", primals_801: "f32[512, 128]", primals_802: "f32[512]", primals_803: "f32[128, 512]", primals_804: "f32[128]", primals_805: "f32[512, 128]", primals_806: "f32[512]", primals_807: "f32[128, 512]", primals_808: "f32[128]", primals_809: "f32[512, 128]", primals_810: "f32[512]", primals_811: "f32[128, 512]", primals_812: "f32[128]", primals_813: "f32[512, 128]", primals_814: "f32[512]", primals_815: "f32[128, 512]", primals_816: "f32[128]", primals_817: "f32[128, 512]", primals_818: "f32[128]", primals_819: "f32[128, 128]", primals_820: "f32[128]", primals_821: "f32[128, 128]", primals_822: "f32[128]", primals_823: "f32[128, 512]", primals_824: "f32[128]", primals_825: "f32[128, 128]", primals_826: "f32[128]", primals_827: "f32[512, 128]", primals_828: "f32[512]", primals_829: "f32[128, 512]", primals_830: "f32[128]", primals_831: "f32[512, 128]", primals_832: "f32[512]", primals_833: "f32[128, 512]", primals_834: "f32[128]", primals_835: "f32[512, 128]", primals_836: "f32[512]", primals_837: "f32[128, 512]", primals_838: "f32[128]", primals_839: "f32[512, 128]", primals_840: "f32[512]", primals_841: "f32[128, 512]", primals_842: "f32[128]", primals_843: "f32[512, 128]", primals_844: "f32[512]", primals_845: "f32[128, 512]", primals_846: "f32[128]", primals_847: "f32[128, 512]", primals_848: "f32[128]", primals_849: "f32[128, 128]", primals_850: "f32[128]", primals_851: "f32[128, 128]", primals_852: "f32[128]", primals_853: "f32[128, 512]", primals_854: "f32[128]", primals_855: "f32[128, 128]", primals_856: "f32[128]", primals_857: "f32[512, 128]", primals_858: "f32[512]", primals_859: "f32[128, 512]", primals_860: "f32[128]", primals_861: "f32[512, 128]", primals_862: "f32[512]", primals_863: "f32[128, 512]", primals_864: "f32[128]", primals_865: "f32[512, 128]", primals_866: "f32[512]", primals_867: "f32[128, 512]", primals_868: "f32[128]", primals_869: "f32[512, 128]", primals_870: "f32[512]", primals_871: "f32[128, 512]", primals_872: "f32[128]", primals_873: "f32[512, 128]", primals_874: "f32[512]", primals_875: "f32[128, 512]", primals_876: "f32[128]", primals_877: "f32[128, 512]", primals_878: "f32[128]", primals_879: "f32[128, 128]", primals_880: "f32[128]", primals_881: "f32[128, 128]", primals_882: "f32[128]", primals_883: "f32[128, 512]", primals_884: "f32[128]", primals_885: "f32[128, 128]", primals_886: "f32[128]", primals_887: "f32[512, 128]", primals_888: "f32[512]", primals_889: "f32[128, 512]", primals_890: "f32[128]", primals_891: "f32[512, 128]", primals_892: "f32[512]", primals_893: "f32[128, 512]", primals_894: "f32[128]", primals_895: "f32[512, 128]", primals_896: "f32[512]", primals_897: "f32[128, 512]", primals_898: "f32[128]", primals_899: "f32[512, 128]", primals_900: "f32[512]", primals_901: "f32[128, 512]", primals_902: "f32[128]", primals_903: "f32[512, 128]", primals_904: "f32[512]", primals_905: "f32[128, 512]", primals_906: "f32[128]", primals_907: "f32[128, 512]", primals_908: "f32[128]", primals_909: "f32[128, 128]", primals_910: "f32[128]", primals_911: "f32[128, 128]", primals_912: "f32[128]", primals_913: "f32[128, 512]", primals_914: "f32[128]", primals_915: "f32[128, 128]", primals_916: "f32[128]", primals_917: "f32[512, 128]", primals_918: "f32[512]", primals_919: "f32[128, 512]", primals_920: "f32[128]", primals_921: "f32[512, 128]", primals_922: "f32[512]", primals_923: "f32[128, 512]", primals_924: "f32[128]", primals_925: "f32[512, 128]", primals_926: "f32[512]", primals_927: "f32[128, 512]", primals_928: "f32[128]", primals_929: "f32[512, 128]", primals_930: "f32[512]", primals_931: "f32[128, 512]", primals_932: "f32[128]", primals_933: "f32[512, 128]", primals_934: "f32[512]", primals_935: "f32[128, 512]", primals_936: "f32[128]", primals_937: "f32[128, 512]", primals_938: "f32[128]", primals_939: "f32[128, 128]", primals_940: "f32[128]", primals_941: "f32[128, 128]", primals_942: "f32[128]", primals_943: "f32[128, 512]", primals_944: "f32[128]", primals_945: "f32[128, 128]", primals_946: "f32[128]", primals_947: "f32[512, 128]", primals_948: "f32[512]", primals_949: "f32[128, 512]", primals_950: "f32[128]", primals_951: "f32[512, 128]", primals_952: "f32[512]", primals_953: "f32[128, 512]", primals_954: "f32[128]", primals_955: "f32[512, 128]", primals_956: "f32[512]", primals_957: "f32[128, 512]", primals_958: "f32[128]", primals_959: "f32[512, 128]", primals_960: "f32[512]", primals_961: "f32[128, 512]", primals_962: "f32[128]", primals_963: "f32[512, 128]", primals_964: "f32[512]", primals_965: "f32[128, 512]", primals_966: "f32[128]", primals_967: "f32[128, 512]", primals_968: "f32[128]", primals_969: "f32[128, 128]", primals_970: "f32[128]", primals_971: "f32[128, 128]", primals_972: "f32[128]", primals_973: "f32[128, 512]", primals_974: "f32[128]", primals_975: "f32[128, 128]", primals_976: "f32[128]", primals_977: "f32[512, 128]", primals_978: "f32[512]", primals_979: "f32[128, 512]", primals_980: "f32[128]", primals_981: "f32[512, 128]", primals_982: "f32[512]", primals_983: "f32[128, 512]", primals_984: "f32[128]", primals_985: "f32[512, 128]", primals_986: "f32[512]", primals_987: "f32[128, 512]", primals_988: "f32[128]", primals_989: "f32[512, 128]", primals_990: "f32[512]", primals_991: "f32[128, 512]", primals_992: "f32[128]", primals_993: "f32[512, 128]", primals_994: "f32[512]", primals_995: "f32[128, 512]", primals_996: "f32[128]", primals_997: "f32[128, 512]", primals_998: "f32[128]", primals_999: "f32[128, 128]", primals_1000: "f32[128]", primals_1001: "f32[128, 128]", primals_1002: "f32[128]", primals_1003: "f32[128, 512]", primals_1004: "f32[128]", primals_1005: "f32[128, 128]", primals_1006: "f32[128]", primals_1007: "f32[512, 128]", primals_1008: "f32[512]", primals_1009: "f32[128, 512]", primals_1010: "f32[128]", primals_1011: "f32[512, 128]", primals_1012: "f32[512]", primals_1013: "f32[128, 512]", primals_1014: "f32[128]", primals_1015: "f32[512, 128]", primals_1016: "f32[512]", primals_1017: "f32[128, 512]", primals_1018: "f32[128]", primals_1019: "f32[512, 128]", primals_1020: "f32[512]", primals_1021: "f32[128, 512]", primals_1022: "f32[128]", primals_1023: "f32[512, 128]", primals_1024: "f32[512]", primals_1025: "f32[128, 512]", primals_1026: "f32[128]", primals_1027: "f32[128, 512]", primals_1028: "f32[128]", primals_1029: "f32[128, 128]", primals_1030: "f32[128]", primals_1031: "f32[128, 128]", primals_1032: "f32[128]", primals_1033: "f32[128, 512]", primals_1034: "f32[128]", primals_1035: "f32[128, 128]", primals_1036: "f32[128]", primals_1037: "f32[512, 128]", primals_1038: "f32[512]", primals_1039: "f32[128, 512]", primals_1040: "f32[128]", primals_1041: "f32[512, 128]", primals_1042: "f32[512]", primals_1043: "f32[128, 512]", primals_1044: "f32[128]", primals_1045: "f32[512, 128]", primals_1046: "f32[512]", primals_1047: "f32[128, 512]", primals_1048: "f32[128]", primals_1049: "f32[512, 128]", primals_1050: "f32[512]", primals_1051: "f32[128, 512]", primals_1052: "f32[128]", primals_1053: "f32[512, 128]", primals_1054: "f32[512]", primals_1055: "f32[128, 512]", primals_1056: "f32[128]", primals_1057: "f32[128, 512]", primals_1058: "f32[128]", primals_1059: "f32[128, 128]", primals_1060: "f32[128]", primals_1061: "f32[128, 128]", primals_1062: "f32[128]", primals_1063: "f32[128, 512]", primals_1064: "f32[128]", primals_1065: "f32[128, 128]", primals_1066: "f32[128]", primals_1067: "f32[512, 128]", primals_1068: "f32[512]", primals_1069: "f32[128, 512]", primals_1070: "f32[128]", primals_1071: "f32[512, 128]", primals_1072: "f32[512]", primals_1073: "f32[128, 512]", primals_1074: "f32[128]", primals_1075: "f32[512, 128]", primals_1076: "f32[512]", primals_1077: "f32[128, 512]", primals_1078: "f32[128]", primals_1079: "f32[512, 128]", primals_1080: "f32[512]", primals_1081: "f32[128, 512]", primals_1082: "f32[128]", primals_1083: "f32[512, 128]", primals_1084: "f32[512]", primals_1085: "f32[128, 512]", primals_1086: "f32[128]", primals_1087: "f32[128, 512]", primals_1088: "f32[128]", primals_1089: "f32[128, 128]", primals_1090: "f32[128]", primals_1091: "f32[128, 128]", primals_1092: "f32[128]", primals_1093: "f32[128, 512]", primals_1094: "f32[128]", primals_1095: "f32[128, 128]", primals_1096: "f32[128]", primals_1097: "f32[512, 128]", primals_1098: "f32[512]", primals_1099: "f32[128, 512]", primals_1100: "f32[128]", primals_1101: "f32[512, 128]", primals_1102: "f32[512]", primals_1103: "f32[128, 512]", primals_1104: "f32[128]", primals_1105: "f32[512, 128]", primals_1106: "f32[512]", primals_1107: "f32[128, 512]", primals_1108: "f32[128]", primals_1109: "f32[512, 128]", primals_1110: "f32[512]", primals_1111: "f32[128, 512]", primals_1112: "f32[128]", primals_1113: "f32[512, 128]", primals_1114: "f32[512]", primals_1115: "f32[512, 512]", primals_1116: "f32[512]", primals_1117: "f32[512]", primals_1118: "f32[512]", primals_1119: "i64[1, 512]", primals_1120: "i64[1, 128]", primals_1121: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:882, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    full_default: "i64[1, 128]" = torch.ops.aten.full.default([1, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:213, code: position_ids = self.position_ids[:, :seq_length]
    slice_3: "i64[1, 512]" = torch.ops.aten.slice.Tensor(primals_1119, 0, 0, 9223372036854775807);  primals_1119 = None
    slice_4: "i64[1, 128]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 128);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:218, code: inputs_embeds = self.word_embeddings(input_ids)
    embedding: "f32[1, 128, 128]" = torch.ops.aten.embedding.default(primals_390, primals_1120, 0);  primals_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:230, code: nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0),
    slice_5: "f32[1, 128, 128]" = torch.ops.aten.slice.Tensor(embedding, 0, 0, 9223372036854775807)
    slice_6: "f32[1, 127, 128]" = torch.ops.aten.slice.Tensor(slice_5, 1, 1, 9223372036854775807)
    constant_pad_nd: "f32[1, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_6, [0, 0, 0, 1, 0, 0], 0.0);  slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:232, code: nn.functional.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0.0),
    slice_8: "f32[1, 127, 128]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, -1);  slice_5 = None
    constant_pad_nd_1: "f32[1, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_8, [0, 0, 1, 0, 0, 0], 0.0);  slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:228, code: inputs_embeds = torch.cat(
    cat: "f32[1, 128, 384]" = torch.ops.aten.cat.default([constant_pad_nd, embedding, constant_pad_nd_1], 2);  constant_pad_nd = embedding = constant_pad_nd_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:237, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
    view: "f32[128, 384]" = torch.ops.aten.view.default(cat, [128, 384]);  cat = None
    permute: "f32[384, 512]" = torch.ops.aten.permute.default(primals_391, [1, 0]);  primals_391 = None
    addmm: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_392, view, permute);  primals_392 = None
    view_1: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm, [1, 128, 512]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:241, code: position_embeddings = self.position_embeddings(position_ids)
    embedding_1: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(primals_393, slice_4);  primals_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:242, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    embedding_2: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(primals_394, full_default);  primals_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:243, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    add: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(view_1, embedding_1);  view_1 = embedding_1 = None
    add_1: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_1: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, primals_1)
    add_2: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  mul_1 = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:245, code: embeddings = self.dropout(embeddings)
    clone: "f32[1, 128, 512]" = torch.ops.aten.clone.default(add_2);  add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_2: "f32[128, 512]" = torch.ops.aten.view.default(clone, [128, 512])
    permute_1: "f32[512, 128]" = torch.ops.aten.permute.default(primals_395, [1, 0]);  primals_395 = None
    addmm_1: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_396, view_2, permute_1);  primals_396 = None
    view_3: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_1, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_2: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_3, primals_3);  view_3 = None
    add_3: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_2, primals_4);  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_2: "f32[512, 128]" = torch.ops.aten.permute.default(primals_397, [1, 0]);  primals_397 = None
    addmm_2: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_398, view_2, permute_2);  primals_398 = None
    view_5: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_2, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_3: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_5, primals_5);  view_5 = None
    add_4: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_3, primals_6);  mul_3 = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_6: "f32[128, 128]" = torch.ops.aten.view.default(add_4, [128, 128]);  add_4 = None
    permute_3: "f32[128, 128]" = torch.ops.aten.permute.default(primals_399, [1, 0]);  primals_399 = None
    addmm_3: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_400, view_6, permute_3);  primals_400 = None
    view_7: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_3, [1, 128, 128]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_4: "f32[128, 128]" = torch.ops.aten.permute.default(primals_401, [1, 0]);  primals_401 = None
    addmm_4: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_402, view_6, permute_4);  primals_402 = None
    view_9: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_4, [1, 128, 128]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_5: "f32[512, 128]" = torch.ops.aten.permute.default(primals_403, [1, 0]);  primals_403 = None
    addmm_5: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_404, view_2, permute_5);  primals_404 = None
    view_11: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_5, [1, 128, 128]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_12: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_7, [1, 128, 4, 32]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_6: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_13: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_9, [1, 128, 4, 32]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_7: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_14: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_11, [1, 128, 4, 32]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_8: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    
    # No stacktrace found for following nodes
    clone_default_69: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
    clone_default_70: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    clone_default_71: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    _scaled_dot_product_efficient_attention_default_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_69, clone_default_70, clone_default_71, None, True, 0.1, scale = 0.17677669529663687)
    getitem_211: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_23[0]
    getitem_212: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_23[1]
    getitem_213: "i64[]" = _scaled_dot_product_efficient_attention_default_23[2]
    getitem_214: "i64[]" = _scaled_dot_product_efficient_attention_default_23[3];  _scaled_dot_product_efficient_attention_default_23 = None
    alias_default_46: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_211)
    alias_default_47: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_46);  alias_default_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_10: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_211, [0, 2, 1, 3]);  getitem_211 = None
    clone_1: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_21: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_1, [1, 128, 128]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_22: "f32[128, 128]" = torch.ops.aten.view.default(view_21, [128, 128]);  view_21 = None
    permute_11: "f32[128, 128]" = torch.ops.aten.permute.default(primals_405, [1, 0]);  primals_405 = None
    addmm_6: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_406, view_22, permute_11);  primals_406 = None
    view_23: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_6, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_6: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_23, add_3);  view_23 = add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_4: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_6, primals_7);  add_6 = None
    add_7: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_4, primals_8);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_24: "f32[128, 128]" = torch.ops.aten.view.default(add_7, [128, 128])
    permute_12: "f32[128, 512]" = torch.ops.aten.permute.default(primals_407, [1, 0]);  primals_407 = None
    addmm_7: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_408, view_24, permute_12);  primals_408 = None
    view_25: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_7, [1, 128, 512]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_25);  view_25 = None
    alias_1: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_26: "f32[128, 512]" = torch.ops.aten.view.default(relu, [128, 512]);  relu = None
    permute_13: "f32[512, 128]" = torch.ops.aten.permute.default(primals_409, [1, 0]);  primals_409 = None
    addmm_8: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_410, view_26, permute_13);  primals_410 = None
    view_27: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_8, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_8: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_27, add_7);  view_27 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_5: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_8, primals_9);  add_8 = None
    add_9: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_5, primals_10);  mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_28: "f32[128, 128]" = torch.ops.aten.view.default(add_9, [128, 128])
    permute_14: "f32[128, 512]" = torch.ops.aten.permute.default(primals_411, [1, 0]);  primals_411 = None
    addmm_9: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_412, view_28, permute_14);  primals_412 = None
    view_29: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_9, [1, 128, 512]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_1: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_29);  view_29 = None
    alias_2: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_30: "f32[128, 512]" = torch.ops.aten.view.default(relu_1, [128, 512]);  relu_1 = None
    permute_15: "f32[512, 128]" = torch.ops.aten.permute.default(primals_413, [1, 0]);  primals_413 = None
    addmm_10: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_414, view_30, permute_15);  primals_414 = None
    view_31: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_10, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_10: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_31, add_9);  view_31 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_6: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_10, primals_11);  add_10 = None
    add_11: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, primals_12);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_32: "f32[128, 128]" = torch.ops.aten.view.default(add_11, [128, 128])
    permute_16: "f32[128, 512]" = torch.ops.aten.permute.default(primals_415, [1, 0]);  primals_415 = None
    addmm_11: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_416, view_32, permute_16);  primals_416 = None
    view_33: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_11, [1, 128, 512]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_2: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_33);  view_33 = None
    alias_3: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_34: "f32[128, 512]" = torch.ops.aten.view.default(relu_2, [128, 512]);  relu_2 = None
    permute_17: "f32[512, 128]" = torch.ops.aten.permute.default(primals_417, [1, 0]);  primals_417 = None
    addmm_12: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_418, view_34, permute_17);  primals_418 = None
    view_35: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_12, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_12: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_35, add_11);  view_35 = add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_7: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_12, primals_13);  add_12 = None
    add_13: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_7, primals_14);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_36: "f32[128, 128]" = torch.ops.aten.view.default(add_13, [128, 128])
    permute_18: "f32[128, 512]" = torch.ops.aten.permute.default(primals_419, [1, 0]);  primals_419 = None
    addmm_13: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_420, view_36, permute_18);  primals_420 = None
    view_37: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_13, [1, 128, 512]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_3: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_37);  view_37 = None
    alias_4: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_38: "f32[128, 512]" = torch.ops.aten.view.default(relu_3, [128, 512]);  relu_3 = None
    permute_19: "f32[512, 128]" = torch.ops.aten.permute.default(primals_421, [1, 0]);  primals_421 = None
    addmm_14: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_422, view_38, permute_19);  primals_422 = None
    view_39: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_14, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_14: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_39, add_13);  view_39 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_8: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_14, primals_15);  add_14 = None
    add_15: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_8, primals_16);  mul_8 = primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_40: "f32[128, 128]" = torch.ops.aten.view.default(add_15, [128, 128]);  add_15 = None
    permute_20: "f32[128, 512]" = torch.ops.aten.permute.default(primals_423, [1, 0]);  primals_423 = None
    addmm_15: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_424, view_40, permute_20);  primals_424 = None
    view_41: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_15, [1, 128, 512]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_2: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_41);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_16: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_2, clone);  clone_2 = clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_9: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_16, primals_17)
    add_17: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_9, primals_18);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_42: "f32[128, 512]" = torch.ops.aten.view.default(add_17, [128, 512])
    permute_21: "f32[512, 128]" = torch.ops.aten.permute.default(primals_425, [1, 0]);  primals_425 = None
    addmm_16: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_426, view_42, permute_21);  primals_426 = None
    view_43: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_16, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_10: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_43, primals_19);  view_43 = None
    add_18: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_10, primals_20);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_22: "f32[512, 128]" = torch.ops.aten.permute.default(primals_427, [1, 0]);  primals_427 = None
    addmm_17: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_428, view_42, permute_22);  primals_428 = None
    view_45: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_17, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_11: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_45, primals_21);  view_45 = None
    add_19: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_11, primals_22);  mul_11 = primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_46: "f32[128, 128]" = torch.ops.aten.view.default(add_19, [128, 128]);  add_19 = None
    permute_23: "f32[128, 128]" = torch.ops.aten.permute.default(primals_429, [1, 0]);  primals_429 = None
    addmm_18: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_430, view_46, permute_23);  primals_430 = None
    view_47: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_18, [1, 128, 128]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_24: "f32[128, 128]" = torch.ops.aten.permute.default(primals_431, [1, 0]);  primals_431 = None
    addmm_19: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_432, view_46, permute_24);  primals_432 = None
    view_49: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_19, [1, 128, 128]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_25: "f32[512, 128]" = torch.ops.aten.permute.default(primals_433, [1, 0]);  primals_433 = None
    addmm_20: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_434, view_42, permute_25);  primals_434 = None
    view_51: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_20, [1, 128, 128]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_52: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_47, [1, 128, 4, 32]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_26: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_53: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_49, [1, 128, 4, 32]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_27: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_54: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_51, [1, 128, 4, 32]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_28: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # No stacktrace found for following nodes
    clone_default_66: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    clone_default_67: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    clone_default_68: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
    _scaled_dot_product_efficient_attention_default_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_66, clone_default_67, clone_default_68, None, True, 0.1, scale = 0.17677669529663687)
    getitem_204: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_22[0]
    getitem_205: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_22[1]
    getitem_206: "i64[]" = _scaled_dot_product_efficient_attention_default_22[2]
    getitem_207: "i64[]" = _scaled_dot_product_efficient_attention_default_22[3];  _scaled_dot_product_efficient_attention_default_22 = None
    alias_default_44: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_204)
    alias_default_45: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_44);  alias_default_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_30: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_204, [0, 2, 1, 3]);  getitem_204 = None
    clone_3: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_61: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_3, [1, 128, 128]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_62: "f32[128, 128]" = torch.ops.aten.view.default(view_61, [128, 128]);  view_61 = None
    permute_31: "f32[128, 128]" = torch.ops.aten.permute.default(primals_435, [1, 0]);  primals_435 = None
    addmm_21: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_436, view_62, permute_31);  primals_436 = None
    view_63: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_21, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_21: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_63, add_18);  view_63 = add_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_12: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_21, primals_23);  add_21 = None
    add_22: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_12, primals_24);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_64: "f32[128, 128]" = torch.ops.aten.view.default(add_22, [128, 128])
    permute_32: "f32[128, 512]" = torch.ops.aten.permute.default(primals_437, [1, 0]);  primals_437 = None
    addmm_22: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_438, view_64, permute_32);  primals_438 = None
    view_65: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_22, [1, 128, 512]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_4: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_65);  view_65 = None
    alias_6: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_66: "f32[128, 512]" = torch.ops.aten.view.default(relu_4, [128, 512]);  relu_4 = None
    permute_33: "f32[512, 128]" = torch.ops.aten.permute.default(primals_439, [1, 0]);  primals_439 = None
    addmm_23: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_440, view_66, permute_33);  primals_440 = None
    view_67: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_23, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_23: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_67, add_22);  view_67 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_13: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_23, primals_25);  add_23 = None
    add_24: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_13, primals_26);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_68: "f32[128, 128]" = torch.ops.aten.view.default(add_24, [128, 128])
    permute_34: "f32[128, 512]" = torch.ops.aten.permute.default(primals_441, [1, 0]);  primals_441 = None
    addmm_24: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_442, view_68, permute_34);  primals_442 = None
    view_69: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_24, [1, 128, 512]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_5: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_69);  view_69 = None
    alias_7: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_70: "f32[128, 512]" = torch.ops.aten.view.default(relu_5, [128, 512]);  relu_5 = None
    permute_35: "f32[512, 128]" = torch.ops.aten.permute.default(primals_443, [1, 0]);  primals_443 = None
    addmm_25: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_444, view_70, permute_35);  primals_444 = None
    view_71: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_25, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_25: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_71, add_24);  view_71 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_14: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_25, primals_27);  add_25 = None
    add_26: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, primals_28);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_72: "f32[128, 128]" = torch.ops.aten.view.default(add_26, [128, 128])
    permute_36: "f32[128, 512]" = torch.ops.aten.permute.default(primals_445, [1, 0]);  primals_445 = None
    addmm_26: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_446, view_72, permute_36);  primals_446 = None
    view_73: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_26, [1, 128, 512]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_6: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_73);  view_73 = None
    alias_8: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_74: "f32[128, 512]" = torch.ops.aten.view.default(relu_6, [128, 512]);  relu_6 = None
    permute_37: "f32[512, 128]" = torch.ops.aten.permute.default(primals_447, [1, 0]);  primals_447 = None
    addmm_27: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_448, view_74, permute_37);  primals_448 = None
    view_75: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_27, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_27: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_75, add_26);  view_75 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_15: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_27, primals_29);  add_27 = None
    add_28: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_15, primals_30);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_76: "f32[128, 128]" = torch.ops.aten.view.default(add_28, [128, 128])
    permute_38: "f32[128, 512]" = torch.ops.aten.permute.default(primals_449, [1, 0]);  primals_449 = None
    addmm_28: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_450, view_76, permute_38);  primals_450 = None
    view_77: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_28, [1, 128, 512]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_7: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_77);  view_77 = None
    alias_9: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_78: "f32[128, 512]" = torch.ops.aten.view.default(relu_7, [128, 512]);  relu_7 = None
    permute_39: "f32[512, 128]" = torch.ops.aten.permute.default(primals_451, [1, 0]);  primals_451 = None
    addmm_29: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_452, view_78, permute_39);  primals_452 = None
    view_79: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_29, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_29: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_79, add_28);  view_79 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_16: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_29, primals_31);  add_29 = None
    add_30: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_16, primals_32);  mul_16 = primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_80: "f32[128, 128]" = torch.ops.aten.view.default(add_30, [128, 128]);  add_30 = None
    permute_40: "f32[128, 512]" = torch.ops.aten.permute.default(primals_453, [1, 0]);  primals_453 = None
    addmm_30: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_454, view_80, permute_40);  primals_454 = None
    view_81: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_30, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_4: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_81);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_31: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_4, add_17);  clone_4 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_17: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_31, primals_33);  add_31 = None
    add_32: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_17, primals_34);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_82: "f32[128, 512]" = torch.ops.aten.view.default(add_32, [128, 512])
    permute_41: "f32[512, 128]" = torch.ops.aten.permute.default(primals_455, [1, 0]);  primals_455 = None
    addmm_31: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_456, view_82, permute_41);  primals_456 = None
    view_83: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_31, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_18: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_83, primals_35);  view_83 = None
    add_33: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_18, primals_36);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_42: "f32[512, 128]" = torch.ops.aten.permute.default(primals_457, [1, 0]);  primals_457 = None
    addmm_32: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_458, view_82, permute_42);  primals_458 = None
    view_85: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_32, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_19: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_85, primals_37);  view_85 = None
    add_34: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_19, primals_38);  mul_19 = primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_86: "f32[128, 128]" = torch.ops.aten.view.default(add_34, [128, 128]);  add_34 = None
    permute_43: "f32[128, 128]" = torch.ops.aten.permute.default(primals_459, [1, 0]);  primals_459 = None
    addmm_33: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_460, view_86, permute_43);  primals_460 = None
    view_87: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_33, [1, 128, 128]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_44: "f32[128, 128]" = torch.ops.aten.permute.default(primals_461, [1, 0]);  primals_461 = None
    addmm_34: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_462, view_86, permute_44);  primals_462 = None
    view_89: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_34, [1, 128, 128]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_45: "f32[512, 128]" = torch.ops.aten.permute.default(primals_463, [1, 0]);  primals_463 = None
    addmm_35: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_464, view_82, permute_45);  primals_464 = None
    view_91: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_35, [1, 128, 128]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_92: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_87, [1, 128, 4, 32]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_46: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_93: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_89, [1, 128, 4, 32]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_47: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_94: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_91, [1, 128, 4, 32]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_48: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # No stacktrace found for following nodes
    clone_default_63: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    clone_default_64: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    clone_default_65: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    _scaled_dot_product_efficient_attention_default_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_63, clone_default_64, clone_default_65, None, True, 0.1, scale = 0.17677669529663687)
    getitem_197: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_21[0]
    getitem_198: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_21[1]
    getitem_199: "i64[]" = _scaled_dot_product_efficient_attention_default_21[2]
    getitem_200: "i64[]" = _scaled_dot_product_efficient_attention_default_21[3];  _scaled_dot_product_efficient_attention_default_21 = None
    alias_default_42: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_197)
    alias_default_43: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_42);  alias_default_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_50: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_197, [0, 2, 1, 3]);  getitem_197 = None
    clone_5: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_101: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_5, [1, 128, 128]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_102: "f32[128, 128]" = torch.ops.aten.view.default(view_101, [128, 128]);  view_101 = None
    permute_51: "f32[128, 128]" = torch.ops.aten.permute.default(primals_465, [1, 0]);  primals_465 = None
    addmm_36: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_466, view_102, permute_51);  primals_466 = None
    view_103: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_36, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_36: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_103, add_33);  view_103 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_20: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_36, primals_39);  add_36 = None
    add_37: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_20, primals_40);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_104: "f32[128, 128]" = torch.ops.aten.view.default(add_37, [128, 128])
    permute_52: "f32[128, 512]" = torch.ops.aten.permute.default(primals_467, [1, 0]);  primals_467 = None
    addmm_37: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_468, view_104, permute_52);  primals_468 = None
    view_105: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_37, [1, 128, 512]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_8: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_105);  view_105 = None
    alias_11: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_106: "f32[128, 512]" = torch.ops.aten.view.default(relu_8, [128, 512]);  relu_8 = None
    permute_53: "f32[512, 128]" = torch.ops.aten.permute.default(primals_469, [1, 0]);  primals_469 = None
    addmm_38: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_470, view_106, permute_53);  primals_470 = None
    view_107: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_38, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_38: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_107, add_37);  view_107 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_21: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_38, primals_41);  add_38 = None
    add_39: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_21, primals_42);  mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_108: "f32[128, 128]" = torch.ops.aten.view.default(add_39, [128, 128])
    permute_54: "f32[128, 512]" = torch.ops.aten.permute.default(primals_471, [1, 0]);  primals_471 = None
    addmm_39: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_472, view_108, permute_54);  primals_472 = None
    view_109: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_39, [1, 128, 512]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_9: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_109);  view_109 = None
    alias_12: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_110: "f32[128, 512]" = torch.ops.aten.view.default(relu_9, [128, 512]);  relu_9 = None
    permute_55: "f32[512, 128]" = torch.ops.aten.permute.default(primals_473, [1, 0]);  primals_473 = None
    addmm_40: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_474, view_110, permute_55);  primals_474 = None
    view_111: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_40, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_40: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_111, add_39);  view_111 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_22: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_40, primals_43);  add_40 = None
    add_41: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_22, primals_44);  mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_112: "f32[128, 128]" = torch.ops.aten.view.default(add_41, [128, 128])
    permute_56: "f32[128, 512]" = torch.ops.aten.permute.default(primals_475, [1, 0]);  primals_475 = None
    addmm_41: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_476, view_112, permute_56);  primals_476 = None
    view_113: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_41, [1, 128, 512]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_10: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_113);  view_113 = None
    alias_13: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_114: "f32[128, 512]" = torch.ops.aten.view.default(relu_10, [128, 512]);  relu_10 = None
    permute_57: "f32[512, 128]" = torch.ops.aten.permute.default(primals_477, [1, 0]);  primals_477 = None
    addmm_42: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_478, view_114, permute_57);  primals_478 = None
    view_115: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_42, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_42: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_115, add_41);  view_115 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_23: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_42, primals_45);  add_42 = None
    add_43: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_23, primals_46);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_116: "f32[128, 128]" = torch.ops.aten.view.default(add_43, [128, 128])
    permute_58: "f32[128, 512]" = torch.ops.aten.permute.default(primals_479, [1, 0]);  primals_479 = None
    addmm_43: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_480, view_116, permute_58);  primals_480 = None
    view_117: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_43, [1, 128, 512]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_11: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_117);  view_117 = None
    alias_14: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_118: "f32[128, 512]" = torch.ops.aten.view.default(relu_11, [128, 512]);  relu_11 = None
    permute_59: "f32[512, 128]" = torch.ops.aten.permute.default(primals_481, [1, 0]);  primals_481 = None
    addmm_44: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_482, view_118, permute_59);  primals_482 = None
    view_119: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_44, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_44: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_119, add_43);  view_119 = add_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_24: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_44, primals_47);  add_44 = None
    add_45: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_24, primals_48);  mul_24 = primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_120: "f32[128, 128]" = torch.ops.aten.view.default(add_45, [128, 128]);  add_45 = None
    permute_60: "f32[128, 512]" = torch.ops.aten.permute.default(primals_483, [1, 0]);  primals_483 = None
    addmm_45: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_484, view_120, permute_60);  primals_484 = None
    view_121: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_45, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_6: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_121);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_46: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_6, add_32);  clone_6 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_25: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_46, primals_49);  add_46 = None
    add_47: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_25, primals_50);  mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_122: "f32[128, 512]" = torch.ops.aten.view.default(add_47, [128, 512])
    permute_61: "f32[512, 128]" = torch.ops.aten.permute.default(primals_485, [1, 0]);  primals_485 = None
    addmm_46: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_486, view_122, permute_61);  primals_486 = None
    view_123: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_46, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_26: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_123, primals_51);  view_123 = None
    add_48: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_26, primals_52);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_62: "f32[512, 128]" = torch.ops.aten.permute.default(primals_487, [1, 0]);  primals_487 = None
    addmm_47: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_488, view_122, permute_62);  primals_488 = None
    view_125: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_47, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_27: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_125, primals_53);  view_125 = None
    add_49: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_27, primals_54);  mul_27 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_126: "f32[128, 128]" = torch.ops.aten.view.default(add_49, [128, 128]);  add_49 = None
    permute_63: "f32[128, 128]" = torch.ops.aten.permute.default(primals_489, [1, 0]);  primals_489 = None
    addmm_48: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_490, view_126, permute_63);  primals_490 = None
    view_127: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_48, [1, 128, 128]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_64: "f32[128, 128]" = torch.ops.aten.permute.default(primals_491, [1, 0]);  primals_491 = None
    addmm_49: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_492, view_126, permute_64);  primals_492 = None
    view_129: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_49, [1, 128, 128]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_65: "f32[512, 128]" = torch.ops.aten.permute.default(primals_493, [1, 0]);  primals_493 = None
    addmm_50: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_494, view_122, permute_65);  primals_494 = None
    view_131: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_50, [1, 128, 128]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_132: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_127, [1, 128, 4, 32]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_66: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_133: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_129, [1, 128, 4, 32]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_67: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_134: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_131, [1, 128, 4, 32]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_68: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
    
    # No stacktrace found for following nodes
    clone_default_60: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    clone_default_61: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    clone_default_62: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    _scaled_dot_product_efficient_attention_default_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_60, clone_default_61, clone_default_62, None, True, 0.1, scale = 0.17677669529663687)
    getitem_190: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_20[0]
    getitem_191: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_20[1]
    getitem_192: "i64[]" = _scaled_dot_product_efficient_attention_default_20[2]
    getitem_193: "i64[]" = _scaled_dot_product_efficient_attention_default_20[3];  _scaled_dot_product_efficient_attention_default_20 = None
    alias_default_40: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_190)
    alias_default_41: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_40);  alias_default_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_70: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_190, [0, 2, 1, 3]);  getitem_190 = None
    clone_7: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_141: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_7, [1, 128, 128]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_142: "f32[128, 128]" = torch.ops.aten.view.default(view_141, [128, 128]);  view_141 = None
    permute_71: "f32[128, 128]" = torch.ops.aten.permute.default(primals_495, [1, 0]);  primals_495 = None
    addmm_51: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_496, view_142, permute_71);  primals_496 = None
    view_143: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_51, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_51: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_143, add_48);  view_143 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_28: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_51, primals_55);  add_51 = None
    add_52: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_28, primals_56);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_144: "f32[128, 128]" = torch.ops.aten.view.default(add_52, [128, 128])
    permute_72: "f32[128, 512]" = torch.ops.aten.permute.default(primals_497, [1, 0]);  primals_497 = None
    addmm_52: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_498, view_144, permute_72);  primals_498 = None
    view_145: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_52, [1, 128, 512]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_12: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_145);  view_145 = None
    alias_16: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_146: "f32[128, 512]" = torch.ops.aten.view.default(relu_12, [128, 512]);  relu_12 = None
    permute_73: "f32[512, 128]" = torch.ops.aten.permute.default(primals_499, [1, 0]);  primals_499 = None
    addmm_53: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_500, view_146, permute_73);  primals_500 = None
    view_147: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_53, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_53: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_147, add_52);  view_147 = add_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_29: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_53, primals_57);  add_53 = None
    add_54: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_29, primals_58);  mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_148: "f32[128, 128]" = torch.ops.aten.view.default(add_54, [128, 128])
    permute_74: "f32[128, 512]" = torch.ops.aten.permute.default(primals_501, [1, 0]);  primals_501 = None
    addmm_54: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_502, view_148, permute_74);  primals_502 = None
    view_149: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_54, [1, 128, 512]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_13: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_149);  view_149 = None
    alias_17: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_150: "f32[128, 512]" = torch.ops.aten.view.default(relu_13, [128, 512]);  relu_13 = None
    permute_75: "f32[512, 128]" = torch.ops.aten.permute.default(primals_503, [1, 0]);  primals_503 = None
    addmm_55: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_504, view_150, permute_75);  primals_504 = None
    view_151: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_55, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_55: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_151, add_54);  view_151 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_30: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_55, primals_59);  add_55 = None
    add_56: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_30, primals_60);  mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_152: "f32[128, 128]" = torch.ops.aten.view.default(add_56, [128, 128])
    permute_76: "f32[128, 512]" = torch.ops.aten.permute.default(primals_505, [1, 0]);  primals_505 = None
    addmm_56: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_506, view_152, permute_76);  primals_506 = None
    view_153: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_56, [1, 128, 512]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_14: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_153);  view_153 = None
    alias_18: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_154: "f32[128, 512]" = torch.ops.aten.view.default(relu_14, [128, 512]);  relu_14 = None
    permute_77: "f32[512, 128]" = torch.ops.aten.permute.default(primals_507, [1, 0]);  primals_507 = None
    addmm_57: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_508, view_154, permute_77);  primals_508 = None
    view_155: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_57, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_57: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_155, add_56);  view_155 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_31: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_57, primals_61);  add_57 = None
    add_58: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_31, primals_62);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_156: "f32[128, 128]" = torch.ops.aten.view.default(add_58, [128, 128])
    permute_78: "f32[128, 512]" = torch.ops.aten.permute.default(primals_509, [1, 0]);  primals_509 = None
    addmm_58: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_510, view_156, permute_78);  primals_510 = None
    view_157: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_58, [1, 128, 512]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_15: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_157);  view_157 = None
    alias_19: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_158: "f32[128, 512]" = torch.ops.aten.view.default(relu_15, [128, 512]);  relu_15 = None
    permute_79: "f32[512, 128]" = torch.ops.aten.permute.default(primals_511, [1, 0]);  primals_511 = None
    addmm_59: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_512, view_158, permute_79);  primals_512 = None
    view_159: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_59, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_59: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_159, add_58);  view_159 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_32: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_59, primals_63);  add_59 = None
    add_60: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_32, primals_64);  mul_32 = primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_160: "f32[128, 128]" = torch.ops.aten.view.default(add_60, [128, 128]);  add_60 = None
    permute_80: "f32[128, 512]" = torch.ops.aten.permute.default(primals_513, [1, 0]);  primals_513 = None
    addmm_60: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_514, view_160, permute_80);  primals_514 = None
    view_161: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_60, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_8: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_161);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_61: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_8, add_47);  clone_8 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_33: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_61, primals_65);  add_61 = None
    add_62: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_33, primals_66);  mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_162: "f32[128, 512]" = torch.ops.aten.view.default(add_62, [128, 512])
    permute_81: "f32[512, 128]" = torch.ops.aten.permute.default(primals_515, [1, 0]);  primals_515 = None
    addmm_61: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_516, view_162, permute_81);  primals_516 = None
    view_163: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_61, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_34: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_163, primals_67);  view_163 = None
    add_63: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_34, primals_68);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_82: "f32[512, 128]" = torch.ops.aten.permute.default(primals_517, [1, 0]);  primals_517 = None
    addmm_62: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_518, view_162, permute_82);  primals_518 = None
    view_165: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_62, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_35: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_165, primals_69);  view_165 = None
    add_64: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_35, primals_70);  mul_35 = primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_166: "f32[128, 128]" = torch.ops.aten.view.default(add_64, [128, 128]);  add_64 = None
    permute_83: "f32[128, 128]" = torch.ops.aten.permute.default(primals_519, [1, 0]);  primals_519 = None
    addmm_63: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_520, view_166, permute_83);  primals_520 = None
    view_167: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_63, [1, 128, 128]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_84: "f32[128, 128]" = torch.ops.aten.permute.default(primals_521, [1, 0]);  primals_521 = None
    addmm_64: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_522, view_166, permute_84);  primals_522 = None
    view_169: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_64, [1, 128, 128]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_85: "f32[512, 128]" = torch.ops.aten.permute.default(primals_523, [1, 0]);  primals_523 = None
    addmm_65: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_524, view_162, permute_85);  primals_524 = None
    view_171: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_65, [1, 128, 128]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_172: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_167, [1, 128, 4, 32]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_86: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_173: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_169, [1, 128, 4, 32]);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_87: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_174: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_171, [1, 128, 4, 32]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_88: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
    
    # No stacktrace found for following nodes
    clone_default_57: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
    clone_default_58: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
    clone_default_59: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
    _scaled_dot_product_efficient_attention_default_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_57, clone_default_58, clone_default_59, None, True, 0.1, scale = 0.17677669529663687)
    getitem_183: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_19[0]
    getitem_184: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_19[1]
    getitem_185: "i64[]" = _scaled_dot_product_efficient_attention_default_19[2]
    getitem_186: "i64[]" = _scaled_dot_product_efficient_attention_default_19[3];  _scaled_dot_product_efficient_attention_default_19 = None
    alias_default_38: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_183)
    alias_default_39: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_38);  alias_default_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_90: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_183, [0, 2, 1, 3]);  getitem_183 = None
    clone_9: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_181: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_9, [1, 128, 128]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_182: "f32[128, 128]" = torch.ops.aten.view.default(view_181, [128, 128]);  view_181 = None
    permute_91: "f32[128, 128]" = torch.ops.aten.permute.default(primals_525, [1, 0]);  primals_525 = None
    addmm_66: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_526, view_182, permute_91);  primals_526 = None
    view_183: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_66, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_66: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_183, add_63);  view_183 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_36: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_66, primals_71);  add_66 = None
    add_67: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_36, primals_72);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_184: "f32[128, 128]" = torch.ops.aten.view.default(add_67, [128, 128])
    permute_92: "f32[128, 512]" = torch.ops.aten.permute.default(primals_527, [1, 0]);  primals_527 = None
    addmm_67: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_528, view_184, permute_92);  primals_528 = None
    view_185: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_67, [1, 128, 512]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_16: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_185);  view_185 = None
    alias_21: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_186: "f32[128, 512]" = torch.ops.aten.view.default(relu_16, [128, 512]);  relu_16 = None
    permute_93: "f32[512, 128]" = torch.ops.aten.permute.default(primals_529, [1, 0]);  primals_529 = None
    addmm_68: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_530, view_186, permute_93);  primals_530 = None
    view_187: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_68, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_68: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_187, add_67);  view_187 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_37: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_68, primals_73);  add_68 = None
    add_69: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_37, primals_74);  mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_188: "f32[128, 128]" = torch.ops.aten.view.default(add_69, [128, 128])
    permute_94: "f32[128, 512]" = torch.ops.aten.permute.default(primals_531, [1, 0]);  primals_531 = None
    addmm_69: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_532, view_188, permute_94);  primals_532 = None
    view_189: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_69, [1, 128, 512]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_17: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_189);  view_189 = None
    alias_22: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_190: "f32[128, 512]" = torch.ops.aten.view.default(relu_17, [128, 512]);  relu_17 = None
    permute_95: "f32[512, 128]" = torch.ops.aten.permute.default(primals_533, [1, 0]);  primals_533 = None
    addmm_70: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_534, view_190, permute_95);  primals_534 = None
    view_191: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_70, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_70: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_191, add_69);  view_191 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_38: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_70, primals_75);  add_70 = None
    add_71: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_38, primals_76);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_192: "f32[128, 128]" = torch.ops.aten.view.default(add_71, [128, 128])
    permute_96: "f32[128, 512]" = torch.ops.aten.permute.default(primals_535, [1, 0]);  primals_535 = None
    addmm_71: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_536, view_192, permute_96);  primals_536 = None
    view_193: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_71, [1, 128, 512]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_18: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_193);  view_193 = None
    alias_23: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_194: "f32[128, 512]" = torch.ops.aten.view.default(relu_18, [128, 512]);  relu_18 = None
    permute_97: "f32[512, 128]" = torch.ops.aten.permute.default(primals_537, [1, 0]);  primals_537 = None
    addmm_72: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_538, view_194, permute_97);  primals_538 = None
    view_195: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_72, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_72: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_195, add_71);  view_195 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_39: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_72, primals_77);  add_72 = None
    add_73: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_39, primals_78);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_196: "f32[128, 128]" = torch.ops.aten.view.default(add_73, [128, 128])
    permute_98: "f32[128, 512]" = torch.ops.aten.permute.default(primals_539, [1, 0]);  primals_539 = None
    addmm_73: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_540, view_196, permute_98);  primals_540 = None
    view_197: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_73, [1, 128, 512]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_19: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_197);  view_197 = None
    alias_24: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_198: "f32[128, 512]" = torch.ops.aten.view.default(relu_19, [128, 512]);  relu_19 = None
    permute_99: "f32[512, 128]" = torch.ops.aten.permute.default(primals_541, [1, 0]);  primals_541 = None
    addmm_74: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_542, view_198, permute_99);  primals_542 = None
    view_199: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_74, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_74: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_199, add_73);  view_199 = add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_40: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_74, primals_79);  add_74 = None
    add_75: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_40, primals_80);  mul_40 = primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_200: "f32[128, 128]" = torch.ops.aten.view.default(add_75, [128, 128]);  add_75 = None
    permute_100: "f32[128, 512]" = torch.ops.aten.permute.default(primals_543, [1, 0]);  primals_543 = None
    addmm_75: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_544, view_200, permute_100);  primals_544 = None
    view_201: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_75, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_10: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_201);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_76: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_10, add_62);  clone_10 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_41: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_76, primals_81);  add_76 = None
    add_77: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_41, primals_82);  mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_202: "f32[128, 512]" = torch.ops.aten.view.default(add_77, [128, 512])
    permute_101: "f32[512, 128]" = torch.ops.aten.permute.default(primals_545, [1, 0]);  primals_545 = None
    addmm_76: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_546, view_202, permute_101);  primals_546 = None
    view_203: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_76, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_42: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_203, primals_83);  view_203 = None
    add_78: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_42, primals_84);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_102: "f32[512, 128]" = torch.ops.aten.permute.default(primals_547, [1, 0]);  primals_547 = None
    addmm_77: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_548, view_202, permute_102);  primals_548 = None
    view_205: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_77, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_43: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_205, primals_85);  view_205 = None
    add_79: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_43, primals_86);  mul_43 = primals_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_206: "f32[128, 128]" = torch.ops.aten.view.default(add_79, [128, 128]);  add_79 = None
    permute_103: "f32[128, 128]" = torch.ops.aten.permute.default(primals_549, [1, 0]);  primals_549 = None
    addmm_78: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_550, view_206, permute_103);  primals_550 = None
    view_207: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_78, [1, 128, 128]);  addmm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_104: "f32[128, 128]" = torch.ops.aten.permute.default(primals_551, [1, 0]);  primals_551 = None
    addmm_79: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_552, view_206, permute_104);  primals_552 = None
    view_209: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_79, [1, 128, 128]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_105: "f32[512, 128]" = torch.ops.aten.permute.default(primals_553, [1, 0]);  primals_553 = None
    addmm_80: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_554, view_202, permute_105);  primals_554 = None
    view_211: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_80, [1, 128, 128]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_212: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_207, [1, 128, 4, 32]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_106: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_213: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_209, [1, 128, 4, 32]);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_107: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_214: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_211, [1, 128, 4, 32]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_108: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    
    # No stacktrace found for following nodes
    clone_default_54: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    clone_default_55: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
    clone_default_56: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_108, memory_format = torch.contiguous_format);  permute_108 = None
    _scaled_dot_product_efficient_attention_default_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_54, clone_default_55, clone_default_56, None, True, 0.1, scale = 0.17677669529663687)
    getitem_176: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_18[0]
    getitem_177: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_18[1]
    getitem_178: "i64[]" = _scaled_dot_product_efficient_attention_default_18[2]
    getitem_179: "i64[]" = _scaled_dot_product_efficient_attention_default_18[3];  _scaled_dot_product_efficient_attention_default_18 = None
    alias_default_36: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_176)
    alias_default_37: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_36);  alias_default_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_110: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_176, [0, 2, 1, 3]);  getitem_176 = None
    clone_11: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_221: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_11, [1, 128, 128]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_222: "f32[128, 128]" = torch.ops.aten.view.default(view_221, [128, 128]);  view_221 = None
    permute_111: "f32[128, 128]" = torch.ops.aten.permute.default(primals_555, [1, 0]);  primals_555 = None
    addmm_81: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_556, view_222, permute_111);  primals_556 = None
    view_223: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_81, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_81: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_223, add_78);  view_223 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_44: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_81, primals_87);  add_81 = None
    add_82: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_44, primals_88);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_224: "f32[128, 128]" = torch.ops.aten.view.default(add_82, [128, 128])
    permute_112: "f32[128, 512]" = torch.ops.aten.permute.default(primals_557, [1, 0]);  primals_557 = None
    addmm_82: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_558, view_224, permute_112);  primals_558 = None
    view_225: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_82, [1, 128, 512]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_20: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_225);  view_225 = None
    alias_26: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_226: "f32[128, 512]" = torch.ops.aten.view.default(relu_20, [128, 512]);  relu_20 = None
    permute_113: "f32[512, 128]" = torch.ops.aten.permute.default(primals_559, [1, 0]);  primals_559 = None
    addmm_83: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_560, view_226, permute_113);  primals_560 = None
    view_227: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_83, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_83: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_227, add_82);  view_227 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_45: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_83, primals_89);  add_83 = None
    add_84: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_45, primals_90);  mul_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_228: "f32[128, 128]" = torch.ops.aten.view.default(add_84, [128, 128])
    permute_114: "f32[128, 512]" = torch.ops.aten.permute.default(primals_561, [1, 0]);  primals_561 = None
    addmm_84: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_562, view_228, permute_114);  primals_562 = None
    view_229: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_84, [1, 128, 512]);  addmm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_21: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_229);  view_229 = None
    alias_27: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_230: "f32[128, 512]" = torch.ops.aten.view.default(relu_21, [128, 512]);  relu_21 = None
    permute_115: "f32[512, 128]" = torch.ops.aten.permute.default(primals_563, [1, 0]);  primals_563 = None
    addmm_85: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_564, view_230, permute_115);  primals_564 = None
    view_231: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_85, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_85: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_231, add_84);  view_231 = add_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_46: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_85, primals_91);  add_85 = None
    add_86: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_46, primals_92);  mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_232: "f32[128, 128]" = torch.ops.aten.view.default(add_86, [128, 128])
    permute_116: "f32[128, 512]" = torch.ops.aten.permute.default(primals_565, [1, 0]);  primals_565 = None
    addmm_86: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_566, view_232, permute_116);  primals_566 = None
    view_233: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_86, [1, 128, 512]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_22: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_233);  view_233 = None
    alias_28: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_234: "f32[128, 512]" = torch.ops.aten.view.default(relu_22, [128, 512]);  relu_22 = None
    permute_117: "f32[512, 128]" = torch.ops.aten.permute.default(primals_567, [1, 0]);  primals_567 = None
    addmm_87: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_568, view_234, permute_117);  primals_568 = None
    view_235: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_87, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_87: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_235, add_86);  view_235 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_47: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_87, primals_93);  add_87 = None
    add_88: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_47, primals_94);  mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_236: "f32[128, 128]" = torch.ops.aten.view.default(add_88, [128, 128])
    permute_118: "f32[128, 512]" = torch.ops.aten.permute.default(primals_569, [1, 0]);  primals_569 = None
    addmm_88: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_570, view_236, permute_118);  primals_570 = None
    view_237: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_88, [1, 128, 512]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_23: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_237);  view_237 = None
    alias_29: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_238: "f32[128, 512]" = torch.ops.aten.view.default(relu_23, [128, 512]);  relu_23 = None
    permute_119: "f32[512, 128]" = torch.ops.aten.permute.default(primals_571, [1, 0]);  primals_571 = None
    addmm_89: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_572, view_238, permute_119);  primals_572 = None
    view_239: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_89, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_89: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_239, add_88);  view_239 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_48: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_89, primals_95);  add_89 = None
    add_90: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_48, primals_96);  mul_48 = primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_240: "f32[128, 128]" = torch.ops.aten.view.default(add_90, [128, 128]);  add_90 = None
    permute_120: "f32[128, 512]" = torch.ops.aten.permute.default(primals_573, [1, 0]);  primals_573 = None
    addmm_90: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_574, view_240, permute_120);  primals_574 = None
    view_241: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_90, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_12: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_241);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_91: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_12, add_77);  clone_12 = add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_49: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_91, primals_97);  add_91 = None
    add_92: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_49, primals_98);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_242: "f32[128, 512]" = torch.ops.aten.view.default(add_92, [128, 512])
    permute_121: "f32[512, 128]" = torch.ops.aten.permute.default(primals_575, [1, 0]);  primals_575 = None
    addmm_91: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_576, view_242, permute_121);  primals_576 = None
    view_243: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_91, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_50: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_243, primals_99);  view_243 = None
    add_93: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_50, primals_100);  mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_122: "f32[512, 128]" = torch.ops.aten.permute.default(primals_577, [1, 0]);  primals_577 = None
    addmm_92: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_578, view_242, permute_122);  primals_578 = None
    view_245: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_92, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_51: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_245, primals_101);  view_245 = None
    add_94: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_51, primals_102);  mul_51 = primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_246: "f32[128, 128]" = torch.ops.aten.view.default(add_94, [128, 128]);  add_94 = None
    permute_123: "f32[128, 128]" = torch.ops.aten.permute.default(primals_579, [1, 0]);  primals_579 = None
    addmm_93: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_580, view_246, permute_123);  primals_580 = None
    view_247: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_93, [1, 128, 128]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_124: "f32[128, 128]" = torch.ops.aten.permute.default(primals_581, [1, 0]);  primals_581 = None
    addmm_94: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_582, view_246, permute_124);  primals_582 = None
    view_249: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_94, [1, 128, 128]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_125: "f32[512, 128]" = torch.ops.aten.permute.default(primals_583, [1, 0]);  primals_583 = None
    addmm_95: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_584, view_242, permute_125);  primals_584 = None
    view_251: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_95, [1, 128, 128]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_252: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_247, [1, 128, 4, 32]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_126: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_253: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_249, [1, 128, 4, 32]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_127: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_254: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_251, [1, 128, 4, 32]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_128: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_254, [0, 2, 1, 3]);  view_254 = None
    
    # No stacktrace found for following nodes
    clone_default_51: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    clone_default_52: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    clone_default_53: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    _scaled_dot_product_efficient_attention_default_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_51, clone_default_52, clone_default_53, None, True, 0.1, scale = 0.17677669529663687)
    getitem_169: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_17[0]
    getitem_170: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_17[1]
    getitem_171: "i64[]" = _scaled_dot_product_efficient_attention_default_17[2]
    getitem_172: "i64[]" = _scaled_dot_product_efficient_attention_default_17[3];  _scaled_dot_product_efficient_attention_default_17 = None
    alias_default_34: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_169)
    alias_default_35: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_34);  alias_default_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_130: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_169, [0, 2, 1, 3]);  getitem_169 = None
    clone_13: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_261: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_13, [1, 128, 128]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_262: "f32[128, 128]" = torch.ops.aten.view.default(view_261, [128, 128]);  view_261 = None
    permute_131: "f32[128, 128]" = torch.ops.aten.permute.default(primals_585, [1, 0]);  primals_585 = None
    addmm_96: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_586, view_262, permute_131);  primals_586 = None
    view_263: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_96, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_96: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_263, add_93);  view_263 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_52: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_96, primals_103);  add_96 = None
    add_97: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_52, primals_104);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_264: "f32[128, 128]" = torch.ops.aten.view.default(add_97, [128, 128])
    permute_132: "f32[128, 512]" = torch.ops.aten.permute.default(primals_587, [1, 0]);  primals_587 = None
    addmm_97: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_588, view_264, permute_132);  primals_588 = None
    view_265: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_97, [1, 128, 512]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_24: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_265);  view_265 = None
    alias_31: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_266: "f32[128, 512]" = torch.ops.aten.view.default(relu_24, [128, 512]);  relu_24 = None
    permute_133: "f32[512, 128]" = torch.ops.aten.permute.default(primals_589, [1, 0]);  primals_589 = None
    addmm_98: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_590, view_266, permute_133);  primals_590 = None
    view_267: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_98, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_98: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_267, add_97);  view_267 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_53: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_98, primals_105);  add_98 = None
    add_99: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_53, primals_106);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_268: "f32[128, 128]" = torch.ops.aten.view.default(add_99, [128, 128])
    permute_134: "f32[128, 512]" = torch.ops.aten.permute.default(primals_591, [1, 0]);  primals_591 = None
    addmm_99: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_592, view_268, permute_134);  primals_592 = None
    view_269: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_99, [1, 128, 512]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_25: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_269);  view_269 = None
    alias_32: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_270: "f32[128, 512]" = torch.ops.aten.view.default(relu_25, [128, 512]);  relu_25 = None
    permute_135: "f32[512, 128]" = torch.ops.aten.permute.default(primals_593, [1, 0]);  primals_593 = None
    addmm_100: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_594, view_270, permute_135);  primals_594 = None
    view_271: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_100, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_100: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_271, add_99);  view_271 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_54: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_100, primals_107);  add_100 = None
    add_101: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_54, primals_108);  mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_272: "f32[128, 128]" = torch.ops.aten.view.default(add_101, [128, 128])
    permute_136: "f32[128, 512]" = torch.ops.aten.permute.default(primals_595, [1, 0]);  primals_595 = None
    addmm_101: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_596, view_272, permute_136);  primals_596 = None
    view_273: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_101, [1, 128, 512]);  addmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_26: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_273);  view_273 = None
    alias_33: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_274: "f32[128, 512]" = torch.ops.aten.view.default(relu_26, [128, 512]);  relu_26 = None
    permute_137: "f32[512, 128]" = torch.ops.aten.permute.default(primals_597, [1, 0]);  primals_597 = None
    addmm_102: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_598, view_274, permute_137);  primals_598 = None
    view_275: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_102, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_102: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_275, add_101);  view_275 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_55: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_102, primals_109);  add_102 = None
    add_103: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_55, primals_110);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_276: "f32[128, 128]" = torch.ops.aten.view.default(add_103, [128, 128])
    permute_138: "f32[128, 512]" = torch.ops.aten.permute.default(primals_599, [1, 0]);  primals_599 = None
    addmm_103: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_600, view_276, permute_138);  primals_600 = None
    view_277: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_103, [1, 128, 512]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_27: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_277);  view_277 = None
    alias_34: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_278: "f32[128, 512]" = torch.ops.aten.view.default(relu_27, [128, 512]);  relu_27 = None
    permute_139: "f32[512, 128]" = torch.ops.aten.permute.default(primals_601, [1, 0]);  primals_601 = None
    addmm_104: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_602, view_278, permute_139);  primals_602 = None
    view_279: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_104, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_104: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_279, add_103);  view_279 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_56: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_104, primals_111);  add_104 = None
    add_105: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_56, primals_112);  mul_56 = primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_280: "f32[128, 128]" = torch.ops.aten.view.default(add_105, [128, 128]);  add_105 = None
    permute_140: "f32[128, 512]" = torch.ops.aten.permute.default(primals_603, [1, 0]);  primals_603 = None
    addmm_105: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_604, view_280, permute_140);  primals_604 = None
    view_281: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_105, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_14: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_281);  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_106: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_14, add_92);  clone_14 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_57: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_106, primals_113);  add_106 = None
    add_107: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_57, primals_114);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_282: "f32[128, 512]" = torch.ops.aten.view.default(add_107, [128, 512])
    permute_141: "f32[512, 128]" = torch.ops.aten.permute.default(primals_605, [1, 0]);  primals_605 = None
    addmm_106: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_606, view_282, permute_141);  primals_606 = None
    view_283: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_106, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_58: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_283, primals_115);  view_283 = None
    add_108: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_58, primals_116);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_142: "f32[512, 128]" = torch.ops.aten.permute.default(primals_607, [1, 0]);  primals_607 = None
    addmm_107: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_608, view_282, permute_142);  primals_608 = None
    view_285: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_107, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_59: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_285, primals_117);  view_285 = None
    add_109: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_59, primals_118);  mul_59 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_286: "f32[128, 128]" = torch.ops.aten.view.default(add_109, [128, 128]);  add_109 = None
    permute_143: "f32[128, 128]" = torch.ops.aten.permute.default(primals_609, [1, 0]);  primals_609 = None
    addmm_108: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_610, view_286, permute_143);  primals_610 = None
    view_287: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_108, [1, 128, 128]);  addmm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_144: "f32[128, 128]" = torch.ops.aten.permute.default(primals_611, [1, 0]);  primals_611 = None
    addmm_109: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_612, view_286, permute_144);  primals_612 = None
    view_289: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_109, [1, 128, 128]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_145: "f32[512, 128]" = torch.ops.aten.permute.default(primals_613, [1, 0]);  primals_613 = None
    addmm_110: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_614, view_282, permute_145);  primals_614 = None
    view_291: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_110, [1, 128, 128]);  addmm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_292: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_287, [1, 128, 4, 32]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_146: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_293: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_289, [1, 128, 4, 32]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_147: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_294: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_291, [1, 128, 4, 32]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_148: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    
    # No stacktrace found for following nodes
    clone_default_48: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
    clone_default_49: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    clone_default_50: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    _scaled_dot_product_efficient_attention_default_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_48, clone_default_49, clone_default_50, None, True, 0.1, scale = 0.17677669529663687)
    getitem_162: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_16[0]
    getitem_163: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_16[1]
    getitem_164: "i64[]" = _scaled_dot_product_efficient_attention_default_16[2]
    getitem_165: "i64[]" = _scaled_dot_product_efficient_attention_default_16[3];  _scaled_dot_product_efficient_attention_default_16 = None
    alias_default_32: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_162)
    alias_default_33: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_32);  alias_default_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_150: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None
    clone_15: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_301: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_15, [1, 128, 128]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_302: "f32[128, 128]" = torch.ops.aten.view.default(view_301, [128, 128]);  view_301 = None
    permute_151: "f32[128, 128]" = torch.ops.aten.permute.default(primals_615, [1, 0]);  primals_615 = None
    addmm_111: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_616, view_302, permute_151);  primals_616 = None
    view_303: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_111, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_111: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_303, add_108);  view_303 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_60: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_111, primals_119);  add_111 = None
    add_112: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_60, primals_120);  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_304: "f32[128, 128]" = torch.ops.aten.view.default(add_112, [128, 128])
    permute_152: "f32[128, 512]" = torch.ops.aten.permute.default(primals_617, [1, 0]);  primals_617 = None
    addmm_112: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_618, view_304, permute_152);  primals_618 = None
    view_305: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_112, [1, 128, 512]);  addmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_28: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_305);  view_305 = None
    alias_36: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_306: "f32[128, 512]" = torch.ops.aten.view.default(relu_28, [128, 512]);  relu_28 = None
    permute_153: "f32[512, 128]" = torch.ops.aten.permute.default(primals_619, [1, 0]);  primals_619 = None
    addmm_113: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_620, view_306, permute_153);  primals_620 = None
    view_307: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_113, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_113: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_307, add_112);  view_307 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_61: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_113, primals_121);  add_113 = None
    add_114: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_61, primals_122);  mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_308: "f32[128, 128]" = torch.ops.aten.view.default(add_114, [128, 128])
    permute_154: "f32[128, 512]" = torch.ops.aten.permute.default(primals_621, [1, 0]);  primals_621 = None
    addmm_114: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_622, view_308, permute_154);  primals_622 = None
    view_309: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_114, [1, 128, 512]);  addmm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_29: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_309);  view_309 = None
    alias_37: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_310: "f32[128, 512]" = torch.ops.aten.view.default(relu_29, [128, 512]);  relu_29 = None
    permute_155: "f32[512, 128]" = torch.ops.aten.permute.default(primals_623, [1, 0]);  primals_623 = None
    addmm_115: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_624, view_310, permute_155);  primals_624 = None
    view_311: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_115, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_115: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_311, add_114);  view_311 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_62: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_115, primals_123);  add_115 = None
    add_116: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_62, primals_124);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_312: "f32[128, 128]" = torch.ops.aten.view.default(add_116, [128, 128])
    permute_156: "f32[128, 512]" = torch.ops.aten.permute.default(primals_625, [1, 0]);  primals_625 = None
    addmm_116: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_626, view_312, permute_156);  primals_626 = None
    view_313: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_116, [1, 128, 512]);  addmm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_30: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_313);  view_313 = None
    alias_38: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_314: "f32[128, 512]" = torch.ops.aten.view.default(relu_30, [128, 512]);  relu_30 = None
    permute_157: "f32[512, 128]" = torch.ops.aten.permute.default(primals_627, [1, 0]);  primals_627 = None
    addmm_117: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_628, view_314, permute_157);  primals_628 = None
    view_315: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_117, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_117: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_315, add_116);  view_315 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_63: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_117, primals_125);  add_117 = None
    add_118: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_63, primals_126);  mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_316: "f32[128, 128]" = torch.ops.aten.view.default(add_118, [128, 128])
    permute_158: "f32[128, 512]" = torch.ops.aten.permute.default(primals_629, [1, 0]);  primals_629 = None
    addmm_118: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_630, view_316, permute_158);  primals_630 = None
    view_317: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_118, [1, 128, 512]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_31: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_317);  view_317 = None
    alias_39: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_318: "f32[128, 512]" = torch.ops.aten.view.default(relu_31, [128, 512]);  relu_31 = None
    permute_159: "f32[512, 128]" = torch.ops.aten.permute.default(primals_631, [1, 0]);  primals_631 = None
    addmm_119: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_632, view_318, permute_159);  primals_632 = None
    view_319: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_119, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_119: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_319, add_118);  view_319 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_64: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_119, primals_127);  add_119 = None
    add_120: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_64, primals_128);  mul_64 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_320: "f32[128, 128]" = torch.ops.aten.view.default(add_120, [128, 128]);  add_120 = None
    permute_160: "f32[128, 512]" = torch.ops.aten.permute.default(primals_633, [1, 0]);  primals_633 = None
    addmm_120: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_634, view_320, permute_160);  primals_634 = None
    view_321: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_120, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_16: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_321);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_121: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_16, add_107);  clone_16 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_65: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_121, primals_129);  add_121 = None
    add_122: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_65, primals_130);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_322: "f32[128, 512]" = torch.ops.aten.view.default(add_122, [128, 512])
    permute_161: "f32[512, 128]" = torch.ops.aten.permute.default(primals_635, [1, 0]);  primals_635 = None
    addmm_121: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_636, view_322, permute_161);  primals_636 = None
    view_323: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_121, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_66: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_323, primals_131);  view_323 = None
    add_123: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_66, primals_132);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_162: "f32[512, 128]" = torch.ops.aten.permute.default(primals_637, [1, 0]);  primals_637 = None
    addmm_122: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_638, view_322, permute_162);  primals_638 = None
    view_325: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_122, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_67: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_325, primals_133);  view_325 = None
    add_124: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_67, primals_134);  mul_67 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_326: "f32[128, 128]" = torch.ops.aten.view.default(add_124, [128, 128]);  add_124 = None
    permute_163: "f32[128, 128]" = torch.ops.aten.permute.default(primals_639, [1, 0]);  primals_639 = None
    addmm_123: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_640, view_326, permute_163);  primals_640 = None
    view_327: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_123, [1, 128, 128]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_164: "f32[128, 128]" = torch.ops.aten.permute.default(primals_641, [1, 0]);  primals_641 = None
    addmm_124: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_642, view_326, permute_164);  primals_642 = None
    view_329: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_124, [1, 128, 128]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_165: "f32[512, 128]" = torch.ops.aten.permute.default(primals_643, [1, 0]);  primals_643 = None
    addmm_125: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_644, view_322, permute_165);  primals_644 = None
    view_331: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_125, [1, 128, 128]);  addmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_332: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_327, [1, 128, 4, 32]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_166: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_333: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_329, [1, 128, 4, 32]);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_167: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_333, [0, 2, 1, 3]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_334: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_331, [1, 128, 4, 32]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_168: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
    
    # No stacktrace found for following nodes
    clone_default_45: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
    clone_default_46: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
    clone_default_47: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    _scaled_dot_product_efficient_attention_default_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_45, clone_default_46, clone_default_47, None, True, 0.1, scale = 0.17677669529663687)
    getitem_155: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_15[0]
    getitem_156: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_15[1]
    getitem_157: "i64[]" = _scaled_dot_product_efficient_attention_default_15[2]
    getitem_158: "i64[]" = _scaled_dot_product_efficient_attention_default_15[3];  _scaled_dot_product_efficient_attention_default_15 = None
    alias_default_30: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_155)
    alias_default_31: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_30);  alias_default_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_170: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3]);  getitem_155 = None
    clone_17: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_341: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_17, [1, 128, 128]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_342: "f32[128, 128]" = torch.ops.aten.view.default(view_341, [128, 128]);  view_341 = None
    permute_171: "f32[128, 128]" = torch.ops.aten.permute.default(primals_645, [1, 0]);  primals_645 = None
    addmm_126: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_646, view_342, permute_171);  primals_646 = None
    view_343: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_126, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_126: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_343, add_123);  view_343 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_68: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_126, primals_135);  add_126 = None
    add_127: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_68, primals_136);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_344: "f32[128, 128]" = torch.ops.aten.view.default(add_127, [128, 128])
    permute_172: "f32[128, 512]" = torch.ops.aten.permute.default(primals_647, [1, 0]);  primals_647 = None
    addmm_127: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_648, view_344, permute_172);  primals_648 = None
    view_345: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_127, [1, 128, 512]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_32: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_345);  view_345 = None
    alias_41: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_346: "f32[128, 512]" = torch.ops.aten.view.default(relu_32, [128, 512]);  relu_32 = None
    permute_173: "f32[512, 128]" = torch.ops.aten.permute.default(primals_649, [1, 0]);  primals_649 = None
    addmm_128: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_650, view_346, permute_173);  primals_650 = None
    view_347: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_128, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_128: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_347, add_127);  view_347 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_69: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_128, primals_137);  add_128 = None
    add_129: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_69, primals_138);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_348: "f32[128, 128]" = torch.ops.aten.view.default(add_129, [128, 128])
    permute_174: "f32[128, 512]" = torch.ops.aten.permute.default(primals_651, [1, 0]);  primals_651 = None
    addmm_129: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_652, view_348, permute_174);  primals_652 = None
    view_349: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_129, [1, 128, 512]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_33: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_349);  view_349 = None
    alias_42: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_350: "f32[128, 512]" = torch.ops.aten.view.default(relu_33, [128, 512]);  relu_33 = None
    permute_175: "f32[512, 128]" = torch.ops.aten.permute.default(primals_653, [1, 0]);  primals_653 = None
    addmm_130: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_654, view_350, permute_175);  primals_654 = None
    view_351: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_130, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_130: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_351, add_129);  view_351 = add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_70: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_130, primals_139);  add_130 = None
    add_131: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_70, primals_140);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_352: "f32[128, 128]" = torch.ops.aten.view.default(add_131, [128, 128])
    permute_176: "f32[128, 512]" = torch.ops.aten.permute.default(primals_655, [1, 0]);  primals_655 = None
    addmm_131: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_656, view_352, permute_176);  primals_656 = None
    view_353: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_131, [1, 128, 512]);  addmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_34: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_353);  view_353 = None
    alias_43: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_354: "f32[128, 512]" = torch.ops.aten.view.default(relu_34, [128, 512]);  relu_34 = None
    permute_177: "f32[512, 128]" = torch.ops.aten.permute.default(primals_657, [1, 0]);  primals_657 = None
    addmm_132: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_658, view_354, permute_177);  primals_658 = None
    view_355: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_132, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_132: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_355, add_131);  view_355 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_71: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_132, primals_141);  add_132 = None
    add_133: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_71, primals_142);  mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_356: "f32[128, 128]" = torch.ops.aten.view.default(add_133, [128, 128])
    permute_178: "f32[128, 512]" = torch.ops.aten.permute.default(primals_659, [1, 0]);  primals_659 = None
    addmm_133: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_660, view_356, permute_178);  primals_660 = None
    view_357: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_133, [1, 128, 512]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_35: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_357);  view_357 = None
    alias_44: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_358: "f32[128, 512]" = torch.ops.aten.view.default(relu_35, [128, 512]);  relu_35 = None
    permute_179: "f32[512, 128]" = torch.ops.aten.permute.default(primals_661, [1, 0]);  primals_661 = None
    addmm_134: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_662, view_358, permute_179);  primals_662 = None
    view_359: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_134, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_134: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_359, add_133);  view_359 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_72: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_134, primals_143);  add_134 = None
    add_135: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_72, primals_144);  mul_72 = primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_360: "f32[128, 128]" = torch.ops.aten.view.default(add_135, [128, 128]);  add_135 = None
    permute_180: "f32[128, 512]" = torch.ops.aten.permute.default(primals_663, [1, 0]);  primals_663 = None
    addmm_135: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_664, view_360, permute_180);  primals_664 = None
    view_361: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_135, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_18: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_361);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_136: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_18, add_122);  clone_18 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_73: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_136, primals_145);  add_136 = None
    add_137: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_73, primals_146);  mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_362: "f32[128, 512]" = torch.ops.aten.view.default(add_137, [128, 512])
    permute_181: "f32[512, 128]" = torch.ops.aten.permute.default(primals_665, [1, 0]);  primals_665 = None
    addmm_136: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_666, view_362, permute_181);  primals_666 = None
    view_363: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_136, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_74: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_363, primals_147);  view_363 = None
    add_138: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_74, primals_148);  mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_182: "f32[512, 128]" = torch.ops.aten.permute.default(primals_667, [1, 0]);  primals_667 = None
    addmm_137: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_668, view_362, permute_182);  primals_668 = None
    view_365: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_137, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_75: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_365, primals_149);  view_365 = None
    add_139: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_75, primals_150);  mul_75 = primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_366: "f32[128, 128]" = torch.ops.aten.view.default(add_139, [128, 128]);  add_139 = None
    permute_183: "f32[128, 128]" = torch.ops.aten.permute.default(primals_669, [1, 0]);  primals_669 = None
    addmm_138: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_670, view_366, permute_183);  primals_670 = None
    view_367: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_138, [1, 128, 128]);  addmm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_184: "f32[128, 128]" = torch.ops.aten.permute.default(primals_671, [1, 0]);  primals_671 = None
    addmm_139: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_672, view_366, permute_184);  primals_672 = None
    view_369: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_139, [1, 128, 128]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_185: "f32[512, 128]" = torch.ops.aten.permute.default(primals_673, [1, 0]);  primals_673 = None
    addmm_140: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_674, view_362, permute_185);  primals_674 = None
    view_371: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_140, [1, 128, 128]);  addmm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_372: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_367, [1, 128, 4, 32]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_186: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_373: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_369, [1, 128, 4, 32]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_187: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_374: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_371, [1, 128, 4, 32]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_188: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
    
    # No stacktrace found for following nodes
    clone_default_42: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    clone_default_43: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
    clone_default_44: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    _scaled_dot_product_efficient_attention_default_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_42, clone_default_43, clone_default_44, None, True, 0.1, scale = 0.17677669529663687)
    getitem_148: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_14[0]
    getitem_149: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_14[1]
    getitem_150: "i64[]" = _scaled_dot_product_efficient_attention_default_14[2]
    getitem_151: "i64[]" = _scaled_dot_product_efficient_attention_default_14[3];  _scaled_dot_product_efficient_attention_default_14 = None
    alias_default_28: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_148)
    alias_default_29: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_28);  alias_default_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_190: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_148, [0, 2, 1, 3]);  getitem_148 = None
    clone_19: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_381: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_19, [1, 128, 128]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_382: "f32[128, 128]" = torch.ops.aten.view.default(view_381, [128, 128]);  view_381 = None
    permute_191: "f32[128, 128]" = torch.ops.aten.permute.default(primals_675, [1, 0]);  primals_675 = None
    addmm_141: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_676, view_382, permute_191);  primals_676 = None
    view_383: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_141, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_141: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_383, add_138);  view_383 = add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_76: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_141, primals_151);  add_141 = None
    add_142: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_76, primals_152);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_384: "f32[128, 128]" = torch.ops.aten.view.default(add_142, [128, 128])
    permute_192: "f32[128, 512]" = torch.ops.aten.permute.default(primals_677, [1, 0]);  primals_677 = None
    addmm_142: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_678, view_384, permute_192);  primals_678 = None
    view_385: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_142, [1, 128, 512]);  addmm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_36: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_385);  view_385 = None
    alias_46: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_386: "f32[128, 512]" = torch.ops.aten.view.default(relu_36, [128, 512]);  relu_36 = None
    permute_193: "f32[512, 128]" = torch.ops.aten.permute.default(primals_679, [1, 0]);  primals_679 = None
    addmm_143: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_680, view_386, permute_193);  primals_680 = None
    view_387: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_143, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_143: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_387, add_142);  view_387 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_77: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_143, primals_153);  add_143 = None
    add_144: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_77, primals_154);  mul_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_388: "f32[128, 128]" = torch.ops.aten.view.default(add_144, [128, 128])
    permute_194: "f32[128, 512]" = torch.ops.aten.permute.default(primals_681, [1, 0]);  primals_681 = None
    addmm_144: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_682, view_388, permute_194);  primals_682 = None
    view_389: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_144, [1, 128, 512]);  addmm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_37: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_389);  view_389 = None
    alias_47: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_390: "f32[128, 512]" = torch.ops.aten.view.default(relu_37, [128, 512]);  relu_37 = None
    permute_195: "f32[512, 128]" = torch.ops.aten.permute.default(primals_683, [1, 0]);  primals_683 = None
    addmm_145: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_684, view_390, permute_195);  primals_684 = None
    view_391: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_145, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_145: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_391, add_144);  view_391 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_78: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_145, primals_155);  add_145 = None
    add_146: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_78, primals_156);  mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_392: "f32[128, 128]" = torch.ops.aten.view.default(add_146, [128, 128])
    permute_196: "f32[128, 512]" = torch.ops.aten.permute.default(primals_685, [1, 0]);  primals_685 = None
    addmm_146: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_686, view_392, permute_196);  primals_686 = None
    view_393: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_146, [1, 128, 512]);  addmm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_38: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_393);  view_393 = None
    alias_48: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_394: "f32[128, 512]" = torch.ops.aten.view.default(relu_38, [128, 512]);  relu_38 = None
    permute_197: "f32[512, 128]" = torch.ops.aten.permute.default(primals_687, [1, 0]);  primals_687 = None
    addmm_147: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_688, view_394, permute_197);  primals_688 = None
    view_395: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_147, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_147: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_395, add_146);  view_395 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_79: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_147, primals_157);  add_147 = None
    add_148: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_79, primals_158);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_396: "f32[128, 128]" = torch.ops.aten.view.default(add_148, [128, 128])
    permute_198: "f32[128, 512]" = torch.ops.aten.permute.default(primals_689, [1, 0]);  primals_689 = None
    addmm_148: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_690, view_396, permute_198);  primals_690 = None
    view_397: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_148, [1, 128, 512]);  addmm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_39: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_397);  view_397 = None
    alias_49: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_398: "f32[128, 512]" = torch.ops.aten.view.default(relu_39, [128, 512]);  relu_39 = None
    permute_199: "f32[512, 128]" = torch.ops.aten.permute.default(primals_691, [1, 0]);  primals_691 = None
    addmm_149: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_692, view_398, permute_199);  primals_692 = None
    view_399: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_149, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_149: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_399, add_148);  view_399 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_80: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_149, primals_159);  add_149 = None
    add_150: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_80, primals_160);  mul_80 = primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_400: "f32[128, 128]" = torch.ops.aten.view.default(add_150, [128, 128]);  add_150 = None
    permute_200: "f32[128, 512]" = torch.ops.aten.permute.default(primals_693, [1, 0]);  primals_693 = None
    addmm_150: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_694, view_400, permute_200);  primals_694 = None
    view_401: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_150, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_20: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_401);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_151: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_20, add_137);  clone_20 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_81: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_151, primals_161);  add_151 = None
    add_152: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_81, primals_162);  mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_402: "f32[128, 512]" = torch.ops.aten.view.default(add_152, [128, 512])
    permute_201: "f32[512, 128]" = torch.ops.aten.permute.default(primals_695, [1, 0]);  primals_695 = None
    addmm_151: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_696, view_402, permute_201);  primals_696 = None
    view_403: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_151, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_82: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_403, primals_163);  view_403 = None
    add_153: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_82, primals_164);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_202: "f32[512, 128]" = torch.ops.aten.permute.default(primals_697, [1, 0]);  primals_697 = None
    addmm_152: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_698, view_402, permute_202);  primals_698 = None
    view_405: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_152, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_83: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_405, primals_165);  view_405 = None
    add_154: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_83, primals_166);  mul_83 = primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_406: "f32[128, 128]" = torch.ops.aten.view.default(add_154, [128, 128]);  add_154 = None
    permute_203: "f32[128, 128]" = torch.ops.aten.permute.default(primals_699, [1, 0]);  primals_699 = None
    addmm_153: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_700, view_406, permute_203);  primals_700 = None
    view_407: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_153, [1, 128, 128]);  addmm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_204: "f32[128, 128]" = torch.ops.aten.permute.default(primals_701, [1, 0]);  primals_701 = None
    addmm_154: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_702, view_406, permute_204);  primals_702 = None
    view_409: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_154, [1, 128, 128]);  addmm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_205: "f32[512, 128]" = torch.ops.aten.permute.default(primals_703, [1, 0]);  primals_703 = None
    addmm_155: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_704, view_402, permute_205);  primals_704 = None
    view_411: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_155, [1, 128, 128]);  addmm_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_412: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_407, [1, 128, 4, 32]);  view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_206: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_413: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_409, [1, 128, 4, 32]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_207: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_414: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_411, [1, 128, 4, 32]);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_208: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
    
    # No stacktrace found for following nodes
    clone_default_39: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
    clone_default_40: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
    clone_default_41: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
    _scaled_dot_product_efficient_attention_default_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_39, clone_default_40, clone_default_41, None, True, 0.1, scale = 0.17677669529663687)
    getitem_141: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_13[0]
    getitem_142: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_13[1]
    getitem_143: "i64[]" = _scaled_dot_product_efficient_attention_default_13[2]
    getitem_144: "i64[]" = _scaled_dot_product_efficient_attention_default_13[3];  _scaled_dot_product_efficient_attention_default_13 = None
    alias_default_26: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_141)
    alias_default_27: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_26);  alias_default_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_210: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_141, [0, 2, 1, 3]);  getitem_141 = None
    clone_21: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_421: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_21, [1, 128, 128]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_422: "f32[128, 128]" = torch.ops.aten.view.default(view_421, [128, 128]);  view_421 = None
    permute_211: "f32[128, 128]" = torch.ops.aten.permute.default(primals_705, [1, 0]);  primals_705 = None
    addmm_156: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_706, view_422, permute_211);  primals_706 = None
    view_423: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_156, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_156: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_423, add_153);  view_423 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_84: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_156, primals_167);  add_156 = None
    add_157: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_84, primals_168);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_424: "f32[128, 128]" = torch.ops.aten.view.default(add_157, [128, 128])
    permute_212: "f32[128, 512]" = torch.ops.aten.permute.default(primals_707, [1, 0]);  primals_707 = None
    addmm_157: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_708, view_424, permute_212);  primals_708 = None
    view_425: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_157, [1, 128, 512]);  addmm_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_40: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_425);  view_425 = None
    alias_51: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_426: "f32[128, 512]" = torch.ops.aten.view.default(relu_40, [128, 512]);  relu_40 = None
    permute_213: "f32[512, 128]" = torch.ops.aten.permute.default(primals_709, [1, 0]);  primals_709 = None
    addmm_158: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_710, view_426, permute_213);  primals_710 = None
    view_427: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_158, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_158: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_427, add_157);  view_427 = add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_85: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_158, primals_169);  add_158 = None
    add_159: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_85, primals_170);  mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_428: "f32[128, 128]" = torch.ops.aten.view.default(add_159, [128, 128])
    permute_214: "f32[128, 512]" = torch.ops.aten.permute.default(primals_711, [1, 0]);  primals_711 = None
    addmm_159: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_712, view_428, permute_214);  primals_712 = None
    view_429: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_159, [1, 128, 512]);  addmm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_41: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_429);  view_429 = None
    alias_52: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_430: "f32[128, 512]" = torch.ops.aten.view.default(relu_41, [128, 512]);  relu_41 = None
    permute_215: "f32[512, 128]" = torch.ops.aten.permute.default(primals_713, [1, 0]);  primals_713 = None
    addmm_160: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_714, view_430, permute_215);  primals_714 = None
    view_431: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_160, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_160: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_431, add_159);  view_431 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_86: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_160, primals_171);  add_160 = None
    add_161: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_86, primals_172);  mul_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_432: "f32[128, 128]" = torch.ops.aten.view.default(add_161, [128, 128])
    permute_216: "f32[128, 512]" = torch.ops.aten.permute.default(primals_715, [1, 0]);  primals_715 = None
    addmm_161: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_716, view_432, permute_216);  primals_716 = None
    view_433: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_161, [1, 128, 512]);  addmm_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_42: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_433);  view_433 = None
    alias_53: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_434: "f32[128, 512]" = torch.ops.aten.view.default(relu_42, [128, 512]);  relu_42 = None
    permute_217: "f32[512, 128]" = torch.ops.aten.permute.default(primals_717, [1, 0]);  primals_717 = None
    addmm_162: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_718, view_434, permute_217);  primals_718 = None
    view_435: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_162, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_162: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_435, add_161);  view_435 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_87: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_162, primals_173);  add_162 = None
    add_163: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_87, primals_174);  mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_436: "f32[128, 128]" = torch.ops.aten.view.default(add_163, [128, 128])
    permute_218: "f32[128, 512]" = torch.ops.aten.permute.default(primals_719, [1, 0]);  primals_719 = None
    addmm_163: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_720, view_436, permute_218);  primals_720 = None
    view_437: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_163, [1, 128, 512]);  addmm_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_43: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_437);  view_437 = None
    alias_54: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_438: "f32[128, 512]" = torch.ops.aten.view.default(relu_43, [128, 512]);  relu_43 = None
    permute_219: "f32[512, 128]" = torch.ops.aten.permute.default(primals_721, [1, 0]);  primals_721 = None
    addmm_164: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_722, view_438, permute_219);  primals_722 = None
    view_439: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_164, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_164: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_439, add_163);  view_439 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_88: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_164, primals_175);  add_164 = None
    add_165: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_88, primals_176);  mul_88 = primals_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_440: "f32[128, 128]" = torch.ops.aten.view.default(add_165, [128, 128]);  add_165 = None
    permute_220: "f32[128, 512]" = torch.ops.aten.permute.default(primals_723, [1, 0]);  primals_723 = None
    addmm_165: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_724, view_440, permute_220);  primals_724 = None
    view_441: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_165, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_22: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_441);  view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_166: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_22, add_152);  clone_22 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_89: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_166, primals_177);  add_166 = None
    add_167: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_89, primals_178);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_442: "f32[128, 512]" = torch.ops.aten.view.default(add_167, [128, 512])
    permute_221: "f32[512, 128]" = torch.ops.aten.permute.default(primals_725, [1, 0]);  primals_725 = None
    addmm_166: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_726, view_442, permute_221);  primals_726 = None
    view_443: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_166, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_90: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_443, primals_179);  view_443 = None
    add_168: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_90, primals_180);  mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_222: "f32[512, 128]" = torch.ops.aten.permute.default(primals_727, [1, 0]);  primals_727 = None
    addmm_167: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_728, view_442, permute_222);  primals_728 = None
    view_445: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_167, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_91: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_445, primals_181);  view_445 = None
    add_169: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_91, primals_182);  mul_91 = primals_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_446: "f32[128, 128]" = torch.ops.aten.view.default(add_169, [128, 128]);  add_169 = None
    permute_223: "f32[128, 128]" = torch.ops.aten.permute.default(primals_729, [1, 0]);  primals_729 = None
    addmm_168: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_730, view_446, permute_223);  primals_730 = None
    view_447: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_168, [1, 128, 128]);  addmm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_224: "f32[128, 128]" = torch.ops.aten.permute.default(primals_731, [1, 0]);  primals_731 = None
    addmm_169: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_732, view_446, permute_224);  primals_732 = None
    view_449: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_169, [1, 128, 128]);  addmm_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_225: "f32[512, 128]" = torch.ops.aten.permute.default(primals_733, [1, 0]);  primals_733 = None
    addmm_170: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_734, view_442, permute_225);  primals_734 = None
    view_451: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_170, [1, 128, 128]);  addmm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_452: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_447, [1, 128, 4, 32]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_226: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_453: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_449, [1, 128, 4, 32]);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_454: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_451, [1, 128, 4, 32]);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_228: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
    
    # No stacktrace found for following nodes
    clone_default_36: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    clone_default_37: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    clone_default_38: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    _scaled_dot_product_efficient_attention_default_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_36, clone_default_37, clone_default_38, None, True, 0.1, scale = 0.17677669529663687)
    getitem_134: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_12[0]
    getitem_135: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_12[1]
    getitem_136: "i64[]" = _scaled_dot_product_efficient_attention_default_12[2]
    getitem_137: "i64[]" = _scaled_dot_product_efficient_attention_default_12[3];  _scaled_dot_product_efficient_attention_default_12 = None
    alias_default_24: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_134)
    alias_default_25: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_24);  alias_default_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_230: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None
    clone_23: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_461: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_23, [1, 128, 128]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_462: "f32[128, 128]" = torch.ops.aten.view.default(view_461, [128, 128]);  view_461 = None
    permute_231: "f32[128, 128]" = torch.ops.aten.permute.default(primals_735, [1, 0]);  primals_735 = None
    addmm_171: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_736, view_462, permute_231);  primals_736 = None
    view_463: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_171, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_171: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_463, add_168);  view_463 = add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_92: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_171, primals_183);  add_171 = None
    add_172: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_92, primals_184);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_464: "f32[128, 128]" = torch.ops.aten.view.default(add_172, [128, 128])
    permute_232: "f32[128, 512]" = torch.ops.aten.permute.default(primals_737, [1, 0]);  primals_737 = None
    addmm_172: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_738, view_464, permute_232);  primals_738 = None
    view_465: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_172, [1, 128, 512]);  addmm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_44: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_465);  view_465 = None
    alias_56: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_466: "f32[128, 512]" = torch.ops.aten.view.default(relu_44, [128, 512]);  relu_44 = None
    permute_233: "f32[512, 128]" = torch.ops.aten.permute.default(primals_739, [1, 0]);  primals_739 = None
    addmm_173: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_740, view_466, permute_233);  primals_740 = None
    view_467: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_173, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_173: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_467, add_172);  view_467 = add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_93: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_173, primals_185);  add_173 = None
    add_174: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_93, primals_186);  mul_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_468: "f32[128, 128]" = torch.ops.aten.view.default(add_174, [128, 128])
    permute_234: "f32[128, 512]" = torch.ops.aten.permute.default(primals_741, [1, 0]);  primals_741 = None
    addmm_174: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_742, view_468, permute_234);  primals_742 = None
    view_469: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_174, [1, 128, 512]);  addmm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_45: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_469);  view_469 = None
    alias_57: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_45)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_470: "f32[128, 512]" = torch.ops.aten.view.default(relu_45, [128, 512]);  relu_45 = None
    permute_235: "f32[512, 128]" = torch.ops.aten.permute.default(primals_743, [1, 0]);  primals_743 = None
    addmm_175: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_744, view_470, permute_235);  primals_744 = None
    view_471: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_175, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_175: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_471, add_174);  view_471 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_94: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_175, primals_187);  add_175 = None
    add_176: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_94, primals_188);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_472: "f32[128, 128]" = torch.ops.aten.view.default(add_176, [128, 128])
    permute_236: "f32[128, 512]" = torch.ops.aten.permute.default(primals_745, [1, 0]);  primals_745 = None
    addmm_176: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_746, view_472, permute_236);  primals_746 = None
    view_473: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_176, [1, 128, 512]);  addmm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_46: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_473);  view_473 = None
    alias_58: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_474: "f32[128, 512]" = torch.ops.aten.view.default(relu_46, [128, 512]);  relu_46 = None
    permute_237: "f32[512, 128]" = torch.ops.aten.permute.default(primals_747, [1, 0]);  primals_747 = None
    addmm_177: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_748, view_474, permute_237);  primals_748 = None
    view_475: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_177, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_177: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_475, add_176);  view_475 = add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_95: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_177, primals_189);  add_177 = None
    add_178: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_95, primals_190);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_476: "f32[128, 128]" = torch.ops.aten.view.default(add_178, [128, 128])
    permute_238: "f32[128, 512]" = torch.ops.aten.permute.default(primals_749, [1, 0]);  primals_749 = None
    addmm_178: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_750, view_476, permute_238);  primals_750 = None
    view_477: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_178, [1, 128, 512]);  addmm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_47: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_477);  view_477 = None
    alias_59: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_478: "f32[128, 512]" = torch.ops.aten.view.default(relu_47, [128, 512]);  relu_47 = None
    permute_239: "f32[512, 128]" = torch.ops.aten.permute.default(primals_751, [1, 0]);  primals_751 = None
    addmm_179: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_752, view_478, permute_239);  primals_752 = None
    view_479: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_179, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_179: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_479, add_178);  view_479 = add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_96: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_179, primals_191);  add_179 = None
    add_180: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_96, primals_192);  mul_96 = primals_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_480: "f32[128, 128]" = torch.ops.aten.view.default(add_180, [128, 128]);  add_180 = None
    permute_240: "f32[128, 512]" = torch.ops.aten.permute.default(primals_753, [1, 0]);  primals_753 = None
    addmm_180: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_754, view_480, permute_240);  primals_754 = None
    view_481: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_180, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_24: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_481);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_181: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_24, add_167);  clone_24 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_97: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_181, primals_193);  add_181 = None
    add_182: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_97, primals_194);  mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_482: "f32[128, 512]" = torch.ops.aten.view.default(add_182, [128, 512])
    permute_241: "f32[512, 128]" = torch.ops.aten.permute.default(primals_755, [1, 0]);  primals_755 = None
    addmm_181: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_756, view_482, permute_241);  primals_756 = None
    view_483: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_181, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_98: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_483, primals_195);  view_483 = None
    add_183: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_98, primals_196);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_242: "f32[512, 128]" = torch.ops.aten.permute.default(primals_757, [1, 0]);  primals_757 = None
    addmm_182: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_758, view_482, permute_242);  primals_758 = None
    view_485: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_182, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_99: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_485, primals_197);  view_485 = None
    add_184: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_99, primals_198);  mul_99 = primals_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_486: "f32[128, 128]" = torch.ops.aten.view.default(add_184, [128, 128]);  add_184 = None
    permute_243: "f32[128, 128]" = torch.ops.aten.permute.default(primals_759, [1, 0]);  primals_759 = None
    addmm_183: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_760, view_486, permute_243);  primals_760 = None
    view_487: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_183, [1, 128, 128]);  addmm_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_244: "f32[128, 128]" = torch.ops.aten.permute.default(primals_761, [1, 0]);  primals_761 = None
    addmm_184: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_762, view_486, permute_244);  primals_762 = None
    view_489: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_184, [1, 128, 128]);  addmm_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_245: "f32[512, 128]" = torch.ops.aten.permute.default(primals_763, [1, 0]);  primals_763 = None
    addmm_185: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_764, view_482, permute_245);  primals_764 = None
    view_491: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_185, [1, 128, 128]);  addmm_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_492: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_487, [1, 128, 4, 32]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_246: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_493: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_489, [1, 128, 4, 32]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_247: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_494: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_491, [1, 128, 4, 32]);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_248: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
    
    # No stacktrace found for following nodes
    clone_default_33: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
    clone_default_34: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    clone_default_35: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
    _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_33, clone_default_34, clone_default_35, None, True, 0.1, scale = 0.17677669529663687)
    getitem_127: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_11[0]
    getitem_128: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_11[1]
    getitem_129: "i64[]" = _scaled_dot_product_efficient_attention_default_11[2]
    getitem_130: "i64[]" = _scaled_dot_product_efficient_attention_default_11[3];  _scaled_dot_product_efficient_attention_default_11 = None
    alias_default_22: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_127)
    alias_default_23: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_22);  alias_default_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_250: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_127, [0, 2, 1, 3]);  getitem_127 = None
    clone_25: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_501: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_25, [1, 128, 128]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_502: "f32[128, 128]" = torch.ops.aten.view.default(view_501, [128, 128]);  view_501 = None
    permute_251: "f32[128, 128]" = torch.ops.aten.permute.default(primals_765, [1, 0]);  primals_765 = None
    addmm_186: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_766, view_502, permute_251);  primals_766 = None
    view_503: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_186, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_186: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_503, add_183);  view_503 = add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_100: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_186, primals_199);  add_186 = None
    add_187: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_100, primals_200);  mul_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_504: "f32[128, 128]" = torch.ops.aten.view.default(add_187, [128, 128])
    permute_252: "f32[128, 512]" = torch.ops.aten.permute.default(primals_767, [1, 0]);  primals_767 = None
    addmm_187: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_768, view_504, permute_252);  primals_768 = None
    view_505: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_187, [1, 128, 512]);  addmm_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_48: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_505);  view_505 = None
    alias_61: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_506: "f32[128, 512]" = torch.ops.aten.view.default(relu_48, [128, 512]);  relu_48 = None
    permute_253: "f32[512, 128]" = torch.ops.aten.permute.default(primals_769, [1, 0]);  primals_769 = None
    addmm_188: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_770, view_506, permute_253);  primals_770 = None
    view_507: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_188, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_188: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_507, add_187);  view_507 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_101: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_188, primals_201);  add_188 = None
    add_189: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_101, primals_202);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_508: "f32[128, 128]" = torch.ops.aten.view.default(add_189, [128, 128])
    permute_254: "f32[128, 512]" = torch.ops.aten.permute.default(primals_771, [1, 0]);  primals_771 = None
    addmm_189: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_772, view_508, permute_254);  primals_772 = None
    view_509: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_189, [1, 128, 512]);  addmm_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_49: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_509);  view_509 = None
    alias_62: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_510: "f32[128, 512]" = torch.ops.aten.view.default(relu_49, [128, 512]);  relu_49 = None
    permute_255: "f32[512, 128]" = torch.ops.aten.permute.default(primals_773, [1, 0]);  primals_773 = None
    addmm_190: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_774, view_510, permute_255);  primals_774 = None
    view_511: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_190, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_190: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_511, add_189);  view_511 = add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_102: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_190, primals_203);  add_190 = None
    add_191: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_102, primals_204);  mul_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_512: "f32[128, 128]" = torch.ops.aten.view.default(add_191, [128, 128])
    permute_256: "f32[128, 512]" = torch.ops.aten.permute.default(primals_775, [1, 0]);  primals_775 = None
    addmm_191: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_776, view_512, permute_256);  primals_776 = None
    view_513: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_191, [1, 128, 512]);  addmm_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_50: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_513);  view_513 = None
    alias_63: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_514: "f32[128, 512]" = torch.ops.aten.view.default(relu_50, [128, 512]);  relu_50 = None
    permute_257: "f32[512, 128]" = torch.ops.aten.permute.default(primals_777, [1, 0]);  primals_777 = None
    addmm_192: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_778, view_514, permute_257);  primals_778 = None
    view_515: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_192, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_192: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_515, add_191);  view_515 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_103: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_192, primals_205);  add_192 = None
    add_193: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_103, primals_206);  mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_516: "f32[128, 128]" = torch.ops.aten.view.default(add_193, [128, 128])
    permute_258: "f32[128, 512]" = torch.ops.aten.permute.default(primals_779, [1, 0]);  primals_779 = None
    addmm_193: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_780, view_516, permute_258);  primals_780 = None
    view_517: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_193, [1, 128, 512]);  addmm_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_51: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_517);  view_517 = None
    alias_64: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_518: "f32[128, 512]" = torch.ops.aten.view.default(relu_51, [128, 512]);  relu_51 = None
    permute_259: "f32[512, 128]" = torch.ops.aten.permute.default(primals_781, [1, 0]);  primals_781 = None
    addmm_194: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_782, view_518, permute_259);  primals_782 = None
    view_519: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_194, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_194: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_519, add_193);  view_519 = add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_104: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_194, primals_207);  add_194 = None
    add_195: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_104, primals_208);  mul_104 = primals_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_520: "f32[128, 128]" = torch.ops.aten.view.default(add_195, [128, 128]);  add_195 = None
    permute_260: "f32[128, 512]" = torch.ops.aten.permute.default(primals_783, [1, 0]);  primals_783 = None
    addmm_195: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_784, view_520, permute_260);  primals_784 = None
    view_521: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_195, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_26: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_521);  view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_196: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_26, add_182);  clone_26 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_105: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_196, primals_209);  add_196 = None
    add_197: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_105, primals_210);  mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_522: "f32[128, 512]" = torch.ops.aten.view.default(add_197, [128, 512])
    permute_261: "f32[512, 128]" = torch.ops.aten.permute.default(primals_785, [1, 0]);  primals_785 = None
    addmm_196: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_786, view_522, permute_261);  primals_786 = None
    view_523: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_196, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_106: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_523, primals_211);  view_523 = None
    add_198: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_106, primals_212);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_262: "f32[512, 128]" = torch.ops.aten.permute.default(primals_787, [1, 0]);  primals_787 = None
    addmm_197: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_788, view_522, permute_262);  primals_788 = None
    view_525: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_197, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_107: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_525, primals_213);  view_525 = None
    add_199: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_107, primals_214);  mul_107 = primals_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_526: "f32[128, 128]" = torch.ops.aten.view.default(add_199, [128, 128]);  add_199 = None
    permute_263: "f32[128, 128]" = torch.ops.aten.permute.default(primals_789, [1, 0]);  primals_789 = None
    addmm_198: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_790, view_526, permute_263);  primals_790 = None
    view_527: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_198, [1, 128, 128]);  addmm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_264: "f32[128, 128]" = torch.ops.aten.permute.default(primals_791, [1, 0]);  primals_791 = None
    addmm_199: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_792, view_526, permute_264);  primals_792 = None
    view_529: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_199, [1, 128, 128]);  addmm_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_265: "f32[512, 128]" = torch.ops.aten.permute.default(primals_793, [1, 0]);  primals_793 = None
    addmm_200: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_794, view_522, permute_265);  primals_794 = None
    view_531: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_200, [1, 128, 128]);  addmm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_532: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_527, [1, 128, 4, 32]);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_266: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_533: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_529, [1, 128, 4, 32]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_267: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_533, [0, 2, 1, 3]);  view_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_534: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_531, [1, 128, 4, 32]);  view_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_268: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
    
    # No stacktrace found for following nodes
    clone_default_30: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    clone_default_31: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
    clone_default_32: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_268, memory_format = torch.contiguous_format);  permute_268 = None
    _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_30, clone_default_31, clone_default_32, None, True, 0.1, scale = 0.17677669529663687)
    getitem_120: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_10[0]
    getitem_121: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_10[1]
    getitem_122: "i64[]" = _scaled_dot_product_efficient_attention_default_10[2]
    getitem_123: "i64[]" = _scaled_dot_product_efficient_attention_default_10[3];  _scaled_dot_product_efficient_attention_default_10 = None
    alias_default_20: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_120)
    alias_default_21: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_20);  alias_default_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_270: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_120, [0, 2, 1, 3]);  getitem_120 = None
    clone_27: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_541: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_27, [1, 128, 128]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_542: "f32[128, 128]" = torch.ops.aten.view.default(view_541, [128, 128]);  view_541 = None
    permute_271: "f32[128, 128]" = torch.ops.aten.permute.default(primals_795, [1, 0]);  primals_795 = None
    addmm_201: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_796, view_542, permute_271);  primals_796 = None
    view_543: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_201, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_201: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_543, add_198);  view_543 = add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_108: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_201, primals_215);  add_201 = None
    add_202: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_108, primals_216);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_544: "f32[128, 128]" = torch.ops.aten.view.default(add_202, [128, 128])
    permute_272: "f32[128, 512]" = torch.ops.aten.permute.default(primals_797, [1, 0]);  primals_797 = None
    addmm_202: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_798, view_544, permute_272);  primals_798 = None
    view_545: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_202, [1, 128, 512]);  addmm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_52: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_545);  view_545 = None
    alias_66: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_546: "f32[128, 512]" = torch.ops.aten.view.default(relu_52, [128, 512]);  relu_52 = None
    permute_273: "f32[512, 128]" = torch.ops.aten.permute.default(primals_799, [1, 0]);  primals_799 = None
    addmm_203: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_800, view_546, permute_273);  primals_800 = None
    view_547: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_203, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_203: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_547, add_202);  view_547 = add_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_109: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_203, primals_217);  add_203 = None
    add_204: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_109, primals_218);  mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_548: "f32[128, 128]" = torch.ops.aten.view.default(add_204, [128, 128])
    permute_274: "f32[128, 512]" = torch.ops.aten.permute.default(primals_801, [1, 0]);  primals_801 = None
    addmm_204: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_802, view_548, permute_274);  primals_802 = None
    view_549: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_204, [1, 128, 512]);  addmm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_53: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_549);  view_549 = None
    alias_67: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_550: "f32[128, 512]" = torch.ops.aten.view.default(relu_53, [128, 512]);  relu_53 = None
    permute_275: "f32[512, 128]" = torch.ops.aten.permute.default(primals_803, [1, 0]);  primals_803 = None
    addmm_205: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_804, view_550, permute_275);  primals_804 = None
    view_551: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_205, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_205: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_551, add_204);  view_551 = add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_110: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_205, primals_219);  add_205 = None
    add_206: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_110, primals_220);  mul_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_552: "f32[128, 128]" = torch.ops.aten.view.default(add_206, [128, 128])
    permute_276: "f32[128, 512]" = torch.ops.aten.permute.default(primals_805, [1, 0]);  primals_805 = None
    addmm_206: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_806, view_552, permute_276);  primals_806 = None
    view_553: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_206, [1, 128, 512]);  addmm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_54: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_553);  view_553 = None
    alias_68: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_554: "f32[128, 512]" = torch.ops.aten.view.default(relu_54, [128, 512]);  relu_54 = None
    permute_277: "f32[512, 128]" = torch.ops.aten.permute.default(primals_807, [1, 0]);  primals_807 = None
    addmm_207: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_808, view_554, permute_277);  primals_808 = None
    view_555: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_207, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_207: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_555, add_206);  view_555 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_111: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_207, primals_221);  add_207 = None
    add_208: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_111, primals_222);  mul_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_556: "f32[128, 128]" = torch.ops.aten.view.default(add_208, [128, 128])
    permute_278: "f32[128, 512]" = torch.ops.aten.permute.default(primals_809, [1, 0]);  primals_809 = None
    addmm_208: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_810, view_556, permute_278);  primals_810 = None
    view_557: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_208, [1, 128, 512]);  addmm_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_55: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_557);  view_557 = None
    alias_69: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_558: "f32[128, 512]" = torch.ops.aten.view.default(relu_55, [128, 512]);  relu_55 = None
    permute_279: "f32[512, 128]" = torch.ops.aten.permute.default(primals_811, [1, 0]);  primals_811 = None
    addmm_209: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_812, view_558, permute_279);  primals_812 = None
    view_559: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_209, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_209: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_559, add_208);  view_559 = add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_112: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_209, primals_223);  add_209 = None
    add_210: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_112, primals_224);  mul_112 = primals_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_560: "f32[128, 128]" = torch.ops.aten.view.default(add_210, [128, 128]);  add_210 = None
    permute_280: "f32[128, 512]" = torch.ops.aten.permute.default(primals_813, [1, 0]);  primals_813 = None
    addmm_210: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_814, view_560, permute_280);  primals_814 = None
    view_561: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_210, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_28: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_561);  view_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_211: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_28, add_197);  clone_28 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_113: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_211, primals_225);  add_211 = None
    add_212: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_113, primals_226);  mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_562: "f32[128, 512]" = torch.ops.aten.view.default(add_212, [128, 512])
    permute_281: "f32[512, 128]" = torch.ops.aten.permute.default(primals_815, [1, 0]);  primals_815 = None
    addmm_211: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_816, view_562, permute_281);  primals_816 = None
    view_563: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_211, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_114: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_563, primals_227);  view_563 = None
    add_213: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_114, primals_228);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_282: "f32[512, 128]" = torch.ops.aten.permute.default(primals_817, [1, 0]);  primals_817 = None
    addmm_212: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_818, view_562, permute_282);  primals_818 = None
    view_565: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_212, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_115: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_565, primals_229);  view_565 = None
    add_214: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_115, primals_230);  mul_115 = primals_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_566: "f32[128, 128]" = torch.ops.aten.view.default(add_214, [128, 128]);  add_214 = None
    permute_283: "f32[128, 128]" = torch.ops.aten.permute.default(primals_819, [1, 0]);  primals_819 = None
    addmm_213: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_820, view_566, permute_283);  primals_820 = None
    view_567: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_213, [1, 128, 128]);  addmm_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_284: "f32[128, 128]" = torch.ops.aten.permute.default(primals_821, [1, 0]);  primals_821 = None
    addmm_214: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_822, view_566, permute_284);  primals_822 = None
    view_569: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_214, [1, 128, 128]);  addmm_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_285: "f32[512, 128]" = torch.ops.aten.permute.default(primals_823, [1, 0]);  primals_823 = None
    addmm_215: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_824, view_562, permute_285);  primals_824 = None
    view_571: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_215, [1, 128, 128]);  addmm_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_572: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_567, [1, 128, 4, 32]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_286: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_573: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_569, [1, 128, 4, 32]);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_287: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_574: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_571, [1, 128, 4, 32]);  view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_288: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None
    
    # No stacktrace found for following nodes
    clone_default_27: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    clone_default_28: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
    clone_default_29: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_27, clone_default_28, clone_default_29, None, True, 0.1, scale = 0.17677669529663687)
    getitem_113: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_9[0]
    getitem_114: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_9[1]
    getitem_115: "i64[]" = _scaled_dot_product_efficient_attention_default_9[2]
    getitem_116: "i64[]" = _scaled_dot_product_efficient_attention_default_9[3];  _scaled_dot_product_efficient_attention_default_9 = None
    alias_default_18: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_113)
    alias_default_19: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_18);  alias_default_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_290: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_113, [0, 2, 1, 3]);  getitem_113 = None
    clone_29: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_581: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_29, [1, 128, 128]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_582: "f32[128, 128]" = torch.ops.aten.view.default(view_581, [128, 128]);  view_581 = None
    permute_291: "f32[128, 128]" = torch.ops.aten.permute.default(primals_825, [1, 0]);  primals_825 = None
    addmm_216: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_826, view_582, permute_291);  primals_826 = None
    view_583: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_216, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_216: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_583, add_213);  view_583 = add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_116: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_216, primals_231);  add_216 = None
    add_217: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_116, primals_232);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_584: "f32[128, 128]" = torch.ops.aten.view.default(add_217, [128, 128])
    permute_292: "f32[128, 512]" = torch.ops.aten.permute.default(primals_827, [1, 0]);  primals_827 = None
    addmm_217: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_828, view_584, permute_292);  primals_828 = None
    view_585: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_217, [1, 128, 512]);  addmm_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_56: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_585);  view_585 = None
    alias_71: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_586: "f32[128, 512]" = torch.ops.aten.view.default(relu_56, [128, 512]);  relu_56 = None
    permute_293: "f32[512, 128]" = torch.ops.aten.permute.default(primals_829, [1, 0]);  primals_829 = None
    addmm_218: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_830, view_586, permute_293);  primals_830 = None
    view_587: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_218, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_218: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_587, add_217);  view_587 = add_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_117: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_218, primals_233);  add_218 = None
    add_219: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_117, primals_234);  mul_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_588: "f32[128, 128]" = torch.ops.aten.view.default(add_219, [128, 128])
    permute_294: "f32[128, 512]" = torch.ops.aten.permute.default(primals_831, [1, 0]);  primals_831 = None
    addmm_219: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_832, view_588, permute_294);  primals_832 = None
    view_589: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_219, [1, 128, 512]);  addmm_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_57: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_589);  view_589 = None
    alias_72: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_590: "f32[128, 512]" = torch.ops.aten.view.default(relu_57, [128, 512]);  relu_57 = None
    permute_295: "f32[512, 128]" = torch.ops.aten.permute.default(primals_833, [1, 0]);  primals_833 = None
    addmm_220: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_834, view_590, permute_295);  primals_834 = None
    view_591: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_220, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_220: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_591, add_219);  view_591 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_118: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_220, primals_235);  add_220 = None
    add_221: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_118, primals_236);  mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_592: "f32[128, 128]" = torch.ops.aten.view.default(add_221, [128, 128])
    permute_296: "f32[128, 512]" = torch.ops.aten.permute.default(primals_835, [1, 0]);  primals_835 = None
    addmm_221: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_836, view_592, permute_296);  primals_836 = None
    view_593: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_221, [1, 128, 512]);  addmm_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_58: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_593);  view_593 = None
    alias_73: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_58)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_594: "f32[128, 512]" = torch.ops.aten.view.default(relu_58, [128, 512]);  relu_58 = None
    permute_297: "f32[512, 128]" = torch.ops.aten.permute.default(primals_837, [1, 0]);  primals_837 = None
    addmm_222: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_838, view_594, permute_297);  primals_838 = None
    view_595: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_222, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_222: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_595, add_221);  view_595 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_119: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_222, primals_237);  add_222 = None
    add_223: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_119, primals_238);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_596: "f32[128, 128]" = torch.ops.aten.view.default(add_223, [128, 128])
    permute_298: "f32[128, 512]" = torch.ops.aten.permute.default(primals_839, [1, 0]);  primals_839 = None
    addmm_223: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_840, view_596, permute_298);  primals_840 = None
    view_597: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_223, [1, 128, 512]);  addmm_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_59: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_597);  view_597 = None
    alias_74: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_598: "f32[128, 512]" = torch.ops.aten.view.default(relu_59, [128, 512]);  relu_59 = None
    permute_299: "f32[512, 128]" = torch.ops.aten.permute.default(primals_841, [1, 0]);  primals_841 = None
    addmm_224: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_842, view_598, permute_299);  primals_842 = None
    view_599: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_224, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_224: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_599, add_223);  view_599 = add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_120: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_224, primals_239);  add_224 = None
    add_225: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_120, primals_240);  mul_120 = primals_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_600: "f32[128, 128]" = torch.ops.aten.view.default(add_225, [128, 128]);  add_225 = None
    permute_300: "f32[128, 512]" = torch.ops.aten.permute.default(primals_843, [1, 0]);  primals_843 = None
    addmm_225: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_844, view_600, permute_300);  primals_844 = None
    view_601: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_225, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_30: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_601);  view_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_226: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_30, add_212);  clone_30 = add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_121: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_226, primals_241);  add_226 = None
    add_227: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_121, primals_242);  mul_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_602: "f32[128, 512]" = torch.ops.aten.view.default(add_227, [128, 512])
    permute_301: "f32[512, 128]" = torch.ops.aten.permute.default(primals_845, [1, 0]);  primals_845 = None
    addmm_226: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_846, view_602, permute_301);  primals_846 = None
    view_603: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_226, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_122: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_603, primals_243);  view_603 = None
    add_228: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_122, primals_244);  mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_302: "f32[512, 128]" = torch.ops.aten.permute.default(primals_847, [1, 0]);  primals_847 = None
    addmm_227: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_848, view_602, permute_302);  primals_848 = None
    view_605: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_227, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_123: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_605, primals_245);  view_605 = None
    add_229: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_123, primals_246);  mul_123 = primals_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_606: "f32[128, 128]" = torch.ops.aten.view.default(add_229, [128, 128]);  add_229 = None
    permute_303: "f32[128, 128]" = torch.ops.aten.permute.default(primals_849, [1, 0]);  primals_849 = None
    addmm_228: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_850, view_606, permute_303);  primals_850 = None
    view_607: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_228, [1, 128, 128]);  addmm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_304: "f32[128, 128]" = torch.ops.aten.permute.default(primals_851, [1, 0]);  primals_851 = None
    addmm_229: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_852, view_606, permute_304);  primals_852 = None
    view_609: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_229, [1, 128, 128]);  addmm_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_305: "f32[512, 128]" = torch.ops.aten.permute.default(primals_853, [1, 0]);  primals_853 = None
    addmm_230: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_854, view_602, permute_305);  primals_854 = None
    view_611: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_230, [1, 128, 128]);  addmm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_612: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_607, [1, 128, 4, 32]);  view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_306: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_612, [0, 2, 1, 3]);  view_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_613: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_609, [1, 128, 4, 32]);  view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_307: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_613, [0, 2, 1, 3]);  view_613 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_614: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_611, [1, 128, 4, 32]);  view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_308: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
    
    # No stacktrace found for following nodes
    clone_default_24: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_306, memory_format = torch.contiguous_format);  permute_306 = None
    clone_default_25: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_307, memory_format = torch.contiguous_format);  permute_307 = None
    clone_default_26: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
    _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_24, clone_default_25, clone_default_26, None, True, 0.1, scale = 0.17677669529663687)
    getitem_106: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_8[0]
    getitem_107: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_8[1]
    getitem_108: "i64[]" = _scaled_dot_product_efficient_attention_default_8[2]
    getitem_109: "i64[]" = _scaled_dot_product_efficient_attention_default_8[3];  _scaled_dot_product_efficient_attention_default_8 = None
    alias_default_16: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_106)
    alias_default_17: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_16);  alias_default_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_310: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_106, [0, 2, 1, 3]);  getitem_106 = None
    clone_31: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_621: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_31, [1, 128, 128]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_622: "f32[128, 128]" = torch.ops.aten.view.default(view_621, [128, 128]);  view_621 = None
    permute_311: "f32[128, 128]" = torch.ops.aten.permute.default(primals_855, [1, 0]);  primals_855 = None
    addmm_231: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_856, view_622, permute_311);  primals_856 = None
    view_623: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_231, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_231: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_623, add_228);  view_623 = add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_124: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_231, primals_247);  add_231 = None
    add_232: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_124, primals_248);  mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_624: "f32[128, 128]" = torch.ops.aten.view.default(add_232, [128, 128])
    permute_312: "f32[128, 512]" = torch.ops.aten.permute.default(primals_857, [1, 0]);  primals_857 = None
    addmm_232: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_858, view_624, permute_312);  primals_858 = None
    view_625: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_232, [1, 128, 512]);  addmm_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_60: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_625);  view_625 = None
    alias_76: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_626: "f32[128, 512]" = torch.ops.aten.view.default(relu_60, [128, 512]);  relu_60 = None
    permute_313: "f32[512, 128]" = torch.ops.aten.permute.default(primals_859, [1, 0]);  primals_859 = None
    addmm_233: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_860, view_626, permute_313);  primals_860 = None
    view_627: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_233, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_233: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_627, add_232);  view_627 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_125: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_233, primals_249);  add_233 = None
    add_234: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_125, primals_250);  mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_628: "f32[128, 128]" = torch.ops.aten.view.default(add_234, [128, 128])
    permute_314: "f32[128, 512]" = torch.ops.aten.permute.default(primals_861, [1, 0]);  primals_861 = None
    addmm_234: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_862, view_628, permute_314);  primals_862 = None
    view_629: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_234, [1, 128, 512]);  addmm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_61: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_629);  view_629 = None
    alias_77: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_630: "f32[128, 512]" = torch.ops.aten.view.default(relu_61, [128, 512]);  relu_61 = None
    permute_315: "f32[512, 128]" = torch.ops.aten.permute.default(primals_863, [1, 0]);  primals_863 = None
    addmm_235: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_864, view_630, permute_315);  primals_864 = None
    view_631: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_235, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_235: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_631, add_234);  view_631 = add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_126: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_235, primals_251);  add_235 = None
    add_236: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_126, primals_252);  mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_632: "f32[128, 128]" = torch.ops.aten.view.default(add_236, [128, 128])
    permute_316: "f32[128, 512]" = torch.ops.aten.permute.default(primals_865, [1, 0]);  primals_865 = None
    addmm_236: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_866, view_632, permute_316);  primals_866 = None
    view_633: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_236, [1, 128, 512]);  addmm_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_62: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_633);  view_633 = None
    alias_78: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_634: "f32[128, 512]" = torch.ops.aten.view.default(relu_62, [128, 512]);  relu_62 = None
    permute_317: "f32[512, 128]" = torch.ops.aten.permute.default(primals_867, [1, 0]);  primals_867 = None
    addmm_237: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_868, view_634, permute_317);  primals_868 = None
    view_635: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_237, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_237: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_635, add_236);  view_635 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_127: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_237, primals_253);  add_237 = None
    add_238: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_127, primals_254);  mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_636: "f32[128, 128]" = torch.ops.aten.view.default(add_238, [128, 128])
    permute_318: "f32[128, 512]" = torch.ops.aten.permute.default(primals_869, [1, 0]);  primals_869 = None
    addmm_238: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_870, view_636, permute_318);  primals_870 = None
    view_637: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_238, [1, 128, 512]);  addmm_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_63: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_637);  view_637 = None
    alias_79: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_638: "f32[128, 512]" = torch.ops.aten.view.default(relu_63, [128, 512]);  relu_63 = None
    permute_319: "f32[512, 128]" = torch.ops.aten.permute.default(primals_871, [1, 0]);  primals_871 = None
    addmm_239: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_872, view_638, permute_319);  primals_872 = None
    view_639: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_239, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_239: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_639, add_238);  view_639 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_128: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_239, primals_255);  add_239 = None
    add_240: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_128, primals_256);  mul_128 = primals_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_640: "f32[128, 128]" = torch.ops.aten.view.default(add_240, [128, 128]);  add_240 = None
    permute_320: "f32[128, 512]" = torch.ops.aten.permute.default(primals_873, [1, 0]);  primals_873 = None
    addmm_240: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_874, view_640, permute_320);  primals_874 = None
    view_641: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_240, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_32: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_641);  view_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_241: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_32, add_227);  clone_32 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_129: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_241, primals_257);  add_241 = None
    add_242: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_129, primals_258);  mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_642: "f32[128, 512]" = torch.ops.aten.view.default(add_242, [128, 512])
    permute_321: "f32[512, 128]" = torch.ops.aten.permute.default(primals_875, [1, 0]);  primals_875 = None
    addmm_241: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_876, view_642, permute_321);  primals_876 = None
    view_643: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_241, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_130: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_643, primals_259);  view_643 = None
    add_243: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_130, primals_260);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_322: "f32[512, 128]" = torch.ops.aten.permute.default(primals_877, [1, 0]);  primals_877 = None
    addmm_242: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_878, view_642, permute_322);  primals_878 = None
    view_645: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_242, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_131: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_645, primals_261);  view_645 = None
    add_244: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_131, primals_262);  mul_131 = primals_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_646: "f32[128, 128]" = torch.ops.aten.view.default(add_244, [128, 128]);  add_244 = None
    permute_323: "f32[128, 128]" = torch.ops.aten.permute.default(primals_879, [1, 0]);  primals_879 = None
    addmm_243: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_880, view_646, permute_323);  primals_880 = None
    view_647: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_243, [1, 128, 128]);  addmm_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_324: "f32[128, 128]" = torch.ops.aten.permute.default(primals_881, [1, 0]);  primals_881 = None
    addmm_244: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_882, view_646, permute_324);  primals_882 = None
    view_649: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_244, [1, 128, 128]);  addmm_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_325: "f32[512, 128]" = torch.ops.aten.permute.default(primals_883, [1, 0]);  primals_883 = None
    addmm_245: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_884, view_642, permute_325);  primals_884 = None
    view_651: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_245, [1, 128, 128]);  addmm_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_652: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_647, [1, 128, 4, 32]);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_652, [0, 2, 1, 3]);  view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_653: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_649, [1, 128, 4, 32]);  view_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_327: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_653, [0, 2, 1, 3]);  view_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_654: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_651, [1, 128, 4, 32]);  view_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_328: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    
    # No stacktrace found for following nodes
    clone_default_21: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    clone_default_22: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
    clone_default_23: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_21, clone_default_22, clone_default_23, None, True, 0.1, scale = 0.17677669529663687)
    getitem_99: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_7[0]
    getitem_100: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_7[1]
    getitem_101: "i64[]" = _scaled_dot_product_efficient_attention_default_7[2]
    getitem_102: "i64[]" = _scaled_dot_product_efficient_attention_default_7[3];  _scaled_dot_product_efficient_attention_default_7 = None
    alias_default_14: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_99)
    alias_default_15: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_14);  alias_default_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_330: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3]);  getitem_99 = None
    clone_33: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_661: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_33, [1, 128, 128]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_662: "f32[128, 128]" = torch.ops.aten.view.default(view_661, [128, 128]);  view_661 = None
    permute_331: "f32[128, 128]" = torch.ops.aten.permute.default(primals_885, [1, 0]);  primals_885 = None
    addmm_246: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_886, view_662, permute_331);  primals_886 = None
    view_663: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_246, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_246: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_663, add_243);  view_663 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_132: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_246, primals_263);  add_246 = None
    add_247: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_132, primals_264);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_664: "f32[128, 128]" = torch.ops.aten.view.default(add_247, [128, 128])
    permute_332: "f32[128, 512]" = torch.ops.aten.permute.default(primals_887, [1, 0]);  primals_887 = None
    addmm_247: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_888, view_664, permute_332);  primals_888 = None
    view_665: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_247, [1, 128, 512]);  addmm_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_64: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_665);  view_665 = None
    alias_81: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_666: "f32[128, 512]" = torch.ops.aten.view.default(relu_64, [128, 512]);  relu_64 = None
    permute_333: "f32[512, 128]" = torch.ops.aten.permute.default(primals_889, [1, 0]);  primals_889 = None
    addmm_248: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_890, view_666, permute_333);  primals_890 = None
    view_667: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_248, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_248: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_667, add_247);  view_667 = add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_133: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_248, primals_265);  add_248 = None
    add_249: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_133, primals_266);  mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_668: "f32[128, 128]" = torch.ops.aten.view.default(add_249, [128, 128])
    permute_334: "f32[128, 512]" = torch.ops.aten.permute.default(primals_891, [1, 0]);  primals_891 = None
    addmm_249: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_892, view_668, permute_334);  primals_892 = None
    view_669: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_249, [1, 128, 512]);  addmm_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_65: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_669);  view_669 = None
    alias_82: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_670: "f32[128, 512]" = torch.ops.aten.view.default(relu_65, [128, 512]);  relu_65 = None
    permute_335: "f32[512, 128]" = torch.ops.aten.permute.default(primals_893, [1, 0]);  primals_893 = None
    addmm_250: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_894, view_670, permute_335);  primals_894 = None
    view_671: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_250, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_250: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_671, add_249);  view_671 = add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_134: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_250, primals_267);  add_250 = None
    add_251: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_134, primals_268);  mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_672: "f32[128, 128]" = torch.ops.aten.view.default(add_251, [128, 128])
    permute_336: "f32[128, 512]" = torch.ops.aten.permute.default(primals_895, [1, 0]);  primals_895 = None
    addmm_251: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_896, view_672, permute_336);  primals_896 = None
    view_673: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_251, [1, 128, 512]);  addmm_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_66: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_673);  view_673 = None
    alias_83: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_674: "f32[128, 512]" = torch.ops.aten.view.default(relu_66, [128, 512]);  relu_66 = None
    permute_337: "f32[512, 128]" = torch.ops.aten.permute.default(primals_897, [1, 0]);  primals_897 = None
    addmm_252: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_898, view_674, permute_337);  primals_898 = None
    view_675: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_252, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_252: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_675, add_251);  view_675 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_135: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_252, primals_269);  add_252 = None
    add_253: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_135, primals_270);  mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_676: "f32[128, 128]" = torch.ops.aten.view.default(add_253, [128, 128])
    permute_338: "f32[128, 512]" = torch.ops.aten.permute.default(primals_899, [1, 0]);  primals_899 = None
    addmm_253: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_900, view_676, permute_338);  primals_900 = None
    view_677: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_253, [1, 128, 512]);  addmm_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_67: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_677);  view_677 = None
    alias_84: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_678: "f32[128, 512]" = torch.ops.aten.view.default(relu_67, [128, 512]);  relu_67 = None
    permute_339: "f32[512, 128]" = torch.ops.aten.permute.default(primals_901, [1, 0]);  primals_901 = None
    addmm_254: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_902, view_678, permute_339);  primals_902 = None
    view_679: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_254, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_254: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_679, add_253);  view_679 = add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_136: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_254, primals_271);  add_254 = None
    add_255: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_136, primals_272);  mul_136 = primals_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_680: "f32[128, 128]" = torch.ops.aten.view.default(add_255, [128, 128]);  add_255 = None
    permute_340: "f32[128, 512]" = torch.ops.aten.permute.default(primals_903, [1, 0]);  primals_903 = None
    addmm_255: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_904, view_680, permute_340);  primals_904 = None
    view_681: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_255, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_34: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_681);  view_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_256: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_34, add_242);  clone_34 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_137: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_256, primals_273);  add_256 = None
    add_257: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_137, primals_274);  mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_682: "f32[128, 512]" = torch.ops.aten.view.default(add_257, [128, 512])
    permute_341: "f32[512, 128]" = torch.ops.aten.permute.default(primals_905, [1, 0]);  primals_905 = None
    addmm_256: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_906, view_682, permute_341);  primals_906 = None
    view_683: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_256, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_138: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_683, primals_275);  view_683 = None
    add_258: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_138, primals_276);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_342: "f32[512, 128]" = torch.ops.aten.permute.default(primals_907, [1, 0]);  primals_907 = None
    addmm_257: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_908, view_682, permute_342);  primals_908 = None
    view_685: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_257, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_139: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_685, primals_277);  view_685 = None
    add_259: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_139, primals_278);  mul_139 = primals_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_686: "f32[128, 128]" = torch.ops.aten.view.default(add_259, [128, 128]);  add_259 = None
    permute_343: "f32[128, 128]" = torch.ops.aten.permute.default(primals_909, [1, 0]);  primals_909 = None
    addmm_258: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_910, view_686, permute_343);  primals_910 = None
    view_687: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_258, [1, 128, 128]);  addmm_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_344: "f32[128, 128]" = torch.ops.aten.permute.default(primals_911, [1, 0]);  primals_911 = None
    addmm_259: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_912, view_686, permute_344);  primals_912 = None
    view_689: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_259, [1, 128, 128]);  addmm_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_345: "f32[512, 128]" = torch.ops.aten.permute.default(primals_913, [1, 0]);  primals_913 = None
    addmm_260: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_914, view_682, permute_345);  primals_914 = None
    view_691: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_260, [1, 128, 128]);  addmm_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_692: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_687, [1, 128, 4, 32]);  view_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_346: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_692, [0, 2, 1, 3]);  view_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_693: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_689, [1, 128, 4, 32]);  view_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_347: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_693, [0, 2, 1, 3]);  view_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_694: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_691, [1, 128, 4, 32]);  view_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_348: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None
    
    # No stacktrace found for following nodes
    clone_default_18: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_346, memory_format = torch.contiguous_format);  permute_346 = None
    clone_default_19: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
    clone_default_20: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
    _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_18, clone_default_19, clone_default_20, None, True, 0.1, scale = 0.17677669529663687)
    getitem_92: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_6[0]
    getitem_93: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_6[1]
    getitem_94: "i64[]" = _scaled_dot_product_efficient_attention_default_6[2]
    getitem_95: "i64[]" = _scaled_dot_product_efficient_attention_default_6[3];  _scaled_dot_product_efficient_attention_default_6 = None
    alias_default_12: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_92)
    alias_default_13: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_12);  alias_default_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_350: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_92, [0, 2, 1, 3]);  getitem_92 = None
    clone_35: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_350, memory_format = torch.contiguous_format);  permute_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_701: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_35, [1, 128, 128]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_702: "f32[128, 128]" = torch.ops.aten.view.default(view_701, [128, 128]);  view_701 = None
    permute_351: "f32[128, 128]" = torch.ops.aten.permute.default(primals_915, [1, 0]);  primals_915 = None
    addmm_261: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_916, view_702, permute_351);  primals_916 = None
    view_703: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_261, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_261: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_703, add_258);  view_703 = add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_140: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_261, primals_279);  add_261 = None
    add_262: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_140, primals_280);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_704: "f32[128, 128]" = torch.ops.aten.view.default(add_262, [128, 128])
    permute_352: "f32[128, 512]" = torch.ops.aten.permute.default(primals_917, [1, 0]);  primals_917 = None
    addmm_262: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_918, view_704, permute_352);  primals_918 = None
    view_705: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_262, [1, 128, 512]);  addmm_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_68: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_705);  view_705 = None
    alias_86: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_706: "f32[128, 512]" = torch.ops.aten.view.default(relu_68, [128, 512]);  relu_68 = None
    permute_353: "f32[512, 128]" = torch.ops.aten.permute.default(primals_919, [1, 0]);  primals_919 = None
    addmm_263: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_920, view_706, permute_353);  primals_920 = None
    view_707: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_263, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_263: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_707, add_262);  view_707 = add_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_141: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_263, primals_281);  add_263 = None
    add_264: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_141, primals_282);  mul_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_708: "f32[128, 128]" = torch.ops.aten.view.default(add_264, [128, 128])
    permute_354: "f32[128, 512]" = torch.ops.aten.permute.default(primals_921, [1, 0]);  primals_921 = None
    addmm_264: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_922, view_708, permute_354);  primals_922 = None
    view_709: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_264, [1, 128, 512]);  addmm_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_69: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_709);  view_709 = None
    alias_87: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_69)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_710: "f32[128, 512]" = torch.ops.aten.view.default(relu_69, [128, 512]);  relu_69 = None
    permute_355: "f32[512, 128]" = torch.ops.aten.permute.default(primals_923, [1, 0]);  primals_923 = None
    addmm_265: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_924, view_710, permute_355);  primals_924 = None
    view_711: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_265, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_265: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_711, add_264);  view_711 = add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_142: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_265, primals_283);  add_265 = None
    add_266: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_142, primals_284);  mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_712: "f32[128, 128]" = torch.ops.aten.view.default(add_266, [128, 128])
    permute_356: "f32[128, 512]" = torch.ops.aten.permute.default(primals_925, [1, 0]);  primals_925 = None
    addmm_266: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_926, view_712, permute_356);  primals_926 = None
    view_713: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_266, [1, 128, 512]);  addmm_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_70: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_713);  view_713 = None
    alias_88: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_70)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_714: "f32[128, 512]" = torch.ops.aten.view.default(relu_70, [128, 512]);  relu_70 = None
    permute_357: "f32[512, 128]" = torch.ops.aten.permute.default(primals_927, [1, 0]);  primals_927 = None
    addmm_267: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_928, view_714, permute_357);  primals_928 = None
    view_715: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_267, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_267: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_715, add_266);  view_715 = add_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_143: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_267, primals_285);  add_267 = None
    add_268: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_143, primals_286);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_716: "f32[128, 128]" = torch.ops.aten.view.default(add_268, [128, 128])
    permute_358: "f32[128, 512]" = torch.ops.aten.permute.default(primals_929, [1, 0]);  primals_929 = None
    addmm_268: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_930, view_716, permute_358);  primals_930 = None
    view_717: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_268, [1, 128, 512]);  addmm_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_71: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_717);  view_717 = None
    alias_89: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_718: "f32[128, 512]" = torch.ops.aten.view.default(relu_71, [128, 512]);  relu_71 = None
    permute_359: "f32[512, 128]" = torch.ops.aten.permute.default(primals_931, [1, 0]);  primals_931 = None
    addmm_269: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_932, view_718, permute_359);  primals_932 = None
    view_719: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_269, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_269: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_719, add_268);  view_719 = add_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_144: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_269, primals_287);  add_269 = None
    add_270: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_144, primals_288);  mul_144 = primals_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_720: "f32[128, 128]" = torch.ops.aten.view.default(add_270, [128, 128]);  add_270 = None
    permute_360: "f32[128, 512]" = torch.ops.aten.permute.default(primals_933, [1, 0]);  primals_933 = None
    addmm_270: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_934, view_720, permute_360);  primals_934 = None
    view_721: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_270, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_36: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_721);  view_721 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_271: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_36, add_257);  clone_36 = add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_145: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_271, primals_289);  add_271 = None
    add_272: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_145, primals_290);  mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_722: "f32[128, 512]" = torch.ops.aten.view.default(add_272, [128, 512])
    permute_361: "f32[512, 128]" = torch.ops.aten.permute.default(primals_935, [1, 0]);  primals_935 = None
    addmm_271: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_936, view_722, permute_361);  primals_936 = None
    view_723: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_271, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_146: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_723, primals_291);  view_723 = None
    add_273: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_146, primals_292);  mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_362: "f32[512, 128]" = torch.ops.aten.permute.default(primals_937, [1, 0]);  primals_937 = None
    addmm_272: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_938, view_722, permute_362);  primals_938 = None
    view_725: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_272, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_147: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_725, primals_293);  view_725 = None
    add_274: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_147, primals_294);  mul_147 = primals_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_726: "f32[128, 128]" = torch.ops.aten.view.default(add_274, [128, 128]);  add_274 = None
    permute_363: "f32[128, 128]" = torch.ops.aten.permute.default(primals_939, [1, 0]);  primals_939 = None
    addmm_273: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_940, view_726, permute_363);  primals_940 = None
    view_727: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_273, [1, 128, 128]);  addmm_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_364: "f32[128, 128]" = torch.ops.aten.permute.default(primals_941, [1, 0]);  primals_941 = None
    addmm_274: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_942, view_726, permute_364);  primals_942 = None
    view_729: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_274, [1, 128, 128]);  addmm_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_365: "f32[512, 128]" = torch.ops.aten.permute.default(primals_943, [1, 0]);  primals_943 = None
    addmm_275: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_944, view_722, permute_365);  primals_944 = None
    view_731: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_275, [1, 128, 128]);  addmm_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_732: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_727, [1, 128, 4, 32]);  view_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_366: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_733: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_729, [1, 128, 4, 32]);  view_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_367: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_733, [0, 2, 1, 3]);  view_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_734: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_731, [1, 128, 4, 32]);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_368: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_734, [0, 2, 1, 3]);  view_734 = None
    
    # No stacktrace found for following nodes
    clone_default_15: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
    clone_default_16: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_367, memory_format = torch.contiguous_format);  permute_367 = None
    clone_default_17: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
    _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_15, clone_default_16, clone_default_17, None, True, 0.1, scale = 0.17677669529663687)
    getitem_85: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_5[0]
    getitem_86: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_5[1]
    getitem_87: "i64[]" = _scaled_dot_product_efficient_attention_default_5[2]
    getitem_88: "i64[]" = _scaled_dot_product_efficient_attention_default_5[3];  _scaled_dot_product_efficient_attention_default_5 = None
    alias_default_10: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_85)
    alias_default_11: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_10);  alias_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_370: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3]);  getitem_85 = None
    clone_37: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_741: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_37, [1, 128, 128]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_742: "f32[128, 128]" = torch.ops.aten.view.default(view_741, [128, 128]);  view_741 = None
    permute_371: "f32[128, 128]" = torch.ops.aten.permute.default(primals_945, [1, 0]);  primals_945 = None
    addmm_276: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_946, view_742, permute_371);  primals_946 = None
    view_743: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_276, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_276: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_743, add_273);  view_743 = add_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_148: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_276, primals_295);  add_276 = None
    add_277: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_148, primals_296);  mul_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_744: "f32[128, 128]" = torch.ops.aten.view.default(add_277, [128, 128])
    permute_372: "f32[128, 512]" = torch.ops.aten.permute.default(primals_947, [1, 0]);  primals_947 = None
    addmm_277: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_948, view_744, permute_372);  primals_948 = None
    view_745: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_277, [1, 128, 512]);  addmm_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_72: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_745);  view_745 = None
    alias_91: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_746: "f32[128, 512]" = torch.ops.aten.view.default(relu_72, [128, 512]);  relu_72 = None
    permute_373: "f32[512, 128]" = torch.ops.aten.permute.default(primals_949, [1, 0]);  primals_949 = None
    addmm_278: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_950, view_746, permute_373);  primals_950 = None
    view_747: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_278, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_278: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_747, add_277);  view_747 = add_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_149: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_278, primals_297);  add_278 = None
    add_279: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_149, primals_298);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_748: "f32[128, 128]" = torch.ops.aten.view.default(add_279, [128, 128])
    permute_374: "f32[128, 512]" = torch.ops.aten.permute.default(primals_951, [1, 0]);  primals_951 = None
    addmm_279: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_952, view_748, permute_374);  primals_952 = None
    view_749: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_279, [1, 128, 512]);  addmm_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_73: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_749);  view_749 = None
    alias_92: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_73)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_750: "f32[128, 512]" = torch.ops.aten.view.default(relu_73, [128, 512]);  relu_73 = None
    permute_375: "f32[512, 128]" = torch.ops.aten.permute.default(primals_953, [1, 0]);  primals_953 = None
    addmm_280: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_954, view_750, permute_375);  primals_954 = None
    view_751: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_280, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_280: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_751, add_279);  view_751 = add_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_150: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_280, primals_299);  add_280 = None
    add_281: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_150, primals_300);  mul_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_752: "f32[128, 128]" = torch.ops.aten.view.default(add_281, [128, 128])
    permute_376: "f32[128, 512]" = torch.ops.aten.permute.default(primals_955, [1, 0]);  primals_955 = None
    addmm_281: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_956, view_752, permute_376);  primals_956 = None
    view_753: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_281, [1, 128, 512]);  addmm_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_74: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_753);  view_753 = None
    alias_93: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_74)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_754: "f32[128, 512]" = torch.ops.aten.view.default(relu_74, [128, 512]);  relu_74 = None
    permute_377: "f32[512, 128]" = torch.ops.aten.permute.default(primals_957, [1, 0]);  primals_957 = None
    addmm_282: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_958, view_754, permute_377);  primals_958 = None
    view_755: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_282, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_282: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_755, add_281);  view_755 = add_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_151: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_282, primals_301);  add_282 = None
    add_283: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_151, primals_302);  mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_756: "f32[128, 128]" = torch.ops.aten.view.default(add_283, [128, 128])
    permute_378: "f32[128, 512]" = torch.ops.aten.permute.default(primals_959, [1, 0]);  primals_959 = None
    addmm_283: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_960, view_756, permute_378);  primals_960 = None
    view_757: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_283, [1, 128, 512]);  addmm_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_75: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_757);  view_757 = None
    alias_94: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_75)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_758: "f32[128, 512]" = torch.ops.aten.view.default(relu_75, [128, 512]);  relu_75 = None
    permute_379: "f32[512, 128]" = torch.ops.aten.permute.default(primals_961, [1, 0]);  primals_961 = None
    addmm_284: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_962, view_758, permute_379);  primals_962 = None
    view_759: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_284, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_284: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_759, add_283);  view_759 = add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_152: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_284, primals_303);  add_284 = None
    add_285: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_152, primals_304);  mul_152 = primals_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_760: "f32[128, 128]" = torch.ops.aten.view.default(add_285, [128, 128]);  add_285 = None
    permute_380: "f32[128, 512]" = torch.ops.aten.permute.default(primals_963, [1, 0]);  primals_963 = None
    addmm_285: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_964, view_760, permute_380);  primals_964 = None
    view_761: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_285, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_38: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_761);  view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_286: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_38, add_272);  clone_38 = add_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_153: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_286, primals_305);  add_286 = None
    add_287: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_153, primals_306);  mul_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_762: "f32[128, 512]" = torch.ops.aten.view.default(add_287, [128, 512])
    permute_381: "f32[512, 128]" = torch.ops.aten.permute.default(primals_965, [1, 0]);  primals_965 = None
    addmm_286: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_966, view_762, permute_381);  primals_966 = None
    view_763: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_286, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_154: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_763, primals_307);  view_763 = None
    add_288: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_154, primals_308);  mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_382: "f32[512, 128]" = torch.ops.aten.permute.default(primals_967, [1, 0]);  primals_967 = None
    addmm_287: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_968, view_762, permute_382);  primals_968 = None
    view_765: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_287, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_155: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_765, primals_309);  view_765 = None
    add_289: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_155, primals_310);  mul_155 = primals_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_766: "f32[128, 128]" = torch.ops.aten.view.default(add_289, [128, 128]);  add_289 = None
    permute_383: "f32[128, 128]" = torch.ops.aten.permute.default(primals_969, [1, 0]);  primals_969 = None
    addmm_288: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_970, view_766, permute_383);  primals_970 = None
    view_767: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_288, [1, 128, 128]);  addmm_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_384: "f32[128, 128]" = torch.ops.aten.permute.default(primals_971, [1, 0]);  primals_971 = None
    addmm_289: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_972, view_766, permute_384);  primals_972 = None
    view_769: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_289, [1, 128, 128]);  addmm_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_385: "f32[512, 128]" = torch.ops.aten.permute.default(primals_973, [1, 0]);  primals_973 = None
    addmm_290: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_974, view_762, permute_385);  primals_974 = None
    view_771: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_290, [1, 128, 128]);  addmm_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_772: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_767, [1, 128, 4, 32]);  view_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_386: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_772, [0, 2, 1, 3]);  view_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_773: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_769, [1, 128, 4, 32]);  view_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_387: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_773, [0, 2, 1, 3]);  view_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_774: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_771, [1, 128, 4, 32]);  view_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_388: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None
    
    # No stacktrace found for following nodes
    clone_default_12: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
    clone_default_13: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    clone_default_14: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
    _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_12, clone_default_13, clone_default_14, None, True, 0.1, scale = 0.17677669529663687)
    getitem_78: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_4[0]
    getitem_79: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_4[1]
    getitem_80: "i64[]" = _scaled_dot_product_efficient_attention_default_4[2]
    getitem_81: "i64[]" = _scaled_dot_product_efficient_attention_default_4[3];  _scaled_dot_product_efficient_attention_default_4 = None
    alias_default_8: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_78)
    alias_default_9: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_8);  alias_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_390: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_78, [0, 2, 1, 3]);  getitem_78 = None
    clone_39: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_781: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_39, [1, 128, 128]);  clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_782: "f32[128, 128]" = torch.ops.aten.view.default(view_781, [128, 128]);  view_781 = None
    permute_391: "f32[128, 128]" = torch.ops.aten.permute.default(primals_975, [1, 0]);  primals_975 = None
    addmm_291: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_976, view_782, permute_391);  primals_976 = None
    view_783: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_291, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_291: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_783, add_288);  view_783 = add_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_156: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_291, primals_311);  add_291 = None
    add_292: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_156, primals_312);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_784: "f32[128, 128]" = torch.ops.aten.view.default(add_292, [128, 128])
    permute_392: "f32[128, 512]" = torch.ops.aten.permute.default(primals_977, [1, 0]);  primals_977 = None
    addmm_292: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_978, view_784, permute_392);  primals_978 = None
    view_785: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_292, [1, 128, 512]);  addmm_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_76: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_785);  view_785 = None
    alias_96: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_76)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_786: "f32[128, 512]" = torch.ops.aten.view.default(relu_76, [128, 512]);  relu_76 = None
    permute_393: "f32[512, 128]" = torch.ops.aten.permute.default(primals_979, [1, 0]);  primals_979 = None
    addmm_293: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_980, view_786, permute_393);  primals_980 = None
    view_787: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_293, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_293: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_787, add_292);  view_787 = add_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_157: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_293, primals_313);  add_293 = None
    add_294: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_157, primals_314);  mul_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_788: "f32[128, 128]" = torch.ops.aten.view.default(add_294, [128, 128])
    permute_394: "f32[128, 512]" = torch.ops.aten.permute.default(primals_981, [1, 0]);  primals_981 = None
    addmm_294: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_982, view_788, permute_394);  primals_982 = None
    view_789: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_294, [1, 128, 512]);  addmm_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_77: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_789);  view_789 = None
    alias_97: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_77)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_790: "f32[128, 512]" = torch.ops.aten.view.default(relu_77, [128, 512]);  relu_77 = None
    permute_395: "f32[512, 128]" = torch.ops.aten.permute.default(primals_983, [1, 0]);  primals_983 = None
    addmm_295: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_984, view_790, permute_395);  primals_984 = None
    view_791: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_295, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_295: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_791, add_294);  view_791 = add_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_158: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_295, primals_315);  add_295 = None
    add_296: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_158, primals_316);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_792: "f32[128, 128]" = torch.ops.aten.view.default(add_296, [128, 128])
    permute_396: "f32[128, 512]" = torch.ops.aten.permute.default(primals_985, [1, 0]);  primals_985 = None
    addmm_296: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_986, view_792, permute_396);  primals_986 = None
    view_793: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_296, [1, 128, 512]);  addmm_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_78: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_793);  view_793 = None
    alias_98: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_78)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_794: "f32[128, 512]" = torch.ops.aten.view.default(relu_78, [128, 512]);  relu_78 = None
    permute_397: "f32[512, 128]" = torch.ops.aten.permute.default(primals_987, [1, 0]);  primals_987 = None
    addmm_297: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_988, view_794, permute_397);  primals_988 = None
    view_795: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_297, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_297: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_795, add_296);  view_795 = add_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_159: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_297, primals_317);  add_297 = None
    add_298: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_159, primals_318);  mul_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_796: "f32[128, 128]" = torch.ops.aten.view.default(add_298, [128, 128])
    permute_398: "f32[128, 512]" = torch.ops.aten.permute.default(primals_989, [1, 0]);  primals_989 = None
    addmm_298: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_990, view_796, permute_398);  primals_990 = None
    view_797: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_298, [1, 128, 512]);  addmm_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_79: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_797);  view_797 = None
    alias_99: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_798: "f32[128, 512]" = torch.ops.aten.view.default(relu_79, [128, 512]);  relu_79 = None
    permute_399: "f32[512, 128]" = torch.ops.aten.permute.default(primals_991, [1, 0]);  primals_991 = None
    addmm_299: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_992, view_798, permute_399);  primals_992 = None
    view_799: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_299, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_299: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_799, add_298);  view_799 = add_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_160: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_299, primals_319);  add_299 = None
    add_300: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_160, primals_320);  mul_160 = primals_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_800: "f32[128, 128]" = torch.ops.aten.view.default(add_300, [128, 128]);  add_300 = None
    permute_400: "f32[128, 512]" = torch.ops.aten.permute.default(primals_993, [1, 0]);  primals_993 = None
    addmm_300: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_994, view_800, permute_400);  primals_994 = None
    view_801: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_300, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_40: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_801);  view_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_301: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_40, add_287);  clone_40 = add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_161: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_301, primals_321);  add_301 = None
    add_302: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_161, primals_322);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_802: "f32[128, 512]" = torch.ops.aten.view.default(add_302, [128, 512])
    permute_401: "f32[512, 128]" = torch.ops.aten.permute.default(primals_995, [1, 0]);  primals_995 = None
    addmm_301: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_996, view_802, permute_401);  primals_996 = None
    view_803: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_301, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_162: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_803, primals_323);  view_803 = None
    add_303: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_162, primals_324);  mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_402: "f32[512, 128]" = torch.ops.aten.permute.default(primals_997, [1, 0]);  primals_997 = None
    addmm_302: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_998, view_802, permute_402);  primals_998 = None
    view_805: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_302, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_163: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_805, primals_325);  view_805 = None
    add_304: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_163, primals_326);  mul_163 = primals_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_806: "f32[128, 128]" = torch.ops.aten.view.default(add_304, [128, 128]);  add_304 = None
    permute_403: "f32[128, 128]" = torch.ops.aten.permute.default(primals_999, [1, 0]);  primals_999 = None
    addmm_303: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1000, view_806, permute_403);  primals_1000 = None
    view_807: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_303, [1, 128, 128]);  addmm_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_404: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1001, [1, 0]);  primals_1001 = None
    addmm_304: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1002, view_806, permute_404);  primals_1002 = None
    view_809: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_304, [1, 128, 128]);  addmm_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_405: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1003, [1, 0]);  primals_1003 = None
    addmm_305: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1004, view_802, permute_405);  primals_1004 = None
    view_811: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_305, [1, 128, 128]);  addmm_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_812: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_807, [1, 128, 4, 32]);  view_807 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_406: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_812, [0, 2, 1, 3]);  view_812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_813: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_809, [1, 128, 4, 32]);  view_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_407: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_813, [0, 2, 1, 3]);  view_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_814: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_811, [1, 128, 4, 32]);  view_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_408: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_814, [0, 2, 1, 3]);  view_814 = None
    
    # No stacktrace found for following nodes
    clone_default_9: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_406, memory_format = torch.contiguous_format);  permute_406 = None
    clone_default_10: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
    clone_default_11: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_408, memory_format = torch.contiguous_format);  permute_408 = None
    _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_9, clone_default_10, clone_default_11, None, True, 0.1, scale = 0.17677669529663687)
    getitem_71: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_3[0]
    getitem_72: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_3[1]
    getitem_73: "i64[]" = _scaled_dot_product_efficient_attention_default_3[2]
    getitem_74: "i64[]" = _scaled_dot_product_efficient_attention_default_3[3];  _scaled_dot_product_efficient_attention_default_3 = None
    alias_default_6: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_71)
    alias_default_7: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_6);  alias_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_410: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
    clone_41: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_821: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_41, [1, 128, 128]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_822: "f32[128, 128]" = torch.ops.aten.view.default(view_821, [128, 128]);  view_821 = None
    permute_411: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1005, [1, 0]);  primals_1005 = None
    addmm_306: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1006, view_822, permute_411);  primals_1006 = None
    view_823: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_306, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_306: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_823, add_303);  view_823 = add_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_164: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_306, primals_327);  add_306 = None
    add_307: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_164, primals_328);  mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_824: "f32[128, 128]" = torch.ops.aten.view.default(add_307, [128, 128])
    permute_412: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1007, [1, 0]);  primals_1007 = None
    addmm_307: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1008, view_824, permute_412);  primals_1008 = None
    view_825: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_307, [1, 128, 512]);  addmm_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_80: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_825);  view_825 = None
    alias_101: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_826: "f32[128, 512]" = torch.ops.aten.view.default(relu_80, [128, 512]);  relu_80 = None
    permute_413: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1009, [1, 0]);  primals_1009 = None
    addmm_308: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1010, view_826, permute_413);  primals_1010 = None
    view_827: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_308, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_308: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_827, add_307);  view_827 = add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_165: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_308, primals_329);  add_308 = None
    add_309: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_165, primals_330);  mul_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_828: "f32[128, 128]" = torch.ops.aten.view.default(add_309, [128, 128])
    permute_414: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1011, [1, 0]);  primals_1011 = None
    addmm_309: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1012, view_828, permute_414);  primals_1012 = None
    view_829: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_309, [1, 128, 512]);  addmm_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_81: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_829);  view_829 = None
    alias_102: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_81)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_830: "f32[128, 512]" = torch.ops.aten.view.default(relu_81, [128, 512]);  relu_81 = None
    permute_415: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1013, [1, 0]);  primals_1013 = None
    addmm_310: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1014, view_830, permute_415);  primals_1014 = None
    view_831: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_310, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_310: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_831, add_309);  view_831 = add_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_166: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_310, primals_331);  add_310 = None
    add_311: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_166, primals_332);  mul_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_832: "f32[128, 128]" = torch.ops.aten.view.default(add_311, [128, 128])
    permute_416: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1015, [1, 0]);  primals_1015 = None
    addmm_311: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1016, view_832, permute_416);  primals_1016 = None
    view_833: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_311, [1, 128, 512]);  addmm_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_82: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_833);  view_833 = None
    alias_103: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_82)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_834: "f32[128, 512]" = torch.ops.aten.view.default(relu_82, [128, 512]);  relu_82 = None
    permute_417: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1017, [1, 0]);  primals_1017 = None
    addmm_312: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1018, view_834, permute_417);  primals_1018 = None
    view_835: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_312, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_312: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_835, add_311);  view_835 = add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_167: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_312, primals_333);  add_312 = None
    add_313: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_167, primals_334);  mul_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_836: "f32[128, 128]" = torch.ops.aten.view.default(add_313, [128, 128])
    permute_418: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1019, [1, 0]);  primals_1019 = None
    addmm_313: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1020, view_836, permute_418);  primals_1020 = None
    view_837: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_313, [1, 128, 512]);  addmm_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_83: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_837);  view_837 = None
    alias_104: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_83)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_838: "f32[128, 512]" = torch.ops.aten.view.default(relu_83, [128, 512]);  relu_83 = None
    permute_419: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1021, [1, 0]);  primals_1021 = None
    addmm_314: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1022, view_838, permute_419);  primals_1022 = None
    view_839: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_314, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_314: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_839, add_313);  view_839 = add_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_168: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_314, primals_335);  add_314 = None
    add_315: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_168, primals_336);  mul_168 = primals_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_840: "f32[128, 128]" = torch.ops.aten.view.default(add_315, [128, 128]);  add_315 = None
    permute_420: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1023, [1, 0]);  primals_1023 = None
    addmm_315: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1024, view_840, permute_420);  primals_1024 = None
    view_841: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_315, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_42: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_841);  view_841 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_316: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_42, add_302);  clone_42 = add_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_169: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_316, primals_337);  add_316 = None
    add_317: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_169, primals_338);  mul_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_842: "f32[128, 512]" = torch.ops.aten.view.default(add_317, [128, 512])
    permute_421: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1025, [1, 0]);  primals_1025 = None
    addmm_316: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1026, view_842, permute_421);  primals_1026 = None
    view_843: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_316, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_170: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_843, primals_339);  view_843 = None
    add_318: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_170, primals_340);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_422: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1027, [1, 0]);  primals_1027 = None
    addmm_317: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1028, view_842, permute_422);  primals_1028 = None
    view_845: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_317, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_171: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_845, primals_341);  view_845 = None
    add_319: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_171, primals_342);  mul_171 = primals_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_846: "f32[128, 128]" = torch.ops.aten.view.default(add_319, [128, 128]);  add_319 = None
    permute_423: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1029, [1, 0]);  primals_1029 = None
    addmm_318: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1030, view_846, permute_423);  primals_1030 = None
    view_847: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_318, [1, 128, 128]);  addmm_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_424: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1031, [1, 0]);  primals_1031 = None
    addmm_319: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1032, view_846, permute_424);  primals_1032 = None
    view_849: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_319, [1, 128, 128]);  addmm_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_425: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1033, [1, 0]);  primals_1033 = None
    addmm_320: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1034, view_842, permute_425);  primals_1034 = None
    view_851: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_320, [1, 128, 128]);  addmm_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_852: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_847, [1, 128, 4, 32]);  view_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_426: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_852, [0, 2, 1, 3]);  view_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_853: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_849, [1, 128, 4, 32]);  view_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_427: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_853, [0, 2, 1, 3]);  view_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_854: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_851, [1, 128, 4, 32]);  view_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_428: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_854, [0, 2, 1, 3]);  view_854 = None
    
    # No stacktrace found for following nodes
    clone_default_6: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    clone_default_7: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_427, memory_format = torch.contiguous_format);  permute_427 = None
    clone_default_8: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_428, memory_format = torch.contiguous_format);  permute_428 = None
    _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_6, clone_default_7, clone_default_8, None, True, 0.1, scale = 0.17677669529663687)
    getitem_64: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_2[0]
    getitem_65: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_2[1]
    getitem_66: "i64[]" = _scaled_dot_product_efficient_attention_default_2[2]
    getitem_67: "i64[]" = _scaled_dot_product_efficient_attention_default_2[3];  _scaled_dot_product_efficient_attention_default_2 = None
    alias_default_4: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_64)
    alias_default_5: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_4);  alias_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_430: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_64, [0, 2, 1, 3]);  getitem_64 = None
    clone_43: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_430, memory_format = torch.contiguous_format);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_861: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_43, [1, 128, 128]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_862: "f32[128, 128]" = torch.ops.aten.view.default(view_861, [128, 128]);  view_861 = None
    permute_431: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1035, [1, 0]);  primals_1035 = None
    addmm_321: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1036, view_862, permute_431);  primals_1036 = None
    view_863: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_321, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_321: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_863, add_318);  view_863 = add_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_172: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_321, primals_343);  add_321 = None
    add_322: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_172, primals_344);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_864: "f32[128, 128]" = torch.ops.aten.view.default(add_322, [128, 128])
    permute_432: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1037, [1, 0]);  primals_1037 = None
    addmm_322: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1038, view_864, permute_432);  primals_1038 = None
    view_865: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_322, [1, 128, 512]);  addmm_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_84: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_865);  view_865 = None
    alias_106: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_866: "f32[128, 512]" = torch.ops.aten.view.default(relu_84, [128, 512]);  relu_84 = None
    permute_433: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1039, [1, 0]);  primals_1039 = None
    addmm_323: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1040, view_866, permute_433);  primals_1040 = None
    view_867: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_323, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_323: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_867, add_322);  view_867 = add_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_173: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_323, primals_345);  add_323 = None
    add_324: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_173, primals_346);  mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_868: "f32[128, 128]" = torch.ops.aten.view.default(add_324, [128, 128])
    permute_434: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1041, [1, 0]);  primals_1041 = None
    addmm_324: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1042, view_868, permute_434);  primals_1042 = None
    view_869: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_324, [1, 128, 512]);  addmm_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_85: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_869);  view_869 = None
    alias_107: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_85)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_870: "f32[128, 512]" = torch.ops.aten.view.default(relu_85, [128, 512]);  relu_85 = None
    permute_435: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1043, [1, 0]);  primals_1043 = None
    addmm_325: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1044, view_870, permute_435);  primals_1044 = None
    view_871: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_325, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_325: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_871, add_324);  view_871 = add_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_174: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_325, primals_347);  add_325 = None
    add_326: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_174, primals_348);  mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_872: "f32[128, 128]" = torch.ops.aten.view.default(add_326, [128, 128])
    permute_436: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1045, [1, 0]);  primals_1045 = None
    addmm_326: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1046, view_872, permute_436);  primals_1046 = None
    view_873: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_326, [1, 128, 512]);  addmm_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_86: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_873);  view_873 = None
    alias_108: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_86)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_874: "f32[128, 512]" = torch.ops.aten.view.default(relu_86, [128, 512]);  relu_86 = None
    permute_437: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1047, [1, 0]);  primals_1047 = None
    addmm_327: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1048, view_874, permute_437);  primals_1048 = None
    view_875: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_327, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_327: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_875, add_326);  view_875 = add_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_175: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_327, primals_349);  add_327 = None
    add_328: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_175, primals_350);  mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_876: "f32[128, 128]" = torch.ops.aten.view.default(add_328, [128, 128])
    permute_438: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1049, [1, 0]);  primals_1049 = None
    addmm_328: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1050, view_876, permute_438);  primals_1050 = None
    view_877: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_328, [1, 128, 512]);  addmm_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_87: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_877);  view_877 = None
    alias_109: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_87)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_878: "f32[128, 512]" = torch.ops.aten.view.default(relu_87, [128, 512]);  relu_87 = None
    permute_439: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1051, [1, 0]);  primals_1051 = None
    addmm_329: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1052, view_878, permute_439);  primals_1052 = None
    view_879: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_329, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_329: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_879, add_328);  view_879 = add_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_176: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_329, primals_351);  add_329 = None
    add_330: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_176, primals_352);  mul_176 = primals_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_880: "f32[128, 128]" = torch.ops.aten.view.default(add_330, [128, 128]);  add_330 = None
    permute_440: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1053, [1, 0]);  primals_1053 = None
    addmm_330: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1054, view_880, permute_440);  primals_1054 = None
    view_881: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_330, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_44: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_881);  view_881 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_331: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_44, add_317);  clone_44 = add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_177: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_331, primals_353);  add_331 = None
    add_332: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_177, primals_354);  mul_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_882: "f32[128, 512]" = torch.ops.aten.view.default(add_332, [128, 512])
    permute_441: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1055, [1, 0]);  primals_1055 = None
    addmm_331: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1056, view_882, permute_441);  primals_1056 = None
    view_883: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_331, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_178: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_883, primals_355);  view_883 = None
    add_333: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_178, primals_356);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_442: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1057, [1, 0]);  primals_1057 = None
    addmm_332: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1058, view_882, permute_442);  primals_1058 = None
    view_885: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_332, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_179: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_885, primals_357);  view_885 = None
    add_334: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_179, primals_358);  mul_179 = primals_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_886: "f32[128, 128]" = torch.ops.aten.view.default(add_334, [128, 128]);  add_334 = None
    permute_443: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1059, [1, 0]);  primals_1059 = None
    addmm_333: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1060, view_886, permute_443);  primals_1060 = None
    view_887: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_333, [1, 128, 128]);  addmm_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_444: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1061, [1, 0]);  primals_1061 = None
    addmm_334: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1062, view_886, permute_444);  primals_1062 = None
    view_889: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_334, [1, 128, 128]);  addmm_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_445: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1063, [1, 0]);  primals_1063 = None
    addmm_335: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1064, view_882, permute_445);  primals_1064 = None
    view_891: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_335, [1, 128, 128]);  addmm_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_892: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_887, [1, 128, 4, 32]);  view_887 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_446: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_892, [0, 2, 1, 3]);  view_892 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_893: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_889, [1, 128, 4, 32]);  view_889 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_447: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_893, [0, 2, 1, 3]);  view_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_894: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_891, [1, 128, 4, 32]);  view_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_448: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_894, [0, 2, 1, 3]);  view_894 = None
    
    # No stacktrace found for following nodes
    clone_default_3: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_446, memory_format = torch.contiguous_format);  permute_446 = None
    clone_default_4: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
    clone_default_5: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_448, memory_format = torch.contiguous_format);  permute_448 = None
    _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default_3, clone_default_4, clone_default_5, None, True, 0.1, scale = 0.17677669529663687)
    getitem_57: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default_1[0]
    getitem_58: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default_1[1]
    getitem_59: "i64[]" = _scaled_dot_product_efficient_attention_default_1[2]
    getitem_60: "i64[]" = _scaled_dot_product_efficient_attention_default_1[3];  _scaled_dot_product_efficient_attention_default_1 = None
    alias_default_2: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_57)
    alias_default_3: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default_2);  alias_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_450: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_57, [0, 2, 1, 3]);  getitem_57 = None
    clone_45: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_450, memory_format = torch.contiguous_format);  permute_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_901: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_45, [1, 128, 128]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_902: "f32[128, 128]" = torch.ops.aten.view.default(view_901, [128, 128]);  view_901 = None
    permute_451: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1065, [1, 0]);  primals_1065 = None
    addmm_336: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1066, view_902, permute_451);  primals_1066 = None
    view_903: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_336, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_336: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_903, add_333);  view_903 = add_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_180: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_336, primals_359);  add_336 = None
    add_337: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_180, primals_360);  mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_904: "f32[128, 128]" = torch.ops.aten.view.default(add_337, [128, 128])
    permute_452: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1067, [1, 0]);  primals_1067 = None
    addmm_337: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1068, view_904, permute_452);  primals_1068 = None
    view_905: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_337, [1, 128, 512]);  addmm_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_88: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_905);  view_905 = None
    alias_111: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_88)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_906: "f32[128, 512]" = torch.ops.aten.view.default(relu_88, [128, 512]);  relu_88 = None
    permute_453: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1069, [1, 0]);  primals_1069 = None
    addmm_338: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1070, view_906, permute_453);  primals_1070 = None
    view_907: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_338, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_338: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_907, add_337);  view_907 = add_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_181: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_338, primals_361);  add_338 = None
    add_339: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_181, primals_362);  mul_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_908: "f32[128, 128]" = torch.ops.aten.view.default(add_339, [128, 128])
    permute_454: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1071, [1, 0]);  primals_1071 = None
    addmm_339: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1072, view_908, permute_454);  primals_1072 = None
    view_909: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_339, [1, 128, 512]);  addmm_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_89: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_909);  view_909 = None
    alias_112: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_89)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_910: "f32[128, 512]" = torch.ops.aten.view.default(relu_89, [128, 512]);  relu_89 = None
    permute_455: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1073, [1, 0]);  primals_1073 = None
    addmm_340: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1074, view_910, permute_455);  primals_1074 = None
    view_911: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_340, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_340: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_911, add_339);  view_911 = add_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_182: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_340, primals_363);  add_340 = None
    add_341: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_182, primals_364);  mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_912: "f32[128, 128]" = torch.ops.aten.view.default(add_341, [128, 128])
    permute_456: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1075, [1, 0]);  primals_1075 = None
    addmm_341: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1076, view_912, permute_456);  primals_1076 = None
    view_913: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_341, [1, 128, 512]);  addmm_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_90: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_913);  view_913 = None
    alias_113: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_914: "f32[128, 512]" = torch.ops.aten.view.default(relu_90, [128, 512]);  relu_90 = None
    permute_457: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1077, [1, 0]);  primals_1077 = None
    addmm_342: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1078, view_914, permute_457);  primals_1078 = None
    view_915: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_342, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_342: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_915, add_341);  view_915 = add_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_183: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_342, primals_365);  add_342 = None
    add_343: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_183, primals_366);  mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_916: "f32[128, 128]" = torch.ops.aten.view.default(add_343, [128, 128])
    permute_458: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1079, [1, 0]);  primals_1079 = None
    addmm_343: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1080, view_916, permute_458);  primals_1080 = None
    view_917: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_343, [1, 128, 512]);  addmm_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_91: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_917);  view_917 = None
    alias_114: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_91)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_918: "f32[128, 512]" = torch.ops.aten.view.default(relu_91, [128, 512]);  relu_91 = None
    permute_459: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1081, [1, 0]);  primals_1081 = None
    addmm_344: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1082, view_918, permute_459);  primals_1082 = None
    view_919: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_344, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_344: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_919, add_343);  view_919 = add_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_184: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_344, primals_367);  add_344 = None
    add_345: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_184, primals_368);  mul_184 = primals_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_920: "f32[128, 128]" = torch.ops.aten.view.default(add_345, [128, 128]);  add_345 = None
    permute_460: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1083, [1, 0]);  primals_1083 = None
    addmm_345: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1084, view_920, permute_460);  primals_1084 = None
    view_921: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_345, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_46: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_921);  view_921 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_346: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_46, add_332);  clone_46 = add_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_185: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_346, primals_369);  add_346 = None
    add_347: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_185, primals_370);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    view_922: "f32[128, 512]" = torch.ops.aten.view.default(add_347, [128, 512])
    permute_461: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1085, [1, 0]);  primals_1085 = None
    addmm_346: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1086, view_922, permute_461);  primals_1086 = None
    view_923: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_346, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_186: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_923, primals_371);  view_923 = None
    add_348: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_186, primals_372);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_462: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1087, [1, 0]);  primals_1087 = None
    addmm_347: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1088, view_922, permute_462);  primals_1088 = None
    view_925: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_347, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_187: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(view_925, primals_373);  view_925 = None
    add_349: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_187, primals_374);  mul_187 = primals_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    view_926: "f32[128, 128]" = torch.ops.aten.view.default(add_349, [128, 128]);  add_349 = None
    permute_463: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1089, [1, 0]);  primals_1089 = None
    addmm_348: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1090, view_926, permute_463);  primals_1090 = None
    view_927: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_348, [1, 128, 128]);  addmm_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_464: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1091, [1, 0]);  primals_1091 = None
    addmm_349: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1092, view_926, permute_464);  primals_1092 = None
    view_929: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_349, [1, 128, 128]);  addmm_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_465: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1093, [1, 0]);  primals_1093 = None
    addmm_350: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1094, view_922, permute_465);  primals_1094 = None
    view_931: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_350, [1, 128, 128]);  addmm_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_932: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_927, [1, 128, 4, 32]);  view_927 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_466: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_932, [0, 2, 1, 3]);  view_932 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_933: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_929, [1, 128, 4, 32]);  view_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_467: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_933, [0, 2, 1, 3]);  view_933 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:265, code: x = x.view(new_x_shape)
    view_934: "f32[1, 128, 4, 32]" = torch.ops.aten.view.default(view_931, [1, 128, 4, 32]);  view_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:266, code: return x.permute(0, 2, 1, 3)
    permute_468: "f32[1, 4, 128, 32]" = torch.ops.aten.permute.default(view_934, [0, 2, 1, 3]);  view_934 = None
    
    # No stacktrace found for following nodes
    clone_default: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_466, memory_format = torch.contiguous_format);  permute_466 = None
    clone_default_1: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
    clone_default_2: "f32[1, 4, 128, 32]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
    _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(clone_default, clone_default_1, clone_default_2, None, True, 0.1, scale = 0.17677669529663687)
    getitem_50: "f32[1, 4, 128, 32]" = _scaled_dot_product_efficient_attention_default[0]
    getitem_51: "f32[1, 4, 128]" = _scaled_dot_product_efficient_attention_default[1]
    getitem_52: "i64[]" = _scaled_dot_product_efficient_attention_default[2]
    getitem_53: "i64[]" = _scaled_dot_product_efficient_attention_default[3];  _scaled_dot_product_efficient_attention_default = None
    alias_default: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(getitem_50)
    alias_default_1: "f32[1, 4, 128, 32]" = torch.ops.aten.alias.default(alias_default);  alias_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:300, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_470: "f32[1, 128, 4, 32]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3]);  getitem_50 = None
    clone_47: "f32[1, 128, 4, 32]" = torch.ops.aten.clone.default(permute_470, memory_format = torch.contiguous_format);  permute_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:302, code: context_layer = context_layer.view(new_context_layer_shape)
    view_941: "f32[1, 128, 128]" = torch.ops.aten.view.default(clone_47, [1, 128, 128]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    view_942: "f32[128, 128]" = torch.ops.aten.view.default(view_941, [128, 128]);  view_941 = None
    permute_471: "f32[128, 128]" = torch.ops.aten.permute.default(primals_1095, [1, 0]);  primals_1095 = None
    addmm_351: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1096, view_942, permute_471);  primals_1096 = None
    view_943: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_351, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:320, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_351: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_943, add_348);  view_943 = add_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_188: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_351, primals_375);  add_351 = None
    add_352: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_188, primals_376);  mul_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_944: "f32[128, 128]" = torch.ops.aten.view.default(add_352, [128, 128])
    permute_472: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1097, [1, 0]);  primals_1097 = None
    addmm_352: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1098, view_944, permute_472);  primals_1098 = None
    view_945: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_352, [1, 128, 512]);  addmm_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_92: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_945);  view_945 = None
    alias_116: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_946: "f32[128, 512]" = torch.ops.aten.view.default(relu_92, [128, 512]);  relu_92 = None
    permute_473: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1099, [1, 0]);  primals_1099 = None
    addmm_353: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1100, view_946, permute_473);  primals_1100 = None
    view_947: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_353, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_353: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_947, add_352);  view_947 = add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_189: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_353, primals_377);  add_353 = None
    add_354: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_189, primals_378);  mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_948: "f32[128, 128]" = torch.ops.aten.view.default(add_354, [128, 128])
    permute_474: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1101, [1, 0]);  primals_1101 = None
    addmm_354: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1102, view_948, permute_474);  primals_1102 = None
    view_949: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_354, [1, 128, 512]);  addmm_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_93: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_949);  view_949 = None
    alias_117: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_93)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_950: "f32[128, 512]" = torch.ops.aten.view.default(relu_93, [128, 512]);  relu_93 = None
    permute_475: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1103, [1, 0]);  primals_1103 = None
    addmm_355: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1104, view_950, permute_475);  primals_1104 = None
    view_951: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_355, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_355: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_951, add_354);  view_951 = add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_190: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_355, primals_379);  add_355 = None
    add_356: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_190, primals_380);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_952: "f32[128, 128]" = torch.ops.aten.view.default(add_356, [128, 128])
    permute_476: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1105, [1, 0]);  primals_1105 = None
    addmm_356: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1106, view_952, permute_476);  primals_1106 = None
    view_953: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_356, [1, 128, 512]);  addmm_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_94: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_953);  view_953 = None
    alias_118: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_94)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    view_954: "f32[128, 512]" = torch.ops.aten.view.default(relu_94, [128, 512]);  relu_94 = None
    permute_477: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1107, [1, 0]);  primals_1107 = None
    addmm_357: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1108, view_954, permute_477);  primals_1108 = None
    view_955: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_357, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:483, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_357: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_955, add_356);  view_955 = add_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_191: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_357, primals_381);  add_357 = None
    add_358: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_191, primals_382);  mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    view_956: "f32[128, 128]" = torch.ops.aten.view.default(add_358, [128, 128])
    permute_478: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1109, [1, 0]);  primals_1109 = None
    addmm_358: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1110, view_956, permute_478);  primals_1110 = None
    view_957: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_358, [1, 128, 512]);  addmm_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    relu_95: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_957);  view_957 = None
    alias_119: "f32[1, 128, 512]" = torch.ops.aten.alias.default(relu_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    view_958: "f32[128, 512]" = torch.ops.aten.view.default(relu_95, [128, 512]);  relu_95 = None
    permute_479: "f32[512, 128]" = torch.ops.aten.permute.default(primals_1111, [1, 0]);  primals_1111 = None
    addmm_359: "f32[128, 128]" = torch.ops.aten.addmm.default(primals_1112, view_958, permute_479);  primals_1112 = None
    view_959: "f32[1, 128, 128]" = torch.ops.aten.view.default(addmm_359, [1, 128, 128])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:422, code: layer_output = self.LayerNorm(layer_output + residual_tensor_1)
    add_359: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(view_959, add_358);  view_959 = add_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_192: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(add_359, primals_383);  add_359 = None
    add_360: "f32[1, 128, 128]" = torch.ops.aten.add.Tensor(mul_192, primals_384);  mul_192 = primals_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    view_960: "f32[128, 128]" = torch.ops.aten.view.default(add_360, [128, 128]);  add_360 = None
    permute_480: "f32[128, 512]" = torch.ops.aten.permute.default(primals_1113, [1, 0]);  primals_1113 = None
    addmm_360: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1114, view_960, permute_480);  primals_1114 = None
    view_961: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_360, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:398, code: layer_outputs = self.dropout(layer_outputs)
    clone_48: "f32[1, 128, 512]" = torch.ops.aten.clone.default(view_961);  view_961 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:399, code: layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
    add_361: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(clone_48, add_347);  clone_48 = add_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:167, code: return input_tensor * self.weight + self.bias
    mul_193: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_361, primals_385);  add_361 = None
    add_362: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_193, primals_386);  mul_193 = primals_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:634, code: hidden_states = self.dense(hidden_states)
    view_962: "f32[128, 512]" = torch.ops.aten.view.default(add_362, [128, 512]);  add_362 = None
    permute_481: "f32[512, 512]" = torch.ops.aten.permute.default(primals_1115, [1, 0]);  primals_1115 = None
    addmm_361: "f32[128, 512]" = torch.ops.aten.addmm.default(primals_1116, view_962, permute_481);  primals_1116 = None
    view_963: "f32[1, 128, 512]" = torch.ops.aten.view.default(addmm_361, [1, 128, 512])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:635, code: hidden_states = self.transform_act_fn(hidden_states)
    relu_96: "f32[1, 128, 512]" = torch.ops.aten.relu.default(view_963);  view_963 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:636, code: hidden_states = self.LayerNorm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(relu_96, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 128, 1]" = var_mean[0]
    getitem_49: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add_363: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_363);  add_363 = None
    sub_25: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(relu_96, getitem_49);  relu_96 = None
    mul_194: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt);  sub_25 = None
    mul_195: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_194, primals_1117);  mul_194 = None
    add_364: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_195, primals_1118);  mul_195 = primals_1118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:654, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
    permute_482: "f32[128, 30522]" = torch.ops.aten.permute.default(primals_387, [1, 0]);  primals_387 = None
    cat_1: "f32[512, 30522]" = torch.ops.aten.cat.default([permute_482, primals_388]);  permute_482 = primals_388 = None
    view_964: "f32[128, 512]" = torch.ops.aten.view.default(add_364, [128, 512]);  add_364 = None
    mm: "f32[128, 30522]" = torch.ops.aten.mm.default(view_964, cat_1)
    view_965: "f32[1, 128, 30522]" = torch.ops.aten.view.default(mm, [1, 128, 30522]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:655, code: hidden_states += self.decoder.bias
    add_365: "f32[1, 128, 30522]" = torch.ops.aten.add.Tensor(view_965, primals_389);  view_965 = primals_389 = None
    view_966: "f32[128, 30522]" = torch.ops.aten.view.default(add_365, [128, 30522]);  add_365 = None
    view_967: "f32[1, 128, 30522]" = torch.ops.aten.view.default(view_966, [1, 128, 30522]);  view_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:1107, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_969: "i64[128]" = torch.ops.aten.view.default(primals_1121, [-1])
    view_971: "f32[128, 30522]" = torch.ops.aten.view.default(view_967, [-1, 30522])
    amax_24: "f32[128, 1]" = torch.ops.aten.amax.default(view_971, [1], True)
    sub_26: "f32[128, 30522]" = torch.ops.aten.sub.Tensor(view_971, amax_24);  view_971 = amax_24 = None
    exp_24: "f32[128, 30522]" = torch.ops.aten.exp.default(sub_26)
    sum_25: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[128, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_27: "f32[128, 30522]" = torch.ops.aten.sub.Tensor(sub_26, log);  sub_26 = log = None
    ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_969, -100)
    full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "i64[128]" = torch.ops.aten.where.self(ne, view_969, full_default_2);  view_969 = full_default_2 = None
    unsqueeze_2: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_27, 1, unsqueeze_2);  unsqueeze_2 = None
    squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "f32[128]" = torch.ops.aten.where.self(ne, neg, full_default_3);  neg = full_default_3 = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_1);  where_1 = None
    div_48: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:654, code: hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
    permute_483: "f32[512, 128]" = torch.ops.aten.permute.default(view_964, [1, 0]);  view_964 = None
    permute_484: "f32[30522, 512]" = torch.ops.aten.permute.default(cat_1, [1, 0]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:634, code: hidden_states = self.dense(hidden_states)
    permute_486: "f32[512, 512]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_490: "f32[512, 128]" = torch.ops.aten.permute.default(permute_480, [1, 0]);  permute_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_494: "f32[128, 512]" = torch.ops.aten.permute.default(permute_479, [1, 0]);  permute_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_124: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_119);  alias_119 = None
    le_1: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_124, 0);  alias_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_498: "f32[512, 128]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_502: "f32[128, 512]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_125: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_118);  alias_118 = None
    le_2: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_125, 0);  alias_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_506: "f32[512, 128]" = torch.ops.aten.permute.default(permute_476, [1, 0]);  permute_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_510: "f32[128, 512]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_126: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_117);  alias_117 = None
    le_3: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_126, 0);  alias_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_514: "f32[512, 128]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_518: "f32[128, 512]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_127: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_116);  alias_116 = None
    le_4: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_127, 0);  alias_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_522: "f32[512, 128]" = torch.ops.aten.permute.default(permute_472, [1, 0]);  permute_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_526: "f32[128, 128]" = torch.ops.aten.permute.default(permute_471, [1, 0]);  permute_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_539: "f32[128, 512]" = torch.ops.aten.permute.default(permute_465, [1, 0]);  permute_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_543: "f32[128, 128]" = torch.ops.aten.permute.default(permute_464, [1, 0]);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_547: "f32[128, 128]" = torch.ops.aten.permute.default(permute_463, [1, 0]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_551: "f32[128, 512]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_555: "f32[128, 512]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_559: "f32[512, 128]" = torch.ops.aten.permute.default(permute_460, [1, 0]);  permute_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_563: "f32[128, 512]" = torch.ops.aten.permute.default(permute_459, [1, 0]);  permute_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_129: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_114);  alias_114 = None
    le_5: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_129, 0);  alias_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_567: "f32[512, 128]" = torch.ops.aten.permute.default(permute_458, [1, 0]);  permute_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_571: "f32[128, 512]" = torch.ops.aten.permute.default(permute_457, [1, 0]);  permute_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_130: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_113);  alias_113 = None
    le_6: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_130, 0);  alias_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_575: "f32[512, 128]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_579: "f32[128, 512]" = torch.ops.aten.permute.default(permute_455, [1, 0]);  permute_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_131: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_112);  alias_112 = None
    le_7: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_131, 0);  alias_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_583: "f32[512, 128]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_587: "f32[128, 512]" = torch.ops.aten.permute.default(permute_453, [1, 0]);  permute_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_132: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_111);  alias_111 = None
    le_8: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_132, 0);  alias_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_591: "f32[512, 128]" = torch.ops.aten.permute.default(permute_452, [1, 0]);  permute_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_595: "f32[128, 128]" = torch.ops.aten.permute.default(permute_451, [1, 0]);  permute_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_608: "f32[128, 512]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_612: "f32[128, 128]" = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_616: "f32[128, 128]" = torch.ops.aten.permute.default(permute_443, [1, 0]);  permute_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_620: "f32[128, 512]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_624: "f32[128, 512]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_628: "f32[512, 128]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_632: "f32[128, 512]" = torch.ops.aten.permute.default(permute_439, [1, 0]);  permute_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_134: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_109);  alias_109 = None
    le_9: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_134, 0);  alias_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_636: "f32[512, 128]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_640: "f32[128, 512]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_135: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_108);  alias_108 = None
    le_10: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_135, 0);  alias_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_644: "f32[512, 128]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_648: "f32[128, 512]" = torch.ops.aten.permute.default(permute_435, [1, 0]);  permute_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_136: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_107);  alias_107 = None
    le_11: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_136, 0);  alias_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_652: "f32[512, 128]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_656: "f32[128, 512]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_137: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_106);  alias_106 = None
    le_12: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_137, 0);  alias_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_660: "f32[512, 128]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_664: "f32[128, 128]" = torch.ops.aten.permute.default(permute_431, [1, 0]);  permute_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_677: "f32[128, 512]" = torch.ops.aten.permute.default(permute_425, [1, 0]);  permute_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_681: "f32[128, 128]" = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_685: "f32[128, 128]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_689: "f32[128, 512]" = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_693: "f32[128, 512]" = torch.ops.aten.permute.default(permute_421, [1, 0]);  permute_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_697: "f32[512, 128]" = torch.ops.aten.permute.default(permute_420, [1, 0]);  permute_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_701: "f32[128, 512]" = torch.ops.aten.permute.default(permute_419, [1, 0]);  permute_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_139: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_104);  alias_104 = None
    le_13: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_139, 0);  alias_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_705: "f32[512, 128]" = torch.ops.aten.permute.default(permute_418, [1, 0]);  permute_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_709: "f32[128, 512]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_140: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_103);  alias_103 = None
    le_14: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_140, 0);  alias_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_713: "f32[512, 128]" = torch.ops.aten.permute.default(permute_416, [1, 0]);  permute_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_717: "f32[128, 512]" = torch.ops.aten.permute.default(permute_415, [1, 0]);  permute_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_141: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    le_15: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_141, 0);  alias_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_721: "f32[512, 128]" = torch.ops.aten.permute.default(permute_414, [1, 0]);  permute_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_725: "f32[128, 512]" = torch.ops.aten.permute.default(permute_413, [1, 0]);  permute_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_142: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_101);  alias_101 = None
    le_16: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_142, 0);  alias_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_729: "f32[512, 128]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_733: "f32[128, 128]" = torch.ops.aten.permute.default(permute_411, [1, 0]);  permute_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_746: "f32[128, 512]" = torch.ops.aten.permute.default(permute_405, [1, 0]);  permute_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_750: "f32[128, 128]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_754: "f32[128, 128]" = torch.ops.aten.permute.default(permute_403, [1, 0]);  permute_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_758: "f32[128, 512]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_762: "f32[128, 512]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_766: "f32[512, 128]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_770: "f32[128, 512]" = torch.ops.aten.permute.default(permute_399, [1, 0]);  permute_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_144: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    le_17: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_144, 0);  alias_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_774: "f32[512, 128]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_778: "f32[128, 512]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_145: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_98);  alias_98 = None
    le_18: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_145, 0);  alias_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_782: "f32[512, 128]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_786: "f32[128, 512]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_146: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_97);  alias_97 = None
    le_19: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_146, 0);  alias_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_790: "f32[512, 128]" = torch.ops.aten.permute.default(permute_394, [1, 0]);  permute_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_794: "f32[128, 512]" = torch.ops.aten.permute.default(permute_393, [1, 0]);  permute_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_147: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_96);  alias_96 = None
    le_20: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_147, 0);  alias_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_798: "f32[512, 128]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_802: "f32[128, 128]" = torch.ops.aten.permute.default(permute_391, [1, 0]);  permute_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_815: "f32[128, 512]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_819: "f32[128, 128]" = torch.ops.aten.permute.default(permute_384, [1, 0]);  permute_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_823: "f32[128, 128]" = torch.ops.aten.permute.default(permute_383, [1, 0]);  permute_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_827: "f32[128, 512]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_831: "f32[128, 512]" = torch.ops.aten.permute.default(permute_381, [1, 0]);  permute_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_835: "f32[512, 128]" = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_839: "f32[128, 512]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_149: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_94);  alias_94 = None
    le_21: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_149, 0);  alias_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_843: "f32[512, 128]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_847: "f32[128, 512]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_150: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_93);  alias_93 = None
    le_22: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_150, 0);  alias_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_851: "f32[512, 128]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_855: "f32[128, 512]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_151: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_92);  alias_92 = None
    le_23: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_151, 0);  alias_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_859: "f32[512, 128]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_863: "f32[128, 512]" = torch.ops.aten.permute.default(permute_373, [1, 0]);  permute_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_152: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_91);  alias_91 = None
    le_24: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_152, 0);  alias_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_867: "f32[512, 128]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_871: "f32[128, 128]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_884: "f32[128, 512]" = torch.ops.aten.permute.default(permute_365, [1, 0]);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_888: "f32[128, 128]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_892: "f32[128, 128]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_896: "f32[128, 512]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_900: "f32[128, 512]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_904: "f32[512, 128]" = torch.ops.aten.permute.default(permute_360, [1, 0]);  permute_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_908: "f32[128, 512]" = torch.ops.aten.permute.default(permute_359, [1, 0]);  permute_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_154: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_89);  alias_89 = None
    le_25: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_154, 0);  alias_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_912: "f32[512, 128]" = torch.ops.aten.permute.default(permute_358, [1, 0]);  permute_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_916: "f32[128, 512]" = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_155: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_88);  alias_88 = None
    le_26: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_155, 0);  alias_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_920: "f32[512, 128]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_924: "f32[128, 512]" = torch.ops.aten.permute.default(permute_355, [1, 0]);  permute_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_156: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_87);  alias_87 = None
    le_27: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_156, 0);  alias_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_928: "f32[512, 128]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_932: "f32[128, 512]" = torch.ops.aten.permute.default(permute_353, [1, 0]);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_157: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_86);  alias_86 = None
    le_28: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_157, 0);  alias_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_936: "f32[512, 128]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_940: "f32[128, 128]" = torch.ops.aten.permute.default(permute_351, [1, 0]);  permute_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_953: "f32[128, 512]" = torch.ops.aten.permute.default(permute_345, [1, 0]);  permute_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_957: "f32[128, 128]" = torch.ops.aten.permute.default(permute_344, [1, 0]);  permute_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_961: "f32[128, 128]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_965: "f32[128, 512]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_969: "f32[128, 512]" = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_973: "f32[512, 128]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_977: "f32[128, 512]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_159: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    le_29: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_159, 0);  alias_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_981: "f32[512, 128]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_985: "f32[128, 512]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_160: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_83);  alias_83 = None
    le_30: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_160, 0);  alias_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_989: "f32[512, 128]" = torch.ops.aten.permute.default(permute_336, [1, 0]);  permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_993: "f32[128, 512]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_161: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_82);  alias_82 = None
    le_31: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_161, 0);  alias_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_997: "f32[512, 128]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1001: "f32[128, 512]" = torch.ops.aten.permute.default(permute_333, [1, 0]);  permute_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_162: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_81);  alias_81 = None
    le_32: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_162, 0);  alias_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1005: "f32[512, 128]" = torch.ops.aten.permute.default(permute_332, [1, 0]);  permute_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1009: "f32[128, 128]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1022: "f32[128, 512]" = torch.ops.aten.permute.default(permute_325, [1, 0]);  permute_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1026: "f32[128, 128]" = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1030: "f32[128, 128]" = torch.ops.aten.permute.default(permute_323, [1, 0]);  permute_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1034: "f32[128, 512]" = torch.ops.aten.permute.default(permute_322, [1, 0]);  permute_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1038: "f32[128, 512]" = torch.ops.aten.permute.default(permute_321, [1, 0]);  permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1042: "f32[512, 128]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1046: "f32[128, 512]" = torch.ops.aten.permute.default(permute_319, [1, 0]);  permute_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_164: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_79);  alias_79 = None
    le_33: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_164, 0);  alias_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1050: "f32[512, 128]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1054: "f32[128, 512]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_165: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_78);  alias_78 = None
    le_34: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_165, 0);  alias_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1058: "f32[512, 128]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1062: "f32[128, 512]" = torch.ops.aten.permute.default(permute_315, [1, 0]);  permute_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_166: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_77);  alias_77 = None
    le_35: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_166, 0);  alias_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1066: "f32[512, 128]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1070: "f32[128, 512]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_167: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_76);  alias_76 = None
    le_36: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_167, 0);  alias_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1074: "f32[512, 128]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1078: "f32[128, 128]" = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1091: "f32[128, 512]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1095: "f32[128, 128]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1099: "f32[128, 128]" = torch.ops.aten.permute.default(permute_303, [1, 0]);  permute_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1103: "f32[128, 512]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1107: "f32[128, 512]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1111: "f32[512, 128]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1115: "f32[128, 512]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_169: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_74);  alias_74 = None
    le_37: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_169, 0);  alias_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1119: "f32[512, 128]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1123: "f32[128, 512]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_170: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_73);  alias_73 = None
    le_38: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_170, 0);  alias_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1127: "f32[512, 128]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1131: "f32[128, 512]" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_171: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    le_39: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_171, 0);  alias_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1135: "f32[512, 128]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1139: "f32[128, 512]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_172: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_71);  alias_71 = None
    le_40: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_172, 0);  alias_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1143: "f32[512, 128]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1147: "f32[128, 128]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1160: "f32[128, 512]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1164: "f32[128, 128]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1168: "f32[128, 128]" = torch.ops.aten.permute.default(permute_283, [1, 0]);  permute_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1172: "f32[128, 512]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1176: "f32[128, 512]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1180: "f32[512, 128]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1184: "f32[128, 512]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_174: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    le_41: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_174, 0);  alias_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1188: "f32[512, 128]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1192: "f32[128, 512]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_175: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_68);  alias_68 = None
    le_42: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_175, 0);  alias_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1196: "f32[512, 128]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1200: "f32[128, 512]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_176: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_67);  alias_67 = None
    le_43: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_176, 0);  alias_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1204: "f32[512, 128]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1208: "f32[128, 512]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_177: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    le_44: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_177, 0);  alias_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1212: "f32[512, 128]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1216: "f32[128, 128]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1229: "f32[128, 512]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1233: "f32[128, 128]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1237: "f32[128, 128]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1241: "f32[128, 512]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1245: "f32[128, 512]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1249: "f32[512, 128]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1253: "f32[128, 512]" = torch.ops.aten.permute.default(permute_259, [1, 0]);  permute_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_179: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_64);  alias_64 = None
    le_45: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_179, 0);  alias_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1257: "f32[512, 128]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1261: "f32[128, 512]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_180: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_63);  alias_63 = None
    le_46: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_180, 0);  alias_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1265: "f32[512, 128]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1269: "f32[128, 512]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_181: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    le_47: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_181, 0);  alias_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1273: "f32[512, 128]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1277: "f32[128, 512]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_182: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_61);  alias_61 = None
    le_48: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_182, 0);  alias_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1281: "f32[512, 128]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1285: "f32[128, 128]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1298: "f32[128, 512]" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1302: "f32[128, 128]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1306: "f32[128, 128]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1310: "f32[128, 512]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1314: "f32[128, 512]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1318: "f32[512, 128]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1322: "f32[128, 512]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_184: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    le_49: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_184, 0);  alias_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1326: "f32[512, 128]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1330: "f32[128, 512]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_185: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    le_50: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_185, 0);  alias_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1334: "f32[512, 128]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1338: "f32[128, 512]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_186: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    le_51: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_186, 0);  alias_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1342: "f32[512, 128]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1346: "f32[128, 512]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_187: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    le_52: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_187, 0);  alias_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1350: "f32[512, 128]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1354: "f32[128, 128]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1367: "f32[128, 512]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1371: "f32[128, 128]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1375: "f32[128, 128]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1379: "f32[128, 512]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1383: "f32[128, 512]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1387: "f32[512, 128]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1391: "f32[128, 512]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_189: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_53: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_189, 0);  alias_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1395: "f32[512, 128]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1399: "f32[128, 512]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_190: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_53);  alias_53 = None
    le_54: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_190, 0);  alias_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1403: "f32[512, 128]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1407: "f32[128, 512]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_191: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    le_55: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_191, 0);  alias_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1411: "f32[512, 128]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1415: "f32[128, 512]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_192: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    le_56: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_192, 0);  alias_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1419: "f32[512, 128]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1423: "f32[128, 128]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1436: "f32[128, 512]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1440: "f32[128, 128]" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1444: "f32[128, 128]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1448: "f32[128, 512]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1452: "f32[128, 512]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1456: "f32[512, 128]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1460: "f32[128, 512]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_194: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_49);  alias_49 = None
    le_57: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_194, 0);  alias_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1464: "f32[512, 128]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1468: "f32[128, 512]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_195: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_58: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_195, 0);  alias_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1472: "f32[512, 128]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1476: "f32[128, 512]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_196: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    le_59: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_196, 0);  alias_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1480: "f32[512, 128]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1484: "f32[128, 512]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_197: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    le_60: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_197, 0);  alias_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1488: "f32[512, 128]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1492: "f32[128, 128]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1505: "f32[128, 512]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1509: "f32[128, 128]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1513: "f32[128, 128]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1517: "f32[128, 512]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1521: "f32[128, 512]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1525: "f32[512, 128]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1529: "f32[128, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_199: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    le_61: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_199, 0);  alias_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1533: "f32[512, 128]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1537: "f32[128, 512]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_200: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_43);  alias_43 = None
    le_62: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_200, 0);  alias_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1541: "f32[512, 128]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1545: "f32[128, 512]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_201: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    le_63: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_201, 0);  alias_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1549: "f32[512, 128]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1553: "f32[128, 512]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_202: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    le_64: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_202, 0);  alias_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1557: "f32[512, 128]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1561: "f32[128, 128]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1574: "f32[128, 512]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1578: "f32[128, 128]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1582: "f32[128, 128]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1586: "f32[128, 512]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1590: "f32[128, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1594: "f32[512, 128]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1598: "f32[128, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_204: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_39);  alias_39 = None
    le_65: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_204, 0);  alias_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1602: "f32[512, 128]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1606: "f32[128, 512]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_205: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    le_66: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_205, 0);  alias_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1610: "f32[512, 128]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1614: "f32[128, 512]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_206: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    le_67: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_206, 0);  alias_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1618: "f32[512, 128]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1622: "f32[128, 512]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_207: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    le_68: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_207, 0);  alias_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1626: "f32[512, 128]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1630: "f32[128, 128]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1643: "f32[128, 512]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1647: "f32[128, 128]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1651: "f32[128, 128]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1655: "f32[128, 512]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1659: "f32[128, 512]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1663: "f32[512, 128]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1667: "f32[128, 512]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_209: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    le_69: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_209, 0);  alias_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1671: "f32[512, 128]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1675: "f32[128, 512]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_210: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    le_70: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_210, 0);  alias_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1679: "f32[512, 128]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1683: "f32[128, 512]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_211: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    le_71: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_211, 0);  alias_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1687: "f32[512, 128]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1691: "f32[128, 512]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_212: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    le_72: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_212, 0);  alias_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1695: "f32[512, 128]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1699: "f32[128, 128]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1712: "f32[128, 512]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1716: "f32[128, 128]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1720: "f32[128, 128]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1724: "f32[128, 512]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1728: "f32[128, 512]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1732: "f32[512, 128]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1736: "f32[128, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_214: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    le_73: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_214, 0);  alias_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1740: "f32[512, 128]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1744: "f32[128, 512]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_215: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    le_74: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_215, 0);  alias_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1748: "f32[512, 128]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1752: "f32[128, 512]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_216: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    le_75: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_216, 0);  alias_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1756: "f32[512, 128]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1760: "f32[128, 512]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_217: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    le_76: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_217, 0);  alias_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1764: "f32[512, 128]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1768: "f32[128, 128]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1781: "f32[128, 512]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1785: "f32[128, 128]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1789: "f32[128, 128]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1793: "f32[128, 512]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1797: "f32[128, 512]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1801: "f32[512, 128]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1805: "f32[128, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_219: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    le_77: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_219, 0);  alias_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1809: "f32[512, 128]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1813: "f32[128, 512]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_220: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    le_78: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_220, 0);  alias_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1817: "f32[512, 128]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1821: "f32[128, 512]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_221: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    le_79: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_221, 0);  alias_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1825: "f32[512, 128]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1829: "f32[128, 512]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_222: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    le_80: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_222, 0);  alias_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1833: "f32[512, 128]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1837: "f32[128, 128]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1850: "f32[128, 512]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1854: "f32[128, 128]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1858: "f32[128, 128]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1862: "f32[128, 512]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1866: "f32[128, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1870: "f32[512, 128]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1874: "f32[128, 512]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_224: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    le_81: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_224, 0);  alias_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1878: "f32[512, 128]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1882: "f32[128, 512]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_225: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    le_82: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_225, 0);  alias_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1886: "f32[512, 128]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1890: "f32[128, 512]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_226: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    le_83: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_226, 0);  alias_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1894: "f32[512, 128]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1898: "f32[128, 512]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_227: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    le_84: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_227, 0);  alias_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1902: "f32[512, 128]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1906: "f32[128, 128]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1919: "f32[128, 512]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1923: "f32[128, 128]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1927: "f32[128, 128]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1931: "f32[128, 512]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_1935: "f32[128, 512]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_1939: "f32[512, 128]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_1943: "f32[128, 512]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_229: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    le_85: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_229, 0);  alias_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1947: "f32[512, 128]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1951: "f32[128, 512]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_230: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    le_86: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_230, 0);  alias_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1955: "f32[512, 128]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1959: "f32[128, 512]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_231: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    le_87: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_231, 0);  alias_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1963: "f32[512, 128]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_1967: "f32[128, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_232: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    le_88: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_232, 0);  alias_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_1971: "f32[512, 128]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_1975: "f32[128, 128]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_1988: "f32[128, 512]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_1992: "f32[128, 128]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_1996: "f32[128, 128]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_2000: "f32[128, 512]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_2004: "f32[128, 512]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_2008: "f32[512, 128]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_2012: "f32[128, 512]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_234: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    le_89: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_234, 0);  alias_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_2016: "f32[512, 128]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_2020: "f32[128, 512]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_235: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    le_90: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_235, 0);  alias_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_2024: "f32[512, 128]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_2028: "f32[128, 512]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_236: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    le_91: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_236, 0);  alias_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_2032: "f32[512, 128]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_2036: "f32[128, 512]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_237: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    le_92: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_237, 0);  alias_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_2040: "f32[512, 128]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_2044: "f32[128, 128]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_2057: "f32[128, 512]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_2061: "f32[128, 128]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_2065: "f32[128, 128]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_2069: "f32[128, 512]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_2073: "f32[128, 512]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:397, code: layer_outputs = self.dense(hidden_states)
    permute_2077: "f32[512, 128]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:417, code: layer_output = self.dense(intermediate_states)
    permute_2081: "f32[128, 512]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_239: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    le_93: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_239, 0);  alias_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_2085: "f32[512, 128]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_2089: "f32[128, 512]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_240: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    le_94: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_240, 0);  alias_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_2093: "f32[512, 128]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_2097: "f32[128, 512]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_241: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    le_95: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_241, 0);  alias_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_2101: "f32[512, 128]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:482, code: layer_outputs = self.dense(hidden_states)
    permute_2105: "f32[128, 512]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:385, code: hidden_states = self.intermediate_act_fn(hidden_states)
    alias_242: "f32[1, 128, 512]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    le_96: "b8[1, 128, 512]" = torch.ops.aten.le.Scalar(alias_242, 0);  alias_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:384, code: hidden_states = self.dense(hidden_states)
    permute_2109: "f32[512, 128]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:317, code: layer_outputs = self.dense(hidden_states)
    permute_2113: "f32[128, 128]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:279, code: mixed_value_layer = self.value(value_tensor)
    permute_2126: "f32[128, 512]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:278, code: mixed_key_layer = self.key(key_tensor)
    permute_2130: "f32[128, 128]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:277, code: mixed_query_layer = self.query(query_tensor)
    permute_2134: "f32[128, 128]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_2138: "f32[128, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:434, code: layer_input = self.dense(hidden_states)
    permute_2142: "f32[128, 512]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mobilebert/modeling_mobilebert.py:237, code: inputs_embeds = self.embedding_transformation(inputs_embeds)
    permute_2146: "f32[512, 384]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [div_48, view_967, primals_1, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_33, primals_34, primals_35, primals_36, primals_37, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_49, primals_50, primals_51, primals_52, primals_53, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_81, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_101, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_113, primals_114, primals_115, primals_116, primals_117, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_129, primals_130, primals_131, primals_132, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_145, primals_146, primals_147, primals_148, primals_149, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_161, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_193, primals_194, primals_195, primals_196, primals_197, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_209, primals_210, primals_211, primals_212, primals_213, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_225, primals_226, primals_227, primals_228, primals_229, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_241, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_257, primals_258, primals_259, primals_260, primals_261, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_273, primals_274, primals_275, primals_276, primals_277, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_289, primals_290, primals_291, primals_292, primals_293, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_305, primals_306, primals_307, primals_308, primals_309, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_321, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_337, primals_338, primals_339, primals_340, primals_341, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_353, primals_354, primals_355, primals_356, primals_357, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_369, primals_370, primals_371, primals_372, primals_373, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_385, primals_1117, primals_1120, primals_1121, full_default, slice_4, view, add_1, view_2, addmm_1, addmm_2, view_6, clone_default_69, clone_default_70, clone_default_71, getitem_212, getitem_213, getitem_214, alias_default_47, view_22, addmm_6, view_24, view_26, addmm_8, view_28, view_30, addmm_10, view_32, view_34, addmm_12, view_36, view_38, addmm_14, view_40, add_16, view_42, addmm_16, addmm_17, view_46, clone_default_66, clone_default_67, clone_default_68, getitem_205, getitem_206, getitem_207, alias_default_45, view_62, addmm_21, view_64, view_66, addmm_23, view_68, view_70, addmm_25, view_72, view_74, addmm_27, view_76, view_78, addmm_29, view_80, addmm_30, view_82, addmm_31, addmm_32, view_86, clone_default_63, clone_default_64, clone_default_65, getitem_198, getitem_199, getitem_200, alias_default_43, view_102, addmm_36, view_104, view_106, addmm_38, view_108, view_110, addmm_40, view_112, view_114, addmm_42, view_116, view_118, addmm_44, view_120, addmm_45, view_122, addmm_46, addmm_47, view_126, clone_default_60, clone_default_61, clone_default_62, getitem_191, getitem_192, getitem_193, alias_default_41, view_142, addmm_51, view_144, view_146, addmm_53, view_148, view_150, addmm_55, view_152, view_154, addmm_57, view_156, view_158, addmm_59, view_160, addmm_60, view_162, addmm_61, addmm_62, view_166, clone_default_57, clone_default_58, clone_default_59, getitem_184, getitem_185, getitem_186, alias_default_39, view_182, addmm_66, view_184, view_186, addmm_68, view_188, view_190, addmm_70, view_192, view_194, addmm_72, view_196, view_198, addmm_74, view_200, addmm_75, view_202, addmm_76, addmm_77, view_206, clone_default_54, clone_default_55, clone_default_56, getitem_177, getitem_178, getitem_179, alias_default_37, view_222, addmm_81, view_224, view_226, addmm_83, view_228, view_230, addmm_85, view_232, view_234, addmm_87, view_236, view_238, addmm_89, view_240, addmm_90, view_242, addmm_91, addmm_92, view_246, clone_default_51, clone_default_52, clone_default_53, getitem_170, getitem_171, getitem_172, alias_default_35, view_262, addmm_96, view_264, view_266, addmm_98, view_268, view_270, addmm_100, view_272, view_274, addmm_102, view_276, view_278, addmm_104, view_280, addmm_105, view_282, addmm_106, addmm_107, view_286, clone_default_48, clone_default_49, clone_default_50, getitem_163, getitem_164, getitem_165, alias_default_33, view_302, addmm_111, view_304, view_306, addmm_113, view_308, view_310, addmm_115, view_312, view_314, addmm_117, view_316, view_318, addmm_119, view_320, addmm_120, view_322, addmm_121, addmm_122, view_326, clone_default_45, clone_default_46, clone_default_47, getitem_156, getitem_157, getitem_158, alias_default_31, view_342, addmm_126, view_344, view_346, addmm_128, view_348, view_350, addmm_130, view_352, view_354, addmm_132, view_356, view_358, addmm_134, view_360, addmm_135, view_362, addmm_136, addmm_137, view_366, clone_default_42, clone_default_43, clone_default_44, getitem_149, getitem_150, getitem_151, alias_default_29, view_382, addmm_141, view_384, view_386, addmm_143, view_388, view_390, addmm_145, view_392, view_394, addmm_147, view_396, view_398, addmm_149, view_400, addmm_150, view_402, addmm_151, addmm_152, view_406, clone_default_39, clone_default_40, clone_default_41, getitem_142, getitem_143, getitem_144, alias_default_27, view_422, addmm_156, view_424, view_426, addmm_158, view_428, view_430, addmm_160, view_432, view_434, addmm_162, view_436, view_438, addmm_164, view_440, addmm_165, view_442, addmm_166, addmm_167, view_446, clone_default_36, clone_default_37, clone_default_38, getitem_135, getitem_136, getitem_137, alias_default_25, view_462, addmm_171, view_464, view_466, addmm_173, view_468, view_470, addmm_175, view_472, view_474, addmm_177, view_476, view_478, addmm_179, view_480, addmm_180, view_482, addmm_181, addmm_182, view_486, clone_default_33, clone_default_34, clone_default_35, getitem_128, getitem_129, getitem_130, alias_default_23, view_502, addmm_186, view_504, view_506, addmm_188, view_508, view_510, addmm_190, view_512, view_514, addmm_192, view_516, view_518, addmm_194, view_520, addmm_195, view_522, addmm_196, addmm_197, view_526, clone_default_30, clone_default_31, clone_default_32, getitem_121, getitem_122, getitem_123, alias_default_21, view_542, addmm_201, view_544, view_546, addmm_203, view_548, view_550, addmm_205, view_552, view_554, addmm_207, view_556, view_558, addmm_209, view_560, addmm_210, view_562, addmm_211, addmm_212, view_566, clone_default_27, clone_default_28, clone_default_29, getitem_114, getitem_115, getitem_116, alias_default_19, view_582, addmm_216, view_584, view_586, addmm_218, view_588, view_590, addmm_220, view_592, view_594, addmm_222, view_596, view_598, addmm_224, view_600, addmm_225, view_602, addmm_226, addmm_227, view_606, clone_default_24, clone_default_25, clone_default_26, getitem_107, getitem_108, getitem_109, alias_default_17, view_622, addmm_231, view_624, view_626, addmm_233, view_628, view_630, addmm_235, view_632, view_634, addmm_237, view_636, view_638, addmm_239, view_640, addmm_240, view_642, addmm_241, addmm_242, view_646, clone_default_21, clone_default_22, clone_default_23, getitem_100, getitem_101, getitem_102, alias_default_15, view_662, addmm_246, view_664, view_666, addmm_248, view_668, view_670, addmm_250, view_672, view_674, addmm_252, view_676, view_678, addmm_254, view_680, addmm_255, view_682, addmm_256, addmm_257, view_686, clone_default_18, clone_default_19, clone_default_20, getitem_93, getitem_94, getitem_95, alias_default_13, view_702, addmm_261, view_704, view_706, addmm_263, view_708, view_710, addmm_265, view_712, view_714, addmm_267, view_716, view_718, addmm_269, view_720, addmm_270, view_722, addmm_271, addmm_272, view_726, clone_default_15, clone_default_16, clone_default_17, getitem_86, getitem_87, getitem_88, alias_default_11, view_742, addmm_276, view_744, view_746, addmm_278, view_748, view_750, addmm_280, view_752, view_754, addmm_282, view_756, view_758, addmm_284, view_760, addmm_285, view_762, addmm_286, addmm_287, view_766, clone_default_12, clone_default_13, clone_default_14, getitem_79, getitem_80, getitem_81, alias_default_9, view_782, addmm_291, view_784, view_786, addmm_293, view_788, view_790, addmm_295, view_792, view_794, addmm_297, view_796, view_798, addmm_299, view_800, addmm_300, view_802, addmm_301, addmm_302, view_806, clone_default_9, clone_default_10, clone_default_11, getitem_72, getitem_73, getitem_74, alias_default_7, view_822, addmm_306, view_824, view_826, addmm_308, view_828, view_830, addmm_310, view_832, view_834, addmm_312, view_836, view_838, addmm_314, view_840, addmm_315, view_842, addmm_316, addmm_317, view_846, clone_default_6, clone_default_7, clone_default_8, getitem_65, getitem_66, getitem_67, alias_default_5, view_862, addmm_321, view_864, view_866, addmm_323, view_868, view_870, addmm_325, view_872, view_874, addmm_327, view_876, view_878, addmm_329, view_880, addmm_330, view_882, addmm_331, addmm_332, view_886, clone_default_3, clone_default_4, clone_default_5, getitem_58, getitem_59, getitem_60, alias_default_3, view_902, addmm_336, view_904, view_906, addmm_338, view_908, view_910, addmm_340, view_912, view_914, addmm_342, view_916, view_918, addmm_344, view_920, addmm_345, view_922, addmm_346, addmm_347, view_926, clone_default, clone_default_1, clone_default_2, getitem_51, getitem_52, getitem_53, alias_default_1, view_942, addmm_351, view_944, view_946, addmm_353, view_948, view_950, addmm_355, view_952, view_954, addmm_357, view_956, view_958, addmm_359, view_960, addmm_360, view_962, addmm_361, getitem_49, rsqrt, sub_27, convert_element_type, permute_483, permute_484, permute_486, permute_490, permute_494, le_1, permute_498, permute_502, le_2, permute_506, permute_510, le_3, permute_514, permute_518, le_4, permute_522, permute_526, permute_539, permute_543, permute_547, permute_551, permute_555, permute_559, permute_563, le_5, permute_567, permute_571, le_6, permute_575, permute_579, le_7, permute_583, permute_587, le_8, permute_591, permute_595, permute_608, permute_612, permute_616, permute_620, permute_624, permute_628, permute_632, le_9, permute_636, permute_640, le_10, permute_644, permute_648, le_11, permute_652, permute_656, le_12, permute_660, permute_664, permute_677, permute_681, permute_685, permute_689, permute_693, permute_697, permute_701, le_13, permute_705, permute_709, le_14, permute_713, permute_717, le_15, permute_721, permute_725, le_16, permute_729, permute_733, permute_746, permute_750, permute_754, permute_758, permute_762, permute_766, permute_770, le_17, permute_774, permute_778, le_18, permute_782, permute_786, le_19, permute_790, permute_794, le_20, permute_798, permute_802, permute_815, permute_819, permute_823, permute_827, permute_831, permute_835, permute_839, le_21, permute_843, permute_847, le_22, permute_851, permute_855, le_23, permute_859, permute_863, le_24, permute_867, permute_871, permute_884, permute_888, permute_892, permute_896, permute_900, permute_904, permute_908, le_25, permute_912, permute_916, le_26, permute_920, permute_924, le_27, permute_928, permute_932, le_28, permute_936, permute_940, permute_953, permute_957, permute_961, permute_965, permute_969, permute_973, permute_977, le_29, permute_981, permute_985, le_30, permute_989, permute_993, le_31, permute_997, permute_1001, le_32, permute_1005, permute_1009, permute_1022, permute_1026, permute_1030, permute_1034, permute_1038, permute_1042, permute_1046, le_33, permute_1050, permute_1054, le_34, permute_1058, permute_1062, le_35, permute_1066, permute_1070, le_36, permute_1074, permute_1078, permute_1091, permute_1095, permute_1099, permute_1103, permute_1107, permute_1111, permute_1115, le_37, permute_1119, permute_1123, le_38, permute_1127, permute_1131, le_39, permute_1135, permute_1139, le_40, permute_1143, permute_1147, permute_1160, permute_1164, permute_1168, permute_1172, permute_1176, permute_1180, permute_1184, le_41, permute_1188, permute_1192, le_42, permute_1196, permute_1200, le_43, permute_1204, permute_1208, le_44, permute_1212, permute_1216, permute_1229, permute_1233, permute_1237, permute_1241, permute_1245, permute_1249, permute_1253, le_45, permute_1257, permute_1261, le_46, permute_1265, permute_1269, le_47, permute_1273, permute_1277, le_48, permute_1281, permute_1285, permute_1298, permute_1302, permute_1306, permute_1310, permute_1314, permute_1318, permute_1322, le_49, permute_1326, permute_1330, le_50, permute_1334, permute_1338, le_51, permute_1342, permute_1346, le_52, permute_1350, permute_1354, permute_1367, permute_1371, permute_1375, permute_1379, permute_1383, permute_1387, permute_1391, le_53, permute_1395, permute_1399, le_54, permute_1403, permute_1407, le_55, permute_1411, permute_1415, le_56, permute_1419, permute_1423, permute_1436, permute_1440, permute_1444, permute_1448, permute_1452, permute_1456, permute_1460, le_57, permute_1464, permute_1468, le_58, permute_1472, permute_1476, le_59, permute_1480, permute_1484, le_60, permute_1488, permute_1492, permute_1505, permute_1509, permute_1513, permute_1517, permute_1521, permute_1525, permute_1529, le_61, permute_1533, permute_1537, le_62, permute_1541, permute_1545, le_63, permute_1549, permute_1553, le_64, permute_1557, permute_1561, permute_1574, permute_1578, permute_1582, permute_1586, permute_1590, permute_1594, permute_1598, le_65, permute_1602, permute_1606, le_66, permute_1610, permute_1614, le_67, permute_1618, permute_1622, le_68, permute_1626, permute_1630, permute_1643, permute_1647, permute_1651, permute_1655, permute_1659, permute_1663, permute_1667, le_69, permute_1671, permute_1675, le_70, permute_1679, permute_1683, le_71, permute_1687, permute_1691, le_72, permute_1695, permute_1699, permute_1712, permute_1716, permute_1720, permute_1724, permute_1728, permute_1732, permute_1736, le_73, permute_1740, permute_1744, le_74, permute_1748, permute_1752, le_75, permute_1756, permute_1760, le_76, permute_1764, permute_1768, permute_1781, permute_1785, permute_1789, permute_1793, permute_1797, permute_1801, permute_1805, le_77, permute_1809, permute_1813, le_78, permute_1817, permute_1821, le_79, permute_1825, permute_1829, le_80, permute_1833, permute_1837, permute_1850, permute_1854, permute_1858, permute_1862, permute_1866, permute_1870, permute_1874, le_81, permute_1878, permute_1882, le_82, permute_1886, permute_1890, le_83, permute_1894, permute_1898, le_84, permute_1902, permute_1906, permute_1919, permute_1923, permute_1927, permute_1931, permute_1935, permute_1939, permute_1943, le_85, permute_1947, permute_1951, le_86, permute_1955, permute_1959, le_87, permute_1963, permute_1967, le_88, permute_1971, permute_1975, permute_1988, permute_1992, permute_1996, permute_2000, permute_2004, permute_2008, permute_2012, le_89, permute_2016, permute_2020, le_90, permute_2024, permute_2028, le_91, permute_2032, permute_2036, le_92, permute_2040, permute_2044, permute_2057, permute_2061, permute_2065, permute_2069, permute_2073, permute_2077, permute_2081, le_93, permute_2085, permute_2089, le_94, permute_2093, permute_2097, le_95, permute_2101, permute_2105, le_96, permute_2109, permute_2113, permute_2126, permute_2130, permute_2134, permute_2138, permute_2142, permute_2146]
    