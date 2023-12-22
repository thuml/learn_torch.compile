from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[768]"; primals_2: "f32[16, 1, 1]"; primals_3: "f32[768]"; primals_4: "f32[768]"; primals_5: "f32[768]"; primals_6: "f32[16, 1, 1]"; primals_7: "f32[768]"; primals_8: "f32[768]"; primals_9: "f32[768]"; primals_10: "f32[16, 1, 1]"; primals_11: "f32[768]"; primals_12: "f32[768]"; primals_13: "f32[768]"; primals_14: "f32[16, 1, 1]"; primals_15: "f32[768]"; primals_16: "f32[768]"; primals_17: "f32[768]"; primals_18: "f32[16, 1, 1]"; primals_19: "f32[768]"; primals_20: "f32[768]"; primals_21: "f32[768]"; primals_22: "f32[16, 1, 1]"; primals_23: "f32[768]"; primals_24: "f32[768]"; primals_25: "f32[768]"; primals_26: "f32[16, 1, 1]"; primals_27: "f32[768]"; primals_28: "f32[768]"; primals_29: "f32[768]"; primals_30: "f32[16, 1, 1]"; primals_31: "f32[768]"; primals_32: "f32[768]"; primals_33: "f32[768]"; primals_34: "f32[16, 1, 1]"; primals_35: "f32[768]"; primals_36: "f32[768]"; primals_37: "f32[768]"; primals_38: "f32[16, 1, 1]"; primals_39: "f32[768]"; primals_40: "f32[768]"; primals_41: "f32[768]"; primals_42: "f32[16, 1, 1]"; primals_43: "f32[768]"; primals_44: "f32[768]"; primals_45: "f32[768]"; primals_46: "f32[16, 1, 1]"; primals_47: "f32[768]"; primals_48: "f32[768]"; primals_49: "f32[768]"; primals_50: "f32[16, 1, 1]"; primals_51: "f32[768]"; primals_52: "f32[768]"; primals_53: "f32[768]"; primals_54: "f32[16, 1, 1]"; primals_55: "f32[768]"; primals_56: "f32[768]"; primals_57: "f32[768]"; primals_58: "f32[16, 1, 1]"; primals_59: "f32[768]"; primals_60: "f32[768]"; primals_61: "f32[768]"; primals_62: "f32[16, 1, 1]"; primals_63: "f32[768]"; primals_64: "f32[768]"; primals_65: "f32[768]"; primals_66: "f32[16, 1, 1]"; primals_67: "f32[768]"; primals_68: "f32[768]"; primals_69: "f32[768]"; primals_70: "f32[16, 1, 1]"; primals_71: "f32[768]"; primals_72: "f32[768]"; primals_73: "f32[768]"; primals_74: "f32[16, 1, 1]"; primals_75: "f32[768]"; primals_76: "f32[768]"; primals_77: "f32[768]"; primals_78: "f32[16, 1, 1]"; primals_79: "f32[768]"; primals_80: "f32[768]"; primals_81: "f32[768]"; primals_82: "f32[16, 1, 1]"; primals_83: "f32[768]"; primals_84: "f32[768]"; primals_85: "f32[768]"; primals_86: "f32[16, 1, 1]"; primals_87: "f32[768]"; primals_88: "f32[768]"; primals_89: "f32[768]"; primals_90: "f32[16, 1, 1]"; primals_91: "f32[768]"; primals_92: "f32[768]"; primals_93: "f32[768]"; primals_94: "f32[16, 1, 1]"; primals_95: "f32[768]"; primals_96: "f32[768]"; primals_97: "f32[1, 1, 768]"; primals_98: "f32[768]"; primals_99: "f32[768]"; primals_100: "f32[768]"; primals_101: "f32[768]"; primals_102: "f32[192, 3, 3, 3]"; primals_103: "f32[192]"; primals_104: "f32[192]"; primals_105: "f32[384, 192, 3, 3]"; primals_106: "f32[384]"; primals_107: "f32[384]"; primals_108: "f32[768, 384, 3, 3]"; primals_109: "f32[768]"; primals_110: "f32[768]"; primals_111: "f32[768, 64, 1, 1]"; primals_112: "f32[768]"; primals_113: "f32[768]"; primals_114: "f32[768]"; primals_115: "f32[2304, 768]"; primals_116: "f32[2304]"; primals_117: "f32[768, 768]"; primals_118: "f32[768]"; primals_119: "f32[768]"; primals_120: "f32[768]"; primals_121: "f32[768, 1, 3, 3]"; primals_122: "f32[768]"; primals_123: "f32[768]"; primals_124: "f32[768]"; primals_125: "f32[768, 1, 3, 3]"; primals_126: "f32[768]"; primals_127: "f32[768]"; primals_128: "f32[768]"; primals_129: "f32[3072, 768]"; primals_130: "f32[3072]"; primals_131: "f32[768, 3072]"; primals_132: "f32[768]"; primals_133: "f32[768]"; primals_134: "f32[768]"; primals_135: "f32[2304, 768]"; primals_136: "f32[2304]"; primals_137: "f32[768, 768]"; primals_138: "f32[768]"; primals_139: "f32[768]"; primals_140: "f32[768]"; primals_141: "f32[768, 1, 3, 3]"; primals_142: "f32[768]"; primals_143: "f32[768]"; primals_144: "f32[768]"; primals_145: "f32[768, 1, 3, 3]"; primals_146: "f32[768]"; primals_147: "f32[768]"; primals_148: "f32[768]"; primals_149: "f32[3072, 768]"; primals_150: "f32[3072]"; primals_151: "f32[768, 3072]"; primals_152: "f32[768]"; primals_153: "f32[768]"; primals_154: "f32[768]"; primals_155: "f32[2304, 768]"; primals_156: "f32[2304]"; primals_157: "f32[768, 768]"; primals_158: "f32[768]"; primals_159: "f32[768]"; primals_160: "f32[768]"; primals_161: "f32[768, 1, 3, 3]"; primals_162: "f32[768]"; primals_163: "f32[768]"; primals_164: "f32[768]"; primals_165: "f32[768, 1, 3, 3]"; primals_166: "f32[768]"; primals_167: "f32[768]"; primals_168: "f32[768]"; primals_169: "f32[3072, 768]"; primals_170: "f32[3072]"; primals_171: "f32[768, 3072]"; primals_172: "f32[768]"; primals_173: "f32[768]"; primals_174: "f32[768]"; primals_175: "f32[2304, 768]"; primals_176: "f32[2304]"; primals_177: "f32[768, 768]"; primals_178: "f32[768]"; primals_179: "f32[768]"; primals_180: "f32[768]"; primals_181: "f32[768, 1, 3, 3]"; primals_182: "f32[768]"; primals_183: "f32[768]"; primals_184: "f32[768]"; primals_185: "f32[768, 1, 3, 3]"; primals_186: "f32[768]"; primals_187: "f32[768]"; primals_188: "f32[768]"; primals_189: "f32[3072, 768]"; primals_190: "f32[3072]"; primals_191: "f32[768, 3072]"; primals_192: "f32[768]"; primals_193: "f32[768]"; primals_194: "f32[768]"; primals_195: "f32[2304, 768]"; primals_196: "f32[2304]"; primals_197: "f32[768, 768]"; primals_198: "f32[768]"; primals_199: "f32[768]"; primals_200: "f32[768]"; primals_201: "f32[768, 1, 3, 3]"; primals_202: "f32[768]"; primals_203: "f32[768]"; primals_204: "f32[768]"; primals_205: "f32[768, 1, 3, 3]"; primals_206: "f32[768]"; primals_207: "f32[768]"; primals_208: "f32[768]"; primals_209: "f32[3072, 768]"; primals_210: "f32[3072]"; primals_211: "f32[768, 3072]"; primals_212: "f32[768]"; primals_213: "f32[768]"; primals_214: "f32[768]"; primals_215: "f32[2304, 768]"; primals_216: "f32[2304]"; primals_217: "f32[768, 768]"; primals_218: "f32[768]"; primals_219: "f32[768]"; primals_220: "f32[768]"; primals_221: "f32[768, 1, 3, 3]"; primals_222: "f32[768]"; primals_223: "f32[768]"; primals_224: "f32[768]"; primals_225: "f32[768, 1, 3, 3]"; primals_226: "f32[768]"; primals_227: "f32[768]"; primals_228: "f32[768]"; primals_229: "f32[3072, 768]"; primals_230: "f32[3072]"; primals_231: "f32[768, 3072]"; primals_232: "f32[768]"; primals_233: "f32[768]"; primals_234: "f32[768]"; primals_235: "f32[2304, 768]"; primals_236: "f32[2304]"; primals_237: "f32[768, 768]"; primals_238: "f32[768]"; primals_239: "f32[768]"; primals_240: "f32[768]"; primals_241: "f32[768, 1, 3, 3]"; primals_242: "f32[768]"; primals_243: "f32[768]"; primals_244: "f32[768]"; primals_245: "f32[768, 1, 3, 3]"; primals_246: "f32[768]"; primals_247: "f32[768]"; primals_248: "f32[768]"; primals_249: "f32[3072, 768]"; primals_250: "f32[3072]"; primals_251: "f32[768, 3072]"; primals_252: "f32[768]"; primals_253: "f32[768]"; primals_254: "f32[768]"; primals_255: "f32[2304, 768]"; primals_256: "f32[2304]"; primals_257: "f32[768, 768]"; primals_258: "f32[768]"; primals_259: "f32[768]"; primals_260: "f32[768]"; primals_261: "f32[768, 1, 3, 3]"; primals_262: "f32[768]"; primals_263: "f32[768]"; primals_264: "f32[768]"; primals_265: "f32[768, 1, 3, 3]"; primals_266: "f32[768]"; primals_267: "f32[768]"; primals_268: "f32[768]"; primals_269: "f32[3072, 768]"; primals_270: "f32[3072]"; primals_271: "f32[768, 3072]"; primals_272: "f32[768]"; primals_273: "f32[768]"; primals_274: "f32[768]"; primals_275: "f32[2304, 768]"; primals_276: "f32[2304]"; primals_277: "f32[768, 768]"; primals_278: "f32[768]"; primals_279: "f32[768]"; primals_280: "f32[768]"; primals_281: "f32[768, 1, 3, 3]"; primals_282: "f32[768]"; primals_283: "f32[768]"; primals_284: "f32[768]"; primals_285: "f32[768, 1, 3, 3]"; primals_286: "f32[768]"; primals_287: "f32[768]"; primals_288: "f32[768]"; primals_289: "f32[3072, 768]"; primals_290: "f32[3072]"; primals_291: "f32[768, 3072]"; primals_292: "f32[768]"; primals_293: "f32[768]"; primals_294: "f32[768]"; primals_295: "f32[2304, 768]"; primals_296: "f32[2304]"; primals_297: "f32[768, 768]"; primals_298: "f32[768]"; primals_299: "f32[768]"; primals_300: "f32[768]"; primals_301: "f32[768, 1, 3, 3]"; primals_302: "f32[768]"; primals_303: "f32[768]"; primals_304: "f32[768]"; primals_305: "f32[768, 1, 3, 3]"; primals_306: "f32[768]"; primals_307: "f32[768]"; primals_308: "f32[768]"; primals_309: "f32[3072, 768]"; primals_310: "f32[3072]"; primals_311: "f32[768, 3072]"; primals_312: "f32[768]"; primals_313: "f32[768]"; primals_314: "f32[768]"; primals_315: "f32[2304, 768]"; primals_316: "f32[2304]"; primals_317: "f32[768, 768]"; primals_318: "f32[768]"; primals_319: "f32[768]"; primals_320: "f32[768]"; primals_321: "f32[768, 1, 3, 3]"; primals_322: "f32[768]"; primals_323: "f32[768]"; primals_324: "f32[768]"; primals_325: "f32[768, 1, 3, 3]"; primals_326: "f32[768]"; primals_327: "f32[768]"; primals_328: "f32[768]"; primals_329: "f32[3072, 768]"; primals_330: "f32[3072]"; primals_331: "f32[768, 3072]"; primals_332: "f32[768]"; primals_333: "f32[768]"; primals_334: "f32[768]"; primals_335: "f32[2304, 768]"; primals_336: "f32[2304]"; primals_337: "f32[768, 768]"; primals_338: "f32[768]"; primals_339: "f32[768]"; primals_340: "f32[768]"; primals_341: "f32[768, 1, 3, 3]"; primals_342: "f32[768]"; primals_343: "f32[768]"; primals_344: "f32[768]"; primals_345: "f32[768, 1, 3, 3]"; primals_346: "f32[768]"; primals_347: "f32[768]"; primals_348: "f32[768]"; primals_349: "f32[3072, 768]"; primals_350: "f32[3072]"; primals_351: "f32[768, 3072]"; primals_352: "f32[768]"; primals_353: "f32[768]"; primals_354: "f32[768]"; primals_355: "f32[2304, 768]"; primals_356: "f32[2304]"; primals_357: "f32[768, 768]"; primals_358: "f32[768]"; primals_359: "f32[768]"; primals_360: "f32[768]"; primals_361: "f32[768, 1, 3, 3]"; primals_362: "f32[768]"; primals_363: "f32[768]"; primals_364: "f32[768]"; primals_365: "f32[768, 1, 3, 3]"; primals_366: "f32[768]"; primals_367: "f32[768]"; primals_368: "f32[768]"; primals_369: "f32[3072, 768]"; primals_370: "f32[3072]"; primals_371: "f32[768, 3072]"; primals_372: "f32[768]"; primals_373: "f32[768]"; primals_374: "f32[768]"; primals_375: "f32[2304, 768]"; primals_376: "f32[2304]"; primals_377: "f32[768, 768]"; primals_378: "f32[768]"; primals_379: "f32[768]"; primals_380: "f32[768]"; primals_381: "f32[768, 1, 3, 3]"; primals_382: "f32[768]"; primals_383: "f32[768]"; primals_384: "f32[768]"; primals_385: "f32[768, 1, 3, 3]"; primals_386: "f32[768]"; primals_387: "f32[768]"; primals_388: "f32[768]"; primals_389: "f32[3072, 768]"; primals_390: "f32[3072]"; primals_391: "f32[768, 3072]"; primals_392: "f32[768]"; primals_393: "f32[768]"; primals_394: "f32[768]"; primals_395: "f32[2304, 768]"; primals_396: "f32[2304]"; primals_397: "f32[768, 768]"; primals_398: "f32[768]"; primals_399: "f32[768]"; primals_400: "f32[768]"; primals_401: "f32[768, 1, 3, 3]"; primals_402: "f32[768]"; primals_403: "f32[768]"; primals_404: "f32[768]"; primals_405: "f32[768, 1, 3, 3]"; primals_406: "f32[768]"; primals_407: "f32[768]"; primals_408: "f32[768]"; primals_409: "f32[3072, 768]"; primals_410: "f32[3072]"; primals_411: "f32[768, 3072]"; primals_412: "f32[768]"; primals_413: "f32[768]"; primals_414: "f32[768]"; primals_415: "f32[2304, 768]"; primals_416: "f32[2304]"; primals_417: "f32[768, 768]"; primals_418: "f32[768]"; primals_419: "f32[768]"; primals_420: "f32[768]"; primals_421: "f32[768, 1, 3, 3]"; primals_422: "f32[768]"; primals_423: "f32[768]"; primals_424: "f32[768]"; primals_425: "f32[768, 1, 3, 3]"; primals_426: "f32[768]"; primals_427: "f32[768]"; primals_428: "f32[768]"; primals_429: "f32[3072, 768]"; primals_430: "f32[3072]"; primals_431: "f32[768, 3072]"; primals_432: "f32[768]"; primals_433: "f32[768]"; primals_434: "f32[768]"; primals_435: "f32[2304, 768]"; primals_436: "f32[2304]"; primals_437: "f32[768, 768]"; primals_438: "f32[768]"; primals_439: "f32[768]"; primals_440: "f32[768]"; primals_441: "f32[768, 1, 3, 3]"; primals_442: "f32[768]"; primals_443: "f32[768]"; primals_444: "f32[768]"; primals_445: "f32[768, 1, 3, 3]"; primals_446: "f32[768]"; primals_447: "f32[768]"; primals_448: "f32[768]"; primals_449: "f32[3072, 768]"; primals_450: "f32[3072]"; primals_451: "f32[768, 3072]"; primals_452: "f32[768]"; primals_453: "f32[768]"; primals_454: "f32[768]"; primals_455: "f32[2304, 768]"; primals_456: "f32[2304]"; primals_457: "f32[768, 768]"; primals_458: "f32[768]"; primals_459: "f32[768]"; primals_460: "f32[768]"; primals_461: "f32[768, 1, 3, 3]"; primals_462: "f32[768]"; primals_463: "f32[768]"; primals_464: "f32[768]"; primals_465: "f32[768, 1, 3, 3]"; primals_466: "f32[768]"; primals_467: "f32[768]"; primals_468: "f32[768]"; primals_469: "f32[3072, 768]"; primals_470: "f32[3072]"; primals_471: "f32[768, 3072]"; primals_472: "f32[768]"; primals_473: "f32[768]"; primals_474: "f32[768]"; primals_475: "f32[2304, 768]"; primals_476: "f32[2304]"; primals_477: "f32[768, 768]"; primals_478: "f32[768]"; primals_479: "f32[768]"; primals_480: "f32[768]"; primals_481: "f32[768, 1, 3, 3]"; primals_482: "f32[768]"; primals_483: "f32[768]"; primals_484: "f32[768]"; primals_485: "f32[768, 1, 3, 3]"; primals_486: "f32[768]"; primals_487: "f32[768]"; primals_488: "f32[768]"; primals_489: "f32[3072, 768]"; primals_490: "f32[3072]"; primals_491: "f32[768, 3072]"; primals_492: "f32[768]"; primals_493: "f32[768]"; primals_494: "f32[768]"; primals_495: "f32[2304, 768]"; primals_496: "f32[2304]"; primals_497: "f32[768, 768]"; primals_498: "f32[768]"; primals_499: "f32[768]"; primals_500: "f32[768]"; primals_501: "f32[768, 1, 3, 3]"; primals_502: "f32[768]"; primals_503: "f32[768]"; primals_504: "f32[768]"; primals_505: "f32[768, 1, 3, 3]"; primals_506: "f32[768]"; primals_507: "f32[768]"; primals_508: "f32[768]"; primals_509: "f32[3072, 768]"; primals_510: "f32[3072]"; primals_511: "f32[768, 3072]"; primals_512: "f32[768]"; primals_513: "f32[768]"; primals_514: "f32[768]"; primals_515: "f32[2304, 768]"; primals_516: "f32[2304]"; primals_517: "f32[768, 768]"; primals_518: "f32[768]"; primals_519: "f32[768]"; primals_520: "f32[768]"; primals_521: "f32[768, 1, 3, 3]"; primals_522: "f32[768]"; primals_523: "f32[768]"; primals_524: "f32[768]"; primals_525: "f32[768, 1, 3, 3]"; primals_526: "f32[768]"; primals_527: "f32[768]"; primals_528: "f32[768]"; primals_529: "f32[3072, 768]"; primals_530: "f32[3072]"; primals_531: "f32[768, 3072]"; primals_532: "f32[768]"; primals_533: "f32[768]"; primals_534: "f32[768]"; primals_535: "f32[2304, 768]"; primals_536: "f32[2304]"; primals_537: "f32[768, 768]"; primals_538: "f32[768]"; primals_539: "f32[768]"; primals_540: "f32[768]"; primals_541: "f32[768, 1, 3, 3]"; primals_542: "f32[768]"; primals_543: "f32[768]"; primals_544: "f32[768]"; primals_545: "f32[768, 1, 3, 3]"; primals_546: "f32[768]"; primals_547: "f32[768]"; primals_548: "f32[768]"; primals_549: "f32[3072, 768]"; primals_550: "f32[3072]"; primals_551: "f32[768, 3072]"; primals_552: "f32[768]"; primals_553: "f32[768]"; primals_554: "f32[768]"; primals_555: "f32[2304, 768]"; primals_556: "f32[2304]"; primals_557: "f32[768, 768]"; primals_558: "f32[768]"; primals_559: "f32[768]"; primals_560: "f32[768]"; primals_561: "f32[768, 1, 3, 3]"; primals_562: "f32[768]"; primals_563: "f32[768]"; primals_564: "f32[768]"; primals_565: "f32[768, 1, 3, 3]"; primals_566: "f32[768]"; primals_567: "f32[768]"; primals_568: "f32[768]"; primals_569: "f32[3072, 768]"; primals_570: "f32[3072]"; primals_571: "f32[768, 3072]"; primals_572: "f32[768]"; primals_573: "f32[768]"; primals_574: "f32[768]"; primals_575: "f32[2304, 768]"; primals_576: "f32[2304]"; primals_577: "f32[768, 768]"; primals_578: "f32[768]"; primals_579: "f32[768]"; primals_580: "f32[768]"; primals_581: "f32[768, 1, 3, 3]"; primals_582: "f32[768]"; primals_583: "f32[768]"; primals_584: "f32[768]"; primals_585: "f32[768, 1, 3, 3]"; primals_586: "f32[768]"; primals_587: "f32[768]"; primals_588: "f32[768]"; primals_589: "f32[3072, 768]"; primals_590: "f32[3072]"; primals_591: "f32[768, 3072]"; primals_592: "f32[768]"; primals_593: "f32[768]"; primals_594: "f32[768]"; primals_595: "f32[768, 768]"; primals_596: "f32[768]"; primals_597: "f32[768, 768]"; primals_598: "f32[768]"; primals_599: "f32[768, 768]"; primals_600: "f32[768]"; primals_601: "f32[768, 768]"; primals_602: "f32[768]"; primals_603: "f32[768]"; primals_604: "f32[768]"; primals_605: "f32[3072, 768]"; primals_606: "f32[3072]"; primals_607: "f32[768, 3072]"; primals_608: "f32[768]"; primals_609: "f32[768]"; primals_610: "f32[768]"; primals_611: "f32[768, 768]"; primals_612: "f32[768]"; primals_613: "f32[768, 768]"; primals_614: "f32[768]"; primals_615: "f32[768, 768]"; primals_616: "f32[768]"; primals_617: "f32[768, 768]"; primals_618: "f32[768]"; primals_619: "f32[768]"; primals_620: "f32[768]"; primals_621: "f32[3072, 768]"; primals_622: "f32[3072]"; primals_623: "f32[768, 3072]"; primals_624: "f32[768]"; primals_625: "f32[768]"; primals_626: "f32[768]"; primals_627: "f32[1000, 768]"; primals_628: "f32[1000]"; primals_629: "f32[192]"; primals_630: "f32[192]"; primals_631: "i64[]"; primals_632: "f32[384]"; primals_633: "f32[384]"; primals_634: "i64[]"; primals_635: "f32[768]"; primals_636: "f32[768]"; primals_637: "i64[]"; primals_638: "f32[768]"; primals_639: "f32[768]"; primals_640: "i64[]"; primals_641: "f32[768]"; primals_642: "f32[768]"; primals_643: "i64[]"; primals_644: "f32[768]"; primals_645: "f32[768]"; primals_646: "i64[]"; primals_647: "f32[768]"; primals_648: "f32[768]"; primals_649: "i64[]"; primals_650: "f32[768]"; primals_651: "f32[768]"; primals_652: "i64[]"; primals_653: "f32[768]"; primals_654: "f32[768]"; primals_655: "i64[]"; primals_656: "f32[768]"; primals_657: "f32[768]"; primals_658: "i64[]"; primals_659: "f32[768]"; primals_660: "f32[768]"; primals_661: "i64[]"; primals_662: "f32[768]"; primals_663: "f32[768]"; primals_664: "i64[]"; primals_665: "f32[768]"; primals_666: "f32[768]"; primals_667: "i64[]"; primals_668: "f32[768]"; primals_669: "f32[768]"; primals_670: "i64[]"; primals_671: "f32[768]"; primals_672: "f32[768]"; primals_673: "i64[]"; primals_674: "f32[768]"; primals_675: "f32[768]"; primals_676: "i64[]"; primals_677: "f32[768]"; primals_678: "f32[768]"; primals_679: "i64[]"; primals_680: "f32[768]"; primals_681: "f32[768]"; primals_682: "i64[]"; primals_683: "f32[768]"; primals_684: "f32[768]"; primals_685: "i64[]"; primals_686: "f32[768]"; primals_687: "f32[768]"; primals_688: "i64[]"; primals_689: "f32[768]"; primals_690: "f32[768]"; primals_691: "i64[]"; primals_692: "f32[768]"; primals_693: "f32[768]"; primals_694: "i64[]"; primals_695: "f32[768]"; primals_696: "f32[768]"; primals_697: "i64[]"; primals_698: "f32[768]"; primals_699: "f32[768]"; primals_700: "i64[]"; primals_701: "f32[768]"; primals_702: "f32[768]"; primals_703: "i64[]"; primals_704: "f32[768]"; primals_705: "f32[768]"; primals_706: "i64[]"; primals_707: "f32[768]"; primals_708: "f32[768]"; primals_709: "i64[]"; primals_710: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:107, code: x = self.proj(x)
    convolution: "f32[8, 192, 112, 112]" = torch.ops.aten.convolution.default(primals_710, primals_102, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_631, 1);  primals_631 = None
    _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(convolution, primals_103, primals_104, primals_629, primals_630, True, 0.1, 1e-05);  primals_104 = primals_629 = primals_630 = None
    getitem: "f32[8, 192, 112, 112]" = _native_batch_norm_legit_functional[0]
    getitem_1: "f32[192]" = _native_batch_norm_legit_functional[1]
    getitem_2: "f32[192]" = _native_batch_norm_legit_functional[2]
    getitem_3: "f32[192]" = _native_batch_norm_legit_functional[3]
    getitem_4: "f32[192]" = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
    gelu: "f32[8, 192, 112, 112]" = torch.ops.aten.gelu.default(getitem)
    convolution_1: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(gelu, primals_105, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_1: "i64[]" = torch.ops.aten.add.Tensor(primals_634, 1);  primals_634 = None
    _native_batch_norm_legit_functional_1 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_1, primals_106, primals_107, primals_632, primals_633, True, 0.1, 1e-05);  primals_107 = primals_632 = primals_633 = None
    getitem_5: "f32[8, 384, 56, 56]" = _native_batch_norm_legit_functional_1[0]
    getitem_6: "f32[384]" = _native_batch_norm_legit_functional_1[1]
    getitem_7: "f32[384]" = _native_batch_norm_legit_functional_1[2]
    getitem_8: "f32[384]" = _native_batch_norm_legit_functional_1[3]
    getitem_9: "f32[384]" = _native_batch_norm_legit_functional_1[4];  _native_batch_norm_legit_functional_1 = None
    gelu_1: "f32[8, 384, 56, 56]" = torch.ops.aten.gelu.default(getitem_5)
    convolution_2: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(gelu_1, primals_108, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_2: "i64[]" = torch.ops.aten.add.Tensor(primals_637, 1);  primals_637 = None
    _native_batch_norm_legit_functional_2 = torch.ops.aten._native_batch_norm_legit_functional.default(convolution_2, primals_109, primals_110, primals_635, primals_636, True, 0.1, 1e-05);  primals_110 = primals_635 = primals_636 = None
    getitem_10: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_2[0]
    getitem_11: "f32[768]" = _native_batch_norm_legit_functional_2[1]
    getitem_12: "f32[768]" = _native_batch_norm_legit_functional_2[2]
    getitem_13: "f32[768]" = _native_batch_norm_legit_functional_2[3]
    getitem_14: "f32[768]" = _native_batch_norm_legit_functional_2[4];  _native_batch_norm_legit_functional_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:109, code: x = x.flatten(2).transpose(1, 2)  # (B, N, C)
    view: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_10, [8, 768, 784]);  getitem_10 = None
    transpose: "f32[8, 784, 768]" = torch.ops.aten.transpose.int(view, 1, 2);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:51, code: y_embed = torch.arange(1, H+1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, 1, W)
    arange: "f32[28]" = torch.ops.aten.arange.start(1, 29, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    unsqueeze: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arange, 1);  arange = None
    repeat: "f32[1, 28, 28]" = torch.ops.aten.repeat.default(unsqueeze, [1, 1, 28]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:52, code: x_embed = torch.arange(1, W+1, dtype=torch.float32, device=device).repeat(1, H, 1)
    arange_1: "f32[28]" = torch.ops.aten.arange.start(1, 29, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    repeat_1: "f32[1, 28, 28]" = torch.ops.aten.repeat.default(arange_1, [1, 28, 1]);  arange_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:53, code: y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
    slice_1: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(repeat, 0, 0, 9223372036854775807)
    slice_2: "f32[1, 1, 28]" = torch.ops.aten.slice.Tensor(slice_1, 1, -1, 9223372036854775807);  slice_1 = None
    slice_3: "f32[1, 1, 28]" = torch.ops.aten.slice.Tensor(slice_2, 2, 0, 9223372036854775807);  slice_2 = None
    add_3: "f32[1, 1, 28]" = torch.ops.aten.add.Tensor(slice_3, 1e-06);  slice_3 = None
    div: "f32[1, 28, 28]" = torch.ops.aten.div.Tensor(repeat, add_3);  repeat = add_3 = None
    mul: "f32[1, 28, 28]" = torch.ops.aten.mul.Tensor(div, 6.283185307179586);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:54, code: x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
    slice_4: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(repeat_1, 0, 0, 9223372036854775807)
    slice_5: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(slice_4, 1, 0, 9223372036854775807);  slice_4 = None
    slice_6: "f32[1, 28, 1]" = torch.ops.aten.slice.Tensor(slice_5, 2, -1, 9223372036854775807);  slice_5 = None
    add_4: "f32[1, 28, 1]" = torch.ops.aten.add.Tensor(slice_6, 1e-06);  slice_6 = None
    div_1: "f32[1, 28, 28]" = torch.ops.aten.div.Tensor(repeat_1, add_4);  repeat_1 = add_4 = None
    mul_1: "f32[1, 28, 28]" = torch.ops.aten.mul.Tensor(div_1, 6.283185307179586);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:55, code: dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
    arange_2: "f32[32]" = torch.ops.aten.arange.default(32, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:56, code: dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.hidden_dim)
    div_2: "f32[32]" = torch.ops.aten.div.Tensor_mode(arange_2, 2, rounding_mode = 'floor');  arange_2 = None
    mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(div_2, 2);  div_2 = None
    div_3: "f32[32]" = torch.ops.aten.div.Tensor(mul_2, 32);  mul_2 = None
    pow_1: "f32[32]" = torch.ops.aten.pow.Scalar(10000, div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:57, code: pos_x = x_embed[:, :, :, None] / dim_t
    slice_7: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(mul_1, 0, 0, 9223372036854775807);  mul_1 = None
    slice_8: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 9223372036854775807);  slice_7 = None
    slice_9: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(slice_8, 2, 0, 9223372036854775807);  slice_8 = None
    unsqueeze_1: "f32[1, 28, 28, 1]" = torch.ops.aten.unsqueeze.default(slice_9, 3);  slice_9 = None
    div_4: "f32[1, 28, 28, 32]" = torch.ops.aten.div.Tensor(unsqueeze_1, pow_1);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:58, code: pos_y = y_embed[:, :, :, None] / dim_t
    slice_10: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(mul, 0, 0, 9223372036854775807);  mul = None
    slice_11: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(slice_10, 1, 0, 9223372036854775807);  slice_10 = None
    slice_12: "f32[1, 28, 28]" = torch.ops.aten.slice.Tensor(slice_11, 2, 0, 9223372036854775807);  slice_11 = None
    unsqueeze_2: "f32[1, 28, 28, 1]" = torch.ops.aten.unsqueeze.default(slice_12, 3);  slice_12 = None
    div_5: "f32[1, 28, 28, 32]" = torch.ops.aten.div.Tensor(unsqueeze_2, pow_1);  unsqueeze_2 = pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:59, code: pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
    slice_13: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(div_4, 0, 0, 9223372036854775807)
    slice_14: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    slice_15: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_14, 2, 0, 9223372036854775807);  slice_14 = None
    slice_16: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 9223372036854775807, 2);  slice_15 = None
    sin: "f32[1, 28, 28, 16]" = torch.ops.aten.sin.default(slice_16);  slice_16 = None
    slice_17: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(div_4, 0, 0, 9223372036854775807);  div_4 = None
    slice_18: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_17, 1, 0, 9223372036854775807);  slice_17 = None
    slice_19: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_18, 2, 0, 9223372036854775807);  slice_18 = None
    slice_20: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(slice_19, 3, 1, 9223372036854775807, 2);  slice_19 = None
    cos: "f32[1, 28, 28, 16]" = torch.ops.aten.cos.default(slice_20);  slice_20 = None
    stack: "f32[1, 28, 28, 16, 2]" = torch.ops.aten.stack.default([sin, cos], 4);  sin = cos = None
    view_1: "f32[1, 28, 28, 32]" = torch.ops.aten.view.default(stack, [1, 28, 28, 32]);  stack = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:60, code: pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
    slice_21: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(div_5, 0, 0, 9223372036854775807)
    slice_22: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    slice_23: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_22, 2, 0, 9223372036854775807);  slice_22 = None
    slice_24: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 9223372036854775807, 2);  slice_23 = None
    sin_1: "f32[1, 28, 28, 16]" = torch.ops.aten.sin.default(slice_24);  slice_24 = None
    slice_25: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(div_5, 0, 0, 9223372036854775807);  div_5 = None
    slice_26: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_25, 1, 0, 9223372036854775807);  slice_25 = None
    slice_27: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_26, 2, 0, 9223372036854775807);  slice_26 = None
    slice_28: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(slice_27, 3, 1, 9223372036854775807, 2);  slice_27 = None
    cos_1: "f32[1, 28, 28, 16]" = torch.ops.aten.cos.default(slice_28);  slice_28 = None
    stack_1: "f32[1, 28, 28, 16, 2]" = torch.ops.aten.stack.default([sin_1, cos_1], 4);  sin_1 = cos_1 = None
    view_2: "f32[1, 28, 28, 32]" = torch.ops.aten.view.default(stack_1, [1, 28, 28, 32]);  stack_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:61, code: pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    cat: "f32[1, 28, 28, 64]" = torch.ops.aten.cat.default([view_2, view_1], 3);  view_2 = view_1 = None
    permute: "f32[1, 64, 28, 28]" = torch.ops.aten.permute.default(cat, [0, 3, 1, 2]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:62, code: pos = self.token_projection(pos)
    convolution_3: "f32[1, 768, 28, 28]" = torch.ops.aten.convolution.default(permute, primals_111, primals_112, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:63, code: return pos.repeat(B, 1, 1, 1)  # (B, C, H, W)
    repeat_2: "f32[8, 768, 28, 28]" = torch.ops.aten.repeat.default(convolution_3, [8, 1, 1, 1]);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:437, code: pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
    view_3: "f32[8, 768, 784]" = torch.ops.aten.view.default(repeat_2, [8, -1, 784]);  repeat_2 = None
    permute_1: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:438, code: x = x + pos_encoding
    add_5: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(transpose, permute_1);  transpose = permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:439, code: x = self.pos_drop(x)
    clone: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm = torch.ops.aten.native_layer_norm.default(clone, [768], primals_113, primals_114, 1e-06)
    getitem_15: "f32[8, 784, 768]" = native_layer_norm[0]
    getitem_16: "f32[8, 784, 1]" = native_layer_norm[1]
    getitem_17: "f32[8, 784, 1]" = native_layer_norm[2];  native_layer_norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_4: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_15, [6272, 768]);  getitem_15 = None
    t: "f32[768, 2304]" = torch.ops.aten.t.default(primals_115);  primals_115 = None
    addmm: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_116, view_4, t);  primals_116 = None
    view_5: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm, [8, 784, 2304]);  addmm = None
    view_6: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_5, [8, 784, 3, 16, 48]);  view_5 = None
    permute_2: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_6, [2, 0, 3, 4, 1]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
    getitem_18: "f32[8, 16, 48, 784]" = unbind[0]
    getitem_19: "f32[8, 16, 48, 784]" = unbind[1]
    getitem_20: "f32[8, 16, 48, 784]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_18, 2.0, [-1], True)
    detach: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm)
    clamp_min: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm, 1e-12)
    expand: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min, [8, 16, 48, 784]);  clamp_min = None
    div_6: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_18, expand)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_1: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_19, 2.0, [-1], True)
    detach_1: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_1)
    clamp_min_1: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_1, 1e-12)
    expand_1: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_1, [8, 16, 48, 784]);  clamp_min_1 = None
    div_7: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_19, expand_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_1: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_7, -2, -1);  div_7 = None
    expand_2: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_6, [8, 16, 48, 784]);  div_6 = None
    clone_1: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    _unsafe_view: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_1, [128, 48, 784]);  clone_1 = None
    expand_3: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_1, [8, 16, 784, 48]);  transpose_1 = None
    clone_2: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    _unsafe_view_1: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_2, [128, 784, 48]);  clone_2 = None
    bmm: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view, _unsafe_view_1)
    view_7: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm, [8, 16, 48, 48]);  bmm = None
    mul_3: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_7, primals_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_3, -1, False);  mul_3 = None
    detach_2: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_3: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax);  _softmax = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_4: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_3, [8, 16, 48, 48]);  clone_3 = None
    view_8: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_4, [128, 48, 48]);  expand_4 = None
    expand_5: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_20, [8, 16, 48, 784]);  getitem_20 = None
    clone_4: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    _unsafe_view_2: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_4, [128, 48, 784]);  clone_4 = None
    bmm_1: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_8, _unsafe_view_2)
    view_9: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_1, [8, 16, 48, 784]);  bmm_1 = None
    permute_3: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_9, [0, 3, 1, 2]);  view_9 = None
    view_10: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_3, [8, 784, 768]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_1: "f32[768, 768]" = torch.ops.aten.t.default(primals_117);  primals_117 = None
    clone_5: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_10, memory_format = torch.contiguous_format);  view_10 = None
    _unsafe_view_3: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_5, [6272, 768]);  clone_5 = None
    mm: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_3, t_1)
    view_11: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm, [8, 784, 768]);  mm = None
    add_6: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_11, primals_118);  view_11 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_6: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_6);  add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_4: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_1, clone_6)
    add_7: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(clone, mul_4);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_1 = torch.ops.aten.native_layer_norm.default(add_7, [768], primals_119, primals_120, 1e-06)
    getitem_21: "f32[8, 784, 768]" = native_layer_norm_1[0]
    getitem_22: "f32[8, 784, 1]" = native_layer_norm_1[1]
    getitem_23: "f32[8, 784, 1]" = native_layer_norm_1[2];  native_layer_norm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_4: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1]);  getitem_21 = None
    view_12: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_4, [8, 768, 28, 28]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_4: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_12, primals_121, primals_122, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_2: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_8: "i64[]" = torch.ops.aten.add.Tensor(primals_640, 1);  primals_640 = None
    _native_batch_norm_legit_functional_3 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_2, primals_123, primals_124, primals_638, primals_639, True, 0.1, 1e-05);  primals_124 = primals_638 = primals_639 = None
    getitem_24: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_3[0]
    getitem_25: "f32[768]" = _native_batch_norm_legit_functional_3[1]
    getitem_26: "f32[768]" = _native_batch_norm_legit_functional_3[2]
    getitem_27: "f32[768]" = _native_batch_norm_legit_functional_3[3]
    getitem_28: "f32[768]" = _native_batch_norm_legit_functional_3[4];  _native_batch_norm_legit_functional_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_5: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_24, primals_125, primals_126, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_13: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_5, [8, 768, 784]);  convolution_5 = None
    permute_5: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_5: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_3, permute_5)
    add_9: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_7, mul_5);  mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_2 = torch.ops.aten.native_layer_norm.default(add_9, [768], primals_127, primals_128, 1e-06)
    getitem_29: "f32[8, 784, 768]" = native_layer_norm_2[0]
    getitem_30: "f32[8, 784, 1]" = native_layer_norm_2[1]
    getitem_31: "f32[8, 784, 1]" = native_layer_norm_2[2];  native_layer_norm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_14: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_29, [6272, 768]);  getitem_29 = None
    t_2: "f32[768, 3072]" = torch.ops.aten.t.default(primals_129);  primals_129 = None
    addmm_1: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_130, view_14, t_2);  primals_130 = None
    view_15: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_1, [8, 784, 3072]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_3: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_7: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_3);  gelu_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_16: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_7, [6272, 3072]);  clone_7 = None
    t_3: "f32[3072, 768]" = torch.ops.aten.t.default(primals_131);  primals_131 = None
    addmm_2: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_132, view_16, t_3);  primals_132 = None
    view_17: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_2, [8, 784, 768]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_8: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_17);  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_6: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_4, clone_8)
    add_10: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_9, mul_6);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_3 = torch.ops.aten.native_layer_norm.default(add_10, [768], primals_133, primals_134, 1e-06)
    getitem_32: "f32[8, 784, 768]" = native_layer_norm_3[0]
    getitem_33: "f32[8, 784, 1]" = native_layer_norm_3[1]
    getitem_34: "f32[8, 784, 1]" = native_layer_norm_3[2];  native_layer_norm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_18: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_32, [6272, 768]);  getitem_32 = None
    t_4: "f32[768, 2304]" = torch.ops.aten.t.default(primals_135);  primals_135 = None
    addmm_3: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_136, view_18, t_4);  primals_136 = None
    view_19: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_3, [8, 784, 2304]);  addmm_3 = None
    view_20: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_19, [8, 784, 3, 16, 48]);  view_19 = None
    permute_6: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_20, [2, 0, 3, 4, 1]);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_1 = torch.ops.aten.unbind.int(permute_6);  permute_6 = None
    getitem_35: "f32[8, 16, 48, 784]" = unbind_1[0]
    getitem_36: "f32[8, 16, 48, 784]" = unbind_1[1]
    getitem_37: "f32[8, 16, 48, 784]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_2: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_35, 2.0, [-1], True)
    detach_3: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_2)
    clamp_min_2: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_2, 1e-12)
    expand_6: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_2, [8, 16, 48, 784]);  clamp_min_2 = None
    div_8: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_35, expand_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_3: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_36, 2.0, [-1], True)
    detach_4: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_3)
    clamp_min_3: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_3, 1e-12)
    expand_7: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_3, [8, 16, 48, 784]);  clamp_min_3 = None
    div_9: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_36, expand_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_2: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_9, -2, -1);  div_9 = None
    expand_8: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_8, [8, 16, 48, 784]);  div_8 = None
    clone_9: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    _unsafe_view_4: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_9, [128, 48, 784]);  clone_9 = None
    expand_9: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_2, [8, 16, 784, 48]);  transpose_2 = None
    clone_10: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    _unsafe_view_5: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_10, [128, 784, 48]);  clone_10 = None
    bmm_2: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_4, _unsafe_view_5)
    view_21: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_2, [8, 16, 48, 48]);  bmm_2 = None
    mul_7: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_21, primals_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_1: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_7, -1, False);  mul_7 = None
    detach_5: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_11: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_1);  _softmax_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_10: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_11, [8, 16, 48, 48]);  clone_11 = None
    view_22: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_10, [128, 48, 48]);  expand_10 = None
    expand_11: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_37, [8, 16, 48, 784]);  getitem_37 = None
    clone_12: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    _unsafe_view_6: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_12, [128, 48, 784]);  clone_12 = None
    bmm_3: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_22, _unsafe_view_6)
    view_23: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_3, [8, 16, 48, 784]);  bmm_3 = None
    permute_7: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_23, [0, 3, 1, 2]);  view_23 = None
    view_24: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_7, [8, 784, 768]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_5: "f32[768, 768]" = torch.ops.aten.t.default(primals_137);  primals_137 = None
    clone_13: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_24, memory_format = torch.contiguous_format);  view_24 = None
    _unsafe_view_7: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_13, [6272, 768]);  clone_13 = None
    mm_1: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_7, t_5)
    view_25: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_1, [8, 784, 768]);  mm_1 = None
    add_11: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_25, primals_138);  view_25 = primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_14: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_11);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_8: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_5, clone_14)
    add_12: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_10, mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_4 = torch.ops.aten.native_layer_norm.default(add_12, [768], primals_139, primals_140, 1e-06)
    getitem_38: "f32[8, 784, 768]" = native_layer_norm_4[0]
    getitem_39: "f32[8, 784, 1]" = native_layer_norm_4[1]
    getitem_40: "f32[8, 784, 1]" = native_layer_norm_4[2];  native_layer_norm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_8: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1]);  getitem_38 = None
    view_26: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_8, [8, 768, 28, 28]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_6: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_26, primals_141, primals_142, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_4: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_13: "i64[]" = torch.ops.aten.add.Tensor(primals_643, 1);  primals_643 = None
    _native_batch_norm_legit_functional_4 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_4, primals_143, primals_144, primals_641, primals_642, True, 0.1, 1e-05);  primals_144 = primals_641 = primals_642 = None
    getitem_41: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_4[0]
    getitem_42: "f32[768]" = _native_batch_norm_legit_functional_4[1]
    getitem_43: "f32[768]" = _native_batch_norm_legit_functional_4[2]
    getitem_44: "f32[768]" = _native_batch_norm_legit_functional_4[3]
    getitem_45: "f32[768]" = _native_batch_norm_legit_functional_4[4];  _native_batch_norm_legit_functional_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_7: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_41, primals_145, primals_146, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_27: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_7, [8, 768, 784]);  convolution_7 = None
    permute_9: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_9: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_7, permute_9)
    add_14: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_12, mul_9);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_5 = torch.ops.aten.native_layer_norm.default(add_14, [768], primals_147, primals_148, 1e-06)
    getitem_46: "f32[8, 784, 768]" = native_layer_norm_5[0]
    getitem_47: "f32[8, 784, 1]" = native_layer_norm_5[1]
    getitem_48: "f32[8, 784, 1]" = native_layer_norm_5[2];  native_layer_norm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_28: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_46, [6272, 768]);  getitem_46 = None
    t_6: "f32[768, 3072]" = torch.ops.aten.t.default(primals_149);  primals_149 = None
    addmm_4: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_150, view_28, t_6);  primals_150 = None
    view_29: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_4, [8, 784, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_5: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_15: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_5);  gelu_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_30: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_15, [6272, 3072]);  clone_15 = None
    t_7: "f32[3072, 768]" = torch.ops.aten.t.default(primals_151);  primals_151 = None
    addmm_5: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_152, view_30, t_7);  primals_152 = None
    view_31: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_5, [8, 784, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_16: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_31);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_10: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_8, clone_16)
    add_15: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_14, mul_10);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_6 = torch.ops.aten.native_layer_norm.default(add_15, [768], primals_153, primals_154, 1e-06)
    getitem_49: "f32[8, 784, 768]" = native_layer_norm_6[0]
    getitem_50: "f32[8, 784, 1]" = native_layer_norm_6[1]
    getitem_51: "f32[8, 784, 1]" = native_layer_norm_6[2];  native_layer_norm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_32: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_49, [6272, 768]);  getitem_49 = None
    t_8: "f32[768, 2304]" = torch.ops.aten.t.default(primals_155);  primals_155 = None
    addmm_6: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_156, view_32, t_8);  primals_156 = None
    view_33: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_6, [8, 784, 2304]);  addmm_6 = None
    view_34: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_33, [8, 784, 3, 16, 48]);  view_33 = None
    permute_10: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_34, [2, 0, 3, 4, 1]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_2 = torch.ops.aten.unbind.int(permute_10);  permute_10 = None
    getitem_52: "f32[8, 16, 48, 784]" = unbind_2[0]
    getitem_53: "f32[8, 16, 48, 784]" = unbind_2[1]
    getitem_54: "f32[8, 16, 48, 784]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_4: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_52, 2.0, [-1], True)
    detach_6: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_4)
    clamp_min_4: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_4, 1e-12)
    expand_12: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_4, [8, 16, 48, 784]);  clamp_min_4 = None
    div_10: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_52, expand_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_5: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_53, 2.0, [-1], True)
    detach_7: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_5)
    clamp_min_5: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_5, 1e-12)
    expand_13: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_5, [8, 16, 48, 784]);  clamp_min_5 = None
    div_11: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_53, expand_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_3: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_11, -2, -1);  div_11 = None
    expand_14: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_10, [8, 16, 48, 784]);  div_10 = None
    clone_17: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    _unsafe_view_8: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_17, [128, 48, 784]);  clone_17 = None
    expand_15: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_3, [8, 16, 784, 48]);  transpose_3 = None
    clone_18: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    _unsafe_view_9: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_18, [128, 784, 48]);  clone_18 = None
    bmm_4: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_8, _unsafe_view_9)
    view_35: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_4, [8, 16, 48, 48]);  bmm_4 = None
    mul_11: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_35, primals_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_2: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_11, -1, False);  mul_11 = None
    detach_8: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_19: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_2);  _softmax_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_16: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_19, [8, 16, 48, 48]);  clone_19 = None
    view_36: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_16, [128, 48, 48]);  expand_16 = None
    expand_17: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_54, [8, 16, 48, 784]);  getitem_54 = None
    clone_20: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    _unsafe_view_10: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_20, [128, 48, 784]);  clone_20 = None
    bmm_5: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_36, _unsafe_view_10)
    view_37: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_5, [8, 16, 48, 784]);  bmm_5 = None
    permute_11: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_37, [0, 3, 1, 2]);  view_37 = None
    view_38: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_11, [8, 784, 768]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_9: "f32[768, 768]" = torch.ops.aten.t.default(primals_157);  primals_157 = None
    clone_21: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_38, memory_format = torch.contiguous_format);  view_38 = None
    _unsafe_view_11: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_21, [6272, 768]);  clone_21 = None
    mm_2: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_11, t_9)
    view_39: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_2, [8, 784, 768]);  mm_2 = None
    add_16: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_39, primals_158);  view_39 = primals_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_22: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_16);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_12: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_9, clone_22)
    add_17: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_15, mul_12);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_7 = torch.ops.aten.native_layer_norm.default(add_17, [768], primals_159, primals_160, 1e-06)
    getitem_55: "f32[8, 784, 768]" = native_layer_norm_7[0]
    getitem_56: "f32[8, 784, 1]" = native_layer_norm_7[1]
    getitem_57: "f32[8, 784, 1]" = native_layer_norm_7[2];  native_layer_norm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_12: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_55, [0, 2, 1]);  getitem_55 = None
    view_40: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_12, [8, 768, 28, 28]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_8: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_40, primals_161, primals_162, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_6: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_18: "i64[]" = torch.ops.aten.add.Tensor(primals_646, 1);  primals_646 = None
    _native_batch_norm_legit_functional_5 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_6, primals_163, primals_164, primals_644, primals_645, True, 0.1, 1e-05);  primals_164 = primals_644 = primals_645 = None
    getitem_58: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_5[0]
    getitem_59: "f32[768]" = _native_batch_norm_legit_functional_5[1]
    getitem_60: "f32[768]" = _native_batch_norm_legit_functional_5[2]
    getitem_61: "f32[768]" = _native_batch_norm_legit_functional_5[3]
    getitem_62: "f32[768]" = _native_batch_norm_legit_functional_5[4];  _native_batch_norm_legit_functional_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_9: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_58, primals_165, primals_166, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_41: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_9, [8, 768, 784]);  convolution_9 = None
    permute_13: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_13: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_11, permute_13)
    add_19: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_17, mul_13);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_8 = torch.ops.aten.native_layer_norm.default(add_19, [768], primals_167, primals_168, 1e-06)
    getitem_63: "f32[8, 784, 768]" = native_layer_norm_8[0]
    getitem_64: "f32[8, 784, 1]" = native_layer_norm_8[1]
    getitem_65: "f32[8, 784, 1]" = native_layer_norm_8[2];  native_layer_norm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_42: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_63, [6272, 768]);  getitem_63 = None
    t_10: "f32[768, 3072]" = torch.ops.aten.t.default(primals_169);  primals_169 = None
    addmm_7: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_170, view_42, t_10);  primals_170 = None
    view_43: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_7, [8, 784, 3072]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_7: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_7);  gelu_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_44: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_23, [6272, 3072]);  clone_23 = None
    t_11: "f32[3072, 768]" = torch.ops.aten.t.default(primals_171);  primals_171 = None
    addmm_8: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_172, view_44, t_11);  primals_172 = None
    view_45: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_8, [8, 784, 768]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_45);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_14: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_12, clone_24)
    add_20: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_19, mul_14);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_9 = torch.ops.aten.native_layer_norm.default(add_20, [768], primals_173, primals_174, 1e-06)
    getitem_66: "f32[8, 784, 768]" = native_layer_norm_9[0]
    getitem_67: "f32[8, 784, 1]" = native_layer_norm_9[1]
    getitem_68: "f32[8, 784, 1]" = native_layer_norm_9[2];  native_layer_norm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_46: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_66, [6272, 768]);  getitem_66 = None
    t_12: "f32[768, 2304]" = torch.ops.aten.t.default(primals_175);  primals_175 = None
    addmm_9: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_176, view_46, t_12);  primals_176 = None
    view_47: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_9, [8, 784, 2304]);  addmm_9 = None
    view_48: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_47, [8, 784, 3, 16, 48]);  view_47 = None
    permute_14: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_48, [2, 0, 3, 4, 1]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_3 = torch.ops.aten.unbind.int(permute_14);  permute_14 = None
    getitem_69: "f32[8, 16, 48, 784]" = unbind_3[0]
    getitem_70: "f32[8, 16, 48, 784]" = unbind_3[1]
    getitem_71: "f32[8, 16, 48, 784]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_6: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_69, 2.0, [-1], True)
    detach_9: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_6)
    clamp_min_6: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_6, 1e-12)
    expand_18: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_6, [8, 16, 48, 784]);  clamp_min_6 = None
    div_12: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_69, expand_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_7: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_70, 2.0, [-1], True)
    detach_10: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_7)
    clamp_min_7: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_7, 1e-12)
    expand_19: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_7, [8, 16, 48, 784]);  clamp_min_7 = None
    div_13: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_70, expand_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_4: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_13, -2, -1);  div_13 = None
    expand_20: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_12, [8, 16, 48, 784]);  div_12 = None
    clone_25: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    _unsafe_view_12: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_25, [128, 48, 784]);  clone_25 = None
    expand_21: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_4, [8, 16, 784, 48]);  transpose_4 = None
    clone_26: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    _unsafe_view_13: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_26, [128, 784, 48]);  clone_26 = None
    bmm_6: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_12, _unsafe_view_13)
    view_49: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_6, [8, 16, 48, 48]);  bmm_6 = None
    mul_15: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_49, primals_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_3: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_15, -1, False);  mul_15 = None
    detach_11: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_27: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_3);  _softmax_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_22: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_27, [8, 16, 48, 48]);  clone_27 = None
    view_50: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_22, [128, 48, 48]);  expand_22 = None
    expand_23: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_71, [8, 16, 48, 784]);  getitem_71 = None
    clone_28: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    _unsafe_view_14: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_28, [128, 48, 784]);  clone_28 = None
    bmm_7: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_50, _unsafe_view_14)
    view_51: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_7, [8, 16, 48, 784]);  bmm_7 = None
    permute_15: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_51, [0, 3, 1, 2]);  view_51 = None
    view_52: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_15, [8, 784, 768]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_13: "f32[768, 768]" = torch.ops.aten.t.default(primals_177);  primals_177 = None
    clone_29: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_52, memory_format = torch.contiguous_format);  view_52 = None
    _unsafe_view_15: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_29, [6272, 768]);  clone_29 = None
    mm_3: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_15, t_13)
    view_53: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_3, [8, 784, 768]);  mm_3 = None
    add_21: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_53, primals_178);  view_53 = primals_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_30: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_16: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_13, clone_30)
    add_22: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_20, mul_16);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_10 = torch.ops.aten.native_layer_norm.default(add_22, [768], primals_179, primals_180, 1e-06)
    getitem_72: "f32[8, 784, 768]" = native_layer_norm_10[0]
    getitem_73: "f32[8, 784, 1]" = native_layer_norm_10[1]
    getitem_74: "f32[8, 784, 1]" = native_layer_norm_10[2];  native_layer_norm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_16: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1]);  getitem_72 = None
    view_54: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_16, [8, 768, 28, 28]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_10: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_54, primals_181, primals_182, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_8: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_23: "i64[]" = torch.ops.aten.add.Tensor(primals_649, 1);  primals_649 = None
    _native_batch_norm_legit_functional_6 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_8, primals_183, primals_184, primals_647, primals_648, True, 0.1, 1e-05);  primals_184 = primals_647 = primals_648 = None
    getitem_75: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_6[0]
    getitem_76: "f32[768]" = _native_batch_norm_legit_functional_6[1]
    getitem_77: "f32[768]" = _native_batch_norm_legit_functional_6[2]
    getitem_78: "f32[768]" = _native_batch_norm_legit_functional_6[3]
    getitem_79: "f32[768]" = _native_batch_norm_legit_functional_6[4];  _native_batch_norm_legit_functional_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_11: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_75, primals_185, primals_186, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_55: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_11, [8, 768, 784]);  convolution_11 = None
    permute_17: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_17: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_15, permute_17)
    add_24: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_22, mul_17);  mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_11 = torch.ops.aten.native_layer_norm.default(add_24, [768], primals_187, primals_188, 1e-06)
    getitem_80: "f32[8, 784, 768]" = native_layer_norm_11[0]
    getitem_81: "f32[8, 784, 1]" = native_layer_norm_11[1]
    getitem_82: "f32[8, 784, 1]" = native_layer_norm_11[2];  native_layer_norm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_56: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_80, [6272, 768]);  getitem_80 = None
    t_14: "f32[768, 3072]" = torch.ops.aten.t.default(primals_189);  primals_189 = None
    addmm_10: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_190, view_56, t_14);  primals_190 = None
    view_57: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 784, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_9: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_31: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_9);  gelu_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_58: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_31, [6272, 3072]);  clone_31 = None
    t_15: "f32[3072, 768]" = torch.ops.aten.t.default(primals_191);  primals_191 = None
    addmm_11: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_192, view_58, t_15);  primals_192 = None
    view_59: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_11, [8, 784, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_32: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_59);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_18: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_16, clone_32)
    add_25: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_24, mul_18);  mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_12 = torch.ops.aten.native_layer_norm.default(add_25, [768], primals_193, primals_194, 1e-06)
    getitem_83: "f32[8, 784, 768]" = native_layer_norm_12[0]
    getitem_84: "f32[8, 784, 1]" = native_layer_norm_12[1]
    getitem_85: "f32[8, 784, 1]" = native_layer_norm_12[2];  native_layer_norm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_60: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_83, [6272, 768]);  getitem_83 = None
    t_16: "f32[768, 2304]" = torch.ops.aten.t.default(primals_195);  primals_195 = None
    addmm_12: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_196, view_60, t_16);  primals_196 = None
    view_61: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_12, [8, 784, 2304]);  addmm_12 = None
    view_62: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_61, [8, 784, 3, 16, 48]);  view_61 = None
    permute_18: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_62, [2, 0, 3, 4, 1]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_4 = torch.ops.aten.unbind.int(permute_18);  permute_18 = None
    getitem_86: "f32[8, 16, 48, 784]" = unbind_4[0]
    getitem_87: "f32[8, 16, 48, 784]" = unbind_4[1]
    getitem_88: "f32[8, 16, 48, 784]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_8: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_86, 2.0, [-1], True)
    detach_12: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_8)
    clamp_min_8: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_8, 1e-12)
    expand_24: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_8, [8, 16, 48, 784]);  clamp_min_8 = None
    div_14: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_86, expand_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_9: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_87, 2.0, [-1], True)
    detach_13: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_9)
    clamp_min_9: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_9, 1e-12)
    expand_25: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_9, [8, 16, 48, 784]);  clamp_min_9 = None
    div_15: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_87, expand_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_5: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_15, -2, -1);  div_15 = None
    expand_26: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_14, [8, 16, 48, 784]);  div_14 = None
    clone_33: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    _unsafe_view_16: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_33, [128, 48, 784]);  clone_33 = None
    expand_27: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_5, [8, 16, 784, 48]);  transpose_5 = None
    clone_34: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    _unsafe_view_17: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_34, [128, 784, 48]);  clone_34 = None
    bmm_8: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_16, _unsafe_view_17)
    view_63: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_8, [8, 16, 48, 48]);  bmm_8 = None
    mul_19: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_63, primals_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_4: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_19, -1, False);  mul_19 = None
    detach_14: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_35: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_4);  _softmax_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_28: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_35, [8, 16, 48, 48]);  clone_35 = None
    view_64: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_28, [128, 48, 48]);  expand_28 = None
    expand_29: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_88, [8, 16, 48, 784]);  getitem_88 = None
    clone_36: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    _unsafe_view_18: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_36, [128, 48, 784]);  clone_36 = None
    bmm_9: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_64, _unsafe_view_18)
    view_65: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_9, [8, 16, 48, 784]);  bmm_9 = None
    permute_19: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_65, [0, 3, 1, 2]);  view_65 = None
    view_66: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_19, [8, 784, 768]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_17: "f32[768, 768]" = torch.ops.aten.t.default(primals_197);  primals_197 = None
    clone_37: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_66, memory_format = torch.contiguous_format);  view_66 = None
    _unsafe_view_19: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_37, [6272, 768]);  clone_37 = None
    mm_4: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_19, t_17)
    view_67: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_4, [8, 784, 768]);  mm_4 = None
    add_26: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_67, primals_198);  view_67 = primals_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_38: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_26);  add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_20: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_17, clone_38)
    add_27: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_25, mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_13 = torch.ops.aten.native_layer_norm.default(add_27, [768], primals_199, primals_200, 1e-06)
    getitem_89: "f32[8, 784, 768]" = native_layer_norm_13[0]
    getitem_90: "f32[8, 784, 1]" = native_layer_norm_13[1]
    getitem_91: "f32[8, 784, 1]" = native_layer_norm_13[2];  native_layer_norm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_20: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_89, [0, 2, 1]);  getitem_89 = None
    view_68: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_20, [8, 768, 28, 28]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_12: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_68, primals_201, primals_202, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_10: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_28: "i64[]" = torch.ops.aten.add.Tensor(primals_652, 1);  primals_652 = None
    _native_batch_norm_legit_functional_7 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_10, primals_203, primals_204, primals_650, primals_651, True, 0.1, 1e-05);  primals_204 = primals_650 = primals_651 = None
    getitem_92: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_7[0]
    getitem_93: "f32[768]" = _native_batch_norm_legit_functional_7[1]
    getitem_94: "f32[768]" = _native_batch_norm_legit_functional_7[2]
    getitem_95: "f32[768]" = _native_batch_norm_legit_functional_7[3]
    getitem_96: "f32[768]" = _native_batch_norm_legit_functional_7[4];  _native_batch_norm_legit_functional_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_13: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_92, primals_205, primals_206, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_69: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_13, [8, 768, 784]);  convolution_13 = None
    permute_21: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_21: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_19, permute_21)
    add_29: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_27, mul_21);  mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_14 = torch.ops.aten.native_layer_norm.default(add_29, [768], primals_207, primals_208, 1e-06)
    getitem_97: "f32[8, 784, 768]" = native_layer_norm_14[0]
    getitem_98: "f32[8, 784, 1]" = native_layer_norm_14[1]
    getitem_99: "f32[8, 784, 1]" = native_layer_norm_14[2];  native_layer_norm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_70: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_97, [6272, 768]);  getitem_97 = None
    t_18: "f32[768, 3072]" = torch.ops.aten.t.default(primals_209);  primals_209 = None
    addmm_13: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_210, view_70, t_18);  primals_210 = None
    view_71: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_13, [8, 784, 3072]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_11: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_39: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_11);  gelu_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_72: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_39, [6272, 3072]);  clone_39 = None
    t_19: "f32[3072, 768]" = torch.ops.aten.t.default(primals_211);  primals_211 = None
    addmm_14: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_212, view_72, t_19);  primals_212 = None
    view_73: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_14, [8, 784, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_40: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_73);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_22: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_20, clone_40)
    add_30: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_29, mul_22);  mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_15 = torch.ops.aten.native_layer_norm.default(add_30, [768], primals_213, primals_214, 1e-06)
    getitem_100: "f32[8, 784, 768]" = native_layer_norm_15[0]
    getitem_101: "f32[8, 784, 1]" = native_layer_norm_15[1]
    getitem_102: "f32[8, 784, 1]" = native_layer_norm_15[2];  native_layer_norm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_74: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_100, [6272, 768]);  getitem_100 = None
    t_20: "f32[768, 2304]" = torch.ops.aten.t.default(primals_215);  primals_215 = None
    addmm_15: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_216, view_74, t_20);  primals_216 = None
    view_75: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_15, [8, 784, 2304]);  addmm_15 = None
    view_76: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_75, [8, 784, 3, 16, 48]);  view_75 = None
    permute_22: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_76, [2, 0, 3, 4, 1]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_5 = torch.ops.aten.unbind.int(permute_22);  permute_22 = None
    getitem_103: "f32[8, 16, 48, 784]" = unbind_5[0]
    getitem_104: "f32[8, 16, 48, 784]" = unbind_5[1]
    getitem_105: "f32[8, 16, 48, 784]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_10: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_103, 2.0, [-1], True)
    detach_15: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_10)
    clamp_min_10: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_10, 1e-12)
    expand_30: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_10, [8, 16, 48, 784]);  clamp_min_10 = None
    div_16: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_103, expand_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_11: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_104, 2.0, [-1], True)
    detach_16: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_11)
    clamp_min_11: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_11, 1e-12)
    expand_31: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_11, [8, 16, 48, 784]);  clamp_min_11 = None
    div_17: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_104, expand_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_6: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_17, -2, -1);  div_17 = None
    expand_32: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_16, [8, 16, 48, 784]);  div_16 = None
    clone_41: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    _unsafe_view_20: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_41, [128, 48, 784]);  clone_41 = None
    expand_33: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_6, [8, 16, 784, 48]);  transpose_6 = None
    clone_42: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    _unsafe_view_21: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_42, [128, 784, 48]);  clone_42 = None
    bmm_10: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_20, _unsafe_view_21)
    view_77: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_10, [8, 16, 48, 48]);  bmm_10 = None
    mul_23: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_77, primals_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_5: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_23, -1, False);  mul_23 = None
    detach_17: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_43: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_5);  _softmax_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_34: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_43, [8, 16, 48, 48]);  clone_43 = None
    view_78: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_34, [128, 48, 48]);  expand_34 = None
    expand_35: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_105, [8, 16, 48, 784]);  getitem_105 = None
    clone_44: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    _unsafe_view_22: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_44, [128, 48, 784]);  clone_44 = None
    bmm_11: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_78, _unsafe_view_22)
    view_79: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_11, [8, 16, 48, 784]);  bmm_11 = None
    permute_23: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_79, [0, 3, 1, 2]);  view_79 = None
    view_80: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_23, [8, 784, 768]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_21: "f32[768, 768]" = torch.ops.aten.t.default(primals_217);  primals_217 = None
    clone_45: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_80, memory_format = torch.contiguous_format);  view_80 = None
    _unsafe_view_23: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_45, [6272, 768]);  clone_45 = None
    mm_5: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_23, t_21)
    view_81: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_5, [8, 784, 768]);  mm_5 = None
    add_31: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_81, primals_218);  view_81 = primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_46: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_31);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_24: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_21, clone_46)
    add_32: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_30, mul_24);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_16 = torch.ops.aten.native_layer_norm.default(add_32, [768], primals_219, primals_220, 1e-06)
    getitem_106: "f32[8, 784, 768]" = native_layer_norm_16[0]
    getitem_107: "f32[8, 784, 1]" = native_layer_norm_16[1]
    getitem_108: "f32[8, 784, 1]" = native_layer_norm_16[2];  native_layer_norm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_24: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_106, [0, 2, 1]);  getitem_106 = None
    view_82: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_24, [8, 768, 28, 28]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_14: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_82, primals_221, primals_222, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_12: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_33: "i64[]" = torch.ops.aten.add.Tensor(primals_655, 1);  primals_655 = None
    _native_batch_norm_legit_functional_8 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_12, primals_223, primals_224, primals_653, primals_654, True, 0.1, 1e-05);  primals_224 = primals_653 = primals_654 = None
    getitem_109: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_8[0]
    getitem_110: "f32[768]" = _native_batch_norm_legit_functional_8[1]
    getitem_111: "f32[768]" = _native_batch_norm_legit_functional_8[2]
    getitem_112: "f32[768]" = _native_batch_norm_legit_functional_8[3]
    getitem_113: "f32[768]" = _native_batch_norm_legit_functional_8[4];  _native_batch_norm_legit_functional_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_15: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_109, primals_225, primals_226, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_83: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_15, [8, 768, 784]);  convolution_15 = None
    permute_25: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_25: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_23, permute_25)
    add_34: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_32, mul_25);  mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_17 = torch.ops.aten.native_layer_norm.default(add_34, [768], primals_227, primals_228, 1e-06)
    getitem_114: "f32[8, 784, 768]" = native_layer_norm_17[0]
    getitem_115: "f32[8, 784, 1]" = native_layer_norm_17[1]
    getitem_116: "f32[8, 784, 1]" = native_layer_norm_17[2];  native_layer_norm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_84: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_114, [6272, 768]);  getitem_114 = None
    t_22: "f32[768, 3072]" = torch.ops.aten.t.default(primals_229);  primals_229 = None
    addmm_16: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_230, view_84, t_22);  primals_230 = None
    view_85: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_16, [8, 784, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_13: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_85)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_47: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_13);  gelu_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_86: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_47, [6272, 3072]);  clone_47 = None
    t_23: "f32[3072, 768]" = torch.ops.aten.t.default(primals_231);  primals_231 = None
    addmm_17: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_232, view_86, t_23);  primals_232 = None
    view_87: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_17, [8, 784, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_48: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_87);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_26: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_24, clone_48)
    add_35: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_34, mul_26);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_18 = torch.ops.aten.native_layer_norm.default(add_35, [768], primals_233, primals_234, 1e-06)
    getitem_117: "f32[8, 784, 768]" = native_layer_norm_18[0]
    getitem_118: "f32[8, 784, 1]" = native_layer_norm_18[1]
    getitem_119: "f32[8, 784, 1]" = native_layer_norm_18[2];  native_layer_norm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_88: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_117, [6272, 768]);  getitem_117 = None
    t_24: "f32[768, 2304]" = torch.ops.aten.t.default(primals_235);  primals_235 = None
    addmm_18: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_236, view_88, t_24);  primals_236 = None
    view_89: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_18, [8, 784, 2304]);  addmm_18 = None
    view_90: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_89, [8, 784, 3, 16, 48]);  view_89 = None
    permute_26: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_90, [2, 0, 3, 4, 1]);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_6 = torch.ops.aten.unbind.int(permute_26);  permute_26 = None
    getitem_120: "f32[8, 16, 48, 784]" = unbind_6[0]
    getitem_121: "f32[8, 16, 48, 784]" = unbind_6[1]
    getitem_122: "f32[8, 16, 48, 784]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_12: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_120, 2.0, [-1], True)
    detach_18: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_12)
    clamp_min_12: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_12, 1e-12)
    expand_36: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_12, [8, 16, 48, 784]);  clamp_min_12 = None
    div_18: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_120, expand_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_13: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_121, 2.0, [-1], True)
    detach_19: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_13)
    clamp_min_13: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_13, 1e-12)
    expand_37: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_13, [8, 16, 48, 784]);  clamp_min_13 = None
    div_19: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_121, expand_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_7: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_19, -2, -1);  div_19 = None
    expand_38: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_18, [8, 16, 48, 784]);  div_18 = None
    clone_49: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    _unsafe_view_24: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_49, [128, 48, 784]);  clone_49 = None
    expand_39: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_7, [8, 16, 784, 48]);  transpose_7 = None
    clone_50: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    _unsafe_view_25: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_50, [128, 784, 48]);  clone_50 = None
    bmm_12: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_24, _unsafe_view_25)
    view_91: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_12, [8, 16, 48, 48]);  bmm_12 = None
    mul_27: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_91, primals_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_6: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_27, -1, False);  mul_27 = None
    detach_20: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_51: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_6);  _softmax_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_40: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_51, [8, 16, 48, 48]);  clone_51 = None
    view_92: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_40, [128, 48, 48]);  expand_40 = None
    expand_41: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_122, [8, 16, 48, 784]);  getitem_122 = None
    clone_52: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    _unsafe_view_26: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_52, [128, 48, 784]);  clone_52 = None
    bmm_13: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_92, _unsafe_view_26)
    view_93: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_13, [8, 16, 48, 784]);  bmm_13 = None
    permute_27: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_93, [0, 3, 1, 2]);  view_93 = None
    view_94: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_27, [8, 784, 768]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_25: "f32[768, 768]" = torch.ops.aten.t.default(primals_237);  primals_237 = None
    clone_53: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_94, memory_format = torch.contiguous_format);  view_94 = None
    _unsafe_view_27: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_53, [6272, 768]);  clone_53 = None
    mm_6: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_27, t_25)
    view_95: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_6, [8, 784, 768]);  mm_6 = None
    add_36: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_95, primals_238);  view_95 = primals_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_54: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_36);  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_28: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_25, clone_54)
    add_37: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_35, mul_28);  mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_19 = torch.ops.aten.native_layer_norm.default(add_37, [768], primals_239, primals_240, 1e-06)
    getitem_123: "f32[8, 784, 768]" = native_layer_norm_19[0]
    getitem_124: "f32[8, 784, 1]" = native_layer_norm_19[1]
    getitem_125: "f32[8, 784, 1]" = native_layer_norm_19[2];  native_layer_norm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_28: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_123, [0, 2, 1]);  getitem_123 = None
    view_96: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_28, [8, 768, 28, 28]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_16: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_96, primals_241, primals_242, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_14: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_38: "i64[]" = torch.ops.aten.add.Tensor(primals_658, 1);  primals_658 = None
    _native_batch_norm_legit_functional_9 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_14, primals_243, primals_244, primals_656, primals_657, True, 0.1, 1e-05);  primals_244 = primals_656 = primals_657 = None
    getitem_126: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_9[0]
    getitem_127: "f32[768]" = _native_batch_norm_legit_functional_9[1]
    getitem_128: "f32[768]" = _native_batch_norm_legit_functional_9[2]
    getitem_129: "f32[768]" = _native_batch_norm_legit_functional_9[3]
    getitem_130: "f32[768]" = _native_batch_norm_legit_functional_9[4];  _native_batch_norm_legit_functional_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_17: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_126, primals_245, primals_246, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_97: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_17, [8, 768, 784]);  convolution_17 = None
    permute_29: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_29: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_27, permute_29)
    add_39: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_37, mul_29);  mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_20 = torch.ops.aten.native_layer_norm.default(add_39, [768], primals_247, primals_248, 1e-06)
    getitem_131: "f32[8, 784, 768]" = native_layer_norm_20[0]
    getitem_132: "f32[8, 784, 1]" = native_layer_norm_20[1]
    getitem_133: "f32[8, 784, 1]" = native_layer_norm_20[2];  native_layer_norm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_98: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_131, [6272, 768]);  getitem_131 = None
    t_26: "f32[768, 3072]" = torch.ops.aten.t.default(primals_249);  primals_249 = None
    addmm_19: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_250, view_98, t_26);  primals_250 = None
    view_99: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_19, [8, 784, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_15: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_55: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_15);  gelu_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_100: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_55, [6272, 3072]);  clone_55 = None
    t_27: "f32[3072, 768]" = torch.ops.aten.t.default(primals_251);  primals_251 = None
    addmm_20: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_252, view_100, t_27);  primals_252 = None
    view_101: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_20, [8, 784, 768]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_56: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_101);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_30: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_28, clone_56)
    add_40: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_39, mul_30);  mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_21 = torch.ops.aten.native_layer_norm.default(add_40, [768], primals_253, primals_254, 1e-06)
    getitem_134: "f32[8, 784, 768]" = native_layer_norm_21[0]
    getitem_135: "f32[8, 784, 1]" = native_layer_norm_21[1]
    getitem_136: "f32[8, 784, 1]" = native_layer_norm_21[2];  native_layer_norm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_102: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_134, [6272, 768]);  getitem_134 = None
    t_28: "f32[768, 2304]" = torch.ops.aten.t.default(primals_255);  primals_255 = None
    addmm_21: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_256, view_102, t_28);  primals_256 = None
    view_103: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_21, [8, 784, 2304]);  addmm_21 = None
    view_104: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_103, [8, 784, 3, 16, 48]);  view_103 = None
    permute_30: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_104, [2, 0, 3, 4, 1]);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_7 = torch.ops.aten.unbind.int(permute_30);  permute_30 = None
    getitem_137: "f32[8, 16, 48, 784]" = unbind_7[0]
    getitem_138: "f32[8, 16, 48, 784]" = unbind_7[1]
    getitem_139: "f32[8, 16, 48, 784]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_14: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_137, 2.0, [-1], True)
    detach_21: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_14)
    clamp_min_14: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_14, 1e-12)
    expand_42: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_14, [8, 16, 48, 784]);  clamp_min_14 = None
    div_20: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_137, expand_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_15: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_138, 2.0, [-1], True)
    detach_22: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_15)
    clamp_min_15: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_15, 1e-12)
    expand_43: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_15, [8, 16, 48, 784]);  clamp_min_15 = None
    div_21: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_138, expand_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_8: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_21, -2, -1);  div_21 = None
    expand_44: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_20, [8, 16, 48, 784]);  div_20 = None
    clone_57: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    _unsafe_view_28: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_57, [128, 48, 784]);  clone_57 = None
    expand_45: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_8, [8, 16, 784, 48]);  transpose_8 = None
    clone_58: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    _unsafe_view_29: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_58, [128, 784, 48]);  clone_58 = None
    bmm_14: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_28, _unsafe_view_29)
    view_105: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_14, [8, 16, 48, 48]);  bmm_14 = None
    mul_31: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_105, primals_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_7: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_31, -1, False);  mul_31 = None
    detach_23: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_59: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_7);  _softmax_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_46: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_59, [8, 16, 48, 48]);  clone_59 = None
    view_106: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_46, [128, 48, 48]);  expand_46 = None
    expand_47: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_139, [8, 16, 48, 784]);  getitem_139 = None
    clone_60: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    _unsafe_view_30: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_60, [128, 48, 784]);  clone_60 = None
    bmm_15: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_106, _unsafe_view_30)
    view_107: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_15, [8, 16, 48, 784]);  bmm_15 = None
    permute_31: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_107, [0, 3, 1, 2]);  view_107 = None
    view_108: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_31, [8, 784, 768]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_29: "f32[768, 768]" = torch.ops.aten.t.default(primals_257);  primals_257 = None
    clone_61: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_108, memory_format = torch.contiguous_format);  view_108 = None
    _unsafe_view_31: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_61, [6272, 768]);  clone_61 = None
    mm_7: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_31, t_29)
    view_109: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_7, [8, 784, 768]);  mm_7 = None
    add_41: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_109, primals_258);  view_109 = primals_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_62: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_41);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_32: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_29, clone_62)
    add_42: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_40, mul_32);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_22 = torch.ops.aten.native_layer_norm.default(add_42, [768], primals_259, primals_260, 1e-06)
    getitem_140: "f32[8, 784, 768]" = native_layer_norm_22[0]
    getitem_141: "f32[8, 784, 1]" = native_layer_norm_22[1]
    getitem_142: "f32[8, 784, 1]" = native_layer_norm_22[2];  native_layer_norm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_32: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_140, [0, 2, 1]);  getitem_140 = None
    view_110: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_32, [8, 768, 28, 28]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_18: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_110, primals_261, primals_262, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_16: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_43: "i64[]" = torch.ops.aten.add.Tensor(primals_661, 1);  primals_661 = None
    _native_batch_norm_legit_functional_10 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_16, primals_263, primals_264, primals_659, primals_660, True, 0.1, 1e-05);  primals_264 = primals_659 = primals_660 = None
    getitem_143: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_10[0]
    getitem_144: "f32[768]" = _native_batch_norm_legit_functional_10[1]
    getitem_145: "f32[768]" = _native_batch_norm_legit_functional_10[2]
    getitem_146: "f32[768]" = _native_batch_norm_legit_functional_10[3]
    getitem_147: "f32[768]" = _native_batch_norm_legit_functional_10[4];  _native_batch_norm_legit_functional_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_19: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_143, primals_265, primals_266, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_111: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_19, [8, 768, 784]);  convolution_19 = None
    permute_33: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_33: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_31, permute_33)
    add_44: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_42, mul_33);  mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_23 = torch.ops.aten.native_layer_norm.default(add_44, [768], primals_267, primals_268, 1e-06)
    getitem_148: "f32[8, 784, 768]" = native_layer_norm_23[0]
    getitem_149: "f32[8, 784, 1]" = native_layer_norm_23[1]
    getitem_150: "f32[8, 784, 1]" = native_layer_norm_23[2];  native_layer_norm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_112: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_148, [6272, 768]);  getitem_148 = None
    t_30: "f32[768, 3072]" = torch.ops.aten.t.default(primals_269);  primals_269 = None
    addmm_22: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_270, view_112, t_30);  primals_270 = None
    view_113: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 784, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_17: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_113)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_63: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_17);  gelu_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_114: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_63, [6272, 3072]);  clone_63 = None
    t_31: "f32[3072, 768]" = torch.ops.aten.t.default(primals_271);  primals_271 = None
    addmm_23: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_272, view_114, t_31);  primals_272 = None
    view_115: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_23, [8, 784, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_64: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_115);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_34: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_32, clone_64)
    add_45: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_44, mul_34);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_24 = torch.ops.aten.native_layer_norm.default(add_45, [768], primals_273, primals_274, 1e-06)
    getitem_151: "f32[8, 784, 768]" = native_layer_norm_24[0]
    getitem_152: "f32[8, 784, 1]" = native_layer_norm_24[1]
    getitem_153: "f32[8, 784, 1]" = native_layer_norm_24[2];  native_layer_norm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_116: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_151, [6272, 768]);  getitem_151 = None
    t_32: "f32[768, 2304]" = torch.ops.aten.t.default(primals_275);  primals_275 = None
    addmm_24: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_276, view_116, t_32);  primals_276 = None
    view_117: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_24, [8, 784, 2304]);  addmm_24 = None
    view_118: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_117, [8, 784, 3, 16, 48]);  view_117 = None
    permute_34: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_118, [2, 0, 3, 4, 1]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_8 = torch.ops.aten.unbind.int(permute_34);  permute_34 = None
    getitem_154: "f32[8, 16, 48, 784]" = unbind_8[0]
    getitem_155: "f32[8, 16, 48, 784]" = unbind_8[1]
    getitem_156: "f32[8, 16, 48, 784]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_16: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_154, 2.0, [-1], True)
    detach_24: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_16)
    clamp_min_16: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_16, 1e-12)
    expand_48: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_16, [8, 16, 48, 784]);  clamp_min_16 = None
    div_22: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_154, expand_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_17: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_155, 2.0, [-1], True)
    detach_25: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_17)
    clamp_min_17: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_17, 1e-12)
    expand_49: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_17, [8, 16, 48, 784]);  clamp_min_17 = None
    div_23: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_155, expand_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_9: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_23, -2, -1);  div_23 = None
    expand_50: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_22, [8, 16, 48, 784]);  div_22 = None
    clone_65: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
    _unsafe_view_32: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_65, [128, 48, 784]);  clone_65 = None
    expand_51: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_9, [8, 16, 784, 48]);  transpose_9 = None
    clone_66: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    _unsafe_view_33: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_66, [128, 784, 48]);  clone_66 = None
    bmm_16: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_32, _unsafe_view_33)
    view_119: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_16, [8, 16, 48, 48]);  bmm_16 = None
    mul_35: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_119, primals_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_8: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_35, -1, False);  mul_35 = None
    detach_26: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_67: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_8);  _softmax_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_52: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_67, [8, 16, 48, 48]);  clone_67 = None
    view_120: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_52, [128, 48, 48]);  expand_52 = None
    expand_53: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_156, [8, 16, 48, 784]);  getitem_156 = None
    clone_68: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    _unsafe_view_34: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_68, [128, 48, 784]);  clone_68 = None
    bmm_17: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_120, _unsafe_view_34)
    view_121: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_17, [8, 16, 48, 784]);  bmm_17 = None
    permute_35: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_121, [0, 3, 1, 2]);  view_121 = None
    view_122: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_35, [8, 784, 768]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_33: "f32[768, 768]" = torch.ops.aten.t.default(primals_277);  primals_277 = None
    clone_69: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_122, memory_format = torch.contiguous_format);  view_122 = None
    _unsafe_view_35: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_69, [6272, 768]);  clone_69 = None
    mm_8: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_35, t_33)
    view_123: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_8, [8, 784, 768]);  mm_8 = None
    add_46: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_123, primals_278);  view_123 = primals_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_70: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_46);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_36: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_33, clone_70)
    add_47: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_45, mul_36);  mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_25 = torch.ops.aten.native_layer_norm.default(add_47, [768], primals_279, primals_280, 1e-06)
    getitem_157: "f32[8, 784, 768]" = native_layer_norm_25[0]
    getitem_158: "f32[8, 784, 1]" = native_layer_norm_25[1]
    getitem_159: "f32[8, 784, 1]" = native_layer_norm_25[2];  native_layer_norm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_36: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_157, [0, 2, 1]);  getitem_157 = None
    view_124: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_36, [8, 768, 28, 28]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_20: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_124, primals_281, primals_282, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_18: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_48: "i64[]" = torch.ops.aten.add.Tensor(primals_664, 1);  primals_664 = None
    _native_batch_norm_legit_functional_11 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_18, primals_283, primals_284, primals_662, primals_663, True, 0.1, 1e-05);  primals_284 = primals_662 = primals_663 = None
    getitem_160: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_11[0]
    getitem_161: "f32[768]" = _native_batch_norm_legit_functional_11[1]
    getitem_162: "f32[768]" = _native_batch_norm_legit_functional_11[2]
    getitem_163: "f32[768]" = _native_batch_norm_legit_functional_11[3]
    getitem_164: "f32[768]" = _native_batch_norm_legit_functional_11[4];  _native_batch_norm_legit_functional_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_21: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_160, primals_285, primals_286, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_125: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_21, [8, 768, 784]);  convolution_21 = None
    permute_37: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_37: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_35, permute_37)
    add_49: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_47, mul_37);  mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_26 = torch.ops.aten.native_layer_norm.default(add_49, [768], primals_287, primals_288, 1e-06)
    getitem_165: "f32[8, 784, 768]" = native_layer_norm_26[0]
    getitem_166: "f32[8, 784, 1]" = native_layer_norm_26[1]
    getitem_167: "f32[8, 784, 1]" = native_layer_norm_26[2];  native_layer_norm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_126: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_165, [6272, 768]);  getitem_165 = None
    t_34: "f32[768, 3072]" = torch.ops.aten.t.default(primals_289);  primals_289 = None
    addmm_25: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_290, view_126, t_34);  primals_290 = None
    view_127: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_25, [8, 784, 3072]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_19: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_127)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_71: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_19);  gelu_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_128: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_71, [6272, 3072]);  clone_71 = None
    t_35: "f32[3072, 768]" = torch.ops.aten.t.default(primals_291);  primals_291 = None
    addmm_26: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_292, view_128, t_35);  primals_292 = None
    view_129: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_26, [8, 784, 768]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_72: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_129);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_38: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_36, clone_72)
    add_50: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_49, mul_38);  mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_27 = torch.ops.aten.native_layer_norm.default(add_50, [768], primals_293, primals_294, 1e-06)
    getitem_168: "f32[8, 784, 768]" = native_layer_norm_27[0]
    getitem_169: "f32[8, 784, 1]" = native_layer_norm_27[1]
    getitem_170: "f32[8, 784, 1]" = native_layer_norm_27[2];  native_layer_norm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_130: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_168, [6272, 768]);  getitem_168 = None
    t_36: "f32[768, 2304]" = torch.ops.aten.t.default(primals_295);  primals_295 = None
    addmm_27: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_296, view_130, t_36);  primals_296 = None
    view_131: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_27, [8, 784, 2304]);  addmm_27 = None
    view_132: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_131, [8, 784, 3, 16, 48]);  view_131 = None
    permute_38: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_132, [2, 0, 3, 4, 1]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_9 = torch.ops.aten.unbind.int(permute_38);  permute_38 = None
    getitem_171: "f32[8, 16, 48, 784]" = unbind_9[0]
    getitem_172: "f32[8, 16, 48, 784]" = unbind_9[1]
    getitem_173: "f32[8, 16, 48, 784]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_18: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_171, 2.0, [-1], True)
    detach_27: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_18)
    clamp_min_18: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_18, 1e-12)
    expand_54: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_18, [8, 16, 48, 784]);  clamp_min_18 = None
    div_24: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_171, expand_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_19: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_172, 2.0, [-1], True)
    detach_28: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_19)
    clamp_min_19: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_19, 1e-12)
    expand_55: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_19, [8, 16, 48, 784]);  clamp_min_19 = None
    div_25: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_172, expand_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_10: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_25, -2, -1);  div_25 = None
    expand_56: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_24, [8, 16, 48, 784]);  div_24 = None
    clone_73: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    _unsafe_view_36: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_73, [128, 48, 784]);  clone_73 = None
    expand_57: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_10, [8, 16, 784, 48]);  transpose_10 = None
    clone_74: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    _unsafe_view_37: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_74, [128, 784, 48]);  clone_74 = None
    bmm_18: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_36, _unsafe_view_37)
    view_133: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_18, [8, 16, 48, 48]);  bmm_18 = None
    mul_39: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_133, primals_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_9: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_39, -1, False);  mul_39 = None
    detach_29: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_75: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_9);  _softmax_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_58: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_75, [8, 16, 48, 48]);  clone_75 = None
    view_134: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_58, [128, 48, 48]);  expand_58 = None
    expand_59: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_173, [8, 16, 48, 784]);  getitem_173 = None
    clone_76: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    _unsafe_view_38: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_76, [128, 48, 784]);  clone_76 = None
    bmm_19: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_134, _unsafe_view_38)
    view_135: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_19, [8, 16, 48, 784]);  bmm_19 = None
    permute_39: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_135, [0, 3, 1, 2]);  view_135 = None
    view_136: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_39, [8, 784, 768]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_37: "f32[768, 768]" = torch.ops.aten.t.default(primals_297);  primals_297 = None
    clone_77: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_136, memory_format = torch.contiguous_format);  view_136 = None
    _unsafe_view_39: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_77, [6272, 768]);  clone_77 = None
    mm_9: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_39, t_37)
    view_137: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_9, [8, 784, 768]);  mm_9 = None
    add_51: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_137, primals_298);  view_137 = primals_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_78: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_40: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_37, clone_78)
    add_52: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_50, mul_40);  mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_28 = torch.ops.aten.native_layer_norm.default(add_52, [768], primals_299, primals_300, 1e-06)
    getitem_174: "f32[8, 784, 768]" = native_layer_norm_28[0]
    getitem_175: "f32[8, 784, 1]" = native_layer_norm_28[1]
    getitem_176: "f32[8, 784, 1]" = native_layer_norm_28[2];  native_layer_norm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_40: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_174, [0, 2, 1]);  getitem_174 = None
    view_138: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_40, [8, 768, 28, 28]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_22: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_138, primals_301, primals_302, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_20: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_53: "i64[]" = torch.ops.aten.add.Tensor(primals_667, 1);  primals_667 = None
    _native_batch_norm_legit_functional_12 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_20, primals_303, primals_304, primals_665, primals_666, True, 0.1, 1e-05);  primals_304 = primals_665 = primals_666 = None
    getitem_177: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_12[0]
    getitem_178: "f32[768]" = _native_batch_norm_legit_functional_12[1]
    getitem_179: "f32[768]" = _native_batch_norm_legit_functional_12[2]
    getitem_180: "f32[768]" = _native_batch_norm_legit_functional_12[3]
    getitem_181: "f32[768]" = _native_batch_norm_legit_functional_12[4];  _native_batch_norm_legit_functional_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_23: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_177, primals_305, primals_306, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_139: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_23, [8, 768, 784]);  convolution_23 = None
    permute_41: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_139, [0, 2, 1]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_41: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_39, permute_41)
    add_54: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_52, mul_41);  mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_29 = torch.ops.aten.native_layer_norm.default(add_54, [768], primals_307, primals_308, 1e-06)
    getitem_182: "f32[8, 784, 768]" = native_layer_norm_29[0]
    getitem_183: "f32[8, 784, 1]" = native_layer_norm_29[1]
    getitem_184: "f32[8, 784, 1]" = native_layer_norm_29[2];  native_layer_norm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_140: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_182, [6272, 768]);  getitem_182 = None
    t_38: "f32[768, 3072]" = torch.ops.aten.t.default(primals_309);  primals_309 = None
    addmm_28: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_310, view_140, t_38);  primals_310 = None
    view_141: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_28, [8, 784, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_21: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_141)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_79: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_21);  gelu_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_142: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_79, [6272, 3072]);  clone_79 = None
    t_39: "f32[3072, 768]" = torch.ops.aten.t.default(primals_311);  primals_311 = None
    addmm_29: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_312, view_142, t_39);  primals_312 = None
    view_143: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_29, [8, 784, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_80: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_143);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_42: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_40, clone_80)
    add_55: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_54, mul_42);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_30 = torch.ops.aten.native_layer_norm.default(add_55, [768], primals_313, primals_314, 1e-06)
    getitem_185: "f32[8, 784, 768]" = native_layer_norm_30[0]
    getitem_186: "f32[8, 784, 1]" = native_layer_norm_30[1]
    getitem_187: "f32[8, 784, 1]" = native_layer_norm_30[2];  native_layer_norm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_144: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_185, [6272, 768]);  getitem_185 = None
    t_40: "f32[768, 2304]" = torch.ops.aten.t.default(primals_315);  primals_315 = None
    addmm_30: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_316, view_144, t_40);  primals_316 = None
    view_145: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_30, [8, 784, 2304]);  addmm_30 = None
    view_146: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_145, [8, 784, 3, 16, 48]);  view_145 = None
    permute_42: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_146, [2, 0, 3, 4, 1]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_10 = torch.ops.aten.unbind.int(permute_42);  permute_42 = None
    getitem_188: "f32[8, 16, 48, 784]" = unbind_10[0]
    getitem_189: "f32[8, 16, 48, 784]" = unbind_10[1]
    getitem_190: "f32[8, 16, 48, 784]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_20: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_188, 2.0, [-1], True)
    detach_30: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_20)
    clamp_min_20: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_20, 1e-12)
    expand_60: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_20, [8, 16, 48, 784]);  clamp_min_20 = None
    div_26: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_188, expand_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_21: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_189, 2.0, [-1], True)
    detach_31: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_21)
    clamp_min_21: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_21, 1e-12)
    expand_61: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_21, [8, 16, 48, 784]);  clamp_min_21 = None
    div_27: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_189, expand_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_11: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_27, -2, -1);  div_27 = None
    expand_62: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_26, [8, 16, 48, 784]);  div_26 = None
    clone_81: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
    _unsafe_view_40: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_81, [128, 48, 784]);  clone_81 = None
    expand_63: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_11, [8, 16, 784, 48]);  transpose_11 = None
    clone_82: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    _unsafe_view_41: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_82, [128, 784, 48]);  clone_82 = None
    bmm_20: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_40, _unsafe_view_41)
    view_147: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_20, [8, 16, 48, 48]);  bmm_20 = None
    mul_43: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_147, primals_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_10: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_43, -1, False);  mul_43 = None
    detach_32: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_83: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_10);  _softmax_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_64: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_83, [8, 16, 48, 48]);  clone_83 = None
    view_148: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_64, [128, 48, 48]);  expand_64 = None
    expand_65: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_190, [8, 16, 48, 784]);  getitem_190 = None
    clone_84: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    _unsafe_view_42: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_84, [128, 48, 784]);  clone_84 = None
    bmm_21: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_148, _unsafe_view_42)
    view_149: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_21, [8, 16, 48, 784]);  bmm_21 = None
    permute_43: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_149, [0, 3, 1, 2]);  view_149 = None
    view_150: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_43, [8, 784, 768]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_41: "f32[768, 768]" = torch.ops.aten.t.default(primals_317);  primals_317 = None
    clone_85: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_150, memory_format = torch.contiguous_format);  view_150 = None
    _unsafe_view_43: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_85, [6272, 768]);  clone_85 = None
    mm_10: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_43, t_41)
    view_151: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_10, [8, 784, 768]);  mm_10 = None
    add_56: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_151, primals_318);  view_151 = primals_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_86: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_56);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_44: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_41, clone_86)
    add_57: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_55, mul_44);  mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_31 = torch.ops.aten.native_layer_norm.default(add_57, [768], primals_319, primals_320, 1e-06)
    getitem_191: "f32[8, 784, 768]" = native_layer_norm_31[0]
    getitem_192: "f32[8, 784, 1]" = native_layer_norm_31[1]
    getitem_193: "f32[8, 784, 1]" = native_layer_norm_31[2];  native_layer_norm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_44: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_191, [0, 2, 1]);  getitem_191 = None
    view_152: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_44, [8, 768, 28, 28]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_24: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_152, primals_321, primals_322, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_22: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_58: "i64[]" = torch.ops.aten.add.Tensor(primals_670, 1);  primals_670 = None
    _native_batch_norm_legit_functional_13 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_22, primals_323, primals_324, primals_668, primals_669, True, 0.1, 1e-05);  primals_324 = primals_668 = primals_669 = None
    getitem_194: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_13[0]
    getitem_195: "f32[768]" = _native_batch_norm_legit_functional_13[1]
    getitem_196: "f32[768]" = _native_batch_norm_legit_functional_13[2]
    getitem_197: "f32[768]" = _native_batch_norm_legit_functional_13[3]
    getitem_198: "f32[768]" = _native_batch_norm_legit_functional_13[4];  _native_batch_norm_legit_functional_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_25: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_194, primals_325, primals_326, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_153: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_25, [8, 768, 784]);  convolution_25 = None
    permute_45: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_45: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_43, permute_45)
    add_59: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_57, mul_45);  mul_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_32 = torch.ops.aten.native_layer_norm.default(add_59, [768], primals_327, primals_328, 1e-06)
    getitem_199: "f32[8, 784, 768]" = native_layer_norm_32[0]
    getitem_200: "f32[8, 784, 1]" = native_layer_norm_32[1]
    getitem_201: "f32[8, 784, 1]" = native_layer_norm_32[2];  native_layer_norm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_154: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_199, [6272, 768]);  getitem_199 = None
    t_42: "f32[768, 3072]" = torch.ops.aten.t.default(primals_329);  primals_329 = None
    addmm_31: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_330, view_154, t_42);  primals_330 = None
    view_155: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_31, [8, 784, 3072]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_23: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_155)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_87: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_23);  gelu_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_156: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_87, [6272, 3072]);  clone_87 = None
    t_43: "f32[3072, 768]" = torch.ops.aten.t.default(primals_331);  primals_331 = None
    addmm_32: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_332, view_156, t_43);  primals_332 = None
    view_157: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_32, [8, 784, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_88: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_157);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_46: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_44, clone_88)
    add_60: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_59, mul_46);  mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_33 = torch.ops.aten.native_layer_norm.default(add_60, [768], primals_333, primals_334, 1e-06)
    getitem_202: "f32[8, 784, 768]" = native_layer_norm_33[0]
    getitem_203: "f32[8, 784, 1]" = native_layer_norm_33[1]
    getitem_204: "f32[8, 784, 1]" = native_layer_norm_33[2];  native_layer_norm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_158: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_202, [6272, 768]);  getitem_202 = None
    t_44: "f32[768, 2304]" = torch.ops.aten.t.default(primals_335);  primals_335 = None
    addmm_33: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_336, view_158, t_44);  primals_336 = None
    view_159: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_33, [8, 784, 2304]);  addmm_33 = None
    view_160: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_159, [8, 784, 3, 16, 48]);  view_159 = None
    permute_46: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_160, [2, 0, 3, 4, 1]);  view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_11 = torch.ops.aten.unbind.int(permute_46);  permute_46 = None
    getitem_205: "f32[8, 16, 48, 784]" = unbind_11[0]
    getitem_206: "f32[8, 16, 48, 784]" = unbind_11[1]
    getitem_207: "f32[8, 16, 48, 784]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_22: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_205, 2.0, [-1], True)
    detach_33: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_22)
    clamp_min_22: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_22, 1e-12)
    expand_66: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_22, [8, 16, 48, 784]);  clamp_min_22 = None
    div_28: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_205, expand_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_23: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_206, 2.0, [-1], True)
    detach_34: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_23)
    clamp_min_23: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_23, 1e-12)
    expand_67: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_23, [8, 16, 48, 784]);  clamp_min_23 = None
    div_29: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_206, expand_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_12: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_29, -2, -1);  div_29 = None
    expand_68: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_28, [8, 16, 48, 784]);  div_28 = None
    clone_89: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    _unsafe_view_44: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_89, [128, 48, 784]);  clone_89 = None
    expand_69: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_12, [8, 16, 784, 48]);  transpose_12 = None
    clone_90: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    _unsafe_view_45: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_90, [128, 784, 48]);  clone_90 = None
    bmm_22: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_44, _unsafe_view_45)
    view_161: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_22, [8, 16, 48, 48]);  bmm_22 = None
    mul_47: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_161, primals_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_11: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_47, -1, False);  mul_47 = None
    detach_35: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_91: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_11);  _softmax_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_70: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_91, [8, 16, 48, 48]);  clone_91 = None
    view_162: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_70, [128, 48, 48]);  expand_70 = None
    expand_71: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_207, [8, 16, 48, 784]);  getitem_207 = None
    clone_92: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    _unsafe_view_46: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_92, [128, 48, 784]);  clone_92 = None
    bmm_23: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_162, _unsafe_view_46)
    view_163: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_23, [8, 16, 48, 784]);  bmm_23 = None
    permute_47: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_163, [0, 3, 1, 2]);  view_163 = None
    view_164: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_47, [8, 784, 768]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_45: "f32[768, 768]" = torch.ops.aten.t.default(primals_337);  primals_337 = None
    clone_93: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_164, memory_format = torch.contiguous_format);  view_164 = None
    _unsafe_view_47: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_93, [6272, 768]);  clone_93 = None
    mm_11: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_47, t_45)
    view_165: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_11, [8, 784, 768]);  mm_11 = None
    add_61: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_165, primals_338);  view_165 = primals_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_94: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_61);  add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_48: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_45, clone_94)
    add_62: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_60, mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_34 = torch.ops.aten.native_layer_norm.default(add_62, [768], primals_339, primals_340, 1e-06)
    getitem_208: "f32[8, 784, 768]" = native_layer_norm_34[0]
    getitem_209: "f32[8, 784, 1]" = native_layer_norm_34[1]
    getitem_210: "f32[8, 784, 1]" = native_layer_norm_34[2];  native_layer_norm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_48: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_208, [0, 2, 1]);  getitem_208 = None
    view_166: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_48, [8, 768, 28, 28]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_26: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_166, primals_341, primals_342, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_24: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_63: "i64[]" = torch.ops.aten.add.Tensor(primals_673, 1);  primals_673 = None
    _native_batch_norm_legit_functional_14 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_24, primals_343, primals_344, primals_671, primals_672, True, 0.1, 1e-05);  primals_344 = primals_671 = primals_672 = None
    getitem_211: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_14[0]
    getitem_212: "f32[768]" = _native_batch_norm_legit_functional_14[1]
    getitem_213: "f32[768]" = _native_batch_norm_legit_functional_14[2]
    getitem_214: "f32[768]" = _native_batch_norm_legit_functional_14[3]
    getitem_215: "f32[768]" = _native_batch_norm_legit_functional_14[4];  _native_batch_norm_legit_functional_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_27: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_211, primals_345, primals_346, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_167: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_27, [8, 768, 784]);  convolution_27 = None
    permute_49: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_49: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_47, permute_49)
    add_64: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_62, mul_49);  mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_35 = torch.ops.aten.native_layer_norm.default(add_64, [768], primals_347, primals_348, 1e-06)
    getitem_216: "f32[8, 784, 768]" = native_layer_norm_35[0]
    getitem_217: "f32[8, 784, 1]" = native_layer_norm_35[1]
    getitem_218: "f32[8, 784, 1]" = native_layer_norm_35[2];  native_layer_norm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_168: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_216, [6272, 768]);  getitem_216 = None
    t_46: "f32[768, 3072]" = torch.ops.aten.t.default(primals_349);  primals_349 = None
    addmm_34: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_350, view_168, t_46);  primals_350 = None
    view_169: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 784, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_25: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_169)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_95: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_25);  gelu_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_170: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_95, [6272, 3072]);  clone_95 = None
    t_47: "f32[3072, 768]" = torch.ops.aten.t.default(primals_351);  primals_351 = None
    addmm_35: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_352, view_170, t_47);  primals_352 = None
    view_171: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_35, [8, 784, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_96: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_171);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_50: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_48, clone_96)
    add_65: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_64, mul_50);  mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_36 = torch.ops.aten.native_layer_norm.default(add_65, [768], primals_353, primals_354, 1e-06)
    getitem_219: "f32[8, 784, 768]" = native_layer_norm_36[0]
    getitem_220: "f32[8, 784, 1]" = native_layer_norm_36[1]
    getitem_221: "f32[8, 784, 1]" = native_layer_norm_36[2];  native_layer_norm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_172: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_219, [6272, 768]);  getitem_219 = None
    t_48: "f32[768, 2304]" = torch.ops.aten.t.default(primals_355);  primals_355 = None
    addmm_36: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_356, view_172, t_48);  primals_356 = None
    view_173: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_36, [8, 784, 2304]);  addmm_36 = None
    view_174: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_173, [8, 784, 3, 16, 48]);  view_173 = None
    permute_50: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_174, [2, 0, 3, 4, 1]);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_12 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
    getitem_222: "f32[8, 16, 48, 784]" = unbind_12[0]
    getitem_223: "f32[8, 16, 48, 784]" = unbind_12[1]
    getitem_224: "f32[8, 16, 48, 784]" = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_24: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_222, 2.0, [-1], True)
    detach_36: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_24)
    clamp_min_24: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_24, 1e-12)
    expand_72: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_24, [8, 16, 48, 784]);  clamp_min_24 = None
    div_30: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_222, expand_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_25: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_223, 2.0, [-1], True)
    detach_37: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_25)
    clamp_min_25: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_25, 1e-12)
    expand_73: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_25, [8, 16, 48, 784]);  clamp_min_25 = None
    div_31: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_223, expand_73)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_13: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_31, -2, -1);  div_31 = None
    expand_74: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_30, [8, 16, 48, 784]);  div_30 = None
    clone_97: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    _unsafe_view_48: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_97, [128, 48, 784]);  clone_97 = None
    expand_75: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_13, [8, 16, 784, 48]);  transpose_13 = None
    clone_98: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    _unsafe_view_49: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_98, [128, 784, 48]);  clone_98 = None
    bmm_24: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_48, _unsafe_view_49)
    view_175: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_24, [8, 16, 48, 48]);  bmm_24 = None
    mul_51: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_175, primals_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_12: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_51, -1, False);  mul_51 = None
    detach_38: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_99: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_12);  _softmax_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_76: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_99, [8, 16, 48, 48]);  clone_99 = None
    view_176: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_76, [128, 48, 48]);  expand_76 = None
    expand_77: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_224, [8, 16, 48, 784]);  getitem_224 = None
    clone_100: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
    _unsafe_view_50: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_100, [128, 48, 784]);  clone_100 = None
    bmm_25: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_176, _unsafe_view_50)
    view_177: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_25, [8, 16, 48, 784]);  bmm_25 = None
    permute_51: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_177, [0, 3, 1, 2]);  view_177 = None
    view_178: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_51, [8, 784, 768]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_49: "f32[768, 768]" = torch.ops.aten.t.default(primals_357);  primals_357 = None
    clone_101: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_178, memory_format = torch.contiguous_format);  view_178 = None
    _unsafe_view_51: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_101, [6272, 768]);  clone_101 = None
    mm_12: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_51, t_49)
    view_179: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_12, [8, 784, 768]);  mm_12 = None
    add_66: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_179, primals_358);  view_179 = primals_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_102: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_66);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_52: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_49, clone_102)
    add_67: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_65, mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_37 = torch.ops.aten.native_layer_norm.default(add_67, [768], primals_359, primals_360, 1e-06)
    getitem_225: "f32[8, 784, 768]" = native_layer_norm_37[0]
    getitem_226: "f32[8, 784, 1]" = native_layer_norm_37[1]
    getitem_227: "f32[8, 784, 1]" = native_layer_norm_37[2];  native_layer_norm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_52: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_225, [0, 2, 1]);  getitem_225 = None
    view_180: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_52, [8, 768, 28, 28]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_28: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_180, primals_361, primals_362, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_26: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_676, 1);  primals_676 = None
    _native_batch_norm_legit_functional_15 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_26, primals_363, primals_364, primals_674, primals_675, True, 0.1, 1e-05);  primals_364 = primals_674 = primals_675 = None
    getitem_228: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_15[0]
    getitem_229: "f32[768]" = _native_batch_norm_legit_functional_15[1]
    getitem_230: "f32[768]" = _native_batch_norm_legit_functional_15[2]
    getitem_231: "f32[768]" = _native_batch_norm_legit_functional_15[3]
    getitem_232: "f32[768]" = _native_batch_norm_legit_functional_15[4];  _native_batch_norm_legit_functional_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_29: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_228, primals_365, primals_366, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_181: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_29, [8, 768, 784]);  convolution_29 = None
    permute_53: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_53: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_51, permute_53)
    add_69: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_67, mul_53);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_38 = torch.ops.aten.native_layer_norm.default(add_69, [768], primals_367, primals_368, 1e-06)
    getitem_233: "f32[8, 784, 768]" = native_layer_norm_38[0]
    getitem_234: "f32[8, 784, 1]" = native_layer_norm_38[1]
    getitem_235: "f32[8, 784, 1]" = native_layer_norm_38[2];  native_layer_norm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_182: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_233, [6272, 768]);  getitem_233 = None
    t_50: "f32[768, 3072]" = torch.ops.aten.t.default(primals_369);  primals_369 = None
    addmm_37: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_370, view_182, t_50);  primals_370 = None
    view_183: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_37, [8, 784, 3072]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_27: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_183)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_103: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_27);  gelu_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_184: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_103, [6272, 3072]);  clone_103 = None
    t_51: "f32[3072, 768]" = torch.ops.aten.t.default(primals_371);  primals_371 = None
    addmm_38: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_372, view_184, t_51);  primals_372 = None
    view_185: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_38, [8, 784, 768]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_104: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_185);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_54: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_52, clone_104)
    add_70: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_69, mul_54);  mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_39 = torch.ops.aten.native_layer_norm.default(add_70, [768], primals_373, primals_374, 1e-06)
    getitem_236: "f32[8, 784, 768]" = native_layer_norm_39[0]
    getitem_237: "f32[8, 784, 1]" = native_layer_norm_39[1]
    getitem_238: "f32[8, 784, 1]" = native_layer_norm_39[2];  native_layer_norm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_186: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_236, [6272, 768]);  getitem_236 = None
    t_52: "f32[768, 2304]" = torch.ops.aten.t.default(primals_375);  primals_375 = None
    addmm_39: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_376, view_186, t_52);  primals_376 = None
    view_187: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_39, [8, 784, 2304]);  addmm_39 = None
    view_188: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_187, [8, 784, 3, 16, 48]);  view_187 = None
    permute_54: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_188, [2, 0, 3, 4, 1]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_13 = torch.ops.aten.unbind.int(permute_54);  permute_54 = None
    getitem_239: "f32[8, 16, 48, 784]" = unbind_13[0]
    getitem_240: "f32[8, 16, 48, 784]" = unbind_13[1]
    getitem_241: "f32[8, 16, 48, 784]" = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_26: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_239, 2.0, [-1], True)
    detach_39: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_26)
    clamp_min_26: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_26, 1e-12)
    expand_78: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_26, [8, 16, 48, 784]);  clamp_min_26 = None
    div_32: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_239, expand_78)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_27: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_240, 2.0, [-1], True)
    detach_40: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_27)
    clamp_min_27: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_27, 1e-12)
    expand_79: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_27, [8, 16, 48, 784]);  clamp_min_27 = None
    div_33: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_240, expand_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_14: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_33, -2, -1);  div_33 = None
    expand_80: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_32, [8, 16, 48, 784]);  div_32 = None
    clone_105: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    _unsafe_view_52: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_105, [128, 48, 784]);  clone_105 = None
    expand_81: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_14, [8, 16, 784, 48]);  transpose_14 = None
    clone_106: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    _unsafe_view_53: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_106, [128, 784, 48]);  clone_106 = None
    bmm_26: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_52, _unsafe_view_53)
    view_189: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_26, [8, 16, 48, 48]);  bmm_26 = None
    mul_55: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_189, primals_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_13: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_55, -1, False);  mul_55 = None
    detach_41: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_107: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_13);  _softmax_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_82: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_107, [8, 16, 48, 48]);  clone_107 = None
    view_190: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_82, [128, 48, 48]);  expand_82 = None
    expand_83: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_241, [8, 16, 48, 784]);  getitem_241 = None
    clone_108: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
    _unsafe_view_54: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_108, [128, 48, 784]);  clone_108 = None
    bmm_27: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_190, _unsafe_view_54)
    view_191: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_27, [8, 16, 48, 784]);  bmm_27 = None
    permute_55: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_191, [0, 3, 1, 2]);  view_191 = None
    view_192: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_55, [8, 784, 768]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_53: "f32[768, 768]" = torch.ops.aten.t.default(primals_377);  primals_377 = None
    clone_109: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_192, memory_format = torch.contiguous_format);  view_192 = None
    _unsafe_view_55: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_109, [6272, 768]);  clone_109 = None
    mm_13: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_55, t_53)
    view_193: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_13, [8, 784, 768]);  mm_13 = None
    add_71: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_193, primals_378);  view_193 = primals_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_110: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_71);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_56: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_53, clone_110)
    add_72: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_70, mul_56);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_40 = torch.ops.aten.native_layer_norm.default(add_72, [768], primals_379, primals_380, 1e-06)
    getitem_242: "f32[8, 784, 768]" = native_layer_norm_40[0]
    getitem_243: "f32[8, 784, 1]" = native_layer_norm_40[1]
    getitem_244: "f32[8, 784, 1]" = native_layer_norm_40[2];  native_layer_norm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_56: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_242, [0, 2, 1]);  getitem_242 = None
    view_194: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_56, [8, 768, 28, 28]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_30: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_194, primals_381, primals_382, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_28: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_679, 1);  primals_679 = None
    _native_batch_norm_legit_functional_16 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_28, primals_383, primals_384, primals_677, primals_678, True, 0.1, 1e-05);  primals_384 = primals_677 = primals_678 = None
    getitem_245: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_16[0]
    getitem_246: "f32[768]" = _native_batch_norm_legit_functional_16[1]
    getitem_247: "f32[768]" = _native_batch_norm_legit_functional_16[2]
    getitem_248: "f32[768]" = _native_batch_norm_legit_functional_16[3]
    getitem_249: "f32[768]" = _native_batch_norm_legit_functional_16[4];  _native_batch_norm_legit_functional_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_31: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_245, primals_385, primals_386, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_195: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_31, [8, 768, 784]);  convolution_31 = None
    permute_57: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_57: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_55, permute_57)
    add_74: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_72, mul_57);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_41 = torch.ops.aten.native_layer_norm.default(add_74, [768], primals_387, primals_388, 1e-06)
    getitem_250: "f32[8, 784, 768]" = native_layer_norm_41[0]
    getitem_251: "f32[8, 784, 1]" = native_layer_norm_41[1]
    getitem_252: "f32[8, 784, 1]" = native_layer_norm_41[2];  native_layer_norm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_196: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_250, [6272, 768]);  getitem_250 = None
    t_54: "f32[768, 3072]" = torch.ops.aten.t.default(primals_389);  primals_389 = None
    addmm_40: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_390, view_196, t_54);  primals_390 = None
    view_197: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_40, [8, 784, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_29: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_197)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_111: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_29);  gelu_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_198: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_111, [6272, 3072]);  clone_111 = None
    t_55: "f32[3072, 768]" = torch.ops.aten.t.default(primals_391);  primals_391 = None
    addmm_41: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_392, view_198, t_55);  primals_392 = None
    view_199: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_41, [8, 784, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_112: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_199);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_58: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_56, clone_112)
    add_75: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_74, mul_58);  mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_42 = torch.ops.aten.native_layer_norm.default(add_75, [768], primals_393, primals_394, 1e-06)
    getitem_253: "f32[8, 784, 768]" = native_layer_norm_42[0]
    getitem_254: "f32[8, 784, 1]" = native_layer_norm_42[1]
    getitem_255: "f32[8, 784, 1]" = native_layer_norm_42[2];  native_layer_norm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_200: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_253, [6272, 768]);  getitem_253 = None
    t_56: "f32[768, 2304]" = torch.ops.aten.t.default(primals_395);  primals_395 = None
    addmm_42: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_396, view_200, t_56);  primals_396 = None
    view_201: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_42, [8, 784, 2304]);  addmm_42 = None
    view_202: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_201, [8, 784, 3, 16, 48]);  view_201 = None
    permute_58: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_202, [2, 0, 3, 4, 1]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_14 = torch.ops.aten.unbind.int(permute_58);  permute_58 = None
    getitem_256: "f32[8, 16, 48, 784]" = unbind_14[0]
    getitem_257: "f32[8, 16, 48, 784]" = unbind_14[1]
    getitem_258: "f32[8, 16, 48, 784]" = unbind_14[2];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_28: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_256, 2.0, [-1], True)
    detach_42: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_28)
    clamp_min_28: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_28, 1e-12)
    expand_84: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_28, [8, 16, 48, 784]);  clamp_min_28 = None
    div_34: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_256, expand_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_29: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_257, 2.0, [-1], True)
    detach_43: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_29)
    clamp_min_29: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_29, 1e-12)
    expand_85: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_29, [8, 16, 48, 784]);  clamp_min_29 = None
    div_35: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_257, expand_85)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_15: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_35, -2, -1);  div_35 = None
    expand_86: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_34, [8, 16, 48, 784]);  div_34 = None
    clone_113: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
    _unsafe_view_56: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_113, [128, 48, 784]);  clone_113 = None
    expand_87: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_15, [8, 16, 784, 48]);  transpose_15 = None
    clone_114: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
    _unsafe_view_57: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_114, [128, 784, 48]);  clone_114 = None
    bmm_28: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_56, _unsafe_view_57)
    view_203: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_28, [8, 16, 48, 48]);  bmm_28 = None
    mul_59: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_203, primals_58)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_14: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_59, -1, False);  mul_59 = None
    detach_44: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_115: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_14);  _softmax_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_88: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_115, [8, 16, 48, 48]);  clone_115 = None
    view_204: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_88, [128, 48, 48]);  expand_88 = None
    expand_89: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_258, [8, 16, 48, 784]);  getitem_258 = None
    clone_116: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    _unsafe_view_58: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_116, [128, 48, 784]);  clone_116 = None
    bmm_29: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_204, _unsafe_view_58)
    view_205: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_29, [8, 16, 48, 784]);  bmm_29 = None
    permute_59: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_205, [0, 3, 1, 2]);  view_205 = None
    view_206: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_59, [8, 784, 768]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_57: "f32[768, 768]" = torch.ops.aten.t.default(primals_397);  primals_397 = None
    clone_117: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_206, memory_format = torch.contiguous_format);  view_206 = None
    _unsafe_view_59: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_117, [6272, 768]);  clone_117 = None
    mm_14: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_59, t_57)
    view_207: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_14, [8, 784, 768]);  mm_14 = None
    add_76: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_207, primals_398);  view_207 = primals_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_118: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_76);  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_60: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_57, clone_118)
    add_77: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_75, mul_60);  mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_43 = torch.ops.aten.native_layer_norm.default(add_77, [768], primals_399, primals_400, 1e-06)
    getitem_259: "f32[8, 784, 768]" = native_layer_norm_43[0]
    getitem_260: "f32[8, 784, 1]" = native_layer_norm_43[1]
    getitem_261: "f32[8, 784, 1]" = native_layer_norm_43[2];  native_layer_norm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_60: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_259, [0, 2, 1]);  getitem_259 = None
    view_208: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_60, [8, 768, 28, 28]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_32: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_208, primals_401, primals_402, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_30: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_78: "i64[]" = torch.ops.aten.add.Tensor(primals_682, 1);  primals_682 = None
    _native_batch_norm_legit_functional_17 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_30, primals_403, primals_404, primals_680, primals_681, True, 0.1, 1e-05);  primals_404 = primals_680 = primals_681 = None
    getitem_262: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_17[0]
    getitem_263: "f32[768]" = _native_batch_norm_legit_functional_17[1]
    getitem_264: "f32[768]" = _native_batch_norm_legit_functional_17[2]
    getitem_265: "f32[768]" = _native_batch_norm_legit_functional_17[3]
    getitem_266: "f32[768]" = _native_batch_norm_legit_functional_17[4];  _native_batch_norm_legit_functional_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_33: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_262, primals_405, primals_406, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_209: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_33, [8, 768, 784]);  convolution_33 = None
    permute_61: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_61: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_59, permute_61)
    add_79: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_77, mul_61);  mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_44 = torch.ops.aten.native_layer_norm.default(add_79, [768], primals_407, primals_408, 1e-06)
    getitem_267: "f32[8, 784, 768]" = native_layer_norm_44[0]
    getitem_268: "f32[8, 784, 1]" = native_layer_norm_44[1]
    getitem_269: "f32[8, 784, 1]" = native_layer_norm_44[2];  native_layer_norm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_210: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_267, [6272, 768]);  getitem_267 = None
    t_58: "f32[768, 3072]" = torch.ops.aten.t.default(primals_409);  primals_409 = None
    addmm_43: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_410, view_210, t_58);  primals_410 = None
    view_211: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_43, [8, 784, 3072]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_31: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_211)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_119: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_31);  gelu_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_212: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_119, [6272, 3072]);  clone_119 = None
    t_59: "f32[3072, 768]" = torch.ops.aten.t.default(primals_411);  primals_411 = None
    addmm_44: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_412, view_212, t_59);  primals_412 = None
    view_213: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_44, [8, 784, 768]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_120: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_213);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_62: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_60, clone_120)
    add_80: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_79, mul_62);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_45 = torch.ops.aten.native_layer_norm.default(add_80, [768], primals_413, primals_414, 1e-06)
    getitem_270: "f32[8, 784, 768]" = native_layer_norm_45[0]
    getitem_271: "f32[8, 784, 1]" = native_layer_norm_45[1]
    getitem_272: "f32[8, 784, 1]" = native_layer_norm_45[2];  native_layer_norm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_214: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_270, [6272, 768]);  getitem_270 = None
    t_60: "f32[768, 2304]" = torch.ops.aten.t.default(primals_415);  primals_415 = None
    addmm_45: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_416, view_214, t_60);  primals_416 = None
    view_215: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_45, [8, 784, 2304]);  addmm_45 = None
    view_216: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_215, [8, 784, 3, 16, 48]);  view_215 = None
    permute_62: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_216, [2, 0, 3, 4, 1]);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_15 = torch.ops.aten.unbind.int(permute_62);  permute_62 = None
    getitem_273: "f32[8, 16, 48, 784]" = unbind_15[0]
    getitem_274: "f32[8, 16, 48, 784]" = unbind_15[1]
    getitem_275: "f32[8, 16, 48, 784]" = unbind_15[2];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_30: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_273, 2.0, [-1], True)
    detach_45: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_30)
    clamp_min_30: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_30, 1e-12)
    expand_90: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_30, [8, 16, 48, 784]);  clamp_min_30 = None
    div_36: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_273, expand_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_31: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_274, 2.0, [-1], True)
    detach_46: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_31)
    clamp_min_31: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_31, 1e-12)
    expand_91: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_31, [8, 16, 48, 784]);  clamp_min_31 = None
    div_37: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_274, expand_91)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_16: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_37, -2, -1);  div_37 = None
    expand_92: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_36, [8, 16, 48, 784]);  div_36 = None
    clone_121: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
    _unsafe_view_60: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_121, [128, 48, 784]);  clone_121 = None
    expand_93: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_16, [8, 16, 784, 48]);  transpose_16 = None
    clone_122: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
    _unsafe_view_61: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_122, [128, 784, 48]);  clone_122 = None
    bmm_30: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_60, _unsafe_view_61)
    view_217: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_30, [8, 16, 48, 48]);  bmm_30 = None
    mul_63: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_217, primals_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_15: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_63, -1, False);  mul_63 = None
    detach_47: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_123: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_15);  _softmax_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_94: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_123, [8, 16, 48, 48]);  clone_123 = None
    view_218: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_94, [128, 48, 48]);  expand_94 = None
    expand_95: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_275, [8, 16, 48, 784]);  getitem_275 = None
    clone_124: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
    _unsafe_view_62: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_124, [128, 48, 784]);  clone_124 = None
    bmm_31: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_218, _unsafe_view_62)
    view_219: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_31, [8, 16, 48, 784]);  bmm_31 = None
    permute_63: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_219, [0, 3, 1, 2]);  view_219 = None
    view_220: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_63, [8, 784, 768]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_61: "f32[768, 768]" = torch.ops.aten.t.default(primals_417);  primals_417 = None
    clone_125: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_220, memory_format = torch.contiguous_format);  view_220 = None
    _unsafe_view_63: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_125, [6272, 768]);  clone_125 = None
    mm_15: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_63, t_61)
    view_221: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_15, [8, 784, 768]);  mm_15 = None
    add_81: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_221, primals_418);  view_221 = primals_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_126: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_81);  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_64: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_61, clone_126)
    add_82: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_80, mul_64);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_46 = torch.ops.aten.native_layer_norm.default(add_82, [768], primals_419, primals_420, 1e-06)
    getitem_276: "f32[8, 784, 768]" = native_layer_norm_46[0]
    getitem_277: "f32[8, 784, 1]" = native_layer_norm_46[1]
    getitem_278: "f32[8, 784, 1]" = native_layer_norm_46[2];  native_layer_norm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_64: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_276, [0, 2, 1]);  getitem_276 = None
    view_222: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_64, [8, 768, 28, 28]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_34: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_222, primals_421, primals_422, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_32: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_685, 1);  primals_685 = None
    _native_batch_norm_legit_functional_18 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_32, primals_423, primals_424, primals_683, primals_684, True, 0.1, 1e-05);  primals_424 = primals_683 = primals_684 = None
    getitem_279: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_18[0]
    getitem_280: "f32[768]" = _native_batch_norm_legit_functional_18[1]
    getitem_281: "f32[768]" = _native_batch_norm_legit_functional_18[2]
    getitem_282: "f32[768]" = _native_batch_norm_legit_functional_18[3]
    getitem_283: "f32[768]" = _native_batch_norm_legit_functional_18[4];  _native_batch_norm_legit_functional_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_35: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_279, primals_425, primals_426, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_223: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_35, [8, 768, 784]);  convolution_35 = None
    permute_65: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_223, [0, 2, 1]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_65: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_63, permute_65)
    add_84: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_82, mul_65);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_47 = torch.ops.aten.native_layer_norm.default(add_84, [768], primals_427, primals_428, 1e-06)
    getitem_284: "f32[8, 784, 768]" = native_layer_norm_47[0]
    getitem_285: "f32[8, 784, 1]" = native_layer_norm_47[1]
    getitem_286: "f32[8, 784, 1]" = native_layer_norm_47[2];  native_layer_norm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_224: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_284, [6272, 768]);  getitem_284 = None
    t_62: "f32[768, 3072]" = torch.ops.aten.t.default(primals_429);  primals_429 = None
    addmm_46: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_430, view_224, t_62);  primals_430 = None
    view_225: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 784, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_33: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_225)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_127: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_33);  gelu_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_226: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_127, [6272, 3072]);  clone_127 = None
    t_63: "f32[3072, 768]" = torch.ops.aten.t.default(primals_431);  primals_431 = None
    addmm_47: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_432, view_226, t_63);  primals_432 = None
    view_227: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_47, [8, 784, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_128: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_227);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_66: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_64, clone_128)
    add_85: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_84, mul_66);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_48 = torch.ops.aten.native_layer_norm.default(add_85, [768], primals_433, primals_434, 1e-06)
    getitem_287: "f32[8, 784, 768]" = native_layer_norm_48[0]
    getitem_288: "f32[8, 784, 1]" = native_layer_norm_48[1]
    getitem_289: "f32[8, 784, 1]" = native_layer_norm_48[2];  native_layer_norm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_228: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_287, [6272, 768]);  getitem_287 = None
    t_64: "f32[768, 2304]" = torch.ops.aten.t.default(primals_435);  primals_435 = None
    addmm_48: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_436, view_228, t_64);  primals_436 = None
    view_229: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_48, [8, 784, 2304]);  addmm_48 = None
    view_230: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_229, [8, 784, 3, 16, 48]);  view_229 = None
    permute_66: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_230, [2, 0, 3, 4, 1]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_16 = torch.ops.aten.unbind.int(permute_66);  permute_66 = None
    getitem_290: "f32[8, 16, 48, 784]" = unbind_16[0]
    getitem_291: "f32[8, 16, 48, 784]" = unbind_16[1]
    getitem_292: "f32[8, 16, 48, 784]" = unbind_16[2];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_32: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_290, 2.0, [-1], True)
    detach_48: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_32)
    clamp_min_32: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_32, 1e-12)
    expand_96: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_32, [8, 16, 48, 784]);  clamp_min_32 = None
    div_38: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_290, expand_96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_33: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_291, 2.0, [-1], True)
    detach_49: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_33)
    clamp_min_33: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_33, 1e-12)
    expand_97: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_33, [8, 16, 48, 784]);  clamp_min_33 = None
    div_39: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_291, expand_97)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_17: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_39, -2, -1);  div_39 = None
    expand_98: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_38, [8, 16, 48, 784]);  div_38 = None
    clone_129: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
    _unsafe_view_64: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_129, [128, 48, 784]);  clone_129 = None
    expand_99: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_17, [8, 16, 784, 48]);  transpose_17 = None
    clone_130: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
    _unsafe_view_65: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_130, [128, 784, 48]);  clone_130 = None
    bmm_32: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_64, _unsafe_view_65)
    view_231: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_32, [8, 16, 48, 48]);  bmm_32 = None
    mul_67: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_231, primals_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_16: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_67, -1, False);  mul_67 = None
    detach_50: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_131: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_16);  _softmax_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_100: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_131, [8, 16, 48, 48]);  clone_131 = None
    view_232: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_100, [128, 48, 48]);  expand_100 = None
    expand_101: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_292, [8, 16, 48, 784]);  getitem_292 = None
    clone_132: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
    _unsafe_view_66: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_132, [128, 48, 784]);  clone_132 = None
    bmm_33: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_232, _unsafe_view_66)
    view_233: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_33, [8, 16, 48, 784]);  bmm_33 = None
    permute_67: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_233, [0, 3, 1, 2]);  view_233 = None
    view_234: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_67, [8, 784, 768]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_65: "f32[768, 768]" = torch.ops.aten.t.default(primals_437);  primals_437 = None
    clone_133: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_234, memory_format = torch.contiguous_format);  view_234 = None
    _unsafe_view_67: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_133, [6272, 768]);  clone_133 = None
    mm_16: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_67, t_65)
    view_235: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_16, [8, 784, 768]);  mm_16 = None
    add_86: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_235, primals_438);  view_235 = primals_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_134: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_86);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_68: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_65, clone_134)
    add_87: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_85, mul_68);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_49 = torch.ops.aten.native_layer_norm.default(add_87, [768], primals_439, primals_440, 1e-06)
    getitem_293: "f32[8, 784, 768]" = native_layer_norm_49[0]
    getitem_294: "f32[8, 784, 1]" = native_layer_norm_49[1]
    getitem_295: "f32[8, 784, 1]" = native_layer_norm_49[2];  native_layer_norm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_68: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_293, [0, 2, 1]);  getitem_293 = None
    view_236: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_68, [8, 768, 28, 28]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_36: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_236, primals_441, primals_442, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_34: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_88: "i64[]" = torch.ops.aten.add.Tensor(primals_688, 1);  primals_688 = None
    _native_batch_norm_legit_functional_19 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_34, primals_443, primals_444, primals_686, primals_687, True, 0.1, 1e-05);  primals_444 = primals_686 = primals_687 = None
    getitem_296: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_19[0]
    getitem_297: "f32[768]" = _native_batch_norm_legit_functional_19[1]
    getitem_298: "f32[768]" = _native_batch_norm_legit_functional_19[2]
    getitem_299: "f32[768]" = _native_batch_norm_legit_functional_19[3]
    getitem_300: "f32[768]" = _native_batch_norm_legit_functional_19[4];  _native_batch_norm_legit_functional_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_37: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_296, primals_445, primals_446, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_237: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_37, [8, 768, 784]);  convolution_37 = None
    permute_69: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_69: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_67, permute_69)
    add_89: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_87, mul_69);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_50 = torch.ops.aten.native_layer_norm.default(add_89, [768], primals_447, primals_448, 1e-06)
    getitem_301: "f32[8, 784, 768]" = native_layer_norm_50[0]
    getitem_302: "f32[8, 784, 1]" = native_layer_norm_50[1]
    getitem_303: "f32[8, 784, 1]" = native_layer_norm_50[2];  native_layer_norm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_238: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_301, [6272, 768]);  getitem_301 = None
    t_66: "f32[768, 3072]" = torch.ops.aten.t.default(primals_449);  primals_449 = None
    addmm_49: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_450, view_238, t_66);  primals_450 = None
    view_239: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_49, [8, 784, 3072]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_35: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_239)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_135: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_35);  gelu_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_240: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_135, [6272, 3072]);  clone_135 = None
    t_67: "f32[3072, 768]" = torch.ops.aten.t.default(primals_451);  primals_451 = None
    addmm_50: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_452, view_240, t_67);  primals_452 = None
    view_241: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_50, [8, 784, 768]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_136: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_241);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_70: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_68, clone_136)
    add_90: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_89, mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_51 = torch.ops.aten.native_layer_norm.default(add_90, [768], primals_453, primals_454, 1e-06)
    getitem_304: "f32[8, 784, 768]" = native_layer_norm_51[0]
    getitem_305: "f32[8, 784, 1]" = native_layer_norm_51[1]
    getitem_306: "f32[8, 784, 1]" = native_layer_norm_51[2];  native_layer_norm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_242: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_304, [6272, 768]);  getitem_304 = None
    t_68: "f32[768, 2304]" = torch.ops.aten.t.default(primals_455);  primals_455 = None
    addmm_51: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_456, view_242, t_68);  primals_456 = None
    view_243: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_51, [8, 784, 2304]);  addmm_51 = None
    view_244: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_243, [8, 784, 3, 16, 48]);  view_243 = None
    permute_70: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_244, [2, 0, 3, 4, 1]);  view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_17 = torch.ops.aten.unbind.int(permute_70);  permute_70 = None
    getitem_307: "f32[8, 16, 48, 784]" = unbind_17[0]
    getitem_308: "f32[8, 16, 48, 784]" = unbind_17[1]
    getitem_309: "f32[8, 16, 48, 784]" = unbind_17[2];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_34: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_307, 2.0, [-1], True)
    detach_51: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_34)
    clamp_min_34: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_34, 1e-12)
    expand_102: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_34, [8, 16, 48, 784]);  clamp_min_34 = None
    div_40: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_307, expand_102)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_35: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_308, 2.0, [-1], True)
    detach_52: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_35)
    clamp_min_35: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_35, 1e-12)
    expand_103: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_35, [8, 16, 48, 784]);  clamp_min_35 = None
    div_41: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_308, expand_103)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_18: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_41, -2, -1);  div_41 = None
    expand_104: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_40, [8, 16, 48, 784]);  div_40 = None
    clone_137: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
    _unsafe_view_68: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_137, [128, 48, 784]);  clone_137 = None
    expand_105: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_18, [8, 16, 784, 48]);  transpose_18 = None
    clone_138: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
    _unsafe_view_69: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_138, [128, 784, 48]);  clone_138 = None
    bmm_34: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_68, _unsafe_view_69)
    view_245: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_34, [8, 16, 48, 48]);  bmm_34 = None
    mul_71: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_245, primals_70)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_17: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_71, -1, False);  mul_71 = None
    detach_53: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_139: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_17);  _softmax_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_106: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_139, [8, 16, 48, 48]);  clone_139 = None
    view_246: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_106, [128, 48, 48]);  expand_106 = None
    expand_107: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_309, [8, 16, 48, 784]);  getitem_309 = None
    clone_140: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
    _unsafe_view_70: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_140, [128, 48, 784]);  clone_140 = None
    bmm_35: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_246, _unsafe_view_70)
    view_247: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_35, [8, 16, 48, 784]);  bmm_35 = None
    permute_71: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_247, [0, 3, 1, 2]);  view_247 = None
    view_248: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_71, [8, 784, 768]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_69: "f32[768, 768]" = torch.ops.aten.t.default(primals_457);  primals_457 = None
    clone_141: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_248, memory_format = torch.contiguous_format);  view_248 = None
    _unsafe_view_71: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_141, [6272, 768]);  clone_141 = None
    mm_17: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_71, t_69)
    view_249: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_17, [8, 784, 768]);  mm_17 = None
    add_91: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_249, primals_458);  view_249 = primals_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_142: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_91);  add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_72: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_69, clone_142)
    add_92: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_90, mul_72);  mul_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_52 = torch.ops.aten.native_layer_norm.default(add_92, [768], primals_459, primals_460, 1e-06)
    getitem_310: "f32[8, 784, 768]" = native_layer_norm_52[0]
    getitem_311: "f32[8, 784, 1]" = native_layer_norm_52[1]
    getitem_312: "f32[8, 784, 1]" = native_layer_norm_52[2];  native_layer_norm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_72: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_310, [0, 2, 1]);  getitem_310 = None
    view_250: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_72, [8, 768, 28, 28]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_38: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_250, primals_461, primals_462, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_36: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_93: "i64[]" = torch.ops.aten.add.Tensor(primals_691, 1);  primals_691 = None
    _native_batch_norm_legit_functional_20 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_36, primals_463, primals_464, primals_689, primals_690, True, 0.1, 1e-05);  primals_464 = primals_689 = primals_690 = None
    getitem_313: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_20[0]
    getitem_314: "f32[768]" = _native_batch_norm_legit_functional_20[1]
    getitem_315: "f32[768]" = _native_batch_norm_legit_functional_20[2]
    getitem_316: "f32[768]" = _native_batch_norm_legit_functional_20[3]
    getitem_317: "f32[768]" = _native_batch_norm_legit_functional_20[4];  _native_batch_norm_legit_functional_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_39: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_313, primals_465, primals_466, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_251: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_39, [8, 768, 784]);  convolution_39 = None
    permute_73: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_73: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_71, permute_73)
    add_94: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_92, mul_73);  mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_53 = torch.ops.aten.native_layer_norm.default(add_94, [768], primals_467, primals_468, 1e-06)
    getitem_318: "f32[8, 784, 768]" = native_layer_norm_53[0]
    getitem_319: "f32[8, 784, 1]" = native_layer_norm_53[1]
    getitem_320: "f32[8, 784, 1]" = native_layer_norm_53[2];  native_layer_norm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_252: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_318, [6272, 768]);  getitem_318 = None
    t_70: "f32[768, 3072]" = torch.ops.aten.t.default(primals_469);  primals_469 = None
    addmm_52: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_470, view_252, t_70);  primals_470 = None
    view_253: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_52, [8, 784, 3072]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_37: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_253)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_143: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_37);  gelu_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_254: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_143, [6272, 3072]);  clone_143 = None
    t_71: "f32[3072, 768]" = torch.ops.aten.t.default(primals_471);  primals_471 = None
    addmm_53: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_472, view_254, t_71);  primals_472 = None
    view_255: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_53, [8, 784, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_144: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_255);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_74: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_72, clone_144)
    add_95: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_94, mul_74);  mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_54 = torch.ops.aten.native_layer_norm.default(add_95, [768], primals_473, primals_474, 1e-06)
    getitem_321: "f32[8, 784, 768]" = native_layer_norm_54[0]
    getitem_322: "f32[8, 784, 1]" = native_layer_norm_54[1]
    getitem_323: "f32[8, 784, 1]" = native_layer_norm_54[2];  native_layer_norm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_256: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_321, [6272, 768]);  getitem_321 = None
    t_72: "f32[768, 2304]" = torch.ops.aten.t.default(primals_475);  primals_475 = None
    addmm_54: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_476, view_256, t_72);  primals_476 = None
    view_257: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_54, [8, 784, 2304]);  addmm_54 = None
    view_258: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_257, [8, 784, 3, 16, 48]);  view_257 = None
    permute_74: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_258, [2, 0, 3, 4, 1]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_18 = torch.ops.aten.unbind.int(permute_74);  permute_74 = None
    getitem_324: "f32[8, 16, 48, 784]" = unbind_18[0]
    getitem_325: "f32[8, 16, 48, 784]" = unbind_18[1]
    getitem_326: "f32[8, 16, 48, 784]" = unbind_18[2];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_36: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_324, 2.0, [-1], True)
    detach_54: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_36)
    clamp_min_36: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_36, 1e-12)
    expand_108: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_36, [8, 16, 48, 784]);  clamp_min_36 = None
    div_42: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_324, expand_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_37: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_325, 2.0, [-1], True)
    detach_55: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_37)
    clamp_min_37: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_37, 1e-12)
    expand_109: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_37, [8, 16, 48, 784]);  clamp_min_37 = None
    div_43: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_325, expand_109)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_19: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_43, -2, -1);  div_43 = None
    expand_110: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_42, [8, 16, 48, 784]);  div_42 = None
    clone_145: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_110, memory_format = torch.contiguous_format);  expand_110 = None
    _unsafe_view_72: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_145, [128, 48, 784]);  clone_145 = None
    expand_111: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_19, [8, 16, 784, 48]);  transpose_19 = None
    clone_146: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
    _unsafe_view_73: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_146, [128, 784, 48]);  clone_146 = None
    bmm_36: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_72, _unsafe_view_73)
    view_259: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_36, [8, 16, 48, 48]);  bmm_36 = None
    mul_75: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_259, primals_74)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_18: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_75, -1, False);  mul_75 = None
    detach_56: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_147: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_18);  _softmax_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_112: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_147, [8, 16, 48, 48]);  clone_147 = None
    view_260: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_112, [128, 48, 48]);  expand_112 = None
    expand_113: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_326, [8, 16, 48, 784]);  getitem_326 = None
    clone_148: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
    _unsafe_view_74: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_148, [128, 48, 784]);  clone_148 = None
    bmm_37: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_260, _unsafe_view_74)
    view_261: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_37, [8, 16, 48, 784]);  bmm_37 = None
    permute_75: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_261, [0, 3, 1, 2]);  view_261 = None
    view_262: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_75, [8, 784, 768]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_73: "f32[768, 768]" = torch.ops.aten.t.default(primals_477);  primals_477 = None
    clone_149: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_262, memory_format = torch.contiguous_format);  view_262 = None
    _unsafe_view_75: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_149, [6272, 768]);  clone_149 = None
    mm_18: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_75, t_73)
    view_263: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_18, [8, 784, 768]);  mm_18 = None
    add_96: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_263, primals_478);  view_263 = primals_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_150: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_96);  add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_76: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_73, clone_150)
    add_97: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_95, mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_55 = torch.ops.aten.native_layer_norm.default(add_97, [768], primals_479, primals_480, 1e-06)
    getitem_327: "f32[8, 784, 768]" = native_layer_norm_55[0]
    getitem_328: "f32[8, 784, 1]" = native_layer_norm_55[1]
    getitem_329: "f32[8, 784, 1]" = native_layer_norm_55[2];  native_layer_norm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_76: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_327, [0, 2, 1]);  getitem_327 = None
    view_264: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_76, [8, 768, 28, 28]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_40: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_264, primals_481, primals_482, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_38: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_98: "i64[]" = torch.ops.aten.add.Tensor(primals_694, 1);  primals_694 = None
    _native_batch_norm_legit_functional_21 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_38, primals_483, primals_484, primals_692, primals_693, True, 0.1, 1e-05);  primals_484 = primals_692 = primals_693 = None
    getitem_330: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_21[0]
    getitem_331: "f32[768]" = _native_batch_norm_legit_functional_21[1]
    getitem_332: "f32[768]" = _native_batch_norm_legit_functional_21[2]
    getitem_333: "f32[768]" = _native_batch_norm_legit_functional_21[3]
    getitem_334: "f32[768]" = _native_batch_norm_legit_functional_21[4];  _native_batch_norm_legit_functional_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_41: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_330, primals_485, primals_486, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_265: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_41, [8, 768, 784]);  convolution_41 = None
    permute_77: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_265, [0, 2, 1]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_77: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_75, permute_77)
    add_99: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_97, mul_77);  mul_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_56 = torch.ops.aten.native_layer_norm.default(add_99, [768], primals_487, primals_488, 1e-06)
    getitem_335: "f32[8, 784, 768]" = native_layer_norm_56[0]
    getitem_336: "f32[8, 784, 1]" = native_layer_norm_56[1]
    getitem_337: "f32[8, 784, 1]" = native_layer_norm_56[2];  native_layer_norm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_266: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_335, [6272, 768]);  getitem_335 = None
    t_74: "f32[768, 3072]" = torch.ops.aten.t.default(primals_489);  primals_489 = None
    addmm_55: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_490, view_266, t_74);  primals_490 = None
    view_267: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_55, [8, 784, 3072]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_39: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_267)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_151: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_39);  gelu_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_268: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_151, [6272, 3072]);  clone_151 = None
    t_75: "f32[3072, 768]" = torch.ops.aten.t.default(primals_491);  primals_491 = None
    addmm_56: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_492, view_268, t_75);  primals_492 = None
    view_269: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_56, [8, 784, 768]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_152: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_269);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_78: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_76, clone_152)
    add_100: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_99, mul_78);  mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_57 = torch.ops.aten.native_layer_norm.default(add_100, [768], primals_493, primals_494, 1e-06)
    getitem_338: "f32[8, 784, 768]" = native_layer_norm_57[0]
    getitem_339: "f32[8, 784, 1]" = native_layer_norm_57[1]
    getitem_340: "f32[8, 784, 1]" = native_layer_norm_57[2];  native_layer_norm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_270: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_338, [6272, 768]);  getitem_338 = None
    t_76: "f32[768, 2304]" = torch.ops.aten.t.default(primals_495);  primals_495 = None
    addmm_57: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_496, view_270, t_76);  primals_496 = None
    view_271: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_57, [8, 784, 2304]);  addmm_57 = None
    view_272: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_271, [8, 784, 3, 16, 48]);  view_271 = None
    permute_78: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_272, [2, 0, 3, 4, 1]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_19 = torch.ops.aten.unbind.int(permute_78);  permute_78 = None
    getitem_341: "f32[8, 16, 48, 784]" = unbind_19[0]
    getitem_342: "f32[8, 16, 48, 784]" = unbind_19[1]
    getitem_343: "f32[8, 16, 48, 784]" = unbind_19[2];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_38: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_341, 2.0, [-1], True)
    detach_57: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_38)
    clamp_min_38: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_38, 1e-12)
    expand_114: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_38, [8, 16, 48, 784]);  clamp_min_38 = None
    div_44: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_341, expand_114)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_39: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_342, 2.0, [-1], True)
    detach_58: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_39)
    clamp_min_39: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_39, 1e-12)
    expand_115: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_39, [8, 16, 48, 784]);  clamp_min_39 = None
    div_45: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_342, expand_115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_20: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_45, -2, -1);  div_45 = None
    expand_116: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_44, [8, 16, 48, 784]);  div_44 = None
    clone_153: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
    _unsafe_view_76: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_153, [128, 48, 784]);  clone_153 = None
    expand_117: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_20, [8, 16, 784, 48]);  transpose_20 = None
    clone_154: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
    _unsafe_view_77: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_154, [128, 784, 48]);  clone_154 = None
    bmm_38: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_76, _unsafe_view_77)
    view_273: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_38, [8, 16, 48, 48]);  bmm_38 = None
    mul_79: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_273, primals_78)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_19: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_79, -1, False);  mul_79 = None
    detach_59: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_155: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_19);  _softmax_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_118: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_155, [8, 16, 48, 48]);  clone_155 = None
    view_274: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_118, [128, 48, 48]);  expand_118 = None
    expand_119: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_343, [8, 16, 48, 784]);  getitem_343 = None
    clone_156: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_119, memory_format = torch.contiguous_format);  expand_119 = None
    _unsafe_view_78: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_156, [128, 48, 784]);  clone_156 = None
    bmm_39: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_274, _unsafe_view_78)
    view_275: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_39, [8, 16, 48, 784]);  bmm_39 = None
    permute_79: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_275, [0, 3, 1, 2]);  view_275 = None
    view_276: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_79, [8, 784, 768]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_77: "f32[768, 768]" = torch.ops.aten.t.default(primals_497);  primals_497 = None
    clone_157: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_276, memory_format = torch.contiguous_format);  view_276 = None
    _unsafe_view_79: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_157, [6272, 768]);  clone_157 = None
    mm_19: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_79, t_77)
    view_277: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_19, [8, 784, 768]);  mm_19 = None
    add_101: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_277, primals_498);  view_277 = primals_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_158: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_101);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_80: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_77, clone_158)
    add_102: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_100, mul_80);  mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_58 = torch.ops.aten.native_layer_norm.default(add_102, [768], primals_499, primals_500, 1e-06)
    getitem_344: "f32[8, 784, 768]" = native_layer_norm_58[0]
    getitem_345: "f32[8, 784, 1]" = native_layer_norm_58[1]
    getitem_346: "f32[8, 784, 1]" = native_layer_norm_58[2];  native_layer_norm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_80: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_344, [0, 2, 1]);  getitem_344 = None
    view_278: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_80, [8, 768, 28, 28]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_42: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_278, primals_501, primals_502, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_40: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_103: "i64[]" = torch.ops.aten.add.Tensor(primals_697, 1);  primals_697 = None
    _native_batch_norm_legit_functional_22 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_40, primals_503, primals_504, primals_695, primals_696, True, 0.1, 1e-05);  primals_504 = primals_695 = primals_696 = None
    getitem_347: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_22[0]
    getitem_348: "f32[768]" = _native_batch_norm_legit_functional_22[1]
    getitem_349: "f32[768]" = _native_batch_norm_legit_functional_22[2]
    getitem_350: "f32[768]" = _native_batch_norm_legit_functional_22[3]
    getitem_351: "f32[768]" = _native_batch_norm_legit_functional_22[4];  _native_batch_norm_legit_functional_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_43: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_347, primals_505, primals_506, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_279: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_43, [8, 768, 784]);  convolution_43 = None
    permute_81: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_279, [0, 2, 1]);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_81: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_79, permute_81)
    add_104: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_102, mul_81);  mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_59 = torch.ops.aten.native_layer_norm.default(add_104, [768], primals_507, primals_508, 1e-06)
    getitem_352: "f32[8, 784, 768]" = native_layer_norm_59[0]
    getitem_353: "f32[8, 784, 1]" = native_layer_norm_59[1]
    getitem_354: "f32[8, 784, 1]" = native_layer_norm_59[2];  native_layer_norm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_280: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_352, [6272, 768]);  getitem_352 = None
    t_78: "f32[768, 3072]" = torch.ops.aten.t.default(primals_509);  primals_509 = None
    addmm_58: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_510, view_280, t_78);  primals_510 = None
    view_281: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_58, [8, 784, 3072]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_41: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_281)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_159: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_41);  gelu_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_282: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_159, [6272, 3072]);  clone_159 = None
    t_79: "f32[3072, 768]" = torch.ops.aten.t.default(primals_511);  primals_511 = None
    addmm_59: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_512, view_282, t_79);  primals_512 = None
    view_283: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_59, [8, 784, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_160: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_283);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_82: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_80, clone_160)
    add_105: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_104, mul_82);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_60 = torch.ops.aten.native_layer_norm.default(add_105, [768], primals_513, primals_514, 1e-06)
    getitem_355: "f32[8, 784, 768]" = native_layer_norm_60[0]
    getitem_356: "f32[8, 784, 1]" = native_layer_norm_60[1]
    getitem_357: "f32[8, 784, 1]" = native_layer_norm_60[2];  native_layer_norm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_284: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_355, [6272, 768]);  getitem_355 = None
    t_80: "f32[768, 2304]" = torch.ops.aten.t.default(primals_515);  primals_515 = None
    addmm_60: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_516, view_284, t_80);  primals_516 = None
    view_285: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_60, [8, 784, 2304]);  addmm_60 = None
    view_286: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_285, [8, 784, 3, 16, 48]);  view_285 = None
    permute_82: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_286, [2, 0, 3, 4, 1]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_20 = torch.ops.aten.unbind.int(permute_82);  permute_82 = None
    getitem_358: "f32[8, 16, 48, 784]" = unbind_20[0]
    getitem_359: "f32[8, 16, 48, 784]" = unbind_20[1]
    getitem_360: "f32[8, 16, 48, 784]" = unbind_20[2];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_40: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_358, 2.0, [-1], True)
    detach_60: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_40)
    clamp_min_40: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_40, 1e-12)
    expand_120: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_40, [8, 16, 48, 784]);  clamp_min_40 = None
    div_46: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_358, expand_120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_41: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_359, 2.0, [-1], True)
    detach_61: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_41)
    clamp_min_41: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_41, 1e-12)
    expand_121: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_41, [8, 16, 48, 784]);  clamp_min_41 = None
    div_47: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_359, expand_121)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_21: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_47, -2, -1);  div_47 = None
    expand_122: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_46, [8, 16, 48, 784]);  div_46 = None
    clone_161: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
    _unsafe_view_80: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_161, [128, 48, 784]);  clone_161 = None
    expand_123: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_21, [8, 16, 784, 48]);  transpose_21 = None
    clone_162: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
    _unsafe_view_81: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_162, [128, 784, 48]);  clone_162 = None
    bmm_40: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_80, _unsafe_view_81)
    view_287: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_40, [8, 16, 48, 48]);  bmm_40 = None
    mul_83: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_287, primals_82)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_20: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_83, -1, False);  mul_83 = None
    detach_62: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_163: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_20);  _softmax_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_124: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_163, [8, 16, 48, 48]);  clone_163 = None
    view_288: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_124, [128, 48, 48]);  expand_124 = None
    expand_125: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_360, [8, 16, 48, 784]);  getitem_360 = None
    clone_164: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
    _unsafe_view_82: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_164, [128, 48, 784]);  clone_164 = None
    bmm_41: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_288, _unsafe_view_82)
    view_289: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_41, [8, 16, 48, 784]);  bmm_41 = None
    permute_83: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_289, [0, 3, 1, 2]);  view_289 = None
    view_290: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_83, [8, 784, 768]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_81: "f32[768, 768]" = torch.ops.aten.t.default(primals_517);  primals_517 = None
    clone_165: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_290, memory_format = torch.contiguous_format);  view_290 = None
    _unsafe_view_83: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_165, [6272, 768]);  clone_165 = None
    mm_20: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_83, t_81)
    view_291: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_20, [8, 784, 768]);  mm_20 = None
    add_106: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_291, primals_518);  view_291 = primals_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_166: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_106);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_84: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_81, clone_166)
    add_107: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_105, mul_84);  mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_61 = torch.ops.aten.native_layer_norm.default(add_107, [768], primals_519, primals_520, 1e-06)
    getitem_361: "f32[8, 784, 768]" = native_layer_norm_61[0]
    getitem_362: "f32[8, 784, 1]" = native_layer_norm_61[1]
    getitem_363: "f32[8, 784, 1]" = native_layer_norm_61[2];  native_layer_norm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_84: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_361, [0, 2, 1]);  getitem_361 = None
    view_292: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_84, [8, 768, 28, 28]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_44: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_292, primals_521, primals_522, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_42: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_108: "i64[]" = torch.ops.aten.add.Tensor(primals_700, 1);  primals_700 = None
    _native_batch_norm_legit_functional_23 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_42, primals_523, primals_524, primals_698, primals_699, True, 0.1, 1e-05);  primals_524 = primals_698 = primals_699 = None
    getitem_364: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_23[0]
    getitem_365: "f32[768]" = _native_batch_norm_legit_functional_23[1]
    getitem_366: "f32[768]" = _native_batch_norm_legit_functional_23[2]
    getitem_367: "f32[768]" = _native_batch_norm_legit_functional_23[3]
    getitem_368: "f32[768]" = _native_batch_norm_legit_functional_23[4];  _native_batch_norm_legit_functional_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_45: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_364, primals_525, primals_526, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_293: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_45, [8, 768, 784]);  convolution_45 = None
    permute_85: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_85: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_83, permute_85)
    add_109: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_107, mul_85);  mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_62 = torch.ops.aten.native_layer_norm.default(add_109, [768], primals_527, primals_528, 1e-06)
    getitem_369: "f32[8, 784, 768]" = native_layer_norm_62[0]
    getitem_370: "f32[8, 784, 1]" = native_layer_norm_62[1]
    getitem_371: "f32[8, 784, 1]" = native_layer_norm_62[2];  native_layer_norm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_294: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_369, [6272, 768]);  getitem_369 = None
    t_82: "f32[768, 3072]" = torch.ops.aten.t.default(primals_529);  primals_529 = None
    addmm_61: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_530, view_294, t_82);  primals_530 = None
    view_295: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_61, [8, 784, 3072]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_43: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_295)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_167: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_43);  gelu_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_296: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_167, [6272, 3072]);  clone_167 = None
    t_83: "f32[3072, 768]" = torch.ops.aten.t.default(primals_531);  primals_531 = None
    addmm_62: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_532, view_296, t_83);  primals_532 = None
    view_297: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_62, [8, 784, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_168: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_297);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_86: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_84, clone_168)
    add_110: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_109, mul_86);  mul_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_63 = torch.ops.aten.native_layer_norm.default(add_110, [768], primals_533, primals_534, 1e-06)
    getitem_372: "f32[8, 784, 768]" = native_layer_norm_63[0]
    getitem_373: "f32[8, 784, 1]" = native_layer_norm_63[1]
    getitem_374: "f32[8, 784, 1]" = native_layer_norm_63[2];  native_layer_norm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_298: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_372, [6272, 768]);  getitem_372 = None
    t_84: "f32[768, 2304]" = torch.ops.aten.t.default(primals_535);  primals_535 = None
    addmm_63: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_536, view_298, t_84);  primals_536 = None
    view_299: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_63, [8, 784, 2304]);  addmm_63 = None
    view_300: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_299, [8, 784, 3, 16, 48]);  view_299 = None
    permute_86: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_300, [2, 0, 3, 4, 1]);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_21 = torch.ops.aten.unbind.int(permute_86);  permute_86 = None
    getitem_375: "f32[8, 16, 48, 784]" = unbind_21[0]
    getitem_376: "f32[8, 16, 48, 784]" = unbind_21[1]
    getitem_377: "f32[8, 16, 48, 784]" = unbind_21[2];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_42: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_375, 2.0, [-1], True)
    detach_63: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_42)
    clamp_min_42: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_42, 1e-12)
    expand_126: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_42, [8, 16, 48, 784]);  clamp_min_42 = None
    div_48: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_375, expand_126)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_43: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_376, 2.0, [-1], True)
    detach_64: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_43)
    clamp_min_43: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_43, 1e-12)
    expand_127: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_43, [8, 16, 48, 784]);  clamp_min_43 = None
    div_49: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_376, expand_127)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_22: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_49, -2, -1);  div_49 = None
    expand_128: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_48, [8, 16, 48, 784]);  div_48 = None
    clone_169: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
    _unsafe_view_84: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_169, [128, 48, 784]);  clone_169 = None
    expand_129: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_22, [8, 16, 784, 48]);  transpose_22 = None
    clone_170: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
    _unsafe_view_85: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_170, [128, 784, 48]);  clone_170 = None
    bmm_42: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_84, _unsafe_view_85)
    view_301: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_42, [8, 16, 48, 48]);  bmm_42 = None
    mul_87: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_301, primals_86)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_21: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_87, -1, False);  mul_87 = None
    detach_65: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_171: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_21);  _softmax_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_130: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_171, [8, 16, 48, 48]);  clone_171 = None
    view_302: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_130, [128, 48, 48]);  expand_130 = None
    expand_131: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_377, [8, 16, 48, 784]);  getitem_377 = None
    clone_172: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
    _unsafe_view_86: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_172, [128, 48, 784]);  clone_172 = None
    bmm_43: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_302, _unsafe_view_86)
    view_303: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_43, [8, 16, 48, 784]);  bmm_43 = None
    permute_87: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_303, [0, 3, 1, 2]);  view_303 = None
    view_304: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_87, [8, 784, 768]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_85: "f32[768, 768]" = torch.ops.aten.t.default(primals_537);  primals_537 = None
    clone_173: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_304, memory_format = torch.contiguous_format);  view_304 = None
    _unsafe_view_87: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_173, [6272, 768]);  clone_173 = None
    mm_21: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_87, t_85)
    view_305: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_21, [8, 784, 768]);  mm_21 = None
    add_111: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_305, primals_538);  view_305 = primals_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_174: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_111);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_88: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_85, clone_174)
    add_112: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_110, mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_64 = torch.ops.aten.native_layer_norm.default(add_112, [768], primals_539, primals_540, 1e-06)
    getitem_378: "f32[8, 784, 768]" = native_layer_norm_64[0]
    getitem_379: "f32[8, 784, 1]" = native_layer_norm_64[1]
    getitem_380: "f32[8, 784, 1]" = native_layer_norm_64[2];  native_layer_norm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_88: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_378, [0, 2, 1]);  getitem_378 = None
    view_306: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_88, [8, 768, 28, 28]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_46: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_306, primals_541, primals_542, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_44: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_113: "i64[]" = torch.ops.aten.add.Tensor(primals_703, 1);  primals_703 = None
    _native_batch_norm_legit_functional_24 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_44, primals_543, primals_544, primals_701, primals_702, True, 0.1, 1e-05);  primals_544 = primals_701 = primals_702 = None
    getitem_381: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_24[0]
    getitem_382: "f32[768]" = _native_batch_norm_legit_functional_24[1]
    getitem_383: "f32[768]" = _native_batch_norm_legit_functional_24[2]
    getitem_384: "f32[768]" = _native_batch_norm_legit_functional_24[3]
    getitem_385: "f32[768]" = _native_batch_norm_legit_functional_24[4];  _native_batch_norm_legit_functional_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_47: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_381, primals_545, primals_546, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_307: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_47, [8, 768, 784]);  convolution_47 = None
    permute_89: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_307, [0, 2, 1]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_89: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_87, permute_89)
    add_114: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_112, mul_89);  mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_65 = torch.ops.aten.native_layer_norm.default(add_114, [768], primals_547, primals_548, 1e-06)
    getitem_386: "f32[8, 784, 768]" = native_layer_norm_65[0]
    getitem_387: "f32[8, 784, 1]" = native_layer_norm_65[1]
    getitem_388: "f32[8, 784, 1]" = native_layer_norm_65[2];  native_layer_norm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_308: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_386, [6272, 768]);  getitem_386 = None
    t_86: "f32[768, 3072]" = torch.ops.aten.t.default(primals_549);  primals_549 = None
    addmm_64: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_550, view_308, t_86);  primals_550 = None
    view_309: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_64, [8, 784, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_45: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_309)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_175: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_45);  gelu_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_310: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_175, [6272, 3072]);  clone_175 = None
    t_87: "f32[3072, 768]" = torch.ops.aten.t.default(primals_551);  primals_551 = None
    addmm_65: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_552, view_310, t_87);  primals_552 = None
    view_311: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_65, [8, 784, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_176: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_311);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_90: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_88, clone_176)
    add_115: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_114, mul_90);  mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_66 = torch.ops.aten.native_layer_norm.default(add_115, [768], primals_553, primals_554, 1e-06)
    getitem_389: "f32[8, 784, 768]" = native_layer_norm_66[0]
    getitem_390: "f32[8, 784, 1]" = native_layer_norm_66[1]
    getitem_391: "f32[8, 784, 1]" = native_layer_norm_66[2];  native_layer_norm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_312: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_389, [6272, 768]);  getitem_389 = None
    t_88: "f32[768, 2304]" = torch.ops.aten.t.default(primals_555);  primals_555 = None
    addmm_66: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_556, view_312, t_88);  primals_556 = None
    view_313: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_66, [8, 784, 2304]);  addmm_66 = None
    view_314: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_313, [8, 784, 3, 16, 48]);  view_313 = None
    permute_90: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_314, [2, 0, 3, 4, 1]);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_22 = torch.ops.aten.unbind.int(permute_90);  permute_90 = None
    getitem_392: "f32[8, 16, 48, 784]" = unbind_22[0]
    getitem_393: "f32[8, 16, 48, 784]" = unbind_22[1]
    getitem_394: "f32[8, 16, 48, 784]" = unbind_22[2];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_44: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_392, 2.0, [-1], True)
    detach_66: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_44)
    clamp_min_44: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_44, 1e-12)
    expand_132: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_44, [8, 16, 48, 784]);  clamp_min_44 = None
    div_50: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_392, expand_132)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_45: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_393, 2.0, [-1], True)
    detach_67: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_45)
    clamp_min_45: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_45, 1e-12)
    expand_133: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_45, [8, 16, 48, 784]);  clamp_min_45 = None
    div_51: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_393, expand_133)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_23: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_51, -2, -1);  div_51 = None
    expand_134: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_50, [8, 16, 48, 784]);  div_50 = None
    clone_177: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_134, memory_format = torch.contiguous_format);  expand_134 = None
    _unsafe_view_88: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_177, [128, 48, 784]);  clone_177 = None
    expand_135: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_23, [8, 16, 784, 48]);  transpose_23 = None
    clone_178: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
    _unsafe_view_89: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_178, [128, 784, 48]);  clone_178 = None
    bmm_44: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_88, _unsafe_view_89)
    view_315: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_44, [8, 16, 48, 48]);  bmm_44 = None
    mul_91: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_315, primals_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_22: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_91, -1, False);  mul_91 = None
    detach_68: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_179: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_22);  _softmax_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_136: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_179, [8, 16, 48, 48]);  clone_179 = None
    view_316: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_136, [128, 48, 48]);  expand_136 = None
    expand_137: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_394, [8, 16, 48, 784]);  getitem_394 = None
    clone_180: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
    _unsafe_view_90: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_180, [128, 48, 784]);  clone_180 = None
    bmm_45: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_316, _unsafe_view_90)
    view_317: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_45, [8, 16, 48, 784]);  bmm_45 = None
    permute_91: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_317, [0, 3, 1, 2]);  view_317 = None
    view_318: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_91, [8, 784, 768]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_89: "f32[768, 768]" = torch.ops.aten.t.default(primals_557);  primals_557 = None
    clone_181: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_318, memory_format = torch.contiguous_format);  view_318 = None
    _unsafe_view_91: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_181, [6272, 768]);  clone_181 = None
    mm_22: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_91, t_89)
    view_319: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_22, [8, 784, 768]);  mm_22 = None
    add_116: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_319, primals_558);  view_319 = primals_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_182: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_116);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_92: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_89, clone_182)
    add_117: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_115, mul_92);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_67 = torch.ops.aten.native_layer_norm.default(add_117, [768], primals_559, primals_560, 1e-06)
    getitem_395: "f32[8, 784, 768]" = native_layer_norm_67[0]
    getitem_396: "f32[8, 784, 1]" = native_layer_norm_67[1]
    getitem_397: "f32[8, 784, 1]" = native_layer_norm_67[2];  native_layer_norm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_92: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_395, [0, 2, 1]);  getitem_395 = None
    view_320: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_92, [8, 768, 28, 28]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_48: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_320, primals_561, primals_562, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_46: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_118: "i64[]" = torch.ops.aten.add.Tensor(primals_706, 1);  primals_706 = None
    _native_batch_norm_legit_functional_25 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_46, primals_563, primals_564, primals_704, primals_705, True, 0.1, 1e-05);  primals_564 = primals_704 = primals_705 = None
    getitem_398: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_25[0]
    getitem_399: "f32[768]" = _native_batch_norm_legit_functional_25[1]
    getitem_400: "f32[768]" = _native_batch_norm_legit_functional_25[2]
    getitem_401: "f32[768]" = _native_batch_norm_legit_functional_25[3]
    getitem_402: "f32[768]" = _native_batch_norm_legit_functional_25[4];  _native_batch_norm_legit_functional_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_49: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_398, primals_565, primals_566, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_321: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_49, [8, 768, 784]);  convolution_49 = None
    permute_93: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_93: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_91, permute_93)
    add_119: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_117, mul_93);  mul_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_68 = torch.ops.aten.native_layer_norm.default(add_119, [768], primals_567, primals_568, 1e-06)
    getitem_403: "f32[8, 784, 768]" = native_layer_norm_68[0]
    getitem_404: "f32[8, 784, 1]" = native_layer_norm_68[1]
    getitem_405: "f32[8, 784, 1]" = native_layer_norm_68[2];  native_layer_norm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_322: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_403, [6272, 768]);  getitem_403 = None
    t_90: "f32[768, 3072]" = torch.ops.aten.t.default(primals_569);  primals_569 = None
    addmm_67: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_570, view_322, t_90);  primals_570 = None
    view_323: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_67, [8, 784, 3072]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_47: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_323)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_183: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_47);  gelu_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_324: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_183, [6272, 3072]);  clone_183 = None
    t_91: "f32[3072, 768]" = torch.ops.aten.t.default(primals_571);  primals_571 = None
    addmm_68: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_572, view_324, t_91);  primals_572 = None
    view_325: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_68, [8, 784, 768]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_184: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_325);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_94: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_92, clone_184)
    add_120: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_119, mul_94);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_69 = torch.ops.aten.native_layer_norm.default(add_120, [768], primals_573, primals_574, 1e-06)
    getitem_406: "f32[8, 784, 768]" = native_layer_norm_69[0]
    getitem_407: "f32[8, 784, 1]" = native_layer_norm_69[1]
    getitem_408: "f32[8, 784, 1]" = native_layer_norm_69[2];  native_layer_norm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_326: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_406, [6272, 768]);  getitem_406 = None
    t_92: "f32[768, 2304]" = torch.ops.aten.t.default(primals_575);  primals_575 = None
    addmm_69: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_576, view_326, t_92);  primals_576 = None
    view_327: "f32[8, 784, 2304]" = torch.ops.aten.view.default(addmm_69, [8, 784, 2304]);  addmm_69 = None
    view_328: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.view.default(view_327, [8, 784, 3, 16, 48]);  view_327 = None
    permute_94: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_328, [2, 0, 3, 4, 1]);  view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_23 = torch.ops.aten.unbind.int(permute_94);  permute_94 = None
    getitem_409: "f32[8, 16, 48, 784]" = unbind_23[0]
    getitem_410: "f32[8, 16, 48, 784]" = unbind_23[1]
    getitem_411: "f32[8, 16, 48, 784]" = unbind_23[2];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    linalg_vector_norm_46: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_409, 2.0, [-1], True)
    detach_69: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_46)
    clamp_min_46: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_46, 1e-12)
    expand_138: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_46, [8, 16, 48, 784]);  clamp_min_46 = None
    div_52: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_409, expand_138)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_47: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_410, 2.0, [-1], True)
    detach_70: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(linalg_vector_norm_47)
    clamp_min_47: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_47, 1e-12)
    expand_139: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_47, [8, 16, 48, 784]);  clamp_min_47 = None
    div_53: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_410, expand_139)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_24: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_53, -2, -1);  div_53 = None
    expand_140: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_52, [8, 16, 48, 784]);  div_52 = None
    clone_185: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
    _unsafe_view_92: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_185, [128, 48, 784]);  clone_185 = None
    expand_141: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_24, [8, 16, 784, 48]);  transpose_24 = None
    clone_186: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
    _unsafe_view_93: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_186, [128, 784, 48]);  clone_186 = None
    bmm_46: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_92, _unsafe_view_93)
    view_329: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_46, [8, 16, 48, 48]);  bmm_46 = None
    mul_95: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_329, primals_94)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    _softmax_23: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax.default(mul_95, -1, False);  mul_95 = None
    detach_71: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(_softmax_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_187: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(_softmax_23);  _softmax_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_142: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_187, [8, 16, 48, 48]);  clone_187 = None
    view_330: "f32[128, 48, 48]" = torch.ops.aten.view.default(expand_142, [128, 48, 48]);  expand_142 = None
    expand_143: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_411, [8, 16, 48, 784]);  getitem_411 = None
    clone_188: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_143, memory_format = torch.contiguous_format);  expand_143 = None
    _unsafe_view_94: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_188, [128, 48, 784]);  clone_188 = None
    bmm_47: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_330, _unsafe_view_94)
    view_331: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_47, [8, 16, 48, 784]);  bmm_47 = None
    permute_95: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_331, [0, 3, 1, 2]);  view_331 = None
    view_332: "f32[8, 784, 768]" = torch.ops.aten.view.default(permute_95, [8, 784, 768]);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_93: "f32[768, 768]" = torch.ops.aten.t.default(primals_577);  primals_577 = None
    clone_189: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_332, memory_format = torch.contiguous_format);  view_332 = None
    _unsafe_view_95: "f32[6272, 768]" = torch.ops.aten._unsafe_view.default(clone_189, [6272, 768]);  clone_189 = None
    mm_23: "f32[6272, 768]" = torch.ops.aten.mm.default(_unsafe_view_95, t_93)
    view_333: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_23, [8, 784, 768]);  mm_23 = None
    add_121: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_333, primals_578);  view_333 = primals_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_190: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_121);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_96: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_93, clone_190)
    add_122: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_120, mul_96);  mul_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_70 = torch.ops.aten.native_layer_norm.default(add_122, [768], primals_579, primals_580, 1e-06)
    getitem_412: "f32[8, 784, 768]" = native_layer_norm_70[0]
    getitem_413: "f32[8, 784, 1]" = native_layer_norm_70[1]
    getitem_414: "f32[8, 784, 1]" = native_layer_norm_70[2];  native_layer_norm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_96: "f32[8, 768, 784]" = torch.ops.aten.permute.default(getitem_412, [0, 2, 1]);  getitem_412 = None
    view_334: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_96, [8, 768, 28, 28]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_50: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_334, primals_581, primals_582, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_48: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu.default(convolution_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_123: "i64[]" = torch.ops.aten.add.Tensor(primals_709, 1);  primals_709 = None
    _native_batch_norm_legit_functional_26 = torch.ops.aten._native_batch_norm_legit_functional.default(gelu_48, primals_583, primals_584, primals_707, primals_708, True, 0.1, 1e-05);  primals_584 = primals_707 = primals_708 = None
    getitem_415: "f32[8, 768, 28, 28]" = _native_batch_norm_legit_functional_26[0]
    getitem_416: "f32[768]" = _native_batch_norm_legit_functional_26[1]
    getitem_417: "f32[768]" = _native_batch_norm_legit_functional_26[2]
    getitem_418: "f32[768]" = _native_batch_norm_legit_functional_26[3]
    getitem_419: "f32[768]" = _native_batch_norm_legit_functional_26[4];  _native_batch_norm_legit_functional_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_51: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(getitem_415, primals_585, primals_586, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_335: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_51, [8, 768, 784]);  convolution_51 = None
    permute_97: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_335, [0, 2, 1]);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_97: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_95, permute_97)
    add_124: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_122, mul_97);  mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_71 = torch.ops.aten.native_layer_norm.default(add_124, [768], primals_587, primals_588, 1e-06)
    getitem_420: "f32[8, 784, 768]" = native_layer_norm_71[0]
    getitem_421: "f32[8, 784, 1]" = native_layer_norm_71[1]
    getitem_422: "f32[8, 784, 1]" = native_layer_norm_71[2];  native_layer_norm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_336: "f32[6272, 768]" = torch.ops.aten.view.default(getitem_420, [6272, 768]);  getitem_420 = None
    t_94: "f32[768, 3072]" = torch.ops.aten.t.default(primals_589);  primals_589 = None
    addmm_70: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_590, view_336, t_94);  primals_590 = None
    view_337: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_70, [8, 784, 3072]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_49: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_337)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_191: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_49);  gelu_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_338: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_191, [6272, 3072]);  clone_191 = None
    t_95: "f32[3072, 768]" = torch.ops.aten.t.default(primals_591);  primals_591 = None
    addmm_71: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_592, view_338, t_95);  primals_592 = None
    view_339: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_71, [8, 784, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_192: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_339);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_98: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_96, clone_192)
    add_125: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_124, mul_98);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:447, code: x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
    expand_144: "f32[8, 1, 768]" = torch.ops.aten.expand.default(primals_97, [8, -1, -1]);  primals_97 = None
    cat_1: "f32[8, 785, 768]" = torch.ops.aten.cat.default([expand_144, add_125], 1);  expand_144 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    native_layer_norm_72 = torch.ops.aten.native_layer_norm.default(cat_1, [768], primals_593, primals_594, 1e-06)
    getitem_423: "f32[8, 785, 768]" = native_layer_norm_72[0]
    getitem_424: "f32[8, 785, 1]" = native_layer_norm_72[1]
    getitem_425: "f32[8, 785, 1]" = native_layer_norm_72[2];  native_layer_norm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_29: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(getitem_423, 0, 0, 9223372036854775807)
    select: "f32[8, 768]" = torch.ops.aten.select.int(slice_29, 1, 0);  slice_29 = None
    t_96: "f32[768, 768]" = torch.ops.aten.t.default(primals_595);  primals_595 = None
    addmm_72: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_596, select, t_96);  primals_596 = None
    unsqueeze_3: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_72, 1);  addmm_72 = None
    view_340: "f32[8, 1, 16, 48]" = torch.ops.aten.view.default(unsqueeze_3, [8, 1, 16, 48]);  unsqueeze_3 = None
    permute_98: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_341: "f32[6280, 768]" = torch.ops.aten.view.default(getitem_423, [6280, 768])
    t_97: "f32[768, 768]" = torch.ops.aten.t.default(primals_597);  primals_597 = None
    addmm_73: "f32[6280, 768]" = torch.ops.aten.addmm.default(primals_598, view_341, t_97);  primals_598 = None
    view_342: "f32[8, 785, 768]" = torch.ops.aten.view.default(addmm_73, [8, 785, 768]);  addmm_73 = None
    view_343: "f32[8, 785, 16, 48]" = torch.ops.aten.view.default(view_342, [8, 785, 16, 48]);  view_342 = None
    permute_99: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_344: "f32[6280, 768]" = torch.ops.aten.view.default(getitem_423, [6280, 768])
    t_98: "f32[768, 768]" = torch.ops.aten.t.default(primals_599);  primals_599 = None
    addmm_74: "f32[6280, 768]" = torch.ops.aten.addmm.default(primals_600, view_344, t_98);  primals_600 = None
    view_345: "f32[8, 785, 768]" = torch.ops.aten.view.default(addmm_74, [8, 785, 768]);  addmm_74 = None
    view_346: "f32[8, 785, 16, 48]" = torch.ops.aten.view.default(view_345, [8, 785, 16, 48]);  view_345 = None
    permute_100: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_98, permute_99, permute_100)
    getitem_426: "f32[8, 16, 1, 48]" = _scaled_dot_product_flash_attention[0]
    getitem_427: "f32[8, 16, 1]" = _scaled_dot_product_flash_attention[1]
    getitem_428: "i32[]" = _scaled_dot_product_flash_attention[2]
    getitem_429: "i32[]" = _scaled_dot_product_flash_attention[3]
    getitem_432: "i64[]" = _scaled_dot_product_flash_attention[6]
    getitem_433: "i64[]" = _scaled_dot_product_flash_attention[7];  _scaled_dot_product_flash_attention = None
    detach_72: "f32[8, 16, 1, 48]" = torch.ops.aten.detach.default(getitem_426)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    transpose_25: "f32[8, 1, 16, 48]" = torch.ops.aten.transpose.int(getitem_426, 1, 2);  getitem_426 = None
    view_347: "f32[8, 1, 768]" = torch.ops.aten.view.default(transpose_25, [8, 1, 768]);  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_348: "f32[8, 768]" = torch.ops.aten.view.default(view_347, [8, 768]);  view_347 = None
    t_99: "f32[768, 768]" = torch.ops.aten.t.default(primals_601);  primals_601 = None
    addmm_75: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_602, view_348, t_99);  primals_602 = None
    view_349: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_75, [8, 1, 768]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:64, code: x_cls = self.proj_drop(x_cls)
    clone_193: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_349);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:182, code: x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
    slice_30: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(getitem_423, 0, 0, 9223372036854775807);  getitem_423 = None
    slice_31: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_30, 1, 1, 9223372036854775807);  slice_30 = None
    cat_2: "f32[8, 785, 768]" = torch.ops.aten.cat.default([clone_193, slice_31], 1);  clone_193 = slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:183, code: x = x + self.drop_path(self.gamma1 * x_attn)
    mul_99: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(primals_98, cat_2)
    add_126: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(cat_1, mul_99);  mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    native_layer_norm_73 = torch.ops.aten.native_layer_norm.default(add_126, [768], primals_603, primals_604, 1e-06)
    getitem_435: "f32[8, 785, 768]" = native_layer_norm_73[0]
    getitem_436: "f32[8, 785, 1]" = native_layer_norm_73[1]
    getitem_437: "f32[8, 785, 1]" = native_layer_norm_73[2];  native_layer_norm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    slice_32: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(getitem_435, 0, 0, 9223372036854775807)
    slice_33: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(slice_32, 1, 0, 1);  slice_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_100: "f32[768, 3072]" = torch.ops.aten.t.default(primals_605);  primals_605 = None
    view_350: "f32[8, 768]" = torch.ops.aten.view.default(slice_33, [8, 768]);  slice_33 = None
    mm_24: "f32[8, 3072]" = torch.ops.aten.mm.default(view_350, t_100)
    view_351: "f32[8, 1, 3072]" = torch.ops.aten.view.default(mm_24, [8, 1, 3072]);  mm_24 = None
    add_127: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(view_351, primals_606);  view_351 = primals_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_50: "f32[8, 1, 3072]" = torch.ops.aten.gelu.default(add_127)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_194: "f32[8, 1, 3072]" = torch.ops.aten.clone.default(gelu_50);  gelu_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_352: "f32[8, 3072]" = torch.ops.aten.view.default(clone_194, [8, 3072]);  clone_194 = None
    t_101: "f32[3072, 768]" = torch.ops.aten.t.default(primals_607);  primals_607 = None
    addmm_76: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_608, view_352, t_101);  primals_608 = None
    view_353: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_76, [8, 1, 768]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_195: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_353);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    mul_100: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(primals_99, clone_195)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    slice_34: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(getitem_435, 0, 0, 9223372036854775807)
    slice_35: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_34, 1, 1, 9223372036854775807);  slice_34 = None
    cat_3: "f32[8, 785, 768]" = torch.ops.aten.cat.default([mul_100, slice_35], 1);  mul_100 = slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:192, code: x = x_res + self.drop_path(x)
    add_128: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(getitem_435, cat_3);  getitem_435 = cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    native_layer_norm_74 = torch.ops.aten.native_layer_norm.default(add_128, [768], primals_609, primals_610, 1e-06)
    getitem_438: "f32[8, 785, 768]" = native_layer_norm_74[0]
    getitem_439: "f32[8, 785, 1]" = native_layer_norm_74[1]
    getitem_440: "f32[8, 785, 1]" = native_layer_norm_74[2];  native_layer_norm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_36: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(getitem_438, 0, 0, 9223372036854775807)
    select_1: "f32[8, 768]" = torch.ops.aten.select.int(slice_36, 1, 0);  slice_36 = None
    t_102: "f32[768, 768]" = torch.ops.aten.t.default(primals_611);  primals_611 = None
    addmm_77: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_612, select_1, t_102);  primals_612 = None
    unsqueeze_4: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_77, 1);  addmm_77 = None
    view_354: "f32[8, 1, 16, 48]" = torch.ops.aten.view.default(unsqueeze_4, [8, 1, 16, 48]);  unsqueeze_4 = None
    permute_101: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_355: "f32[6280, 768]" = torch.ops.aten.view.default(getitem_438, [6280, 768])
    t_103: "f32[768, 768]" = torch.ops.aten.t.default(primals_613);  primals_613 = None
    addmm_78: "f32[6280, 768]" = torch.ops.aten.addmm.default(primals_614, view_355, t_103);  primals_614 = None
    view_356: "f32[8, 785, 768]" = torch.ops.aten.view.default(addmm_78, [8, 785, 768]);  addmm_78 = None
    view_357: "f32[8, 785, 16, 48]" = torch.ops.aten.view.default(view_356, [8, 785, 16, 48]);  view_356 = None
    permute_102: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_358: "f32[6280, 768]" = torch.ops.aten.view.default(getitem_438, [6280, 768])
    t_104: "f32[768, 768]" = torch.ops.aten.t.default(primals_615);  primals_615 = None
    addmm_79: "f32[6280, 768]" = torch.ops.aten.addmm.default(primals_616, view_358, t_104);  primals_616 = None
    view_359: "f32[8, 785, 768]" = torch.ops.aten.view.default(addmm_79, [8, 785, 768]);  addmm_79 = None
    view_360: "f32[8, 785, 16, 48]" = torch.ops.aten.view.default(view_359, [8, 785, 16, 48]);  view_359 = None
    permute_103: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_101, permute_102, permute_103)
    getitem_441: "f32[8, 16, 1, 48]" = _scaled_dot_product_flash_attention_1[0]
    getitem_442: "f32[8, 16, 1]" = _scaled_dot_product_flash_attention_1[1]
    getitem_443: "i32[]" = _scaled_dot_product_flash_attention_1[2]
    getitem_444: "i32[]" = _scaled_dot_product_flash_attention_1[3]
    getitem_447: "i64[]" = _scaled_dot_product_flash_attention_1[6]
    getitem_448: "i64[]" = _scaled_dot_product_flash_attention_1[7];  _scaled_dot_product_flash_attention_1 = None
    detach_73: "f32[8, 16, 1, 48]" = torch.ops.aten.detach.default(getitem_441)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    transpose_26: "f32[8, 1, 16, 48]" = torch.ops.aten.transpose.int(getitem_441, 1, 2);  getitem_441 = None
    view_361: "f32[8, 1, 768]" = torch.ops.aten.view.default(transpose_26, [8, 1, 768]);  transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_362: "f32[8, 768]" = torch.ops.aten.view.default(view_361, [8, 768]);  view_361 = None
    t_105: "f32[768, 768]" = torch.ops.aten.t.default(primals_617);  primals_617 = None
    addmm_80: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_618, view_362, t_105);  primals_618 = None
    view_363: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_80, [8, 1, 768]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:64, code: x_cls = self.proj_drop(x_cls)
    clone_196: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_363);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:182, code: x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
    slice_37: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(getitem_438, 0, 0, 9223372036854775807);  getitem_438 = None
    slice_38: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_37, 1, 1, 9223372036854775807);  slice_37 = None
    cat_4: "f32[8, 785, 768]" = torch.ops.aten.cat.default([clone_196, slice_38], 1);  clone_196 = slice_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:183, code: x = x + self.drop_path(self.gamma1 * x_attn)
    mul_101: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(primals_100, cat_4)
    add_129: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_128, mul_101);  mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    native_layer_norm_75 = torch.ops.aten.native_layer_norm.default(add_129, [768], primals_619, primals_620, 1e-06)
    getitem_450: "f32[8, 785, 768]" = native_layer_norm_75[0]
    getitem_451: "f32[8, 785, 1]" = native_layer_norm_75[1]
    getitem_452: "f32[8, 785, 1]" = native_layer_norm_75[2];  native_layer_norm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    slice_39: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(getitem_450, 0, 0, 9223372036854775807)
    slice_40: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(slice_39, 1, 0, 1);  slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_106: "f32[768, 3072]" = torch.ops.aten.t.default(primals_621);  primals_621 = None
    view_364: "f32[8, 768]" = torch.ops.aten.view.default(slice_40, [8, 768]);  slice_40 = None
    mm_25: "f32[8, 3072]" = torch.ops.aten.mm.default(view_364, t_106)
    view_365: "f32[8, 1, 3072]" = torch.ops.aten.view.default(mm_25, [8, 1, 3072]);  mm_25 = None
    add_130: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(view_365, primals_622);  view_365 = primals_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_51: "f32[8, 1, 3072]" = torch.ops.aten.gelu.default(add_130)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_197: "f32[8, 1, 3072]" = torch.ops.aten.clone.default(gelu_51);  gelu_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_366: "f32[8, 3072]" = torch.ops.aten.view.default(clone_197, [8, 3072]);  clone_197 = None
    t_107: "f32[3072, 768]" = torch.ops.aten.t.default(primals_623);  primals_623 = None
    addmm_81: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_624, view_366, t_107);  primals_624 = None
    view_367: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_81, [8, 1, 768]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_198: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_367);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    mul_102: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(primals_101, clone_198)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    slice_41: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(getitem_450, 0, 0, 9223372036854775807)
    slice_42: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_41, 1, 1, 9223372036854775807);  slice_41 = None
    cat_5: "f32[8, 785, 768]" = torch.ops.aten.cat.default([mul_102, slice_42], 1);  mul_102 = slice_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:192, code: x = x_res + self.drop_path(x)
    add_131: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(getitem_450, cat_5);  getitem_450 = cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:455, code: x = self.norm(x)
    native_layer_norm_76 = torch.ops.aten.native_layer_norm.default(add_131, [768], primals_625, primals_626, 1e-06)
    getitem_453: "f32[8, 785, 768]" = native_layer_norm_76[0]
    getitem_454: "f32[8, 785, 1]" = native_layer_norm_76[1]
    getitem_455: "f32[8, 785, 1]" = native_layer_norm_76[2];  native_layer_norm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:460, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    slice_43: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(getitem_453, 0, 0, 9223372036854775807);  getitem_453 = None
    select_2: "f32[8, 768]" = torch.ops.aten.select.int(slice_43, 1, 0);  slice_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:461, code: x = self.head_drop(x)
    clone_199: "f32[8, 768]" = torch.ops.aten.clone.default(select_2);  select_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:462, code: return x if pre_logits else self.head(x)
    t_108: "f32[768, 1000]" = torch.ops.aten.t.default(primals_627);  primals_627 = None
    addmm_82: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_628, clone_199, t_108);  primals_628 = None
    t_109: "f32[1000, 768]" = torch.ops.aten.t.default(t_108);  t_108 = None
    mm_26: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, t_109);  t_109 = None
    t_110: "f32[1000, 8]" = torch.ops.aten.t.default(tangents_1)
    mm_27: "f32[1000, 768]" = torch.ops.aten.mm.default(t_110, clone_199);  t_110 = clone_199 = None
    t_111: "f32[768, 1000]" = torch.ops.aten.t.default(mm_27);  mm_27 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_368: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    t_112: "f32[1000, 768]" = torch.ops.aten.t.default(t_111);  t_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:460, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    select_backward: "f32[8, 785, 768]" = torch.ops.aten.select_backward.default(mm_26, [8, 785, 768], 1, 0);  mm_26 = None
    slice_backward: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(select_backward, [8, 785, 768], 0, 0, 9223372036854775807, 1);  select_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:455, code: x = self.norm(x)
    native_layer_norm_backward = torch.ops.aten.native_layer_norm_backward.default(slice_backward, add_131, [768], getitem_454, getitem_455, primals_625, primals_626, [True, True, True]);  slice_backward = add_131 = getitem_454 = getitem_455 = primals_625 = primals_626 = None
    getitem_456: "f32[8, 785, 768]" = native_layer_norm_backward[0]
    getitem_457: "f32[768]" = native_layer_norm_backward[1]
    getitem_458: "f32[768]" = native_layer_norm_backward[2];  native_layer_norm_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    slice_44: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(getitem_456, 1, 0, 1)
    slice_45: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(getitem_456, 1, 1, 785)
    slice_backward_1: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(slice_45, [8, 785, 768], 1, 1, 9223372036854775807, 1);  slice_45 = None
    slice_backward_2: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(slice_backward_1, [8, 785, 768], 0, 0, 9223372036854775807, 1);  slice_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    add_132: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(getitem_456, slice_backward_2);  getitem_456 = slice_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    mul_103: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(slice_44, primals_101);  primals_101 = None
    mul_104: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(slice_44, clone_198);  slice_44 = clone_198 = None
    sum_2: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_104, [0, 1], True);  mul_104 = None
    view_369: "f32[768]" = torch.ops.aten.view.default(sum_2, [768]);  sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_370: "f32[8, 768]" = torch.ops.aten.view.default(mul_103, [8, 768]);  mul_103 = None
    t_113: "f32[768, 3072]" = torch.ops.aten.t.default(t_107);  t_107 = None
    mm_28: "f32[8, 3072]" = torch.ops.aten.mm.default(view_370, t_113);  t_113 = None
    t_114: "f32[768, 8]" = torch.ops.aten.t.default(view_370)
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(t_114, view_366);  t_114 = view_366 = None
    t_115: "f32[3072, 768]" = torch.ops.aten.t.default(mm_29);  mm_29 = None
    sum_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_370, [0], True);  view_370 = None
    view_371: "f32[768]" = torch.ops.aten.view.default(sum_3, [768]);  sum_3 = None
    t_116: "f32[768, 3072]" = torch.ops.aten.t.default(t_115);  t_115 = None
    view_372: "f32[8, 1, 3072]" = torch.ops.aten.view.default(mm_28, [8, 1, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward: "f32[8, 1, 3072]" = torch.ops.aten.gelu_backward.default(view_372, add_130);  view_372 = add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_4: "f32[1, 1, 3072]" = torch.ops.aten.sum.dim_IntList(gelu_backward, [0, 1], True)
    view_373: "f32[3072]" = torch.ops.aten.view.default(sum_4, [3072]);  sum_4 = None
    view_374: "f32[8, 3072]" = torch.ops.aten.view.default(gelu_backward, [8, 3072]);  gelu_backward = None
    t_117: "f32[3072, 8]" = torch.ops.aten.t.default(view_374)
    mm_30: "f32[3072, 768]" = torch.ops.aten.mm.default(t_117, view_364);  t_117 = view_364 = None
    t_118: "f32[768, 3072]" = torch.ops.aten.t.default(mm_30);  mm_30 = None
    t_119: "f32[3072, 768]" = torch.ops.aten.t.default(t_106);  t_106 = None
    mm_31: "f32[8, 768]" = torch.ops.aten.mm.default(view_374, t_119);  view_374 = t_119 = None
    view_375: "f32[8, 1, 768]" = torch.ops.aten.view.default(mm_31, [8, 1, 768]);  mm_31 = None
    t_120: "f32[3072, 768]" = torch.ops.aten.t.default(t_118);  t_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    slice_backward_3: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(view_375, [8, 785, 768], 1, 0, 1, 1);  view_375 = None
    slice_backward_4: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(slice_backward_3, [8, 785, 768], 0, 0, 9223372036854775807, 1);  slice_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    add_133: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_132, slice_backward_4);  add_132 = slice_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    native_layer_norm_backward_1 = torch.ops.aten.native_layer_norm_backward.default(add_133, add_129, [768], getitem_451, getitem_452, primals_619, primals_620, [True, True, True]);  add_133 = add_129 = getitem_451 = getitem_452 = primals_619 = primals_620 = None
    getitem_459: "f32[8, 785, 768]" = native_layer_norm_backward_1[0]
    getitem_460: "f32[768]" = native_layer_norm_backward_1[1]
    getitem_461: "f32[768]" = native_layer_norm_backward_1[2];  native_layer_norm_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:183, code: x = x + self.drop_path(self.gamma1 * x_attn)
    mul_105: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(getitem_459, primals_100);  primals_100 = None
    mul_106: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(getitem_459, cat_4);  cat_4 = None
    sum_5: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_106, [0, 1], True);  mul_106 = None
    view_376: "f32[768]" = torch.ops.aten.view.default(sum_5, [768]);  sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:182, code: x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
    slice_46: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(mul_105, 1, 0, 1)
    slice_47: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(mul_105, 1, 1, 785);  mul_105 = None
    slice_backward_5: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(slice_47, [8, 785, 768], 1, 1, 9223372036854775807, 1);  slice_47 = None
    slice_backward_6: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(slice_backward_5, [8, 785, 768], 0, 0, 9223372036854775807, 1);  slice_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_377: "f32[8, 768]" = torch.ops.aten.view.default(slice_46, [8, 768]);  slice_46 = None
    t_121: "f32[768, 768]" = torch.ops.aten.t.default(t_105);  t_105 = None
    mm_32: "f32[8, 768]" = torch.ops.aten.mm.default(view_377, t_121);  t_121 = None
    t_122: "f32[768, 8]" = torch.ops.aten.t.default(view_377)
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(t_122, view_362);  t_122 = view_362 = None
    t_123: "f32[768, 768]" = torch.ops.aten.t.default(mm_33);  mm_33 = None
    sum_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
    view_378: "f32[768]" = torch.ops.aten.view.default(sum_6, [768]);  sum_6 = None
    t_124: "f32[768, 768]" = torch.ops.aten.t.default(t_123);  t_123 = None
    view_379: "f32[8, 1, 768]" = torch.ops.aten.view.default(mm_32, [8, 1, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    view_380: "f32[8, 1, 16, 48]" = torch.ops.aten.view.default(view_379, [8, 1, 16, 48]);  view_379 = None
    transpose_27: "f32[8, 16, 1, 48]" = torch.ops.aten.transpose.int(view_380, 1, 2);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    detach_74: "f32[8, 16, 1, 48]" = torch.ops.aten.detach.default(detach_73);  detach_73 = None
    _scaled_dot_product_flash_attention_backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(transpose_27, permute_101, permute_102, permute_103, detach_74, getitem_442, getitem_443, getitem_444, 0, 0, 0.0, False, getitem_447, getitem_448);  transpose_27 = permute_101 = permute_102 = permute_103 = detach_74 = getitem_442 = getitem_443 = getitem_444 = getitem_447 = getitem_448 = None
    getitem_462: "f32[8, 16, 1, 48]" = _scaled_dot_product_flash_attention_backward[0]
    getitem_463: "f32[8, 16, 785, 48]" = _scaled_dot_product_flash_attention_backward[1]
    getitem_464: "f32[8, 16, 785, 48]" = _scaled_dot_product_flash_attention_backward[2];  _scaled_dot_product_flash_attention_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_104: "f32[8, 785, 16, 48]" = torch.ops.aten.permute.default(getitem_464, [0, 2, 1, 3]);  getitem_464 = None
    view_381: "f32[8, 785, 768]" = torch.ops.aten.view.default(permute_104, [8, 785, 768]);  permute_104 = None
    view_382: "f32[6280, 768]" = torch.ops.aten.view.default(view_381, [6280, 768]);  view_381 = None
    t_125: "f32[768, 768]" = torch.ops.aten.t.default(t_104);  t_104 = None
    mm_34: "f32[6280, 768]" = torch.ops.aten.mm.default(view_382, t_125);  t_125 = None
    t_126: "f32[768, 6280]" = torch.ops.aten.t.default(view_382)
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(t_126, view_358);  t_126 = view_358 = None
    t_127: "f32[768, 768]" = torch.ops.aten.t.default(mm_35);  mm_35 = None
    sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_382, [0], True);  view_382 = None
    view_383: "f32[768]" = torch.ops.aten.view.default(sum_7, [768]);  sum_7 = None
    t_128: "f32[768, 768]" = torch.ops.aten.t.default(t_127);  t_127 = None
    view_384: "f32[8, 785, 768]" = torch.ops.aten.view.default(mm_34, [8, 785, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_134: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(slice_backward_6, view_384);  slice_backward_6 = view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_105: "f32[8, 785, 16, 48]" = torch.ops.aten.permute.default(getitem_463, [0, 2, 1, 3]);  getitem_463 = None
    view_385: "f32[8, 785, 768]" = torch.ops.aten.view.default(permute_105, [8, 785, 768]);  permute_105 = None
    view_386: "f32[6280, 768]" = torch.ops.aten.view.default(view_385, [6280, 768]);  view_385 = None
    t_129: "f32[768, 768]" = torch.ops.aten.t.default(t_103);  t_103 = None
    mm_36: "f32[6280, 768]" = torch.ops.aten.mm.default(view_386, t_129);  t_129 = None
    t_130: "f32[768, 6280]" = torch.ops.aten.t.default(view_386)
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(t_130, view_355);  t_130 = view_355 = None
    t_131: "f32[768, 768]" = torch.ops.aten.t.default(mm_37);  mm_37 = None
    sum_8: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
    view_387: "f32[768]" = torch.ops.aten.view.default(sum_8, [768]);  sum_8 = None
    t_132: "f32[768, 768]" = torch.ops.aten.t.default(t_131);  t_131 = None
    view_388: "f32[8, 785, 768]" = torch.ops.aten.view.default(mm_36, [8, 785, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_135: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_134, view_388);  add_134 = view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_106: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_462, [0, 2, 1, 3]);  getitem_462 = None
    view_389: "f32[8, 1, 768]" = torch.ops.aten.view.default(permute_106, [8, 1, 768]);  permute_106 = None
    squeeze: "f32[8, 768]" = torch.ops.aten.squeeze.dim(view_389, 1);  view_389 = None
    t_133: "f32[768, 768]" = torch.ops.aten.t.default(t_102);  t_102 = None
    mm_38: "f32[8, 768]" = torch.ops.aten.mm.default(squeeze, t_133);  t_133 = None
    t_134: "f32[768, 8]" = torch.ops.aten.t.default(squeeze)
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(t_134, select_1);  t_134 = select_1 = None
    t_135: "f32[768, 768]" = torch.ops.aten.t.default(mm_39);  mm_39 = None
    sum_9: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(squeeze, [0], True);  squeeze = None
    view_390: "f32[768]" = torch.ops.aten.view.default(sum_9, [768]);  sum_9 = None
    t_136: "f32[768, 768]" = torch.ops.aten.t.default(t_135);  t_135 = None
    select_backward_1: "f32[8, 785, 768]" = torch.ops.aten.select_backward.default(mm_38, [8, 785, 768], 1, 0);  mm_38 = None
    slice_backward_7: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(select_backward_1, [8, 785, 768], 0, 0, 9223372036854775807, 1);  select_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_136: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_135, slice_backward_7);  add_135 = slice_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    native_layer_norm_backward_2 = torch.ops.aten.native_layer_norm_backward.default(add_136, add_128, [768], getitem_439, getitem_440, primals_609, primals_610, [True, True, True]);  add_136 = add_128 = getitem_439 = getitem_440 = primals_609 = primals_610 = None
    getitem_465: "f32[8, 785, 768]" = native_layer_norm_backward_2[0]
    getitem_466: "f32[768]" = native_layer_norm_backward_2[1]
    getitem_467: "f32[768]" = native_layer_norm_backward_2[2];  native_layer_norm_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    add_137: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(getitem_459, getitem_465);  getitem_459 = getitem_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    slice_48: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_137, 1, 0, 1)
    slice_49: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(add_137, 1, 1, 785)
    slice_backward_8: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(slice_49, [8, 785, 768], 1, 1, 9223372036854775807, 1);  slice_49 = None
    slice_backward_9: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(slice_backward_8, [8, 785, 768], 0, 0, 9223372036854775807, 1);  slice_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    add_138: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_137, slice_backward_9);  add_137 = slice_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    mul_107: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(slice_48, primals_99);  primals_99 = None
    mul_108: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(slice_48, clone_195);  slice_48 = clone_195 = None
    sum_10: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_108, [0, 1], True);  mul_108 = None
    view_391: "f32[768]" = torch.ops.aten.view.default(sum_10, [768]);  sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_392: "f32[8, 768]" = torch.ops.aten.view.default(mul_107, [8, 768]);  mul_107 = None
    t_137: "f32[768, 3072]" = torch.ops.aten.t.default(t_101);  t_101 = None
    mm_40: "f32[8, 3072]" = torch.ops.aten.mm.default(view_392, t_137);  t_137 = None
    t_138: "f32[768, 8]" = torch.ops.aten.t.default(view_392)
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(t_138, view_352);  t_138 = view_352 = None
    t_139: "f32[3072, 768]" = torch.ops.aten.t.default(mm_41);  mm_41 = None
    sum_11: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_392, [0], True);  view_392 = None
    view_393: "f32[768]" = torch.ops.aten.view.default(sum_11, [768]);  sum_11 = None
    t_140: "f32[768, 3072]" = torch.ops.aten.t.default(t_139);  t_139 = None
    view_394: "f32[8, 1, 3072]" = torch.ops.aten.view.default(mm_40, [8, 1, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_1: "f32[8, 1, 3072]" = torch.ops.aten.gelu_backward.default(view_394, add_127);  view_394 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_12: "f32[1, 1, 3072]" = torch.ops.aten.sum.dim_IntList(gelu_backward_1, [0, 1], True)
    view_395: "f32[3072]" = torch.ops.aten.view.default(sum_12, [3072]);  sum_12 = None
    view_396: "f32[8, 3072]" = torch.ops.aten.view.default(gelu_backward_1, [8, 3072]);  gelu_backward_1 = None
    t_141: "f32[3072, 8]" = torch.ops.aten.t.default(view_396)
    mm_42: "f32[3072, 768]" = torch.ops.aten.mm.default(t_141, view_350);  t_141 = view_350 = None
    t_142: "f32[768, 3072]" = torch.ops.aten.t.default(mm_42);  mm_42 = None
    t_143: "f32[3072, 768]" = torch.ops.aten.t.default(t_100);  t_100 = None
    mm_43: "f32[8, 768]" = torch.ops.aten.mm.default(view_396, t_143);  view_396 = t_143 = None
    view_397: "f32[8, 1, 768]" = torch.ops.aten.view.default(mm_43, [8, 1, 768]);  mm_43 = None
    t_144: "f32[3072, 768]" = torch.ops.aten.t.default(t_142);  t_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    slice_backward_10: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(view_397, [8, 785, 768], 1, 0, 1, 1);  view_397 = None
    slice_backward_11: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(slice_backward_10, [8, 785, 768], 0, 0, 9223372036854775807, 1);  slice_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    add_139: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_138, slice_backward_11);  add_138 = slice_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    native_layer_norm_backward_3 = torch.ops.aten.native_layer_norm_backward.default(add_139, add_126, [768], getitem_436, getitem_437, primals_603, primals_604, [True, True, True]);  add_139 = add_126 = getitem_436 = getitem_437 = primals_603 = primals_604 = None
    getitem_468: "f32[8, 785, 768]" = native_layer_norm_backward_3[0]
    getitem_469: "f32[768]" = native_layer_norm_backward_3[1]
    getitem_470: "f32[768]" = native_layer_norm_backward_3[2];  native_layer_norm_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:183, code: x = x + self.drop_path(self.gamma1 * x_attn)
    mul_109: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(getitem_468, primals_98);  primals_98 = None
    mul_110: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(getitem_468, cat_2);  cat_2 = None
    sum_13: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_110, [0, 1], True);  mul_110 = None
    view_398: "f32[768]" = torch.ops.aten.view.default(sum_13, [768]);  sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:182, code: x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
    slice_50: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(mul_109, 1, 0, 1)
    slice_51: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(mul_109, 1, 1, 785);  mul_109 = None
    slice_backward_12: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(slice_51, [8, 785, 768], 1, 1, 9223372036854775807, 1);  slice_51 = None
    slice_backward_13: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(slice_backward_12, [8, 785, 768], 0, 0, 9223372036854775807, 1);  slice_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_399: "f32[8, 768]" = torch.ops.aten.view.default(slice_50, [8, 768]);  slice_50 = None
    t_145: "f32[768, 768]" = torch.ops.aten.t.default(t_99);  t_99 = None
    mm_44: "f32[8, 768]" = torch.ops.aten.mm.default(view_399, t_145);  t_145 = None
    t_146: "f32[768, 8]" = torch.ops.aten.t.default(view_399)
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(t_146, view_348);  t_146 = view_348 = None
    t_147: "f32[768, 768]" = torch.ops.aten.t.default(mm_45);  mm_45 = None
    sum_14: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_399, [0], True);  view_399 = None
    view_400: "f32[768]" = torch.ops.aten.view.default(sum_14, [768]);  sum_14 = None
    t_148: "f32[768, 768]" = torch.ops.aten.t.default(t_147);  t_147 = None
    view_401: "f32[8, 1, 768]" = torch.ops.aten.view.default(mm_44, [8, 1, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    view_402: "f32[8, 1, 16, 48]" = torch.ops.aten.view.default(view_401, [8, 1, 16, 48]);  view_401 = None
    transpose_28: "f32[8, 16, 1, 48]" = torch.ops.aten.transpose.int(view_402, 1, 2);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    detach_75: "f32[8, 16, 1, 48]" = torch.ops.aten.detach.default(detach_72);  detach_72 = None
    _scaled_dot_product_flash_attention_backward_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(transpose_28, permute_98, permute_99, permute_100, detach_75, getitem_427, getitem_428, getitem_429, 0, 0, 0.0, False, getitem_432, getitem_433);  transpose_28 = permute_98 = permute_99 = permute_100 = detach_75 = getitem_427 = getitem_428 = getitem_429 = getitem_432 = getitem_433 = None
    getitem_471: "f32[8, 16, 1, 48]" = _scaled_dot_product_flash_attention_backward_1[0]
    getitem_472: "f32[8, 16, 785, 48]" = _scaled_dot_product_flash_attention_backward_1[1]
    getitem_473: "f32[8, 16, 785, 48]" = _scaled_dot_product_flash_attention_backward_1[2];  _scaled_dot_product_flash_attention_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_107: "f32[8, 785, 16, 48]" = torch.ops.aten.permute.default(getitem_473, [0, 2, 1, 3]);  getitem_473 = None
    view_403: "f32[8, 785, 768]" = torch.ops.aten.view.default(permute_107, [8, 785, 768]);  permute_107 = None
    view_404: "f32[6280, 768]" = torch.ops.aten.view.default(view_403, [6280, 768]);  view_403 = None
    t_149: "f32[768, 768]" = torch.ops.aten.t.default(t_98);  t_98 = None
    mm_46: "f32[6280, 768]" = torch.ops.aten.mm.default(view_404, t_149);  t_149 = None
    t_150: "f32[768, 6280]" = torch.ops.aten.t.default(view_404)
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(t_150, view_344);  t_150 = view_344 = None
    t_151: "f32[768, 768]" = torch.ops.aten.t.default(mm_47);  mm_47 = None
    sum_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_404, [0], True);  view_404 = None
    view_405: "f32[768]" = torch.ops.aten.view.default(sum_15, [768]);  sum_15 = None
    t_152: "f32[768, 768]" = torch.ops.aten.t.default(t_151);  t_151 = None
    view_406: "f32[8, 785, 768]" = torch.ops.aten.view.default(mm_46, [8, 785, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_140: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(slice_backward_13, view_406);  slice_backward_13 = view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_108: "f32[8, 785, 16, 48]" = torch.ops.aten.permute.default(getitem_472, [0, 2, 1, 3]);  getitem_472 = None
    view_407: "f32[8, 785, 768]" = torch.ops.aten.view.default(permute_108, [8, 785, 768]);  permute_108 = None
    view_408: "f32[6280, 768]" = torch.ops.aten.view.default(view_407, [6280, 768]);  view_407 = None
    t_153: "f32[768, 768]" = torch.ops.aten.t.default(t_97);  t_97 = None
    mm_48: "f32[6280, 768]" = torch.ops.aten.mm.default(view_408, t_153);  t_153 = None
    t_154: "f32[768, 6280]" = torch.ops.aten.t.default(view_408)
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(t_154, view_341);  t_154 = view_341 = None
    t_155: "f32[768, 768]" = torch.ops.aten.t.default(mm_49);  mm_49 = None
    sum_16: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
    view_409: "f32[768]" = torch.ops.aten.view.default(sum_16, [768]);  sum_16 = None
    t_156: "f32[768, 768]" = torch.ops.aten.t.default(t_155);  t_155 = None
    view_410: "f32[8, 785, 768]" = torch.ops.aten.view.default(mm_48, [8, 785, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_141: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_140, view_410);  add_140 = view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_109: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_471, [0, 2, 1, 3]);  getitem_471 = None
    view_411: "f32[8, 1, 768]" = torch.ops.aten.view.default(permute_109, [8, 1, 768]);  permute_109 = None
    squeeze_1: "f32[8, 768]" = torch.ops.aten.squeeze.dim(view_411, 1);  view_411 = None
    t_157: "f32[768, 768]" = torch.ops.aten.t.default(t_96);  t_96 = None
    mm_50: "f32[8, 768]" = torch.ops.aten.mm.default(squeeze_1, t_157);  t_157 = None
    t_158: "f32[768, 8]" = torch.ops.aten.t.default(squeeze_1)
    mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(t_158, select);  t_158 = select = None
    t_159: "f32[768, 768]" = torch.ops.aten.t.default(mm_51);  mm_51 = None
    sum_17: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(squeeze_1, [0], True);  squeeze_1 = None
    view_412: "f32[768]" = torch.ops.aten.view.default(sum_17, [768]);  sum_17 = None
    t_160: "f32[768, 768]" = torch.ops.aten.t.default(t_159);  t_159 = None
    select_backward_2: "f32[8, 785, 768]" = torch.ops.aten.select_backward.default(mm_50, [8, 785, 768], 1, 0);  mm_50 = None
    slice_backward_14: "f32[8, 785, 768]" = torch.ops.aten.slice_backward.default(select_backward_2, [8, 785, 768], 0, 0, 9223372036854775807, 1);  select_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_142: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_141, slice_backward_14);  add_141 = slice_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    native_layer_norm_backward_4 = torch.ops.aten.native_layer_norm_backward.default(add_142, cat_1, [768], getitem_424, getitem_425, primals_593, primals_594, [True, True, True]);  add_142 = cat_1 = getitem_424 = getitem_425 = primals_593 = primals_594 = None
    getitem_474: "f32[8, 785, 768]" = native_layer_norm_backward_4[0]
    getitem_475: "f32[768]" = native_layer_norm_backward_4[1]
    getitem_476: "f32[768]" = native_layer_norm_backward_4[2];  native_layer_norm_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    add_143: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(getitem_468, getitem_474);  getitem_468 = getitem_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:447, code: x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
    slice_52: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_143, 1, 0, 1)
    slice_53: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(add_143, 1, 1, 785);  add_143 = None
    sum_18: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_52, [0], True);  slice_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_111: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(slice_53, primals_96);  primals_96 = None
    mul_112: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(slice_53, clone_192);  clone_192 = None
    sum_19: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_112, [0, 1], True);  mul_112 = None
    view_413: "f32[768]" = torch.ops.aten.view.default(sum_19, [768]);  sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_414: "f32[6272, 768]" = torch.ops.aten.view.default(mul_111, [6272, 768]);  mul_111 = None
    t_161: "f32[768, 3072]" = torch.ops.aten.t.default(t_95);  t_95 = None
    mm_52: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_414, t_161);  t_161 = None
    t_162: "f32[768, 6272]" = torch.ops.aten.t.default(view_414)
    mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(t_162, view_338);  t_162 = view_338 = None
    t_163: "f32[3072, 768]" = torch.ops.aten.t.default(mm_53);  mm_53 = None
    sum_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[768]" = torch.ops.aten.view.default(sum_20, [768]);  sum_20 = None
    t_164: "f32[768, 3072]" = torch.ops.aten.t.default(t_163);  t_163 = None
    view_416: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_52, [8, 784, 3072]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_2: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_416, view_337);  view_416 = view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_417: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_2, [6272, 3072]);  gelu_backward_2 = None
    t_165: "f32[3072, 768]" = torch.ops.aten.t.default(t_94);  t_94 = None
    mm_54: "f32[6272, 768]" = torch.ops.aten.mm.default(view_417, t_165);  t_165 = None
    t_166: "f32[3072, 6272]" = torch.ops.aten.t.default(view_417)
    mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(t_166, view_336);  t_166 = view_336 = None
    t_167: "f32[768, 3072]" = torch.ops.aten.t.default(mm_55);  mm_55 = None
    sum_21: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True);  view_417 = None
    view_418: "f32[3072]" = torch.ops.aten.view.default(sum_21, [3072]);  sum_21 = None
    t_168: "f32[3072, 768]" = torch.ops.aten.t.default(t_167);  t_167 = None
    view_419: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_54, [8, 784, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_5 = torch.ops.aten.native_layer_norm_backward.default(view_419, add_124, [768], getitem_421, getitem_422, primals_587, primals_588, [True, True, True]);  view_419 = add_124 = getitem_421 = getitem_422 = primals_587 = primals_588 = None
    getitem_477: "f32[8, 784, 768]" = native_layer_norm_backward_5[0]
    getitem_478: "f32[768]" = native_layer_norm_backward_5[1]
    getitem_479: "f32[768]" = native_layer_norm_backward_5[2];  native_layer_norm_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_144: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(slice_53, getitem_477);  slice_53 = getitem_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_113: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_144, primals_95);  primals_95 = None
    mul_114: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_144, permute_97);  permute_97 = None
    sum_22: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_114, [0, 1], True);  mul_114 = None
    view_420: "f32[768]" = torch.ops.aten.view.default(sum_22, [768]);  sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_110: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_113, [0, 2, 1]);  mul_113 = None
    view_421: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_110, [8, 768, 28, 28]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(view_421, getitem_415, primals_585, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_421 = getitem_415 = primals_585 = None
    getitem_480: "f32[8, 768, 28, 28]" = convolution_backward[0]
    getitem_481: "f32[768, 1, 3, 3]" = convolution_backward[1]
    getitem_482: "f32[768]" = convolution_backward[2];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward = torch.ops.aten.native_batch_norm_backward.default(getitem_480, gelu_48, primals_583, getitem_418, getitem_419, getitem_416, getitem_417, True, 1e-05, [True, True, True]);  getitem_480 = gelu_48 = primals_583 = getitem_416 = getitem_417 = None
    getitem_483: "f32[8, 768, 28, 28]" = native_batch_norm_backward[0]
    getitem_484: "f32[768]" = native_batch_norm_backward[1]
    getitem_485: "f32[768]" = native_batch_norm_backward[2];  native_batch_norm_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_3: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_483, convolution_50);  getitem_483 = convolution_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(gelu_backward_3, view_334, primals_581, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_3 = view_334 = primals_581 = None
    getitem_486: "f32[8, 768, 28, 28]" = convolution_backward_1[0]
    getitem_487: "f32[768, 1, 3, 3]" = convolution_backward_1[1]
    getitem_488: "f32[768]" = convolution_backward_1[2];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_422: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_486, [8, 768, 784]);  getitem_486 = None
    permute_111: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_422, [0, 2, 1]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_6 = torch.ops.aten.native_layer_norm_backward.default(permute_111, add_122, [768], getitem_413, getitem_414, primals_579, primals_580, [True, True, True]);  permute_111 = add_122 = getitem_413 = getitem_414 = primals_579 = primals_580 = None
    getitem_489: "f32[8, 784, 768]" = native_layer_norm_backward_6[0]
    getitem_490: "f32[768]" = native_layer_norm_backward_6[1]
    getitem_491: "f32[768]" = native_layer_norm_backward_6[2];  native_layer_norm_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_145: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_144, getitem_489);  add_144 = getitem_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_115: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_145, primals_93);  primals_93 = None
    mul_116: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_145, clone_190);  clone_190 = None
    sum_23: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_116, [0, 1], True);  mul_116 = None
    view_423: "f32[768]" = torch.ops.aten.view.default(sum_23, [768]);  sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_24: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_115, [0, 1], True)
    view_424: "f32[768]" = torch.ops.aten.view.default(sum_24, [768]);  sum_24 = None
    view_425: "f32[6272, 768]" = torch.ops.aten.view.default(mul_115, [6272, 768]);  mul_115 = None
    t_169: "f32[768, 6272]" = torch.ops.aten.t.default(view_425)
    mm_56: "f32[768, 768]" = torch.ops.aten.mm.default(t_169, _unsafe_view_95);  t_169 = _unsafe_view_95 = None
    t_170: "f32[768, 768]" = torch.ops.aten.t.default(mm_56);  mm_56 = None
    t_171: "f32[768, 768]" = torch.ops.aten.t.default(t_93);  t_93 = None
    mm_57: "f32[6272, 768]" = torch.ops.aten.mm.default(view_425, t_171);  view_425 = t_171 = None
    view_426: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_57, [8, 784, 768]);  mm_57 = None
    t_172: "f32[768, 768]" = torch.ops.aten.t.default(t_170);  t_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_427: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_426, [8, 784, 16, 48]);  view_426 = None
    permute_112: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_427, [0, 2, 3, 1]);  view_427 = None
    clone_200: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    _unsafe_view_96: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_200, [128, 48, 784]);  clone_200 = None
    transpose_29: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_330, 1, 2);  view_330 = None
    bmm_48: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_29, _unsafe_view_96);  transpose_29 = None
    transpose_30: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_94, 1, 2);  _unsafe_view_94 = None
    bmm_49: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_96, transpose_30);  _unsafe_view_96 = transpose_30 = None
    view_428: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_48, [8, 16, 48, 784]);  bmm_48 = None
    view_429: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_49, [8, 16, 48, 48]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_76: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_71);  detach_71 = None
    _softmax_backward_data: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_429, detach_76, -1, torch.float32);  view_429 = detach_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_117: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data, view_329);  view_329 = None
    mul_118: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data, primals_94);  _softmax_backward_data = primals_94 = None
    sum_25: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [0, 2, 3], True);  mul_117 = None
    view_430: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_25, [16, 1, 1]);  sum_25 = None
    view_431: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_118, [128, 48, 48]);  mul_118 = None
    transpose_31: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_92, 1, 2);  _unsafe_view_92 = None
    bmm_50: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_31, view_431);  transpose_31 = None
    transpose_32: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_93, 1, 2);  _unsafe_view_93 = None
    bmm_51: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_431, transpose_32);  view_431 = transpose_32 = None
    view_432: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_50, [8, 16, 784, 48]);  bmm_50 = None
    view_433: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_51, [8, 16, 48, 784]);  bmm_51 = None
    transpose_33: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_432, -2, -1);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_54: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_410, expand_139)
    div_55: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_54, expand_139);  div_54 = None
    neg: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_33)
    mul_119: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg, div_55);  neg = div_55 = None
    div_56: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_33, expand_139);  transpose_33 = expand_139 = None
    sum_26: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [3], True);  mul_119 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_47, 1e-12);  linalg_vector_norm_47 = None
    where: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge, sum_26, scalar_tensor);  ge = sum_26 = scalar_tensor = None
    detach_77: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_70);  detach_70 = None
    div_57: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_410, detach_77);  getitem_410 = None
    eq: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_77, 0);  detach_77 = None
    masked_fill: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_57, eq, 0);  div_57 = eq = None
    mul_120: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where, masked_fill);  where = masked_fill = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_146: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_56, mul_120);  div_56 = mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_58: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_409, expand_138)
    div_59: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_58, expand_138);  div_58 = None
    neg_1: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_433)
    mul_121: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_1, div_59);  neg_1 = div_59 = None
    div_60: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_433, expand_138);  view_433 = expand_138 = None
    sum_27: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [3], True);  mul_121 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_1: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_46, 1e-12);  linalg_vector_norm_46 = None
    where_1: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_1, sum_27, scalar_tensor_1);  ge_1 = sum_27 = scalar_tensor_1 = None
    detach_78: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_69);  detach_69 = None
    div_61: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_409, detach_78);  getitem_409 = None
    eq_1: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_78, 0);  detach_78 = None
    masked_fill_1: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_61, eq_1, 0);  div_61 = eq_1 = None
    mul_122: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_1, masked_fill_1);  where_1 = masked_fill_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_147: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_60, mul_122);  div_60 = mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_2: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_147, add_146, view_428]);  add_147 = add_146 = view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_113: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_2, [1, 4, 0, 2, 3]);  stack_2 = None
    clone_201: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    _unsafe_view_97: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_201, [8, 784, 2304]);  clone_201 = None
    view_434: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_97, [6272, 2304]);  _unsafe_view_97 = None
    t_173: "f32[2304, 768]" = torch.ops.aten.t.default(t_92);  t_92 = None
    mm_58: "f32[6272, 768]" = torch.ops.aten.mm.default(view_434, t_173);  t_173 = None
    t_174: "f32[2304, 6272]" = torch.ops.aten.t.default(view_434)
    mm_59: "f32[2304, 768]" = torch.ops.aten.mm.default(t_174, view_326);  t_174 = view_326 = None
    t_175: "f32[768, 2304]" = torch.ops.aten.t.default(mm_59);  mm_59 = None
    sum_28: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[2304]" = torch.ops.aten.view.default(sum_28, [2304]);  sum_28 = None
    t_176: "f32[2304, 768]" = torch.ops.aten.t.default(t_175);  t_175 = None
    view_436: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_58, [8, 784, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_7 = torch.ops.aten.native_layer_norm_backward.default(view_436, add_120, [768], getitem_407, getitem_408, primals_573, primals_574, [True, True, True]);  view_436 = add_120 = getitem_407 = getitem_408 = primals_573 = primals_574 = None
    getitem_492: "f32[8, 784, 768]" = native_layer_norm_backward_7[0]
    getitem_493: "f32[768]" = native_layer_norm_backward_7[1]
    getitem_494: "f32[768]" = native_layer_norm_backward_7[2];  native_layer_norm_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_148: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_145, getitem_492);  add_145 = getitem_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_123: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_148, primals_92);  primals_92 = None
    mul_124: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_148, clone_184);  clone_184 = None
    sum_29: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_124, [0, 1], True);  mul_124 = None
    view_437: "f32[768]" = torch.ops.aten.view.default(sum_29, [768]);  sum_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_438: "f32[6272, 768]" = torch.ops.aten.view.default(mul_123, [6272, 768]);  mul_123 = None
    t_177: "f32[768, 3072]" = torch.ops.aten.t.default(t_91);  t_91 = None
    mm_60: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_438, t_177);  t_177 = None
    t_178: "f32[768, 6272]" = torch.ops.aten.t.default(view_438)
    mm_61: "f32[768, 3072]" = torch.ops.aten.mm.default(t_178, view_324);  t_178 = view_324 = None
    t_179: "f32[3072, 768]" = torch.ops.aten.t.default(mm_61);  mm_61 = None
    sum_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_438, [0], True);  view_438 = None
    view_439: "f32[768]" = torch.ops.aten.view.default(sum_30, [768]);  sum_30 = None
    t_180: "f32[768, 3072]" = torch.ops.aten.t.default(t_179);  t_179 = None
    view_440: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_60, [8, 784, 3072]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_4: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_440, view_323);  view_440 = view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_441: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_4, [6272, 3072]);  gelu_backward_4 = None
    t_181: "f32[3072, 768]" = torch.ops.aten.t.default(t_90);  t_90 = None
    mm_62: "f32[6272, 768]" = torch.ops.aten.mm.default(view_441, t_181);  t_181 = None
    t_182: "f32[3072, 6272]" = torch.ops.aten.t.default(view_441)
    mm_63: "f32[3072, 768]" = torch.ops.aten.mm.default(t_182, view_322);  t_182 = view_322 = None
    t_183: "f32[768, 3072]" = torch.ops.aten.t.default(mm_63);  mm_63 = None
    sum_31: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_441, [0], True);  view_441 = None
    view_442: "f32[3072]" = torch.ops.aten.view.default(sum_31, [3072]);  sum_31 = None
    t_184: "f32[3072, 768]" = torch.ops.aten.t.default(t_183);  t_183 = None
    view_443: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_62, [8, 784, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_8 = torch.ops.aten.native_layer_norm_backward.default(view_443, add_119, [768], getitem_404, getitem_405, primals_567, primals_568, [True, True, True]);  view_443 = add_119 = getitem_404 = getitem_405 = primals_567 = primals_568 = None
    getitem_495: "f32[8, 784, 768]" = native_layer_norm_backward_8[0]
    getitem_496: "f32[768]" = native_layer_norm_backward_8[1]
    getitem_497: "f32[768]" = native_layer_norm_backward_8[2];  native_layer_norm_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_149: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_148, getitem_495);  add_148 = getitem_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_125: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_149, primals_91);  primals_91 = None
    mul_126: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_149, permute_93);  permute_93 = None
    sum_32: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_126, [0, 1], True);  mul_126 = None
    view_444: "f32[768]" = torch.ops.aten.view.default(sum_32, [768]);  sum_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_114: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_125, [0, 2, 1]);  mul_125 = None
    view_445: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_114, [8, 768, 28, 28]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(view_445, getitem_398, primals_565, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_445 = getitem_398 = primals_565 = None
    getitem_498: "f32[8, 768, 28, 28]" = convolution_backward_2[0]
    getitem_499: "f32[768, 1, 3, 3]" = convolution_backward_2[1]
    getitem_500: "f32[768]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_1 = torch.ops.aten.native_batch_norm_backward.default(getitem_498, gelu_46, primals_563, getitem_401, getitem_402, getitem_399, getitem_400, True, 1e-05, [True, True, True]);  getitem_498 = gelu_46 = primals_563 = getitem_399 = getitem_400 = None
    getitem_501: "f32[8, 768, 28, 28]" = native_batch_norm_backward_1[0]
    getitem_502: "f32[768]" = native_batch_norm_backward_1[1]
    getitem_503: "f32[768]" = native_batch_norm_backward_1[2];  native_batch_norm_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_5: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_501, convolution_48);  getitem_501 = convolution_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(gelu_backward_5, view_320, primals_561, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_5 = view_320 = primals_561 = None
    getitem_504: "f32[8, 768, 28, 28]" = convolution_backward_3[0]
    getitem_505: "f32[768, 1, 3, 3]" = convolution_backward_3[1]
    getitem_506: "f32[768]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_446: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_504, [8, 768, 784]);  getitem_504 = None
    permute_115: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_446, [0, 2, 1]);  view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_9 = torch.ops.aten.native_layer_norm_backward.default(permute_115, add_117, [768], getitem_396, getitem_397, primals_559, primals_560, [True, True, True]);  permute_115 = add_117 = getitem_396 = getitem_397 = primals_559 = primals_560 = None
    getitem_507: "f32[8, 784, 768]" = native_layer_norm_backward_9[0]
    getitem_508: "f32[768]" = native_layer_norm_backward_9[1]
    getitem_509: "f32[768]" = native_layer_norm_backward_9[2];  native_layer_norm_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_150: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_149, getitem_507);  add_149 = getitem_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_127: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_150, primals_89);  primals_89 = None
    mul_128: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_150, clone_182);  clone_182 = None
    sum_33: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_128, [0, 1], True);  mul_128 = None
    view_447: "f32[768]" = torch.ops.aten.view.default(sum_33, [768]);  sum_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_34: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_127, [0, 1], True)
    view_448: "f32[768]" = torch.ops.aten.view.default(sum_34, [768]);  sum_34 = None
    view_449: "f32[6272, 768]" = torch.ops.aten.view.default(mul_127, [6272, 768]);  mul_127 = None
    t_185: "f32[768, 6272]" = torch.ops.aten.t.default(view_449)
    mm_64: "f32[768, 768]" = torch.ops.aten.mm.default(t_185, _unsafe_view_91);  t_185 = _unsafe_view_91 = None
    t_186: "f32[768, 768]" = torch.ops.aten.t.default(mm_64);  mm_64 = None
    t_187: "f32[768, 768]" = torch.ops.aten.t.default(t_89);  t_89 = None
    mm_65: "f32[6272, 768]" = torch.ops.aten.mm.default(view_449, t_187);  view_449 = t_187 = None
    view_450: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_65, [8, 784, 768]);  mm_65 = None
    t_188: "f32[768, 768]" = torch.ops.aten.t.default(t_186);  t_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_451: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_450, [8, 784, 16, 48]);  view_450 = None
    permute_116: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_451, [0, 2, 3, 1]);  view_451 = None
    clone_202: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
    _unsafe_view_98: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_202, [128, 48, 784]);  clone_202 = None
    transpose_34: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_316, 1, 2);  view_316 = None
    bmm_52: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_34, _unsafe_view_98);  transpose_34 = None
    transpose_35: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_90, 1, 2);  _unsafe_view_90 = None
    bmm_53: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_98, transpose_35);  _unsafe_view_98 = transpose_35 = None
    view_452: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_52, [8, 16, 48, 784]);  bmm_52 = None
    view_453: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_53, [8, 16, 48, 48]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_79: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_68);  detach_68 = None
    _softmax_backward_data_1: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_453, detach_79, -1, torch.float32);  view_453 = detach_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_129: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_1, view_315);  view_315 = None
    mul_130: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_1, primals_90);  _softmax_backward_data_1 = primals_90 = None
    sum_35: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_129, [0, 2, 3], True);  mul_129 = None
    view_454: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_35, [16, 1, 1]);  sum_35 = None
    view_455: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_130, [128, 48, 48]);  mul_130 = None
    transpose_36: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_88, 1, 2);  _unsafe_view_88 = None
    bmm_54: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_36, view_455);  transpose_36 = None
    transpose_37: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_89, 1, 2);  _unsafe_view_89 = None
    bmm_55: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_455, transpose_37);  view_455 = transpose_37 = None
    view_456: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_54, [8, 16, 784, 48]);  bmm_54 = None
    view_457: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_55, [8, 16, 48, 784]);  bmm_55 = None
    transpose_38: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_456, -2, -1);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_62: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_393, expand_133)
    div_63: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_62, expand_133);  div_62 = None
    neg_2: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_38)
    mul_131: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_2, div_63);  neg_2 = div_63 = None
    div_64: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_38, expand_133);  transpose_38 = expand_133 = None
    sum_36: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [3], True);  mul_131 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_2: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_45, 1e-12);  linalg_vector_norm_45 = None
    where_2: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_2, sum_36, scalar_tensor_2);  ge_2 = sum_36 = scalar_tensor_2 = None
    detach_80: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_67);  detach_67 = None
    div_65: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_393, detach_80);  getitem_393 = None
    eq_2: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_80, 0);  detach_80 = None
    masked_fill_2: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_65, eq_2, 0);  div_65 = eq_2 = None
    mul_132: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_2, masked_fill_2);  where_2 = masked_fill_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_151: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_64, mul_132);  div_64 = mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_66: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_392, expand_132)
    div_67: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_66, expand_132);  div_66 = None
    neg_3: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_457)
    mul_133: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_3, div_67);  neg_3 = div_67 = None
    div_68: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_457, expand_132);  view_457 = expand_132 = None
    sum_37: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_133, [3], True);  mul_133 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_3: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_44, 1e-12);  linalg_vector_norm_44 = None
    where_3: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_3, sum_37, scalar_tensor_3);  ge_3 = sum_37 = scalar_tensor_3 = None
    detach_81: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_66);  detach_66 = None
    div_69: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_392, detach_81);  getitem_392 = None
    eq_3: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_81, 0);  detach_81 = None
    masked_fill_3: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_69, eq_3, 0);  div_69 = eq_3 = None
    mul_134: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_3, masked_fill_3);  where_3 = masked_fill_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_152: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_68, mul_134);  div_68 = mul_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_3: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_152, add_151, view_452]);  add_152 = add_151 = view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_117: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_3, [1, 4, 0, 2, 3]);  stack_3 = None
    clone_203: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    _unsafe_view_99: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_203, [8, 784, 2304]);  clone_203 = None
    view_458: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_99, [6272, 2304]);  _unsafe_view_99 = None
    t_189: "f32[2304, 768]" = torch.ops.aten.t.default(t_88);  t_88 = None
    mm_66: "f32[6272, 768]" = torch.ops.aten.mm.default(view_458, t_189);  t_189 = None
    t_190: "f32[2304, 6272]" = torch.ops.aten.t.default(view_458)
    mm_67: "f32[2304, 768]" = torch.ops.aten.mm.default(t_190, view_312);  t_190 = view_312 = None
    t_191: "f32[768, 2304]" = torch.ops.aten.t.default(mm_67);  mm_67 = None
    sum_38: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_458, [0], True);  view_458 = None
    view_459: "f32[2304]" = torch.ops.aten.view.default(sum_38, [2304]);  sum_38 = None
    t_192: "f32[2304, 768]" = torch.ops.aten.t.default(t_191);  t_191 = None
    view_460: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_66, [8, 784, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_10 = torch.ops.aten.native_layer_norm_backward.default(view_460, add_115, [768], getitem_390, getitem_391, primals_553, primals_554, [True, True, True]);  view_460 = add_115 = getitem_390 = getitem_391 = primals_553 = primals_554 = None
    getitem_510: "f32[8, 784, 768]" = native_layer_norm_backward_10[0]
    getitem_511: "f32[768]" = native_layer_norm_backward_10[1]
    getitem_512: "f32[768]" = native_layer_norm_backward_10[2];  native_layer_norm_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_153: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_150, getitem_510);  add_150 = getitem_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_135: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_153, primals_88);  primals_88 = None
    mul_136: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_153, clone_176);  clone_176 = None
    sum_39: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_136, [0, 1], True);  mul_136 = None
    view_461: "f32[768]" = torch.ops.aten.view.default(sum_39, [768]);  sum_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_462: "f32[6272, 768]" = torch.ops.aten.view.default(mul_135, [6272, 768]);  mul_135 = None
    t_193: "f32[768, 3072]" = torch.ops.aten.t.default(t_87);  t_87 = None
    mm_68: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_462, t_193);  t_193 = None
    t_194: "f32[768, 6272]" = torch.ops.aten.t.default(view_462)
    mm_69: "f32[768, 3072]" = torch.ops.aten.mm.default(t_194, view_310);  t_194 = view_310 = None
    t_195: "f32[3072, 768]" = torch.ops.aten.t.default(mm_69);  mm_69 = None
    sum_40: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
    view_463: "f32[768]" = torch.ops.aten.view.default(sum_40, [768]);  sum_40 = None
    t_196: "f32[768, 3072]" = torch.ops.aten.t.default(t_195);  t_195 = None
    view_464: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_68, [8, 784, 3072]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_6: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_464, view_309);  view_464 = view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_465: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_6, [6272, 3072]);  gelu_backward_6 = None
    t_197: "f32[3072, 768]" = torch.ops.aten.t.default(t_86);  t_86 = None
    mm_70: "f32[6272, 768]" = torch.ops.aten.mm.default(view_465, t_197);  t_197 = None
    t_198: "f32[3072, 6272]" = torch.ops.aten.t.default(view_465)
    mm_71: "f32[3072, 768]" = torch.ops.aten.mm.default(t_198, view_308);  t_198 = view_308 = None
    t_199: "f32[768, 3072]" = torch.ops.aten.t.default(mm_71);  mm_71 = None
    sum_41: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_465, [0], True);  view_465 = None
    view_466: "f32[3072]" = torch.ops.aten.view.default(sum_41, [3072]);  sum_41 = None
    t_200: "f32[3072, 768]" = torch.ops.aten.t.default(t_199);  t_199 = None
    view_467: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_70, [8, 784, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_11 = torch.ops.aten.native_layer_norm_backward.default(view_467, add_114, [768], getitem_387, getitem_388, primals_547, primals_548, [True, True, True]);  view_467 = add_114 = getitem_387 = getitem_388 = primals_547 = primals_548 = None
    getitem_513: "f32[8, 784, 768]" = native_layer_norm_backward_11[0]
    getitem_514: "f32[768]" = native_layer_norm_backward_11[1]
    getitem_515: "f32[768]" = native_layer_norm_backward_11[2];  native_layer_norm_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_154: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_153, getitem_513);  add_153 = getitem_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_137: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_154, primals_87);  primals_87 = None
    mul_138: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_154, permute_89);  permute_89 = None
    sum_42: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_138, [0, 1], True);  mul_138 = None
    view_468: "f32[768]" = torch.ops.aten.view.default(sum_42, [768]);  sum_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_118: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_137, [0, 2, 1]);  mul_137 = None
    view_469: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_118, [8, 768, 28, 28]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(view_469, getitem_381, primals_545, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_469 = getitem_381 = primals_545 = None
    getitem_516: "f32[8, 768, 28, 28]" = convolution_backward_4[0]
    getitem_517: "f32[768, 1, 3, 3]" = convolution_backward_4[1]
    getitem_518: "f32[768]" = convolution_backward_4[2];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_2 = torch.ops.aten.native_batch_norm_backward.default(getitem_516, gelu_44, primals_543, getitem_384, getitem_385, getitem_382, getitem_383, True, 1e-05, [True, True, True]);  getitem_516 = gelu_44 = primals_543 = getitem_382 = getitem_383 = None
    getitem_519: "f32[8, 768, 28, 28]" = native_batch_norm_backward_2[0]
    getitem_520: "f32[768]" = native_batch_norm_backward_2[1]
    getitem_521: "f32[768]" = native_batch_norm_backward_2[2];  native_batch_norm_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_7: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_519, convolution_46);  getitem_519 = convolution_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(gelu_backward_7, view_306, primals_541, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_7 = view_306 = primals_541 = None
    getitem_522: "f32[8, 768, 28, 28]" = convolution_backward_5[0]
    getitem_523: "f32[768, 1, 3, 3]" = convolution_backward_5[1]
    getitem_524: "f32[768]" = convolution_backward_5[2];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_470: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_522, [8, 768, 784]);  getitem_522 = None
    permute_119: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_470, [0, 2, 1]);  view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_12 = torch.ops.aten.native_layer_norm_backward.default(permute_119, add_112, [768], getitem_379, getitem_380, primals_539, primals_540, [True, True, True]);  permute_119 = add_112 = getitem_379 = getitem_380 = primals_539 = primals_540 = None
    getitem_525: "f32[8, 784, 768]" = native_layer_norm_backward_12[0]
    getitem_526: "f32[768]" = native_layer_norm_backward_12[1]
    getitem_527: "f32[768]" = native_layer_norm_backward_12[2];  native_layer_norm_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_155: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_154, getitem_525);  add_154 = getitem_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_139: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_155, primals_85);  primals_85 = None
    mul_140: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_155, clone_174);  clone_174 = None
    sum_43: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_140, [0, 1], True);  mul_140 = None
    view_471: "f32[768]" = torch.ops.aten.view.default(sum_43, [768]);  sum_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_44: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_139, [0, 1], True)
    view_472: "f32[768]" = torch.ops.aten.view.default(sum_44, [768]);  sum_44 = None
    view_473: "f32[6272, 768]" = torch.ops.aten.view.default(mul_139, [6272, 768]);  mul_139 = None
    t_201: "f32[768, 6272]" = torch.ops.aten.t.default(view_473)
    mm_72: "f32[768, 768]" = torch.ops.aten.mm.default(t_201, _unsafe_view_87);  t_201 = _unsafe_view_87 = None
    t_202: "f32[768, 768]" = torch.ops.aten.t.default(mm_72);  mm_72 = None
    t_203: "f32[768, 768]" = torch.ops.aten.t.default(t_85);  t_85 = None
    mm_73: "f32[6272, 768]" = torch.ops.aten.mm.default(view_473, t_203);  view_473 = t_203 = None
    view_474: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_73, [8, 784, 768]);  mm_73 = None
    t_204: "f32[768, 768]" = torch.ops.aten.t.default(t_202);  t_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_475: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_474, [8, 784, 16, 48]);  view_474 = None
    permute_120: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_475, [0, 2, 3, 1]);  view_475 = None
    clone_204: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_120, memory_format = torch.contiguous_format);  permute_120 = None
    _unsafe_view_100: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_204, [128, 48, 784]);  clone_204 = None
    transpose_39: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_302, 1, 2);  view_302 = None
    bmm_56: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_39, _unsafe_view_100);  transpose_39 = None
    transpose_40: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_86, 1, 2);  _unsafe_view_86 = None
    bmm_57: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_100, transpose_40);  _unsafe_view_100 = transpose_40 = None
    view_476: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_56, [8, 16, 48, 784]);  bmm_56 = None
    view_477: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_57, [8, 16, 48, 48]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_82: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_65);  detach_65 = None
    _softmax_backward_data_2: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_477, detach_82, -1, torch.float32);  view_477 = detach_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_141: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_2, view_301);  view_301 = None
    mul_142: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_2, primals_86);  _softmax_backward_data_2 = primals_86 = None
    sum_45: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_141, [0, 2, 3], True);  mul_141 = None
    view_478: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_45, [16, 1, 1]);  sum_45 = None
    view_479: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_142, [128, 48, 48]);  mul_142 = None
    transpose_41: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_84, 1, 2);  _unsafe_view_84 = None
    bmm_58: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_41, view_479);  transpose_41 = None
    transpose_42: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_85, 1, 2);  _unsafe_view_85 = None
    bmm_59: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_479, transpose_42);  view_479 = transpose_42 = None
    view_480: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_58, [8, 16, 784, 48]);  bmm_58 = None
    view_481: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_59, [8, 16, 48, 784]);  bmm_59 = None
    transpose_43: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_480, -2, -1);  view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_70: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_376, expand_127)
    div_71: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_70, expand_127);  div_70 = None
    neg_4: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_43)
    mul_143: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_4, div_71);  neg_4 = div_71 = None
    div_72: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_43, expand_127);  transpose_43 = expand_127 = None
    sum_46: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_143, [3], True);  mul_143 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_4: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_43, 1e-12);  linalg_vector_norm_43 = None
    where_4: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_4, sum_46, scalar_tensor_4);  ge_4 = sum_46 = scalar_tensor_4 = None
    detach_83: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_64);  detach_64 = None
    div_73: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_376, detach_83);  getitem_376 = None
    eq_4: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_83, 0);  detach_83 = None
    masked_fill_4: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_73, eq_4, 0);  div_73 = eq_4 = None
    mul_144: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_4, masked_fill_4);  where_4 = masked_fill_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_156: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_72, mul_144);  div_72 = mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_74: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_375, expand_126)
    div_75: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_74, expand_126);  div_74 = None
    neg_5: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_481)
    mul_145: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_5, div_75);  neg_5 = div_75 = None
    div_76: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_481, expand_126);  view_481 = expand_126 = None
    sum_47: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [3], True);  mul_145 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_5: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_42, 1e-12);  linalg_vector_norm_42 = None
    where_5: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_5, sum_47, scalar_tensor_5);  ge_5 = sum_47 = scalar_tensor_5 = None
    detach_84: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_63);  detach_63 = None
    div_77: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_375, detach_84);  getitem_375 = None
    eq_5: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_84, 0);  detach_84 = None
    masked_fill_5: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_77, eq_5, 0);  div_77 = eq_5 = None
    mul_146: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_5, masked_fill_5);  where_5 = masked_fill_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_157: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_76, mul_146);  div_76 = mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_4: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_157, add_156, view_476]);  add_157 = add_156 = view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_121: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_4, [1, 4, 0, 2, 3]);  stack_4 = None
    clone_205: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    _unsafe_view_101: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_205, [8, 784, 2304]);  clone_205 = None
    view_482: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_101, [6272, 2304]);  _unsafe_view_101 = None
    t_205: "f32[2304, 768]" = torch.ops.aten.t.default(t_84);  t_84 = None
    mm_74: "f32[6272, 768]" = torch.ops.aten.mm.default(view_482, t_205);  t_205 = None
    t_206: "f32[2304, 6272]" = torch.ops.aten.t.default(view_482)
    mm_75: "f32[2304, 768]" = torch.ops.aten.mm.default(t_206, view_298);  t_206 = view_298 = None
    t_207: "f32[768, 2304]" = torch.ops.aten.t.default(mm_75);  mm_75 = None
    sum_48: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_482, [0], True);  view_482 = None
    view_483: "f32[2304]" = torch.ops.aten.view.default(sum_48, [2304]);  sum_48 = None
    t_208: "f32[2304, 768]" = torch.ops.aten.t.default(t_207);  t_207 = None
    view_484: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_74, [8, 784, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_13 = torch.ops.aten.native_layer_norm_backward.default(view_484, add_110, [768], getitem_373, getitem_374, primals_533, primals_534, [True, True, True]);  view_484 = add_110 = getitem_373 = getitem_374 = primals_533 = primals_534 = None
    getitem_528: "f32[8, 784, 768]" = native_layer_norm_backward_13[0]
    getitem_529: "f32[768]" = native_layer_norm_backward_13[1]
    getitem_530: "f32[768]" = native_layer_norm_backward_13[2];  native_layer_norm_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_158: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_155, getitem_528);  add_155 = getitem_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_147: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_158, primals_84);  primals_84 = None
    mul_148: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_158, clone_168);  clone_168 = None
    sum_49: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 1], True);  mul_148 = None
    view_485: "f32[768]" = torch.ops.aten.view.default(sum_49, [768]);  sum_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_486: "f32[6272, 768]" = torch.ops.aten.view.default(mul_147, [6272, 768]);  mul_147 = None
    t_209: "f32[768, 3072]" = torch.ops.aten.t.default(t_83);  t_83 = None
    mm_76: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_486, t_209);  t_209 = None
    t_210: "f32[768, 6272]" = torch.ops.aten.t.default(view_486)
    mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(t_210, view_296);  t_210 = view_296 = None
    t_211: "f32[3072, 768]" = torch.ops.aten.t.default(mm_77);  mm_77 = None
    sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_486, [0], True);  view_486 = None
    view_487: "f32[768]" = torch.ops.aten.view.default(sum_50, [768]);  sum_50 = None
    t_212: "f32[768, 3072]" = torch.ops.aten.t.default(t_211);  t_211 = None
    view_488: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_76, [8, 784, 3072]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_8: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_488, view_295);  view_488 = view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_489: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_8, [6272, 3072]);  gelu_backward_8 = None
    t_213: "f32[3072, 768]" = torch.ops.aten.t.default(t_82);  t_82 = None
    mm_78: "f32[6272, 768]" = torch.ops.aten.mm.default(view_489, t_213);  t_213 = None
    t_214: "f32[3072, 6272]" = torch.ops.aten.t.default(view_489)
    mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(t_214, view_294);  t_214 = view_294 = None
    t_215: "f32[768, 3072]" = torch.ops.aten.t.default(mm_79);  mm_79 = None
    sum_51: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_489, [0], True);  view_489 = None
    view_490: "f32[3072]" = torch.ops.aten.view.default(sum_51, [3072]);  sum_51 = None
    t_216: "f32[3072, 768]" = torch.ops.aten.t.default(t_215);  t_215 = None
    view_491: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_78, [8, 784, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_14 = torch.ops.aten.native_layer_norm_backward.default(view_491, add_109, [768], getitem_370, getitem_371, primals_527, primals_528, [True, True, True]);  view_491 = add_109 = getitem_370 = getitem_371 = primals_527 = primals_528 = None
    getitem_531: "f32[8, 784, 768]" = native_layer_norm_backward_14[0]
    getitem_532: "f32[768]" = native_layer_norm_backward_14[1]
    getitem_533: "f32[768]" = native_layer_norm_backward_14[2];  native_layer_norm_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_159: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_158, getitem_531);  add_158 = getitem_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_149: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_159, primals_83);  primals_83 = None
    mul_150: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_159, permute_85);  permute_85 = None
    sum_52: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_150, [0, 1], True);  mul_150 = None
    view_492: "f32[768]" = torch.ops.aten.view.default(sum_52, [768]);  sum_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_122: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_149, [0, 2, 1]);  mul_149 = None
    view_493: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_122, [8, 768, 28, 28]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(view_493, getitem_364, primals_525, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_493 = getitem_364 = primals_525 = None
    getitem_534: "f32[8, 768, 28, 28]" = convolution_backward_6[0]
    getitem_535: "f32[768, 1, 3, 3]" = convolution_backward_6[1]
    getitem_536: "f32[768]" = convolution_backward_6[2];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_3 = torch.ops.aten.native_batch_norm_backward.default(getitem_534, gelu_42, primals_523, getitem_367, getitem_368, getitem_365, getitem_366, True, 1e-05, [True, True, True]);  getitem_534 = gelu_42 = primals_523 = getitem_365 = getitem_366 = None
    getitem_537: "f32[8, 768, 28, 28]" = native_batch_norm_backward_3[0]
    getitem_538: "f32[768]" = native_batch_norm_backward_3[1]
    getitem_539: "f32[768]" = native_batch_norm_backward_3[2];  native_batch_norm_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_9: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_537, convolution_44);  getitem_537 = convolution_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(gelu_backward_9, view_292, primals_521, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_9 = view_292 = primals_521 = None
    getitem_540: "f32[8, 768, 28, 28]" = convolution_backward_7[0]
    getitem_541: "f32[768, 1, 3, 3]" = convolution_backward_7[1]
    getitem_542: "f32[768]" = convolution_backward_7[2];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_494: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_540, [8, 768, 784]);  getitem_540 = None
    permute_123: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_15 = torch.ops.aten.native_layer_norm_backward.default(permute_123, add_107, [768], getitem_362, getitem_363, primals_519, primals_520, [True, True, True]);  permute_123 = add_107 = getitem_362 = getitem_363 = primals_519 = primals_520 = None
    getitem_543: "f32[8, 784, 768]" = native_layer_norm_backward_15[0]
    getitem_544: "f32[768]" = native_layer_norm_backward_15[1]
    getitem_545: "f32[768]" = native_layer_norm_backward_15[2];  native_layer_norm_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_160: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_159, getitem_543);  add_159 = getitem_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_151: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_160, primals_81);  primals_81 = None
    mul_152: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_160, clone_166);  clone_166 = None
    sum_53: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_152, [0, 1], True);  mul_152 = None
    view_495: "f32[768]" = torch.ops.aten.view.default(sum_53, [768]);  sum_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_54: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_151, [0, 1], True)
    view_496: "f32[768]" = torch.ops.aten.view.default(sum_54, [768]);  sum_54 = None
    view_497: "f32[6272, 768]" = torch.ops.aten.view.default(mul_151, [6272, 768]);  mul_151 = None
    t_217: "f32[768, 6272]" = torch.ops.aten.t.default(view_497)
    mm_80: "f32[768, 768]" = torch.ops.aten.mm.default(t_217, _unsafe_view_83);  t_217 = _unsafe_view_83 = None
    t_218: "f32[768, 768]" = torch.ops.aten.t.default(mm_80);  mm_80 = None
    t_219: "f32[768, 768]" = torch.ops.aten.t.default(t_81);  t_81 = None
    mm_81: "f32[6272, 768]" = torch.ops.aten.mm.default(view_497, t_219);  view_497 = t_219 = None
    view_498: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_81, [8, 784, 768]);  mm_81 = None
    t_220: "f32[768, 768]" = torch.ops.aten.t.default(t_218);  t_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_499: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_498, [8, 784, 16, 48]);  view_498 = None
    permute_124: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_499, [0, 2, 3, 1]);  view_499 = None
    clone_206: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    _unsafe_view_102: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_206, [128, 48, 784]);  clone_206 = None
    transpose_44: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_288, 1, 2);  view_288 = None
    bmm_60: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_44, _unsafe_view_102);  transpose_44 = None
    transpose_45: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_82, 1, 2);  _unsafe_view_82 = None
    bmm_61: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_102, transpose_45);  _unsafe_view_102 = transpose_45 = None
    view_500: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_60, [8, 16, 48, 784]);  bmm_60 = None
    view_501: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_61, [8, 16, 48, 48]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_85: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_62);  detach_62 = None
    _softmax_backward_data_3: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_501, detach_85, -1, torch.float32);  view_501 = detach_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_153: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_3, view_287);  view_287 = None
    mul_154: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_3, primals_82);  _softmax_backward_data_3 = primals_82 = None
    sum_55: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 2, 3], True);  mul_153 = None
    view_502: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_55, [16, 1, 1]);  sum_55 = None
    view_503: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_154, [128, 48, 48]);  mul_154 = None
    transpose_46: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_80, 1, 2);  _unsafe_view_80 = None
    bmm_62: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_46, view_503);  transpose_46 = None
    transpose_47: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_81, 1, 2);  _unsafe_view_81 = None
    bmm_63: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_503, transpose_47);  view_503 = transpose_47 = None
    view_504: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_62, [8, 16, 784, 48]);  bmm_62 = None
    view_505: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_63, [8, 16, 48, 784]);  bmm_63 = None
    transpose_48: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_504, -2, -1);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_78: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_359, expand_121)
    div_79: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_78, expand_121);  div_78 = None
    neg_6: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_48)
    mul_155: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_6, div_79);  neg_6 = div_79 = None
    div_80: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_48, expand_121);  transpose_48 = expand_121 = None
    sum_56: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [3], True);  mul_155 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_6: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_41, 1e-12);  linalg_vector_norm_41 = None
    where_6: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_6, sum_56, scalar_tensor_6);  ge_6 = sum_56 = scalar_tensor_6 = None
    detach_86: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_61);  detach_61 = None
    div_81: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_359, detach_86);  getitem_359 = None
    eq_6: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_86, 0);  detach_86 = None
    masked_fill_6: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_81, eq_6, 0);  div_81 = eq_6 = None
    mul_156: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_6, masked_fill_6);  where_6 = masked_fill_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_161: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_80, mul_156);  div_80 = mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_82: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_358, expand_120)
    div_83: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_82, expand_120);  div_82 = None
    neg_7: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_505)
    mul_157: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_7, div_83);  neg_7 = div_83 = None
    div_84: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_505, expand_120);  view_505 = expand_120 = None
    sum_57: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [3], True);  mul_157 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_7: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_40, 1e-12);  linalg_vector_norm_40 = None
    where_7: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_7, sum_57, scalar_tensor_7);  ge_7 = sum_57 = scalar_tensor_7 = None
    detach_87: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_60);  detach_60 = None
    div_85: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_358, detach_87);  getitem_358 = None
    eq_7: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_87, 0);  detach_87 = None
    masked_fill_7: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_85, eq_7, 0);  div_85 = eq_7 = None
    mul_158: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_7, masked_fill_7);  where_7 = masked_fill_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_162: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_84, mul_158);  div_84 = mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_5: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_162, add_161, view_500]);  add_162 = add_161 = view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_125: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_5, [1, 4, 0, 2, 3]);  stack_5 = None
    clone_207: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    _unsafe_view_103: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_207, [8, 784, 2304]);  clone_207 = None
    view_506: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_103, [6272, 2304]);  _unsafe_view_103 = None
    t_221: "f32[2304, 768]" = torch.ops.aten.t.default(t_80);  t_80 = None
    mm_82: "f32[6272, 768]" = torch.ops.aten.mm.default(view_506, t_221);  t_221 = None
    t_222: "f32[2304, 6272]" = torch.ops.aten.t.default(view_506)
    mm_83: "f32[2304, 768]" = torch.ops.aten.mm.default(t_222, view_284);  t_222 = view_284 = None
    t_223: "f32[768, 2304]" = torch.ops.aten.t.default(mm_83);  mm_83 = None
    sum_58: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
    view_507: "f32[2304]" = torch.ops.aten.view.default(sum_58, [2304]);  sum_58 = None
    t_224: "f32[2304, 768]" = torch.ops.aten.t.default(t_223);  t_223 = None
    view_508: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_82, [8, 784, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_16 = torch.ops.aten.native_layer_norm_backward.default(view_508, add_105, [768], getitem_356, getitem_357, primals_513, primals_514, [True, True, True]);  view_508 = add_105 = getitem_356 = getitem_357 = primals_513 = primals_514 = None
    getitem_546: "f32[8, 784, 768]" = native_layer_norm_backward_16[0]
    getitem_547: "f32[768]" = native_layer_norm_backward_16[1]
    getitem_548: "f32[768]" = native_layer_norm_backward_16[2];  native_layer_norm_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_163: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_160, getitem_546);  add_160 = getitem_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_159: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_163, primals_80);  primals_80 = None
    mul_160: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_163, clone_160);  clone_160 = None
    sum_59: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_160, [0, 1], True);  mul_160 = None
    view_509: "f32[768]" = torch.ops.aten.view.default(sum_59, [768]);  sum_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_510: "f32[6272, 768]" = torch.ops.aten.view.default(mul_159, [6272, 768]);  mul_159 = None
    t_225: "f32[768, 3072]" = torch.ops.aten.t.default(t_79);  t_79 = None
    mm_84: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_510, t_225);  t_225 = None
    t_226: "f32[768, 6272]" = torch.ops.aten.t.default(view_510)
    mm_85: "f32[768, 3072]" = torch.ops.aten.mm.default(t_226, view_282);  t_226 = view_282 = None
    t_227: "f32[3072, 768]" = torch.ops.aten.t.default(mm_85);  mm_85 = None
    sum_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_510, [0], True);  view_510 = None
    view_511: "f32[768]" = torch.ops.aten.view.default(sum_60, [768]);  sum_60 = None
    t_228: "f32[768, 3072]" = torch.ops.aten.t.default(t_227);  t_227 = None
    view_512: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_84, [8, 784, 3072]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_10: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_512, view_281);  view_512 = view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_513: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_10, [6272, 3072]);  gelu_backward_10 = None
    t_229: "f32[3072, 768]" = torch.ops.aten.t.default(t_78);  t_78 = None
    mm_86: "f32[6272, 768]" = torch.ops.aten.mm.default(view_513, t_229);  t_229 = None
    t_230: "f32[3072, 6272]" = torch.ops.aten.t.default(view_513)
    mm_87: "f32[3072, 768]" = torch.ops.aten.mm.default(t_230, view_280);  t_230 = view_280 = None
    t_231: "f32[768, 3072]" = torch.ops.aten.t.default(mm_87);  mm_87 = None
    sum_61: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_513, [0], True);  view_513 = None
    view_514: "f32[3072]" = torch.ops.aten.view.default(sum_61, [3072]);  sum_61 = None
    t_232: "f32[3072, 768]" = torch.ops.aten.t.default(t_231);  t_231 = None
    view_515: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_86, [8, 784, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_17 = torch.ops.aten.native_layer_norm_backward.default(view_515, add_104, [768], getitem_353, getitem_354, primals_507, primals_508, [True, True, True]);  view_515 = add_104 = getitem_353 = getitem_354 = primals_507 = primals_508 = None
    getitem_549: "f32[8, 784, 768]" = native_layer_norm_backward_17[0]
    getitem_550: "f32[768]" = native_layer_norm_backward_17[1]
    getitem_551: "f32[768]" = native_layer_norm_backward_17[2];  native_layer_norm_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_164: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_163, getitem_549);  add_163 = getitem_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_161: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_164, primals_79);  primals_79 = None
    mul_162: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_164, permute_81);  permute_81 = None
    sum_62: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_162, [0, 1], True);  mul_162 = None
    view_516: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_126: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_161, [0, 2, 1]);  mul_161 = None
    view_517: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_126, [8, 768, 28, 28]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(view_517, getitem_347, primals_505, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_517 = getitem_347 = primals_505 = None
    getitem_552: "f32[8, 768, 28, 28]" = convolution_backward_8[0]
    getitem_553: "f32[768, 1, 3, 3]" = convolution_backward_8[1]
    getitem_554: "f32[768]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_4 = torch.ops.aten.native_batch_norm_backward.default(getitem_552, gelu_40, primals_503, getitem_350, getitem_351, getitem_348, getitem_349, True, 1e-05, [True, True, True]);  getitem_552 = gelu_40 = primals_503 = getitem_348 = getitem_349 = None
    getitem_555: "f32[8, 768, 28, 28]" = native_batch_norm_backward_4[0]
    getitem_556: "f32[768]" = native_batch_norm_backward_4[1]
    getitem_557: "f32[768]" = native_batch_norm_backward_4[2];  native_batch_norm_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_11: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_555, convolution_42);  getitem_555 = convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(gelu_backward_11, view_278, primals_501, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_11 = view_278 = primals_501 = None
    getitem_558: "f32[8, 768, 28, 28]" = convolution_backward_9[0]
    getitem_559: "f32[768, 1, 3, 3]" = convolution_backward_9[1]
    getitem_560: "f32[768]" = convolution_backward_9[2];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_518: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_558, [8, 768, 784]);  getitem_558 = None
    permute_127: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_518, [0, 2, 1]);  view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_18 = torch.ops.aten.native_layer_norm_backward.default(permute_127, add_102, [768], getitem_345, getitem_346, primals_499, primals_500, [True, True, True]);  permute_127 = add_102 = getitem_345 = getitem_346 = primals_499 = primals_500 = None
    getitem_561: "f32[8, 784, 768]" = native_layer_norm_backward_18[0]
    getitem_562: "f32[768]" = native_layer_norm_backward_18[1]
    getitem_563: "f32[768]" = native_layer_norm_backward_18[2];  native_layer_norm_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_165: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_164, getitem_561);  add_164 = getitem_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_163: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_165, primals_77);  primals_77 = None
    mul_164: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_165, clone_158);  clone_158 = None
    sum_63: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_164, [0, 1], True);  mul_164 = None
    view_519: "f32[768]" = torch.ops.aten.view.default(sum_63, [768]);  sum_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_64: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_163, [0, 1], True)
    view_520: "f32[768]" = torch.ops.aten.view.default(sum_64, [768]);  sum_64 = None
    view_521: "f32[6272, 768]" = torch.ops.aten.view.default(mul_163, [6272, 768]);  mul_163 = None
    t_233: "f32[768, 6272]" = torch.ops.aten.t.default(view_521)
    mm_88: "f32[768, 768]" = torch.ops.aten.mm.default(t_233, _unsafe_view_79);  t_233 = _unsafe_view_79 = None
    t_234: "f32[768, 768]" = torch.ops.aten.t.default(mm_88);  mm_88 = None
    t_235: "f32[768, 768]" = torch.ops.aten.t.default(t_77);  t_77 = None
    mm_89: "f32[6272, 768]" = torch.ops.aten.mm.default(view_521, t_235);  view_521 = t_235 = None
    view_522: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_89, [8, 784, 768]);  mm_89 = None
    t_236: "f32[768, 768]" = torch.ops.aten.t.default(t_234);  t_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_523: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_522, [8, 784, 16, 48]);  view_522 = None
    permute_128: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_523, [0, 2, 3, 1]);  view_523 = None
    clone_208: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    _unsafe_view_104: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_208, [128, 48, 784]);  clone_208 = None
    transpose_49: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_274, 1, 2);  view_274 = None
    bmm_64: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_49, _unsafe_view_104);  transpose_49 = None
    transpose_50: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_78, 1, 2);  _unsafe_view_78 = None
    bmm_65: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_104, transpose_50);  _unsafe_view_104 = transpose_50 = None
    view_524: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_64, [8, 16, 48, 784]);  bmm_64 = None
    view_525: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_65, [8, 16, 48, 48]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_88: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_59);  detach_59 = None
    _softmax_backward_data_4: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_525, detach_88, -1, torch.float32);  view_525 = detach_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_165: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_4, view_273);  view_273 = None
    mul_166: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_4, primals_78);  _softmax_backward_data_4 = primals_78 = None
    sum_65: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [0, 2, 3], True);  mul_165 = None
    view_526: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_65, [16, 1, 1]);  sum_65 = None
    view_527: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_166, [128, 48, 48]);  mul_166 = None
    transpose_51: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_76, 1, 2);  _unsafe_view_76 = None
    bmm_66: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_51, view_527);  transpose_51 = None
    transpose_52: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_77, 1, 2);  _unsafe_view_77 = None
    bmm_67: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_527, transpose_52);  view_527 = transpose_52 = None
    view_528: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_66, [8, 16, 784, 48]);  bmm_66 = None
    view_529: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_67, [8, 16, 48, 784]);  bmm_67 = None
    transpose_53: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_528, -2, -1);  view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_86: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_342, expand_115)
    div_87: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_86, expand_115);  div_86 = None
    neg_8: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_53)
    mul_167: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_8, div_87);  neg_8 = div_87 = None
    div_88: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_53, expand_115);  transpose_53 = expand_115 = None
    sum_66: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [3], True);  mul_167 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_8: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_39, 1e-12);  linalg_vector_norm_39 = None
    where_8: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_8, sum_66, scalar_tensor_8);  ge_8 = sum_66 = scalar_tensor_8 = None
    detach_89: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_58);  detach_58 = None
    div_89: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_342, detach_89);  getitem_342 = None
    eq_8: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_89, 0);  detach_89 = None
    masked_fill_8: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_89, eq_8, 0);  div_89 = eq_8 = None
    mul_168: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_8, masked_fill_8);  where_8 = masked_fill_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_166: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_88, mul_168);  div_88 = mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_90: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_341, expand_114)
    div_91: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_90, expand_114);  div_90 = None
    neg_9: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_529)
    mul_169: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_9, div_91);  neg_9 = div_91 = None
    div_92: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_529, expand_114);  view_529 = expand_114 = None
    sum_67: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [3], True);  mul_169 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_9: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_38, 1e-12);  linalg_vector_norm_38 = None
    where_9: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_9, sum_67, scalar_tensor_9);  ge_9 = sum_67 = scalar_tensor_9 = None
    detach_90: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_57);  detach_57 = None
    div_93: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_341, detach_90);  getitem_341 = None
    eq_9: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_90, 0);  detach_90 = None
    masked_fill_9: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_93, eq_9, 0);  div_93 = eq_9 = None
    mul_170: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_9, masked_fill_9);  where_9 = masked_fill_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_167: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_92, mul_170);  div_92 = mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_6: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_167, add_166, view_524]);  add_167 = add_166 = view_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_129: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_6, [1, 4, 0, 2, 3]);  stack_6 = None
    clone_209: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
    _unsafe_view_105: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_209, [8, 784, 2304]);  clone_209 = None
    view_530: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_105, [6272, 2304]);  _unsafe_view_105 = None
    t_237: "f32[2304, 768]" = torch.ops.aten.t.default(t_76);  t_76 = None
    mm_90: "f32[6272, 768]" = torch.ops.aten.mm.default(view_530, t_237);  t_237 = None
    t_238: "f32[2304, 6272]" = torch.ops.aten.t.default(view_530)
    mm_91: "f32[2304, 768]" = torch.ops.aten.mm.default(t_238, view_270);  t_238 = view_270 = None
    t_239: "f32[768, 2304]" = torch.ops.aten.t.default(mm_91);  mm_91 = None
    sum_68: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_530, [0], True);  view_530 = None
    view_531: "f32[2304]" = torch.ops.aten.view.default(sum_68, [2304]);  sum_68 = None
    t_240: "f32[2304, 768]" = torch.ops.aten.t.default(t_239);  t_239 = None
    view_532: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_90, [8, 784, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_19 = torch.ops.aten.native_layer_norm_backward.default(view_532, add_100, [768], getitem_339, getitem_340, primals_493, primals_494, [True, True, True]);  view_532 = add_100 = getitem_339 = getitem_340 = primals_493 = primals_494 = None
    getitem_564: "f32[8, 784, 768]" = native_layer_norm_backward_19[0]
    getitem_565: "f32[768]" = native_layer_norm_backward_19[1]
    getitem_566: "f32[768]" = native_layer_norm_backward_19[2];  native_layer_norm_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_168: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_165, getitem_564);  add_165 = getitem_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_171: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_168, primals_76);  primals_76 = None
    mul_172: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_168, clone_152);  clone_152 = None
    sum_69: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1], True);  mul_172 = None
    view_533: "f32[768]" = torch.ops.aten.view.default(sum_69, [768]);  sum_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_534: "f32[6272, 768]" = torch.ops.aten.view.default(mul_171, [6272, 768]);  mul_171 = None
    t_241: "f32[768, 3072]" = torch.ops.aten.t.default(t_75);  t_75 = None
    mm_92: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_534, t_241);  t_241 = None
    t_242: "f32[768, 6272]" = torch.ops.aten.t.default(view_534)
    mm_93: "f32[768, 3072]" = torch.ops.aten.mm.default(t_242, view_268);  t_242 = view_268 = None
    t_243: "f32[3072, 768]" = torch.ops.aten.t.default(mm_93);  mm_93 = None
    sum_70: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
    view_535: "f32[768]" = torch.ops.aten.view.default(sum_70, [768]);  sum_70 = None
    t_244: "f32[768, 3072]" = torch.ops.aten.t.default(t_243);  t_243 = None
    view_536: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_92, [8, 784, 3072]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_12: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_536, view_267);  view_536 = view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_537: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_12, [6272, 3072]);  gelu_backward_12 = None
    t_245: "f32[3072, 768]" = torch.ops.aten.t.default(t_74);  t_74 = None
    mm_94: "f32[6272, 768]" = torch.ops.aten.mm.default(view_537, t_245);  t_245 = None
    t_246: "f32[3072, 6272]" = torch.ops.aten.t.default(view_537)
    mm_95: "f32[3072, 768]" = torch.ops.aten.mm.default(t_246, view_266);  t_246 = view_266 = None
    t_247: "f32[768, 3072]" = torch.ops.aten.t.default(mm_95);  mm_95 = None
    sum_71: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
    view_538: "f32[3072]" = torch.ops.aten.view.default(sum_71, [3072]);  sum_71 = None
    t_248: "f32[3072, 768]" = torch.ops.aten.t.default(t_247);  t_247 = None
    view_539: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_94, [8, 784, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_20 = torch.ops.aten.native_layer_norm_backward.default(view_539, add_99, [768], getitem_336, getitem_337, primals_487, primals_488, [True, True, True]);  view_539 = add_99 = getitem_336 = getitem_337 = primals_487 = primals_488 = None
    getitem_567: "f32[8, 784, 768]" = native_layer_norm_backward_20[0]
    getitem_568: "f32[768]" = native_layer_norm_backward_20[1]
    getitem_569: "f32[768]" = native_layer_norm_backward_20[2];  native_layer_norm_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_169: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_168, getitem_567);  add_168 = getitem_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_173: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_169, primals_75);  primals_75 = None
    mul_174: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_169, permute_77);  permute_77 = None
    sum_72: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_174, [0, 1], True);  mul_174 = None
    view_540: "f32[768]" = torch.ops.aten.view.default(sum_72, [768]);  sum_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_130: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_173, [0, 2, 1]);  mul_173 = None
    view_541: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_130, [8, 768, 28, 28]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(view_541, getitem_330, primals_485, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_541 = getitem_330 = primals_485 = None
    getitem_570: "f32[8, 768, 28, 28]" = convolution_backward_10[0]
    getitem_571: "f32[768, 1, 3, 3]" = convolution_backward_10[1]
    getitem_572: "f32[768]" = convolution_backward_10[2];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_5 = torch.ops.aten.native_batch_norm_backward.default(getitem_570, gelu_38, primals_483, getitem_333, getitem_334, getitem_331, getitem_332, True, 1e-05, [True, True, True]);  getitem_570 = gelu_38 = primals_483 = getitem_331 = getitem_332 = None
    getitem_573: "f32[8, 768, 28, 28]" = native_batch_norm_backward_5[0]
    getitem_574: "f32[768]" = native_batch_norm_backward_5[1]
    getitem_575: "f32[768]" = native_batch_norm_backward_5[2];  native_batch_norm_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_13: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_573, convolution_40);  getitem_573 = convolution_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(gelu_backward_13, view_264, primals_481, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_13 = view_264 = primals_481 = None
    getitem_576: "f32[8, 768, 28, 28]" = convolution_backward_11[0]
    getitem_577: "f32[768, 1, 3, 3]" = convolution_backward_11[1]
    getitem_578: "f32[768]" = convolution_backward_11[2];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_542: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_576, [8, 768, 784]);  getitem_576 = None
    permute_131: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_542, [0, 2, 1]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_21 = torch.ops.aten.native_layer_norm_backward.default(permute_131, add_97, [768], getitem_328, getitem_329, primals_479, primals_480, [True, True, True]);  permute_131 = add_97 = getitem_328 = getitem_329 = primals_479 = primals_480 = None
    getitem_579: "f32[8, 784, 768]" = native_layer_norm_backward_21[0]
    getitem_580: "f32[768]" = native_layer_norm_backward_21[1]
    getitem_581: "f32[768]" = native_layer_norm_backward_21[2];  native_layer_norm_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_170: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_169, getitem_579);  add_169 = getitem_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_175: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_170, primals_73);  primals_73 = None
    mul_176: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_170, clone_150);  clone_150 = None
    sum_73: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 1], True);  mul_176 = None
    view_543: "f32[768]" = torch.ops.aten.view.default(sum_73, [768]);  sum_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_74: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 1], True)
    view_544: "f32[768]" = torch.ops.aten.view.default(sum_74, [768]);  sum_74 = None
    view_545: "f32[6272, 768]" = torch.ops.aten.view.default(mul_175, [6272, 768]);  mul_175 = None
    t_249: "f32[768, 6272]" = torch.ops.aten.t.default(view_545)
    mm_96: "f32[768, 768]" = torch.ops.aten.mm.default(t_249, _unsafe_view_75);  t_249 = _unsafe_view_75 = None
    t_250: "f32[768, 768]" = torch.ops.aten.t.default(mm_96);  mm_96 = None
    t_251: "f32[768, 768]" = torch.ops.aten.t.default(t_73);  t_73 = None
    mm_97: "f32[6272, 768]" = torch.ops.aten.mm.default(view_545, t_251);  view_545 = t_251 = None
    view_546: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_97, [8, 784, 768]);  mm_97 = None
    t_252: "f32[768, 768]" = torch.ops.aten.t.default(t_250);  t_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_547: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_546, [8, 784, 16, 48]);  view_546 = None
    permute_132: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_547, [0, 2, 3, 1]);  view_547 = None
    clone_210: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    _unsafe_view_106: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_210, [128, 48, 784]);  clone_210 = None
    transpose_54: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_260, 1, 2);  view_260 = None
    bmm_68: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_54, _unsafe_view_106);  transpose_54 = None
    transpose_55: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_74, 1, 2);  _unsafe_view_74 = None
    bmm_69: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_106, transpose_55);  _unsafe_view_106 = transpose_55 = None
    view_548: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_68, [8, 16, 48, 784]);  bmm_68 = None
    view_549: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_69, [8, 16, 48, 48]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_91: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_56);  detach_56 = None
    _softmax_backward_data_5: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_549, detach_91, -1, torch.float32);  view_549 = detach_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_177: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_5, view_259);  view_259 = None
    mul_178: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_5, primals_74);  _softmax_backward_data_5 = primals_74 = None
    sum_75: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [0, 2, 3], True);  mul_177 = None
    view_550: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_75, [16, 1, 1]);  sum_75 = None
    view_551: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_178, [128, 48, 48]);  mul_178 = None
    transpose_56: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_72, 1, 2);  _unsafe_view_72 = None
    bmm_70: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_56, view_551);  transpose_56 = None
    transpose_57: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_73, 1, 2);  _unsafe_view_73 = None
    bmm_71: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_551, transpose_57);  view_551 = transpose_57 = None
    view_552: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_70, [8, 16, 784, 48]);  bmm_70 = None
    view_553: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_71, [8, 16, 48, 784]);  bmm_71 = None
    transpose_58: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_552, -2, -1);  view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_94: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_325, expand_109)
    div_95: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_94, expand_109);  div_94 = None
    neg_10: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_58)
    mul_179: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_10, div_95);  neg_10 = div_95 = None
    div_96: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_58, expand_109);  transpose_58 = expand_109 = None
    sum_76: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [3], True);  mul_179 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_10: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_37, 1e-12);  linalg_vector_norm_37 = None
    where_10: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_10, sum_76, scalar_tensor_10);  ge_10 = sum_76 = scalar_tensor_10 = None
    detach_92: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_55);  detach_55 = None
    div_97: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_325, detach_92);  getitem_325 = None
    eq_10: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_92, 0);  detach_92 = None
    masked_fill_10: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_97, eq_10, 0);  div_97 = eq_10 = None
    mul_180: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_10, masked_fill_10);  where_10 = masked_fill_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_171: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_96, mul_180);  div_96 = mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_98: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_324, expand_108)
    div_99: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_98, expand_108);  div_98 = None
    neg_11: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_553)
    mul_181: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_11, div_99);  neg_11 = div_99 = None
    div_100: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_553, expand_108);  view_553 = expand_108 = None
    sum_77: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [3], True);  mul_181 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_11: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_36, 1e-12);  linalg_vector_norm_36 = None
    where_11: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_11, sum_77, scalar_tensor_11);  ge_11 = sum_77 = scalar_tensor_11 = None
    detach_93: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_54);  detach_54 = None
    div_101: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_324, detach_93);  getitem_324 = None
    eq_11: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_93, 0);  detach_93 = None
    masked_fill_11: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_101, eq_11, 0);  div_101 = eq_11 = None
    mul_182: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_11, masked_fill_11);  where_11 = masked_fill_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_172: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_100, mul_182);  div_100 = mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_7: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_172, add_171, view_548]);  add_172 = add_171 = view_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_133: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_7, [1, 4, 0, 2, 3]);  stack_7 = None
    clone_211: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    _unsafe_view_107: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_211, [8, 784, 2304]);  clone_211 = None
    view_554: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_107, [6272, 2304]);  _unsafe_view_107 = None
    t_253: "f32[2304, 768]" = torch.ops.aten.t.default(t_72);  t_72 = None
    mm_98: "f32[6272, 768]" = torch.ops.aten.mm.default(view_554, t_253);  t_253 = None
    t_254: "f32[2304, 6272]" = torch.ops.aten.t.default(view_554)
    mm_99: "f32[2304, 768]" = torch.ops.aten.mm.default(t_254, view_256);  t_254 = view_256 = None
    t_255: "f32[768, 2304]" = torch.ops.aten.t.default(mm_99);  mm_99 = None
    sum_78: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
    view_555: "f32[2304]" = torch.ops.aten.view.default(sum_78, [2304]);  sum_78 = None
    t_256: "f32[2304, 768]" = torch.ops.aten.t.default(t_255);  t_255 = None
    view_556: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_98, [8, 784, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_22 = torch.ops.aten.native_layer_norm_backward.default(view_556, add_95, [768], getitem_322, getitem_323, primals_473, primals_474, [True, True, True]);  view_556 = add_95 = getitem_322 = getitem_323 = primals_473 = primals_474 = None
    getitem_582: "f32[8, 784, 768]" = native_layer_norm_backward_22[0]
    getitem_583: "f32[768]" = native_layer_norm_backward_22[1]
    getitem_584: "f32[768]" = native_layer_norm_backward_22[2];  native_layer_norm_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_173: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_170, getitem_582);  add_170 = getitem_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_183: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_173, primals_72);  primals_72 = None
    mul_184: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_173, clone_144);  clone_144 = None
    sum_79: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1], True);  mul_184 = None
    view_557: "f32[768]" = torch.ops.aten.view.default(sum_79, [768]);  sum_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_558: "f32[6272, 768]" = torch.ops.aten.view.default(mul_183, [6272, 768]);  mul_183 = None
    t_257: "f32[768, 3072]" = torch.ops.aten.t.default(t_71);  t_71 = None
    mm_100: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_558, t_257);  t_257 = None
    t_258: "f32[768, 6272]" = torch.ops.aten.t.default(view_558)
    mm_101: "f32[768, 3072]" = torch.ops.aten.mm.default(t_258, view_254);  t_258 = view_254 = None
    t_259: "f32[3072, 768]" = torch.ops.aten.t.default(mm_101);  mm_101 = None
    sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_558, [0], True);  view_558 = None
    view_559: "f32[768]" = torch.ops.aten.view.default(sum_80, [768]);  sum_80 = None
    t_260: "f32[768, 3072]" = torch.ops.aten.t.default(t_259);  t_259 = None
    view_560: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_100, [8, 784, 3072]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_14: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_560, view_253);  view_560 = view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_561: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_14, [6272, 3072]);  gelu_backward_14 = None
    t_261: "f32[3072, 768]" = torch.ops.aten.t.default(t_70);  t_70 = None
    mm_102: "f32[6272, 768]" = torch.ops.aten.mm.default(view_561, t_261);  t_261 = None
    t_262: "f32[3072, 6272]" = torch.ops.aten.t.default(view_561)
    mm_103: "f32[3072, 768]" = torch.ops.aten.mm.default(t_262, view_252);  t_262 = view_252 = None
    t_263: "f32[768, 3072]" = torch.ops.aten.t.default(mm_103);  mm_103 = None
    sum_81: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
    view_562: "f32[3072]" = torch.ops.aten.view.default(sum_81, [3072]);  sum_81 = None
    t_264: "f32[3072, 768]" = torch.ops.aten.t.default(t_263);  t_263 = None
    view_563: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_102, [8, 784, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_23 = torch.ops.aten.native_layer_norm_backward.default(view_563, add_94, [768], getitem_319, getitem_320, primals_467, primals_468, [True, True, True]);  view_563 = add_94 = getitem_319 = getitem_320 = primals_467 = primals_468 = None
    getitem_585: "f32[8, 784, 768]" = native_layer_norm_backward_23[0]
    getitem_586: "f32[768]" = native_layer_norm_backward_23[1]
    getitem_587: "f32[768]" = native_layer_norm_backward_23[2];  native_layer_norm_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_174: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_173, getitem_585);  add_173 = getitem_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_185: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_174, primals_71);  primals_71 = None
    mul_186: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_174, permute_73);  permute_73 = None
    sum_82: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_186, [0, 1], True);  mul_186 = None
    view_564: "f32[768]" = torch.ops.aten.view.default(sum_82, [768]);  sum_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_134: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_185, [0, 2, 1]);  mul_185 = None
    view_565: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_134, [8, 768, 28, 28]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(view_565, getitem_313, primals_465, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_565 = getitem_313 = primals_465 = None
    getitem_588: "f32[8, 768, 28, 28]" = convolution_backward_12[0]
    getitem_589: "f32[768, 1, 3, 3]" = convolution_backward_12[1]
    getitem_590: "f32[768]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_6 = torch.ops.aten.native_batch_norm_backward.default(getitem_588, gelu_36, primals_463, getitem_316, getitem_317, getitem_314, getitem_315, True, 1e-05, [True, True, True]);  getitem_588 = gelu_36 = primals_463 = getitem_314 = getitem_315 = None
    getitem_591: "f32[8, 768, 28, 28]" = native_batch_norm_backward_6[0]
    getitem_592: "f32[768]" = native_batch_norm_backward_6[1]
    getitem_593: "f32[768]" = native_batch_norm_backward_6[2];  native_batch_norm_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_15: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_591, convolution_38);  getitem_591 = convolution_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(gelu_backward_15, view_250, primals_461, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_15 = view_250 = primals_461 = None
    getitem_594: "f32[8, 768, 28, 28]" = convolution_backward_13[0]
    getitem_595: "f32[768, 1, 3, 3]" = convolution_backward_13[1]
    getitem_596: "f32[768]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_566: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_594, [8, 768, 784]);  getitem_594 = None
    permute_135: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_566, [0, 2, 1]);  view_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_24 = torch.ops.aten.native_layer_norm_backward.default(permute_135, add_92, [768], getitem_311, getitem_312, primals_459, primals_460, [True, True, True]);  permute_135 = add_92 = getitem_311 = getitem_312 = primals_459 = primals_460 = None
    getitem_597: "f32[8, 784, 768]" = native_layer_norm_backward_24[0]
    getitem_598: "f32[768]" = native_layer_norm_backward_24[1]
    getitem_599: "f32[768]" = native_layer_norm_backward_24[2];  native_layer_norm_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_175: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_174, getitem_597);  add_174 = getitem_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_187: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_175, primals_69);  primals_69 = None
    mul_188: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_175, clone_142);  clone_142 = None
    sum_83: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 1], True);  mul_188 = None
    view_567: "f32[768]" = torch.ops.aten.view.default(sum_83, [768]);  sum_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_84: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_187, [0, 1], True)
    view_568: "f32[768]" = torch.ops.aten.view.default(sum_84, [768]);  sum_84 = None
    view_569: "f32[6272, 768]" = torch.ops.aten.view.default(mul_187, [6272, 768]);  mul_187 = None
    t_265: "f32[768, 6272]" = torch.ops.aten.t.default(view_569)
    mm_104: "f32[768, 768]" = torch.ops.aten.mm.default(t_265, _unsafe_view_71);  t_265 = _unsafe_view_71 = None
    t_266: "f32[768, 768]" = torch.ops.aten.t.default(mm_104);  mm_104 = None
    t_267: "f32[768, 768]" = torch.ops.aten.t.default(t_69);  t_69 = None
    mm_105: "f32[6272, 768]" = torch.ops.aten.mm.default(view_569, t_267);  view_569 = t_267 = None
    view_570: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_105, [8, 784, 768]);  mm_105 = None
    t_268: "f32[768, 768]" = torch.ops.aten.t.default(t_266);  t_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_571: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_570, [8, 784, 16, 48]);  view_570 = None
    permute_136: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_571, [0, 2, 3, 1]);  view_571 = None
    clone_212: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    _unsafe_view_108: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_212, [128, 48, 784]);  clone_212 = None
    transpose_59: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_246, 1, 2);  view_246 = None
    bmm_72: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_59, _unsafe_view_108);  transpose_59 = None
    transpose_60: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_70, 1, 2);  _unsafe_view_70 = None
    bmm_73: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_108, transpose_60);  _unsafe_view_108 = transpose_60 = None
    view_572: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_72, [8, 16, 48, 784]);  bmm_72 = None
    view_573: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_73, [8, 16, 48, 48]);  bmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_94: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_53);  detach_53 = None
    _softmax_backward_data_6: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_573, detach_94, -1, torch.float32);  view_573 = detach_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_189: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_6, view_245);  view_245 = None
    mul_190: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_6, primals_70);  _softmax_backward_data_6 = primals_70 = None
    sum_85: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 2, 3], True);  mul_189 = None
    view_574: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_85, [16, 1, 1]);  sum_85 = None
    view_575: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_190, [128, 48, 48]);  mul_190 = None
    transpose_61: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_68, 1, 2);  _unsafe_view_68 = None
    bmm_74: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_61, view_575);  transpose_61 = None
    transpose_62: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_69, 1, 2);  _unsafe_view_69 = None
    bmm_75: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_575, transpose_62);  view_575 = transpose_62 = None
    view_576: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_74, [8, 16, 784, 48]);  bmm_74 = None
    view_577: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_75, [8, 16, 48, 784]);  bmm_75 = None
    transpose_63: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_576, -2, -1);  view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_102: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_308, expand_103)
    div_103: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_102, expand_103);  div_102 = None
    neg_12: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_63)
    mul_191: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_12, div_103);  neg_12 = div_103 = None
    div_104: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_63, expand_103);  transpose_63 = expand_103 = None
    sum_86: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_191, [3], True);  mul_191 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_12: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_35, 1e-12);  linalg_vector_norm_35 = None
    where_12: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_12, sum_86, scalar_tensor_12);  ge_12 = sum_86 = scalar_tensor_12 = None
    detach_95: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_52);  detach_52 = None
    div_105: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_308, detach_95);  getitem_308 = None
    eq_12: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_95, 0);  detach_95 = None
    masked_fill_12: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_105, eq_12, 0);  div_105 = eq_12 = None
    mul_192: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_12, masked_fill_12);  where_12 = masked_fill_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_176: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_104, mul_192);  div_104 = mul_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_106: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_307, expand_102)
    div_107: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_106, expand_102);  div_106 = None
    neg_13: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_577)
    mul_193: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_13, div_107);  neg_13 = div_107 = None
    div_108: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_577, expand_102);  view_577 = expand_102 = None
    sum_87: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [3], True);  mul_193 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_13: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_34, 1e-12);  linalg_vector_norm_34 = None
    where_13: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_13, sum_87, scalar_tensor_13);  ge_13 = sum_87 = scalar_tensor_13 = None
    detach_96: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_51);  detach_51 = None
    div_109: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_307, detach_96);  getitem_307 = None
    eq_13: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_96, 0);  detach_96 = None
    masked_fill_13: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_109, eq_13, 0);  div_109 = eq_13 = None
    mul_194: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_13, masked_fill_13);  where_13 = masked_fill_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_177: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_108, mul_194);  div_108 = mul_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_8: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_177, add_176, view_572]);  add_177 = add_176 = view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_137: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_8, [1, 4, 0, 2, 3]);  stack_8 = None
    clone_213: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    _unsafe_view_109: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_213, [8, 784, 2304]);  clone_213 = None
    view_578: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_109, [6272, 2304]);  _unsafe_view_109 = None
    t_269: "f32[2304, 768]" = torch.ops.aten.t.default(t_68);  t_68 = None
    mm_106: "f32[6272, 768]" = torch.ops.aten.mm.default(view_578, t_269);  t_269 = None
    t_270: "f32[2304, 6272]" = torch.ops.aten.t.default(view_578)
    mm_107: "f32[2304, 768]" = torch.ops.aten.mm.default(t_270, view_242);  t_270 = view_242 = None
    t_271: "f32[768, 2304]" = torch.ops.aten.t.default(mm_107);  mm_107 = None
    sum_88: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_578, [0], True);  view_578 = None
    view_579: "f32[2304]" = torch.ops.aten.view.default(sum_88, [2304]);  sum_88 = None
    t_272: "f32[2304, 768]" = torch.ops.aten.t.default(t_271);  t_271 = None
    view_580: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_106, [8, 784, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_25 = torch.ops.aten.native_layer_norm_backward.default(view_580, add_90, [768], getitem_305, getitem_306, primals_453, primals_454, [True, True, True]);  view_580 = add_90 = getitem_305 = getitem_306 = primals_453 = primals_454 = None
    getitem_600: "f32[8, 784, 768]" = native_layer_norm_backward_25[0]
    getitem_601: "f32[768]" = native_layer_norm_backward_25[1]
    getitem_602: "f32[768]" = native_layer_norm_backward_25[2];  native_layer_norm_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_178: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_175, getitem_600);  add_175 = getitem_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_195: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_178, primals_68);  primals_68 = None
    mul_196: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_178, clone_136);  clone_136 = None
    sum_89: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_196, [0, 1], True);  mul_196 = None
    view_581: "f32[768]" = torch.ops.aten.view.default(sum_89, [768]);  sum_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_582: "f32[6272, 768]" = torch.ops.aten.view.default(mul_195, [6272, 768]);  mul_195 = None
    t_273: "f32[768, 3072]" = torch.ops.aten.t.default(t_67);  t_67 = None
    mm_108: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_582, t_273);  t_273 = None
    t_274: "f32[768, 6272]" = torch.ops.aten.t.default(view_582)
    mm_109: "f32[768, 3072]" = torch.ops.aten.mm.default(t_274, view_240);  t_274 = view_240 = None
    t_275: "f32[3072, 768]" = torch.ops.aten.t.default(mm_109);  mm_109 = None
    sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
    view_583: "f32[768]" = torch.ops.aten.view.default(sum_90, [768]);  sum_90 = None
    t_276: "f32[768, 3072]" = torch.ops.aten.t.default(t_275);  t_275 = None
    view_584: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_108, [8, 784, 3072]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_16: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_584, view_239);  view_584 = view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_585: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_16, [6272, 3072]);  gelu_backward_16 = None
    t_277: "f32[3072, 768]" = torch.ops.aten.t.default(t_66);  t_66 = None
    mm_110: "f32[6272, 768]" = torch.ops.aten.mm.default(view_585, t_277);  t_277 = None
    t_278: "f32[3072, 6272]" = torch.ops.aten.t.default(view_585)
    mm_111: "f32[3072, 768]" = torch.ops.aten.mm.default(t_278, view_238);  t_278 = view_238 = None
    t_279: "f32[768, 3072]" = torch.ops.aten.t.default(mm_111);  mm_111 = None
    sum_91: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
    view_586: "f32[3072]" = torch.ops.aten.view.default(sum_91, [3072]);  sum_91 = None
    t_280: "f32[3072, 768]" = torch.ops.aten.t.default(t_279);  t_279 = None
    view_587: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_110, [8, 784, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_26 = torch.ops.aten.native_layer_norm_backward.default(view_587, add_89, [768], getitem_302, getitem_303, primals_447, primals_448, [True, True, True]);  view_587 = add_89 = getitem_302 = getitem_303 = primals_447 = primals_448 = None
    getitem_603: "f32[8, 784, 768]" = native_layer_norm_backward_26[0]
    getitem_604: "f32[768]" = native_layer_norm_backward_26[1]
    getitem_605: "f32[768]" = native_layer_norm_backward_26[2];  native_layer_norm_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_179: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_178, getitem_603);  add_178 = getitem_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_197: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_179, primals_67);  primals_67 = None
    mul_198: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_179, permute_69);  permute_69 = None
    sum_92: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1], True);  mul_198 = None
    view_588: "f32[768]" = torch.ops.aten.view.default(sum_92, [768]);  sum_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_138: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_197, [0, 2, 1]);  mul_197 = None
    view_589: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_138, [8, 768, 28, 28]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(view_589, getitem_296, primals_445, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_589 = getitem_296 = primals_445 = None
    getitem_606: "f32[8, 768, 28, 28]" = convolution_backward_14[0]
    getitem_607: "f32[768, 1, 3, 3]" = convolution_backward_14[1]
    getitem_608: "f32[768]" = convolution_backward_14[2];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_7 = torch.ops.aten.native_batch_norm_backward.default(getitem_606, gelu_34, primals_443, getitem_299, getitem_300, getitem_297, getitem_298, True, 1e-05, [True, True, True]);  getitem_606 = gelu_34 = primals_443 = getitem_297 = getitem_298 = None
    getitem_609: "f32[8, 768, 28, 28]" = native_batch_norm_backward_7[0]
    getitem_610: "f32[768]" = native_batch_norm_backward_7[1]
    getitem_611: "f32[768]" = native_batch_norm_backward_7[2];  native_batch_norm_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_17: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_609, convolution_36);  getitem_609 = convolution_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(gelu_backward_17, view_236, primals_441, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_17 = view_236 = primals_441 = None
    getitem_612: "f32[8, 768, 28, 28]" = convolution_backward_15[0]
    getitem_613: "f32[768, 1, 3, 3]" = convolution_backward_15[1]
    getitem_614: "f32[768]" = convolution_backward_15[2];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_590: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_612, [8, 768, 784]);  getitem_612 = None
    permute_139: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_590, [0, 2, 1]);  view_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_27 = torch.ops.aten.native_layer_norm_backward.default(permute_139, add_87, [768], getitem_294, getitem_295, primals_439, primals_440, [True, True, True]);  permute_139 = add_87 = getitem_294 = getitem_295 = primals_439 = primals_440 = None
    getitem_615: "f32[8, 784, 768]" = native_layer_norm_backward_27[0]
    getitem_616: "f32[768]" = native_layer_norm_backward_27[1]
    getitem_617: "f32[768]" = native_layer_norm_backward_27[2];  native_layer_norm_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_180: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_179, getitem_615);  add_179 = getitem_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_199: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_180, primals_65);  primals_65 = None
    mul_200: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_180, clone_134);  clone_134 = None
    sum_93: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1], True);  mul_200 = None
    view_591: "f32[768]" = torch.ops.aten.view.default(sum_93, [768]);  sum_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_94: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_199, [0, 1], True)
    view_592: "f32[768]" = torch.ops.aten.view.default(sum_94, [768]);  sum_94 = None
    view_593: "f32[6272, 768]" = torch.ops.aten.view.default(mul_199, [6272, 768]);  mul_199 = None
    t_281: "f32[768, 6272]" = torch.ops.aten.t.default(view_593)
    mm_112: "f32[768, 768]" = torch.ops.aten.mm.default(t_281, _unsafe_view_67);  t_281 = _unsafe_view_67 = None
    t_282: "f32[768, 768]" = torch.ops.aten.t.default(mm_112);  mm_112 = None
    t_283: "f32[768, 768]" = torch.ops.aten.t.default(t_65);  t_65 = None
    mm_113: "f32[6272, 768]" = torch.ops.aten.mm.default(view_593, t_283);  view_593 = t_283 = None
    view_594: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_113, [8, 784, 768]);  mm_113 = None
    t_284: "f32[768, 768]" = torch.ops.aten.t.default(t_282);  t_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_595: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_594, [8, 784, 16, 48]);  view_594 = None
    permute_140: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_595, [0, 2, 3, 1]);  view_595 = None
    clone_214: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    _unsafe_view_110: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_214, [128, 48, 784]);  clone_214 = None
    transpose_64: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_232, 1, 2);  view_232 = None
    bmm_76: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_64, _unsafe_view_110);  transpose_64 = None
    transpose_65: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_66, 1, 2);  _unsafe_view_66 = None
    bmm_77: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_110, transpose_65);  _unsafe_view_110 = transpose_65 = None
    view_596: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_76, [8, 16, 48, 784]);  bmm_76 = None
    view_597: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_77, [8, 16, 48, 48]);  bmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_97: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_50);  detach_50 = None
    _softmax_backward_data_7: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_597, detach_97, -1, torch.float32);  view_597 = detach_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_201: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_7, view_231);  view_231 = None
    mul_202: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_7, primals_66);  _softmax_backward_data_7 = primals_66 = None
    sum_95: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [0, 2, 3], True);  mul_201 = None
    view_598: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_95, [16, 1, 1]);  sum_95 = None
    view_599: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_202, [128, 48, 48]);  mul_202 = None
    transpose_66: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_64, 1, 2);  _unsafe_view_64 = None
    bmm_78: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_66, view_599);  transpose_66 = None
    transpose_67: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_65, 1, 2);  _unsafe_view_65 = None
    bmm_79: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_599, transpose_67);  view_599 = transpose_67 = None
    view_600: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_78, [8, 16, 784, 48]);  bmm_78 = None
    view_601: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_79, [8, 16, 48, 784]);  bmm_79 = None
    transpose_68: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_600, -2, -1);  view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_110: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_291, expand_97)
    div_111: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_110, expand_97);  div_110 = None
    neg_14: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_68)
    mul_203: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_14, div_111);  neg_14 = div_111 = None
    div_112: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_68, expand_97);  transpose_68 = expand_97 = None
    sum_96: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [3], True);  mul_203 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_14: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_33, 1e-12);  linalg_vector_norm_33 = None
    where_14: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_14, sum_96, scalar_tensor_14);  ge_14 = sum_96 = scalar_tensor_14 = None
    detach_98: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_49);  detach_49 = None
    div_113: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_291, detach_98);  getitem_291 = None
    eq_14: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_98, 0);  detach_98 = None
    masked_fill_14: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_113, eq_14, 0);  div_113 = eq_14 = None
    mul_204: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_14, masked_fill_14);  where_14 = masked_fill_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_181: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_112, mul_204);  div_112 = mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_114: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_290, expand_96)
    div_115: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_114, expand_96);  div_114 = None
    neg_15: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_601)
    mul_205: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_15, div_115);  neg_15 = div_115 = None
    div_116: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_601, expand_96);  view_601 = expand_96 = None
    sum_97: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_205, [3], True);  mul_205 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_15: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_32, 1e-12);  linalg_vector_norm_32 = None
    where_15: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_15, sum_97, scalar_tensor_15);  ge_15 = sum_97 = scalar_tensor_15 = None
    detach_99: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_48);  detach_48 = None
    div_117: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_290, detach_99);  getitem_290 = None
    eq_15: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_99, 0);  detach_99 = None
    masked_fill_15: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_117, eq_15, 0);  div_117 = eq_15 = None
    mul_206: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_15, masked_fill_15);  where_15 = masked_fill_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_182: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_116, mul_206);  div_116 = mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_9: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_182, add_181, view_596]);  add_182 = add_181 = view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_141: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_9, [1, 4, 0, 2, 3]);  stack_9 = None
    clone_215: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
    _unsafe_view_111: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_215, [8, 784, 2304]);  clone_215 = None
    view_602: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_111, [6272, 2304]);  _unsafe_view_111 = None
    t_285: "f32[2304, 768]" = torch.ops.aten.t.default(t_64);  t_64 = None
    mm_114: "f32[6272, 768]" = torch.ops.aten.mm.default(view_602, t_285);  t_285 = None
    t_286: "f32[2304, 6272]" = torch.ops.aten.t.default(view_602)
    mm_115: "f32[2304, 768]" = torch.ops.aten.mm.default(t_286, view_228);  t_286 = view_228 = None
    t_287: "f32[768, 2304]" = torch.ops.aten.t.default(mm_115);  mm_115 = None
    sum_98: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_602, [0], True);  view_602 = None
    view_603: "f32[2304]" = torch.ops.aten.view.default(sum_98, [2304]);  sum_98 = None
    t_288: "f32[2304, 768]" = torch.ops.aten.t.default(t_287);  t_287 = None
    view_604: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_114, [8, 784, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_28 = torch.ops.aten.native_layer_norm_backward.default(view_604, add_85, [768], getitem_288, getitem_289, primals_433, primals_434, [True, True, True]);  view_604 = add_85 = getitem_288 = getitem_289 = primals_433 = primals_434 = None
    getitem_618: "f32[8, 784, 768]" = native_layer_norm_backward_28[0]
    getitem_619: "f32[768]" = native_layer_norm_backward_28[1]
    getitem_620: "f32[768]" = native_layer_norm_backward_28[2];  native_layer_norm_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_183: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_180, getitem_618);  add_180 = getitem_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_207: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_183, primals_64);  primals_64 = None
    mul_208: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_183, clone_128);  clone_128 = None
    sum_99: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_208, [0, 1], True);  mul_208 = None
    view_605: "f32[768]" = torch.ops.aten.view.default(sum_99, [768]);  sum_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_606: "f32[6272, 768]" = torch.ops.aten.view.default(mul_207, [6272, 768]);  mul_207 = None
    t_289: "f32[768, 3072]" = torch.ops.aten.t.default(t_63);  t_63 = None
    mm_116: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_606, t_289);  t_289 = None
    t_290: "f32[768, 6272]" = torch.ops.aten.t.default(view_606)
    mm_117: "f32[768, 3072]" = torch.ops.aten.mm.default(t_290, view_226);  t_290 = view_226 = None
    t_291: "f32[3072, 768]" = torch.ops.aten.t.default(mm_117);  mm_117 = None
    sum_100: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_606, [0], True);  view_606 = None
    view_607: "f32[768]" = torch.ops.aten.view.default(sum_100, [768]);  sum_100 = None
    t_292: "f32[768, 3072]" = torch.ops.aten.t.default(t_291);  t_291 = None
    view_608: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_116, [8, 784, 3072]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_18: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_608, view_225);  view_608 = view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_609: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_18, [6272, 3072]);  gelu_backward_18 = None
    t_293: "f32[3072, 768]" = torch.ops.aten.t.default(t_62);  t_62 = None
    mm_118: "f32[6272, 768]" = torch.ops.aten.mm.default(view_609, t_293);  t_293 = None
    t_294: "f32[3072, 6272]" = torch.ops.aten.t.default(view_609)
    mm_119: "f32[3072, 768]" = torch.ops.aten.mm.default(t_294, view_224);  t_294 = view_224 = None
    t_295: "f32[768, 3072]" = torch.ops.aten.t.default(mm_119);  mm_119 = None
    sum_101: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_609, [0], True);  view_609 = None
    view_610: "f32[3072]" = torch.ops.aten.view.default(sum_101, [3072]);  sum_101 = None
    t_296: "f32[3072, 768]" = torch.ops.aten.t.default(t_295);  t_295 = None
    view_611: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_118, [8, 784, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_29 = torch.ops.aten.native_layer_norm_backward.default(view_611, add_84, [768], getitem_285, getitem_286, primals_427, primals_428, [True, True, True]);  view_611 = add_84 = getitem_285 = getitem_286 = primals_427 = primals_428 = None
    getitem_621: "f32[8, 784, 768]" = native_layer_norm_backward_29[0]
    getitem_622: "f32[768]" = native_layer_norm_backward_29[1]
    getitem_623: "f32[768]" = native_layer_norm_backward_29[2];  native_layer_norm_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_184: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_183, getitem_621);  add_183 = getitem_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_209: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_184, primals_63);  primals_63 = None
    mul_210: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_184, permute_65);  permute_65 = None
    sum_102: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_210, [0, 1], True);  mul_210 = None
    view_612: "f32[768]" = torch.ops.aten.view.default(sum_102, [768]);  sum_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_142: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_209, [0, 2, 1]);  mul_209 = None
    view_613: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_142, [8, 768, 28, 28]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(view_613, getitem_279, primals_425, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_613 = getitem_279 = primals_425 = None
    getitem_624: "f32[8, 768, 28, 28]" = convolution_backward_16[0]
    getitem_625: "f32[768, 1, 3, 3]" = convolution_backward_16[1]
    getitem_626: "f32[768]" = convolution_backward_16[2];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_8 = torch.ops.aten.native_batch_norm_backward.default(getitem_624, gelu_32, primals_423, getitem_282, getitem_283, getitem_280, getitem_281, True, 1e-05, [True, True, True]);  getitem_624 = gelu_32 = primals_423 = getitem_280 = getitem_281 = None
    getitem_627: "f32[8, 768, 28, 28]" = native_batch_norm_backward_8[0]
    getitem_628: "f32[768]" = native_batch_norm_backward_8[1]
    getitem_629: "f32[768]" = native_batch_norm_backward_8[2];  native_batch_norm_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_19: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_627, convolution_34);  getitem_627 = convolution_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(gelu_backward_19, view_222, primals_421, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_19 = view_222 = primals_421 = None
    getitem_630: "f32[8, 768, 28, 28]" = convolution_backward_17[0]
    getitem_631: "f32[768, 1, 3, 3]" = convolution_backward_17[1]
    getitem_632: "f32[768]" = convolution_backward_17[2];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_614: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_630, [8, 768, 784]);  getitem_630 = None
    permute_143: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_614, [0, 2, 1]);  view_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_30 = torch.ops.aten.native_layer_norm_backward.default(permute_143, add_82, [768], getitem_277, getitem_278, primals_419, primals_420, [True, True, True]);  permute_143 = add_82 = getitem_277 = getitem_278 = primals_419 = primals_420 = None
    getitem_633: "f32[8, 784, 768]" = native_layer_norm_backward_30[0]
    getitem_634: "f32[768]" = native_layer_norm_backward_30[1]
    getitem_635: "f32[768]" = native_layer_norm_backward_30[2];  native_layer_norm_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_185: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_184, getitem_633);  add_184 = getitem_633 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_211: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_185, primals_61);  primals_61 = None
    mul_212: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_185, clone_126);  clone_126 = None
    sum_103: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_212, [0, 1], True);  mul_212 = None
    view_615: "f32[768]" = torch.ops.aten.view.default(sum_103, [768]);  sum_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_104: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1], True)
    view_616: "f32[768]" = torch.ops.aten.view.default(sum_104, [768]);  sum_104 = None
    view_617: "f32[6272, 768]" = torch.ops.aten.view.default(mul_211, [6272, 768]);  mul_211 = None
    t_297: "f32[768, 6272]" = torch.ops.aten.t.default(view_617)
    mm_120: "f32[768, 768]" = torch.ops.aten.mm.default(t_297, _unsafe_view_63);  t_297 = _unsafe_view_63 = None
    t_298: "f32[768, 768]" = torch.ops.aten.t.default(mm_120);  mm_120 = None
    t_299: "f32[768, 768]" = torch.ops.aten.t.default(t_61);  t_61 = None
    mm_121: "f32[6272, 768]" = torch.ops.aten.mm.default(view_617, t_299);  view_617 = t_299 = None
    view_618: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_121, [8, 784, 768]);  mm_121 = None
    t_300: "f32[768, 768]" = torch.ops.aten.t.default(t_298);  t_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_619: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_618, [8, 784, 16, 48]);  view_618 = None
    permute_144: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_619, [0, 2, 3, 1]);  view_619 = None
    clone_216: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    _unsafe_view_112: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_216, [128, 48, 784]);  clone_216 = None
    transpose_69: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_218, 1, 2);  view_218 = None
    bmm_80: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_69, _unsafe_view_112);  transpose_69 = None
    transpose_70: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_62, 1, 2);  _unsafe_view_62 = None
    bmm_81: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_112, transpose_70);  _unsafe_view_112 = transpose_70 = None
    view_620: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_80, [8, 16, 48, 784]);  bmm_80 = None
    view_621: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_81, [8, 16, 48, 48]);  bmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_100: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_47);  detach_47 = None
    _softmax_backward_data_8: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_621, detach_100, -1, torch.float32);  view_621 = detach_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_213: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_8, view_217);  view_217 = None
    mul_214: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_8, primals_62);  _softmax_backward_data_8 = primals_62 = None
    sum_105: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 2, 3], True);  mul_213 = None
    view_622: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_105, [16, 1, 1]);  sum_105 = None
    view_623: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_214, [128, 48, 48]);  mul_214 = None
    transpose_71: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_60, 1, 2);  _unsafe_view_60 = None
    bmm_82: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_71, view_623);  transpose_71 = None
    transpose_72: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_61, 1, 2);  _unsafe_view_61 = None
    bmm_83: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_623, transpose_72);  view_623 = transpose_72 = None
    view_624: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_82, [8, 16, 784, 48]);  bmm_82 = None
    view_625: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_83, [8, 16, 48, 784]);  bmm_83 = None
    transpose_73: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_624, -2, -1);  view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_118: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_274, expand_91)
    div_119: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_118, expand_91);  div_118 = None
    neg_16: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_73)
    mul_215: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_16, div_119);  neg_16 = div_119 = None
    div_120: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_73, expand_91);  transpose_73 = expand_91 = None
    sum_106: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [3], True);  mul_215 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_16: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_31, 1e-12);  linalg_vector_norm_31 = None
    where_16: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_16, sum_106, scalar_tensor_16);  ge_16 = sum_106 = scalar_tensor_16 = None
    detach_101: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_46);  detach_46 = None
    div_121: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_274, detach_101);  getitem_274 = None
    eq_16: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_101, 0);  detach_101 = None
    masked_fill_16: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_121, eq_16, 0);  div_121 = eq_16 = None
    mul_216: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_16, masked_fill_16);  where_16 = masked_fill_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_186: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_120, mul_216);  div_120 = mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_122: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_273, expand_90)
    div_123: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_122, expand_90);  div_122 = None
    neg_17: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_625)
    mul_217: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_17, div_123);  neg_17 = div_123 = None
    div_124: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_625, expand_90);  view_625 = expand_90 = None
    sum_107: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [3], True);  mul_217 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_17: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_30, 1e-12);  linalg_vector_norm_30 = None
    where_17: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_17, sum_107, scalar_tensor_17);  ge_17 = sum_107 = scalar_tensor_17 = None
    detach_102: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_45);  detach_45 = None
    div_125: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_273, detach_102);  getitem_273 = None
    eq_17: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_102, 0);  detach_102 = None
    masked_fill_17: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_125, eq_17, 0);  div_125 = eq_17 = None
    mul_218: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_17, masked_fill_17);  where_17 = masked_fill_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_187: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_124, mul_218);  div_124 = mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_10: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_187, add_186, view_620]);  add_187 = add_186 = view_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_145: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_10, [1, 4, 0, 2, 3]);  stack_10 = None
    clone_217: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    _unsafe_view_113: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_217, [8, 784, 2304]);  clone_217 = None
    view_626: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_113, [6272, 2304]);  _unsafe_view_113 = None
    t_301: "f32[2304, 768]" = torch.ops.aten.t.default(t_60);  t_60 = None
    mm_122: "f32[6272, 768]" = torch.ops.aten.mm.default(view_626, t_301);  t_301 = None
    t_302: "f32[2304, 6272]" = torch.ops.aten.t.default(view_626)
    mm_123: "f32[2304, 768]" = torch.ops.aten.mm.default(t_302, view_214);  t_302 = view_214 = None
    t_303: "f32[768, 2304]" = torch.ops.aten.t.default(mm_123);  mm_123 = None
    sum_108: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_626, [0], True);  view_626 = None
    view_627: "f32[2304]" = torch.ops.aten.view.default(sum_108, [2304]);  sum_108 = None
    t_304: "f32[2304, 768]" = torch.ops.aten.t.default(t_303);  t_303 = None
    view_628: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_122, [8, 784, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_31 = torch.ops.aten.native_layer_norm_backward.default(view_628, add_80, [768], getitem_271, getitem_272, primals_413, primals_414, [True, True, True]);  view_628 = add_80 = getitem_271 = getitem_272 = primals_413 = primals_414 = None
    getitem_636: "f32[8, 784, 768]" = native_layer_norm_backward_31[0]
    getitem_637: "f32[768]" = native_layer_norm_backward_31[1]
    getitem_638: "f32[768]" = native_layer_norm_backward_31[2];  native_layer_norm_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_188: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_185, getitem_636);  add_185 = getitem_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_219: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_188, primals_60);  primals_60 = None
    mul_220: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_188, clone_120);  clone_120 = None
    sum_109: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1], True);  mul_220 = None
    view_629: "f32[768]" = torch.ops.aten.view.default(sum_109, [768]);  sum_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_630: "f32[6272, 768]" = torch.ops.aten.view.default(mul_219, [6272, 768]);  mul_219 = None
    t_305: "f32[768, 3072]" = torch.ops.aten.t.default(t_59);  t_59 = None
    mm_124: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_630, t_305);  t_305 = None
    t_306: "f32[768, 6272]" = torch.ops.aten.t.default(view_630)
    mm_125: "f32[768, 3072]" = torch.ops.aten.mm.default(t_306, view_212);  t_306 = view_212 = None
    t_307: "f32[3072, 768]" = torch.ops.aten.t.default(mm_125);  mm_125 = None
    sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_630, [0], True);  view_630 = None
    view_631: "f32[768]" = torch.ops.aten.view.default(sum_110, [768]);  sum_110 = None
    t_308: "f32[768, 3072]" = torch.ops.aten.t.default(t_307);  t_307 = None
    view_632: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_124, [8, 784, 3072]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_20: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_632, view_211);  view_632 = view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_633: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_20, [6272, 3072]);  gelu_backward_20 = None
    t_309: "f32[3072, 768]" = torch.ops.aten.t.default(t_58);  t_58 = None
    mm_126: "f32[6272, 768]" = torch.ops.aten.mm.default(view_633, t_309);  t_309 = None
    t_310: "f32[3072, 6272]" = torch.ops.aten.t.default(view_633)
    mm_127: "f32[3072, 768]" = torch.ops.aten.mm.default(t_310, view_210);  t_310 = view_210 = None
    t_311: "f32[768, 3072]" = torch.ops.aten.t.default(mm_127);  mm_127 = None
    sum_111: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_633, [0], True);  view_633 = None
    view_634: "f32[3072]" = torch.ops.aten.view.default(sum_111, [3072]);  sum_111 = None
    t_312: "f32[3072, 768]" = torch.ops.aten.t.default(t_311);  t_311 = None
    view_635: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_126, [8, 784, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_32 = torch.ops.aten.native_layer_norm_backward.default(view_635, add_79, [768], getitem_268, getitem_269, primals_407, primals_408, [True, True, True]);  view_635 = add_79 = getitem_268 = getitem_269 = primals_407 = primals_408 = None
    getitem_639: "f32[8, 784, 768]" = native_layer_norm_backward_32[0]
    getitem_640: "f32[768]" = native_layer_norm_backward_32[1]
    getitem_641: "f32[768]" = native_layer_norm_backward_32[2];  native_layer_norm_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_189: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_188, getitem_639);  add_188 = getitem_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_221: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_189, primals_59);  primals_59 = None
    mul_222: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_189, permute_61);  permute_61 = None
    sum_112: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_222, [0, 1], True);  mul_222 = None
    view_636: "f32[768]" = torch.ops.aten.view.default(sum_112, [768]);  sum_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_146: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_221, [0, 2, 1]);  mul_221 = None
    view_637: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_146, [8, 768, 28, 28]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(view_637, getitem_262, primals_405, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_637 = getitem_262 = primals_405 = None
    getitem_642: "f32[8, 768, 28, 28]" = convolution_backward_18[0]
    getitem_643: "f32[768, 1, 3, 3]" = convolution_backward_18[1]
    getitem_644: "f32[768]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_9 = torch.ops.aten.native_batch_norm_backward.default(getitem_642, gelu_30, primals_403, getitem_265, getitem_266, getitem_263, getitem_264, True, 1e-05, [True, True, True]);  getitem_642 = gelu_30 = primals_403 = getitem_263 = getitem_264 = None
    getitem_645: "f32[8, 768, 28, 28]" = native_batch_norm_backward_9[0]
    getitem_646: "f32[768]" = native_batch_norm_backward_9[1]
    getitem_647: "f32[768]" = native_batch_norm_backward_9[2];  native_batch_norm_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_21: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_645, convolution_32);  getitem_645 = convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(gelu_backward_21, view_208, primals_401, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_21 = view_208 = primals_401 = None
    getitem_648: "f32[8, 768, 28, 28]" = convolution_backward_19[0]
    getitem_649: "f32[768, 1, 3, 3]" = convolution_backward_19[1]
    getitem_650: "f32[768]" = convolution_backward_19[2];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_638: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_648, [8, 768, 784]);  getitem_648 = None
    permute_147: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_638, [0, 2, 1]);  view_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_33 = torch.ops.aten.native_layer_norm_backward.default(permute_147, add_77, [768], getitem_260, getitem_261, primals_399, primals_400, [True, True, True]);  permute_147 = add_77 = getitem_260 = getitem_261 = primals_399 = primals_400 = None
    getitem_651: "f32[8, 784, 768]" = native_layer_norm_backward_33[0]
    getitem_652: "f32[768]" = native_layer_norm_backward_33[1]
    getitem_653: "f32[768]" = native_layer_norm_backward_33[2];  native_layer_norm_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_190: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_189, getitem_651);  add_189 = getitem_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_223: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_190, primals_57);  primals_57 = None
    mul_224: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_190, clone_118);  clone_118 = None
    sum_113: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_224, [0, 1], True);  mul_224 = None
    view_639: "f32[768]" = torch.ops.aten.view.default(sum_113, [768]);  sum_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_114: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_223, [0, 1], True)
    view_640: "f32[768]" = torch.ops.aten.view.default(sum_114, [768]);  sum_114 = None
    view_641: "f32[6272, 768]" = torch.ops.aten.view.default(mul_223, [6272, 768]);  mul_223 = None
    t_313: "f32[768, 6272]" = torch.ops.aten.t.default(view_641)
    mm_128: "f32[768, 768]" = torch.ops.aten.mm.default(t_313, _unsafe_view_59);  t_313 = _unsafe_view_59 = None
    t_314: "f32[768, 768]" = torch.ops.aten.t.default(mm_128);  mm_128 = None
    t_315: "f32[768, 768]" = torch.ops.aten.t.default(t_57);  t_57 = None
    mm_129: "f32[6272, 768]" = torch.ops.aten.mm.default(view_641, t_315);  view_641 = t_315 = None
    view_642: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_129, [8, 784, 768]);  mm_129 = None
    t_316: "f32[768, 768]" = torch.ops.aten.t.default(t_314);  t_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_643: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_642, [8, 784, 16, 48]);  view_642 = None
    permute_148: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_643, [0, 2, 3, 1]);  view_643 = None
    clone_218: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    _unsafe_view_114: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_218, [128, 48, 784]);  clone_218 = None
    transpose_74: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_204, 1, 2);  view_204 = None
    bmm_84: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_74, _unsafe_view_114);  transpose_74 = None
    transpose_75: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_58, 1, 2);  _unsafe_view_58 = None
    bmm_85: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_114, transpose_75);  _unsafe_view_114 = transpose_75 = None
    view_644: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_84, [8, 16, 48, 784]);  bmm_84 = None
    view_645: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_85, [8, 16, 48, 48]);  bmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_103: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_44);  detach_44 = None
    _softmax_backward_data_9: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_645, detach_103, -1, torch.float32);  view_645 = detach_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_225: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_9, view_203);  view_203 = None
    mul_226: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_9, primals_58);  _softmax_backward_data_9 = primals_58 = None
    sum_115: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 2, 3], True);  mul_225 = None
    view_646: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_115, [16, 1, 1]);  sum_115 = None
    view_647: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_226, [128, 48, 48]);  mul_226 = None
    transpose_76: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_56, 1, 2);  _unsafe_view_56 = None
    bmm_86: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_76, view_647);  transpose_76 = None
    transpose_77: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_57, 1, 2);  _unsafe_view_57 = None
    bmm_87: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_647, transpose_77);  view_647 = transpose_77 = None
    view_648: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_86, [8, 16, 784, 48]);  bmm_86 = None
    view_649: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_87, [8, 16, 48, 784]);  bmm_87 = None
    transpose_78: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_648, -2, -1);  view_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_126: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_257, expand_85)
    div_127: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_126, expand_85);  div_126 = None
    neg_18: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_78)
    mul_227: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_18, div_127);  neg_18 = div_127 = None
    div_128: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_78, expand_85);  transpose_78 = expand_85 = None
    sum_116: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [3], True);  mul_227 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_18: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_29, 1e-12);  linalg_vector_norm_29 = None
    where_18: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_18, sum_116, scalar_tensor_18);  ge_18 = sum_116 = scalar_tensor_18 = None
    detach_104: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_43);  detach_43 = None
    div_129: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_257, detach_104);  getitem_257 = None
    eq_18: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_104, 0);  detach_104 = None
    masked_fill_18: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_129, eq_18, 0);  div_129 = eq_18 = None
    mul_228: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_18, masked_fill_18);  where_18 = masked_fill_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_191: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_128, mul_228);  div_128 = mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_130: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_256, expand_84)
    div_131: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_130, expand_84);  div_130 = None
    neg_19: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_649)
    mul_229: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_19, div_131);  neg_19 = div_131 = None
    div_132: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_649, expand_84);  view_649 = expand_84 = None
    sum_117: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [3], True);  mul_229 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_19: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_28, 1e-12);  linalg_vector_norm_28 = None
    where_19: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_19, sum_117, scalar_tensor_19);  ge_19 = sum_117 = scalar_tensor_19 = None
    detach_105: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_42);  detach_42 = None
    div_133: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_256, detach_105);  getitem_256 = None
    eq_19: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_105, 0);  detach_105 = None
    masked_fill_19: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_133, eq_19, 0);  div_133 = eq_19 = None
    mul_230: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_19, masked_fill_19);  where_19 = masked_fill_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_192: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_132, mul_230);  div_132 = mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_11: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_192, add_191, view_644]);  add_192 = add_191 = view_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_149: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_11, [1, 4, 0, 2, 3]);  stack_11 = None
    clone_219: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
    _unsafe_view_115: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_219, [8, 784, 2304]);  clone_219 = None
    view_650: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_115, [6272, 2304]);  _unsafe_view_115 = None
    t_317: "f32[2304, 768]" = torch.ops.aten.t.default(t_56);  t_56 = None
    mm_130: "f32[6272, 768]" = torch.ops.aten.mm.default(view_650, t_317);  t_317 = None
    t_318: "f32[2304, 6272]" = torch.ops.aten.t.default(view_650)
    mm_131: "f32[2304, 768]" = torch.ops.aten.mm.default(t_318, view_200);  t_318 = view_200 = None
    t_319: "f32[768, 2304]" = torch.ops.aten.t.default(mm_131);  mm_131 = None
    sum_118: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
    view_651: "f32[2304]" = torch.ops.aten.view.default(sum_118, [2304]);  sum_118 = None
    t_320: "f32[2304, 768]" = torch.ops.aten.t.default(t_319);  t_319 = None
    view_652: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_130, [8, 784, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_34 = torch.ops.aten.native_layer_norm_backward.default(view_652, add_75, [768], getitem_254, getitem_255, primals_393, primals_394, [True, True, True]);  view_652 = add_75 = getitem_254 = getitem_255 = primals_393 = primals_394 = None
    getitem_654: "f32[8, 784, 768]" = native_layer_norm_backward_34[0]
    getitem_655: "f32[768]" = native_layer_norm_backward_34[1]
    getitem_656: "f32[768]" = native_layer_norm_backward_34[2];  native_layer_norm_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_193: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_190, getitem_654);  add_190 = getitem_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_231: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_193, primals_56);  primals_56 = None
    mul_232: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_193, clone_112);  clone_112 = None
    sum_119: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1], True);  mul_232 = None
    view_653: "f32[768]" = torch.ops.aten.view.default(sum_119, [768]);  sum_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_654: "f32[6272, 768]" = torch.ops.aten.view.default(mul_231, [6272, 768]);  mul_231 = None
    t_321: "f32[768, 3072]" = torch.ops.aten.t.default(t_55);  t_55 = None
    mm_132: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_654, t_321);  t_321 = None
    t_322: "f32[768, 6272]" = torch.ops.aten.t.default(view_654)
    mm_133: "f32[768, 3072]" = torch.ops.aten.mm.default(t_322, view_198);  t_322 = view_198 = None
    t_323: "f32[3072, 768]" = torch.ops.aten.t.default(mm_133);  mm_133 = None
    sum_120: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_654, [0], True);  view_654 = None
    view_655: "f32[768]" = torch.ops.aten.view.default(sum_120, [768]);  sum_120 = None
    t_324: "f32[768, 3072]" = torch.ops.aten.t.default(t_323);  t_323 = None
    view_656: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_132, [8, 784, 3072]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_22: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_656, view_197);  view_656 = view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_657: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_22, [6272, 3072]);  gelu_backward_22 = None
    t_325: "f32[3072, 768]" = torch.ops.aten.t.default(t_54);  t_54 = None
    mm_134: "f32[6272, 768]" = torch.ops.aten.mm.default(view_657, t_325);  t_325 = None
    t_326: "f32[3072, 6272]" = torch.ops.aten.t.default(view_657)
    mm_135: "f32[3072, 768]" = torch.ops.aten.mm.default(t_326, view_196);  t_326 = view_196 = None
    t_327: "f32[768, 3072]" = torch.ops.aten.t.default(mm_135);  mm_135 = None
    sum_121: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_657, [0], True);  view_657 = None
    view_658: "f32[3072]" = torch.ops.aten.view.default(sum_121, [3072]);  sum_121 = None
    t_328: "f32[3072, 768]" = torch.ops.aten.t.default(t_327);  t_327 = None
    view_659: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_134, [8, 784, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_35 = torch.ops.aten.native_layer_norm_backward.default(view_659, add_74, [768], getitem_251, getitem_252, primals_387, primals_388, [True, True, True]);  view_659 = add_74 = getitem_251 = getitem_252 = primals_387 = primals_388 = None
    getitem_657: "f32[8, 784, 768]" = native_layer_norm_backward_35[0]
    getitem_658: "f32[768]" = native_layer_norm_backward_35[1]
    getitem_659: "f32[768]" = native_layer_norm_backward_35[2];  native_layer_norm_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_194: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_193, getitem_657);  add_193 = getitem_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_233: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_194, primals_55);  primals_55 = None
    mul_234: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_194, permute_57);  permute_57 = None
    sum_122: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_234, [0, 1], True);  mul_234 = None
    view_660: "f32[768]" = torch.ops.aten.view.default(sum_122, [768]);  sum_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_150: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_233, [0, 2, 1]);  mul_233 = None
    view_661: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_150, [8, 768, 28, 28]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(view_661, getitem_245, primals_385, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_661 = getitem_245 = primals_385 = None
    getitem_660: "f32[8, 768, 28, 28]" = convolution_backward_20[0]
    getitem_661: "f32[768, 1, 3, 3]" = convolution_backward_20[1]
    getitem_662: "f32[768]" = convolution_backward_20[2];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_10 = torch.ops.aten.native_batch_norm_backward.default(getitem_660, gelu_28, primals_383, getitem_248, getitem_249, getitem_246, getitem_247, True, 1e-05, [True, True, True]);  getitem_660 = gelu_28 = primals_383 = getitem_246 = getitem_247 = None
    getitem_663: "f32[8, 768, 28, 28]" = native_batch_norm_backward_10[0]
    getitem_664: "f32[768]" = native_batch_norm_backward_10[1]
    getitem_665: "f32[768]" = native_batch_norm_backward_10[2];  native_batch_norm_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_23: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_663, convolution_30);  getitem_663 = convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(gelu_backward_23, view_194, primals_381, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_23 = view_194 = primals_381 = None
    getitem_666: "f32[8, 768, 28, 28]" = convolution_backward_21[0]
    getitem_667: "f32[768, 1, 3, 3]" = convolution_backward_21[1]
    getitem_668: "f32[768]" = convolution_backward_21[2];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_662: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_666, [8, 768, 784]);  getitem_666 = None
    permute_151: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_662, [0, 2, 1]);  view_662 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_36 = torch.ops.aten.native_layer_norm_backward.default(permute_151, add_72, [768], getitem_243, getitem_244, primals_379, primals_380, [True, True, True]);  permute_151 = add_72 = getitem_243 = getitem_244 = primals_379 = primals_380 = None
    getitem_669: "f32[8, 784, 768]" = native_layer_norm_backward_36[0]
    getitem_670: "f32[768]" = native_layer_norm_backward_36[1]
    getitem_671: "f32[768]" = native_layer_norm_backward_36[2];  native_layer_norm_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_195: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_194, getitem_669);  add_194 = getitem_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_235: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_195, primals_53);  primals_53 = None
    mul_236: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_195, clone_110);  clone_110 = None
    sum_123: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_236, [0, 1], True);  mul_236 = None
    view_663: "f32[768]" = torch.ops.aten.view.default(sum_123, [768]);  sum_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_124: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_235, [0, 1], True)
    view_664: "f32[768]" = torch.ops.aten.view.default(sum_124, [768]);  sum_124 = None
    view_665: "f32[6272, 768]" = torch.ops.aten.view.default(mul_235, [6272, 768]);  mul_235 = None
    t_329: "f32[768, 6272]" = torch.ops.aten.t.default(view_665)
    mm_136: "f32[768, 768]" = torch.ops.aten.mm.default(t_329, _unsafe_view_55);  t_329 = _unsafe_view_55 = None
    t_330: "f32[768, 768]" = torch.ops.aten.t.default(mm_136);  mm_136 = None
    t_331: "f32[768, 768]" = torch.ops.aten.t.default(t_53);  t_53 = None
    mm_137: "f32[6272, 768]" = torch.ops.aten.mm.default(view_665, t_331);  view_665 = t_331 = None
    view_666: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_137, [8, 784, 768]);  mm_137 = None
    t_332: "f32[768, 768]" = torch.ops.aten.t.default(t_330);  t_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_667: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_666, [8, 784, 16, 48]);  view_666 = None
    permute_152: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_667, [0, 2, 3, 1]);  view_667 = None
    clone_220: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
    _unsafe_view_116: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_220, [128, 48, 784]);  clone_220 = None
    transpose_79: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_190, 1, 2);  view_190 = None
    bmm_88: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_79, _unsafe_view_116);  transpose_79 = None
    transpose_80: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_54, 1, 2);  _unsafe_view_54 = None
    bmm_89: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_116, transpose_80);  _unsafe_view_116 = transpose_80 = None
    view_668: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_88, [8, 16, 48, 784]);  bmm_88 = None
    view_669: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_89, [8, 16, 48, 48]);  bmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_106: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_41);  detach_41 = None
    _softmax_backward_data_10: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_669, detach_106, -1, torch.float32);  view_669 = detach_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_237: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_10, view_189);  view_189 = None
    mul_238: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_10, primals_54);  _softmax_backward_data_10 = primals_54 = None
    sum_125: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [0, 2, 3], True);  mul_237 = None
    view_670: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_125, [16, 1, 1]);  sum_125 = None
    view_671: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_238, [128, 48, 48]);  mul_238 = None
    transpose_81: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_52, 1, 2);  _unsafe_view_52 = None
    bmm_90: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_81, view_671);  transpose_81 = None
    transpose_82: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_53, 1, 2);  _unsafe_view_53 = None
    bmm_91: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_671, transpose_82);  view_671 = transpose_82 = None
    view_672: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_90, [8, 16, 784, 48]);  bmm_90 = None
    view_673: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_91, [8, 16, 48, 784]);  bmm_91 = None
    transpose_83: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_672, -2, -1);  view_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_134: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_240, expand_79)
    div_135: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_134, expand_79);  div_134 = None
    neg_20: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_83)
    mul_239: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_20, div_135);  neg_20 = div_135 = None
    div_136: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_83, expand_79);  transpose_83 = expand_79 = None
    sum_126: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [3], True);  mul_239 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_20: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_27, 1e-12);  linalg_vector_norm_27 = None
    where_20: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_20, sum_126, scalar_tensor_20);  ge_20 = sum_126 = scalar_tensor_20 = None
    detach_107: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_40);  detach_40 = None
    div_137: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_240, detach_107);  getitem_240 = None
    eq_20: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_107, 0);  detach_107 = None
    masked_fill_20: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_137, eq_20, 0);  div_137 = eq_20 = None
    mul_240: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_20, masked_fill_20);  where_20 = masked_fill_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_196: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_136, mul_240);  div_136 = mul_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_138: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_239, expand_78)
    div_139: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_138, expand_78);  div_138 = None
    neg_21: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_673)
    mul_241: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_21, div_139);  neg_21 = div_139 = None
    div_140: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_673, expand_78);  view_673 = expand_78 = None
    sum_127: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [3], True);  mul_241 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_21: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_26, 1e-12);  linalg_vector_norm_26 = None
    where_21: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_21, sum_127, scalar_tensor_21);  ge_21 = sum_127 = scalar_tensor_21 = None
    detach_108: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_39);  detach_39 = None
    div_141: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_239, detach_108);  getitem_239 = None
    eq_21: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_108, 0);  detach_108 = None
    masked_fill_21: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_141, eq_21, 0);  div_141 = eq_21 = None
    mul_242: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_21, masked_fill_21);  where_21 = masked_fill_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_197: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_140, mul_242);  div_140 = mul_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_12: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_197, add_196, view_668]);  add_197 = add_196 = view_668 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_153: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_12, [1, 4, 0, 2, 3]);  stack_12 = None
    clone_221: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
    _unsafe_view_117: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_221, [8, 784, 2304]);  clone_221 = None
    view_674: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_117, [6272, 2304]);  _unsafe_view_117 = None
    t_333: "f32[2304, 768]" = torch.ops.aten.t.default(t_52);  t_52 = None
    mm_138: "f32[6272, 768]" = torch.ops.aten.mm.default(view_674, t_333);  t_333 = None
    t_334: "f32[2304, 6272]" = torch.ops.aten.t.default(view_674)
    mm_139: "f32[2304, 768]" = torch.ops.aten.mm.default(t_334, view_186);  t_334 = view_186 = None
    t_335: "f32[768, 2304]" = torch.ops.aten.t.default(mm_139);  mm_139 = None
    sum_128: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_674, [0], True);  view_674 = None
    view_675: "f32[2304]" = torch.ops.aten.view.default(sum_128, [2304]);  sum_128 = None
    t_336: "f32[2304, 768]" = torch.ops.aten.t.default(t_335);  t_335 = None
    view_676: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_138, [8, 784, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_37 = torch.ops.aten.native_layer_norm_backward.default(view_676, add_70, [768], getitem_237, getitem_238, primals_373, primals_374, [True, True, True]);  view_676 = add_70 = getitem_237 = getitem_238 = primals_373 = primals_374 = None
    getitem_672: "f32[8, 784, 768]" = native_layer_norm_backward_37[0]
    getitem_673: "f32[768]" = native_layer_norm_backward_37[1]
    getitem_674: "f32[768]" = native_layer_norm_backward_37[2];  native_layer_norm_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_198: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_195, getitem_672);  add_195 = getitem_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_243: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_198, primals_52);  primals_52 = None
    mul_244: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_198, clone_104);  clone_104 = None
    sum_129: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_244, [0, 1], True);  mul_244 = None
    view_677: "f32[768]" = torch.ops.aten.view.default(sum_129, [768]);  sum_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_678: "f32[6272, 768]" = torch.ops.aten.view.default(mul_243, [6272, 768]);  mul_243 = None
    t_337: "f32[768, 3072]" = torch.ops.aten.t.default(t_51);  t_51 = None
    mm_140: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_678, t_337);  t_337 = None
    t_338: "f32[768, 6272]" = torch.ops.aten.t.default(view_678)
    mm_141: "f32[768, 3072]" = torch.ops.aten.mm.default(t_338, view_184);  t_338 = view_184 = None
    t_339: "f32[3072, 768]" = torch.ops.aten.t.default(mm_141);  mm_141 = None
    sum_130: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_678, [0], True);  view_678 = None
    view_679: "f32[768]" = torch.ops.aten.view.default(sum_130, [768]);  sum_130 = None
    t_340: "f32[768, 3072]" = torch.ops.aten.t.default(t_339);  t_339 = None
    view_680: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_140, [8, 784, 3072]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_24: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_680, view_183);  view_680 = view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_681: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_24, [6272, 3072]);  gelu_backward_24 = None
    t_341: "f32[3072, 768]" = torch.ops.aten.t.default(t_50);  t_50 = None
    mm_142: "f32[6272, 768]" = torch.ops.aten.mm.default(view_681, t_341);  t_341 = None
    t_342: "f32[3072, 6272]" = torch.ops.aten.t.default(view_681)
    mm_143: "f32[3072, 768]" = torch.ops.aten.mm.default(t_342, view_182);  t_342 = view_182 = None
    t_343: "f32[768, 3072]" = torch.ops.aten.t.default(mm_143);  mm_143 = None
    sum_131: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_681, [0], True);  view_681 = None
    view_682: "f32[3072]" = torch.ops.aten.view.default(sum_131, [3072]);  sum_131 = None
    t_344: "f32[3072, 768]" = torch.ops.aten.t.default(t_343);  t_343 = None
    view_683: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_142, [8, 784, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_38 = torch.ops.aten.native_layer_norm_backward.default(view_683, add_69, [768], getitem_234, getitem_235, primals_367, primals_368, [True, True, True]);  view_683 = add_69 = getitem_234 = getitem_235 = primals_367 = primals_368 = None
    getitem_675: "f32[8, 784, 768]" = native_layer_norm_backward_38[0]
    getitem_676: "f32[768]" = native_layer_norm_backward_38[1]
    getitem_677: "f32[768]" = native_layer_norm_backward_38[2];  native_layer_norm_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_199: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_198, getitem_675);  add_198 = getitem_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_245: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_199, primals_51);  primals_51 = None
    mul_246: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_199, permute_53);  permute_53 = None
    sum_132: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1], True);  mul_246 = None
    view_684: "f32[768]" = torch.ops.aten.view.default(sum_132, [768]);  sum_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_154: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_245, [0, 2, 1]);  mul_245 = None
    view_685: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_154, [8, 768, 28, 28]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(view_685, getitem_228, primals_365, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_685 = getitem_228 = primals_365 = None
    getitem_678: "f32[8, 768, 28, 28]" = convolution_backward_22[0]
    getitem_679: "f32[768, 1, 3, 3]" = convolution_backward_22[1]
    getitem_680: "f32[768]" = convolution_backward_22[2];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_11 = torch.ops.aten.native_batch_norm_backward.default(getitem_678, gelu_26, primals_363, getitem_231, getitem_232, getitem_229, getitem_230, True, 1e-05, [True, True, True]);  getitem_678 = gelu_26 = primals_363 = getitem_229 = getitem_230 = None
    getitem_681: "f32[8, 768, 28, 28]" = native_batch_norm_backward_11[0]
    getitem_682: "f32[768]" = native_batch_norm_backward_11[1]
    getitem_683: "f32[768]" = native_batch_norm_backward_11[2];  native_batch_norm_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_25: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_681, convolution_28);  getitem_681 = convolution_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(gelu_backward_25, view_180, primals_361, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_25 = view_180 = primals_361 = None
    getitem_684: "f32[8, 768, 28, 28]" = convolution_backward_23[0]
    getitem_685: "f32[768, 1, 3, 3]" = convolution_backward_23[1]
    getitem_686: "f32[768]" = convolution_backward_23[2];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_686: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_684, [8, 768, 784]);  getitem_684 = None
    permute_155: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_686, [0, 2, 1]);  view_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_39 = torch.ops.aten.native_layer_norm_backward.default(permute_155, add_67, [768], getitem_226, getitem_227, primals_359, primals_360, [True, True, True]);  permute_155 = add_67 = getitem_226 = getitem_227 = primals_359 = primals_360 = None
    getitem_687: "f32[8, 784, 768]" = native_layer_norm_backward_39[0]
    getitem_688: "f32[768]" = native_layer_norm_backward_39[1]
    getitem_689: "f32[768]" = native_layer_norm_backward_39[2];  native_layer_norm_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_200: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_199, getitem_687);  add_199 = getitem_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_247: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_200, primals_49);  primals_49 = None
    mul_248: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_200, clone_102);  clone_102 = None
    sum_133: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1], True);  mul_248 = None
    view_687: "f32[768]" = torch.ops.aten.view.default(sum_133, [768]);  sum_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_134: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_247, [0, 1], True)
    view_688: "f32[768]" = torch.ops.aten.view.default(sum_134, [768]);  sum_134 = None
    view_689: "f32[6272, 768]" = torch.ops.aten.view.default(mul_247, [6272, 768]);  mul_247 = None
    t_345: "f32[768, 6272]" = torch.ops.aten.t.default(view_689)
    mm_144: "f32[768, 768]" = torch.ops.aten.mm.default(t_345, _unsafe_view_51);  t_345 = _unsafe_view_51 = None
    t_346: "f32[768, 768]" = torch.ops.aten.t.default(mm_144);  mm_144 = None
    t_347: "f32[768, 768]" = torch.ops.aten.t.default(t_49);  t_49 = None
    mm_145: "f32[6272, 768]" = torch.ops.aten.mm.default(view_689, t_347);  view_689 = t_347 = None
    view_690: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_145, [8, 784, 768]);  mm_145 = None
    t_348: "f32[768, 768]" = torch.ops.aten.t.default(t_346);  t_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_691: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_690, [8, 784, 16, 48]);  view_690 = None
    permute_156: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_691, [0, 2, 3, 1]);  view_691 = None
    clone_222: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    _unsafe_view_118: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_222, [128, 48, 784]);  clone_222 = None
    transpose_84: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_176, 1, 2);  view_176 = None
    bmm_92: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_84, _unsafe_view_118);  transpose_84 = None
    transpose_85: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_50, 1, 2);  _unsafe_view_50 = None
    bmm_93: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_118, transpose_85);  _unsafe_view_118 = transpose_85 = None
    view_692: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_92, [8, 16, 48, 784]);  bmm_92 = None
    view_693: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_93, [8, 16, 48, 48]);  bmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_109: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_38);  detach_38 = None
    _softmax_backward_data_11: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_693, detach_109, -1, torch.float32);  view_693 = detach_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_249: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_11, view_175);  view_175 = None
    mul_250: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_11, primals_50);  _softmax_backward_data_11 = primals_50 = None
    sum_135: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [0, 2, 3], True);  mul_249 = None
    view_694: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_135, [16, 1, 1]);  sum_135 = None
    view_695: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_250, [128, 48, 48]);  mul_250 = None
    transpose_86: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_48, 1, 2);  _unsafe_view_48 = None
    bmm_94: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_86, view_695);  transpose_86 = None
    transpose_87: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_49, 1, 2);  _unsafe_view_49 = None
    bmm_95: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_695, transpose_87);  view_695 = transpose_87 = None
    view_696: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_94, [8, 16, 784, 48]);  bmm_94 = None
    view_697: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_95, [8, 16, 48, 784]);  bmm_95 = None
    transpose_88: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_696, -2, -1);  view_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_142: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_223, expand_73)
    div_143: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_142, expand_73);  div_142 = None
    neg_22: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_88)
    mul_251: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_22, div_143);  neg_22 = div_143 = None
    div_144: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_88, expand_73);  transpose_88 = expand_73 = None
    sum_136: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [3], True);  mul_251 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_22: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_25, 1e-12);  linalg_vector_norm_25 = None
    where_22: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_22, sum_136, scalar_tensor_22);  ge_22 = sum_136 = scalar_tensor_22 = None
    detach_110: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_37);  detach_37 = None
    div_145: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_223, detach_110);  getitem_223 = None
    eq_22: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_110, 0);  detach_110 = None
    masked_fill_22: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_145, eq_22, 0);  div_145 = eq_22 = None
    mul_252: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_22, masked_fill_22);  where_22 = masked_fill_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_201: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_144, mul_252);  div_144 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_146: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_222, expand_72)
    div_147: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_146, expand_72);  div_146 = None
    neg_23: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_697)
    mul_253: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_23, div_147);  neg_23 = div_147 = None
    div_148: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_697, expand_72);  view_697 = expand_72 = None
    sum_137: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [3], True);  mul_253 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_23: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_24, 1e-12);  linalg_vector_norm_24 = None
    where_23: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_23, sum_137, scalar_tensor_23);  ge_23 = sum_137 = scalar_tensor_23 = None
    detach_111: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_36);  detach_36 = None
    div_149: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_222, detach_111);  getitem_222 = None
    eq_23: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_111, 0);  detach_111 = None
    masked_fill_23: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_149, eq_23, 0);  div_149 = eq_23 = None
    mul_254: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_23, masked_fill_23);  where_23 = masked_fill_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_202: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_148, mul_254);  div_148 = mul_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_13: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_202, add_201, view_692]);  add_202 = add_201 = view_692 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_157: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_13, [1, 4, 0, 2, 3]);  stack_13 = None
    clone_223: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    _unsafe_view_119: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_223, [8, 784, 2304]);  clone_223 = None
    view_698: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_119, [6272, 2304]);  _unsafe_view_119 = None
    t_349: "f32[2304, 768]" = torch.ops.aten.t.default(t_48);  t_48 = None
    mm_146: "f32[6272, 768]" = torch.ops.aten.mm.default(view_698, t_349);  t_349 = None
    t_350: "f32[2304, 6272]" = torch.ops.aten.t.default(view_698)
    mm_147: "f32[2304, 768]" = torch.ops.aten.mm.default(t_350, view_172);  t_350 = view_172 = None
    t_351: "f32[768, 2304]" = torch.ops.aten.t.default(mm_147);  mm_147 = None
    sum_138: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_698, [0], True);  view_698 = None
    view_699: "f32[2304]" = torch.ops.aten.view.default(sum_138, [2304]);  sum_138 = None
    t_352: "f32[2304, 768]" = torch.ops.aten.t.default(t_351);  t_351 = None
    view_700: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_146, [8, 784, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_40 = torch.ops.aten.native_layer_norm_backward.default(view_700, add_65, [768], getitem_220, getitem_221, primals_353, primals_354, [True, True, True]);  view_700 = add_65 = getitem_220 = getitem_221 = primals_353 = primals_354 = None
    getitem_690: "f32[8, 784, 768]" = native_layer_norm_backward_40[0]
    getitem_691: "f32[768]" = native_layer_norm_backward_40[1]
    getitem_692: "f32[768]" = native_layer_norm_backward_40[2];  native_layer_norm_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_203: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_200, getitem_690);  add_200 = getitem_690 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_255: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_203, primals_48);  primals_48 = None
    mul_256: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_203, clone_96);  clone_96 = None
    sum_139: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1], True);  mul_256 = None
    view_701: "f32[768]" = torch.ops.aten.view.default(sum_139, [768]);  sum_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_702: "f32[6272, 768]" = torch.ops.aten.view.default(mul_255, [6272, 768]);  mul_255 = None
    t_353: "f32[768, 3072]" = torch.ops.aten.t.default(t_47);  t_47 = None
    mm_148: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_702, t_353);  t_353 = None
    t_354: "f32[768, 6272]" = torch.ops.aten.t.default(view_702)
    mm_149: "f32[768, 3072]" = torch.ops.aten.mm.default(t_354, view_170);  t_354 = view_170 = None
    t_355: "f32[3072, 768]" = torch.ops.aten.t.default(mm_149);  mm_149 = None
    sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_702, [0], True);  view_702 = None
    view_703: "f32[768]" = torch.ops.aten.view.default(sum_140, [768]);  sum_140 = None
    t_356: "f32[768, 3072]" = torch.ops.aten.t.default(t_355);  t_355 = None
    view_704: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_148, [8, 784, 3072]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_26: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_704, view_169);  view_704 = view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_705: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_26, [6272, 3072]);  gelu_backward_26 = None
    t_357: "f32[3072, 768]" = torch.ops.aten.t.default(t_46);  t_46 = None
    mm_150: "f32[6272, 768]" = torch.ops.aten.mm.default(view_705, t_357);  t_357 = None
    t_358: "f32[3072, 6272]" = torch.ops.aten.t.default(view_705)
    mm_151: "f32[3072, 768]" = torch.ops.aten.mm.default(t_358, view_168);  t_358 = view_168 = None
    t_359: "f32[768, 3072]" = torch.ops.aten.t.default(mm_151);  mm_151 = None
    sum_141: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_705, [0], True);  view_705 = None
    view_706: "f32[3072]" = torch.ops.aten.view.default(sum_141, [3072]);  sum_141 = None
    t_360: "f32[3072, 768]" = torch.ops.aten.t.default(t_359);  t_359 = None
    view_707: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_150, [8, 784, 768]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_41 = torch.ops.aten.native_layer_norm_backward.default(view_707, add_64, [768], getitem_217, getitem_218, primals_347, primals_348, [True, True, True]);  view_707 = add_64 = getitem_217 = getitem_218 = primals_347 = primals_348 = None
    getitem_693: "f32[8, 784, 768]" = native_layer_norm_backward_41[0]
    getitem_694: "f32[768]" = native_layer_norm_backward_41[1]
    getitem_695: "f32[768]" = native_layer_norm_backward_41[2];  native_layer_norm_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_204: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_203, getitem_693);  add_203 = getitem_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_257: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_204, primals_47);  primals_47 = None
    mul_258: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_204, permute_49);  permute_49 = None
    sum_142: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1], True);  mul_258 = None
    view_708: "f32[768]" = torch.ops.aten.view.default(sum_142, [768]);  sum_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_158: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_257, [0, 2, 1]);  mul_257 = None
    view_709: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_158, [8, 768, 28, 28]);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(view_709, getitem_211, primals_345, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_709 = getitem_211 = primals_345 = None
    getitem_696: "f32[8, 768, 28, 28]" = convolution_backward_24[0]
    getitem_697: "f32[768, 1, 3, 3]" = convolution_backward_24[1]
    getitem_698: "f32[768]" = convolution_backward_24[2];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_12 = torch.ops.aten.native_batch_norm_backward.default(getitem_696, gelu_24, primals_343, getitem_214, getitem_215, getitem_212, getitem_213, True, 1e-05, [True, True, True]);  getitem_696 = gelu_24 = primals_343 = getitem_212 = getitem_213 = None
    getitem_699: "f32[8, 768, 28, 28]" = native_batch_norm_backward_12[0]
    getitem_700: "f32[768]" = native_batch_norm_backward_12[1]
    getitem_701: "f32[768]" = native_batch_norm_backward_12[2];  native_batch_norm_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_27: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_699, convolution_26);  getitem_699 = convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(gelu_backward_27, view_166, primals_341, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_27 = view_166 = primals_341 = None
    getitem_702: "f32[8, 768, 28, 28]" = convolution_backward_25[0]
    getitem_703: "f32[768, 1, 3, 3]" = convolution_backward_25[1]
    getitem_704: "f32[768]" = convolution_backward_25[2];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_710: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_702, [8, 768, 784]);  getitem_702 = None
    permute_159: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_710, [0, 2, 1]);  view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_42 = torch.ops.aten.native_layer_norm_backward.default(permute_159, add_62, [768], getitem_209, getitem_210, primals_339, primals_340, [True, True, True]);  permute_159 = add_62 = getitem_209 = getitem_210 = primals_339 = primals_340 = None
    getitem_705: "f32[8, 784, 768]" = native_layer_norm_backward_42[0]
    getitem_706: "f32[768]" = native_layer_norm_backward_42[1]
    getitem_707: "f32[768]" = native_layer_norm_backward_42[2];  native_layer_norm_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_205: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_204, getitem_705);  add_204 = getitem_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_259: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_205, primals_45);  primals_45 = None
    mul_260: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_205, clone_94);  clone_94 = None
    sum_143: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 1], True);  mul_260 = None
    view_711: "f32[768]" = torch.ops.aten.view.default(sum_143, [768]);  sum_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_144: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1], True)
    view_712: "f32[768]" = torch.ops.aten.view.default(sum_144, [768]);  sum_144 = None
    view_713: "f32[6272, 768]" = torch.ops.aten.view.default(mul_259, [6272, 768]);  mul_259 = None
    t_361: "f32[768, 6272]" = torch.ops.aten.t.default(view_713)
    mm_152: "f32[768, 768]" = torch.ops.aten.mm.default(t_361, _unsafe_view_47);  t_361 = _unsafe_view_47 = None
    t_362: "f32[768, 768]" = torch.ops.aten.t.default(mm_152);  mm_152 = None
    t_363: "f32[768, 768]" = torch.ops.aten.t.default(t_45);  t_45 = None
    mm_153: "f32[6272, 768]" = torch.ops.aten.mm.default(view_713, t_363);  view_713 = t_363 = None
    view_714: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_153, [8, 784, 768]);  mm_153 = None
    t_364: "f32[768, 768]" = torch.ops.aten.t.default(t_362);  t_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_715: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_714, [8, 784, 16, 48]);  view_714 = None
    permute_160: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_715, [0, 2, 3, 1]);  view_715 = None
    clone_224: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    _unsafe_view_120: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_224, [128, 48, 784]);  clone_224 = None
    transpose_89: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_162, 1, 2);  view_162 = None
    bmm_96: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_89, _unsafe_view_120);  transpose_89 = None
    transpose_90: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_46, 1, 2);  _unsafe_view_46 = None
    bmm_97: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_120, transpose_90);  _unsafe_view_120 = transpose_90 = None
    view_716: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_96, [8, 16, 48, 784]);  bmm_96 = None
    view_717: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_97, [8, 16, 48, 48]);  bmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_112: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_35);  detach_35 = None
    _softmax_backward_data_12: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_717, detach_112, -1, torch.float32);  view_717 = detach_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_261: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_12, view_161);  view_161 = None
    mul_262: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_12, primals_46);  _softmax_backward_data_12 = primals_46 = None
    sum_145: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [0, 2, 3], True);  mul_261 = None
    view_718: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_145, [16, 1, 1]);  sum_145 = None
    view_719: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_262, [128, 48, 48]);  mul_262 = None
    transpose_91: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_44, 1, 2);  _unsafe_view_44 = None
    bmm_98: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_91, view_719);  transpose_91 = None
    transpose_92: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_45, 1, 2);  _unsafe_view_45 = None
    bmm_99: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_719, transpose_92);  view_719 = transpose_92 = None
    view_720: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_98, [8, 16, 784, 48]);  bmm_98 = None
    view_721: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_99, [8, 16, 48, 784]);  bmm_99 = None
    transpose_93: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_720, -2, -1);  view_720 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_150: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_206, expand_67)
    div_151: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_150, expand_67);  div_150 = None
    neg_24: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_93)
    mul_263: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_24, div_151);  neg_24 = div_151 = None
    div_152: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_93, expand_67);  transpose_93 = expand_67 = None
    sum_146: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [3], True);  mul_263 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_24: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_23, 1e-12);  linalg_vector_norm_23 = None
    where_24: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_24, sum_146, scalar_tensor_24);  ge_24 = sum_146 = scalar_tensor_24 = None
    detach_113: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_34);  detach_34 = None
    div_153: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_206, detach_113);  getitem_206 = None
    eq_24: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_113, 0);  detach_113 = None
    masked_fill_24: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_153, eq_24, 0);  div_153 = eq_24 = None
    mul_264: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_24, masked_fill_24);  where_24 = masked_fill_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_206: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_152, mul_264);  div_152 = mul_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_154: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_205, expand_66)
    div_155: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_154, expand_66);  div_154 = None
    neg_25: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_721)
    mul_265: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_25, div_155);  neg_25 = div_155 = None
    div_156: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_721, expand_66);  view_721 = expand_66 = None
    sum_147: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [3], True);  mul_265 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_25: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_22, 1e-12);  linalg_vector_norm_22 = None
    where_25: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_25, sum_147, scalar_tensor_25);  ge_25 = sum_147 = scalar_tensor_25 = None
    detach_114: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_33);  detach_33 = None
    div_157: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_205, detach_114);  getitem_205 = None
    eq_25: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_114, 0);  detach_114 = None
    masked_fill_25: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_157, eq_25, 0);  div_157 = eq_25 = None
    mul_266: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_25, masked_fill_25);  where_25 = masked_fill_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_207: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_156, mul_266);  div_156 = mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_14: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_207, add_206, view_716]);  add_207 = add_206 = view_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_161: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_14, [1, 4, 0, 2, 3]);  stack_14 = None
    clone_225: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    _unsafe_view_121: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_225, [8, 784, 2304]);  clone_225 = None
    view_722: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_121, [6272, 2304]);  _unsafe_view_121 = None
    t_365: "f32[2304, 768]" = torch.ops.aten.t.default(t_44);  t_44 = None
    mm_154: "f32[6272, 768]" = torch.ops.aten.mm.default(view_722, t_365);  t_365 = None
    t_366: "f32[2304, 6272]" = torch.ops.aten.t.default(view_722)
    mm_155: "f32[2304, 768]" = torch.ops.aten.mm.default(t_366, view_158);  t_366 = view_158 = None
    t_367: "f32[768, 2304]" = torch.ops.aten.t.default(mm_155);  mm_155 = None
    sum_148: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_722, [0], True);  view_722 = None
    view_723: "f32[2304]" = torch.ops.aten.view.default(sum_148, [2304]);  sum_148 = None
    t_368: "f32[2304, 768]" = torch.ops.aten.t.default(t_367);  t_367 = None
    view_724: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_154, [8, 784, 768]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_43 = torch.ops.aten.native_layer_norm_backward.default(view_724, add_60, [768], getitem_203, getitem_204, primals_333, primals_334, [True, True, True]);  view_724 = add_60 = getitem_203 = getitem_204 = primals_333 = primals_334 = None
    getitem_708: "f32[8, 784, 768]" = native_layer_norm_backward_43[0]
    getitem_709: "f32[768]" = native_layer_norm_backward_43[1]
    getitem_710: "f32[768]" = native_layer_norm_backward_43[2];  native_layer_norm_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_208: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_205, getitem_708);  add_205 = getitem_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_267: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_208, primals_44);  primals_44 = None
    mul_268: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_208, clone_88);  clone_88 = None
    sum_149: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 1], True);  mul_268 = None
    view_725: "f32[768]" = torch.ops.aten.view.default(sum_149, [768]);  sum_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_726: "f32[6272, 768]" = torch.ops.aten.view.default(mul_267, [6272, 768]);  mul_267 = None
    t_369: "f32[768, 3072]" = torch.ops.aten.t.default(t_43);  t_43 = None
    mm_156: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_726, t_369);  t_369 = None
    t_370: "f32[768, 6272]" = torch.ops.aten.t.default(view_726)
    mm_157: "f32[768, 3072]" = torch.ops.aten.mm.default(t_370, view_156);  t_370 = view_156 = None
    t_371: "f32[3072, 768]" = torch.ops.aten.t.default(mm_157);  mm_157 = None
    sum_150: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_726, [0], True);  view_726 = None
    view_727: "f32[768]" = torch.ops.aten.view.default(sum_150, [768]);  sum_150 = None
    t_372: "f32[768, 3072]" = torch.ops.aten.t.default(t_371);  t_371 = None
    view_728: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_156, [8, 784, 3072]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_28: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_728, view_155);  view_728 = view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_729: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_28, [6272, 3072]);  gelu_backward_28 = None
    t_373: "f32[3072, 768]" = torch.ops.aten.t.default(t_42);  t_42 = None
    mm_158: "f32[6272, 768]" = torch.ops.aten.mm.default(view_729, t_373);  t_373 = None
    t_374: "f32[3072, 6272]" = torch.ops.aten.t.default(view_729)
    mm_159: "f32[3072, 768]" = torch.ops.aten.mm.default(t_374, view_154);  t_374 = view_154 = None
    t_375: "f32[768, 3072]" = torch.ops.aten.t.default(mm_159);  mm_159 = None
    sum_151: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_729, [0], True);  view_729 = None
    view_730: "f32[3072]" = torch.ops.aten.view.default(sum_151, [3072]);  sum_151 = None
    t_376: "f32[3072, 768]" = torch.ops.aten.t.default(t_375);  t_375 = None
    view_731: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_158, [8, 784, 768]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_44 = torch.ops.aten.native_layer_norm_backward.default(view_731, add_59, [768], getitem_200, getitem_201, primals_327, primals_328, [True, True, True]);  view_731 = add_59 = getitem_200 = getitem_201 = primals_327 = primals_328 = None
    getitem_711: "f32[8, 784, 768]" = native_layer_norm_backward_44[0]
    getitem_712: "f32[768]" = native_layer_norm_backward_44[1]
    getitem_713: "f32[768]" = native_layer_norm_backward_44[2];  native_layer_norm_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_209: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_208, getitem_711);  add_208 = getitem_711 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_269: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_209, primals_43);  primals_43 = None
    mul_270: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_209, permute_45);  permute_45 = None
    sum_152: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_270, [0, 1], True);  mul_270 = None
    view_732: "f32[768]" = torch.ops.aten.view.default(sum_152, [768]);  sum_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_162: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_269, [0, 2, 1]);  mul_269 = None
    view_733: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_162, [8, 768, 28, 28]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(view_733, getitem_194, primals_325, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_733 = getitem_194 = primals_325 = None
    getitem_714: "f32[8, 768, 28, 28]" = convolution_backward_26[0]
    getitem_715: "f32[768, 1, 3, 3]" = convolution_backward_26[1]
    getitem_716: "f32[768]" = convolution_backward_26[2];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_13 = torch.ops.aten.native_batch_norm_backward.default(getitem_714, gelu_22, primals_323, getitem_197, getitem_198, getitem_195, getitem_196, True, 1e-05, [True, True, True]);  getitem_714 = gelu_22 = primals_323 = getitem_195 = getitem_196 = None
    getitem_717: "f32[8, 768, 28, 28]" = native_batch_norm_backward_13[0]
    getitem_718: "f32[768]" = native_batch_norm_backward_13[1]
    getitem_719: "f32[768]" = native_batch_norm_backward_13[2];  native_batch_norm_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_29: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_717, convolution_24);  getitem_717 = convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(gelu_backward_29, view_152, primals_321, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_29 = view_152 = primals_321 = None
    getitem_720: "f32[8, 768, 28, 28]" = convolution_backward_27[0]
    getitem_721: "f32[768, 1, 3, 3]" = convolution_backward_27[1]
    getitem_722: "f32[768]" = convolution_backward_27[2];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_734: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_720, [8, 768, 784]);  getitem_720 = None
    permute_163: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_734, [0, 2, 1]);  view_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_45 = torch.ops.aten.native_layer_norm_backward.default(permute_163, add_57, [768], getitem_192, getitem_193, primals_319, primals_320, [True, True, True]);  permute_163 = add_57 = getitem_192 = getitem_193 = primals_319 = primals_320 = None
    getitem_723: "f32[8, 784, 768]" = native_layer_norm_backward_45[0]
    getitem_724: "f32[768]" = native_layer_norm_backward_45[1]
    getitem_725: "f32[768]" = native_layer_norm_backward_45[2];  native_layer_norm_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_210: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_209, getitem_723);  add_209 = getitem_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_271: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_210, primals_41);  primals_41 = None
    mul_272: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_210, clone_86);  clone_86 = None
    sum_153: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1], True);  mul_272 = None
    view_735: "f32[768]" = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_154: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 1], True)
    view_736: "f32[768]" = torch.ops.aten.view.default(sum_154, [768]);  sum_154 = None
    view_737: "f32[6272, 768]" = torch.ops.aten.view.default(mul_271, [6272, 768]);  mul_271 = None
    t_377: "f32[768, 6272]" = torch.ops.aten.t.default(view_737)
    mm_160: "f32[768, 768]" = torch.ops.aten.mm.default(t_377, _unsafe_view_43);  t_377 = _unsafe_view_43 = None
    t_378: "f32[768, 768]" = torch.ops.aten.t.default(mm_160);  mm_160 = None
    t_379: "f32[768, 768]" = torch.ops.aten.t.default(t_41);  t_41 = None
    mm_161: "f32[6272, 768]" = torch.ops.aten.mm.default(view_737, t_379);  view_737 = t_379 = None
    view_738: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_161, [8, 784, 768]);  mm_161 = None
    t_380: "f32[768, 768]" = torch.ops.aten.t.default(t_378);  t_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_739: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_738, [8, 784, 16, 48]);  view_738 = None
    permute_164: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_739, [0, 2, 3, 1]);  view_739 = None
    clone_226: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    _unsafe_view_122: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_226, [128, 48, 784]);  clone_226 = None
    transpose_94: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_148, 1, 2);  view_148 = None
    bmm_100: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_94, _unsafe_view_122);  transpose_94 = None
    transpose_95: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_42, 1, 2);  _unsafe_view_42 = None
    bmm_101: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_122, transpose_95);  _unsafe_view_122 = transpose_95 = None
    view_740: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_100, [8, 16, 48, 784]);  bmm_100 = None
    view_741: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_101, [8, 16, 48, 48]);  bmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_115: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_32);  detach_32 = None
    _softmax_backward_data_13: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_741, detach_115, -1, torch.float32);  view_741 = detach_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_273: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_13, view_147);  view_147 = None
    mul_274: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_13, primals_42);  _softmax_backward_data_13 = primals_42 = None
    sum_155: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [0, 2, 3], True);  mul_273 = None
    view_742: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_155, [16, 1, 1]);  sum_155 = None
    view_743: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_274, [128, 48, 48]);  mul_274 = None
    transpose_96: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_40, 1, 2);  _unsafe_view_40 = None
    bmm_102: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_96, view_743);  transpose_96 = None
    transpose_97: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_41, 1, 2);  _unsafe_view_41 = None
    bmm_103: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_743, transpose_97);  view_743 = transpose_97 = None
    view_744: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_102, [8, 16, 784, 48]);  bmm_102 = None
    view_745: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_103, [8, 16, 48, 784]);  bmm_103 = None
    transpose_98: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_744, -2, -1);  view_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_158: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_189, expand_61)
    div_159: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_158, expand_61);  div_158 = None
    neg_26: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_98)
    mul_275: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_26, div_159);  neg_26 = div_159 = None
    div_160: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_98, expand_61);  transpose_98 = expand_61 = None
    sum_156: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [3], True);  mul_275 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_26: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_21, 1e-12);  linalg_vector_norm_21 = None
    where_26: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_26, sum_156, scalar_tensor_26);  ge_26 = sum_156 = scalar_tensor_26 = None
    detach_116: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_31);  detach_31 = None
    div_161: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_189, detach_116);  getitem_189 = None
    eq_26: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_116, 0);  detach_116 = None
    masked_fill_26: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_161, eq_26, 0);  div_161 = eq_26 = None
    mul_276: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_26, masked_fill_26);  where_26 = masked_fill_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_211: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_160, mul_276);  div_160 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_162: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_188, expand_60)
    div_163: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_162, expand_60);  div_162 = None
    neg_27: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_745)
    mul_277: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_27, div_163);  neg_27 = div_163 = None
    div_164: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_745, expand_60);  view_745 = expand_60 = None
    sum_157: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [3], True);  mul_277 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_27: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_20, 1e-12);  linalg_vector_norm_20 = None
    where_27: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_27, sum_157, scalar_tensor_27);  ge_27 = sum_157 = scalar_tensor_27 = None
    detach_117: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_30);  detach_30 = None
    div_165: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_188, detach_117);  getitem_188 = None
    eq_27: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_117, 0);  detach_117 = None
    masked_fill_27: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_165, eq_27, 0);  div_165 = eq_27 = None
    mul_278: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_27, masked_fill_27);  where_27 = masked_fill_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_212: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_164, mul_278);  div_164 = mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_15: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_212, add_211, view_740]);  add_212 = add_211 = view_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_165: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_15, [1, 4, 0, 2, 3]);  stack_15 = None
    clone_227: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
    _unsafe_view_123: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_227, [8, 784, 2304]);  clone_227 = None
    view_746: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_123, [6272, 2304]);  _unsafe_view_123 = None
    t_381: "f32[2304, 768]" = torch.ops.aten.t.default(t_40);  t_40 = None
    mm_162: "f32[6272, 768]" = torch.ops.aten.mm.default(view_746, t_381);  t_381 = None
    t_382: "f32[2304, 6272]" = torch.ops.aten.t.default(view_746)
    mm_163: "f32[2304, 768]" = torch.ops.aten.mm.default(t_382, view_144);  t_382 = view_144 = None
    t_383: "f32[768, 2304]" = torch.ops.aten.t.default(mm_163);  mm_163 = None
    sum_158: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_746, [0], True);  view_746 = None
    view_747: "f32[2304]" = torch.ops.aten.view.default(sum_158, [2304]);  sum_158 = None
    t_384: "f32[2304, 768]" = torch.ops.aten.t.default(t_383);  t_383 = None
    view_748: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_162, [8, 784, 768]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_46 = torch.ops.aten.native_layer_norm_backward.default(view_748, add_55, [768], getitem_186, getitem_187, primals_313, primals_314, [True, True, True]);  view_748 = add_55 = getitem_186 = getitem_187 = primals_313 = primals_314 = None
    getitem_726: "f32[8, 784, 768]" = native_layer_norm_backward_46[0]
    getitem_727: "f32[768]" = native_layer_norm_backward_46[1]
    getitem_728: "f32[768]" = native_layer_norm_backward_46[2];  native_layer_norm_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_213: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_210, getitem_726);  add_210 = getitem_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_279: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_213, primals_40);  primals_40 = None
    mul_280: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_213, clone_80);  clone_80 = None
    sum_159: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 1], True);  mul_280 = None
    view_749: "f32[768]" = torch.ops.aten.view.default(sum_159, [768]);  sum_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_750: "f32[6272, 768]" = torch.ops.aten.view.default(mul_279, [6272, 768]);  mul_279 = None
    t_385: "f32[768, 3072]" = torch.ops.aten.t.default(t_39);  t_39 = None
    mm_164: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_750, t_385);  t_385 = None
    t_386: "f32[768, 6272]" = torch.ops.aten.t.default(view_750)
    mm_165: "f32[768, 3072]" = torch.ops.aten.mm.default(t_386, view_142);  t_386 = view_142 = None
    t_387: "f32[3072, 768]" = torch.ops.aten.t.default(mm_165);  mm_165 = None
    sum_160: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_750, [0], True);  view_750 = None
    view_751: "f32[768]" = torch.ops.aten.view.default(sum_160, [768]);  sum_160 = None
    t_388: "f32[768, 3072]" = torch.ops.aten.t.default(t_387);  t_387 = None
    view_752: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_164, [8, 784, 3072]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_30: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_752, view_141);  view_752 = view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_753: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_30, [6272, 3072]);  gelu_backward_30 = None
    t_389: "f32[3072, 768]" = torch.ops.aten.t.default(t_38);  t_38 = None
    mm_166: "f32[6272, 768]" = torch.ops.aten.mm.default(view_753, t_389);  t_389 = None
    t_390: "f32[3072, 6272]" = torch.ops.aten.t.default(view_753)
    mm_167: "f32[3072, 768]" = torch.ops.aten.mm.default(t_390, view_140);  t_390 = view_140 = None
    t_391: "f32[768, 3072]" = torch.ops.aten.t.default(mm_167);  mm_167 = None
    sum_161: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_753, [0], True);  view_753 = None
    view_754: "f32[3072]" = torch.ops.aten.view.default(sum_161, [3072]);  sum_161 = None
    t_392: "f32[3072, 768]" = torch.ops.aten.t.default(t_391);  t_391 = None
    view_755: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_166, [8, 784, 768]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_47 = torch.ops.aten.native_layer_norm_backward.default(view_755, add_54, [768], getitem_183, getitem_184, primals_307, primals_308, [True, True, True]);  view_755 = add_54 = getitem_183 = getitem_184 = primals_307 = primals_308 = None
    getitem_729: "f32[8, 784, 768]" = native_layer_norm_backward_47[0]
    getitem_730: "f32[768]" = native_layer_norm_backward_47[1]
    getitem_731: "f32[768]" = native_layer_norm_backward_47[2];  native_layer_norm_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_214: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_213, getitem_729);  add_213 = getitem_729 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_281: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_214, primals_39);  primals_39 = None
    mul_282: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_214, permute_41);  permute_41 = None
    sum_162: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 1], True);  mul_282 = None
    view_756: "f32[768]" = torch.ops.aten.view.default(sum_162, [768]);  sum_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_166: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_281, [0, 2, 1]);  mul_281 = None
    view_757: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_166, [8, 768, 28, 28]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(view_757, getitem_177, primals_305, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_757 = getitem_177 = primals_305 = None
    getitem_732: "f32[8, 768, 28, 28]" = convolution_backward_28[0]
    getitem_733: "f32[768, 1, 3, 3]" = convolution_backward_28[1]
    getitem_734: "f32[768]" = convolution_backward_28[2];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_14 = torch.ops.aten.native_batch_norm_backward.default(getitem_732, gelu_20, primals_303, getitem_180, getitem_181, getitem_178, getitem_179, True, 1e-05, [True, True, True]);  getitem_732 = gelu_20 = primals_303 = getitem_178 = getitem_179 = None
    getitem_735: "f32[8, 768, 28, 28]" = native_batch_norm_backward_14[0]
    getitem_736: "f32[768]" = native_batch_norm_backward_14[1]
    getitem_737: "f32[768]" = native_batch_norm_backward_14[2];  native_batch_norm_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_31: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_735, convolution_22);  getitem_735 = convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(gelu_backward_31, view_138, primals_301, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_31 = view_138 = primals_301 = None
    getitem_738: "f32[8, 768, 28, 28]" = convolution_backward_29[0]
    getitem_739: "f32[768, 1, 3, 3]" = convolution_backward_29[1]
    getitem_740: "f32[768]" = convolution_backward_29[2];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_758: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_738, [8, 768, 784]);  getitem_738 = None
    permute_167: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_758, [0, 2, 1]);  view_758 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_48 = torch.ops.aten.native_layer_norm_backward.default(permute_167, add_52, [768], getitem_175, getitem_176, primals_299, primals_300, [True, True, True]);  permute_167 = add_52 = getitem_175 = getitem_176 = primals_299 = primals_300 = None
    getitem_741: "f32[8, 784, 768]" = native_layer_norm_backward_48[0]
    getitem_742: "f32[768]" = native_layer_norm_backward_48[1]
    getitem_743: "f32[768]" = native_layer_norm_backward_48[2];  native_layer_norm_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_215: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_214, getitem_741);  add_214 = getitem_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_283: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_215, primals_37);  primals_37 = None
    mul_284: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_215, clone_78);  clone_78 = None
    sum_163: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_284, [0, 1], True);  mul_284 = None
    view_759: "f32[768]" = torch.ops.aten.view.default(sum_163, [768]);  sum_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_164: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1], True)
    view_760: "f32[768]" = torch.ops.aten.view.default(sum_164, [768]);  sum_164 = None
    view_761: "f32[6272, 768]" = torch.ops.aten.view.default(mul_283, [6272, 768]);  mul_283 = None
    t_393: "f32[768, 6272]" = torch.ops.aten.t.default(view_761)
    mm_168: "f32[768, 768]" = torch.ops.aten.mm.default(t_393, _unsafe_view_39);  t_393 = _unsafe_view_39 = None
    t_394: "f32[768, 768]" = torch.ops.aten.t.default(mm_168);  mm_168 = None
    t_395: "f32[768, 768]" = torch.ops.aten.t.default(t_37);  t_37 = None
    mm_169: "f32[6272, 768]" = torch.ops.aten.mm.default(view_761, t_395);  view_761 = t_395 = None
    view_762: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_169, [8, 784, 768]);  mm_169 = None
    t_396: "f32[768, 768]" = torch.ops.aten.t.default(t_394);  t_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_763: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_762, [8, 784, 16, 48]);  view_762 = None
    permute_168: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_763, [0, 2, 3, 1]);  view_763 = None
    clone_228: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    _unsafe_view_124: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_228, [128, 48, 784]);  clone_228 = None
    transpose_99: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_134, 1, 2);  view_134 = None
    bmm_104: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_99, _unsafe_view_124);  transpose_99 = None
    transpose_100: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_38, 1, 2);  _unsafe_view_38 = None
    bmm_105: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_124, transpose_100);  _unsafe_view_124 = transpose_100 = None
    view_764: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_104, [8, 16, 48, 784]);  bmm_104 = None
    view_765: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_105, [8, 16, 48, 48]);  bmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_118: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_29);  detach_29 = None
    _softmax_backward_data_14: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_765, detach_118, -1, torch.float32);  view_765 = detach_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_285: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_14, view_133);  view_133 = None
    mul_286: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_14, primals_38);  _softmax_backward_data_14 = primals_38 = None
    sum_165: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 2, 3], True);  mul_285 = None
    view_766: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_165, [16, 1, 1]);  sum_165 = None
    view_767: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_286, [128, 48, 48]);  mul_286 = None
    transpose_101: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_36, 1, 2);  _unsafe_view_36 = None
    bmm_106: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_101, view_767);  transpose_101 = None
    transpose_102: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_37, 1, 2);  _unsafe_view_37 = None
    bmm_107: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_767, transpose_102);  view_767 = transpose_102 = None
    view_768: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_106, [8, 16, 784, 48]);  bmm_106 = None
    view_769: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_107, [8, 16, 48, 784]);  bmm_107 = None
    transpose_103: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_768, -2, -1);  view_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_166: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_172, expand_55)
    div_167: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_166, expand_55);  div_166 = None
    neg_28: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_103)
    mul_287: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_28, div_167);  neg_28 = div_167 = None
    div_168: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_103, expand_55);  transpose_103 = expand_55 = None
    sum_166: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [3], True);  mul_287 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_28: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_19, 1e-12);  linalg_vector_norm_19 = None
    where_28: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_28, sum_166, scalar_tensor_28);  ge_28 = sum_166 = scalar_tensor_28 = None
    detach_119: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_28);  detach_28 = None
    div_169: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_172, detach_119);  getitem_172 = None
    eq_28: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_119, 0);  detach_119 = None
    masked_fill_28: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_169, eq_28, 0);  div_169 = eq_28 = None
    mul_288: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_28, masked_fill_28);  where_28 = masked_fill_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_216: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_168, mul_288);  div_168 = mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_170: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_171, expand_54)
    div_171: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_170, expand_54);  div_170 = None
    neg_29: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_769)
    mul_289: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_29, div_171);  neg_29 = div_171 = None
    div_172: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_769, expand_54);  view_769 = expand_54 = None
    sum_167: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_289, [3], True);  mul_289 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_29: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_18, 1e-12);  linalg_vector_norm_18 = None
    where_29: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_29, sum_167, scalar_tensor_29);  ge_29 = sum_167 = scalar_tensor_29 = None
    detach_120: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_27);  detach_27 = None
    div_173: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_171, detach_120);  getitem_171 = None
    eq_29: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_120, 0);  detach_120 = None
    masked_fill_29: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_173, eq_29, 0);  div_173 = eq_29 = None
    mul_290: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_29, masked_fill_29);  where_29 = masked_fill_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_217: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_172, mul_290);  div_172 = mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_16: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_217, add_216, view_764]);  add_217 = add_216 = view_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_169: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_16, [1, 4, 0, 2, 3]);  stack_16 = None
    clone_229: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
    _unsafe_view_125: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_229, [8, 784, 2304]);  clone_229 = None
    view_770: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_125, [6272, 2304]);  _unsafe_view_125 = None
    t_397: "f32[2304, 768]" = torch.ops.aten.t.default(t_36);  t_36 = None
    mm_170: "f32[6272, 768]" = torch.ops.aten.mm.default(view_770, t_397);  t_397 = None
    t_398: "f32[2304, 6272]" = torch.ops.aten.t.default(view_770)
    mm_171: "f32[2304, 768]" = torch.ops.aten.mm.default(t_398, view_130);  t_398 = view_130 = None
    t_399: "f32[768, 2304]" = torch.ops.aten.t.default(mm_171);  mm_171 = None
    sum_168: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_770, [0], True);  view_770 = None
    view_771: "f32[2304]" = torch.ops.aten.view.default(sum_168, [2304]);  sum_168 = None
    t_400: "f32[2304, 768]" = torch.ops.aten.t.default(t_399);  t_399 = None
    view_772: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_170, [8, 784, 768]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_49 = torch.ops.aten.native_layer_norm_backward.default(view_772, add_50, [768], getitem_169, getitem_170, primals_293, primals_294, [True, True, True]);  view_772 = add_50 = getitem_169 = getitem_170 = primals_293 = primals_294 = None
    getitem_744: "f32[8, 784, 768]" = native_layer_norm_backward_49[0]
    getitem_745: "f32[768]" = native_layer_norm_backward_49[1]
    getitem_746: "f32[768]" = native_layer_norm_backward_49[2];  native_layer_norm_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_218: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_215, getitem_744);  add_215 = getitem_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_291: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_218, primals_36);  primals_36 = None
    mul_292: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_218, clone_72);  clone_72 = None
    sum_169: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 1], True);  mul_292 = None
    view_773: "f32[768]" = torch.ops.aten.view.default(sum_169, [768]);  sum_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_774: "f32[6272, 768]" = torch.ops.aten.view.default(mul_291, [6272, 768]);  mul_291 = None
    t_401: "f32[768, 3072]" = torch.ops.aten.t.default(t_35);  t_35 = None
    mm_172: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_774, t_401);  t_401 = None
    t_402: "f32[768, 6272]" = torch.ops.aten.t.default(view_774)
    mm_173: "f32[768, 3072]" = torch.ops.aten.mm.default(t_402, view_128);  t_402 = view_128 = None
    t_403: "f32[3072, 768]" = torch.ops.aten.t.default(mm_173);  mm_173 = None
    sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_774, [0], True);  view_774 = None
    view_775: "f32[768]" = torch.ops.aten.view.default(sum_170, [768]);  sum_170 = None
    t_404: "f32[768, 3072]" = torch.ops.aten.t.default(t_403);  t_403 = None
    view_776: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_172, [8, 784, 3072]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_32: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_776, view_127);  view_776 = view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_777: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_32, [6272, 3072]);  gelu_backward_32 = None
    t_405: "f32[3072, 768]" = torch.ops.aten.t.default(t_34);  t_34 = None
    mm_174: "f32[6272, 768]" = torch.ops.aten.mm.default(view_777, t_405);  t_405 = None
    t_406: "f32[3072, 6272]" = torch.ops.aten.t.default(view_777)
    mm_175: "f32[3072, 768]" = torch.ops.aten.mm.default(t_406, view_126);  t_406 = view_126 = None
    t_407: "f32[768, 3072]" = torch.ops.aten.t.default(mm_175);  mm_175 = None
    sum_171: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_777, [0], True);  view_777 = None
    view_778: "f32[3072]" = torch.ops.aten.view.default(sum_171, [3072]);  sum_171 = None
    t_408: "f32[3072, 768]" = torch.ops.aten.t.default(t_407);  t_407 = None
    view_779: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_174, [8, 784, 768]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_50 = torch.ops.aten.native_layer_norm_backward.default(view_779, add_49, [768], getitem_166, getitem_167, primals_287, primals_288, [True, True, True]);  view_779 = add_49 = getitem_166 = getitem_167 = primals_287 = primals_288 = None
    getitem_747: "f32[8, 784, 768]" = native_layer_norm_backward_50[0]
    getitem_748: "f32[768]" = native_layer_norm_backward_50[1]
    getitem_749: "f32[768]" = native_layer_norm_backward_50[2];  native_layer_norm_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_219: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_218, getitem_747);  add_218 = getitem_747 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_293: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_219, primals_35);  primals_35 = None
    mul_294: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_219, permute_37);  permute_37 = None
    sum_172: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_294, [0, 1], True);  mul_294 = None
    view_780: "f32[768]" = torch.ops.aten.view.default(sum_172, [768]);  sum_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_170: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_293, [0, 2, 1]);  mul_293 = None
    view_781: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_170, [8, 768, 28, 28]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(view_781, getitem_160, primals_285, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_781 = getitem_160 = primals_285 = None
    getitem_750: "f32[8, 768, 28, 28]" = convolution_backward_30[0]
    getitem_751: "f32[768, 1, 3, 3]" = convolution_backward_30[1]
    getitem_752: "f32[768]" = convolution_backward_30[2];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_15 = torch.ops.aten.native_batch_norm_backward.default(getitem_750, gelu_18, primals_283, getitem_163, getitem_164, getitem_161, getitem_162, True, 1e-05, [True, True, True]);  getitem_750 = gelu_18 = primals_283 = getitem_161 = getitem_162 = None
    getitem_753: "f32[8, 768, 28, 28]" = native_batch_norm_backward_15[0]
    getitem_754: "f32[768]" = native_batch_norm_backward_15[1]
    getitem_755: "f32[768]" = native_batch_norm_backward_15[2];  native_batch_norm_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_33: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_753, convolution_20);  getitem_753 = convolution_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(gelu_backward_33, view_124, primals_281, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_33 = view_124 = primals_281 = None
    getitem_756: "f32[8, 768, 28, 28]" = convolution_backward_31[0]
    getitem_757: "f32[768, 1, 3, 3]" = convolution_backward_31[1]
    getitem_758: "f32[768]" = convolution_backward_31[2];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_782: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_756, [8, 768, 784]);  getitem_756 = None
    permute_171: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_782, [0, 2, 1]);  view_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_51 = torch.ops.aten.native_layer_norm_backward.default(permute_171, add_47, [768], getitem_158, getitem_159, primals_279, primals_280, [True, True, True]);  permute_171 = add_47 = getitem_158 = getitem_159 = primals_279 = primals_280 = None
    getitem_759: "f32[8, 784, 768]" = native_layer_norm_backward_51[0]
    getitem_760: "f32[768]" = native_layer_norm_backward_51[1]
    getitem_761: "f32[768]" = native_layer_norm_backward_51[2];  native_layer_norm_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_220: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_219, getitem_759);  add_219 = getitem_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_295: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_220, primals_33);  primals_33 = None
    mul_296: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_220, clone_70);  clone_70 = None
    sum_173: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 1], True);  mul_296 = None
    view_783: "f32[768]" = torch.ops.aten.view.default(sum_173, [768]);  sum_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_174: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1], True)
    view_784: "f32[768]" = torch.ops.aten.view.default(sum_174, [768]);  sum_174 = None
    view_785: "f32[6272, 768]" = torch.ops.aten.view.default(mul_295, [6272, 768]);  mul_295 = None
    t_409: "f32[768, 6272]" = torch.ops.aten.t.default(view_785)
    mm_176: "f32[768, 768]" = torch.ops.aten.mm.default(t_409, _unsafe_view_35);  t_409 = _unsafe_view_35 = None
    t_410: "f32[768, 768]" = torch.ops.aten.t.default(mm_176);  mm_176 = None
    t_411: "f32[768, 768]" = torch.ops.aten.t.default(t_33);  t_33 = None
    mm_177: "f32[6272, 768]" = torch.ops.aten.mm.default(view_785, t_411);  view_785 = t_411 = None
    view_786: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_177, [8, 784, 768]);  mm_177 = None
    t_412: "f32[768, 768]" = torch.ops.aten.t.default(t_410);  t_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_787: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_786, [8, 784, 16, 48]);  view_786 = None
    permute_172: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_787, [0, 2, 3, 1]);  view_787 = None
    clone_230: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    _unsafe_view_126: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_230, [128, 48, 784]);  clone_230 = None
    transpose_104: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_120, 1, 2);  view_120 = None
    bmm_108: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_104, _unsafe_view_126);  transpose_104 = None
    transpose_105: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_34, 1, 2);  _unsafe_view_34 = None
    bmm_109: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_126, transpose_105);  _unsafe_view_126 = transpose_105 = None
    view_788: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_108, [8, 16, 48, 784]);  bmm_108 = None
    view_789: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_109, [8, 16, 48, 48]);  bmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_121: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_26);  detach_26 = None
    _softmax_backward_data_15: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_789, detach_121, -1, torch.float32);  view_789 = detach_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_297: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_15, view_119);  view_119 = None
    mul_298: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_15, primals_34);  _softmax_backward_data_15 = primals_34 = None
    sum_175: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 2, 3], True);  mul_297 = None
    view_790: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_175, [16, 1, 1]);  sum_175 = None
    view_791: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_298, [128, 48, 48]);  mul_298 = None
    transpose_106: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_32, 1, 2);  _unsafe_view_32 = None
    bmm_110: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_106, view_791);  transpose_106 = None
    transpose_107: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_33, 1, 2);  _unsafe_view_33 = None
    bmm_111: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_791, transpose_107);  view_791 = transpose_107 = None
    view_792: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_110, [8, 16, 784, 48]);  bmm_110 = None
    view_793: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_111, [8, 16, 48, 784]);  bmm_111 = None
    transpose_108: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_792, -2, -1);  view_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_174: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_155, expand_49)
    div_175: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_174, expand_49);  div_174 = None
    neg_30: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_108)
    mul_299: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_30, div_175);  neg_30 = div_175 = None
    div_176: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_108, expand_49);  transpose_108 = expand_49 = None
    sum_176: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [3], True);  mul_299 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_30: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_17, 1e-12);  linalg_vector_norm_17 = None
    where_30: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_30, sum_176, scalar_tensor_30);  ge_30 = sum_176 = scalar_tensor_30 = None
    detach_122: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_25);  detach_25 = None
    div_177: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_155, detach_122);  getitem_155 = None
    eq_30: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_122, 0);  detach_122 = None
    masked_fill_30: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_177, eq_30, 0);  div_177 = eq_30 = None
    mul_300: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_30, masked_fill_30);  where_30 = masked_fill_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_221: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_176, mul_300);  div_176 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_178: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_154, expand_48)
    div_179: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_178, expand_48);  div_178 = None
    neg_31: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_793)
    mul_301: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_31, div_179);  neg_31 = div_179 = None
    div_180: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_793, expand_48);  view_793 = expand_48 = None
    sum_177: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [3], True);  mul_301 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_31: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_16, 1e-12);  linalg_vector_norm_16 = None
    where_31: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_31, sum_177, scalar_tensor_31);  ge_31 = sum_177 = scalar_tensor_31 = None
    detach_123: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_24);  detach_24 = None
    div_181: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_154, detach_123);  getitem_154 = None
    eq_31: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_123, 0);  detach_123 = None
    masked_fill_31: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_181, eq_31, 0);  div_181 = eq_31 = None
    mul_302: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_31, masked_fill_31);  where_31 = masked_fill_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_222: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_180, mul_302);  div_180 = mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_17: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_222, add_221, view_788]);  add_222 = add_221 = view_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_173: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_17, [1, 4, 0, 2, 3]);  stack_17 = None
    clone_231: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
    _unsafe_view_127: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_231, [8, 784, 2304]);  clone_231 = None
    view_794: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_127, [6272, 2304]);  _unsafe_view_127 = None
    t_413: "f32[2304, 768]" = torch.ops.aten.t.default(t_32);  t_32 = None
    mm_178: "f32[6272, 768]" = torch.ops.aten.mm.default(view_794, t_413);  t_413 = None
    t_414: "f32[2304, 6272]" = torch.ops.aten.t.default(view_794)
    mm_179: "f32[2304, 768]" = torch.ops.aten.mm.default(t_414, view_116);  t_414 = view_116 = None
    t_415: "f32[768, 2304]" = torch.ops.aten.t.default(mm_179);  mm_179 = None
    sum_178: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_794, [0], True);  view_794 = None
    view_795: "f32[2304]" = torch.ops.aten.view.default(sum_178, [2304]);  sum_178 = None
    t_416: "f32[2304, 768]" = torch.ops.aten.t.default(t_415);  t_415 = None
    view_796: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_178, [8, 784, 768]);  mm_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_52 = torch.ops.aten.native_layer_norm_backward.default(view_796, add_45, [768], getitem_152, getitem_153, primals_273, primals_274, [True, True, True]);  view_796 = add_45 = getitem_152 = getitem_153 = primals_273 = primals_274 = None
    getitem_762: "f32[8, 784, 768]" = native_layer_norm_backward_52[0]
    getitem_763: "f32[768]" = native_layer_norm_backward_52[1]
    getitem_764: "f32[768]" = native_layer_norm_backward_52[2];  native_layer_norm_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_223: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_220, getitem_762);  add_220 = getitem_762 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_303: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_223, primals_32);  primals_32 = None
    mul_304: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_223, clone_64);  clone_64 = None
    sum_179: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1], True);  mul_304 = None
    view_797: "f32[768]" = torch.ops.aten.view.default(sum_179, [768]);  sum_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_798: "f32[6272, 768]" = torch.ops.aten.view.default(mul_303, [6272, 768]);  mul_303 = None
    t_417: "f32[768, 3072]" = torch.ops.aten.t.default(t_31);  t_31 = None
    mm_180: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_798, t_417);  t_417 = None
    t_418: "f32[768, 6272]" = torch.ops.aten.t.default(view_798)
    mm_181: "f32[768, 3072]" = torch.ops.aten.mm.default(t_418, view_114);  t_418 = view_114 = None
    t_419: "f32[3072, 768]" = torch.ops.aten.t.default(mm_181);  mm_181 = None
    sum_180: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_798, [0], True);  view_798 = None
    view_799: "f32[768]" = torch.ops.aten.view.default(sum_180, [768]);  sum_180 = None
    t_420: "f32[768, 3072]" = torch.ops.aten.t.default(t_419);  t_419 = None
    view_800: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_180, [8, 784, 3072]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_34: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_800, view_113);  view_800 = view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_801: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_34, [6272, 3072]);  gelu_backward_34 = None
    t_421: "f32[3072, 768]" = torch.ops.aten.t.default(t_30);  t_30 = None
    mm_182: "f32[6272, 768]" = torch.ops.aten.mm.default(view_801, t_421);  t_421 = None
    t_422: "f32[3072, 6272]" = torch.ops.aten.t.default(view_801)
    mm_183: "f32[3072, 768]" = torch.ops.aten.mm.default(t_422, view_112);  t_422 = view_112 = None
    t_423: "f32[768, 3072]" = torch.ops.aten.t.default(mm_183);  mm_183 = None
    sum_181: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_801, [0], True);  view_801 = None
    view_802: "f32[3072]" = torch.ops.aten.view.default(sum_181, [3072]);  sum_181 = None
    t_424: "f32[3072, 768]" = torch.ops.aten.t.default(t_423);  t_423 = None
    view_803: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_182, [8, 784, 768]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_53 = torch.ops.aten.native_layer_norm_backward.default(view_803, add_44, [768], getitem_149, getitem_150, primals_267, primals_268, [True, True, True]);  view_803 = add_44 = getitem_149 = getitem_150 = primals_267 = primals_268 = None
    getitem_765: "f32[8, 784, 768]" = native_layer_norm_backward_53[0]
    getitem_766: "f32[768]" = native_layer_norm_backward_53[1]
    getitem_767: "f32[768]" = native_layer_norm_backward_53[2];  native_layer_norm_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_224: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_223, getitem_765);  add_223 = getitem_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_305: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_224, primals_31);  primals_31 = None
    mul_306: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_224, permute_33);  permute_33 = None
    sum_182: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_306, [0, 1], True);  mul_306 = None
    view_804: "f32[768]" = torch.ops.aten.view.default(sum_182, [768]);  sum_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_174: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_305, [0, 2, 1]);  mul_305 = None
    view_805: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_174, [8, 768, 28, 28]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(view_805, getitem_143, primals_265, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_805 = getitem_143 = primals_265 = None
    getitem_768: "f32[8, 768, 28, 28]" = convolution_backward_32[0]
    getitem_769: "f32[768, 1, 3, 3]" = convolution_backward_32[1]
    getitem_770: "f32[768]" = convolution_backward_32[2];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_16 = torch.ops.aten.native_batch_norm_backward.default(getitem_768, gelu_16, primals_263, getitem_146, getitem_147, getitem_144, getitem_145, True, 1e-05, [True, True, True]);  getitem_768 = gelu_16 = primals_263 = getitem_144 = getitem_145 = None
    getitem_771: "f32[8, 768, 28, 28]" = native_batch_norm_backward_16[0]
    getitem_772: "f32[768]" = native_batch_norm_backward_16[1]
    getitem_773: "f32[768]" = native_batch_norm_backward_16[2];  native_batch_norm_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_35: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_771, convolution_18);  getitem_771 = convolution_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(gelu_backward_35, view_110, primals_261, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_35 = view_110 = primals_261 = None
    getitem_774: "f32[8, 768, 28, 28]" = convolution_backward_33[0]
    getitem_775: "f32[768, 1, 3, 3]" = convolution_backward_33[1]
    getitem_776: "f32[768]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_806: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_774, [8, 768, 784]);  getitem_774 = None
    permute_175: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_806, [0, 2, 1]);  view_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_54 = torch.ops.aten.native_layer_norm_backward.default(permute_175, add_42, [768], getitem_141, getitem_142, primals_259, primals_260, [True, True, True]);  permute_175 = add_42 = getitem_141 = getitem_142 = primals_259 = primals_260 = None
    getitem_777: "f32[8, 784, 768]" = native_layer_norm_backward_54[0]
    getitem_778: "f32[768]" = native_layer_norm_backward_54[1]
    getitem_779: "f32[768]" = native_layer_norm_backward_54[2];  native_layer_norm_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_225: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_224, getitem_777);  add_224 = getitem_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_307: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_225, primals_29);  primals_29 = None
    mul_308: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_225, clone_62);  clone_62 = None
    sum_183: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 1], True);  mul_308 = None
    view_807: "f32[768]" = torch.ops.aten.view.default(sum_183, [768]);  sum_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_184: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 1], True)
    view_808: "f32[768]" = torch.ops.aten.view.default(sum_184, [768]);  sum_184 = None
    view_809: "f32[6272, 768]" = torch.ops.aten.view.default(mul_307, [6272, 768]);  mul_307 = None
    t_425: "f32[768, 6272]" = torch.ops.aten.t.default(view_809)
    mm_184: "f32[768, 768]" = torch.ops.aten.mm.default(t_425, _unsafe_view_31);  t_425 = _unsafe_view_31 = None
    t_426: "f32[768, 768]" = torch.ops.aten.t.default(mm_184);  mm_184 = None
    t_427: "f32[768, 768]" = torch.ops.aten.t.default(t_29);  t_29 = None
    mm_185: "f32[6272, 768]" = torch.ops.aten.mm.default(view_809, t_427);  view_809 = t_427 = None
    view_810: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_185, [8, 784, 768]);  mm_185 = None
    t_428: "f32[768, 768]" = torch.ops.aten.t.default(t_426);  t_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_811: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_810, [8, 784, 16, 48]);  view_810 = None
    permute_176: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_811, [0, 2, 3, 1]);  view_811 = None
    clone_232: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
    _unsafe_view_128: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_232, [128, 48, 784]);  clone_232 = None
    transpose_109: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_106, 1, 2);  view_106 = None
    bmm_112: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_109, _unsafe_view_128);  transpose_109 = None
    transpose_110: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_30, 1, 2);  _unsafe_view_30 = None
    bmm_113: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_128, transpose_110);  _unsafe_view_128 = transpose_110 = None
    view_812: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_112, [8, 16, 48, 784]);  bmm_112 = None
    view_813: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_113, [8, 16, 48, 48]);  bmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_124: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_23);  detach_23 = None
    _softmax_backward_data_16: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_813, detach_124, -1, torch.float32);  view_813 = detach_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_309: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_16, view_105);  view_105 = None
    mul_310: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_16, primals_30);  _softmax_backward_data_16 = primals_30 = None
    sum_185: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 2, 3], True);  mul_309 = None
    view_814: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_185, [16, 1, 1]);  sum_185 = None
    view_815: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_310, [128, 48, 48]);  mul_310 = None
    transpose_111: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_28, 1, 2);  _unsafe_view_28 = None
    bmm_114: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_111, view_815);  transpose_111 = None
    transpose_112: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_29, 1, 2);  _unsafe_view_29 = None
    bmm_115: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_815, transpose_112);  view_815 = transpose_112 = None
    view_816: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_114, [8, 16, 784, 48]);  bmm_114 = None
    view_817: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_115, [8, 16, 48, 784]);  bmm_115 = None
    transpose_113: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_816, -2, -1);  view_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_182: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_138, expand_43)
    div_183: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_182, expand_43);  div_182 = None
    neg_32: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_113)
    mul_311: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_32, div_183);  neg_32 = div_183 = None
    div_184: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_113, expand_43);  transpose_113 = expand_43 = None
    sum_186: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [3], True);  mul_311 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_32: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_15, 1e-12);  linalg_vector_norm_15 = None
    where_32: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_32, sum_186, scalar_tensor_32);  ge_32 = sum_186 = scalar_tensor_32 = None
    detach_125: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_22);  detach_22 = None
    div_185: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_138, detach_125);  getitem_138 = None
    eq_32: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_125, 0);  detach_125 = None
    masked_fill_32: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_185, eq_32, 0);  div_185 = eq_32 = None
    mul_312: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_32, masked_fill_32);  where_32 = masked_fill_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_226: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_184, mul_312);  div_184 = mul_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_186: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_137, expand_42)
    div_187: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_186, expand_42);  div_186 = None
    neg_33: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_817)
    mul_313: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_33, div_187);  neg_33 = div_187 = None
    div_188: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_817, expand_42);  view_817 = expand_42 = None
    sum_187: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [3], True);  mul_313 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_33: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_14, 1e-12);  linalg_vector_norm_14 = None
    where_33: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_33, sum_187, scalar_tensor_33);  ge_33 = sum_187 = scalar_tensor_33 = None
    detach_126: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_21);  detach_21 = None
    div_189: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_137, detach_126);  getitem_137 = None
    eq_33: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_126, 0);  detach_126 = None
    masked_fill_33: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_189, eq_33, 0);  div_189 = eq_33 = None
    mul_314: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_33, masked_fill_33);  where_33 = masked_fill_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_227: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_188, mul_314);  div_188 = mul_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_18: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_227, add_226, view_812]);  add_227 = add_226 = view_812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_177: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_18, [1, 4, 0, 2, 3]);  stack_18 = None
    clone_233: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    _unsafe_view_129: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_233, [8, 784, 2304]);  clone_233 = None
    view_818: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_129, [6272, 2304]);  _unsafe_view_129 = None
    t_429: "f32[2304, 768]" = torch.ops.aten.t.default(t_28);  t_28 = None
    mm_186: "f32[6272, 768]" = torch.ops.aten.mm.default(view_818, t_429);  t_429 = None
    t_430: "f32[2304, 6272]" = torch.ops.aten.t.default(view_818)
    mm_187: "f32[2304, 768]" = torch.ops.aten.mm.default(t_430, view_102);  t_430 = view_102 = None
    t_431: "f32[768, 2304]" = torch.ops.aten.t.default(mm_187);  mm_187 = None
    sum_188: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
    view_819: "f32[2304]" = torch.ops.aten.view.default(sum_188, [2304]);  sum_188 = None
    t_432: "f32[2304, 768]" = torch.ops.aten.t.default(t_431);  t_431 = None
    view_820: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_186, [8, 784, 768]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_55 = torch.ops.aten.native_layer_norm_backward.default(view_820, add_40, [768], getitem_135, getitem_136, primals_253, primals_254, [True, True, True]);  view_820 = add_40 = getitem_135 = getitem_136 = primals_253 = primals_254 = None
    getitem_780: "f32[8, 784, 768]" = native_layer_norm_backward_55[0]
    getitem_781: "f32[768]" = native_layer_norm_backward_55[1]
    getitem_782: "f32[768]" = native_layer_norm_backward_55[2];  native_layer_norm_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_228: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_225, getitem_780);  add_225 = getitem_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_315: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_228, primals_28);  primals_28 = None
    mul_316: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_228, clone_56);  clone_56 = None
    sum_189: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1], True);  mul_316 = None
    view_821: "f32[768]" = torch.ops.aten.view.default(sum_189, [768]);  sum_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_822: "f32[6272, 768]" = torch.ops.aten.view.default(mul_315, [6272, 768]);  mul_315 = None
    t_433: "f32[768, 3072]" = torch.ops.aten.t.default(t_27);  t_27 = None
    mm_188: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_822, t_433);  t_433 = None
    t_434: "f32[768, 6272]" = torch.ops.aten.t.default(view_822)
    mm_189: "f32[768, 3072]" = torch.ops.aten.mm.default(t_434, view_100);  t_434 = view_100 = None
    t_435: "f32[3072, 768]" = torch.ops.aten.t.default(mm_189);  mm_189 = None
    sum_190: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_822, [0], True);  view_822 = None
    view_823: "f32[768]" = torch.ops.aten.view.default(sum_190, [768]);  sum_190 = None
    t_436: "f32[768, 3072]" = torch.ops.aten.t.default(t_435);  t_435 = None
    view_824: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_188, [8, 784, 3072]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_36: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_824, view_99);  view_824 = view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_825: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_36, [6272, 3072]);  gelu_backward_36 = None
    t_437: "f32[3072, 768]" = torch.ops.aten.t.default(t_26);  t_26 = None
    mm_190: "f32[6272, 768]" = torch.ops.aten.mm.default(view_825, t_437);  t_437 = None
    t_438: "f32[3072, 6272]" = torch.ops.aten.t.default(view_825)
    mm_191: "f32[3072, 768]" = torch.ops.aten.mm.default(t_438, view_98);  t_438 = view_98 = None
    t_439: "f32[768, 3072]" = torch.ops.aten.t.default(mm_191);  mm_191 = None
    sum_191: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_825, [0], True);  view_825 = None
    view_826: "f32[3072]" = torch.ops.aten.view.default(sum_191, [3072]);  sum_191 = None
    t_440: "f32[3072, 768]" = torch.ops.aten.t.default(t_439);  t_439 = None
    view_827: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_190, [8, 784, 768]);  mm_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_56 = torch.ops.aten.native_layer_norm_backward.default(view_827, add_39, [768], getitem_132, getitem_133, primals_247, primals_248, [True, True, True]);  view_827 = add_39 = getitem_132 = getitem_133 = primals_247 = primals_248 = None
    getitem_783: "f32[8, 784, 768]" = native_layer_norm_backward_56[0]
    getitem_784: "f32[768]" = native_layer_norm_backward_56[1]
    getitem_785: "f32[768]" = native_layer_norm_backward_56[2];  native_layer_norm_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_229: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_228, getitem_783);  add_228 = getitem_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_317: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_229, primals_27);  primals_27 = None
    mul_318: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_229, permute_29);  permute_29 = None
    sum_192: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1], True);  mul_318 = None
    view_828: "f32[768]" = torch.ops.aten.view.default(sum_192, [768]);  sum_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_178: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_317, [0, 2, 1]);  mul_317 = None
    view_829: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_178, [8, 768, 28, 28]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(view_829, getitem_126, primals_245, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_829 = getitem_126 = primals_245 = None
    getitem_786: "f32[8, 768, 28, 28]" = convolution_backward_34[0]
    getitem_787: "f32[768, 1, 3, 3]" = convolution_backward_34[1]
    getitem_788: "f32[768]" = convolution_backward_34[2];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_17 = torch.ops.aten.native_batch_norm_backward.default(getitem_786, gelu_14, primals_243, getitem_129, getitem_130, getitem_127, getitem_128, True, 1e-05, [True, True, True]);  getitem_786 = gelu_14 = primals_243 = getitem_127 = getitem_128 = None
    getitem_789: "f32[8, 768, 28, 28]" = native_batch_norm_backward_17[0]
    getitem_790: "f32[768]" = native_batch_norm_backward_17[1]
    getitem_791: "f32[768]" = native_batch_norm_backward_17[2];  native_batch_norm_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_37: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_789, convolution_16);  getitem_789 = convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(gelu_backward_37, view_96, primals_241, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_37 = view_96 = primals_241 = None
    getitem_792: "f32[8, 768, 28, 28]" = convolution_backward_35[0]
    getitem_793: "f32[768, 1, 3, 3]" = convolution_backward_35[1]
    getitem_794: "f32[768]" = convolution_backward_35[2];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_830: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_792, [8, 768, 784]);  getitem_792 = None
    permute_179: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_830, [0, 2, 1]);  view_830 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_57 = torch.ops.aten.native_layer_norm_backward.default(permute_179, add_37, [768], getitem_124, getitem_125, primals_239, primals_240, [True, True, True]);  permute_179 = add_37 = getitem_124 = getitem_125 = primals_239 = primals_240 = None
    getitem_795: "f32[8, 784, 768]" = native_layer_norm_backward_57[0]
    getitem_796: "f32[768]" = native_layer_norm_backward_57[1]
    getitem_797: "f32[768]" = native_layer_norm_backward_57[2];  native_layer_norm_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_230: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_229, getitem_795);  add_229 = getitem_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_319: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_230, primals_25);  primals_25 = None
    mul_320: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_230, clone_54);  clone_54 = None
    sum_193: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_320, [0, 1], True);  mul_320 = None
    view_831: "f32[768]" = torch.ops.aten.view.default(sum_193, [768]);  sum_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_194: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_319, [0, 1], True)
    view_832: "f32[768]" = torch.ops.aten.view.default(sum_194, [768]);  sum_194 = None
    view_833: "f32[6272, 768]" = torch.ops.aten.view.default(mul_319, [6272, 768]);  mul_319 = None
    t_441: "f32[768, 6272]" = torch.ops.aten.t.default(view_833)
    mm_192: "f32[768, 768]" = torch.ops.aten.mm.default(t_441, _unsafe_view_27);  t_441 = _unsafe_view_27 = None
    t_442: "f32[768, 768]" = torch.ops.aten.t.default(mm_192);  mm_192 = None
    t_443: "f32[768, 768]" = torch.ops.aten.t.default(t_25);  t_25 = None
    mm_193: "f32[6272, 768]" = torch.ops.aten.mm.default(view_833, t_443);  view_833 = t_443 = None
    view_834: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_193, [8, 784, 768]);  mm_193 = None
    t_444: "f32[768, 768]" = torch.ops.aten.t.default(t_442);  t_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_835: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_834, [8, 784, 16, 48]);  view_834 = None
    permute_180: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_835, [0, 2, 3, 1]);  view_835 = None
    clone_234: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
    _unsafe_view_130: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_234, [128, 48, 784]);  clone_234 = None
    transpose_114: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_92, 1, 2);  view_92 = None
    bmm_116: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_114, _unsafe_view_130);  transpose_114 = None
    transpose_115: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_26, 1, 2);  _unsafe_view_26 = None
    bmm_117: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_130, transpose_115);  _unsafe_view_130 = transpose_115 = None
    view_836: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_116, [8, 16, 48, 784]);  bmm_116 = None
    view_837: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_117, [8, 16, 48, 48]);  bmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_127: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_20);  detach_20 = None
    _softmax_backward_data_17: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_837, detach_127, -1, torch.float32);  view_837 = detach_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_321: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_17, view_91);  view_91 = None
    mul_322: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_17, primals_26);  _softmax_backward_data_17 = primals_26 = None
    sum_195: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [0, 2, 3], True);  mul_321 = None
    view_838: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_195, [16, 1, 1]);  sum_195 = None
    view_839: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_322, [128, 48, 48]);  mul_322 = None
    transpose_116: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_24, 1, 2);  _unsafe_view_24 = None
    bmm_118: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_116, view_839);  transpose_116 = None
    transpose_117: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_25, 1, 2);  _unsafe_view_25 = None
    bmm_119: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_839, transpose_117);  view_839 = transpose_117 = None
    view_840: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_118, [8, 16, 784, 48]);  bmm_118 = None
    view_841: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_119, [8, 16, 48, 784]);  bmm_119 = None
    transpose_118: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_840, -2, -1);  view_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_190: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_121, expand_37)
    div_191: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_190, expand_37);  div_190 = None
    neg_34: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_118)
    mul_323: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_34, div_191);  neg_34 = div_191 = None
    div_192: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_118, expand_37);  transpose_118 = expand_37 = None
    sum_196: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [3], True);  mul_323 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_34: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_13, 1e-12);  linalg_vector_norm_13 = None
    where_34: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_34, sum_196, scalar_tensor_34);  ge_34 = sum_196 = scalar_tensor_34 = None
    detach_128: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_19);  detach_19 = None
    div_193: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_121, detach_128);  getitem_121 = None
    eq_34: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_128, 0);  detach_128 = None
    masked_fill_34: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_193, eq_34, 0);  div_193 = eq_34 = None
    mul_324: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_34, masked_fill_34);  where_34 = masked_fill_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_231: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_192, mul_324);  div_192 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_194: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_120, expand_36)
    div_195: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_194, expand_36);  div_194 = None
    neg_35: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_841)
    mul_325: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_35, div_195);  neg_35 = div_195 = None
    div_196: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_841, expand_36);  view_841 = expand_36 = None
    sum_197: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [3], True);  mul_325 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_35: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_12, 1e-12);  linalg_vector_norm_12 = None
    where_35: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_35, sum_197, scalar_tensor_35);  ge_35 = sum_197 = scalar_tensor_35 = None
    detach_129: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_18);  detach_18 = None
    div_197: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_120, detach_129);  getitem_120 = None
    eq_35: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_129, 0);  detach_129 = None
    masked_fill_35: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_197, eq_35, 0);  div_197 = eq_35 = None
    mul_326: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_35, masked_fill_35);  where_35 = masked_fill_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_232: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_196, mul_326);  div_196 = mul_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_19: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_232, add_231, view_836]);  add_232 = add_231 = view_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_181: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_19, [1, 4, 0, 2, 3]);  stack_19 = None
    clone_235: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    _unsafe_view_131: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_235, [8, 784, 2304]);  clone_235 = None
    view_842: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_131, [6272, 2304]);  _unsafe_view_131 = None
    t_445: "f32[2304, 768]" = torch.ops.aten.t.default(t_24);  t_24 = None
    mm_194: "f32[6272, 768]" = torch.ops.aten.mm.default(view_842, t_445);  t_445 = None
    t_446: "f32[2304, 6272]" = torch.ops.aten.t.default(view_842)
    mm_195: "f32[2304, 768]" = torch.ops.aten.mm.default(t_446, view_88);  t_446 = view_88 = None
    t_447: "f32[768, 2304]" = torch.ops.aten.t.default(mm_195);  mm_195 = None
    sum_198: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_842, [0], True);  view_842 = None
    view_843: "f32[2304]" = torch.ops.aten.view.default(sum_198, [2304]);  sum_198 = None
    t_448: "f32[2304, 768]" = torch.ops.aten.t.default(t_447);  t_447 = None
    view_844: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_194, [8, 784, 768]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_58 = torch.ops.aten.native_layer_norm_backward.default(view_844, add_35, [768], getitem_118, getitem_119, primals_233, primals_234, [True, True, True]);  view_844 = add_35 = getitem_118 = getitem_119 = primals_233 = primals_234 = None
    getitem_798: "f32[8, 784, 768]" = native_layer_norm_backward_58[0]
    getitem_799: "f32[768]" = native_layer_norm_backward_58[1]
    getitem_800: "f32[768]" = native_layer_norm_backward_58[2];  native_layer_norm_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_233: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_230, getitem_798);  add_230 = getitem_798 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_327: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_233, primals_24);  primals_24 = None
    mul_328: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_233, clone_48);  clone_48 = None
    sum_199: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1], True);  mul_328 = None
    view_845: "f32[768]" = torch.ops.aten.view.default(sum_199, [768]);  sum_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_846: "f32[6272, 768]" = torch.ops.aten.view.default(mul_327, [6272, 768]);  mul_327 = None
    t_449: "f32[768, 3072]" = torch.ops.aten.t.default(t_23);  t_23 = None
    mm_196: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_846, t_449);  t_449 = None
    t_450: "f32[768, 6272]" = torch.ops.aten.t.default(view_846)
    mm_197: "f32[768, 3072]" = torch.ops.aten.mm.default(t_450, view_86);  t_450 = view_86 = None
    t_451: "f32[3072, 768]" = torch.ops.aten.t.default(mm_197);  mm_197 = None
    sum_200: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_846, [0], True);  view_846 = None
    view_847: "f32[768]" = torch.ops.aten.view.default(sum_200, [768]);  sum_200 = None
    t_452: "f32[768, 3072]" = torch.ops.aten.t.default(t_451);  t_451 = None
    view_848: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_196, [8, 784, 3072]);  mm_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_38: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_848, view_85);  view_848 = view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_849: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_38, [6272, 3072]);  gelu_backward_38 = None
    t_453: "f32[3072, 768]" = torch.ops.aten.t.default(t_22);  t_22 = None
    mm_198: "f32[6272, 768]" = torch.ops.aten.mm.default(view_849, t_453);  t_453 = None
    t_454: "f32[3072, 6272]" = torch.ops.aten.t.default(view_849)
    mm_199: "f32[3072, 768]" = torch.ops.aten.mm.default(t_454, view_84);  t_454 = view_84 = None
    t_455: "f32[768, 3072]" = torch.ops.aten.t.default(mm_199);  mm_199 = None
    sum_201: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
    view_850: "f32[3072]" = torch.ops.aten.view.default(sum_201, [3072]);  sum_201 = None
    t_456: "f32[3072, 768]" = torch.ops.aten.t.default(t_455);  t_455 = None
    view_851: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_198, [8, 784, 768]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_59 = torch.ops.aten.native_layer_norm_backward.default(view_851, add_34, [768], getitem_115, getitem_116, primals_227, primals_228, [True, True, True]);  view_851 = add_34 = getitem_115 = getitem_116 = primals_227 = primals_228 = None
    getitem_801: "f32[8, 784, 768]" = native_layer_norm_backward_59[0]
    getitem_802: "f32[768]" = native_layer_norm_backward_59[1]
    getitem_803: "f32[768]" = native_layer_norm_backward_59[2];  native_layer_norm_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_234: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_233, getitem_801);  add_233 = getitem_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_329: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_234, primals_23);  primals_23 = None
    mul_330: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_234, permute_25);  permute_25 = None
    sum_202: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1], True);  mul_330 = None
    view_852: "f32[768]" = torch.ops.aten.view.default(sum_202, [768]);  sum_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_182: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_329, [0, 2, 1]);  mul_329 = None
    view_853: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_182, [8, 768, 28, 28]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(view_853, getitem_109, primals_225, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_853 = getitem_109 = primals_225 = None
    getitem_804: "f32[8, 768, 28, 28]" = convolution_backward_36[0]
    getitem_805: "f32[768, 1, 3, 3]" = convolution_backward_36[1]
    getitem_806: "f32[768]" = convolution_backward_36[2];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_18 = torch.ops.aten.native_batch_norm_backward.default(getitem_804, gelu_12, primals_223, getitem_112, getitem_113, getitem_110, getitem_111, True, 1e-05, [True, True, True]);  getitem_804 = gelu_12 = primals_223 = getitem_110 = getitem_111 = None
    getitem_807: "f32[8, 768, 28, 28]" = native_batch_norm_backward_18[0]
    getitem_808: "f32[768]" = native_batch_norm_backward_18[1]
    getitem_809: "f32[768]" = native_batch_norm_backward_18[2];  native_batch_norm_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_39: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_807, convolution_14);  getitem_807 = convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(gelu_backward_39, view_82, primals_221, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_39 = view_82 = primals_221 = None
    getitem_810: "f32[8, 768, 28, 28]" = convolution_backward_37[0]
    getitem_811: "f32[768, 1, 3, 3]" = convolution_backward_37[1]
    getitem_812: "f32[768]" = convolution_backward_37[2];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_854: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_810, [8, 768, 784]);  getitem_810 = None
    permute_183: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_854, [0, 2, 1]);  view_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_60 = torch.ops.aten.native_layer_norm_backward.default(permute_183, add_32, [768], getitem_107, getitem_108, primals_219, primals_220, [True, True, True]);  permute_183 = add_32 = getitem_107 = getitem_108 = primals_219 = primals_220 = None
    getitem_813: "f32[8, 784, 768]" = native_layer_norm_backward_60[0]
    getitem_814: "f32[768]" = native_layer_norm_backward_60[1]
    getitem_815: "f32[768]" = native_layer_norm_backward_60[2];  native_layer_norm_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_235: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_234, getitem_813);  add_234 = getitem_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_331: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_235, primals_21);  primals_21 = None
    mul_332: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_235, clone_46);  clone_46 = None
    sum_203: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1], True);  mul_332 = None
    view_855: "f32[768]" = torch.ops.aten.view.default(sum_203, [768]);  sum_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_204: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_331, [0, 1], True)
    view_856: "f32[768]" = torch.ops.aten.view.default(sum_204, [768]);  sum_204 = None
    view_857: "f32[6272, 768]" = torch.ops.aten.view.default(mul_331, [6272, 768]);  mul_331 = None
    t_457: "f32[768, 6272]" = torch.ops.aten.t.default(view_857)
    mm_200: "f32[768, 768]" = torch.ops.aten.mm.default(t_457, _unsafe_view_23);  t_457 = _unsafe_view_23 = None
    t_458: "f32[768, 768]" = torch.ops.aten.t.default(mm_200);  mm_200 = None
    t_459: "f32[768, 768]" = torch.ops.aten.t.default(t_21);  t_21 = None
    mm_201: "f32[6272, 768]" = torch.ops.aten.mm.default(view_857, t_459);  view_857 = t_459 = None
    view_858: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_201, [8, 784, 768]);  mm_201 = None
    t_460: "f32[768, 768]" = torch.ops.aten.t.default(t_458);  t_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_859: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_858, [8, 784, 16, 48]);  view_858 = None
    permute_184: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_859, [0, 2, 3, 1]);  view_859 = None
    clone_236: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
    _unsafe_view_132: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_236, [128, 48, 784]);  clone_236 = None
    transpose_119: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_78, 1, 2);  view_78 = None
    bmm_120: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_119, _unsafe_view_132);  transpose_119 = None
    transpose_120: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_22, 1, 2);  _unsafe_view_22 = None
    bmm_121: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_132, transpose_120);  _unsafe_view_132 = transpose_120 = None
    view_860: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_120, [8, 16, 48, 784]);  bmm_120 = None
    view_861: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_121, [8, 16, 48, 48]);  bmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_130: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_17);  detach_17 = None
    _softmax_backward_data_18: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_861, detach_130, -1, torch.float32);  view_861 = detach_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_333: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_18, view_77);  view_77 = None
    mul_334: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_18, primals_22);  _softmax_backward_data_18 = primals_22 = None
    sum_205: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 2, 3], True);  mul_333 = None
    view_862: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_205, [16, 1, 1]);  sum_205 = None
    view_863: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_334, [128, 48, 48]);  mul_334 = None
    transpose_121: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_20, 1, 2);  _unsafe_view_20 = None
    bmm_122: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_121, view_863);  transpose_121 = None
    transpose_122: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_21, 1, 2);  _unsafe_view_21 = None
    bmm_123: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_863, transpose_122);  view_863 = transpose_122 = None
    view_864: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_122, [8, 16, 784, 48]);  bmm_122 = None
    view_865: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_123, [8, 16, 48, 784]);  bmm_123 = None
    transpose_123: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_864, -2, -1);  view_864 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_198: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_104, expand_31)
    div_199: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_198, expand_31);  div_198 = None
    neg_36: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_123)
    mul_335: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_36, div_199);  neg_36 = div_199 = None
    div_200: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_123, expand_31);  transpose_123 = expand_31 = None
    sum_206: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [3], True);  mul_335 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_36: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_11, 1e-12);  linalg_vector_norm_11 = None
    where_36: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_36, sum_206, scalar_tensor_36);  ge_36 = sum_206 = scalar_tensor_36 = None
    detach_131: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_16);  detach_16 = None
    div_201: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_104, detach_131);  getitem_104 = None
    eq_36: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_131, 0);  detach_131 = None
    masked_fill_36: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_201, eq_36, 0);  div_201 = eq_36 = None
    mul_336: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_36, masked_fill_36);  where_36 = masked_fill_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_236: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_200, mul_336);  div_200 = mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_202: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_103, expand_30)
    div_203: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_202, expand_30);  div_202 = None
    neg_37: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_865)
    mul_337: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_37, div_203);  neg_37 = div_203 = None
    div_204: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_865, expand_30);  view_865 = expand_30 = None
    sum_207: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_337, [3], True);  mul_337 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_37: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_10, 1e-12);  linalg_vector_norm_10 = None
    where_37: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_37, sum_207, scalar_tensor_37);  ge_37 = sum_207 = scalar_tensor_37 = None
    detach_132: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_15);  detach_15 = None
    div_205: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_103, detach_132);  getitem_103 = None
    eq_37: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_132, 0);  detach_132 = None
    masked_fill_37: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_205, eq_37, 0);  div_205 = eq_37 = None
    mul_338: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_37, masked_fill_37);  where_37 = masked_fill_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_237: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_204, mul_338);  div_204 = mul_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_20: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_237, add_236, view_860]);  add_237 = add_236 = view_860 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_185: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_20, [1, 4, 0, 2, 3]);  stack_20 = None
    clone_237: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
    _unsafe_view_133: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_237, [8, 784, 2304]);  clone_237 = None
    view_866: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_133, [6272, 2304]);  _unsafe_view_133 = None
    t_461: "f32[2304, 768]" = torch.ops.aten.t.default(t_20);  t_20 = None
    mm_202: "f32[6272, 768]" = torch.ops.aten.mm.default(view_866, t_461);  t_461 = None
    t_462: "f32[2304, 6272]" = torch.ops.aten.t.default(view_866)
    mm_203: "f32[2304, 768]" = torch.ops.aten.mm.default(t_462, view_74);  t_462 = view_74 = None
    t_463: "f32[768, 2304]" = torch.ops.aten.t.default(mm_203);  mm_203 = None
    sum_208: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_866, [0], True);  view_866 = None
    view_867: "f32[2304]" = torch.ops.aten.view.default(sum_208, [2304]);  sum_208 = None
    t_464: "f32[2304, 768]" = torch.ops.aten.t.default(t_463);  t_463 = None
    view_868: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_202, [8, 784, 768]);  mm_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_61 = torch.ops.aten.native_layer_norm_backward.default(view_868, add_30, [768], getitem_101, getitem_102, primals_213, primals_214, [True, True, True]);  view_868 = add_30 = getitem_101 = getitem_102 = primals_213 = primals_214 = None
    getitem_816: "f32[8, 784, 768]" = native_layer_norm_backward_61[0]
    getitem_817: "f32[768]" = native_layer_norm_backward_61[1]
    getitem_818: "f32[768]" = native_layer_norm_backward_61[2];  native_layer_norm_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_238: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_235, getitem_816);  add_235 = getitem_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_339: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_238, primals_20);  primals_20 = None
    mul_340: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_238, clone_40);  clone_40 = None
    sum_209: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 1], True);  mul_340 = None
    view_869: "f32[768]" = torch.ops.aten.view.default(sum_209, [768]);  sum_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_870: "f32[6272, 768]" = torch.ops.aten.view.default(mul_339, [6272, 768]);  mul_339 = None
    t_465: "f32[768, 3072]" = torch.ops.aten.t.default(t_19);  t_19 = None
    mm_204: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_870, t_465);  t_465 = None
    t_466: "f32[768, 6272]" = torch.ops.aten.t.default(view_870)
    mm_205: "f32[768, 3072]" = torch.ops.aten.mm.default(t_466, view_72);  t_466 = view_72 = None
    t_467: "f32[3072, 768]" = torch.ops.aten.t.default(mm_205);  mm_205 = None
    sum_210: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_870, [0], True);  view_870 = None
    view_871: "f32[768]" = torch.ops.aten.view.default(sum_210, [768]);  sum_210 = None
    t_468: "f32[768, 3072]" = torch.ops.aten.t.default(t_467);  t_467 = None
    view_872: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_204, [8, 784, 3072]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_40: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_872, view_71);  view_872 = view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_873: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_40, [6272, 3072]);  gelu_backward_40 = None
    t_469: "f32[3072, 768]" = torch.ops.aten.t.default(t_18);  t_18 = None
    mm_206: "f32[6272, 768]" = torch.ops.aten.mm.default(view_873, t_469);  t_469 = None
    t_470: "f32[3072, 6272]" = torch.ops.aten.t.default(view_873)
    mm_207: "f32[3072, 768]" = torch.ops.aten.mm.default(t_470, view_70);  t_470 = view_70 = None
    t_471: "f32[768, 3072]" = torch.ops.aten.t.default(mm_207);  mm_207 = None
    sum_211: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_873, [0], True);  view_873 = None
    view_874: "f32[3072]" = torch.ops.aten.view.default(sum_211, [3072]);  sum_211 = None
    t_472: "f32[3072, 768]" = torch.ops.aten.t.default(t_471);  t_471 = None
    view_875: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_206, [8, 784, 768]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_62 = torch.ops.aten.native_layer_norm_backward.default(view_875, add_29, [768], getitem_98, getitem_99, primals_207, primals_208, [True, True, True]);  view_875 = add_29 = getitem_98 = getitem_99 = primals_207 = primals_208 = None
    getitem_819: "f32[8, 784, 768]" = native_layer_norm_backward_62[0]
    getitem_820: "f32[768]" = native_layer_norm_backward_62[1]
    getitem_821: "f32[768]" = native_layer_norm_backward_62[2];  native_layer_norm_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_239: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_238, getitem_819);  add_238 = getitem_819 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_341: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_239, primals_19);  primals_19 = None
    mul_342: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_239, permute_21);  permute_21 = None
    sum_212: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1], True);  mul_342 = None
    view_876: "f32[768]" = torch.ops.aten.view.default(sum_212, [768]);  sum_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_186: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_341, [0, 2, 1]);  mul_341 = None
    view_877: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_186, [8, 768, 28, 28]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(view_877, getitem_92, primals_205, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_877 = getitem_92 = primals_205 = None
    getitem_822: "f32[8, 768, 28, 28]" = convolution_backward_38[0]
    getitem_823: "f32[768, 1, 3, 3]" = convolution_backward_38[1]
    getitem_824: "f32[768]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_19 = torch.ops.aten.native_batch_norm_backward.default(getitem_822, gelu_10, primals_203, getitem_95, getitem_96, getitem_93, getitem_94, True, 1e-05, [True, True, True]);  getitem_822 = gelu_10 = primals_203 = getitem_93 = getitem_94 = None
    getitem_825: "f32[8, 768, 28, 28]" = native_batch_norm_backward_19[0]
    getitem_826: "f32[768]" = native_batch_norm_backward_19[1]
    getitem_827: "f32[768]" = native_batch_norm_backward_19[2];  native_batch_norm_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_41: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_825, convolution_12);  getitem_825 = convolution_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(gelu_backward_41, view_68, primals_201, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_41 = view_68 = primals_201 = None
    getitem_828: "f32[8, 768, 28, 28]" = convolution_backward_39[0]
    getitem_829: "f32[768, 1, 3, 3]" = convolution_backward_39[1]
    getitem_830: "f32[768]" = convolution_backward_39[2];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_878: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_828, [8, 768, 784]);  getitem_828 = None
    permute_187: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_878, [0, 2, 1]);  view_878 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_63 = torch.ops.aten.native_layer_norm_backward.default(permute_187, add_27, [768], getitem_90, getitem_91, primals_199, primals_200, [True, True, True]);  permute_187 = add_27 = getitem_90 = getitem_91 = primals_199 = primals_200 = None
    getitem_831: "f32[8, 784, 768]" = native_layer_norm_backward_63[0]
    getitem_832: "f32[768]" = native_layer_norm_backward_63[1]
    getitem_833: "f32[768]" = native_layer_norm_backward_63[2];  native_layer_norm_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_240: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_239, getitem_831);  add_239 = getitem_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_343: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_240, primals_17);  primals_17 = None
    mul_344: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_240, clone_38);  clone_38 = None
    sum_213: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 1], True);  mul_344 = None
    view_879: "f32[768]" = torch.ops.aten.view.default(sum_213, [768]);  sum_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_214: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1], True)
    view_880: "f32[768]" = torch.ops.aten.view.default(sum_214, [768]);  sum_214 = None
    view_881: "f32[6272, 768]" = torch.ops.aten.view.default(mul_343, [6272, 768]);  mul_343 = None
    t_473: "f32[768, 6272]" = torch.ops.aten.t.default(view_881)
    mm_208: "f32[768, 768]" = torch.ops.aten.mm.default(t_473, _unsafe_view_19);  t_473 = _unsafe_view_19 = None
    t_474: "f32[768, 768]" = torch.ops.aten.t.default(mm_208);  mm_208 = None
    t_475: "f32[768, 768]" = torch.ops.aten.t.default(t_17);  t_17 = None
    mm_209: "f32[6272, 768]" = torch.ops.aten.mm.default(view_881, t_475);  view_881 = t_475 = None
    view_882: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_209, [8, 784, 768]);  mm_209 = None
    t_476: "f32[768, 768]" = torch.ops.aten.t.default(t_474);  t_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_883: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_882, [8, 784, 16, 48]);  view_882 = None
    permute_188: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_883, [0, 2, 3, 1]);  view_883 = None
    clone_238: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    _unsafe_view_134: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_238, [128, 48, 784]);  clone_238 = None
    transpose_124: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_64, 1, 2);  view_64 = None
    bmm_124: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_124, _unsafe_view_134);  transpose_124 = None
    transpose_125: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_18, 1, 2);  _unsafe_view_18 = None
    bmm_125: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_134, transpose_125);  _unsafe_view_134 = transpose_125 = None
    view_884: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_124, [8, 16, 48, 784]);  bmm_124 = None
    view_885: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_125, [8, 16, 48, 48]);  bmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_133: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_14);  detach_14 = None
    _softmax_backward_data_19: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_885, detach_133, -1, torch.float32);  view_885 = detach_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_345: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_19, view_63);  view_63 = None
    mul_346: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_19, primals_18);  _softmax_backward_data_19 = primals_18 = None
    sum_215: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 2, 3], True);  mul_345 = None
    view_886: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_215, [16, 1, 1]);  sum_215 = None
    view_887: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_346, [128, 48, 48]);  mul_346 = None
    transpose_126: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_16, 1, 2);  _unsafe_view_16 = None
    bmm_126: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_126, view_887);  transpose_126 = None
    transpose_127: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_17, 1, 2);  _unsafe_view_17 = None
    bmm_127: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_887, transpose_127);  view_887 = transpose_127 = None
    view_888: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_126, [8, 16, 784, 48]);  bmm_126 = None
    view_889: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_127, [8, 16, 48, 784]);  bmm_127 = None
    transpose_128: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_888, -2, -1);  view_888 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_206: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_87, expand_25)
    div_207: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_206, expand_25);  div_206 = None
    neg_38: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_128)
    mul_347: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_38, div_207);  neg_38 = div_207 = None
    div_208: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_128, expand_25);  transpose_128 = expand_25 = None
    sum_216: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [3], True);  mul_347 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_38: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_9, 1e-12);  linalg_vector_norm_9 = None
    where_38: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_38, sum_216, scalar_tensor_38);  ge_38 = sum_216 = scalar_tensor_38 = None
    detach_134: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_13);  detach_13 = None
    div_209: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_87, detach_134);  getitem_87 = None
    eq_38: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_134, 0);  detach_134 = None
    masked_fill_38: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_209, eq_38, 0);  div_209 = eq_38 = None
    mul_348: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_38, masked_fill_38);  where_38 = masked_fill_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_241: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_208, mul_348);  div_208 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_210: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_86, expand_24)
    div_211: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_210, expand_24);  div_210 = None
    neg_39: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_889)
    mul_349: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_39, div_211);  neg_39 = div_211 = None
    div_212: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_889, expand_24);  view_889 = expand_24 = None
    sum_217: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [3], True);  mul_349 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_39: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_8, 1e-12);  linalg_vector_norm_8 = None
    where_39: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_39, sum_217, scalar_tensor_39);  ge_39 = sum_217 = scalar_tensor_39 = None
    detach_135: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_12);  detach_12 = None
    div_213: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_86, detach_135);  getitem_86 = None
    eq_39: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_135, 0);  detach_135 = None
    masked_fill_39: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_213, eq_39, 0);  div_213 = eq_39 = None
    mul_350: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_39, masked_fill_39);  where_39 = masked_fill_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_242: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_212, mul_350);  div_212 = mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_21: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_242, add_241, view_884]);  add_242 = add_241 = view_884 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_189: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_21, [1, 4, 0, 2, 3]);  stack_21 = None
    clone_239: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    _unsafe_view_135: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_239, [8, 784, 2304]);  clone_239 = None
    view_890: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_135, [6272, 2304]);  _unsafe_view_135 = None
    t_477: "f32[2304, 768]" = torch.ops.aten.t.default(t_16);  t_16 = None
    mm_210: "f32[6272, 768]" = torch.ops.aten.mm.default(view_890, t_477);  t_477 = None
    t_478: "f32[2304, 6272]" = torch.ops.aten.t.default(view_890)
    mm_211: "f32[2304, 768]" = torch.ops.aten.mm.default(t_478, view_60);  t_478 = view_60 = None
    t_479: "f32[768, 2304]" = torch.ops.aten.t.default(mm_211);  mm_211 = None
    sum_218: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_890, [0], True);  view_890 = None
    view_891: "f32[2304]" = torch.ops.aten.view.default(sum_218, [2304]);  sum_218 = None
    t_480: "f32[2304, 768]" = torch.ops.aten.t.default(t_479);  t_479 = None
    view_892: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_210, [8, 784, 768]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_64 = torch.ops.aten.native_layer_norm_backward.default(view_892, add_25, [768], getitem_84, getitem_85, primals_193, primals_194, [True, True, True]);  view_892 = add_25 = getitem_84 = getitem_85 = primals_193 = primals_194 = None
    getitem_834: "f32[8, 784, 768]" = native_layer_norm_backward_64[0]
    getitem_835: "f32[768]" = native_layer_norm_backward_64[1]
    getitem_836: "f32[768]" = native_layer_norm_backward_64[2];  native_layer_norm_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_243: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_240, getitem_834);  add_240 = getitem_834 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_351: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_243, primals_16);  primals_16 = None
    mul_352: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_243, clone_32);  clone_32 = None
    sum_219: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1], True);  mul_352 = None
    view_893: "f32[768]" = torch.ops.aten.view.default(sum_219, [768]);  sum_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_894: "f32[6272, 768]" = torch.ops.aten.view.default(mul_351, [6272, 768]);  mul_351 = None
    t_481: "f32[768, 3072]" = torch.ops.aten.t.default(t_15);  t_15 = None
    mm_212: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_894, t_481);  t_481 = None
    t_482: "f32[768, 6272]" = torch.ops.aten.t.default(view_894)
    mm_213: "f32[768, 3072]" = torch.ops.aten.mm.default(t_482, view_58);  t_482 = view_58 = None
    t_483: "f32[3072, 768]" = torch.ops.aten.t.default(mm_213);  mm_213 = None
    sum_220: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_894, [0], True);  view_894 = None
    view_895: "f32[768]" = torch.ops.aten.view.default(sum_220, [768]);  sum_220 = None
    t_484: "f32[768, 3072]" = torch.ops.aten.t.default(t_483);  t_483 = None
    view_896: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_212, [8, 784, 3072]);  mm_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_42: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_896, view_57);  view_896 = view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_897: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_42, [6272, 3072]);  gelu_backward_42 = None
    t_485: "f32[3072, 768]" = torch.ops.aten.t.default(t_14);  t_14 = None
    mm_214: "f32[6272, 768]" = torch.ops.aten.mm.default(view_897, t_485);  t_485 = None
    t_486: "f32[3072, 6272]" = torch.ops.aten.t.default(view_897)
    mm_215: "f32[3072, 768]" = torch.ops.aten.mm.default(t_486, view_56);  t_486 = view_56 = None
    t_487: "f32[768, 3072]" = torch.ops.aten.t.default(mm_215);  mm_215 = None
    sum_221: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_897, [0], True);  view_897 = None
    view_898: "f32[3072]" = torch.ops.aten.view.default(sum_221, [3072]);  sum_221 = None
    t_488: "f32[3072, 768]" = torch.ops.aten.t.default(t_487);  t_487 = None
    view_899: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_214, [8, 784, 768]);  mm_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_65 = torch.ops.aten.native_layer_norm_backward.default(view_899, add_24, [768], getitem_81, getitem_82, primals_187, primals_188, [True, True, True]);  view_899 = add_24 = getitem_81 = getitem_82 = primals_187 = primals_188 = None
    getitem_837: "f32[8, 784, 768]" = native_layer_norm_backward_65[0]
    getitem_838: "f32[768]" = native_layer_norm_backward_65[1]
    getitem_839: "f32[768]" = native_layer_norm_backward_65[2];  native_layer_norm_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_244: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_243, getitem_837);  add_243 = getitem_837 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_353: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_244, primals_15);  primals_15 = None
    mul_354: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_244, permute_17);  permute_17 = None
    sum_222: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_354, [0, 1], True);  mul_354 = None
    view_900: "f32[768]" = torch.ops.aten.view.default(sum_222, [768]);  sum_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_190: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_353, [0, 2, 1]);  mul_353 = None
    view_901: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_190, [8, 768, 28, 28]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(view_901, getitem_75, primals_185, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_901 = getitem_75 = primals_185 = None
    getitem_840: "f32[8, 768, 28, 28]" = convolution_backward_40[0]
    getitem_841: "f32[768, 1, 3, 3]" = convolution_backward_40[1]
    getitem_842: "f32[768]" = convolution_backward_40[2];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_20 = torch.ops.aten.native_batch_norm_backward.default(getitem_840, gelu_8, primals_183, getitem_78, getitem_79, getitem_76, getitem_77, True, 1e-05, [True, True, True]);  getitem_840 = gelu_8 = primals_183 = getitem_76 = getitem_77 = None
    getitem_843: "f32[8, 768, 28, 28]" = native_batch_norm_backward_20[0]
    getitem_844: "f32[768]" = native_batch_norm_backward_20[1]
    getitem_845: "f32[768]" = native_batch_norm_backward_20[2];  native_batch_norm_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_43: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_843, convolution_10);  getitem_843 = convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(gelu_backward_43, view_54, primals_181, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_43 = view_54 = primals_181 = None
    getitem_846: "f32[8, 768, 28, 28]" = convolution_backward_41[0]
    getitem_847: "f32[768, 1, 3, 3]" = convolution_backward_41[1]
    getitem_848: "f32[768]" = convolution_backward_41[2];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_902: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_846, [8, 768, 784]);  getitem_846 = None
    permute_191: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_902, [0, 2, 1]);  view_902 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_66 = torch.ops.aten.native_layer_norm_backward.default(permute_191, add_22, [768], getitem_73, getitem_74, primals_179, primals_180, [True, True, True]);  permute_191 = add_22 = getitem_73 = getitem_74 = primals_179 = primals_180 = None
    getitem_849: "f32[8, 784, 768]" = native_layer_norm_backward_66[0]
    getitem_850: "f32[768]" = native_layer_norm_backward_66[1]
    getitem_851: "f32[768]" = native_layer_norm_backward_66[2];  native_layer_norm_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_245: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_244, getitem_849);  add_244 = getitem_849 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_355: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_245, primals_13);  primals_13 = None
    mul_356: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_245, clone_30);  clone_30 = None
    sum_223: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_356, [0, 1], True);  mul_356 = None
    view_903: "f32[768]" = torch.ops.aten.view.default(sum_223, [768]);  sum_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_224: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_355, [0, 1], True)
    view_904: "f32[768]" = torch.ops.aten.view.default(sum_224, [768]);  sum_224 = None
    view_905: "f32[6272, 768]" = torch.ops.aten.view.default(mul_355, [6272, 768]);  mul_355 = None
    t_489: "f32[768, 6272]" = torch.ops.aten.t.default(view_905)
    mm_216: "f32[768, 768]" = torch.ops.aten.mm.default(t_489, _unsafe_view_15);  t_489 = _unsafe_view_15 = None
    t_490: "f32[768, 768]" = torch.ops.aten.t.default(mm_216);  mm_216 = None
    t_491: "f32[768, 768]" = torch.ops.aten.t.default(t_13);  t_13 = None
    mm_217: "f32[6272, 768]" = torch.ops.aten.mm.default(view_905, t_491);  view_905 = t_491 = None
    view_906: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_217, [8, 784, 768]);  mm_217 = None
    t_492: "f32[768, 768]" = torch.ops.aten.t.default(t_490);  t_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_907: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_906, [8, 784, 16, 48]);  view_906 = None
    permute_192: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_907, [0, 2, 3, 1]);  view_907 = None
    clone_240: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
    _unsafe_view_136: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_240, [128, 48, 784]);  clone_240 = None
    transpose_129: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_50, 1, 2);  view_50 = None
    bmm_128: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_129, _unsafe_view_136);  transpose_129 = None
    transpose_130: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_14, 1, 2);  _unsafe_view_14 = None
    bmm_129: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_136, transpose_130);  _unsafe_view_136 = transpose_130 = None
    view_908: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_128, [8, 16, 48, 784]);  bmm_128 = None
    view_909: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_129, [8, 16, 48, 48]);  bmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_136: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_11);  detach_11 = None
    _softmax_backward_data_20: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_909, detach_136, -1, torch.float32);  view_909 = detach_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_357: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_20, view_49);  view_49 = None
    mul_358: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_20, primals_14);  _softmax_backward_data_20 = primals_14 = None
    sum_225: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 2, 3], True);  mul_357 = None
    view_910: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_225, [16, 1, 1]);  sum_225 = None
    view_911: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_358, [128, 48, 48]);  mul_358 = None
    transpose_131: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_12, 1, 2);  _unsafe_view_12 = None
    bmm_130: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_131, view_911);  transpose_131 = None
    transpose_132: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_13, 1, 2);  _unsafe_view_13 = None
    bmm_131: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_911, transpose_132);  view_911 = transpose_132 = None
    view_912: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_130, [8, 16, 784, 48]);  bmm_130 = None
    view_913: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_131, [8, 16, 48, 784]);  bmm_131 = None
    transpose_133: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_912, -2, -1);  view_912 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_214: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_70, expand_19)
    div_215: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_214, expand_19);  div_214 = None
    neg_40: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_133)
    mul_359: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_40, div_215);  neg_40 = div_215 = None
    div_216: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_133, expand_19);  transpose_133 = expand_19 = None
    sum_226: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_359, [3], True);  mul_359 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_40: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_7, 1e-12);  linalg_vector_norm_7 = None
    where_40: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_40, sum_226, scalar_tensor_40);  ge_40 = sum_226 = scalar_tensor_40 = None
    detach_137: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_10);  detach_10 = None
    div_217: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_70, detach_137);  getitem_70 = None
    eq_40: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_137, 0);  detach_137 = None
    masked_fill_40: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_217, eq_40, 0);  div_217 = eq_40 = None
    mul_360: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_40, masked_fill_40);  where_40 = masked_fill_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_246: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_216, mul_360);  div_216 = mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_218: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_69, expand_18)
    div_219: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_218, expand_18);  div_218 = None
    neg_41: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_913)
    mul_361: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_41, div_219);  neg_41 = div_219 = None
    div_220: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_913, expand_18);  view_913 = expand_18 = None
    sum_227: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_361, [3], True);  mul_361 = None
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_41: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_6, 1e-12);  linalg_vector_norm_6 = None
    where_41: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_41, sum_227, scalar_tensor_41);  ge_41 = sum_227 = scalar_tensor_41 = None
    detach_138: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_9);  detach_9 = None
    div_221: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_69, detach_138);  getitem_69 = None
    eq_41: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_138, 0);  detach_138 = None
    masked_fill_41: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_221, eq_41, 0);  div_221 = eq_41 = None
    mul_362: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_41, masked_fill_41);  where_41 = masked_fill_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_247: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_220, mul_362);  div_220 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_22: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_247, add_246, view_908]);  add_247 = add_246 = view_908 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_193: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_22, [1, 4, 0, 2, 3]);  stack_22 = None
    clone_241: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    _unsafe_view_137: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_241, [8, 784, 2304]);  clone_241 = None
    view_914: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_137, [6272, 2304]);  _unsafe_view_137 = None
    t_493: "f32[2304, 768]" = torch.ops.aten.t.default(t_12);  t_12 = None
    mm_218: "f32[6272, 768]" = torch.ops.aten.mm.default(view_914, t_493);  t_493 = None
    t_494: "f32[2304, 6272]" = torch.ops.aten.t.default(view_914)
    mm_219: "f32[2304, 768]" = torch.ops.aten.mm.default(t_494, view_46);  t_494 = view_46 = None
    t_495: "f32[768, 2304]" = torch.ops.aten.t.default(mm_219);  mm_219 = None
    sum_228: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_914, [0], True);  view_914 = None
    view_915: "f32[2304]" = torch.ops.aten.view.default(sum_228, [2304]);  sum_228 = None
    t_496: "f32[2304, 768]" = torch.ops.aten.t.default(t_495);  t_495 = None
    view_916: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_218, [8, 784, 768]);  mm_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_67 = torch.ops.aten.native_layer_norm_backward.default(view_916, add_20, [768], getitem_67, getitem_68, primals_173, primals_174, [True, True, True]);  view_916 = add_20 = getitem_67 = getitem_68 = primals_173 = primals_174 = None
    getitem_852: "f32[8, 784, 768]" = native_layer_norm_backward_67[0]
    getitem_853: "f32[768]" = native_layer_norm_backward_67[1]
    getitem_854: "f32[768]" = native_layer_norm_backward_67[2];  native_layer_norm_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_248: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_245, getitem_852);  add_245 = getitem_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_363: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_248, primals_12);  primals_12 = None
    mul_364: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_248, clone_24);  clone_24 = None
    sum_229: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 1], True);  mul_364 = None
    view_917: "f32[768]" = torch.ops.aten.view.default(sum_229, [768]);  sum_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_918: "f32[6272, 768]" = torch.ops.aten.view.default(mul_363, [6272, 768]);  mul_363 = None
    t_497: "f32[768, 3072]" = torch.ops.aten.t.default(t_11);  t_11 = None
    mm_220: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_918, t_497);  t_497 = None
    t_498: "f32[768, 6272]" = torch.ops.aten.t.default(view_918)
    mm_221: "f32[768, 3072]" = torch.ops.aten.mm.default(t_498, view_44);  t_498 = view_44 = None
    t_499: "f32[3072, 768]" = torch.ops.aten.t.default(mm_221);  mm_221 = None
    sum_230: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_918, [0], True);  view_918 = None
    view_919: "f32[768]" = torch.ops.aten.view.default(sum_230, [768]);  sum_230 = None
    t_500: "f32[768, 3072]" = torch.ops.aten.t.default(t_499);  t_499 = None
    view_920: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_220, [8, 784, 3072]);  mm_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_44: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_920, view_43);  view_920 = view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_921: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_44, [6272, 3072]);  gelu_backward_44 = None
    t_501: "f32[3072, 768]" = torch.ops.aten.t.default(t_10);  t_10 = None
    mm_222: "f32[6272, 768]" = torch.ops.aten.mm.default(view_921, t_501);  t_501 = None
    t_502: "f32[3072, 6272]" = torch.ops.aten.t.default(view_921)
    mm_223: "f32[3072, 768]" = torch.ops.aten.mm.default(t_502, view_42);  t_502 = view_42 = None
    t_503: "f32[768, 3072]" = torch.ops.aten.t.default(mm_223);  mm_223 = None
    sum_231: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_921, [0], True);  view_921 = None
    view_922: "f32[3072]" = torch.ops.aten.view.default(sum_231, [3072]);  sum_231 = None
    t_504: "f32[3072, 768]" = torch.ops.aten.t.default(t_503);  t_503 = None
    view_923: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_222, [8, 784, 768]);  mm_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_68 = torch.ops.aten.native_layer_norm_backward.default(view_923, add_19, [768], getitem_64, getitem_65, primals_167, primals_168, [True, True, True]);  view_923 = add_19 = getitem_64 = getitem_65 = primals_167 = primals_168 = None
    getitem_855: "f32[8, 784, 768]" = native_layer_norm_backward_68[0]
    getitem_856: "f32[768]" = native_layer_norm_backward_68[1]
    getitem_857: "f32[768]" = native_layer_norm_backward_68[2];  native_layer_norm_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_249: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_248, getitem_855);  add_248 = getitem_855 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_365: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_249, primals_11);  primals_11 = None
    mul_366: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_249, permute_13);  permute_13 = None
    sum_232: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 1], True);  mul_366 = None
    view_924: "f32[768]" = torch.ops.aten.view.default(sum_232, [768]);  sum_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_194: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_365, [0, 2, 1]);  mul_365 = None
    view_925: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_194, [8, 768, 28, 28]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(view_925, getitem_58, primals_165, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_925 = getitem_58 = primals_165 = None
    getitem_858: "f32[8, 768, 28, 28]" = convolution_backward_42[0]
    getitem_859: "f32[768, 1, 3, 3]" = convolution_backward_42[1]
    getitem_860: "f32[768]" = convolution_backward_42[2];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_21 = torch.ops.aten.native_batch_norm_backward.default(getitem_858, gelu_6, primals_163, getitem_61, getitem_62, getitem_59, getitem_60, True, 1e-05, [True, True, True]);  getitem_858 = gelu_6 = primals_163 = getitem_59 = getitem_60 = None
    getitem_861: "f32[8, 768, 28, 28]" = native_batch_norm_backward_21[0]
    getitem_862: "f32[768]" = native_batch_norm_backward_21[1]
    getitem_863: "f32[768]" = native_batch_norm_backward_21[2];  native_batch_norm_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_45: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_861, convolution_8);  getitem_861 = convolution_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(gelu_backward_45, view_40, primals_161, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_45 = view_40 = primals_161 = None
    getitem_864: "f32[8, 768, 28, 28]" = convolution_backward_43[0]
    getitem_865: "f32[768, 1, 3, 3]" = convolution_backward_43[1]
    getitem_866: "f32[768]" = convolution_backward_43[2];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_926: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_864, [8, 768, 784]);  getitem_864 = None
    permute_195: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_926, [0, 2, 1]);  view_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_69 = torch.ops.aten.native_layer_norm_backward.default(permute_195, add_17, [768], getitem_56, getitem_57, primals_159, primals_160, [True, True, True]);  permute_195 = add_17 = getitem_56 = getitem_57 = primals_159 = primals_160 = None
    getitem_867: "f32[8, 784, 768]" = native_layer_norm_backward_69[0]
    getitem_868: "f32[768]" = native_layer_norm_backward_69[1]
    getitem_869: "f32[768]" = native_layer_norm_backward_69[2];  native_layer_norm_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_250: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_249, getitem_867);  add_249 = getitem_867 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_367: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_250, primals_9);  primals_9 = None
    mul_368: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_250, clone_22);  clone_22 = None
    sum_233: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 1], True);  mul_368 = None
    view_927: "f32[768]" = torch.ops.aten.view.default(sum_233, [768]);  sum_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_234: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_367, [0, 1], True)
    view_928: "f32[768]" = torch.ops.aten.view.default(sum_234, [768]);  sum_234 = None
    view_929: "f32[6272, 768]" = torch.ops.aten.view.default(mul_367, [6272, 768]);  mul_367 = None
    t_505: "f32[768, 6272]" = torch.ops.aten.t.default(view_929)
    mm_224: "f32[768, 768]" = torch.ops.aten.mm.default(t_505, _unsafe_view_11);  t_505 = _unsafe_view_11 = None
    t_506: "f32[768, 768]" = torch.ops.aten.t.default(mm_224);  mm_224 = None
    t_507: "f32[768, 768]" = torch.ops.aten.t.default(t_9);  t_9 = None
    mm_225: "f32[6272, 768]" = torch.ops.aten.mm.default(view_929, t_507);  view_929 = t_507 = None
    view_930: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_225, [8, 784, 768]);  mm_225 = None
    t_508: "f32[768, 768]" = torch.ops.aten.t.default(t_506);  t_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_931: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_930, [8, 784, 16, 48]);  view_930 = None
    permute_196: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_931, [0, 2, 3, 1]);  view_931 = None
    clone_242: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_196, memory_format = torch.contiguous_format);  permute_196 = None
    _unsafe_view_138: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_242, [128, 48, 784]);  clone_242 = None
    transpose_134: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_36, 1, 2);  view_36 = None
    bmm_132: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_134, _unsafe_view_138);  transpose_134 = None
    transpose_135: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_10, 1, 2);  _unsafe_view_10 = None
    bmm_133: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_138, transpose_135);  _unsafe_view_138 = transpose_135 = None
    view_932: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_132, [8, 16, 48, 784]);  bmm_132 = None
    view_933: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_133, [8, 16, 48, 48]);  bmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_139: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_8);  detach_8 = None
    _softmax_backward_data_21: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_933, detach_139, -1, torch.float32);  view_933 = detach_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_369: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_21, view_35);  view_35 = None
    mul_370: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_21, primals_10);  _softmax_backward_data_21 = primals_10 = None
    sum_235: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [0, 2, 3], True);  mul_369 = None
    view_934: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_235, [16, 1, 1]);  sum_235 = None
    view_935: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_370, [128, 48, 48]);  mul_370 = None
    transpose_136: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_8, 1, 2);  _unsafe_view_8 = None
    bmm_134: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_136, view_935);  transpose_136 = None
    transpose_137: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_9, 1, 2);  _unsafe_view_9 = None
    bmm_135: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_935, transpose_137);  view_935 = transpose_137 = None
    view_936: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_134, [8, 16, 784, 48]);  bmm_134 = None
    view_937: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_135, [8, 16, 48, 784]);  bmm_135 = None
    transpose_138: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_936, -2, -1);  view_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_222: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_53, expand_13)
    div_223: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_222, expand_13);  div_222 = None
    neg_42: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_138)
    mul_371: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_42, div_223);  neg_42 = div_223 = None
    div_224: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_138, expand_13);  transpose_138 = expand_13 = None
    sum_236: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [3], True);  mul_371 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_42: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_5, 1e-12);  linalg_vector_norm_5 = None
    where_42: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_42, sum_236, scalar_tensor_42);  ge_42 = sum_236 = scalar_tensor_42 = None
    detach_140: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_7);  detach_7 = None
    div_225: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_53, detach_140);  getitem_53 = None
    eq_42: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_140, 0);  detach_140 = None
    masked_fill_42: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_225, eq_42, 0);  div_225 = eq_42 = None
    mul_372: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_42, masked_fill_42);  where_42 = masked_fill_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_251: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_224, mul_372);  div_224 = mul_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_226: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_52, expand_12)
    div_227: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_226, expand_12);  div_226 = None
    neg_43: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_937)
    mul_373: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_43, div_227);  neg_43 = div_227 = None
    div_228: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_937, expand_12);  view_937 = expand_12 = None
    sum_237: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [3], True);  mul_373 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_43: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_4, 1e-12);  linalg_vector_norm_4 = None
    where_43: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_43, sum_237, scalar_tensor_43);  ge_43 = sum_237 = scalar_tensor_43 = None
    detach_141: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_6);  detach_6 = None
    div_229: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_52, detach_141);  getitem_52 = None
    eq_43: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_141, 0);  detach_141 = None
    masked_fill_43: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_229, eq_43, 0);  div_229 = eq_43 = None
    mul_374: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_43, masked_fill_43);  where_43 = masked_fill_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_252: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_228, mul_374);  div_228 = mul_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_23: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_252, add_251, view_932]);  add_252 = add_251 = view_932 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_197: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_23, [1, 4, 0, 2, 3]);  stack_23 = None
    clone_243: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
    _unsafe_view_139: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_243, [8, 784, 2304]);  clone_243 = None
    view_938: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_139, [6272, 2304]);  _unsafe_view_139 = None
    t_509: "f32[2304, 768]" = torch.ops.aten.t.default(t_8);  t_8 = None
    mm_226: "f32[6272, 768]" = torch.ops.aten.mm.default(view_938, t_509);  t_509 = None
    t_510: "f32[2304, 6272]" = torch.ops.aten.t.default(view_938)
    mm_227: "f32[2304, 768]" = torch.ops.aten.mm.default(t_510, view_32);  t_510 = view_32 = None
    t_511: "f32[768, 2304]" = torch.ops.aten.t.default(mm_227);  mm_227 = None
    sum_238: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_938, [0], True);  view_938 = None
    view_939: "f32[2304]" = torch.ops.aten.view.default(sum_238, [2304]);  sum_238 = None
    t_512: "f32[2304, 768]" = torch.ops.aten.t.default(t_511);  t_511 = None
    view_940: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_226, [8, 784, 768]);  mm_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_70 = torch.ops.aten.native_layer_norm_backward.default(view_940, add_15, [768], getitem_50, getitem_51, primals_153, primals_154, [True, True, True]);  view_940 = add_15 = getitem_50 = getitem_51 = primals_153 = primals_154 = None
    getitem_870: "f32[8, 784, 768]" = native_layer_norm_backward_70[0]
    getitem_871: "f32[768]" = native_layer_norm_backward_70[1]
    getitem_872: "f32[768]" = native_layer_norm_backward_70[2];  native_layer_norm_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_253: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_250, getitem_870);  add_250 = getitem_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_375: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_253, primals_8);  primals_8 = None
    mul_376: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_253, clone_16);  clone_16 = None
    sum_239: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1], True);  mul_376 = None
    view_941: "f32[768]" = torch.ops.aten.view.default(sum_239, [768]);  sum_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_942: "f32[6272, 768]" = torch.ops.aten.view.default(mul_375, [6272, 768]);  mul_375 = None
    t_513: "f32[768, 3072]" = torch.ops.aten.t.default(t_7);  t_7 = None
    mm_228: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_942, t_513);  t_513 = None
    t_514: "f32[768, 6272]" = torch.ops.aten.t.default(view_942)
    mm_229: "f32[768, 3072]" = torch.ops.aten.mm.default(t_514, view_30);  t_514 = view_30 = None
    t_515: "f32[3072, 768]" = torch.ops.aten.t.default(mm_229);  mm_229 = None
    sum_240: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_942, [0], True);  view_942 = None
    view_943: "f32[768]" = torch.ops.aten.view.default(sum_240, [768]);  sum_240 = None
    t_516: "f32[768, 3072]" = torch.ops.aten.t.default(t_515);  t_515 = None
    view_944: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_228, [8, 784, 3072]);  mm_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_46: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_944, view_29);  view_944 = view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_945: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_46, [6272, 3072]);  gelu_backward_46 = None
    t_517: "f32[3072, 768]" = torch.ops.aten.t.default(t_6);  t_6 = None
    mm_230: "f32[6272, 768]" = torch.ops.aten.mm.default(view_945, t_517);  t_517 = None
    t_518: "f32[3072, 6272]" = torch.ops.aten.t.default(view_945)
    mm_231: "f32[3072, 768]" = torch.ops.aten.mm.default(t_518, view_28);  t_518 = view_28 = None
    t_519: "f32[768, 3072]" = torch.ops.aten.t.default(mm_231);  mm_231 = None
    sum_241: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_945, [0], True);  view_945 = None
    view_946: "f32[3072]" = torch.ops.aten.view.default(sum_241, [3072]);  sum_241 = None
    t_520: "f32[3072, 768]" = torch.ops.aten.t.default(t_519);  t_519 = None
    view_947: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_230, [8, 784, 768]);  mm_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_71 = torch.ops.aten.native_layer_norm_backward.default(view_947, add_14, [768], getitem_47, getitem_48, primals_147, primals_148, [True, True, True]);  view_947 = add_14 = getitem_47 = getitem_48 = primals_147 = primals_148 = None
    getitem_873: "f32[8, 784, 768]" = native_layer_norm_backward_71[0]
    getitem_874: "f32[768]" = native_layer_norm_backward_71[1]
    getitem_875: "f32[768]" = native_layer_norm_backward_71[2];  native_layer_norm_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_254: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_253, getitem_873);  add_253 = getitem_873 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_377: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_254, primals_7);  primals_7 = None
    mul_378: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_254, permute_9);  permute_9 = None
    sum_242: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_378, [0, 1], True);  mul_378 = None
    view_948: "f32[768]" = torch.ops.aten.view.default(sum_242, [768]);  sum_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_198: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_377, [0, 2, 1]);  mul_377 = None
    view_949: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_198, [8, 768, 28, 28]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(view_949, getitem_41, primals_145, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_949 = getitem_41 = primals_145 = None
    getitem_876: "f32[8, 768, 28, 28]" = convolution_backward_44[0]
    getitem_877: "f32[768, 1, 3, 3]" = convolution_backward_44[1]
    getitem_878: "f32[768]" = convolution_backward_44[2];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_22 = torch.ops.aten.native_batch_norm_backward.default(getitem_876, gelu_4, primals_143, getitem_44, getitem_45, getitem_42, getitem_43, True, 1e-05, [True, True, True]);  getitem_876 = gelu_4 = primals_143 = getitem_42 = getitem_43 = None
    getitem_879: "f32[8, 768, 28, 28]" = native_batch_norm_backward_22[0]
    getitem_880: "f32[768]" = native_batch_norm_backward_22[1]
    getitem_881: "f32[768]" = native_batch_norm_backward_22[2];  native_batch_norm_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_47: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_879, convolution_6);  getitem_879 = convolution_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(gelu_backward_47, view_26, primals_141, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_47 = view_26 = primals_141 = None
    getitem_882: "f32[8, 768, 28, 28]" = convolution_backward_45[0]
    getitem_883: "f32[768, 1, 3, 3]" = convolution_backward_45[1]
    getitem_884: "f32[768]" = convolution_backward_45[2];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_950: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_882, [8, 768, 784]);  getitem_882 = None
    permute_199: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_950, [0, 2, 1]);  view_950 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_72 = torch.ops.aten.native_layer_norm_backward.default(permute_199, add_12, [768], getitem_39, getitem_40, primals_139, primals_140, [True, True, True]);  permute_199 = add_12 = getitem_39 = getitem_40 = primals_139 = primals_140 = None
    getitem_885: "f32[8, 784, 768]" = native_layer_norm_backward_72[0]
    getitem_886: "f32[768]" = native_layer_norm_backward_72[1]
    getitem_887: "f32[768]" = native_layer_norm_backward_72[2];  native_layer_norm_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_255: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_254, getitem_885);  add_254 = getitem_885 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_379: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_255, primals_5);  primals_5 = None
    mul_380: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_255, clone_14);  clone_14 = None
    sum_243: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_380, [0, 1], True);  mul_380 = None
    view_951: "f32[768]" = torch.ops.aten.view.default(sum_243, [768]);  sum_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_244: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 1], True)
    view_952: "f32[768]" = torch.ops.aten.view.default(sum_244, [768]);  sum_244 = None
    view_953: "f32[6272, 768]" = torch.ops.aten.view.default(mul_379, [6272, 768]);  mul_379 = None
    t_521: "f32[768, 6272]" = torch.ops.aten.t.default(view_953)
    mm_232: "f32[768, 768]" = torch.ops.aten.mm.default(t_521, _unsafe_view_7);  t_521 = _unsafe_view_7 = None
    t_522: "f32[768, 768]" = torch.ops.aten.t.default(mm_232);  mm_232 = None
    t_523: "f32[768, 768]" = torch.ops.aten.t.default(t_5);  t_5 = None
    mm_233: "f32[6272, 768]" = torch.ops.aten.mm.default(view_953, t_523);  view_953 = t_523 = None
    view_954: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_233, [8, 784, 768]);  mm_233 = None
    t_524: "f32[768, 768]" = torch.ops.aten.t.default(t_522);  t_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_955: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_954, [8, 784, 16, 48]);  view_954 = None
    permute_200: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_955, [0, 2, 3, 1]);  view_955 = None
    clone_244: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
    _unsafe_view_140: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_244, [128, 48, 784]);  clone_244 = None
    transpose_139: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_22, 1, 2);  view_22 = None
    bmm_136: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_139, _unsafe_view_140);  transpose_139 = None
    transpose_140: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_6, 1, 2);  _unsafe_view_6 = None
    bmm_137: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_140, transpose_140);  _unsafe_view_140 = transpose_140 = None
    view_956: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_136, [8, 16, 48, 784]);  bmm_136 = None
    view_957: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_137, [8, 16, 48, 48]);  bmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_142: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_5);  detach_5 = None
    _softmax_backward_data_22: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_957, detach_142, -1, torch.float32);  view_957 = detach_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_381: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_22, view_21);  view_21 = None
    mul_382: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_22, primals_6);  _softmax_backward_data_22 = primals_6 = None
    sum_245: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [0, 2, 3], True);  mul_381 = None
    view_958: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_245, [16, 1, 1]);  sum_245 = None
    view_959: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_382, [128, 48, 48]);  mul_382 = None
    transpose_141: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_4, 1, 2);  _unsafe_view_4 = None
    bmm_138: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_141, view_959);  transpose_141 = None
    transpose_142: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_5, 1, 2);  _unsafe_view_5 = None
    bmm_139: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_959, transpose_142);  view_959 = transpose_142 = None
    view_960: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_138, [8, 16, 784, 48]);  bmm_138 = None
    view_961: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_139, [8, 16, 48, 784]);  bmm_139 = None
    transpose_143: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_960, -2, -1);  view_960 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_230: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_36, expand_7)
    div_231: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_230, expand_7);  div_230 = None
    neg_44: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_143)
    mul_383: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_44, div_231);  neg_44 = div_231 = None
    div_232: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_143, expand_7);  transpose_143 = expand_7 = None
    sum_246: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [3], True);  mul_383 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_44: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_3, 1e-12);  linalg_vector_norm_3 = None
    where_44: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_44, sum_246, scalar_tensor_44);  ge_44 = sum_246 = scalar_tensor_44 = None
    detach_143: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_4);  detach_4 = None
    div_233: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_36, detach_143);  getitem_36 = None
    eq_44: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_143, 0);  detach_143 = None
    masked_fill_44: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_233, eq_44, 0);  div_233 = eq_44 = None
    mul_384: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_44, masked_fill_44);  where_44 = masked_fill_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_256: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_232, mul_384);  div_232 = mul_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_234: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_35, expand_6)
    div_235: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_234, expand_6);  div_234 = None
    neg_45: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_961)
    mul_385: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_45, div_235);  neg_45 = div_235 = None
    div_236: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_961, expand_6);  view_961 = expand_6 = None
    sum_247: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [3], True);  mul_385 = None
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_45: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_2, 1e-12);  linalg_vector_norm_2 = None
    where_45: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_45, sum_247, scalar_tensor_45);  ge_45 = sum_247 = scalar_tensor_45 = None
    detach_144: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_3);  detach_3 = None
    div_237: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_35, detach_144);  getitem_35 = None
    eq_45: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_144, 0);  detach_144 = None
    masked_fill_45: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_237, eq_45, 0);  div_237 = eq_45 = None
    mul_386: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_45, masked_fill_45);  where_45 = masked_fill_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_257: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_236, mul_386);  div_236 = mul_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_24: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_257, add_256, view_956]);  add_257 = add_256 = view_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_201: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_24, [1, 4, 0, 2, 3]);  stack_24 = None
    clone_245: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
    _unsafe_view_141: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_245, [8, 784, 2304]);  clone_245 = None
    view_962: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_141, [6272, 2304]);  _unsafe_view_141 = None
    t_525: "f32[2304, 768]" = torch.ops.aten.t.default(t_4);  t_4 = None
    mm_234: "f32[6272, 768]" = torch.ops.aten.mm.default(view_962, t_525);  t_525 = None
    t_526: "f32[2304, 6272]" = torch.ops.aten.t.default(view_962)
    mm_235: "f32[2304, 768]" = torch.ops.aten.mm.default(t_526, view_18);  t_526 = view_18 = None
    t_527: "f32[768, 2304]" = torch.ops.aten.t.default(mm_235);  mm_235 = None
    sum_248: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_962, [0], True);  view_962 = None
    view_963: "f32[2304]" = torch.ops.aten.view.default(sum_248, [2304]);  sum_248 = None
    t_528: "f32[2304, 768]" = torch.ops.aten.t.default(t_527);  t_527 = None
    view_964: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_234, [8, 784, 768]);  mm_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_73 = torch.ops.aten.native_layer_norm_backward.default(view_964, add_10, [768], getitem_33, getitem_34, primals_133, primals_134, [True, True, True]);  view_964 = add_10 = getitem_33 = getitem_34 = primals_133 = primals_134 = None
    getitem_888: "f32[8, 784, 768]" = native_layer_norm_backward_73[0]
    getitem_889: "f32[768]" = native_layer_norm_backward_73[1]
    getitem_890: "f32[768]" = native_layer_norm_backward_73[2];  native_layer_norm_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_258: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_255, getitem_888);  add_255 = getitem_888 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_387: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_258, primals_4);  primals_4 = None
    mul_388: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_258, clone_8);  clone_8 = None
    sum_249: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1], True);  mul_388 = None
    view_965: "f32[768]" = torch.ops.aten.view.default(sum_249, [768]);  sum_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_966: "f32[6272, 768]" = torch.ops.aten.view.default(mul_387, [6272, 768]);  mul_387 = None
    t_529: "f32[768, 3072]" = torch.ops.aten.t.default(t_3);  t_3 = None
    mm_236: "f32[6272, 3072]" = torch.ops.aten.mm.default(view_966, t_529);  t_529 = None
    t_530: "f32[768, 6272]" = torch.ops.aten.t.default(view_966)
    mm_237: "f32[768, 3072]" = torch.ops.aten.mm.default(t_530, view_16);  t_530 = view_16 = None
    t_531: "f32[3072, 768]" = torch.ops.aten.t.default(mm_237);  mm_237 = None
    sum_250: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_966, [0], True);  view_966 = None
    view_967: "f32[768]" = torch.ops.aten.view.default(sum_250, [768]);  sum_250 = None
    t_532: "f32[768, 3072]" = torch.ops.aten.t.default(t_531);  t_531 = None
    view_968: "f32[8, 784, 3072]" = torch.ops.aten.view.default(mm_236, [8, 784, 3072]);  mm_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_backward_48: "f32[8, 784, 3072]" = torch.ops.aten.gelu_backward.default(view_968, view_15);  view_968 = view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_969: "f32[6272, 3072]" = torch.ops.aten.view.default(gelu_backward_48, [6272, 3072]);  gelu_backward_48 = None
    t_533: "f32[3072, 768]" = torch.ops.aten.t.default(t_2);  t_2 = None
    mm_238: "f32[6272, 768]" = torch.ops.aten.mm.default(view_969, t_533);  t_533 = None
    t_534: "f32[3072, 6272]" = torch.ops.aten.t.default(view_969)
    mm_239: "f32[3072, 768]" = torch.ops.aten.mm.default(t_534, view_14);  t_534 = view_14 = None
    t_535: "f32[768, 3072]" = torch.ops.aten.t.default(mm_239);  mm_239 = None
    sum_251: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_969, [0], True);  view_969 = None
    view_970: "f32[3072]" = torch.ops.aten.view.default(sum_251, [3072]);  sum_251 = None
    t_536: "f32[3072, 768]" = torch.ops.aten.t.default(t_535);  t_535 = None
    view_971: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_238, [8, 784, 768]);  mm_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    native_layer_norm_backward_74 = torch.ops.aten.native_layer_norm_backward.default(view_971, add_9, [768], getitem_30, getitem_31, primals_127, primals_128, [True, True, True]);  view_971 = add_9 = getitem_30 = getitem_31 = primals_127 = primals_128 = None
    getitem_891: "f32[8, 784, 768]" = native_layer_norm_backward_74[0]
    getitem_892: "f32[768]" = native_layer_norm_backward_74[1]
    getitem_893: "f32[768]" = native_layer_norm_backward_74[2];  native_layer_norm_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    add_259: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_258, getitem_891);  add_258 = getitem_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_389: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_259, primals_3);  primals_3 = None
    mul_390: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_259, permute_5);  permute_5 = None
    sum_252: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 1], True);  mul_390 = None
    view_972: "f32[768]" = torch.ops.aten.view.default(sum_252, [768]);  sum_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    permute_202: "f32[8, 768, 784]" = torch.ops.aten.permute.default(mul_389, [0, 2, 1]);  mul_389 = None
    view_973: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_202, [8, 768, 28, 28]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(view_973, getitem_24, primals_125, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  view_973 = getitem_24 = primals_125 = None
    getitem_894: "f32[8, 768, 28, 28]" = convolution_backward_46[0]
    getitem_895: "f32[768, 1, 3, 3]" = convolution_backward_46[1]
    getitem_896: "f32[768]" = convolution_backward_46[2];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    native_batch_norm_backward_23 = torch.ops.aten.native_batch_norm_backward.default(getitem_894, gelu_2, primals_123, getitem_27, getitem_28, getitem_25, getitem_26, True, 1e-05, [True, True, True]);  getitem_894 = gelu_2 = primals_123 = getitem_25 = getitem_26 = None
    getitem_897: "f32[8, 768, 28, 28]" = native_batch_norm_backward_23[0]
    getitem_898: "f32[768]" = native_batch_norm_backward_23[1]
    getitem_899: "f32[768]" = native_batch_norm_backward_23[2];  native_batch_norm_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    gelu_backward_49: "f32[8, 768, 28, 28]" = torch.ops.aten.gelu_backward.default(getitem_897, convolution_4);  getitem_897 = convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(gelu_backward_49, view_12, primals_121, [768], [1, 1], [1, 1], [1, 1], False, [0, 0], 768, [True, True, True]);  gelu_backward_49 = view_12 = primals_121 = None
    getitem_900: "f32[8, 768, 28, 28]" = convolution_backward_47[0]
    getitem_901: "f32[768, 1, 3, 3]" = convolution_backward_47[1]
    getitem_902: "f32[768]" = convolution_backward_47[2];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    view_974: "f32[8, 768, 784]" = torch.ops.aten.view.default(getitem_900, [8, 768, 784]);  getitem_900 = None
    permute_203: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_974, [0, 2, 1]);  view_974 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    native_layer_norm_backward_75 = torch.ops.aten.native_layer_norm_backward.default(permute_203, add_7, [768], getitem_22, getitem_23, primals_119, primals_120, [True, True, True]);  permute_203 = add_7 = getitem_22 = getitem_23 = primals_119 = primals_120 = None
    getitem_903: "f32[8, 784, 768]" = native_layer_norm_backward_75[0]
    getitem_904: "f32[768]" = native_layer_norm_backward_75[1]
    getitem_905: "f32[768]" = native_layer_norm_backward_75[2];  native_layer_norm_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    add_260: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_259, getitem_903);  add_259 = getitem_903 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_391: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_260, primals_1);  primals_1 = None
    mul_392: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(add_260, clone_6);  clone_6 = None
    sum_253: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 1], True);  mul_392 = None
    view_975: "f32[768]" = torch.ops.aten.view.default(sum_253, [768]);  sum_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    sum_254: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1], True)
    view_976: "f32[768]" = torch.ops.aten.view.default(sum_254, [768]);  sum_254 = None
    view_977: "f32[6272, 768]" = torch.ops.aten.view.default(mul_391, [6272, 768]);  mul_391 = None
    t_537: "f32[768, 6272]" = torch.ops.aten.t.default(view_977)
    mm_240: "f32[768, 768]" = torch.ops.aten.mm.default(t_537, _unsafe_view_3);  t_537 = _unsafe_view_3 = None
    t_538: "f32[768, 768]" = torch.ops.aten.t.default(mm_240);  mm_240 = None
    t_539: "f32[768, 768]" = torch.ops.aten.t.default(t_1);  t_1 = None
    mm_241: "f32[6272, 768]" = torch.ops.aten.mm.default(view_977, t_539);  view_977 = t_539 = None
    view_978: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_241, [8, 784, 768]);  mm_241 = None
    t_540: "f32[768, 768]" = torch.ops.aten.t.default(t_538);  t_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    view_979: "f32[8, 784, 16, 48]" = torch.ops.aten.view.default(view_978, [8, 784, 16, 48]);  view_978 = None
    permute_204: "f32[8, 16, 48, 784]" = torch.ops.aten.permute.default(view_979, [0, 2, 3, 1]);  view_979 = None
    clone_246: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
    _unsafe_view_142: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_246, [128, 48, 784]);  clone_246 = None
    transpose_144: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_8, 1, 2);  view_8 = None
    bmm_140: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(transpose_144, _unsafe_view_142);  transpose_144 = None
    transpose_145: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_2, 1, 2);  _unsafe_view_2 = None
    bmm_141: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_142, transpose_145);  _unsafe_view_142 = transpose_145 = None
    view_980: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_140, [8, 16, 48, 784]);  bmm_140 = None
    view_981: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_141, [8, 16, 48, 48]);  bmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_145: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_2);  detach_2 = None
    _softmax_backward_data_23: "f32[8, 16, 48, 48]" = torch.ops.aten._softmax_backward_data.default(view_981, detach_145, -1, torch.float32);  view_981 = detach_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    mul_393: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_23, view_7);  view_7 = None
    mul_394: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(_softmax_backward_data_23, primals_2);  _softmax_backward_data_23 = primals_2 = None
    sum_255: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 2, 3], True);  mul_393 = None
    view_982: "f32[16, 1, 1]" = torch.ops.aten.view.default(sum_255, [16, 1, 1]);  sum_255 = None
    view_983: "f32[128, 48, 48]" = torch.ops.aten.view.default(mul_394, [128, 48, 48]);  mul_394 = None
    transpose_146: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view, 1, 2);  _unsafe_view = None
    bmm_142: "f32[128, 784, 48]" = torch.ops.aten.bmm.default(transpose_146, view_983);  transpose_146 = None
    transpose_147: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_1, 1, 2);  _unsafe_view_1 = None
    bmm_143: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_983, transpose_147);  view_983 = transpose_147 = None
    view_984: "f32[8, 16, 784, 48]" = torch.ops.aten.view.default(bmm_142, [8, 16, 784, 48]);  bmm_142 = None
    view_985: "f32[8, 16, 48, 784]" = torch.ops.aten.view.default(bmm_143, [8, 16, 48, 784]);  bmm_143 = None
    transpose_148: "f32[8, 16, 48, 784]" = torch.ops.aten.transpose.int(view_984, -2, -1);  view_984 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    div_238: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_19, expand_1)
    div_239: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_238, expand_1);  div_238 = None
    neg_46: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(transpose_148)
    mul_395: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_46, div_239);  neg_46 = div_239 = None
    div_240: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(transpose_148, expand_1);  transpose_148 = expand_1 = None
    sum_256: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [3], True);  mul_395 = None
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_46: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm_1, 1e-12);  linalg_vector_norm_1 = None
    where_46: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_46, sum_256, scalar_tensor_46);  ge_46 = sum_256 = scalar_tensor_46 = None
    detach_146: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach_1);  detach_1 = None
    div_241: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_19, detach_146);  getitem_19 = None
    eq_46: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_146, 0);  detach_146 = None
    masked_fill_46: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_241, eq_46, 0);  div_241 = eq_46 = None
    mul_396: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_46, masked_fill_46);  where_46 = masked_fill_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    add_261: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_240, mul_396);  div_240 = mul_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    div_242: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_18, expand)
    div_243: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(div_242, expand);  div_242 = None
    neg_47: "f32[8, 16, 48, 784]" = torch.ops.aten.neg.default(view_985)
    mul_397: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(neg_47, div_243);  neg_47 = div_243 = None
    div_244: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(view_985, expand);  view_985 = expand = None
    sum_257: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [3], True);  mul_397 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    ge_47: "b8[8, 16, 48, 1]" = torch.ops.aten.ge.Scalar(linalg_vector_norm, 1e-12);  linalg_vector_norm = None
    where_47: "f32[8, 16, 48, 1]" = torch.ops.aten.where.self(ge_47, sum_257, scalar_tensor_47);  ge_47 = sum_257 = scalar_tensor_47 = None
    detach_147: "f32[8, 16, 48, 1]" = torch.ops.aten.detach.default(detach);  detach = None
    div_245: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_18, detach_147);  getitem_18 = None
    eq_47: "b8[8, 16, 48, 1]" = torch.ops.aten.eq.Scalar(detach_147, 0);  detach_147 = None
    masked_fill_47: "f32[8, 16, 48, 784]" = torch.ops.aten.masked_fill.Scalar(div_245, eq_47, 0);  div_245 = eq_47 = None
    mul_398: "f32[8, 16, 48, 784]" = torch.ops.aten.mul.Tensor(where_47, masked_fill_47);  where_47 = masked_fill_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    add_262: "f32[8, 16, 48, 784]" = torch.ops.aten.add.Tensor(div_244, mul_398);  div_244 = mul_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    stack_25: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.stack.default([add_262, add_261, view_980]);  add_262 = add_261 = view_980 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_205: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.permute.default(stack_25, [1, 4, 0, 2, 3]);  stack_25 = None
    clone_247: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    _unsafe_view_143: "f32[8, 784, 2304]" = torch.ops.aten._unsafe_view.default(clone_247, [8, 784, 2304]);  clone_247 = None
    view_986: "f32[6272, 2304]" = torch.ops.aten.view.default(_unsafe_view_143, [6272, 2304]);  _unsafe_view_143 = None
    t_541: "f32[2304, 768]" = torch.ops.aten.t.default(t);  t = None
    mm_242: "f32[6272, 768]" = torch.ops.aten.mm.default(view_986, t_541);  t_541 = None
    t_542: "f32[2304, 6272]" = torch.ops.aten.t.default(view_986)
    mm_243: "f32[2304, 768]" = torch.ops.aten.mm.default(t_542, view_4);  t_542 = view_4 = None
    t_543: "f32[768, 2304]" = torch.ops.aten.t.default(mm_243);  mm_243 = None
    sum_258: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_986, [0], True);  view_986 = None
    view_987: "f32[2304]" = torch.ops.aten.view.default(sum_258, [2304]);  sum_258 = None
    t_544: "f32[2304, 768]" = torch.ops.aten.t.default(t_543);  t_543 = None
    view_988: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_242, [8, 784, 768]);  mm_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    native_layer_norm_backward_76 = torch.ops.aten.native_layer_norm_backward.default(view_988, clone, [768], getitem_16, getitem_17, primals_113, primals_114, [True, True, True]);  view_988 = clone = getitem_16 = getitem_17 = primals_113 = primals_114 = None
    getitem_906: "f32[8, 784, 768]" = native_layer_norm_backward_76[0]
    getitem_907: "f32[768]" = native_layer_norm_backward_76[1]
    getitem_908: "f32[768]" = native_layer_norm_backward_76[2];  native_layer_norm_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    add_263: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_260, getitem_906);  add_260 = getitem_906 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:437, code: pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
    permute_206: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_263, [0, 2, 1])
    view_989: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(permute_206, [8, 768, 28, 28]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:63, code: return pos.repeat(B, 1, 1, 1)  # (B, C, H, W)
    view_990: "f32[8, 1, 768, 28, 28]" = torch.ops.aten.view.default(view_989, [8, 1, 768, 28, 28]);  view_989 = None
    sum_259: "f32[1, 768, 28, 28]" = torch.ops.aten.sum.dim_IntList(view_990, [0]);  view_990 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:62, code: pos = self.token_projection(pos)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(sum_259, permute, primals_111, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  sum_259 = permute = primals_111 = None
    getitem_910: "f32[768, 64, 1, 1]" = convolution_backward_48[1]
    getitem_911: "f32[768]" = convolution_backward_48[2];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:109, code: x = x.flatten(2).transpose(1, 2)  # (B, N, C)
    transpose_149: "f32[8, 768, 784]" = torch.ops.aten.transpose.int(add_263, 1, 2);  add_263 = None
    view_991: "f32[8, 768, 28, 28]" = torch.ops.aten.view.default(transpose_149, [8, 768, 28, 28]);  transpose_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:107, code: x = self.proj(x)
    native_batch_norm_backward_24 = torch.ops.aten.native_batch_norm_backward.default(view_991, convolution_2, primals_109, getitem_13, getitem_14, getitem_11, getitem_12, True, 1e-05, [True, True, True]);  view_991 = convolution_2 = primals_109 = getitem_11 = getitem_12 = None
    getitem_912: "f32[8, 768, 28, 28]" = native_batch_norm_backward_24[0]
    getitem_913: "f32[768]" = native_batch_norm_backward_24[1]
    getitem_914: "f32[768]" = native_batch_norm_backward_24[2];  native_batch_norm_backward_24 = None
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(getitem_912, gelu_1, primals_108, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_912 = gelu_1 = primals_108 = None
    getitem_915: "f32[8, 384, 56, 56]" = convolution_backward_49[0]
    getitem_916: "f32[768, 384, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    gelu_backward_50: "f32[8, 384, 56, 56]" = torch.ops.aten.gelu_backward.default(getitem_915, getitem_5);  getitem_915 = getitem_5 = None
    native_batch_norm_backward_25 = torch.ops.aten.native_batch_norm_backward.default(gelu_backward_50, convolution_1, primals_106, getitem_8, getitem_9, getitem_6, getitem_7, True, 1e-05, [True, True, True]);  gelu_backward_50 = convolution_1 = primals_106 = getitem_6 = getitem_7 = None
    getitem_918: "f32[8, 384, 56, 56]" = native_batch_norm_backward_25[0]
    getitem_919: "f32[384]" = native_batch_norm_backward_25[1]
    getitem_920: "f32[384]" = native_batch_norm_backward_25[2];  native_batch_norm_backward_25 = None
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(getitem_918, gelu, primals_105, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  getitem_918 = gelu = primals_105 = None
    getitem_921: "f32[8, 192, 112, 112]" = convolution_backward_50[0]
    getitem_922: "f32[384, 192, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    gelu_backward_51: "f32[8, 192, 112, 112]" = torch.ops.aten.gelu_backward.default(getitem_921, getitem);  getitem_921 = getitem = None
    native_batch_norm_backward_26 = torch.ops.aten.native_batch_norm_backward.default(gelu_backward_51, convolution, primals_103, getitem_3, getitem_4, getitem_1, getitem_2, True, 1e-05, [True, True, True]);  gelu_backward_51 = convolution = primals_103 = getitem_1 = getitem_2 = None
    getitem_924: "f32[8, 192, 112, 112]" = native_batch_norm_backward_26[0]
    getitem_925: "f32[192]" = native_batch_norm_backward_26[1]
    getitem_926: "f32[192]" = native_batch_norm_backward_26[2];  native_batch_norm_backward_26 = None
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(getitem_924, primals_710, primals_102, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  getitem_924 = primals_710 = primals_102 = None
    getitem_928: "f32[192, 3, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    return pytree.tree_unflatten([getitem_3, getitem_4, add, getitem_8, getitem_9, add_1, getitem_13, getitem_14, add_2, getitem_27, getitem_28, add_8, getitem_44, getitem_45, add_13, getitem_61, getitem_62, add_18, getitem_78, getitem_79, add_23, getitem_95, getitem_96, add_28, getitem_112, getitem_113, add_33, getitem_129, getitem_130, add_38, getitem_146, getitem_147, add_43, getitem_163, getitem_164, add_48, getitem_180, getitem_181, add_53, getitem_197, getitem_198, add_58, getitem_214, getitem_215, add_63, getitem_231, getitem_232, add_68, getitem_248, getitem_249, add_73, getitem_265, getitem_266, add_78, getitem_282, getitem_283, add_83, getitem_299, getitem_300, add_88, getitem_316, getitem_317, add_93, getitem_333, getitem_334, add_98, getitem_350, getitem_351, add_103, getitem_367, getitem_368, add_108, getitem_384, getitem_385, add_113, getitem_401, getitem_402, add_118, getitem_418, getitem_419, add_123, addmm_82, view_975, view_982, view_972, view_965, view_951, view_958, view_948, view_941, view_927, view_934, view_924, view_917, view_903, view_910, view_900, view_893, view_879, view_886, view_876, view_869, view_855, view_862, view_852, view_845, view_831, view_838, view_828, view_821, view_807, view_814, view_804, view_797, view_783, view_790, view_780, view_773, view_759, view_766, view_756, view_749, view_735, view_742, view_732, view_725, view_711, view_718, view_708, view_701, view_687, view_694, view_684, view_677, view_663, view_670, view_660, view_653, view_639, view_646, view_636, view_629, view_615, view_622, view_612, view_605, view_591, view_598, view_588, view_581, view_567, view_574, view_564, view_557, view_543, view_550, view_540, view_533, view_519, view_526, view_516, view_509, view_495, view_502, view_492, view_485, view_471, view_478, view_468, view_461, view_447, view_454, view_444, view_437, view_423, view_430, view_420, view_413, sum_18, view_398, view_391, view_376, view_369, getitem_928, getitem_925, getitem_926, getitem_922, getitem_919, getitem_920, getitem_916, getitem_913, getitem_914, getitem_910, getitem_911, getitem_907, getitem_908, t_544, view_987, t_540, view_976, getitem_904, getitem_905, getitem_901, getitem_902, getitem_898, getitem_899, getitem_895, getitem_896, getitem_892, getitem_893, t_536, view_970, t_532, view_967, getitem_889, getitem_890, t_528, view_963, t_524, view_952, getitem_886, getitem_887, getitem_883, getitem_884, getitem_880, getitem_881, getitem_877, getitem_878, getitem_874, getitem_875, t_520, view_946, t_516, view_943, getitem_871, getitem_872, t_512, view_939, t_508, view_928, getitem_868, getitem_869, getitem_865, getitem_866, getitem_862, getitem_863, getitem_859, getitem_860, getitem_856, getitem_857, t_504, view_922, t_500, view_919, getitem_853, getitem_854, t_496, view_915, t_492, view_904, getitem_850, getitem_851, getitem_847, getitem_848, getitem_844, getitem_845, getitem_841, getitem_842, getitem_838, getitem_839, t_488, view_898, t_484, view_895, getitem_835, getitem_836, t_480, view_891, t_476, view_880, getitem_832, getitem_833, getitem_829, getitem_830, getitem_826, getitem_827, getitem_823, getitem_824, getitem_820, getitem_821, t_472, view_874, t_468, view_871, getitem_817, getitem_818, t_464, view_867, t_460, view_856, getitem_814, getitem_815, getitem_811, getitem_812, getitem_808, getitem_809, getitem_805, getitem_806, getitem_802, getitem_803, t_456, view_850, t_452, view_847, getitem_799, getitem_800, t_448, view_843, t_444, view_832, getitem_796, getitem_797, getitem_793, getitem_794, getitem_790, getitem_791, getitem_787, getitem_788, getitem_784, getitem_785, t_440, view_826, t_436, view_823, getitem_781, getitem_782, t_432, view_819, t_428, view_808, getitem_778, getitem_779, getitem_775, getitem_776, getitem_772, getitem_773, getitem_769, getitem_770, getitem_766, getitem_767, t_424, view_802, t_420, view_799, getitem_763, getitem_764, t_416, view_795, t_412, view_784, getitem_760, getitem_761, getitem_757, getitem_758, getitem_754, getitem_755, getitem_751, getitem_752, getitem_748, getitem_749, t_408, view_778, t_404, view_775, getitem_745, getitem_746, t_400, view_771, t_396, view_760, getitem_742, getitem_743, getitem_739, getitem_740, getitem_736, getitem_737, getitem_733, getitem_734, getitem_730, getitem_731, t_392, view_754, t_388, view_751, getitem_727, getitem_728, t_384, view_747, t_380, view_736, getitem_724, getitem_725, getitem_721, getitem_722, getitem_718, getitem_719, getitem_715, getitem_716, getitem_712, getitem_713, t_376, view_730, t_372, view_727, getitem_709, getitem_710, t_368, view_723, t_364, view_712, getitem_706, getitem_707, getitem_703, getitem_704, getitem_700, getitem_701, getitem_697, getitem_698, getitem_694, getitem_695, t_360, view_706, t_356, view_703, getitem_691, getitem_692, t_352, view_699, t_348, view_688, getitem_688, getitem_689, getitem_685, getitem_686, getitem_682, getitem_683, getitem_679, getitem_680, getitem_676, getitem_677, t_344, view_682, t_340, view_679, getitem_673, getitem_674, t_336, view_675, t_332, view_664, getitem_670, getitem_671, getitem_667, getitem_668, getitem_664, getitem_665, getitem_661, getitem_662, getitem_658, getitem_659, t_328, view_658, t_324, view_655, getitem_655, getitem_656, t_320, view_651, t_316, view_640, getitem_652, getitem_653, getitem_649, getitem_650, getitem_646, getitem_647, getitem_643, getitem_644, getitem_640, getitem_641, t_312, view_634, t_308, view_631, getitem_637, getitem_638, t_304, view_627, t_300, view_616, getitem_634, getitem_635, getitem_631, getitem_632, getitem_628, getitem_629, getitem_625, getitem_626, getitem_622, getitem_623, t_296, view_610, t_292, view_607, getitem_619, getitem_620, t_288, view_603, t_284, view_592, getitem_616, getitem_617, getitem_613, getitem_614, getitem_610, getitem_611, getitem_607, getitem_608, getitem_604, getitem_605, t_280, view_586, t_276, view_583, getitem_601, getitem_602, t_272, view_579, t_268, view_568, getitem_598, getitem_599, getitem_595, getitem_596, getitem_592, getitem_593, getitem_589, getitem_590, getitem_586, getitem_587, t_264, view_562, t_260, view_559, getitem_583, getitem_584, t_256, view_555, t_252, view_544, getitem_580, getitem_581, getitem_577, getitem_578, getitem_574, getitem_575, getitem_571, getitem_572, getitem_568, getitem_569, t_248, view_538, t_244, view_535, getitem_565, getitem_566, t_240, view_531, t_236, view_520, getitem_562, getitem_563, getitem_559, getitem_560, getitem_556, getitem_557, getitem_553, getitem_554, getitem_550, getitem_551, t_232, view_514, t_228, view_511, getitem_547, getitem_548, t_224, view_507, t_220, view_496, getitem_544, getitem_545, getitem_541, getitem_542, getitem_538, getitem_539, getitem_535, getitem_536, getitem_532, getitem_533, t_216, view_490, t_212, view_487, getitem_529, getitem_530, t_208, view_483, t_204, view_472, getitem_526, getitem_527, getitem_523, getitem_524, getitem_520, getitem_521, getitem_517, getitem_518, getitem_514, getitem_515, t_200, view_466, t_196, view_463, getitem_511, getitem_512, t_192, view_459, t_188, view_448, getitem_508, getitem_509, getitem_505, getitem_506, getitem_502, getitem_503, getitem_499, getitem_500, getitem_496, getitem_497, t_184, view_442, t_180, view_439, getitem_493, getitem_494, t_176, view_435, t_172, view_424, getitem_490, getitem_491, getitem_487, getitem_488, getitem_484, getitem_485, getitem_481, getitem_482, getitem_478, getitem_479, t_168, view_418, t_164, view_415, getitem_475, getitem_476, t_160, view_412, t_156, view_409, t_152, view_405, t_148, view_400, getitem_469, getitem_470, t_144, view_395, t_140, view_393, getitem_466, getitem_467, t_136, view_390, t_132, view_387, t_128, view_383, t_124, view_378, getitem_460, getitem_461, t_120, view_373, t_116, view_371, getitem_457, getitem_458, t_112, view_368, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    