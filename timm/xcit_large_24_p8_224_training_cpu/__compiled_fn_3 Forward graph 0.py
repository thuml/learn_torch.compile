from __future__ import annotations



def forward(self, primals_1: "f32[768]", primals_2: "f32[16, 1, 1]", primals_3: "f32[768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[16, 1, 1]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[16, 1, 1]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_14: "f32[16, 1, 1]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[16, 1, 1]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[16, 1, 1]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[16, 1, 1]", primals_27: "f32[768]", primals_28: "f32[768]", primals_29: "f32[768]", primals_30: "f32[16, 1, 1]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[16, 1, 1]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[16, 1, 1]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[16, 1, 1]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[768]", primals_46: "f32[16, 1, 1]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[768]", primals_50: "f32[16, 1, 1]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[16, 1, 1]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[16, 1, 1]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[16, 1, 1]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[16, 1, 1]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[16, 1, 1]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[16, 1, 1]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[768]", primals_78: "f32[16, 1, 1]", primals_79: "f32[768]", primals_80: "f32[768]", primals_81: "f32[768]", primals_82: "f32[16, 1, 1]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[16, 1, 1]", primals_87: "f32[768]", primals_88: "f32[768]", primals_89: "f32[768]", primals_90: "f32[16, 1, 1]", primals_91: "f32[768]", primals_92: "f32[768]", primals_93: "f32[768]", primals_94: "f32[16, 1, 1]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[1, 1, 768]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[192, 3, 3, 3]", primals_103: "f32[192]", primals_104: "f32[192]", primals_105: "f32[384, 192, 3, 3]", primals_106: "f32[384]", primals_107: "f32[384]", primals_108: "f32[768, 384, 3, 3]", primals_109: "f32[768]", primals_110: "f32[768]", primals_111: "f32[768, 64, 1, 1]", primals_112: "f32[768]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[2304, 768]", primals_116: "f32[2304]", primals_117: "f32[768, 768]", primals_118: "f32[768]", primals_119: "f32[768]", primals_120: "f32[768]", primals_121: "f32[768, 1, 3, 3]", primals_122: "f32[768]", primals_123: "f32[768]", primals_124: "f32[768]", primals_125: "f32[768, 1, 3, 3]", primals_126: "f32[768]", primals_127: "f32[768]", primals_128: "f32[768]", primals_129: "f32[3072, 768]", primals_130: "f32[3072]", primals_131: "f32[768, 3072]", primals_132: "f32[768]", primals_133: "f32[768]", primals_134: "f32[768]", primals_135: "f32[2304, 768]", primals_136: "f32[2304]", primals_137: "f32[768, 768]", primals_138: "f32[768]", primals_139: "f32[768]", primals_140: "f32[768]", primals_141: "f32[768, 1, 3, 3]", primals_142: "f32[768]", primals_143: "f32[768]", primals_144: "f32[768]", primals_145: "f32[768, 1, 3, 3]", primals_146: "f32[768]", primals_147: "f32[768]", primals_148: "f32[768]", primals_149: "f32[3072, 768]", primals_150: "f32[3072]", primals_151: "f32[768, 3072]", primals_152: "f32[768]", primals_153: "f32[768]", primals_154: "f32[768]", primals_155: "f32[2304, 768]", primals_156: "f32[2304]", primals_157: "f32[768, 768]", primals_158: "f32[768]", primals_159: "f32[768]", primals_160: "f32[768]", primals_161: "f32[768, 1, 3, 3]", primals_162: "f32[768]", primals_163: "f32[768]", primals_164: "f32[768]", primals_165: "f32[768, 1, 3, 3]", primals_166: "f32[768]", primals_167: "f32[768]", primals_168: "f32[768]", primals_169: "f32[3072, 768]", primals_170: "f32[3072]", primals_171: "f32[768, 3072]", primals_172: "f32[768]", primals_173: "f32[768]", primals_174: "f32[768]", primals_175: "f32[2304, 768]", primals_176: "f32[2304]", primals_177: "f32[768, 768]", primals_178: "f32[768]", primals_179: "f32[768]", primals_180: "f32[768]", primals_181: "f32[768, 1, 3, 3]", primals_182: "f32[768]", primals_183: "f32[768]", primals_184: "f32[768]", primals_185: "f32[768, 1, 3, 3]", primals_186: "f32[768]", primals_187: "f32[768]", primals_188: "f32[768]", primals_189: "f32[3072, 768]", primals_190: "f32[3072]", primals_191: "f32[768, 3072]", primals_192: "f32[768]", primals_193: "f32[768]", primals_194: "f32[768]", primals_195: "f32[2304, 768]", primals_196: "f32[2304]", primals_197: "f32[768, 768]", primals_198: "f32[768]", primals_199: "f32[768]", primals_200: "f32[768]", primals_201: "f32[768, 1, 3, 3]", primals_202: "f32[768]", primals_203: "f32[768]", primals_204: "f32[768]", primals_205: "f32[768, 1, 3, 3]", primals_206: "f32[768]", primals_207: "f32[768]", primals_208: "f32[768]", primals_209: "f32[3072, 768]", primals_210: "f32[3072]", primals_211: "f32[768, 3072]", primals_212: "f32[768]", primals_213: "f32[768]", primals_214: "f32[768]", primals_215: "f32[2304, 768]", primals_216: "f32[2304]", primals_217: "f32[768, 768]", primals_218: "f32[768]", primals_219: "f32[768]", primals_220: "f32[768]", primals_221: "f32[768, 1, 3, 3]", primals_222: "f32[768]", primals_223: "f32[768]", primals_224: "f32[768]", primals_225: "f32[768, 1, 3, 3]", primals_226: "f32[768]", primals_227: "f32[768]", primals_228: "f32[768]", primals_229: "f32[3072, 768]", primals_230: "f32[3072]", primals_231: "f32[768, 3072]", primals_232: "f32[768]", primals_233: "f32[768]", primals_234: "f32[768]", primals_235: "f32[2304, 768]", primals_236: "f32[2304]", primals_237: "f32[768, 768]", primals_238: "f32[768]", primals_239: "f32[768]", primals_240: "f32[768]", primals_241: "f32[768, 1, 3, 3]", primals_242: "f32[768]", primals_243: "f32[768]", primals_244: "f32[768]", primals_245: "f32[768, 1, 3, 3]", primals_246: "f32[768]", primals_247: "f32[768]", primals_248: "f32[768]", primals_249: "f32[3072, 768]", primals_250: "f32[3072]", primals_251: "f32[768, 3072]", primals_252: "f32[768]", primals_253: "f32[768]", primals_254: "f32[768]", primals_255: "f32[2304, 768]", primals_256: "f32[2304]", primals_257: "f32[768, 768]", primals_258: "f32[768]", primals_259: "f32[768]", primals_260: "f32[768]", primals_261: "f32[768, 1, 3, 3]", primals_262: "f32[768]", primals_263: "f32[768]", primals_264: "f32[768]", primals_265: "f32[768, 1, 3, 3]", primals_266: "f32[768]", primals_267: "f32[768]", primals_268: "f32[768]", primals_269: "f32[3072, 768]", primals_270: "f32[3072]", primals_271: "f32[768, 3072]", primals_272: "f32[768]", primals_273: "f32[768]", primals_274: "f32[768]", primals_275: "f32[2304, 768]", primals_276: "f32[2304]", primals_277: "f32[768, 768]", primals_278: "f32[768]", primals_279: "f32[768]", primals_280: "f32[768]", primals_281: "f32[768, 1, 3, 3]", primals_282: "f32[768]", primals_283: "f32[768]", primals_284: "f32[768]", primals_285: "f32[768, 1, 3, 3]", primals_286: "f32[768]", primals_287: "f32[768]", primals_288: "f32[768]", primals_289: "f32[3072, 768]", primals_290: "f32[3072]", primals_291: "f32[768, 3072]", primals_292: "f32[768]", primals_293: "f32[768]", primals_294: "f32[768]", primals_295: "f32[2304, 768]", primals_296: "f32[2304]", primals_297: "f32[768, 768]", primals_298: "f32[768]", primals_299: "f32[768]", primals_300: "f32[768]", primals_301: "f32[768, 1, 3, 3]", primals_302: "f32[768]", primals_303: "f32[768]", primals_304: "f32[768]", primals_305: "f32[768, 1, 3, 3]", primals_306: "f32[768]", primals_307: "f32[768]", primals_308: "f32[768]", primals_309: "f32[3072, 768]", primals_310: "f32[3072]", primals_311: "f32[768, 3072]", primals_312: "f32[768]", primals_313: "f32[768]", primals_314: "f32[768]", primals_315: "f32[2304, 768]", primals_316: "f32[2304]", primals_317: "f32[768, 768]", primals_318: "f32[768]", primals_319: "f32[768]", primals_320: "f32[768]", primals_321: "f32[768, 1, 3, 3]", primals_322: "f32[768]", primals_323: "f32[768]", primals_324: "f32[768]", primals_325: "f32[768, 1, 3, 3]", primals_326: "f32[768]", primals_327: "f32[768]", primals_328: "f32[768]", primals_329: "f32[3072, 768]", primals_330: "f32[3072]", primals_331: "f32[768, 3072]", primals_332: "f32[768]", primals_333: "f32[768]", primals_334: "f32[768]", primals_335: "f32[2304, 768]", primals_336: "f32[2304]", primals_337: "f32[768, 768]", primals_338: "f32[768]", primals_339: "f32[768]", primals_340: "f32[768]", primals_341: "f32[768, 1, 3, 3]", primals_342: "f32[768]", primals_343: "f32[768]", primals_344: "f32[768]", primals_345: "f32[768, 1, 3, 3]", primals_346: "f32[768]", primals_347: "f32[768]", primals_348: "f32[768]", primals_349: "f32[3072, 768]", primals_350: "f32[3072]", primals_351: "f32[768, 3072]", primals_352: "f32[768]", primals_353: "f32[768]", primals_354: "f32[768]", primals_355: "f32[2304, 768]", primals_356: "f32[2304]", primals_357: "f32[768, 768]", primals_358: "f32[768]", primals_359: "f32[768]", primals_360: "f32[768]", primals_361: "f32[768, 1, 3, 3]", primals_362: "f32[768]", primals_363: "f32[768]", primals_364: "f32[768]", primals_365: "f32[768, 1, 3, 3]", primals_366: "f32[768]", primals_367: "f32[768]", primals_368: "f32[768]", primals_369: "f32[3072, 768]", primals_370: "f32[3072]", primals_371: "f32[768, 3072]", primals_372: "f32[768]", primals_373: "f32[768]", primals_374: "f32[768]", primals_375: "f32[2304, 768]", primals_376: "f32[2304]", primals_377: "f32[768, 768]", primals_378: "f32[768]", primals_379: "f32[768]", primals_380: "f32[768]", primals_381: "f32[768, 1, 3, 3]", primals_382: "f32[768]", primals_383: "f32[768]", primals_384: "f32[768]", primals_385: "f32[768, 1, 3, 3]", primals_386: "f32[768]", primals_387: "f32[768]", primals_388: "f32[768]", primals_389: "f32[3072, 768]", primals_390: "f32[3072]", primals_391: "f32[768, 3072]", primals_392: "f32[768]", primals_393: "f32[768]", primals_394: "f32[768]", primals_395: "f32[2304, 768]", primals_396: "f32[2304]", primals_397: "f32[768, 768]", primals_398: "f32[768]", primals_399: "f32[768]", primals_400: "f32[768]", primals_401: "f32[768, 1, 3, 3]", primals_402: "f32[768]", primals_403: "f32[768]", primals_404: "f32[768]", primals_405: "f32[768, 1, 3, 3]", primals_406: "f32[768]", primals_407: "f32[768]", primals_408: "f32[768]", primals_409: "f32[3072, 768]", primals_410: "f32[3072]", primals_411: "f32[768, 3072]", primals_412: "f32[768]", primals_413: "f32[768]", primals_414: "f32[768]", primals_415: "f32[2304, 768]", primals_416: "f32[2304]", primals_417: "f32[768, 768]", primals_418: "f32[768]", primals_419: "f32[768]", primals_420: "f32[768]", primals_421: "f32[768, 1, 3, 3]", primals_422: "f32[768]", primals_423: "f32[768]", primals_424: "f32[768]", primals_425: "f32[768, 1, 3, 3]", primals_426: "f32[768]", primals_427: "f32[768]", primals_428: "f32[768]", primals_429: "f32[3072, 768]", primals_430: "f32[3072]", primals_431: "f32[768, 3072]", primals_432: "f32[768]", primals_433: "f32[768]", primals_434: "f32[768]", primals_435: "f32[2304, 768]", primals_436: "f32[2304]", primals_437: "f32[768, 768]", primals_438: "f32[768]", primals_439: "f32[768]", primals_440: "f32[768]", primals_441: "f32[768, 1, 3, 3]", primals_442: "f32[768]", primals_443: "f32[768]", primals_444: "f32[768]", primals_445: "f32[768, 1, 3, 3]", primals_446: "f32[768]", primals_447: "f32[768]", primals_448: "f32[768]", primals_449: "f32[3072, 768]", primals_450: "f32[3072]", primals_451: "f32[768, 3072]", primals_452: "f32[768]", primals_453: "f32[768]", primals_454: "f32[768]", primals_455: "f32[2304, 768]", primals_456: "f32[2304]", primals_457: "f32[768, 768]", primals_458: "f32[768]", primals_459: "f32[768]", primals_460: "f32[768]", primals_461: "f32[768, 1, 3, 3]", primals_462: "f32[768]", primals_463: "f32[768]", primals_464: "f32[768]", primals_465: "f32[768, 1, 3, 3]", primals_466: "f32[768]", primals_467: "f32[768]", primals_468: "f32[768]", primals_469: "f32[3072, 768]", primals_470: "f32[3072]", primals_471: "f32[768, 3072]", primals_472: "f32[768]", primals_473: "f32[768]", primals_474: "f32[768]", primals_475: "f32[2304, 768]", primals_476: "f32[2304]", primals_477: "f32[768, 768]", primals_478: "f32[768]", primals_479: "f32[768]", primals_480: "f32[768]", primals_481: "f32[768, 1, 3, 3]", primals_482: "f32[768]", primals_483: "f32[768]", primals_484: "f32[768]", primals_485: "f32[768, 1, 3, 3]", primals_486: "f32[768]", primals_487: "f32[768]", primals_488: "f32[768]", primals_489: "f32[3072, 768]", primals_490: "f32[3072]", primals_491: "f32[768, 3072]", primals_492: "f32[768]", primals_493: "f32[768]", primals_494: "f32[768]", primals_495: "f32[2304, 768]", primals_496: "f32[2304]", primals_497: "f32[768, 768]", primals_498: "f32[768]", primals_499: "f32[768]", primals_500: "f32[768]", primals_501: "f32[768, 1, 3, 3]", primals_502: "f32[768]", primals_503: "f32[768]", primals_504: "f32[768]", primals_505: "f32[768, 1, 3, 3]", primals_506: "f32[768]", primals_507: "f32[768]", primals_508: "f32[768]", primals_509: "f32[3072, 768]", primals_510: "f32[3072]", primals_511: "f32[768, 3072]", primals_512: "f32[768]", primals_513: "f32[768]", primals_514: "f32[768]", primals_515: "f32[2304, 768]", primals_516: "f32[2304]", primals_517: "f32[768, 768]", primals_518: "f32[768]", primals_519: "f32[768]", primals_520: "f32[768]", primals_521: "f32[768, 1, 3, 3]", primals_522: "f32[768]", primals_523: "f32[768]", primals_524: "f32[768]", primals_525: "f32[768, 1, 3, 3]", primals_526: "f32[768]", primals_527: "f32[768]", primals_528: "f32[768]", primals_529: "f32[3072, 768]", primals_530: "f32[3072]", primals_531: "f32[768, 3072]", primals_532: "f32[768]", primals_533: "f32[768]", primals_534: "f32[768]", primals_535: "f32[2304, 768]", primals_536: "f32[2304]", primals_537: "f32[768, 768]", primals_538: "f32[768]", primals_539: "f32[768]", primals_540: "f32[768]", primals_541: "f32[768, 1, 3, 3]", primals_542: "f32[768]", primals_543: "f32[768]", primals_544: "f32[768]", primals_545: "f32[768, 1, 3, 3]", primals_546: "f32[768]", primals_547: "f32[768]", primals_548: "f32[768]", primals_549: "f32[3072, 768]", primals_550: "f32[3072]", primals_551: "f32[768, 3072]", primals_552: "f32[768]", primals_553: "f32[768]", primals_554: "f32[768]", primals_555: "f32[2304, 768]", primals_556: "f32[2304]", primals_557: "f32[768, 768]", primals_558: "f32[768]", primals_559: "f32[768]", primals_560: "f32[768]", primals_561: "f32[768, 1, 3, 3]", primals_562: "f32[768]", primals_563: "f32[768]", primals_564: "f32[768]", primals_565: "f32[768, 1, 3, 3]", primals_566: "f32[768]", primals_567: "f32[768]", primals_568: "f32[768]", primals_569: "f32[3072, 768]", primals_570: "f32[3072]", primals_571: "f32[768, 3072]", primals_572: "f32[768]", primals_573: "f32[768]", primals_574: "f32[768]", primals_575: "f32[2304, 768]", primals_576: "f32[2304]", primals_577: "f32[768, 768]", primals_578: "f32[768]", primals_579: "f32[768]", primals_580: "f32[768]", primals_581: "f32[768, 1, 3, 3]", primals_582: "f32[768]", primals_583: "f32[768]", primals_584: "f32[768]", primals_585: "f32[768, 1, 3, 3]", primals_586: "f32[768]", primals_587: "f32[768]", primals_588: "f32[768]", primals_589: "f32[3072, 768]", primals_590: "f32[3072]", primals_591: "f32[768, 3072]", primals_592: "f32[768]", primals_593: "f32[768]", primals_594: "f32[768]", primals_595: "f32[768, 768]", primals_596: "f32[768]", primals_597: "f32[768, 768]", primals_598: "f32[768]", primals_599: "f32[768, 768]", primals_600: "f32[768]", primals_601: "f32[768, 768]", primals_602: "f32[768]", primals_603: "f32[768]", primals_604: "f32[768]", primals_605: "f32[3072, 768]", primals_606: "f32[3072]", primals_607: "f32[768, 3072]", primals_608: "f32[768]", primals_609: "f32[768]", primals_610: "f32[768]", primals_611: "f32[768, 768]", primals_612: "f32[768]", primals_613: "f32[768, 768]", primals_614: "f32[768]", primals_615: "f32[768, 768]", primals_616: "f32[768]", primals_617: "f32[768, 768]", primals_618: "f32[768]", primals_619: "f32[768]", primals_620: "f32[768]", primals_621: "f32[3072, 768]", primals_622: "f32[3072]", primals_623: "f32[768, 3072]", primals_624: "f32[768]", primals_625: "f32[768]", primals_626: "f32[768]", primals_627: "f32[1000, 768]", primals_628: "f32[1000]", primals_629: "f32[192]", primals_630: "f32[192]", primals_631: "i64[]", primals_632: "f32[384]", primals_633: "f32[384]", primals_634: "i64[]", primals_635: "f32[768]", primals_636: "f32[768]", primals_637: "i64[]", primals_638: "f32[768]", primals_639: "f32[768]", primals_640: "i64[]", primals_641: "f32[768]", primals_642: "f32[768]", primals_643: "i64[]", primals_644: "f32[768]", primals_645: "f32[768]", primals_646: "i64[]", primals_647: "f32[768]", primals_648: "f32[768]", primals_649: "i64[]", primals_650: "f32[768]", primals_651: "f32[768]", primals_652: "i64[]", primals_653: "f32[768]", primals_654: "f32[768]", primals_655: "i64[]", primals_656: "f32[768]", primals_657: "f32[768]", primals_658: "i64[]", primals_659: "f32[768]", primals_660: "f32[768]", primals_661: "i64[]", primals_662: "f32[768]", primals_663: "f32[768]", primals_664: "i64[]", primals_665: "f32[768]", primals_666: "f32[768]", primals_667: "i64[]", primals_668: "f32[768]", primals_669: "f32[768]", primals_670: "i64[]", primals_671: "f32[768]", primals_672: "f32[768]", primals_673: "i64[]", primals_674: "f32[768]", primals_675: "f32[768]", primals_676: "i64[]", primals_677: "f32[768]", primals_678: "f32[768]", primals_679: "i64[]", primals_680: "f32[768]", primals_681: "f32[768]", primals_682: "i64[]", primals_683: "f32[768]", primals_684: "f32[768]", primals_685: "i64[]", primals_686: "f32[768]", primals_687: "f32[768]", primals_688: "i64[]", primals_689: "f32[768]", primals_690: "f32[768]", primals_691: "i64[]", primals_692: "f32[768]", primals_693: "f32[768]", primals_694: "i64[]", primals_695: "f32[768]", primals_696: "f32[768]", primals_697: "i64[]", primals_698: "f32[768]", primals_699: "f32[768]", primals_700: "i64[]", primals_701: "f32[768]", primals_702: "f32[768]", primals_703: "i64[]", primals_704: "f32[768]", primals_705: "f32[768]", primals_706: "i64[]", primals_707: "f32[768]", primals_708: "f32[768]", primals_709: "i64[]", primals_710: "f32[8, 3, 224, 224]"):
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
    unsqueeze: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(arange, 1)
    repeat: "f32[1, 28, 28]" = torch.ops.aten.repeat.default(unsqueeze, [1, 1, 28]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:52, code: x_embed = torch.arange(1, W+1, dtype=torch.float32, device=device).repeat(1, H, 1)
    repeat_1: "f32[1, 28, 28]" = torch.ops.aten.repeat.default(arange, [1, 28, 1]);  arange = None
    
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
    slice_13: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(div_4, 0, 0, 9223372036854775807);  div_4 = None
    slice_14: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_13, 1, 0, 9223372036854775807);  slice_13 = None
    slice_15: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_14, 2, 0, 9223372036854775807);  slice_14 = None
    slice_16: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 9223372036854775807, 2)
    sin: "f32[1, 28, 28, 16]" = torch.ops.aten.sin.default(slice_16);  slice_16 = None
    slice_20: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(slice_15, 3, 1, 9223372036854775807, 2);  slice_15 = None
    cos: "f32[1, 28, 28, 16]" = torch.ops.aten.cos.default(slice_20);  slice_20 = None
    stack: "f32[1, 28, 28, 16, 2]" = torch.ops.aten.stack.default([sin, cos], 4);  sin = cos = None
    view_1: "f32[1, 28, 28, 32]" = torch.ops.aten.view.default(stack, [1, 28, 28, 32]);  stack = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:60, code: pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
    slice_21: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(div_5, 0, 0, 9223372036854775807);  div_5 = None
    slice_22: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_21, 1, 0, 9223372036854775807);  slice_21 = None
    slice_23: "f32[1, 28, 28, 32]" = torch.ops.aten.slice.Tensor(slice_22, 2, 0, 9223372036854775807);  slice_22 = None
    slice_24: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 9223372036854775807, 2)
    sin_1: "f32[1, 28, 28, 16]" = torch.ops.aten.sin.default(slice_24);  slice_24 = None
    slice_28: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(slice_23, 3, 1, 9223372036854775807, 2);  slice_23 = None
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
    clamp_min: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm, 1e-12)
    expand: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min, [8, 16, 48, 784]);  clamp_min = None
    div_6: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_18, expand);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_1: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_19, 2.0, [-1], True)
    clamp_min_1: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_1, 1e-12)
    expand_1: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_1, [8, 16, 48, 784]);  clamp_min_1 = None
    div_7: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_19, expand_1);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_1: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_7, -2, -1);  div_7 = None
    expand_2: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_6, [8, 16, 48, 784]);  div_6 = None
    clone_1: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    _unsafe_view: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_1, [128, 48, 784]);  clone_1 = None
    expand_3: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_1, [8, 16, 784, 48]);  transpose_1 = None
    clone_2: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    _unsafe_view_1: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_2, [128, 784, 48]);  clone_2 = None
    bmm: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view, _unsafe_view_1)
    view_7: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm, [8, 16, 48, 48])
    mul_3: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_7, primals_2);  view_7 = None
    
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
    view_11: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm, [8, 784, 768])
    add_6: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_11, primals_118);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_6: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_6);  add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_4: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_1, clone_6);  clone_6 = None
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
    view_13: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_5, [8, 768, 784])
    permute_5: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_5: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_3, permute_5);  permute_5 = None
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
    view_15: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_1, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_3: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_15);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_7: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_3);  gelu_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_16: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_7, [6272, 3072]);  clone_7 = None
    t_3: "f32[3072, 768]" = torch.ops.aten.t.default(primals_131);  primals_131 = None
    addmm_2: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_132, view_16, t_3);  primals_132 = None
    view_17: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_2, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_8: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_17);  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_6: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_4, clone_8);  clone_8 = None
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
    clamp_min_2: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_2, 1e-12)
    expand_6: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_2, [8, 16, 48, 784]);  clamp_min_2 = None
    div_8: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_35, expand_6);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_3: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_36, 2.0, [-1], True)
    clamp_min_3: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_3, 1e-12)
    expand_7: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_3, [8, 16, 48, 784]);  clamp_min_3 = None
    div_9: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_36, expand_7);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_2: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_9, -2, -1);  div_9 = None
    expand_8: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_8, [8, 16, 48, 784]);  div_8 = None
    clone_9: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    _unsafe_view_4: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_9, [128, 48, 784]);  clone_9 = None
    expand_9: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_2, [8, 16, 784, 48]);  transpose_2 = None
    clone_10: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    _unsafe_view_5: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_10, [128, 784, 48]);  clone_10 = None
    bmm_2: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_4, _unsafe_view_5)
    view_21: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_2, [8, 16, 48, 48])
    mul_7: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_21, primals_6);  view_21 = None
    
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
    view_25: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_1, [8, 784, 768])
    add_11: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_25, primals_138);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_14: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_11);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_8: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_5, clone_14);  clone_14 = None
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
    view_27: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_7, [8, 768, 784])
    permute_9: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_9: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_7, permute_9);  permute_9 = None
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
    view_29: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_4, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_5: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_29);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_15: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_5);  gelu_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_30: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_15, [6272, 3072]);  clone_15 = None
    t_7: "f32[3072, 768]" = torch.ops.aten.t.default(primals_151);  primals_151 = None
    addmm_5: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_152, view_30, t_7);  primals_152 = None
    view_31: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_5, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_16: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_31);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_10: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_8, clone_16);  clone_16 = None
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
    clamp_min_4: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_4, 1e-12)
    expand_12: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_4, [8, 16, 48, 784]);  clamp_min_4 = None
    div_10: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_52, expand_12);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_5: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_53, 2.0, [-1], True)
    clamp_min_5: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_5, 1e-12)
    expand_13: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_5, [8, 16, 48, 784]);  clamp_min_5 = None
    div_11: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_53, expand_13);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_3: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_11, -2, -1);  div_11 = None
    expand_14: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_10, [8, 16, 48, 784]);  div_10 = None
    clone_17: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    _unsafe_view_8: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_17, [128, 48, 784]);  clone_17 = None
    expand_15: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_3, [8, 16, 784, 48]);  transpose_3 = None
    clone_18: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    _unsafe_view_9: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_18, [128, 784, 48]);  clone_18 = None
    bmm_4: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_8, _unsafe_view_9)
    view_35: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_4, [8, 16, 48, 48])
    mul_11: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_35, primals_10);  view_35 = None
    
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
    view_39: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_2, [8, 784, 768])
    add_16: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_39, primals_158);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_22: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_16);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_12: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_9, clone_22);  clone_22 = None
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
    view_41: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_9, [8, 768, 784])
    permute_13: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_13: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_11, permute_13);  permute_13 = None
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
    view_43: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_7, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_7: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_43);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_7);  gelu_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_44: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_23, [6272, 3072]);  clone_23 = None
    t_11: "f32[3072, 768]" = torch.ops.aten.t.default(primals_171);  primals_171 = None
    addmm_8: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_172, view_44, t_11);  primals_172 = None
    view_45: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_8, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_45);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_14: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_12, clone_24);  clone_24 = None
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
    clamp_min_6: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_6, 1e-12)
    expand_18: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_6, [8, 16, 48, 784]);  clamp_min_6 = None
    div_12: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_69, expand_18);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_7: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_70, 2.0, [-1], True)
    clamp_min_7: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_7, 1e-12)
    expand_19: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_7, [8, 16, 48, 784]);  clamp_min_7 = None
    div_13: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_70, expand_19);  expand_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_4: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_13, -2, -1);  div_13 = None
    expand_20: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_12, [8, 16, 48, 784]);  div_12 = None
    clone_25: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    _unsafe_view_12: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_25, [128, 48, 784]);  clone_25 = None
    expand_21: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_4, [8, 16, 784, 48]);  transpose_4 = None
    clone_26: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    _unsafe_view_13: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_26, [128, 784, 48]);  clone_26 = None
    bmm_6: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_12, _unsafe_view_13)
    view_49: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_6, [8, 16, 48, 48])
    mul_15: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_49, primals_14);  view_49 = None
    
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
    view_53: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_3, [8, 784, 768])
    add_21: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_53, primals_178);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_30: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_21);  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_16: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_13, clone_30);  clone_30 = None
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
    view_55: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_11, [8, 768, 784])
    permute_17: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_17: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_15, permute_17);  permute_17 = None
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
    view_57: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_9: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_57);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_31: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_9);  gelu_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_58: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_31, [6272, 3072]);  clone_31 = None
    t_15: "f32[3072, 768]" = torch.ops.aten.t.default(primals_191);  primals_191 = None
    addmm_11: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_192, view_58, t_15);  primals_192 = None
    view_59: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_11, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_32: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_59);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_18: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_16, clone_32);  clone_32 = None
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
    clamp_min_8: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_8, 1e-12)
    expand_24: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_8, [8, 16, 48, 784]);  clamp_min_8 = None
    div_14: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_86, expand_24);  expand_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_9: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_87, 2.0, [-1], True)
    clamp_min_9: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_9, 1e-12)
    expand_25: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_9, [8, 16, 48, 784]);  clamp_min_9 = None
    div_15: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_87, expand_25);  expand_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_5: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_15, -2, -1);  div_15 = None
    expand_26: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_14, [8, 16, 48, 784]);  div_14 = None
    clone_33: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    _unsafe_view_16: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_33, [128, 48, 784]);  clone_33 = None
    expand_27: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_5, [8, 16, 784, 48]);  transpose_5 = None
    clone_34: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    _unsafe_view_17: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_34, [128, 784, 48]);  clone_34 = None
    bmm_8: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_16, _unsafe_view_17)
    view_63: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_8, [8, 16, 48, 48])
    mul_19: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_63, primals_18);  view_63 = None
    
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
    view_67: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_4, [8, 784, 768])
    add_26: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_67, primals_198);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_38: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_26);  add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_20: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_17, clone_38);  clone_38 = None
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
    view_69: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_13, [8, 768, 784])
    permute_21: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_21: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_19, permute_21);  permute_21 = None
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
    view_71: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_13, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_11: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_71);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_39: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_11);  gelu_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_72: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_39, [6272, 3072]);  clone_39 = None
    t_19: "f32[3072, 768]" = torch.ops.aten.t.default(primals_211);  primals_211 = None
    addmm_14: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_212, view_72, t_19);  primals_212 = None
    view_73: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_14, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_40: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_73);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_22: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_20, clone_40);  clone_40 = None
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
    clamp_min_10: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_10, 1e-12)
    expand_30: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_10, [8, 16, 48, 784]);  clamp_min_10 = None
    div_16: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_103, expand_30);  expand_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_11: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_104, 2.0, [-1], True)
    clamp_min_11: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_11, 1e-12)
    expand_31: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_11, [8, 16, 48, 784]);  clamp_min_11 = None
    div_17: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_104, expand_31);  expand_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_6: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_17, -2, -1);  div_17 = None
    expand_32: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_16, [8, 16, 48, 784]);  div_16 = None
    clone_41: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    _unsafe_view_20: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_41, [128, 48, 784]);  clone_41 = None
    expand_33: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_6, [8, 16, 784, 48]);  transpose_6 = None
    clone_42: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    _unsafe_view_21: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_42, [128, 784, 48]);  clone_42 = None
    bmm_10: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_20, _unsafe_view_21)
    view_77: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_10, [8, 16, 48, 48])
    mul_23: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_77, primals_22);  view_77 = None
    
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
    view_81: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_5, [8, 784, 768])
    add_31: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_81, primals_218);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_46: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_31);  add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_24: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_21, clone_46);  clone_46 = None
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
    view_83: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_15, [8, 768, 784])
    permute_25: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_25: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_23, permute_25);  permute_25 = None
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
    view_85: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_16, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_13: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_85);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_47: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_13);  gelu_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_86: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_47, [6272, 3072]);  clone_47 = None
    t_23: "f32[3072, 768]" = torch.ops.aten.t.default(primals_231);  primals_231 = None
    addmm_17: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_232, view_86, t_23);  primals_232 = None
    view_87: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_17, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_48: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_87);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_26: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_24, clone_48);  clone_48 = None
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
    clamp_min_12: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_12, 1e-12)
    expand_36: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_12, [8, 16, 48, 784]);  clamp_min_12 = None
    div_18: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_120, expand_36);  expand_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_13: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_121, 2.0, [-1], True)
    clamp_min_13: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_13, 1e-12)
    expand_37: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_13, [8, 16, 48, 784]);  clamp_min_13 = None
    div_19: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_121, expand_37);  expand_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_7: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_19, -2, -1);  div_19 = None
    expand_38: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_18, [8, 16, 48, 784]);  div_18 = None
    clone_49: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    _unsafe_view_24: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_49, [128, 48, 784]);  clone_49 = None
    expand_39: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_7, [8, 16, 784, 48]);  transpose_7 = None
    clone_50: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    _unsafe_view_25: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_50, [128, 784, 48]);  clone_50 = None
    bmm_12: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_24, _unsafe_view_25)
    view_91: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_12, [8, 16, 48, 48])
    mul_27: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_91, primals_26);  view_91 = None
    
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
    view_95: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_6, [8, 784, 768])
    add_36: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_95, primals_238);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_54: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_36);  add_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_28: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_25, clone_54);  clone_54 = None
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
    view_97: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_17, [8, 768, 784])
    permute_29: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_29: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_27, permute_29);  permute_29 = None
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
    view_99: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_19, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_15: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_99);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_55: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_15);  gelu_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_100: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_55, [6272, 3072]);  clone_55 = None
    t_27: "f32[3072, 768]" = torch.ops.aten.t.default(primals_251);  primals_251 = None
    addmm_20: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_252, view_100, t_27);  primals_252 = None
    view_101: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_20, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_56: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_101);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_30: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_28, clone_56);  clone_56 = None
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
    clamp_min_14: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_14, 1e-12)
    expand_42: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_14, [8, 16, 48, 784]);  clamp_min_14 = None
    div_20: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_137, expand_42);  expand_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_15: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_138, 2.0, [-1], True)
    clamp_min_15: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_15, 1e-12)
    expand_43: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_15, [8, 16, 48, 784]);  clamp_min_15 = None
    div_21: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_138, expand_43);  expand_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_8: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_21, -2, -1);  div_21 = None
    expand_44: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_20, [8, 16, 48, 784]);  div_20 = None
    clone_57: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    _unsafe_view_28: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_57, [128, 48, 784]);  clone_57 = None
    expand_45: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_8, [8, 16, 784, 48]);  transpose_8 = None
    clone_58: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    _unsafe_view_29: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_58, [128, 784, 48]);  clone_58 = None
    bmm_14: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_28, _unsafe_view_29)
    view_105: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_14, [8, 16, 48, 48])
    mul_31: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_105, primals_30);  view_105 = None
    
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
    view_109: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_7, [8, 784, 768])
    add_41: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_109, primals_258);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_62: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_41);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_32: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_29, clone_62);  clone_62 = None
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
    view_111: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_19, [8, 768, 784])
    permute_33: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_111, [0, 2, 1]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_33: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_31, permute_33);  permute_33 = None
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
    view_113: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_17: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_113);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_63: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_17);  gelu_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_114: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_63, [6272, 3072]);  clone_63 = None
    t_31: "f32[3072, 768]" = torch.ops.aten.t.default(primals_271);  primals_271 = None
    addmm_23: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_272, view_114, t_31);  primals_272 = None
    view_115: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_23, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_64: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_115);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_34: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_32, clone_64);  clone_64 = None
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
    clamp_min_16: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_16, 1e-12)
    expand_48: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_16, [8, 16, 48, 784]);  clamp_min_16 = None
    div_22: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_154, expand_48);  expand_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_17: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_155, 2.0, [-1], True)
    clamp_min_17: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_17, 1e-12)
    expand_49: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_17, [8, 16, 48, 784]);  clamp_min_17 = None
    div_23: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_155, expand_49);  expand_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_9: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_23, -2, -1);  div_23 = None
    expand_50: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_22, [8, 16, 48, 784]);  div_22 = None
    clone_65: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
    _unsafe_view_32: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_65, [128, 48, 784]);  clone_65 = None
    expand_51: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_9, [8, 16, 784, 48]);  transpose_9 = None
    clone_66: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    _unsafe_view_33: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_66, [128, 784, 48]);  clone_66 = None
    bmm_16: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_32, _unsafe_view_33)
    view_119: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_16, [8, 16, 48, 48])
    mul_35: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_119, primals_34);  view_119 = None
    
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
    view_123: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_8, [8, 784, 768])
    add_46: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_123, primals_278);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_70: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_46);  add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_36: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_33, clone_70);  clone_70 = None
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
    view_125: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_21, [8, 768, 784])
    permute_37: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_37: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_35, permute_37);  permute_37 = None
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
    view_127: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_25, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_19: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_127);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_71: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_19);  gelu_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_128: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_71, [6272, 3072]);  clone_71 = None
    t_35: "f32[3072, 768]" = torch.ops.aten.t.default(primals_291);  primals_291 = None
    addmm_26: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_292, view_128, t_35);  primals_292 = None
    view_129: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_26, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_72: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_129);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_38: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_36, clone_72);  clone_72 = None
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
    clamp_min_18: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_18, 1e-12)
    expand_54: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_18, [8, 16, 48, 784]);  clamp_min_18 = None
    div_24: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_171, expand_54);  expand_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_19: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_172, 2.0, [-1], True)
    clamp_min_19: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_19, 1e-12)
    expand_55: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_19, [8, 16, 48, 784]);  clamp_min_19 = None
    div_25: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_172, expand_55);  expand_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_10: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_25, -2, -1);  div_25 = None
    expand_56: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_24, [8, 16, 48, 784]);  div_24 = None
    clone_73: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    _unsafe_view_36: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_73, [128, 48, 784]);  clone_73 = None
    expand_57: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_10, [8, 16, 784, 48]);  transpose_10 = None
    clone_74: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    _unsafe_view_37: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_74, [128, 784, 48]);  clone_74 = None
    bmm_18: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_36, _unsafe_view_37)
    view_133: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_18, [8, 16, 48, 48])
    mul_39: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_133, primals_38);  view_133 = None
    
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
    view_137: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_9, [8, 784, 768])
    add_51: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_137, primals_298);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_78: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_51);  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_40: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_37, clone_78);  clone_78 = None
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
    view_139: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_23, [8, 768, 784])
    permute_41: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_139, [0, 2, 1]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_41: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_39, permute_41);  permute_41 = None
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
    view_141: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_28, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_21: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_141);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_79: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_21);  gelu_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_142: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_79, [6272, 3072]);  clone_79 = None
    t_39: "f32[3072, 768]" = torch.ops.aten.t.default(primals_311);  primals_311 = None
    addmm_29: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_312, view_142, t_39);  primals_312 = None
    view_143: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_29, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_80: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_143);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_42: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_40, clone_80);  clone_80 = None
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
    clamp_min_20: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_20, 1e-12)
    expand_60: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_20, [8, 16, 48, 784]);  clamp_min_20 = None
    div_26: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_188, expand_60);  expand_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_21: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_189, 2.0, [-1], True)
    clamp_min_21: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_21, 1e-12)
    expand_61: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_21, [8, 16, 48, 784]);  clamp_min_21 = None
    div_27: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_189, expand_61);  expand_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_11: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_27, -2, -1);  div_27 = None
    expand_62: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_26, [8, 16, 48, 784]);  div_26 = None
    clone_81: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
    _unsafe_view_40: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_81, [128, 48, 784]);  clone_81 = None
    expand_63: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_11, [8, 16, 784, 48]);  transpose_11 = None
    clone_82: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    _unsafe_view_41: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_82, [128, 784, 48]);  clone_82 = None
    bmm_20: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_40, _unsafe_view_41)
    view_147: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_20, [8, 16, 48, 48])
    mul_43: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_147, primals_42);  view_147 = None
    
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
    view_151: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_10, [8, 784, 768])
    add_56: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_151, primals_318);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_86: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_56);  add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_44: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_41, clone_86);  clone_86 = None
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
    view_153: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_25, [8, 768, 784])
    permute_45: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_45: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_43, permute_45);  permute_45 = None
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
    view_155: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_31, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_23: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_155);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_87: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_23);  gelu_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_156: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_87, [6272, 3072]);  clone_87 = None
    t_43: "f32[3072, 768]" = torch.ops.aten.t.default(primals_331);  primals_331 = None
    addmm_32: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_332, view_156, t_43);  primals_332 = None
    view_157: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_32, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_88: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_157);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_46: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_44, clone_88);  clone_88 = None
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
    clamp_min_22: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_22, 1e-12)
    expand_66: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_22, [8, 16, 48, 784]);  clamp_min_22 = None
    div_28: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_205, expand_66);  expand_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_23: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_206, 2.0, [-1], True)
    clamp_min_23: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_23, 1e-12)
    expand_67: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_23, [8, 16, 48, 784]);  clamp_min_23 = None
    div_29: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_206, expand_67);  expand_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_12: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_29, -2, -1);  div_29 = None
    expand_68: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_28, [8, 16, 48, 784]);  div_28 = None
    clone_89: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    _unsafe_view_44: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_89, [128, 48, 784]);  clone_89 = None
    expand_69: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_12, [8, 16, 784, 48]);  transpose_12 = None
    clone_90: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    _unsafe_view_45: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_90, [128, 784, 48]);  clone_90 = None
    bmm_22: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_44, _unsafe_view_45)
    view_161: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_22, [8, 16, 48, 48])
    mul_47: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_161, primals_46);  view_161 = None
    
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
    view_165: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_11, [8, 784, 768])
    add_61: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_165, primals_338);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_94: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_61);  add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_48: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_45, clone_94);  clone_94 = None
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
    view_167: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_27, [8, 768, 784])
    permute_49: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_49: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_47, permute_49);  permute_49 = None
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
    view_169: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_25: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_169);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_95: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_25);  gelu_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_170: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_95, [6272, 3072]);  clone_95 = None
    t_47: "f32[3072, 768]" = torch.ops.aten.t.default(primals_351);  primals_351 = None
    addmm_35: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_352, view_170, t_47);  primals_352 = None
    view_171: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_35, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_96: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_171);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_50: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_48, clone_96);  clone_96 = None
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
    clamp_min_24: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_24, 1e-12)
    expand_72: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_24, [8, 16, 48, 784]);  clamp_min_24 = None
    div_30: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_222, expand_72);  expand_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_25: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_223, 2.0, [-1], True)
    clamp_min_25: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_25, 1e-12)
    expand_73: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_25, [8, 16, 48, 784]);  clamp_min_25 = None
    div_31: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_223, expand_73);  expand_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_13: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_31, -2, -1);  div_31 = None
    expand_74: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_30, [8, 16, 48, 784]);  div_30 = None
    clone_97: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    _unsafe_view_48: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_97, [128, 48, 784]);  clone_97 = None
    expand_75: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_13, [8, 16, 784, 48]);  transpose_13 = None
    clone_98: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    _unsafe_view_49: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_98, [128, 784, 48]);  clone_98 = None
    bmm_24: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_48, _unsafe_view_49)
    view_175: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_24, [8, 16, 48, 48])
    mul_51: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_175, primals_50);  view_175 = None
    
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
    view_179: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_12, [8, 784, 768])
    add_66: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_179, primals_358);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_102: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_66);  add_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_52: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_49, clone_102);  clone_102 = None
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
    view_181: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_29, [8, 768, 784])
    permute_53: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_53: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_51, permute_53);  permute_53 = None
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
    view_183: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_37, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_27: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_183);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_103: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_27);  gelu_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_184: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_103, [6272, 3072]);  clone_103 = None
    t_51: "f32[3072, 768]" = torch.ops.aten.t.default(primals_371);  primals_371 = None
    addmm_38: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_372, view_184, t_51);  primals_372 = None
    view_185: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_38, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_104: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_185);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_54: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_52, clone_104);  clone_104 = None
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
    clamp_min_26: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_26, 1e-12)
    expand_78: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_26, [8, 16, 48, 784]);  clamp_min_26 = None
    div_32: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_239, expand_78);  expand_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_27: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_240, 2.0, [-1], True)
    clamp_min_27: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_27, 1e-12)
    expand_79: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_27, [8, 16, 48, 784]);  clamp_min_27 = None
    div_33: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_240, expand_79);  expand_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_14: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_33, -2, -1);  div_33 = None
    expand_80: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_32, [8, 16, 48, 784]);  div_32 = None
    clone_105: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    _unsafe_view_52: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_105, [128, 48, 784]);  clone_105 = None
    expand_81: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_14, [8, 16, 784, 48]);  transpose_14 = None
    clone_106: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    _unsafe_view_53: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_106, [128, 784, 48]);  clone_106 = None
    bmm_26: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_52, _unsafe_view_53)
    view_189: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_26, [8, 16, 48, 48])
    mul_55: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_189, primals_54);  view_189 = None
    
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
    view_193: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_13, [8, 784, 768])
    add_71: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_193, primals_378);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_110: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_71);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_56: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_53, clone_110);  clone_110 = None
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
    view_195: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_31, [8, 768, 784])
    permute_57: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_57: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_55, permute_57);  permute_57 = None
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
    view_197: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_40, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_29: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_197);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_111: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_29);  gelu_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_198: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_111, [6272, 3072]);  clone_111 = None
    t_55: "f32[3072, 768]" = torch.ops.aten.t.default(primals_391);  primals_391 = None
    addmm_41: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_392, view_198, t_55);  primals_392 = None
    view_199: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_41, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_112: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_199);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_58: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_56, clone_112);  clone_112 = None
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
    clamp_min_28: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_28, 1e-12)
    expand_84: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_28, [8, 16, 48, 784]);  clamp_min_28 = None
    div_34: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_256, expand_84);  expand_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_29: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_257, 2.0, [-1], True)
    clamp_min_29: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_29, 1e-12)
    expand_85: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_29, [8, 16, 48, 784]);  clamp_min_29 = None
    div_35: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_257, expand_85);  expand_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_15: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_35, -2, -1);  div_35 = None
    expand_86: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_34, [8, 16, 48, 784]);  div_34 = None
    clone_113: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
    _unsafe_view_56: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_113, [128, 48, 784]);  clone_113 = None
    expand_87: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_15, [8, 16, 784, 48]);  transpose_15 = None
    clone_114: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
    _unsafe_view_57: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_114, [128, 784, 48]);  clone_114 = None
    bmm_28: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_56, _unsafe_view_57)
    view_203: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_28, [8, 16, 48, 48])
    mul_59: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_203, primals_58);  view_203 = None
    
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
    view_207: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_14, [8, 784, 768])
    add_76: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_207, primals_398);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_118: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_76);  add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_60: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_57, clone_118);  clone_118 = None
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
    view_209: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_33, [8, 768, 784])
    permute_61: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_61: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_59, permute_61);  permute_61 = None
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
    view_211: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_43, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_31: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_211);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_119: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_31);  gelu_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_212: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_119, [6272, 3072]);  clone_119 = None
    t_59: "f32[3072, 768]" = torch.ops.aten.t.default(primals_411);  primals_411 = None
    addmm_44: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_412, view_212, t_59);  primals_412 = None
    view_213: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_44, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_120: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_213);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_62: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_60, clone_120);  clone_120 = None
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
    clamp_min_30: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_30, 1e-12)
    expand_90: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_30, [8, 16, 48, 784]);  clamp_min_30 = None
    div_36: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_273, expand_90);  expand_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_31: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_274, 2.0, [-1], True)
    clamp_min_31: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_31, 1e-12)
    expand_91: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_31, [8, 16, 48, 784]);  clamp_min_31 = None
    div_37: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_274, expand_91);  expand_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_16: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_37, -2, -1);  div_37 = None
    expand_92: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_36, [8, 16, 48, 784]);  div_36 = None
    clone_121: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
    _unsafe_view_60: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_121, [128, 48, 784]);  clone_121 = None
    expand_93: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_16, [8, 16, 784, 48]);  transpose_16 = None
    clone_122: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
    _unsafe_view_61: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_122, [128, 784, 48]);  clone_122 = None
    bmm_30: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_60, _unsafe_view_61)
    view_217: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_30, [8, 16, 48, 48])
    mul_63: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_217, primals_62);  view_217 = None
    
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
    view_221: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_15, [8, 784, 768])
    add_81: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_221, primals_418);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_126: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_81);  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_64: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_61, clone_126);  clone_126 = None
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
    view_223: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_35, [8, 768, 784])
    permute_65: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_223, [0, 2, 1]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_65: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_63, permute_65);  permute_65 = None
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
    view_225: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_33: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_225);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_127: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_33);  gelu_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_226: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_127, [6272, 3072]);  clone_127 = None
    t_63: "f32[3072, 768]" = torch.ops.aten.t.default(primals_431);  primals_431 = None
    addmm_47: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_432, view_226, t_63);  primals_432 = None
    view_227: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_47, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_128: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_227);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_66: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_64, clone_128);  clone_128 = None
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
    clamp_min_32: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_32, 1e-12)
    expand_96: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_32, [8, 16, 48, 784]);  clamp_min_32 = None
    div_38: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_290, expand_96);  expand_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_33: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_291, 2.0, [-1], True)
    clamp_min_33: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_33, 1e-12)
    expand_97: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_33, [8, 16, 48, 784]);  clamp_min_33 = None
    div_39: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_291, expand_97);  expand_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_17: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_39, -2, -1);  div_39 = None
    expand_98: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_38, [8, 16, 48, 784]);  div_38 = None
    clone_129: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
    _unsafe_view_64: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_129, [128, 48, 784]);  clone_129 = None
    expand_99: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_17, [8, 16, 784, 48]);  transpose_17 = None
    clone_130: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
    _unsafe_view_65: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_130, [128, 784, 48]);  clone_130 = None
    bmm_32: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_64, _unsafe_view_65)
    view_231: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_32, [8, 16, 48, 48])
    mul_67: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_231, primals_66);  view_231 = None
    
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
    view_235: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_16, [8, 784, 768])
    add_86: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_235, primals_438);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_134: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_86);  add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_68: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_65, clone_134);  clone_134 = None
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
    view_237: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_37, [8, 768, 784])
    permute_69: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_69: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_67, permute_69);  permute_69 = None
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
    view_239: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_49, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_35: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_239);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_135: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_35);  gelu_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_240: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_135, [6272, 3072]);  clone_135 = None
    t_67: "f32[3072, 768]" = torch.ops.aten.t.default(primals_451);  primals_451 = None
    addmm_50: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_452, view_240, t_67);  primals_452 = None
    view_241: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_50, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_136: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_241);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_70: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_68, clone_136);  clone_136 = None
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
    clamp_min_34: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_34, 1e-12)
    expand_102: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_34, [8, 16, 48, 784]);  clamp_min_34 = None
    div_40: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_307, expand_102);  expand_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_35: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_308, 2.0, [-1], True)
    clamp_min_35: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_35, 1e-12)
    expand_103: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_35, [8, 16, 48, 784]);  clamp_min_35 = None
    div_41: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_308, expand_103);  expand_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_18: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_41, -2, -1);  div_41 = None
    expand_104: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_40, [8, 16, 48, 784]);  div_40 = None
    clone_137: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
    _unsafe_view_68: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_137, [128, 48, 784]);  clone_137 = None
    expand_105: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_18, [8, 16, 784, 48]);  transpose_18 = None
    clone_138: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
    _unsafe_view_69: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_138, [128, 784, 48]);  clone_138 = None
    bmm_34: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_68, _unsafe_view_69)
    view_245: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_34, [8, 16, 48, 48])
    mul_71: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_245, primals_70);  view_245 = None
    
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
    view_249: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_17, [8, 784, 768])
    add_91: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_249, primals_458);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_142: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_91);  add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_72: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_69, clone_142);  clone_142 = None
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
    view_251: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_39, [8, 768, 784])
    permute_73: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_73: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_71, permute_73);  permute_73 = None
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
    view_253: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_52, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_37: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_253);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_143: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_37);  gelu_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_254: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_143, [6272, 3072]);  clone_143 = None
    t_71: "f32[3072, 768]" = torch.ops.aten.t.default(primals_471);  primals_471 = None
    addmm_53: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_472, view_254, t_71);  primals_472 = None
    view_255: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_53, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_144: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_255);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_74: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_72, clone_144);  clone_144 = None
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
    clamp_min_36: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_36, 1e-12)
    expand_108: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_36, [8, 16, 48, 784]);  clamp_min_36 = None
    div_42: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_324, expand_108);  expand_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_37: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_325, 2.0, [-1], True)
    clamp_min_37: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_37, 1e-12)
    expand_109: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_37, [8, 16, 48, 784]);  clamp_min_37 = None
    div_43: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_325, expand_109);  expand_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_19: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_43, -2, -1);  div_43 = None
    expand_110: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_42, [8, 16, 48, 784]);  div_42 = None
    clone_145: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_110, memory_format = torch.contiguous_format);  expand_110 = None
    _unsafe_view_72: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_145, [128, 48, 784]);  clone_145 = None
    expand_111: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_19, [8, 16, 784, 48]);  transpose_19 = None
    clone_146: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
    _unsafe_view_73: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_146, [128, 784, 48]);  clone_146 = None
    bmm_36: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_72, _unsafe_view_73)
    view_259: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_36, [8, 16, 48, 48])
    mul_75: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_259, primals_74);  view_259 = None
    
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
    view_263: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_18, [8, 784, 768])
    add_96: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_263, primals_478);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_150: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_96);  add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_76: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_73, clone_150);  clone_150 = None
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
    view_265: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_41, [8, 768, 784])
    permute_77: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_265, [0, 2, 1]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_77: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_75, permute_77);  permute_77 = None
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
    view_267: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_55, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_39: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_267);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_151: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_39);  gelu_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_268: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_151, [6272, 3072]);  clone_151 = None
    t_75: "f32[3072, 768]" = torch.ops.aten.t.default(primals_491);  primals_491 = None
    addmm_56: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_492, view_268, t_75);  primals_492 = None
    view_269: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_56, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_152: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_269);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_78: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_76, clone_152);  clone_152 = None
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
    clamp_min_38: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_38, 1e-12)
    expand_114: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_38, [8, 16, 48, 784]);  clamp_min_38 = None
    div_44: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_341, expand_114);  expand_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_39: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_342, 2.0, [-1], True)
    clamp_min_39: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_39, 1e-12)
    expand_115: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_39, [8, 16, 48, 784]);  clamp_min_39 = None
    div_45: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_342, expand_115);  expand_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_20: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_45, -2, -1);  div_45 = None
    expand_116: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_44, [8, 16, 48, 784]);  div_44 = None
    clone_153: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
    _unsafe_view_76: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_153, [128, 48, 784]);  clone_153 = None
    expand_117: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_20, [8, 16, 784, 48]);  transpose_20 = None
    clone_154: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
    _unsafe_view_77: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_154, [128, 784, 48]);  clone_154 = None
    bmm_38: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_76, _unsafe_view_77)
    view_273: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_38, [8, 16, 48, 48])
    mul_79: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_273, primals_78);  view_273 = None
    
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
    view_277: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_19, [8, 784, 768])
    add_101: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_277, primals_498);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_158: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_101);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_80: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_77, clone_158);  clone_158 = None
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
    view_279: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_43, [8, 768, 784])
    permute_81: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_279, [0, 2, 1]);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_81: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_79, permute_81);  permute_81 = None
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
    view_281: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_58, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_41: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_281);  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_159: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_41);  gelu_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_282: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_159, [6272, 3072]);  clone_159 = None
    t_79: "f32[3072, 768]" = torch.ops.aten.t.default(primals_511);  primals_511 = None
    addmm_59: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_512, view_282, t_79);  primals_512 = None
    view_283: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_59, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_160: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_283);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_82: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_80, clone_160);  clone_160 = None
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
    clamp_min_40: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_40, 1e-12)
    expand_120: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_40, [8, 16, 48, 784]);  clamp_min_40 = None
    div_46: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_358, expand_120);  expand_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_41: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_359, 2.0, [-1], True)
    clamp_min_41: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_41, 1e-12)
    expand_121: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_41, [8, 16, 48, 784]);  clamp_min_41 = None
    div_47: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_359, expand_121);  expand_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_21: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_47, -2, -1);  div_47 = None
    expand_122: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_46, [8, 16, 48, 784]);  div_46 = None
    clone_161: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
    _unsafe_view_80: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_161, [128, 48, 784]);  clone_161 = None
    expand_123: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_21, [8, 16, 784, 48]);  transpose_21 = None
    clone_162: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
    _unsafe_view_81: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_162, [128, 784, 48]);  clone_162 = None
    bmm_40: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_80, _unsafe_view_81)
    view_287: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_40, [8, 16, 48, 48])
    mul_83: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_287, primals_82);  view_287 = None
    
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
    view_291: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_20, [8, 784, 768])
    add_106: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_291, primals_518);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_166: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_106);  add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_84: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_81, clone_166);  clone_166 = None
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
    view_293: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_45, [8, 768, 784])
    permute_85: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_85: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_83, permute_85);  permute_85 = None
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
    view_295: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_61, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_43: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_295);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_167: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_43);  gelu_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_296: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_167, [6272, 3072]);  clone_167 = None
    t_83: "f32[3072, 768]" = torch.ops.aten.t.default(primals_531);  primals_531 = None
    addmm_62: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_532, view_296, t_83);  primals_532 = None
    view_297: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_62, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_168: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_297);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_86: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_84, clone_168);  clone_168 = None
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
    clamp_min_42: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_42, 1e-12)
    expand_126: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_42, [8, 16, 48, 784]);  clamp_min_42 = None
    div_48: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_375, expand_126);  expand_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_43: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_376, 2.0, [-1], True)
    clamp_min_43: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_43, 1e-12)
    expand_127: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_43, [8, 16, 48, 784]);  clamp_min_43 = None
    div_49: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_376, expand_127);  expand_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_22: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_49, -2, -1);  div_49 = None
    expand_128: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_48, [8, 16, 48, 784]);  div_48 = None
    clone_169: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
    _unsafe_view_84: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_169, [128, 48, 784]);  clone_169 = None
    expand_129: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_22, [8, 16, 784, 48]);  transpose_22 = None
    clone_170: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
    _unsafe_view_85: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_170, [128, 784, 48]);  clone_170 = None
    bmm_42: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_84, _unsafe_view_85)
    view_301: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_42, [8, 16, 48, 48])
    mul_87: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_301, primals_86);  view_301 = None
    
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
    view_305: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_21, [8, 784, 768])
    add_111: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_305, primals_538);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_174: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_111);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_88: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_85, clone_174);  clone_174 = None
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
    view_307: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_47, [8, 768, 784])
    permute_89: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_307, [0, 2, 1]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_89: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_87, permute_89);  permute_89 = None
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
    view_309: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_64, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_45: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_309);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_175: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_45);  gelu_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_310: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_175, [6272, 3072]);  clone_175 = None
    t_87: "f32[3072, 768]" = torch.ops.aten.t.default(primals_551);  primals_551 = None
    addmm_65: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_552, view_310, t_87);  primals_552 = None
    view_311: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_65, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_176: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_311);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_90: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_88, clone_176);  clone_176 = None
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
    clamp_min_44: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_44, 1e-12)
    expand_132: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_44, [8, 16, 48, 784]);  clamp_min_44 = None
    div_50: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_392, expand_132);  expand_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_45: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_393, 2.0, [-1], True)
    clamp_min_45: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_45, 1e-12)
    expand_133: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_45, [8, 16, 48, 784]);  clamp_min_45 = None
    div_51: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_393, expand_133);  expand_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_23: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_51, -2, -1);  div_51 = None
    expand_134: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_50, [8, 16, 48, 784]);  div_50 = None
    clone_177: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_134, memory_format = torch.contiguous_format);  expand_134 = None
    _unsafe_view_88: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_177, [128, 48, 784]);  clone_177 = None
    expand_135: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_23, [8, 16, 784, 48]);  transpose_23 = None
    clone_178: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
    _unsafe_view_89: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_178, [128, 784, 48]);  clone_178 = None
    bmm_44: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_88, _unsafe_view_89)
    view_315: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_44, [8, 16, 48, 48])
    mul_91: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_315, primals_90);  view_315 = None
    
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
    view_319: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_22, [8, 784, 768])
    add_116: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_319, primals_558);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_182: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_116);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_92: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_89, clone_182);  clone_182 = None
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
    view_321: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_49, [8, 768, 784])
    permute_93: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_93: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_91, permute_93);  permute_93 = None
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
    view_323: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_67, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_47: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_323);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_183: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_47);  gelu_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_324: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_183, [6272, 3072]);  clone_183 = None
    t_91: "f32[3072, 768]" = torch.ops.aten.t.default(primals_571);  primals_571 = None
    addmm_68: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_572, view_324, t_91);  primals_572 = None
    view_325: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_68, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_184: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_325);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_94: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_92, clone_184);  clone_184 = None
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
    clamp_min_46: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_46, 1e-12)
    expand_138: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_46, [8, 16, 48, 784]);  clamp_min_46 = None
    div_52: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_409, expand_138);  expand_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    linalg_vector_norm_47: "f32[8, 16, 48, 1]" = torch.ops.aten.linalg_vector_norm.default(getitem_410, 2.0, [-1], True)
    clamp_min_47: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(linalg_vector_norm_47, 1e-12)
    expand_139: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_47, [8, 16, 48, 784]);  clamp_min_47 = None
    div_53: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_410, expand_139);  expand_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_24: "f32[8, 16, 784, 48]" = torch.ops.aten.transpose.int(div_53, -2, -1);  div_53 = None
    expand_140: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_52, [8, 16, 48, 784]);  div_52 = None
    clone_185: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
    _unsafe_view_92: "f32[128, 48, 784]" = torch.ops.aten._unsafe_view.default(clone_185, [128, 48, 784]);  clone_185 = None
    expand_141: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(transpose_24, [8, 16, 784, 48]);  transpose_24 = None
    clone_186: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
    _unsafe_view_93: "f32[128, 784, 48]" = torch.ops.aten._unsafe_view.default(clone_186, [128, 784, 48]);  clone_186 = None
    bmm_46: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(_unsafe_view_92, _unsafe_view_93)
    view_329: "f32[8, 16, 48, 48]" = torch.ops.aten.view.default(bmm_46, [8, 16, 48, 48])
    mul_95: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_329, primals_94);  view_329 = None
    
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
    view_333: "f32[8, 784, 768]" = torch.ops.aten.view.default(mm_23, [8, 784, 768])
    add_121: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_333, primals_578);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:227, code: x = self.proj_drop(x)
    clone_190: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_121);  add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_96: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_93, clone_190);  clone_190 = None
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
    view_335: "f32[8, 768, 784]" = torch.ops.aten.view.default(convolution_51, [8, 768, 784])
    permute_97: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_335, [0, 2, 1]);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_97: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_95, permute_97);  permute_97 = None
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
    view_337: "f32[8, 784, 3072]" = torch.ops.aten.view.default(addmm_70, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_49: "f32[8, 784, 3072]" = torch.ops.aten.gelu.default(view_337);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_191: "f32[8, 784, 3072]" = torch.ops.aten.clone.default(gelu_49);  gelu_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_338: "f32[6272, 3072]" = torch.ops.aten.view.default(clone_191, [6272, 3072]);  clone_191 = None
    t_95: "f32[3072, 768]" = torch.ops.aten.t.default(primals_591);  primals_591 = None
    addmm_71: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_592, view_338, t_95);  primals_592 = None
    view_339: "f32[8, 784, 768]" = torch.ops.aten.view.default(addmm_71, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_192: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_339);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_98: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_96, clone_192);  clone_192 = None
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
    select: "f32[8, 768]" = torch.ops.aten.select.int(slice_29, 1, 0)
    t_96: "f32[768, 768]" = torch.ops.aten.t.default(primals_595);  primals_595 = None
    addmm_72: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_596, select, t_96);  primals_596 = None
    unsqueeze_3: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_72, 1);  addmm_72 = None
    view_340: "f32[8, 1, 16, 48]" = torch.ops.aten.view.default(unsqueeze_3, [8, 1, 16, 48]);  unsqueeze_3 = None
    permute_98: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_341: "f32[6280, 768]" = torch.ops.aten.view.default(getitem_423, [6280, 768]);  getitem_423 = None
    t_97: "f32[768, 768]" = torch.ops.aten.t.default(primals_597);  primals_597 = None
    addmm_73: "f32[6280, 768]" = torch.ops.aten.addmm.default(primals_598, view_341, t_97);  primals_598 = None
    view_342: "f32[8, 785, 768]" = torch.ops.aten.view.default(addmm_73, [8, 785, 768]);  addmm_73 = None
    view_343: "f32[8, 785, 16, 48]" = torch.ops.aten.view.default(view_342, [8, 785, 16, 48]);  view_342 = None
    permute_99: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    t_98: "f32[768, 768]" = torch.ops.aten.t.default(primals_599);  primals_599 = None
    addmm_74: "f32[6280, 768]" = torch.ops.aten.addmm.default(primals_600, view_341, t_98);  primals_600 = None
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
    slice_31: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_29, 1, 1, 9223372036854775807);  slice_29 = None
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
    slice_33: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(slice_32, 1, 0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_100: "f32[768, 3072]" = torch.ops.aten.t.default(primals_605);  primals_605 = None
    view_350: "f32[8, 768]" = torch.ops.aten.view.default(slice_33, [8, 768]);  slice_33 = None
    mm_24: "f32[8, 3072]" = torch.ops.aten.mm.default(view_350, t_100)
    view_351: "f32[8, 1, 3072]" = torch.ops.aten.view.default(mm_24, [8, 1, 3072])
    add_127: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(view_351, primals_606);  view_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_50: "f32[8, 1, 3072]" = torch.ops.aten.gelu.default(add_127);  add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_194: "f32[8, 1, 3072]" = torch.ops.aten.clone.default(gelu_50);  gelu_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_352: "f32[8, 3072]" = torch.ops.aten.view.default(clone_194, [8, 3072]);  clone_194 = None
    t_101: "f32[3072, 768]" = torch.ops.aten.t.default(primals_607);  primals_607 = None
    addmm_76: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_608, view_352, t_101);  primals_608 = None
    view_353: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_76, [8, 1, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_195: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_353);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    mul_100: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(primals_99, clone_195);  clone_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    slice_35: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_32, 1, 1, 9223372036854775807);  slice_32 = None
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
    select_1: "f32[8, 768]" = torch.ops.aten.select.int(slice_36, 1, 0)
    t_102: "f32[768, 768]" = torch.ops.aten.t.default(primals_611);  primals_611 = None
    addmm_77: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_612, select_1, t_102);  primals_612 = None
    unsqueeze_4: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_77, 1);  addmm_77 = None
    view_354: "f32[8, 1, 16, 48]" = torch.ops.aten.view.default(unsqueeze_4, [8, 1, 16, 48]);  unsqueeze_4 = None
    permute_101: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_355: "f32[6280, 768]" = torch.ops.aten.view.default(getitem_438, [6280, 768]);  getitem_438 = None
    t_103: "f32[768, 768]" = torch.ops.aten.t.default(primals_613);  primals_613 = None
    addmm_78: "f32[6280, 768]" = torch.ops.aten.addmm.default(primals_614, view_355, t_103);  primals_614 = None
    view_356: "f32[8, 785, 768]" = torch.ops.aten.view.default(addmm_78, [8, 785, 768]);  addmm_78 = None
    view_357: "f32[8, 785, 16, 48]" = torch.ops.aten.view.default(view_356, [8, 785, 16, 48]);  view_356 = None
    permute_102: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    t_104: "f32[768, 768]" = torch.ops.aten.t.default(primals_615);  primals_615 = None
    addmm_79: "f32[6280, 768]" = torch.ops.aten.addmm.default(primals_616, view_355, t_104);  primals_616 = None
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
    slice_38: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_36, 1, 1, 9223372036854775807);  slice_36 = None
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
    slice_40: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(slice_39, 1, 0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_106: "f32[768, 3072]" = torch.ops.aten.t.default(primals_621);  primals_621 = None
    view_364: "f32[8, 768]" = torch.ops.aten.view.default(slice_40, [8, 768]);  slice_40 = None
    mm_25: "f32[8, 3072]" = torch.ops.aten.mm.default(view_364, t_106)
    view_365: "f32[8, 1, 3072]" = torch.ops.aten.view.default(mm_25, [8, 1, 3072])
    add_130: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(view_365, primals_622);  view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    gelu_51: "f32[8, 1, 3072]" = torch.ops.aten.gelu.default(add_130);  add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_197: "f32[8, 1, 3072]" = torch.ops.aten.clone.default(gelu_51);  gelu_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_366: "f32[8, 3072]" = torch.ops.aten.view.default(clone_197, [8, 3072]);  clone_197 = None
    t_107: "f32[3072, 768]" = torch.ops.aten.t.default(primals_623);  primals_623 = None
    addmm_81: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_624, view_366, t_107);  primals_624 = None
    view_367: "f32[8, 1, 768]" = torch.ops.aten.view.default(addmm_81, [8, 1, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_198: "f32[8, 1, 768]" = torch.ops.aten.clone.default(view_367);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    mul_102: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(primals_101, clone_198);  clone_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    slice_42: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_39, 1, 1, 9223372036854775807);  slice_39 = None
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_113: "f32[768, 3072]" = torch.ops.aten.t.default(t_107);  t_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_119: "f32[3072, 768]" = torch.ops.aten.t.default(t_106);  t_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    t_121: "f32[768, 768]" = torch.ops.aten.t.default(t_105);  t_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    detach_74: "f32[8, 16, 1, 48]" = torch.ops.aten.detach.default(detach_73);  detach_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    t_125: "f32[768, 768]" = torch.ops.aten.t.default(t_104);  t_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    t_129: "f32[768, 768]" = torch.ops.aten.t.default(t_103);  t_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    t_133: "f32[768, 768]" = torch.ops.aten.t.default(t_102);  t_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_137: "f32[768, 3072]" = torch.ops.aten.t.default(t_101);  t_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_143: "f32[3072, 768]" = torch.ops.aten.t.default(t_100);  t_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    t_145: "f32[768, 768]" = torch.ops.aten.t.default(t_99);  t_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    detach_75: "f32[8, 16, 1, 48]" = torch.ops.aten.detach.default(detach_72);  detach_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    t_149: "f32[768, 768]" = torch.ops.aten.t.default(t_98);  t_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    t_153: "f32[768, 768]" = torch.ops.aten.t.default(t_97);  t_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    t_157: "f32[768, 768]" = torch.ops.aten.t.default(t_96);  t_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_161: "f32[768, 3072]" = torch.ops.aten.t.default(t_95);  t_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_165: "f32[3072, 768]" = torch.ops.aten.t.default(t_94);  t_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_171: "f32[768, 768]" = torch.ops.aten.t.default(t_93);  t_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_29: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_330, 1, 2);  view_330 = None
    transpose_30: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_94, 1, 2);  _unsafe_view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_76: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_71);  detach_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_31: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_92, 1, 2);  _unsafe_view_92 = None
    transpose_32: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_93, 1, 2);  _unsafe_view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_173: "f32[2304, 768]" = torch.ops.aten.t.default(t_92);  t_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_177: "f32[768, 3072]" = torch.ops.aten.t.default(t_91);  t_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_181: "f32[3072, 768]" = torch.ops.aten.t.default(t_90);  t_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_187: "f32[768, 768]" = torch.ops.aten.t.default(t_89);  t_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_34: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_316, 1, 2);  view_316 = None
    transpose_35: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_90, 1, 2);  _unsafe_view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_79: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_68);  detach_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_36: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_88, 1, 2);  _unsafe_view_88 = None
    transpose_37: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_89, 1, 2);  _unsafe_view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_189: "f32[2304, 768]" = torch.ops.aten.t.default(t_88);  t_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_193: "f32[768, 3072]" = torch.ops.aten.t.default(t_87);  t_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_197: "f32[3072, 768]" = torch.ops.aten.t.default(t_86);  t_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_203: "f32[768, 768]" = torch.ops.aten.t.default(t_85);  t_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_39: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_302, 1, 2);  view_302 = None
    transpose_40: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_86, 1, 2);  _unsafe_view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_82: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_65);  detach_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_41: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_84, 1, 2);  _unsafe_view_84 = None
    transpose_42: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_85, 1, 2);  _unsafe_view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_205: "f32[2304, 768]" = torch.ops.aten.t.default(t_84);  t_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_209: "f32[768, 3072]" = torch.ops.aten.t.default(t_83);  t_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_213: "f32[3072, 768]" = torch.ops.aten.t.default(t_82);  t_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_219: "f32[768, 768]" = torch.ops.aten.t.default(t_81);  t_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_44: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_288, 1, 2);  view_288 = None
    transpose_45: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_82, 1, 2);  _unsafe_view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_85: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_62);  detach_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_46: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_80, 1, 2);  _unsafe_view_80 = None
    transpose_47: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_81, 1, 2);  _unsafe_view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_221: "f32[2304, 768]" = torch.ops.aten.t.default(t_80);  t_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_225: "f32[768, 3072]" = torch.ops.aten.t.default(t_79);  t_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_229: "f32[3072, 768]" = torch.ops.aten.t.default(t_78);  t_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_235: "f32[768, 768]" = torch.ops.aten.t.default(t_77);  t_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_49: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_274, 1, 2);  view_274 = None
    transpose_50: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_78, 1, 2);  _unsafe_view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_88: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_59);  detach_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_51: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_76, 1, 2);  _unsafe_view_76 = None
    transpose_52: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_77, 1, 2);  _unsafe_view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_237: "f32[2304, 768]" = torch.ops.aten.t.default(t_76);  t_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_241: "f32[768, 3072]" = torch.ops.aten.t.default(t_75);  t_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_245: "f32[3072, 768]" = torch.ops.aten.t.default(t_74);  t_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_251: "f32[768, 768]" = torch.ops.aten.t.default(t_73);  t_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_54: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_260, 1, 2);  view_260 = None
    transpose_55: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_74, 1, 2);  _unsafe_view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_91: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_56);  detach_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_56: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_72, 1, 2);  _unsafe_view_72 = None
    transpose_57: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_73, 1, 2);  _unsafe_view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_253: "f32[2304, 768]" = torch.ops.aten.t.default(t_72);  t_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_257: "f32[768, 3072]" = torch.ops.aten.t.default(t_71);  t_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_261: "f32[3072, 768]" = torch.ops.aten.t.default(t_70);  t_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_267: "f32[768, 768]" = torch.ops.aten.t.default(t_69);  t_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_59: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_246, 1, 2);  view_246 = None
    transpose_60: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_70, 1, 2);  _unsafe_view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_94: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_53);  detach_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_61: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_68, 1, 2);  _unsafe_view_68 = None
    transpose_62: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_69, 1, 2);  _unsafe_view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_269: "f32[2304, 768]" = torch.ops.aten.t.default(t_68);  t_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_273: "f32[768, 3072]" = torch.ops.aten.t.default(t_67);  t_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_277: "f32[3072, 768]" = torch.ops.aten.t.default(t_66);  t_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_283: "f32[768, 768]" = torch.ops.aten.t.default(t_65);  t_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_64: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_232, 1, 2);  view_232 = None
    transpose_65: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_66, 1, 2);  _unsafe_view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_97: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_50);  detach_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_66: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_64, 1, 2);  _unsafe_view_64 = None
    transpose_67: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_65, 1, 2);  _unsafe_view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_285: "f32[2304, 768]" = torch.ops.aten.t.default(t_64);  t_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_289: "f32[768, 3072]" = torch.ops.aten.t.default(t_63);  t_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_293: "f32[3072, 768]" = torch.ops.aten.t.default(t_62);  t_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_299: "f32[768, 768]" = torch.ops.aten.t.default(t_61);  t_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_69: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_218, 1, 2);  view_218 = None
    transpose_70: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_62, 1, 2);  _unsafe_view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_100: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_47);  detach_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_71: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_60, 1, 2);  _unsafe_view_60 = None
    transpose_72: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_61, 1, 2);  _unsafe_view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_301: "f32[2304, 768]" = torch.ops.aten.t.default(t_60);  t_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_305: "f32[768, 3072]" = torch.ops.aten.t.default(t_59);  t_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_309: "f32[3072, 768]" = torch.ops.aten.t.default(t_58);  t_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_315: "f32[768, 768]" = torch.ops.aten.t.default(t_57);  t_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_74: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_204, 1, 2);  view_204 = None
    transpose_75: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_58, 1, 2);  _unsafe_view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_103: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_44);  detach_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_76: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_56, 1, 2);  _unsafe_view_56 = None
    transpose_77: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_57, 1, 2);  _unsafe_view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_317: "f32[2304, 768]" = torch.ops.aten.t.default(t_56);  t_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_321: "f32[768, 3072]" = torch.ops.aten.t.default(t_55);  t_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_325: "f32[3072, 768]" = torch.ops.aten.t.default(t_54);  t_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_331: "f32[768, 768]" = torch.ops.aten.t.default(t_53);  t_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_79: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_190, 1, 2);  view_190 = None
    transpose_80: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_54, 1, 2);  _unsafe_view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_106: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_41);  detach_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_81: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_52, 1, 2);  _unsafe_view_52 = None
    transpose_82: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_53, 1, 2);  _unsafe_view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_333: "f32[2304, 768]" = torch.ops.aten.t.default(t_52);  t_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_337: "f32[768, 3072]" = torch.ops.aten.t.default(t_51);  t_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_341: "f32[3072, 768]" = torch.ops.aten.t.default(t_50);  t_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_347: "f32[768, 768]" = torch.ops.aten.t.default(t_49);  t_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_84: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_176, 1, 2);  view_176 = None
    transpose_85: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_50, 1, 2);  _unsafe_view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_109: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_38);  detach_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_86: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_48, 1, 2);  _unsafe_view_48 = None
    transpose_87: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_49, 1, 2);  _unsafe_view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_349: "f32[2304, 768]" = torch.ops.aten.t.default(t_48);  t_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_353: "f32[768, 3072]" = torch.ops.aten.t.default(t_47);  t_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_357: "f32[3072, 768]" = torch.ops.aten.t.default(t_46);  t_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_363: "f32[768, 768]" = torch.ops.aten.t.default(t_45);  t_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_89: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_162, 1, 2);  view_162 = None
    transpose_90: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_46, 1, 2);  _unsafe_view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_112: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_35);  detach_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_91: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_44, 1, 2);  _unsafe_view_44 = None
    transpose_92: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_45, 1, 2);  _unsafe_view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_365: "f32[2304, 768]" = torch.ops.aten.t.default(t_44);  t_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_369: "f32[768, 3072]" = torch.ops.aten.t.default(t_43);  t_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_373: "f32[3072, 768]" = torch.ops.aten.t.default(t_42);  t_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_379: "f32[768, 768]" = torch.ops.aten.t.default(t_41);  t_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_94: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_148, 1, 2);  view_148 = None
    transpose_95: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_42, 1, 2);  _unsafe_view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_115: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_32);  detach_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_96: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_40, 1, 2);  _unsafe_view_40 = None
    transpose_97: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_41, 1, 2);  _unsafe_view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_381: "f32[2304, 768]" = torch.ops.aten.t.default(t_40);  t_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_385: "f32[768, 3072]" = torch.ops.aten.t.default(t_39);  t_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_389: "f32[3072, 768]" = torch.ops.aten.t.default(t_38);  t_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_395: "f32[768, 768]" = torch.ops.aten.t.default(t_37);  t_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_99: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_134, 1, 2);  view_134 = None
    transpose_100: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_38, 1, 2);  _unsafe_view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_118: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_29);  detach_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_101: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_36, 1, 2);  _unsafe_view_36 = None
    transpose_102: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_37, 1, 2);  _unsafe_view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_397: "f32[2304, 768]" = torch.ops.aten.t.default(t_36);  t_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_401: "f32[768, 3072]" = torch.ops.aten.t.default(t_35);  t_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_405: "f32[3072, 768]" = torch.ops.aten.t.default(t_34);  t_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_411: "f32[768, 768]" = torch.ops.aten.t.default(t_33);  t_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_104: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_120, 1, 2);  view_120 = None
    transpose_105: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_34, 1, 2);  _unsafe_view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_121: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_26);  detach_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_106: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_32, 1, 2);  _unsafe_view_32 = None
    transpose_107: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_33, 1, 2);  _unsafe_view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_413: "f32[2304, 768]" = torch.ops.aten.t.default(t_32);  t_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_417: "f32[768, 3072]" = torch.ops.aten.t.default(t_31);  t_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_421: "f32[3072, 768]" = torch.ops.aten.t.default(t_30);  t_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_427: "f32[768, 768]" = torch.ops.aten.t.default(t_29);  t_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_109: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_106, 1, 2);  view_106 = None
    transpose_110: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_30, 1, 2);  _unsafe_view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_124: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_23);  detach_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_111: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_28, 1, 2);  _unsafe_view_28 = None
    transpose_112: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_29, 1, 2);  _unsafe_view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_429: "f32[2304, 768]" = torch.ops.aten.t.default(t_28);  t_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_433: "f32[768, 3072]" = torch.ops.aten.t.default(t_27);  t_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_437: "f32[3072, 768]" = torch.ops.aten.t.default(t_26);  t_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_443: "f32[768, 768]" = torch.ops.aten.t.default(t_25);  t_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_114: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_92, 1, 2);  view_92 = None
    transpose_115: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_26, 1, 2);  _unsafe_view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_127: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_20);  detach_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_116: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_24, 1, 2);  _unsafe_view_24 = None
    transpose_117: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_25, 1, 2);  _unsafe_view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_445: "f32[2304, 768]" = torch.ops.aten.t.default(t_24);  t_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_449: "f32[768, 3072]" = torch.ops.aten.t.default(t_23);  t_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_453: "f32[3072, 768]" = torch.ops.aten.t.default(t_22);  t_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_459: "f32[768, 768]" = torch.ops.aten.t.default(t_21);  t_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_119: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_78, 1, 2);  view_78 = None
    transpose_120: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_22, 1, 2);  _unsafe_view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_130: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_17);  detach_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_121: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_20, 1, 2);  _unsafe_view_20 = None
    transpose_122: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_21, 1, 2);  _unsafe_view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_461: "f32[2304, 768]" = torch.ops.aten.t.default(t_20);  t_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_465: "f32[768, 3072]" = torch.ops.aten.t.default(t_19);  t_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_469: "f32[3072, 768]" = torch.ops.aten.t.default(t_18);  t_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_475: "f32[768, 768]" = torch.ops.aten.t.default(t_17);  t_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_124: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_64, 1, 2);  view_64 = None
    transpose_125: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_18, 1, 2);  _unsafe_view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_133: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_14);  detach_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_126: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_16, 1, 2);  _unsafe_view_16 = None
    transpose_127: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_17, 1, 2);  _unsafe_view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_477: "f32[2304, 768]" = torch.ops.aten.t.default(t_16);  t_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_481: "f32[768, 3072]" = torch.ops.aten.t.default(t_15);  t_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_485: "f32[3072, 768]" = torch.ops.aten.t.default(t_14);  t_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_491: "f32[768, 768]" = torch.ops.aten.t.default(t_13);  t_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_129: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_50, 1, 2);  view_50 = None
    transpose_130: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_14, 1, 2);  _unsafe_view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_136: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_11);  detach_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_131: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_12, 1, 2);  _unsafe_view_12 = None
    transpose_132: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_13, 1, 2);  _unsafe_view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_493: "f32[2304, 768]" = torch.ops.aten.t.default(t_12);  t_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_497: "f32[768, 3072]" = torch.ops.aten.t.default(t_11);  t_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_501: "f32[3072, 768]" = torch.ops.aten.t.default(t_10);  t_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_507: "f32[768, 768]" = torch.ops.aten.t.default(t_9);  t_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_134: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_36, 1, 2);  view_36 = None
    transpose_135: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_10, 1, 2);  _unsafe_view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_139: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_8);  detach_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_136: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_8, 1, 2);  _unsafe_view_8 = None
    transpose_137: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_9, 1, 2);  _unsafe_view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_509: "f32[2304, 768]" = torch.ops.aten.t.default(t_8);  t_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_513: "f32[768, 3072]" = torch.ops.aten.t.default(t_7);  t_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_517: "f32[3072, 768]" = torch.ops.aten.t.default(t_6);  t_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_523: "f32[768, 768]" = torch.ops.aten.t.default(t_5);  t_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_139: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_22, 1, 2);  view_22 = None
    transpose_140: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_6, 1, 2);  _unsafe_view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_142: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_5);  detach_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_141: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_4, 1, 2);  _unsafe_view_4 = None
    transpose_142: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_5, 1, 2);  _unsafe_view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_525: "f32[2304, 768]" = torch.ops.aten.t.default(t_4);  t_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    t_529: "f32[768, 3072]" = torch.ops.aten.t.default(t_3);  t_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    t_533: "f32[3072, 768]" = torch.ops.aten.t.default(t_2);  t_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    t_539: "f32[768, 768]" = torch.ops.aten.t.default(t_1);  t_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    transpose_144: "f32[128, 48, 48]" = torch.ops.aten.transpose.int(view_8, 1, 2);  view_8 = None
    transpose_145: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view_2, 1, 2);  _unsafe_view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    detach_145: "f32[8, 16, 48, 48]" = torch.ops.aten.detach.default(detach_2);  detach_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    transpose_146: "f32[128, 784, 48]" = torch.ops.aten.transpose.int(_unsafe_view, 1, 2);  _unsafe_view = None
    transpose_147: "f32[128, 48, 784]" = torch.ops.aten.transpose.int(_unsafe_view_1, 1, 2);  _unsafe_view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    t_541: "f32[2304, 768]" = torch.ops.aten.t.default(t);  t = None
    return [getitem_3, getitem_4, add, getitem_8, getitem_9, add_1, getitem_13, getitem_14, add_2, getitem_27, getitem_28, add_8, getitem_44, getitem_45, add_13, getitem_61, getitem_62, add_18, getitem_78, getitem_79, add_23, getitem_95, getitem_96, add_28, getitem_112, getitem_113, add_33, getitem_129, getitem_130, add_38, getitem_146, getitem_147, add_43, getitem_163, getitem_164, add_48, getitem_180, getitem_181, add_53, getitem_197, getitem_198, add_58, getitem_214, getitem_215, add_63, getitem_231, getitem_232, add_68, getitem_248, getitem_249, add_73, getitem_265, getitem_266, add_78, getitem_282, getitem_283, add_83, getitem_299, getitem_300, add_88, getitem_316, getitem_317, add_93, getitem_333, getitem_334, add_98, getitem_350, getitem_351, add_103, getitem_367, getitem_368, add_108, getitem_384, getitem_385, add_113, getitem_401, getitem_402, add_118, getitem_418, getitem_419, add_123, addmm_82, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_113, primals_114, primals_118, primals_119, primals_120, primals_121, primals_123, primals_125, primals_127, primals_128, primals_133, primals_134, primals_138, primals_139, primals_140, primals_141, primals_143, primals_145, primals_147, primals_148, primals_153, primals_154, primals_158, primals_159, primals_160, primals_161, primals_163, primals_165, primals_167, primals_168, primals_173, primals_174, primals_178, primals_179, primals_180, primals_181, primals_183, primals_185, primals_187, primals_188, primals_193, primals_194, primals_198, primals_199, primals_200, primals_201, primals_203, primals_205, primals_207, primals_208, primals_213, primals_214, primals_218, primals_219, primals_220, primals_221, primals_223, primals_225, primals_227, primals_228, primals_233, primals_234, primals_238, primals_239, primals_240, primals_241, primals_243, primals_245, primals_247, primals_248, primals_253, primals_254, primals_258, primals_259, primals_260, primals_261, primals_263, primals_265, primals_267, primals_268, primals_273, primals_274, primals_278, primals_279, primals_280, primals_281, primals_283, primals_285, primals_287, primals_288, primals_293, primals_294, primals_298, primals_299, primals_300, primals_301, primals_303, primals_305, primals_307, primals_308, primals_313, primals_314, primals_318, primals_319, primals_320, primals_321, primals_323, primals_325, primals_327, primals_328, primals_333, primals_334, primals_338, primals_339, primals_340, primals_341, primals_343, primals_345, primals_347, primals_348, primals_353, primals_354, primals_358, primals_359, primals_360, primals_361, primals_363, primals_365, primals_367, primals_368, primals_373, primals_374, primals_378, primals_379, primals_380, primals_381, primals_383, primals_385, primals_387, primals_388, primals_393, primals_394, primals_398, primals_399, primals_400, primals_401, primals_403, primals_405, primals_407, primals_408, primals_413, primals_414, primals_418, primals_419, primals_420, primals_421, primals_423, primals_425, primals_427, primals_428, primals_433, primals_434, primals_438, primals_439, primals_440, primals_441, primals_443, primals_445, primals_447, primals_448, primals_453, primals_454, primals_458, primals_459, primals_460, primals_461, primals_463, primals_465, primals_467, primals_468, primals_473, primals_474, primals_478, primals_479, primals_480, primals_481, primals_483, primals_485, primals_487, primals_488, primals_493, primals_494, primals_498, primals_499, primals_500, primals_501, primals_503, primals_505, primals_507, primals_508, primals_513, primals_514, primals_518, primals_519, primals_520, primals_521, primals_523, primals_525, primals_527, primals_528, primals_533, primals_534, primals_538, primals_539, primals_540, primals_541, primals_543, primals_545, primals_547, primals_548, primals_553, primals_554, primals_558, primals_559, primals_560, primals_561, primals_563, primals_565, primals_567, primals_568, primals_573, primals_574, primals_578, primals_579, primals_580, primals_581, primals_583, primals_585, primals_587, primals_588, primals_593, primals_594, primals_603, primals_604, primals_606, primals_609, primals_610, primals_619, primals_620, primals_622, primals_625, primals_626, primals_710, convolution, getitem, getitem_1, getitem_2, getitem_3, getitem_4, gelu, convolution_1, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, gelu_1, convolution_2, getitem_11, getitem_12, getitem_13, getitem_14, permute, clone, getitem_16, getitem_17, view_4, getitem_18, getitem_19, linalg_vector_norm, linalg_vector_norm_1, bmm, _unsafe_view_3, mm, add_7, getitem_22, getitem_23, view_12, convolution_4, gelu_2, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, convolution_5, add_9, getitem_30, getitem_31, view_14, addmm_1, view_16, addmm_2, add_10, getitem_33, getitem_34, view_18, getitem_35, getitem_36, linalg_vector_norm_2, linalg_vector_norm_3, bmm_2, _unsafe_view_7, mm_1, add_12, getitem_39, getitem_40, view_26, convolution_6, gelu_4, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, convolution_7, add_14, getitem_47, getitem_48, view_28, addmm_4, view_30, addmm_5, add_15, getitem_50, getitem_51, view_32, getitem_52, getitem_53, linalg_vector_norm_4, linalg_vector_norm_5, bmm_4, _unsafe_view_11, mm_2, add_17, getitem_56, getitem_57, view_40, convolution_8, gelu_6, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, convolution_9, add_19, getitem_64, getitem_65, view_42, addmm_7, view_44, addmm_8, add_20, getitem_67, getitem_68, view_46, getitem_69, getitem_70, linalg_vector_norm_6, linalg_vector_norm_7, bmm_6, _unsafe_view_15, mm_3, add_22, getitem_73, getitem_74, view_54, convolution_10, gelu_8, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, convolution_11, add_24, getitem_81, getitem_82, view_56, addmm_10, view_58, addmm_11, add_25, getitem_84, getitem_85, view_60, getitem_86, getitem_87, linalg_vector_norm_8, linalg_vector_norm_9, bmm_8, _unsafe_view_19, mm_4, add_27, getitem_90, getitem_91, view_68, convolution_12, gelu_10, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, convolution_13, add_29, getitem_98, getitem_99, view_70, addmm_13, view_72, addmm_14, add_30, getitem_101, getitem_102, view_74, getitem_103, getitem_104, linalg_vector_norm_10, linalg_vector_norm_11, bmm_10, _unsafe_view_23, mm_5, add_32, getitem_107, getitem_108, view_82, convolution_14, gelu_12, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, convolution_15, add_34, getitem_115, getitem_116, view_84, addmm_16, view_86, addmm_17, add_35, getitem_118, getitem_119, view_88, getitem_120, getitem_121, linalg_vector_norm_12, linalg_vector_norm_13, bmm_12, _unsafe_view_27, mm_6, add_37, getitem_124, getitem_125, view_96, convolution_16, gelu_14, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, convolution_17, add_39, getitem_132, getitem_133, view_98, addmm_19, view_100, addmm_20, add_40, getitem_135, getitem_136, view_102, getitem_137, getitem_138, linalg_vector_norm_14, linalg_vector_norm_15, bmm_14, _unsafe_view_31, mm_7, add_42, getitem_141, getitem_142, view_110, convolution_18, gelu_16, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, convolution_19, add_44, getitem_149, getitem_150, view_112, addmm_22, view_114, addmm_23, add_45, getitem_152, getitem_153, view_116, getitem_154, getitem_155, linalg_vector_norm_16, linalg_vector_norm_17, bmm_16, _unsafe_view_35, mm_8, add_47, getitem_158, getitem_159, view_124, convolution_20, gelu_18, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, convolution_21, add_49, getitem_166, getitem_167, view_126, addmm_25, view_128, addmm_26, add_50, getitem_169, getitem_170, view_130, getitem_171, getitem_172, linalg_vector_norm_18, linalg_vector_norm_19, bmm_18, _unsafe_view_39, mm_9, add_52, getitem_175, getitem_176, view_138, convolution_22, gelu_20, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, convolution_23, add_54, getitem_183, getitem_184, view_140, addmm_28, view_142, addmm_29, add_55, getitem_186, getitem_187, view_144, getitem_188, getitem_189, linalg_vector_norm_20, linalg_vector_norm_21, bmm_20, _unsafe_view_43, mm_10, add_57, getitem_192, getitem_193, view_152, convolution_24, gelu_22, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, convolution_25, add_59, getitem_200, getitem_201, view_154, addmm_31, view_156, addmm_32, add_60, getitem_203, getitem_204, view_158, getitem_205, getitem_206, linalg_vector_norm_22, linalg_vector_norm_23, bmm_22, _unsafe_view_47, mm_11, add_62, getitem_209, getitem_210, view_166, convolution_26, gelu_24, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215, convolution_27, add_64, getitem_217, getitem_218, view_168, addmm_34, view_170, addmm_35, add_65, getitem_220, getitem_221, view_172, getitem_222, getitem_223, linalg_vector_norm_24, linalg_vector_norm_25, bmm_24, _unsafe_view_51, mm_12, add_67, getitem_226, getitem_227, view_180, convolution_28, gelu_26, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, convolution_29, add_69, getitem_234, getitem_235, view_182, addmm_37, view_184, addmm_38, add_70, getitem_237, getitem_238, view_186, getitem_239, getitem_240, linalg_vector_norm_26, linalg_vector_norm_27, bmm_26, _unsafe_view_55, mm_13, add_72, getitem_243, getitem_244, view_194, convolution_30, gelu_28, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, convolution_31, add_74, getitem_251, getitem_252, view_196, addmm_40, view_198, addmm_41, add_75, getitem_254, getitem_255, view_200, getitem_256, getitem_257, linalg_vector_norm_28, linalg_vector_norm_29, bmm_28, _unsafe_view_59, mm_14, add_77, getitem_260, getitem_261, view_208, convolution_32, gelu_30, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, convolution_33, add_79, getitem_268, getitem_269, view_210, addmm_43, view_212, addmm_44, add_80, getitem_271, getitem_272, view_214, getitem_273, getitem_274, linalg_vector_norm_30, linalg_vector_norm_31, bmm_30, _unsafe_view_63, mm_15, add_82, getitem_277, getitem_278, view_222, convolution_34, gelu_32, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, convolution_35, add_84, getitem_285, getitem_286, view_224, addmm_46, view_226, addmm_47, add_85, getitem_288, getitem_289, view_228, getitem_290, getitem_291, linalg_vector_norm_32, linalg_vector_norm_33, bmm_32, _unsafe_view_67, mm_16, add_87, getitem_294, getitem_295, view_236, convolution_36, gelu_34, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300, convolution_37, add_89, getitem_302, getitem_303, view_238, addmm_49, view_240, addmm_50, add_90, getitem_305, getitem_306, view_242, getitem_307, getitem_308, linalg_vector_norm_34, linalg_vector_norm_35, bmm_34, _unsafe_view_71, mm_17, add_92, getitem_311, getitem_312, view_250, convolution_38, gelu_36, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317, convolution_39, add_94, getitem_319, getitem_320, view_252, addmm_52, view_254, addmm_53, add_95, getitem_322, getitem_323, view_256, getitem_324, getitem_325, linalg_vector_norm_36, linalg_vector_norm_37, bmm_36, _unsafe_view_75, mm_18, add_97, getitem_328, getitem_329, view_264, convolution_40, gelu_38, getitem_330, getitem_331, getitem_332, getitem_333, getitem_334, convolution_41, add_99, getitem_336, getitem_337, view_266, addmm_55, view_268, addmm_56, add_100, getitem_339, getitem_340, view_270, getitem_341, getitem_342, linalg_vector_norm_38, linalg_vector_norm_39, bmm_38, _unsafe_view_79, mm_19, add_102, getitem_345, getitem_346, view_278, convolution_42, gelu_40, getitem_347, getitem_348, getitem_349, getitem_350, getitem_351, convolution_43, add_104, getitem_353, getitem_354, view_280, addmm_58, view_282, addmm_59, add_105, getitem_356, getitem_357, view_284, getitem_358, getitem_359, linalg_vector_norm_40, linalg_vector_norm_41, bmm_40, _unsafe_view_83, mm_20, add_107, getitem_362, getitem_363, view_292, convolution_44, gelu_42, getitem_364, getitem_365, getitem_366, getitem_367, getitem_368, convolution_45, add_109, getitem_370, getitem_371, view_294, addmm_61, view_296, addmm_62, add_110, getitem_373, getitem_374, view_298, getitem_375, getitem_376, linalg_vector_norm_42, linalg_vector_norm_43, bmm_42, _unsafe_view_87, mm_21, add_112, getitem_379, getitem_380, view_306, convolution_46, gelu_44, getitem_381, getitem_382, getitem_383, getitem_384, getitem_385, convolution_47, add_114, getitem_387, getitem_388, view_308, addmm_64, view_310, addmm_65, add_115, getitem_390, getitem_391, view_312, getitem_392, getitem_393, linalg_vector_norm_44, linalg_vector_norm_45, bmm_44, _unsafe_view_91, mm_22, add_117, getitem_396, getitem_397, view_320, convolution_48, gelu_46, getitem_398, getitem_399, getitem_400, getitem_401, getitem_402, convolution_49, add_119, getitem_404, getitem_405, view_322, addmm_67, view_324, addmm_68, add_120, getitem_407, getitem_408, view_326, getitem_409, getitem_410, linalg_vector_norm_46, linalg_vector_norm_47, bmm_46, _unsafe_view_95, mm_23, add_122, getitem_413, getitem_414, view_334, convolution_50, gelu_48, getitem_415, getitem_416, getitem_417, getitem_418, getitem_419, convolution_51, add_124, getitem_421, getitem_422, view_336, addmm_70, view_338, addmm_71, cat_1, getitem_424, getitem_425, select, permute_98, view_341, permute_99, permute_100, getitem_427, getitem_428, getitem_429, getitem_432, getitem_433, view_348, cat_2, add_126, getitem_436, getitem_437, view_350, mm_24, view_352, addmm_76, add_128, getitem_439, getitem_440, select_1, permute_101, view_355, permute_102, permute_103, getitem_442, getitem_443, getitem_444, getitem_447, getitem_448, view_362, cat_4, add_129, getitem_451, getitem_452, view_364, mm_25, view_366, addmm_81, add_131, getitem_454, getitem_455, clone_199, t_109, t_113, t_119, t_121, detach_74, t_125, t_129, t_133, t_137, t_143, t_145, detach_75, t_149, t_153, t_157, t_161, t_165, t_171, transpose_29, transpose_30, detach_76, transpose_31, transpose_32, t_173, t_177, t_181, t_187, transpose_34, transpose_35, detach_79, transpose_36, transpose_37, t_189, t_193, t_197, t_203, transpose_39, transpose_40, detach_82, transpose_41, transpose_42, t_205, t_209, t_213, t_219, transpose_44, transpose_45, detach_85, transpose_46, transpose_47, t_221, t_225, t_229, t_235, transpose_49, transpose_50, detach_88, transpose_51, transpose_52, t_237, t_241, t_245, t_251, transpose_54, transpose_55, detach_91, transpose_56, transpose_57, t_253, t_257, t_261, t_267, transpose_59, transpose_60, detach_94, transpose_61, transpose_62, t_269, t_273, t_277, t_283, transpose_64, transpose_65, detach_97, transpose_66, transpose_67, t_285, t_289, t_293, t_299, transpose_69, transpose_70, detach_100, transpose_71, transpose_72, t_301, t_305, t_309, t_315, transpose_74, transpose_75, detach_103, transpose_76, transpose_77, t_317, t_321, t_325, t_331, transpose_79, transpose_80, detach_106, transpose_81, transpose_82, t_333, t_337, t_341, t_347, transpose_84, transpose_85, detach_109, transpose_86, transpose_87, t_349, t_353, t_357, t_363, transpose_89, transpose_90, detach_112, transpose_91, transpose_92, t_365, t_369, t_373, t_379, transpose_94, transpose_95, detach_115, transpose_96, transpose_97, t_381, t_385, t_389, t_395, transpose_99, transpose_100, detach_118, transpose_101, transpose_102, t_397, t_401, t_405, t_411, transpose_104, transpose_105, detach_121, transpose_106, transpose_107, t_413, t_417, t_421, t_427, transpose_109, transpose_110, detach_124, transpose_111, transpose_112, t_429, t_433, t_437, t_443, transpose_114, transpose_115, detach_127, transpose_116, transpose_117, t_445, t_449, t_453, t_459, transpose_119, transpose_120, detach_130, transpose_121, transpose_122, t_461, t_465, t_469, t_475, transpose_124, transpose_125, detach_133, transpose_126, transpose_127, t_477, t_481, t_485, t_491, transpose_129, transpose_130, detach_136, transpose_131, transpose_132, t_493, t_497, t_501, t_507, transpose_134, transpose_135, detach_139, transpose_136, transpose_137, t_509, t_513, t_517, t_523, transpose_139, transpose_140, detach_142, transpose_141, transpose_142, t_525, t_529, t_533, t_539, transpose_144, transpose_145, detach_145, transpose_146, transpose_147, t_541]
    