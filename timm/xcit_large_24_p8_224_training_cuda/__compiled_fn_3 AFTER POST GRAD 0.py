from __future__ import annotations



def forward(self, primals_1: "f32[768]", primals_2: "f32[16, 1, 1]", primals_3: "f32[768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[16, 1, 1]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[16, 1, 1]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_14: "f32[16, 1, 1]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[16, 1, 1]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[16, 1, 1]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[16, 1, 1]", primals_27: "f32[768]", primals_28: "f32[768]", primals_29: "f32[768]", primals_30: "f32[16, 1, 1]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[16, 1, 1]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[768]", primals_38: "f32[16, 1, 1]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[16, 1, 1]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[768]", primals_46: "f32[16, 1, 1]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[768]", primals_50: "f32[16, 1, 1]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[16, 1, 1]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[16, 1, 1]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[16, 1, 1]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[16, 1, 1]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[16, 1, 1]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[16, 1, 1]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[768]", primals_78: "f32[16, 1, 1]", primals_79: "f32[768]", primals_80: "f32[768]", primals_81: "f32[768]", primals_82: "f32[16, 1, 1]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[16, 1, 1]", primals_87: "f32[768]", primals_88: "f32[768]", primals_89: "f32[768]", primals_90: "f32[16, 1, 1]", primals_91: "f32[768]", primals_92: "f32[768]", primals_93: "f32[768]", primals_94: "f32[16, 1, 1]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[1, 1, 768]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[192, 3, 3, 3]", primals_103: "f32[192]", primals_104: "f32[192]", primals_105: "f32[384, 192, 3, 3]", primals_106: "f32[384]", primals_107: "f32[384]", primals_108: "f32[768, 384, 3, 3]", primals_109: "f32[768]", primals_110: "f32[768]", primals_111: "f32[768, 64, 1, 1]", primals_112: "f32[768]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[2304, 768]", primals_116: "f32[2304]", primals_117: "f32[768, 768]", primals_118: "f32[768]", primals_119: "f32[768]", primals_120: "f32[768]", primals_121: "f32[768, 1, 3, 3]", primals_122: "f32[768]", primals_123: "f32[768]", primals_124: "f32[768]", primals_125: "f32[768, 1, 3, 3]", primals_126: "f32[768]", primals_127: "f32[768]", primals_128: "f32[768]", primals_129: "f32[3072, 768]", primals_130: "f32[3072]", primals_131: "f32[768, 3072]", primals_132: "f32[768]", primals_133: "f32[768]", primals_134: "f32[768]", primals_135: "f32[2304, 768]", primals_136: "f32[2304]", primals_137: "f32[768, 768]", primals_138: "f32[768]", primals_139: "f32[768]", primals_140: "f32[768]", primals_141: "f32[768, 1, 3, 3]", primals_142: "f32[768]", primals_143: "f32[768]", primals_144: "f32[768]", primals_145: "f32[768, 1, 3, 3]", primals_146: "f32[768]", primals_147: "f32[768]", primals_148: "f32[768]", primals_149: "f32[3072, 768]", primals_150: "f32[3072]", primals_151: "f32[768, 3072]", primals_152: "f32[768]", primals_153: "f32[768]", primals_154: "f32[768]", primals_155: "f32[2304, 768]", primals_156: "f32[2304]", primals_157: "f32[768, 768]", primals_158: "f32[768]", primals_159: "f32[768]", primals_160: "f32[768]", primals_161: "f32[768, 1, 3, 3]", primals_162: "f32[768]", primals_163: "f32[768]", primals_164: "f32[768]", primals_165: "f32[768, 1, 3, 3]", primals_166: "f32[768]", primals_167: "f32[768]", primals_168: "f32[768]", primals_169: "f32[3072, 768]", primals_170: "f32[3072]", primals_171: "f32[768, 3072]", primals_172: "f32[768]", primals_173: "f32[768]", primals_174: "f32[768]", primals_175: "f32[2304, 768]", primals_176: "f32[2304]", primals_177: "f32[768, 768]", primals_178: "f32[768]", primals_179: "f32[768]", primals_180: "f32[768]", primals_181: "f32[768, 1, 3, 3]", primals_182: "f32[768]", primals_183: "f32[768]", primals_184: "f32[768]", primals_185: "f32[768, 1, 3, 3]", primals_186: "f32[768]", primals_187: "f32[768]", primals_188: "f32[768]", primals_189: "f32[3072, 768]", primals_190: "f32[3072]", primals_191: "f32[768, 3072]", primals_192: "f32[768]", primals_193: "f32[768]", primals_194: "f32[768]", primals_195: "f32[2304, 768]", primals_196: "f32[2304]", primals_197: "f32[768, 768]", primals_198: "f32[768]", primals_199: "f32[768]", primals_200: "f32[768]", primals_201: "f32[768, 1, 3, 3]", primals_202: "f32[768]", primals_203: "f32[768]", primals_204: "f32[768]", primals_205: "f32[768, 1, 3, 3]", primals_206: "f32[768]", primals_207: "f32[768]", primals_208: "f32[768]", primals_209: "f32[3072, 768]", primals_210: "f32[3072]", primals_211: "f32[768, 3072]", primals_212: "f32[768]", primals_213: "f32[768]", primals_214: "f32[768]", primals_215: "f32[2304, 768]", primals_216: "f32[2304]", primals_217: "f32[768, 768]", primals_218: "f32[768]", primals_219: "f32[768]", primals_220: "f32[768]", primals_221: "f32[768, 1, 3, 3]", primals_222: "f32[768]", primals_223: "f32[768]", primals_224: "f32[768]", primals_225: "f32[768, 1, 3, 3]", primals_226: "f32[768]", primals_227: "f32[768]", primals_228: "f32[768]", primals_229: "f32[3072, 768]", primals_230: "f32[3072]", primals_231: "f32[768, 3072]", primals_232: "f32[768]", primals_233: "f32[768]", primals_234: "f32[768]", primals_235: "f32[2304, 768]", primals_236: "f32[2304]", primals_237: "f32[768, 768]", primals_238: "f32[768]", primals_239: "f32[768]", primals_240: "f32[768]", primals_241: "f32[768, 1, 3, 3]", primals_242: "f32[768]", primals_243: "f32[768]", primals_244: "f32[768]", primals_245: "f32[768, 1, 3, 3]", primals_246: "f32[768]", primals_247: "f32[768]", primals_248: "f32[768]", primals_249: "f32[3072, 768]", primals_250: "f32[3072]", primals_251: "f32[768, 3072]", primals_252: "f32[768]", primals_253: "f32[768]", primals_254: "f32[768]", primals_255: "f32[2304, 768]", primals_256: "f32[2304]", primals_257: "f32[768, 768]", primals_258: "f32[768]", primals_259: "f32[768]", primals_260: "f32[768]", primals_261: "f32[768, 1, 3, 3]", primals_262: "f32[768]", primals_263: "f32[768]", primals_264: "f32[768]", primals_265: "f32[768, 1, 3, 3]", primals_266: "f32[768]", primals_267: "f32[768]", primals_268: "f32[768]", primals_269: "f32[3072, 768]", primals_270: "f32[3072]", primals_271: "f32[768, 3072]", primals_272: "f32[768]", primals_273: "f32[768]", primals_274: "f32[768]", primals_275: "f32[2304, 768]", primals_276: "f32[2304]", primals_277: "f32[768, 768]", primals_278: "f32[768]", primals_279: "f32[768]", primals_280: "f32[768]", primals_281: "f32[768, 1, 3, 3]", primals_282: "f32[768]", primals_283: "f32[768]", primals_284: "f32[768]", primals_285: "f32[768, 1, 3, 3]", primals_286: "f32[768]", primals_287: "f32[768]", primals_288: "f32[768]", primals_289: "f32[3072, 768]", primals_290: "f32[3072]", primals_291: "f32[768, 3072]", primals_292: "f32[768]", primals_293: "f32[768]", primals_294: "f32[768]", primals_295: "f32[2304, 768]", primals_296: "f32[2304]", primals_297: "f32[768, 768]", primals_298: "f32[768]", primals_299: "f32[768]", primals_300: "f32[768]", primals_301: "f32[768, 1, 3, 3]", primals_302: "f32[768]", primals_303: "f32[768]", primals_304: "f32[768]", primals_305: "f32[768, 1, 3, 3]", primals_306: "f32[768]", primals_307: "f32[768]", primals_308: "f32[768]", primals_309: "f32[3072, 768]", primals_310: "f32[3072]", primals_311: "f32[768, 3072]", primals_312: "f32[768]", primals_313: "f32[768]", primals_314: "f32[768]", primals_315: "f32[2304, 768]", primals_316: "f32[2304]", primals_317: "f32[768, 768]", primals_318: "f32[768]", primals_319: "f32[768]", primals_320: "f32[768]", primals_321: "f32[768, 1, 3, 3]", primals_322: "f32[768]", primals_323: "f32[768]", primals_324: "f32[768]", primals_325: "f32[768, 1, 3, 3]", primals_326: "f32[768]", primals_327: "f32[768]", primals_328: "f32[768]", primals_329: "f32[3072, 768]", primals_330: "f32[3072]", primals_331: "f32[768, 3072]", primals_332: "f32[768]", primals_333: "f32[768]", primals_334: "f32[768]", primals_335: "f32[2304, 768]", primals_336: "f32[2304]", primals_337: "f32[768, 768]", primals_338: "f32[768]", primals_339: "f32[768]", primals_340: "f32[768]", primals_341: "f32[768, 1, 3, 3]", primals_342: "f32[768]", primals_343: "f32[768]", primals_344: "f32[768]", primals_345: "f32[768, 1, 3, 3]", primals_346: "f32[768]", primals_347: "f32[768]", primals_348: "f32[768]", primals_349: "f32[3072, 768]", primals_350: "f32[3072]", primals_351: "f32[768, 3072]", primals_352: "f32[768]", primals_353: "f32[768]", primals_354: "f32[768]", primals_355: "f32[2304, 768]", primals_356: "f32[2304]", primals_357: "f32[768, 768]", primals_358: "f32[768]", primals_359: "f32[768]", primals_360: "f32[768]", primals_361: "f32[768, 1, 3, 3]", primals_362: "f32[768]", primals_363: "f32[768]", primals_364: "f32[768]", primals_365: "f32[768, 1, 3, 3]", primals_366: "f32[768]", primals_367: "f32[768]", primals_368: "f32[768]", primals_369: "f32[3072, 768]", primals_370: "f32[3072]", primals_371: "f32[768, 3072]", primals_372: "f32[768]", primals_373: "f32[768]", primals_374: "f32[768]", primals_375: "f32[2304, 768]", primals_376: "f32[2304]", primals_377: "f32[768, 768]", primals_378: "f32[768]", primals_379: "f32[768]", primals_380: "f32[768]", primals_381: "f32[768, 1, 3, 3]", primals_382: "f32[768]", primals_383: "f32[768]", primals_384: "f32[768]", primals_385: "f32[768, 1, 3, 3]", primals_386: "f32[768]", primals_387: "f32[768]", primals_388: "f32[768]", primals_389: "f32[3072, 768]", primals_390: "f32[3072]", primals_391: "f32[768, 3072]", primals_392: "f32[768]", primals_393: "f32[768]", primals_394: "f32[768]", primals_395: "f32[2304, 768]", primals_396: "f32[2304]", primals_397: "f32[768, 768]", primals_398: "f32[768]", primals_399: "f32[768]", primals_400: "f32[768]", primals_401: "f32[768, 1, 3, 3]", primals_402: "f32[768]", primals_403: "f32[768]", primals_404: "f32[768]", primals_405: "f32[768, 1, 3, 3]", primals_406: "f32[768]", primals_407: "f32[768]", primals_408: "f32[768]", primals_409: "f32[3072, 768]", primals_410: "f32[3072]", primals_411: "f32[768, 3072]", primals_412: "f32[768]", primals_413: "f32[768]", primals_414: "f32[768]", primals_415: "f32[2304, 768]", primals_416: "f32[2304]", primals_417: "f32[768, 768]", primals_418: "f32[768]", primals_419: "f32[768]", primals_420: "f32[768]", primals_421: "f32[768, 1, 3, 3]", primals_422: "f32[768]", primals_423: "f32[768]", primals_424: "f32[768]", primals_425: "f32[768, 1, 3, 3]", primals_426: "f32[768]", primals_427: "f32[768]", primals_428: "f32[768]", primals_429: "f32[3072, 768]", primals_430: "f32[3072]", primals_431: "f32[768, 3072]", primals_432: "f32[768]", primals_433: "f32[768]", primals_434: "f32[768]", primals_435: "f32[2304, 768]", primals_436: "f32[2304]", primals_437: "f32[768, 768]", primals_438: "f32[768]", primals_439: "f32[768]", primals_440: "f32[768]", primals_441: "f32[768, 1, 3, 3]", primals_442: "f32[768]", primals_443: "f32[768]", primals_444: "f32[768]", primals_445: "f32[768, 1, 3, 3]", primals_446: "f32[768]", primals_447: "f32[768]", primals_448: "f32[768]", primals_449: "f32[3072, 768]", primals_450: "f32[3072]", primals_451: "f32[768, 3072]", primals_452: "f32[768]", primals_453: "f32[768]", primals_454: "f32[768]", primals_455: "f32[2304, 768]", primals_456: "f32[2304]", primals_457: "f32[768, 768]", primals_458: "f32[768]", primals_459: "f32[768]", primals_460: "f32[768]", primals_461: "f32[768, 1, 3, 3]", primals_462: "f32[768]", primals_463: "f32[768]", primals_464: "f32[768]", primals_465: "f32[768, 1, 3, 3]", primals_466: "f32[768]", primals_467: "f32[768]", primals_468: "f32[768]", primals_469: "f32[3072, 768]", primals_470: "f32[3072]", primals_471: "f32[768, 3072]", primals_472: "f32[768]", primals_473: "f32[768]", primals_474: "f32[768]", primals_475: "f32[2304, 768]", primals_476: "f32[2304]", primals_477: "f32[768, 768]", primals_478: "f32[768]", primals_479: "f32[768]", primals_480: "f32[768]", primals_481: "f32[768, 1, 3, 3]", primals_482: "f32[768]", primals_483: "f32[768]", primals_484: "f32[768]", primals_485: "f32[768, 1, 3, 3]", primals_486: "f32[768]", primals_487: "f32[768]", primals_488: "f32[768]", primals_489: "f32[3072, 768]", primals_490: "f32[3072]", primals_491: "f32[768, 3072]", primals_492: "f32[768]", primals_493: "f32[768]", primals_494: "f32[768]", primals_495: "f32[2304, 768]", primals_496: "f32[2304]", primals_497: "f32[768, 768]", primals_498: "f32[768]", primals_499: "f32[768]", primals_500: "f32[768]", primals_501: "f32[768, 1, 3, 3]", primals_502: "f32[768]", primals_503: "f32[768]", primals_504: "f32[768]", primals_505: "f32[768, 1, 3, 3]", primals_506: "f32[768]", primals_507: "f32[768]", primals_508: "f32[768]", primals_509: "f32[3072, 768]", primals_510: "f32[3072]", primals_511: "f32[768, 3072]", primals_512: "f32[768]", primals_513: "f32[768]", primals_514: "f32[768]", primals_515: "f32[2304, 768]", primals_516: "f32[2304]", primals_517: "f32[768, 768]", primals_518: "f32[768]", primals_519: "f32[768]", primals_520: "f32[768]", primals_521: "f32[768, 1, 3, 3]", primals_522: "f32[768]", primals_523: "f32[768]", primals_524: "f32[768]", primals_525: "f32[768, 1, 3, 3]", primals_526: "f32[768]", primals_527: "f32[768]", primals_528: "f32[768]", primals_529: "f32[3072, 768]", primals_530: "f32[3072]", primals_531: "f32[768, 3072]", primals_532: "f32[768]", primals_533: "f32[768]", primals_534: "f32[768]", primals_535: "f32[2304, 768]", primals_536: "f32[2304]", primals_537: "f32[768, 768]", primals_538: "f32[768]", primals_539: "f32[768]", primals_540: "f32[768]", primals_541: "f32[768, 1, 3, 3]", primals_542: "f32[768]", primals_543: "f32[768]", primals_544: "f32[768]", primals_545: "f32[768, 1, 3, 3]", primals_546: "f32[768]", primals_547: "f32[768]", primals_548: "f32[768]", primals_549: "f32[3072, 768]", primals_550: "f32[3072]", primals_551: "f32[768, 3072]", primals_552: "f32[768]", primals_553: "f32[768]", primals_554: "f32[768]", primals_555: "f32[2304, 768]", primals_556: "f32[2304]", primals_557: "f32[768, 768]", primals_558: "f32[768]", primals_559: "f32[768]", primals_560: "f32[768]", primals_561: "f32[768, 1, 3, 3]", primals_562: "f32[768]", primals_563: "f32[768]", primals_564: "f32[768]", primals_565: "f32[768, 1, 3, 3]", primals_566: "f32[768]", primals_567: "f32[768]", primals_568: "f32[768]", primals_569: "f32[3072, 768]", primals_570: "f32[3072]", primals_571: "f32[768, 3072]", primals_572: "f32[768]", primals_573: "f32[768]", primals_574: "f32[768]", primals_575: "f32[2304, 768]", primals_576: "f32[2304]", primals_577: "f32[768, 768]", primals_578: "f32[768]", primals_579: "f32[768]", primals_580: "f32[768]", primals_581: "f32[768, 1, 3, 3]", primals_582: "f32[768]", primals_583: "f32[768]", primals_584: "f32[768]", primals_585: "f32[768, 1, 3, 3]", primals_586: "f32[768]", primals_587: "f32[768]", primals_588: "f32[768]", primals_589: "f32[3072, 768]", primals_590: "f32[3072]", primals_591: "f32[768, 3072]", primals_592: "f32[768]", primals_593: "f32[768]", primals_594: "f32[768]", primals_595: "f32[768, 768]", primals_596: "f32[768]", primals_597: "f32[768, 768]", primals_598: "f32[768]", primals_599: "f32[768, 768]", primals_600: "f32[768]", primals_601: "f32[768, 768]", primals_602: "f32[768]", primals_603: "f32[768]", primals_604: "f32[768]", primals_605: "f32[3072, 768]", primals_606: "f32[3072]", primals_607: "f32[768, 3072]", primals_608: "f32[768]", primals_609: "f32[768]", primals_610: "f32[768]", primals_611: "f32[768, 768]", primals_612: "f32[768]", primals_613: "f32[768, 768]", primals_614: "f32[768]", primals_615: "f32[768, 768]", primals_616: "f32[768]", primals_617: "f32[768, 768]", primals_618: "f32[768]", primals_619: "f32[768]", primals_620: "f32[768]", primals_621: "f32[3072, 768]", primals_622: "f32[3072]", primals_623: "f32[768, 3072]", primals_624: "f32[768]", primals_625: "f32[768]", primals_626: "f32[768]", primals_627: "f32[1000, 768]", primals_628: "f32[1000]", primals_629: "f32[192]", primals_630: "f32[192]", primals_631: "i64[]", primals_632: "f32[384]", primals_633: "f32[384]", primals_634: "i64[]", primals_635: "f32[768]", primals_636: "f32[768]", primals_637: "i64[]", primals_638: "f32[768]", primals_639: "f32[768]", primals_640: "i64[]", primals_641: "f32[768]", primals_642: "f32[768]", primals_643: "i64[]", primals_644: "f32[768]", primals_645: "f32[768]", primals_646: "i64[]", primals_647: "f32[768]", primals_648: "f32[768]", primals_649: "i64[]", primals_650: "f32[768]", primals_651: "f32[768]", primals_652: "i64[]", primals_653: "f32[768]", primals_654: "f32[768]", primals_655: "i64[]", primals_656: "f32[768]", primals_657: "f32[768]", primals_658: "i64[]", primals_659: "f32[768]", primals_660: "f32[768]", primals_661: "i64[]", primals_662: "f32[768]", primals_663: "f32[768]", primals_664: "i64[]", primals_665: "f32[768]", primals_666: "f32[768]", primals_667: "i64[]", primals_668: "f32[768]", primals_669: "f32[768]", primals_670: "i64[]", primals_671: "f32[768]", primals_672: "f32[768]", primals_673: "i64[]", primals_674: "f32[768]", primals_675: "f32[768]", primals_676: "i64[]", primals_677: "f32[768]", primals_678: "f32[768]", primals_679: "i64[]", primals_680: "f32[768]", primals_681: "f32[768]", primals_682: "i64[]", primals_683: "f32[768]", primals_684: "f32[768]", primals_685: "i64[]", primals_686: "f32[768]", primals_687: "f32[768]", primals_688: "i64[]", primals_689: "f32[768]", primals_690: "f32[768]", primals_691: "i64[]", primals_692: "f32[768]", primals_693: "f32[768]", primals_694: "i64[]", primals_695: "f32[768]", primals_696: "f32[768]", primals_697: "i64[]", primals_698: "f32[768]", primals_699: "f32[768]", primals_700: "i64[]", primals_701: "f32[768]", primals_702: "f32[768]", primals_703: "i64[]", primals_704: "f32[768]", primals_705: "f32[768]", primals_706: "i64[]", primals_707: "f32[768]", primals_708: "f32[768]", primals_709: "i64[]", primals_710: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:107, code: x = self.proj(x)
    convolution: "f32[8, 192, 112, 112]" = torch.ops.aten.convolution.default(primals_710, primals_102, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_631, 1)
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 192, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 192, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 192, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[192]" = torch.ops.aten.mul.Tensor(primals_629, 0.9)
    add_2: "f32[192]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[192]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[192]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[192]" = torch.ops.aten.mul.Tensor(primals_630, 0.9)
    add_3: "f32[192]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
    unsqueeze_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
    unsqueeze_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 192, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    mul_7: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, 0.5)
    mul_8: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, 0.7071067811865476)
    erf: "f32[8, 192, 112, 112]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_5: "f32[8, 192, 112, 112]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_9: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, add_5);  mul_7 = None
    convolution_1: "f32[8, 384, 56, 56]" = torch.ops.aten.convolution.default(mul_9, primals_105, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_6: "i64[]" = torch.ops.aten.add.Tensor(primals_634, 1)
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 384, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 384, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_7: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_1: "f32[8, 384, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_10: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_11: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_12: "f32[384]" = torch.ops.aten.mul.Tensor(primals_632, 0.9)
    add_8: "f32[384]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
    squeeze_5: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_13: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000398612827361);  squeeze_5 = None
    mul_14: "f32[384]" = torch.ops.aten.mul.Tensor(mul_13, 0.1);  mul_13 = None
    mul_15: "f32[384]" = torch.ops.aten.mul.Tensor(primals_633, 0.9)
    add_9: "f32[384]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None
    unsqueeze_4: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1)
    unsqueeze_5: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_16: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_5);  mul_10 = unsqueeze_5 = None
    unsqueeze_6: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1);  primals_107 = None
    unsqueeze_7: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_10: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(mul_16, unsqueeze_7);  mul_16 = unsqueeze_7 = None
    mul_17: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(add_10, 0.5)
    mul_18: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(add_10, 0.7071067811865476)
    erf_1: "f32[8, 384, 56, 56]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_11: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_19: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_17, add_11);  mul_17 = None
    convolution_2: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(mul_19, primals_108, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    add_12: "i64[]" = torch.ops.aten.add.Tensor(primals_637, 1)
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 768, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 768, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_13: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
    sub_2: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_20: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_21: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_22: "f32[768]" = torch.ops.aten.mul.Tensor(primals_635, 0.9)
    add_14: "f32[768]" = torch.ops.aten.add.Tensor(mul_21, mul_22);  mul_21 = mul_22 = None
    squeeze_8: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_23: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0001594642002871);  squeeze_8 = None
    mul_24: "f32[768]" = torch.ops.aten.mul.Tensor(mul_23, 0.1);  mul_23 = None
    mul_25: "f32[768]" = torch.ops.aten.mul.Tensor(primals_636, 0.9)
    add_15: "f32[768]" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None
    unsqueeze_8: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1)
    unsqueeze_9: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_26: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_20, unsqueeze_9);  mul_20 = unsqueeze_9 = None
    unsqueeze_10: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
    unsqueeze_11: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_16: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_11);  mul_26 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:109, code: x = x.flatten(2).transpose(1, 2)  # (B, N, C)
    view: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(add_16, [8, 768, 784]);  add_16 = None
    permute: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:51, code: y_embed = torch.arange(1, H+1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, 1, W)
    iota: "i64[28]" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    convert_element_type: "f32[28]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
    mul_27: "f32[28]" = torch.ops.aten.mul.Tensor(convert_element_type, 1);  convert_element_type = None
    add_17: "f32[28]" = torch.ops.aten.add.Tensor(mul_27, 1);  mul_27 = None
    unsqueeze_12: "f32[28, 1]" = torch.ops.aten.unsqueeze.default(add_17, 1)
    repeat: "f32[1, 28, 28]" = torch.ops.aten.repeat.default(unsqueeze_12, [1, 1, 28]);  unsqueeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:52, code: x_embed = torch.arange(1, W+1, dtype=torch.float32, device=device).repeat(1, H, 1)
    repeat_1: "f32[1, 28, 28]" = torch.ops.aten.repeat.default(add_17, [1, 28, 1]);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:53, code: y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
    full_default: "f32[1, 1, 28]" = torch.ops.aten.full.default([1, 1, 28], 28.000001907348633, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div: "f32[1, 28, 28]" = torch.ops.aten.div.Tensor(repeat, full_default);  repeat = full_default = None
    mul_29: "f32[1, 28, 28]" = torch.ops.aten.mul.Tensor(div, 6.283185307179586);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:54, code: x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
    full_default_1: "f32[1, 28, 1]" = torch.ops.aten.full.default([1, 28, 1], 28.000001907348633, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_1: "f32[1, 28, 28]" = torch.ops.aten.div.Tensor(repeat_1, full_default_1);  repeat_1 = full_default_1 = None
    mul_30: "f32[1, 28, 28]" = torch.ops.aten.mul.Tensor(div_1, 6.283185307179586);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:55, code: dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
    iota_2: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    convert_element_type_2: "f32[32]" = torch.ops.prims.convert_element_type.default(iota_2, torch.float32);  iota_2 = None
    mul_31: "f32[32]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
    add_21: "f32[32]" = torch.ops.aten.add.Tensor(mul_31, 0);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:56, code: dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.hidden_dim)
    div_2: "f32[32]" = torch.ops.aten.div.Tensor_mode(add_21, 2, rounding_mode = 'floor');  add_21 = None
    mul_32: "f32[32]" = torch.ops.aten.mul.Tensor(div_2, 2);  div_2 = None
    div_3: "f32[32]" = torch.ops.aten.div.Tensor(mul_32, 32);  mul_32 = None
    pow_1: "f32[32]" = torch.ops.aten.pow.Scalar(10000, div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:57, code: pos_x = x_embed[:, :, :, None] / dim_t
    unsqueeze_13: "f32[1, 28, 28, 1]" = torch.ops.aten.unsqueeze.default(mul_30, 3);  mul_30 = None
    div_4: "f32[1, 28, 28, 32]" = torch.ops.aten.div.Tensor(unsqueeze_13, pow_1);  unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:58, code: pos_y = y_embed[:, :, :, None] / dim_t
    unsqueeze_14: "f32[1, 28, 28, 1]" = torch.ops.aten.unsqueeze.default(mul_29, 3);  mul_29 = None
    div_5: "f32[1, 28, 28, 32]" = torch.ops.aten.div.Tensor(unsqueeze_14, pow_1);  unsqueeze_14 = pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:59, code: pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
    slice_16: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(div_4, 3, 0, 9223372036854775807, 2)
    sin: "f32[1, 28, 28, 16]" = torch.ops.aten.sin.default(slice_16);  slice_16 = None
    slice_20: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(div_4, 3, 1, 9223372036854775807, 2);  div_4 = None
    cos: "f32[1, 28, 28, 16]" = torch.ops.aten.cos.default(slice_20);  slice_20 = None
    unsqueeze_15: "f32[1, 28, 28, 16, 1]" = torch.ops.aten.unsqueeze.default(sin, 4);  sin = None
    unsqueeze_16: "f32[1, 28, 28, 16, 1]" = torch.ops.aten.unsqueeze.default(cos, 4);  cos = None
    cat: "f32[1, 28, 28, 16, 2]" = torch.ops.aten.cat.default([unsqueeze_15, unsqueeze_16], 4);  unsqueeze_15 = unsqueeze_16 = None
    view_1: "f32[1, 28, 28, 32]" = torch.ops.aten.reshape.default(cat, [1, 28, 28, 32]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:60, code: pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
    slice_24: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(div_5, 3, 0, 9223372036854775807, 2)
    sin_1: "f32[1, 28, 28, 16]" = torch.ops.aten.sin.default(slice_24);  slice_24 = None
    slice_28: "f32[1, 28, 28, 16]" = torch.ops.aten.slice.Tensor(div_5, 3, 1, 9223372036854775807, 2);  div_5 = None
    cos_1: "f32[1, 28, 28, 16]" = torch.ops.aten.cos.default(slice_28);  slice_28 = None
    unsqueeze_17: "f32[1, 28, 28, 16, 1]" = torch.ops.aten.unsqueeze.default(sin_1, 4);  sin_1 = None
    unsqueeze_18: "f32[1, 28, 28, 16, 1]" = torch.ops.aten.unsqueeze.default(cos_1, 4);  cos_1 = None
    cat_1: "f32[1, 28, 28, 16, 2]" = torch.ops.aten.cat.default([unsqueeze_17, unsqueeze_18], 4);  unsqueeze_17 = unsqueeze_18 = None
    view_2: "f32[1, 28, 28, 32]" = torch.ops.aten.reshape.default(cat_1, [1, 28, 28, 32]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:61, code: pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    cat_2: "f32[1, 28, 28, 64]" = torch.ops.aten.cat.default([view_2, view_1], 3);  view_2 = view_1 = None
    permute_1: "f32[1, 64, 28, 28]" = torch.ops.aten.permute.default(cat_2, [0, 3, 1, 2]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:62, code: pos = self.token_projection(pos)
    convolution_3: "f32[1, 768, 28, 28]" = torch.ops.aten.convolution.default(permute_1, primals_111, primals_112, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:63, code: return pos.repeat(B, 1, 1, 1)  # (B, C, H, W)
    repeat_2: "f32[8, 768, 28, 28]" = torch.ops.aten.repeat.default(convolution_3, [8, 1, 1, 1]);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:437, code: pos_encoding = self.pos_embed(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
    view_3: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(repeat_2, [8, -1, 784]);  repeat_2 = None
    permute_2: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:438, code: x = x + pos_encoding
    add_22: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(permute, permute_2);  permute = permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_1: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_22, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_1, [2], correction = 0, keepdim = True)
    getitem_6: "f32[8, 784, 1]" = var_mean_3[0]
    getitem_7: "f32[8, 784, 1]" = var_mean_3[1];  var_mean_3 = None
    add_23: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-06);  getitem_6 = None
    rsqrt_3: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_3: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_1, getitem_7);  clone_1 = getitem_7 = None
    mul_33: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_34: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_33, primals_113)
    add_24: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_114);  mul_34 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_4: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_24, [6272, 768]);  add_24 = None
    permute_3: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_116, view_4, permute_3);  primals_116 = None
    view_5: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm, [8, 784, 2304]);  addmm = None
    view_6: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_5, [8, 784, 3, 16, 48]);  view_5 = None
    permute_4: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_6, [2, 0, 3, 4, 1]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind = torch.ops.aten.unbind.int(permute_4);  permute_4 = None
    getitem_8: "f32[8, 16, 48, 784]" = unbind[0]
    getitem_9: "f32[8, 16, 48, 784]" = unbind[1]
    getitem_10: "f32[8, 16, 48, 784]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_2: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_8, 2.0)
    sum_1: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_2, [-1], True);  pow_2 = None
    pow_3: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_1, 0.5);  sum_1 = None
    clamp_min: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_3, 1e-12)
    expand: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min, [8, 16, 48, 784]);  clamp_min = None
    div_6: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_8, expand);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_4: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_9, 2.0)
    sum_2: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_4, [-1], True);  pow_4 = None
    pow_5: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_2, 0.5);  sum_2 = None
    clamp_min_1: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_5, 1e-12)
    expand_1: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_1, [8, 16, 48, 784]);  clamp_min_1 = None
    div_7: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_9, expand_1);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_5: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_7, [0, 1, 3, 2]);  div_7 = None
    expand_2: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_6, [8, 16, 48, 784]);  div_6 = None
    clone_2: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_7: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_2, [128, 48, 784]);  clone_2 = None
    expand_3: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_5, [8, 16, 784, 48]);  permute_5 = None
    clone_3: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_8: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_3, [128, 784, 48]);  clone_3 = None
    bmm: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_7, view_8)
    view_9: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm, [8, 16, 48, 48])
    mul_35: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_9, primals_2);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_35, [-1], True)
    sub_4: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_35, amax);  mul_35 = amax = None
    exp: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_3: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_8: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp, sum_3);  exp = sum_3 = None
    alias_2: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_4: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_4: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_4, [8, 16, 48, 48]);  clone_4 = None
    view_10: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_4, [128, 48, 48]);  expand_4 = None
    expand_5: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_10, [8, 16, 48, 784]);  getitem_10 = None
    clone_5: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_11: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_5, [128, 48, 784]);  clone_5 = None
    bmm_1: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_10, view_11)
    view_12: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_1, [8, 16, 48, 784]);  bmm_1 = None
    permute_6: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_12, [0, 3, 1, 2]);  view_12 = None
    view_13: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_6, [8, 784, 768]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_7: "f32[768, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    clone_6: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_13, memory_format = torch.contiguous_format);  view_13 = None
    view_14: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_6, [6272, 768]);  clone_6 = None
    mm: "f32[6272, 768]" = torch.ops.aten.mm.default(view_14, permute_7)
    view_15: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm, [8, 784, 768])
    add_25: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_15, primals_118);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_36: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_1, add_25);  add_25 = None
    add_26: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_22, mul_36);  add_22 = mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_8: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_26, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_8, [2], correction = 0, keepdim = True)
    getitem_11: "f32[8, 784, 1]" = var_mean_4[0]
    getitem_12: "f32[8, 784, 1]" = var_mean_4[1];  var_mean_4 = None
    add_27: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-06);  getitem_11 = None
    rsqrt_4: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_5: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_8, getitem_12);  clone_8 = getitem_12 = None
    mul_37: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = None
    mul_38: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_119)
    add_28: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_120);  mul_38 = primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_8: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_28, [0, 2, 1]);  add_28 = None
    view_16: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_8, [8, 768, 28, 28]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_4: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_16, primals_121, primals_122, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_39: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_4, 0.5)
    mul_40: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_4, 0.7071067811865476)
    erf_2: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_29: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_41: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_39, add_29);  mul_39 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_640, 1)
    var_mean_5 = torch.ops.aten.var_mean.correction(mul_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_13: "f32[1, 768, 1, 1]" = var_mean_5[0]
    getitem_14: "f32[1, 768, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_31: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_13, 1e-05)
    rsqrt_5: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_6: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_41, getitem_14);  mul_41 = None
    mul_42: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = None
    squeeze_9: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    squeeze_10: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_43: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_44: "f32[768]" = torch.ops.aten.mul.Tensor(primals_638, 0.9)
    add_32: "f32[768]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    squeeze_11: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    mul_45: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0001594642002871);  squeeze_11 = None
    mul_46: "f32[768]" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
    mul_47: "f32[768]" = torch.ops.aten.mul.Tensor(primals_639, 0.9)
    add_33: "f32[768]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    unsqueeze_19: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1)
    unsqueeze_20: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_19, -1);  unsqueeze_19 = None
    mul_48: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_20);  mul_42 = unsqueeze_20 = None
    unsqueeze_21: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1);  primals_124 = None
    unsqueeze_22: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, -1);  unsqueeze_21 = None
    add_34: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_22);  mul_48 = unsqueeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_5: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_34, primals_125, primals_126, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_17: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_5, [8, 768, 784])
    permute_9: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_49: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_3, permute_9);  permute_9 = None
    add_35: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_26, mul_49);  add_26 = mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_9: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_35, memory_format = torch.contiguous_format)
    var_mean_6 = torch.ops.aten.var_mean.correction(clone_9, [2], correction = 0, keepdim = True)
    getitem_15: "f32[8, 784, 1]" = var_mean_6[0]
    getitem_16: "f32[8, 784, 1]" = var_mean_6[1];  var_mean_6 = None
    add_36: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_15, 1e-06);  getitem_15 = None
    rsqrt_6: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_7: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_9, getitem_16);  clone_9 = getitem_16 = None
    mul_50: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = None
    mul_51: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_50, primals_127)
    add_37: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_51, primals_128);  mul_51 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_18: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_37, [6272, 768]);  add_37 = None
    permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_1: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_130, view_18, permute_10);  primals_130 = None
    view_19: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_1, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_52: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_53: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
    erf_3: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_53);  mul_53 = None
    add_38: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_54: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_38);  mul_52 = add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_20: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_54, [6272, 3072]);  mul_54 = None
    permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_2: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_132, view_20, permute_11);  primals_132 = None
    view_21: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_2, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_55: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_4, view_21);  view_21 = None
    add_39: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_35, mul_55);  add_35 = mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_12: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_39, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_12, [2], correction = 0, keepdim = True)
    getitem_17: "f32[8, 784, 1]" = var_mean_7[0]
    getitem_18: "f32[8, 784, 1]" = var_mean_7[1];  var_mean_7 = None
    add_40: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_17, 1e-06);  getitem_17 = None
    rsqrt_7: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_8: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_12, getitem_18);  clone_12 = getitem_18 = None
    mul_56: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = None
    mul_57: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_56, primals_133)
    add_41: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_57, primals_134);  mul_57 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_22: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_41, [6272, 768]);  add_41 = None
    permute_12: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_3: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_136, view_22, permute_12);  primals_136 = None
    view_23: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_3, [8, 784, 2304]);  addmm_3 = None
    view_24: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_23, [8, 784, 3, 16, 48]);  view_23 = None
    permute_13: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_24, [2, 0, 3, 4, 1]);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_1 = torch.ops.aten.unbind.int(permute_13);  permute_13 = None
    getitem_19: "f32[8, 16, 48, 784]" = unbind_1[0]
    getitem_20: "f32[8, 16, 48, 784]" = unbind_1[1]
    getitem_21: "f32[8, 16, 48, 784]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_6: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_19, 2.0)
    sum_4: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_6, [-1], True);  pow_6 = None
    pow_7: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_4, 0.5);  sum_4 = None
    clamp_min_2: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_7, 1e-12)
    expand_6: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_2, [8, 16, 48, 784]);  clamp_min_2 = None
    div_9: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_19, expand_6);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_8: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_20, 2.0)
    sum_5: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_8, [-1], True);  pow_8 = None
    pow_9: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_5, 0.5);  sum_5 = None
    clamp_min_3: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_9, 1e-12)
    expand_7: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_3, [8, 16, 48, 784]);  clamp_min_3 = None
    div_10: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_20, expand_7);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_14: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_10, [0, 1, 3, 2]);  div_10 = None
    expand_8: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_9, [8, 16, 48, 784]);  div_9 = None
    clone_13: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_25: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_13, [128, 48, 784]);  clone_13 = None
    expand_9: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_14, [8, 16, 784, 48]);  permute_14 = None
    clone_14: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_26: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_14, [128, 784, 48]);  clone_14 = None
    bmm_2: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_25, view_26)
    view_27: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_2, [8, 16, 48, 48])
    mul_58: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_27, primals_6);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_58, [-1], True)
    sub_9: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_58, amax_1);  mul_58 = amax_1 = None
    exp_1: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_6: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_11: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_1, sum_6);  exp_1 = sum_6 = None
    alias_5: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_15: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_10: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_15, [8, 16, 48, 48]);  clone_15 = None
    view_28: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_10, [128, 48, 48]);  expand_10 = None
    expand_11: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_21, [8, 16, 48, 784]);  getitem_21 = None
    clone_16: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_29: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_16, [128, 48, 784]);  clone_16 = None
    bmm_3: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_28, view_29)
    view_30: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_3, [8, 16, 48, 784]);  bmm_3 = None
    permute_15: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_30, [0, 3, 1, 2]);  view_30 = None
    view_31: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_15, [8, 784, 768]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_16: "f32[768, 768]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    clone_17: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_31, memory_format = torch.contiguous_format);  view_31 = None
    view_32: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_17, [6272, 768]);  clone_17 = None
    mm_1: "f32[6272, 768]" = torch.ops.aten.mm.default(view_32, permute_16)
    view_33: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_1, [8, 784, 768])
    add_42: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_33, primals_138);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_59: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_5, add_42);  add_42 = None
    add_43: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_39, mul_59);  add_39 = mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_19: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_43, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_19, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 784, 1]" = var_mean_8[0]
    getitem_23: "f32[8, 784, 1]" = var_mean_8[1];  var_mean_8 = None
    add_44: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_8: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_10: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_19, getitem_23);  clone_19 = getitem_23 = None
    mul_60: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_8);  sub_10 = None
    mul_61: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_60, primals_139)
    add_45: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_140);  mul_61 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_17: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_45, [0, 2, 1]);  add_45 = None
    view_34: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_17, [8, 768, 28, 28]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_6: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_34, primals_141, primals_142, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_62: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.5)
    mul_63: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.7071067811865476)
    erf_4: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_46: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_64: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_62, add_46);  mul_62 = add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_643, 1)
    var_mean_9 = torch.ops.aten.var_mean.correction(mul_64, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 768, 1, 1]" = var_mean_9[0]
    getitem_25: "f32[1, 768, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_48: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_9: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_11: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_64, getitem_25);  mul_64 = None
    mul_65: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_9);  sub_11 = None
    squeeze_12: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_13: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_66: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_67: "f32[768]" = torch.ops.aten.mul.Tensor(primals_641, 0.9)
    add_49: "f32[768]" = torch.ops.aten.add.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
    squeeze_14: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_68: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0001594642002871);  squeeze_14 = None
    mul_69: "f32[768]" = torch.ops.aten.mul.Tensor(mul_68, 0.1);  mul_68 = None
    mul_70: "f32[768]" = torch.ops.aten.mul.Tensor(primals_642, 0.9)
    add_50: "f32[768]" = torch.ops.aten.add.Tensor(mul_69, mul_70);  mul_69 = mul_70 = None
    unsqueeze_23: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_24: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, -1);  unsqueeze_23 = None
    mul_71: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_65, unsqueeze_24);  mul_65 = unsqueeze_24 = None
    unsqueeze_25: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_26: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, -1);  unsqueeze_25 = None
    add_51: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_26);  mul_71 = unsqueeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_7: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_51, primals_145, primals_146, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_35: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_7, [8, 768, 784])
    permute_18: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_72: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_7, permute_18);  permute_18 = None
    add_52: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_43, mul_72);  add_43 = mul_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_20: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_52, memory_format = torch.contiguous_format)
    var_mean_10 = torch.ops.aten.var_mean.correction(clone_20, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 784, 1]" = var_mean_10[0]
    getitem_27: "f32[8, 784, 1]" = var_mean_10[1];  var_mean_10 = None
    add_53: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_10: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_12: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_20, getitem_27);  clone_20 = getitem_27 = None
    mul_73: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_10);  sub_12 = None
    mul_74: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_73, primals_147)
    add_54: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_74, primals_148);  mul_74 = primals_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_36: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_54, [6272, 768]);  add_54 = None
    permute_19: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    addmm_4: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_150, view_36, permute_19);  primals_150 = None
    view_37: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_4, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_75: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_37, 0.5)
    mul_76: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_37, 0.7071067811865476);  view_37 = None
    erf_5: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_55: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_77: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_75, add_55);  mul_75 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_38: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_77, [6272, 3072]);  mul_77 = None
    permute_20: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_5: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_152, view_38, permute_20);  primals_152 = None
    view_39: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_5, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_78: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_8, view_39);  view_39 = None
    add_56: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_52, mul_78);  add_52 = mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_23: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_56, memory_format = torch.contiguous_format)
    var_mean_11 = torch.ops.aten.var_mean.correction(clone_23, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 784, 1]" = var_mean_11[0]
    getitem_29: "f32[8, 784, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_11: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_13: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_23, getitem_29);  clone_23 = getitem_29 = None
    mul_79: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_11);  sub_13 = None
    mul_80: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_79, primals_153)
    add_58: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_80, primals_154);  mul_80 = primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_40: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_58, [6272, 768]);  add_58 = None
    permute_21: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_6: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_156, view_40, permute_21);  primals_156 = None
    view_41: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_6, [8, 784, 2304]);  addmm_6 = None
    view_42: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_41, [8, 784, 3, 16, 48]);  view_41 = None
    permute_22: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_42, [2, 0, 3, 4, 1]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_2 = torch.ops.aten.unbind.int(permute_22);  permute_22 = None
    getitem_30: "f32[8, 16, 48, 784]" = unbind_2[0]
    getitem_31: "f32[8, 16, 48, 784]" = unbind_2[1]
    getitem_32: "f32[8, 16, 48, 784]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_10: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_30, 2.0)
    sum_7: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_10, [-1], True);  pow_10 = None
    pow_11: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_7, 0.5);  sum_7 = None
    clamp_min_4: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_11, 1e-12)
    expand_12: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_4, [8, 16, 48, 784]);  clamp_min_4 = None
    div_12: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_30, expand_12);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_12: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_31, 2.0)
    sum_8: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_12, [-1], True);  pow_12 = None
    pow_13: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_8, 0.5);  sum_8 = None
    clamp_min_5: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_13, 1e-12)
    expand_13: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_5, [8, 16, 48, 784]);  clamp_min_5 = None
    div_13: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_31, expand_13);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_23: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_13, [0, 1, 3, 2]);  div_13 = None
    expand_14: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_12, [8, 16, 48, 784]);  div_12 = None
    clone_24: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_43: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_24, [128, 48, 784]);  clone_24 = None
    expand_15: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_23, [8, 16, 784, 48]);  permute_23 = None
    clone_25: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_44: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_25, [128, 784, 48]);  clone_25 = None
    bmm_4: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_43, view_44)
    view_45: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_4, [8, 16, 48, 48])
    mul_81: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_45, primals_10);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_81, [-1], True)
    sub_14: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_81, amax_2);  mul_81 = amax_2 = None
    exp_2: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_9: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_14: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_2, sum_9);  exp_2 = sum_9 = None
    alias_8: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_26: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_16: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_26, [8, 16, 48, 48]);  clone_26 = None
    view_46: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_16, [128, 48, 48]);  expand_16 = None
    expand_17: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_32, [8, 16, 48, 784]);  getitem_32 = None
    clone_27: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_47: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_27, [128, 48, 784]);  clone_27 = None
    bmm_5: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_46, view_47)
    view_48: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_5, [8, 16, 48, 784]);  bmm_5 = None
    permute_24: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_48, [0, 3, 1, 2]);  view_48 = None
    view_49: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_24, [8, 784, 768]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    clone_28: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_49, memory_format = torch.contiguous_format);  view_49 = None
    view_50: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_28, [6272, 768]);  clone_28 = None
    mm_2: "f32[6272, 768]" = torch.ops.aten.mm.default(view_50, permute_25)
    view_51: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_2, [8, 784, 768])
    add_59: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_51, primals_158);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_82: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_9, add_59);  add_59 = None
    add_60: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_56, mul_82);  add_56 = mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_30: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_60, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_30, [2], correction = 0, keepdim = True)
    getitem_33: "f32[8, 784, 1]" = var_mean_12[0]
    getitem_34: "f32[8, 784, 1]" = var_mean_12[1];  var_mean_12 = None
    add_61: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
    rsqrt_12: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_15: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_30, getitem_34);  clone_30 = getitem_34 = None
    mul_83: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_12);  sub_15 = None
    mul_84: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_83, primals_159)
    add_62: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_84, primals_160);  mul_84 = primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_26: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_62, [0, 2, 1]);  add_62 = None
    view_52: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_26, [8, 768, 28, 28]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_8: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_52, primals_161, primals_162, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_85: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, 0.5)
    mul_86: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, 0.7071067811865476)
    erf_6: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_86);  mul_86 = None
    add_63: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_87: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_85, add_63);  mul_85 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_64: "i64[]" = torch.ops.aten.add.Tensor(primals_646, 1)
    var_mean_13 = torch.ops.aten.var_mean.correction(mul_87, [0, 2, 3], correction = 0, keepdim = True)
    getitem_35: "f32[1, 768, 1, 1]" = var_mean_13[0]
    getitem_36: "f32[1, 768, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_65: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_35, 1e-05)
    rsqrt_13: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_16: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_87, getitem_36);  mul_87 = None
    mul_88: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_13);  sub_16 = None
    squeeze_15: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    squeeze_16: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_89: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_90: "f32[768]" = torch.ops.aten.mul.Tensor(primals_644, 0.9)
    add_66: "f32[768]" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
    squeeze_17: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    mul_91: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0001594642002871);  squeeze_17 = None
    mul_92: "f32[768]" = torch.ops.aten.mul.Tensor(mul_91, 0.1);  mul_91 = None
    mul_93: "f32[768]" = torch.ops.aten.mul.Tensor(primals_645, 0.9)
    add_67: "f32[768]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    unsqueeze_27: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1)
    unsqueeze_28: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_27, -1);  unsqueeze_27 = None
    mul_94: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_28);  mul_88 = unsqueeze_28 = None
    unsqueeze_29: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_164, -1);  primals_164 = None
    unsqueeze_30: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_29, -1);  unsqueeze_29 = None
    add_68: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_30);  mul_94 = unsqueeze_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_9: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_68, primals_165, primals_166, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_53: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_9, [8, 768, 784])
    permute_27: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_95: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_11, permute_27);  permute_27 = None
    add_69: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_60, mul_95);  add_60 = mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_31: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_69, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_31, [2], correction = 0, keepdim = True)
    getitem_37: "f32[8, 784, 1]" = var_mean_14[0]
    getitem_38: "f32[8, 784, 1]" = var_mean_14[1];  var_mean_14 = None
    add_70: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_37, 1e-06);  getitem_37 = None
    rsqrt_14: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_17: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_31, getitem_38);  clone_31 = getitem_38 = None
    mul_96: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_14);  sub_17 = None
    mul_97: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_96, primals_167)
    add_71: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_97, primals_168);  mul_97 = primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_54: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_71, [6272, 768]);  add_71 = None
    permute_28: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    addmm_7: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_170, view_54, permute_28);  primals_170 = None
    view_55: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_7, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_98: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_55, 0.5)
    mul_99: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_55, 0.7071067811865476);  view_55 = None
    erf_7: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_72: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_100: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_98, add_72);  mul_98 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_56: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_100, [6272, 3072]);  mul_100 = None
    permute_29: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm_8: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_172, view_56, permute_29);  primals_172 = None
    view_57: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_8, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_101: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_12, view_57);  view_57 = None
    add_73: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_69, mul_101);  add_69 = mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_34: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_73, memory_format = torch.contiguous_format)
    var_mean_15 = torch.ops.aten.var_mean.correction(clone_34, [2], correction = 0, keepdim = True)
    getitem_39: "f32[8, 784, 1]" = var_mean_15[0]
    getitem_40: "f32[8, 784, 1]" = var_mean_15[1];  var_mean_15 = None
    add_74: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_39, 1e-06);  getitem_39 = None
    rsqrt_15: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_18: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_34, getitem_40);  clone_34 = getitem_40 = None
    mul_102: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_15);  sub_18 = None
    mul_103: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_102, primals_173)
    add_75: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_103, primals_174);  mul_103 = primals_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_58: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_75, [6272, 768]);  add_75 = None
    permute_30: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_9: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_176, view_58, permute_30);  primals_176 = None
    view_59: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_9, [8, 784, 2304]);  addmm_9 = None
    view_60: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_59, [8, 784, 3, 16, 48]);  view_59 = None
    permute_31: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_60, [2, 0, 3, 4, 1]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_3 = torch.ops.aten.unbind.int(permute_31);  permute_31 = None
    getitem_41: "f32[8, 16, 48, 784]" = unbind_3[0]
    getitem_42: "f32[8, 16, 48, 784]" = unbind_3[1]
    getitem_43: "f32[8, 16, 48, 784]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_14: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_41, 2.0)
    sum_10: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_14, [-1], True);  pow_14 = None
    pow_15: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_10, 0.5);  sum_10 = None
    clamp_min_6: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_15, 1e-12)
    expand_18: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_6, [8, 16, 48, 784]);  clamp_min_6 = None
    div_15: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_41, expand_18);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_16: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_42, 2.0)
    sum_11: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_16, [-1], True);  pow_16 = None
    pow_17: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_11, 0.5);  sum_11 = None
    clamp_min_7: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_17, 1e-12)
    expand_19: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_7, [8, 16, 48, 784]);  clamp_min_7 = None
    div_16: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_42, expand_19);  expand_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_32: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_16, [0, 1, 3, 2]);  div_16 = None
    expand_20: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_15, [8, 16, 48, 784]);  div_15 = None
    clone_35: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_61: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_35, [128, 48, 784]);  clone_35 = None
    expand_21: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_32, [8, 16, 784, 48]);  permute_32 = None
    clone_36: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_62: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_36, [128, 784, 48]);  clone_36 = None
    bmm_6: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_61, view_62)
    view_63: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_6, [8, 16, 48, 48])
    mul_104: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_63, primals_14);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_3: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_104, [-1], True)
    sub_19: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_104, amax_3);  mul_104 = amax_3 = None
    exp_3: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_12: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_17: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_3, sum_12);  exp_3 = sum_12 = None
    alias_11: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_37: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_22: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_37, [8, 16, 48, 48]);  clone_37 = None
    view_64: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_22, [128, 48, 48]);  expand_22 = None
    expand_23: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_43, [8, 16, 48, 784]);  getitem_43 = None
    clone_38: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_65: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_38, [128, 48, 784]);  clone_38 = None
    bmm_7: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_64, view_65)
    view_66: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_7, [8, 16, 48, 784]);  bmm_7 = None
    permute_33: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_66, [0, 3, 1, 2]);  view_66 = None
    view_67: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_33, [8, 784, 768]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    clone_39: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_67, memory_format = torch.contiguous_format);  view_67 = None
    view_68: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_39, [6272, 768]);  clone_39 = None
    mm_3: "f32[6272, 768]" = torch.ops.aten.mm.default(view_68, permute_34)
    view_69: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_3, [8, 784, 768])
    add_76: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_69, primals_178);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_105: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_13, add_76);  add_76 = None
    add_77: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_73, mul_105);  add_73 = mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_41: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_77, memory_format = torch.contiguous_format)
    var_mean_16 = torch.ops.aten.var_mean.correction(clone_41, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 784, 1]" = var_mean_16[0]
    getitem_45: "f32[8, 784, 1]" = var_mean_16[1];  var_mean_16 = None
    add_78: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_16: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_20: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_41, getitem_45);  clone_41 = getitem_45 = None
    mul_106: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_16);  sub_20 = None
    mul_107: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_106, primals_179)
    add_79: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_107, primals_180);  mul_107 = primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_35: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_79, [0, 2, 1]);  add_79 = None
    view_70: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_35, [8, 768, 28, 28]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_10: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_70, primals_181, primals_182, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_108: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_10, 0.5)
    mul_109: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_10, 0.7071067811865476)
    erf_8: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_109);  mul_109 = None
    add_80: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_110: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_108, add_80);  mul_108 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_81: "i64[]" = torch.ops.aten.add.Tensor(primals_649, 1)
    var_mean_17 = torch.ops.aten.var_mean.correction(mul_110, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 768, 1, 1]" = var_mean_17[0]
    getitem_47: "f32[1, 768, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_82: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_17: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_21: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_110, getitem_47);  mul_110 = None
    mul_111: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_17);  sub_21 = None
    squeeze_18: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_19: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_112: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_113: "f32[768]" = torch.ops.aten.mul.Tensor(primals_647, 0.9)
    add_83: "f32[768]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    squeeze_20: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_114: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0001594642002871);  squeeze_20 = None
    mul_115: "f32[768]" = torch.ops.aten.mul.Tensor(mul_114, 0.1);  mul_114 = None
    mul_116: "f32[768]" = torch.ops.aten.mul.Tensor(primals_648, 0.9)
    add_84: "f32[768]" = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
    unsqueeze_31: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_183, -1)
    unsqueeze_32: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_31, -1);  unsqueeze_31 = None
    mul_117: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_111, unsqueeze_32);  mul_111 = unsqueeze_32 = None
    unsqueeze_33: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_184, -1);  primals_184 = None
    unsqueeze_34: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, -1);  unsqueeze_33 = None
    add_85: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_117, unsqueeze_34);  mul_117 = unsqueeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_11: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_85, primals_185, primals_186, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_71: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_11, [8, 768, 784])
    permute_36: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_118: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_15, permute_36);  permute_36 = None
    add_86: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_77, mul_118);  add_77 = mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_42: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_86, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 784, 1]" = var_mean_18[0]
    getitem_49: "f32[8, 784, 1]" = var_mean_18[1];  var_mean_18 = None
    add_87: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_18: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_22: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_42, getitem_49);  clone_42 = getitem_49 = None
    mul_119: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_18);  sub_22 = None
    mul_120: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_119, primals_187)
    add_88: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_120, primals_188);  mul_120 = primals_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_72: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_88, [6272, 768]);  add_88 = None
    permute_37: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    addmm_10: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_190, view_72, permute_37);  primals_190 = None
    view_73: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_121: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_73, 0.5)
    mul_122: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_73, 0.7071067811865476);  view_73 = None
    erf_9: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_122);  mul_122 = None
    add_89: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_123: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_121, add_89);  mul_121 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_74: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_123, [6272, 3072]);  mul_123 = None
    permute_38: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
    addmm_11: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_192, view_74, permute_38);  primals_192 = None
    view_75: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_11, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_124: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_16, view_75);  view_75 = None
    add_90: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_86, mul_124);  add_86 = mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_45: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_90, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_45, [2], correction = 0, keepdim = True)
    getitem_50: "f32[8, 784, 1]" = var_mean_19[0]
    getitem_51: "f32[8, 784, 1]" = var_mean_19[1];  var_mean_19 = None
    add_91: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
    rsqrt_19: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_23: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_45, getitem_51);  clone_45 = getitem_51 = None
    mul_125: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_19);  sub_23 = None
    mul_126: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_125, primals_193)
    add_92: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_126, primals_194);  mul_126 = primals_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_76: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_92, [6272, 768]);  add_92 = None
    permute_39: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
    addmm_12: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_196, view_76, permute_39);  primals_196 = None
    view_77: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_12, [8, 784, 2304]);  addmm_12 = None
    view_78: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_77, [8, 784, 3, 16, 48]);  view_77 = None
    permute_40: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_78, [2, 0, 3, 4, 1]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_4 = torch.ops.aten.unbind.int(permute_40);  permute_40 = None
    getitem_52: "f32[8, 16, 48, 784]" = unbind_4[0]
    getitem_53: "f32[8, 16, 48, 784]" = unbind_4[1]
    getitem_54: "f32[8, 16, 48, 784]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_18: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_52, 2.0)
    sum_13: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_18, [-1], True);  pow_18 = None
    pow_19: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_13, 0.5);  sum_13 = None
    clamp_min_8: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_19, 1e-12)
    expand_24: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_8, [8, 16, 48, 784]);  clamp_min_8 = None
    div_18: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_52, expand_24);  expand_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_20: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_53, 2.0)
    sum_14: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_20, [-1], True);  pow_20 = None
    pow_21: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_14, 0.5);  sum_14 = None
    clamp_min_9: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_21, 1e-12)
    expand_25: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_9, [8, 16, 48, 784]);  clamp_min_9 = None
    div_19: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_53, expand_25);  expand_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_41: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_19, [0, 1, 3, 2]);  div_19 = None
    expand_26: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_18, [8, 16, 48, 784]);  div_18 = None
    clone_46: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_79: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_46, [128, 48, 784]);  clone_46 = None
    expand_27: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_41, [8, 16, 784, 48]);  permute_41 = None
    clone_47: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_80: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_47, [128, 784, 48]);  clone_47 = None
    bmm_8: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_79, view_80)
    view_81: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_8, [8, 16, 48, 48])
    mul_127: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_81, primals_18);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_127, [-1], True)
    sub_24: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_127, amax_4);  mul_127 = amax_4 = None
    exp_4: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_15: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_20: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_4, sum_15);  exp_4 = sum_15 = None
    alias_14: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_48: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_28: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_48, [8, 16, 48, 48]);  clone_48 = None
    view_82: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_28, [128, 48, 48]);  expand_28 = None
    expand_29: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_54, [8, 16, 48, 784]);  getitem_54 = None
    clone_49: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_83: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_49, [128, 48, 784]);  clone_49 = None
    bmm_9: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_82, view_83)
    view_84: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_9, [8, 16, 48, 784]);  bmm_9 = None
    permute_42: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_84, [0, 3, 1, 2]);  view_84 = None
    view_85: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_42, [8, 784, 768]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_43: "f32[768, 768]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
    clone_50: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_85, memory_format = torch.contiguous_format);  view_85 = None
    view_86: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_50, [6272, 768]);  clone_50 = None
    mm_4: "f32[6272, 768]" = torch.ops.aten.mm.default(view_86, permute_43)
    view_87: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_4, [8, 784, 768])
    add_93: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_87, primals_198);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_128: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_17, add_93);  add_93 = None
    add_94: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_90, mul_128);  add_90 = mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_52: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_94, memory_format = torch.contiguous_format)
    var_mean_20 = torch.ops.aten.var_mean.correction(clone_52, [2], correction = 0, keepdim = True)
    getitem_55: "f32[8, 784, 1]" = var_mean_20[0]
    getitem_56: "f32[8, 784, 1]" = var_mean_20[1];  var_mean_20 = None
    add_95: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-06);  getitem_55 = None
    rsqrt_20: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_25: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_52, getitem_56);  clone_52 = getitem_56 = None
    mul_129: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_20);  sub_25 = None
    mul_130: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_129, primals_199)
    add_96: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_130, primals_200);  mul_130 = primals_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_44: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_96, [0, 2, 1]);  add_96 = None
    view_88: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_44, [8, 768, 28, 28]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_12: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_88, primals_201, primals_202, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_131: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.5)
    mul_132: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.7071067811865476)
    erf_10: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
    add_97: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_133: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_131, add_97);  mul_131 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_98: "i64[]" = torch.ops.aten.add.Tensor(primals_652, 1)
    var_mean_21 = torch.ops.aten.var_mean.correction(mul_133, [0, 2, 3], correction = 0, keepdim = True)
    getitem_57: "f32[1, 768, 1, 1]" = var_mean_21[0]
    getitem_58: "f32[1, 768, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_99: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_57, 1e-05)
    rsqrt_21: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_26: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_133, getitem_58);  mul_133 = None
    mul_134: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_21);  sub_26 = None
    squeeze_21: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    squeeze_22: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_135: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_136: "f32[768]" = torch.ops.aten.mul.Tensor(primals_650, 0.9)
    add_100: "f32[768]" = torch.ops.aten.add.Tensor(mul_135, mul_136);  mul_135 = mul_136 = None
    squeeze_23: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    mul_137: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0001594642002871);  squeeze_23 = None
    mul_138: "f32[768]" = torch.ops.aten.mul.Tensor(mul_137, 0.1);  mul_137 = None
    mul_139: "f32[768]" = torch.ops.aten.mul.Tensor(primals_651, 0.9)
    add_101: "f32[768]" = torch.ops.aten.add.Tensor(mul_138, mul_139);  mul_138 = mul_139 = None
    unsqueeze_35: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_203, -1)
    unsqueeze_36: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_35, -1);  unsqueeze_35 = None
    mul_140: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_134, unsqueeze_36);  mul_134 = unsqueeze_36 = None
    unsqueeze_37: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
    unsqueeze_38: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, -1);  unsqueeze_37 = None
    add_102: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_38);  mul_140 = unsqueeze_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_13: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_102, primals_205, primals_206, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_89: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_13, [8, 768, 784])
    permute_45: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_89, [0, 2, 1]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_141: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_19, permute_45);  permute_45 = None
    add_103: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_94, mul_141);  add_94 = mul_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_53: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_103, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_53, [2], correction = 0, keepdim = True)
    getitem_59: "f32[8, 784, 1]" = var_mean_22[0]
    getitem_60: "f32[8, 784, 1]" = var_mean_22[1];  var_mean_22 = None
    add_104: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_59, 1e-06);  getitem_59 = None
    rsqrt_22: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    sub_27: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_53, getitem_60);  clone_53 = getitem_60 = None
    mul_142: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_22);  sub_27 = None
    mul_143: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_142, primals_207)
    add_105: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_143, primals_208);  mul_143 = primals_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_90: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_105, [6272, 768]);  add_105 = None
    permute_46: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    addmm_13: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_210, view_90, permute_46);  primals_210 = None
    view_91: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_13, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_144: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_91, 0.5)
    mul_145: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_91, 0.7071067811865476);  view_91 = None
    erf_11: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_145);  mul_145 = None
    add_106: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_146: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_144, add_106);  mul_144 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_92: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_146, [6272, 3072]);  mul_146 = None
    permute_47: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
    addmm_14: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_212, view_92, permute_47);  primals_212 = None
    view_93: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_14, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_147: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_20, view_93);  view_93 = None
    add_107: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_103, mul_147);  add_103 = mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_56: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_107, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_56, [2], correction = 0, keepdim = True)
    getitem_61: "f32[8, 784, 1]" = var_mean_23[0]
    getitem_62: "f32[8, 784, 1]" = var_mean_23[1];  var_mean_23 = None
    add_108: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_61, 1e-06);  getitem_61 = None
    rsqrt_23: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_28: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_56, getitem_62);  clone_56 = getitem_62 = None
    mul_148: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_23);  sub_28 = None
    mul_149: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_148, primals_213)
    add_109: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_149, primals_214);  mul_149 = primals_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_94: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_109, [6272, 768]);  add_109 = None
    permute_48: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    addmm_15: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_216, view_94, permute_48);  primals_216 = None
    view_95: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_15, [8, 784, 2304]);  addmm_15 = None
    view_96: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_95, [8, 784, 3, 16, 48]);  view_95 = None
    permute_49: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_96, [2, 0, 3, 4, 1]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_5 = torch.ops.aten.unbind.int(permute_49);  permute_49 = None
    getitem_63: "f32[8, 16, 48, 784]" = unbind_5[0]
    getitem_64: "f32[8, 16, 48, 784]" = unbind_5[1]
    getitem_65: "f32[8, 16, 48, 784]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_22: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_63, 2.0)
    sum_16: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_22, [-1], True);  pow_22 = None
    pow_23: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_16, 0.5);  sum_16 = None
    clamp_min_10: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_23, 1e-12)
    expand_30: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_10, [8, 16, 48, 784]);  clamp_min_10 = None
    div_21: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_63, expand_30);  expand_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_24: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_64, 2.0)
    sum_17: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_24, [-1], True);  pow_24 = None
    pow_25: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_17, 0.5);  sum_17 = None
    clamp_min_11: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_25, 1e-12)
    expand_31: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_11, [8, 16, 48, 784]);  clamp_min_11 = None
    div_22: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_64, expand_31);  expand_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_50: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_22, [0, 1, 3, 2]);  div_22 = None
    expand_32: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_21, [8, 16, 48, 784]);  div_21 = None
    clone_57: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_97: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_57, [128, 48, 784]);  clone_57 = None
    expand_33: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_50, [8, 16, 784, 48]);  permute_50 = None
    clone_58: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_98: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_58, [128, 784, 48]);  clone_58 = None
    bmm_10: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_97, view_98)
    view_99: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_10, [8, 16, 48, 48])
    mul_150: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_99, primals_22);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_150, [-1], True)
    sub_29: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_150, amax_5);  mul_150 = amax_5 = None
    exp_5: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_18: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_23: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_5, sum_18);  exp_5 = sum_18 = None
    alias_17: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_59: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_34: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_59, [8, 16, 48, 48]);  clone_59 = None
    view_100: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_34, [128, 48, 48]);  expand_34 = None
    expand_35: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_65, [8, 16, 48, 784]);  getitem_65 = None
    clone_60: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_101: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_60, [128, 48, 784]);  clone_60 = None
    bmm_11: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_100, view_101)
    view_102: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_11, [8, 16, 48, 784]);  bmm_11 = None
    permute_51: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_102, [0, 3, 1, 2]);  view_102 = None
    view_103: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_51, [8, 784, 768]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
    clone_61: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_103, memory_format = torch.contiguous_format);  view_103 = None
    view_104: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_61, [6272, 768]);  clone_61 = None
    mm_5: "f32[6272, 768]" = torch.ops.aten.mm.default(view_104, permute_52)
    view_105: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_5, [8, 784, 768])
    add_110: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_105, primals_218);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_151: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_21, add_110);  add_110 = None
    add_111: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_107, mul_151);  add_107 = mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_63: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_111, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_63, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 784, 1]" = var_mean_24[0]
    getitem_67: "f32[8, 784, 1]" = var_mean_24[1];  var_mean_24 = None
    add_112: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_24: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    sub_30: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_63, getitem_67);  clone_63 = getitem_67 = None
    mul_152: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_24);  sub_30 = None
    mul_153: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_152, primals_219)
    add_113: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_153, primals_220);  mul_153 = primals_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_53: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_113, [0, 2, 1]);  add_113 = None
    view_106: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_53, [8, 768, 28, 28]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_14: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_106, primals_221, primals_222, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_154: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.5)
    mul_155: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.7071067811865476)
    erf_12: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_114: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_156: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_154, add_114);  mul_154 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_655, 1)
    var_mean_25 = torch.ops.aten.var_mean.correction(mul_156, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 768, 1, 1]" = var_mean_25[0]
    getitem_69: "f32[1, 768, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_116: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_25: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_31: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_156, getitem_69);  mul_156 = None
    mul_157: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_25);  sub_31 = None
    squeeze_24: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_25: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_158: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_159: "f32[768]" = torch.ops.aten.mul.Tensor(primals_653, 0.9)
    add_117: "f32[768]" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
    squeeze_26: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_160: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0001594642002871);  squeeze_26 = None
    mul_161: "f32[768]" = torch.ops.aten.mul.Tensor(mul_160, 0.1);  mul_160 = None
    mul_162: "f32[768]" = torch.ops.aten.mul.Tensor(primals_654, 0.9)
    add_118: "f32[768]" = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
    unsqueeze_39: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_223, -1)
    unsqueeze_40: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, -1);  unsqueeze_39 = None
    mul_163: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_40);  mul_157 = unsqueeze_40 = None
    unsqueeze_41: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_224, -1);  primals_224 = None
    unsqueeze_42: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_41, -1);  unsqueeze_41 = None
    add_119: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_163, unsqueeze_42);  mul_163 = unsqueeze_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_15: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_119, primals_225, primals_226, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_107: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_15, [8, 768, 784])
    permute_54: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_164: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_23, permute_54);  permute_54 = None
    add_120: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_111, mul_164);  add_111 = mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_64: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_120, memory_format = torch.contiguous_format)
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_64, [2], correction = 0, keepdim = True)
    getitem_70: "f32[8, 784, 1]" = var_mean_26[0]
    getitem_71: "f32[8, 784, 1]" = var_mean_26[1];  var_mean_26 = None
    add_121: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
    rsqrt_26: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_32: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_64, getitem_71);  clone_64 = getitem_71 = None
    mul_165: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_26);  sub_32 = None
    mul_166: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_165, primals_227)
    add_122: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_166, primals_228);  mul_166 = primals_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_108: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_122, [6272, 768]);  add_122 = None
    permute_55: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
    addmm_16: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_230, view_108, permute_55);  primals_230 = None
    view_109: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_16, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_167: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    mul_168: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
    erf_13: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
    add_123: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_169: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_167, add_123);  mul_167 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_110: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_169, [6272, 3072]);  mul_169 = None
    permute_56: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
    addmm_17: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_232, view_110, permute_56);  primals_232 = None
    view_111: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_17, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_170: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_24, view_111);  view_111 = None
    add_124: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_120, mul_170);  add_120 = mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_67: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_124, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_67, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 784, 1]" = var_mean_27[0]
    getitem_73: "f32[8, 784, 1]" = var_mean_27[1];  var_mean_27 = None
    add_125: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_27: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
    sub_33: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_67, getitem_73);  clone_67 = getitem_73 = None
    mul_171: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_27);  sub_33 = None
    mul_172: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_171, primals_233)
    add_126: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_172, primals_234);  mul_172 = primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_112: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_126, [6272, 768]);  add_126 = None
    permute_57: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    addmm_18: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_236, view_112, permute_57);  primals_236 = None
    view_113: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_18, [8, 784, 2304]);  addmm_18 = None
    view_114: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_113, [8, 784, 3, 16, 48]);  view_113 = None
    permute_58: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_114, [2, 0, 3, 4, 1]);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_6 = torch.ops.aten.unbind.int(permute_58);  permute_58 = None
    getitem_74: "f32[8, 16, 48, 784]" = unbind_6[0]
    getitem_75: "f32[8, 16, 48, 784]" = unbind_6[1]
    getitem_76: "f32[8, 16, 48, 784]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_26: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_74, 2.0)
    sum_19: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_26, [-1], True);  pow_26 = None
    pow_27: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_19, 0.5);  sum_19 = None
    clamp_min_12: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_27, 1e-12)
    expand_36: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_12, [8, 16, 48, 784]);  clamp_min_12 = None
    div_24: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_74, expand_36);  expand_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_28: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_75, 2.0)
    sum_20: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_28, [-1], True);  pow_28 = None
    pow_29: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_20, 0.5);  sum_20 = None
    clamp_min_13: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_29, 1e-12)
    expand_37: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_13, [8, 16, 48, 784]);  clamp_min_13 = None
    div_25: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_75, expand_37);  expand_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_59: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_25, [0, 1, 3, 2]);  div_25 = None
    expand_38: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_24, [8, 16, 48, 784]);  div_24 = None
    clone_68: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_115: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_68, [128, 48, 784]);  clone_68 = None
    expand_39: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_59, [8, 16, 784, 48]);  permute_59 = None
    clone_69: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_116: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_69, [128, 784, 48]);  clone_69 = None
    bmm_12: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_115, view_116)
    view_117: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_12, [8, 16, 48, 48])
    mul_173: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_117, primals_26);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_6: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_173, [-1], True)
    sub_34: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_173, amax_6);  mul_173 = amax_6 = None
    exp_6: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_21: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_26: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_6, sum_21);  exp_6 = sum_21 = None
    alias_20: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_70: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_26);  div_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_40: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_70, [8, 16, 48, 48]);  clone_70 = None
    view_118: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_40, [128, 48, 48]);  expand_40 = None
    expand_41: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_76, [8, 16, 48, 784]);  getitem_76 = None
    clone_71: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_119: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_71, [128, 48, 784]);  clone_71 = None
    bmm_13: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_118, view_119)
    view_120: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_13, [8, 16, 48, 784]);  bmm_13 = None
    permute_60: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_120, [0, 3, 1, 2]);  view_120 = None
    view_121: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_60, [8, 784, 768]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_61: "f32[768, 768]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
    clone_72: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_121, memory_format = torch.contiguous_format);  view_121 = None
    view_122: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_72, [6272, 768]);  clone_72 = None
    mm_6: "f32[6272, 768]" = torch.ops.aten.mm.default(view_122, permute_61)
    view_123: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_6, [8, 784, 768])
    add_127: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_123, primals_238);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_174: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_25, add_127);  add_127 = None
    add_128: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_124, mul_174);  add_124 = mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_74: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_128, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_74, [2], correction = 0, keepdim = True)
    getitem_77: "f32[8, 784, 1]" = var_mean_28[0]
    getitem_78: "f32[8, 784, 1]" = var_mean_28[1];  var_mean_28 = None
    add_129: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-06);  getitem_77 = None
    rsqrt_28: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    sub_35: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_74, getitem_78);  clone_74 = getitem_78 = None
    mul_175: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_28);  sub_35 = None
    mul_176: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_175, primals_239)
    add_130: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_176, primals_240);  mul_176 = primals_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_62: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_130, [0, 2, 1]);  add_130 = None
    view_124: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_62, [8, 768, 28, 28]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_16: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_124, primals_241, primals_242, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_177: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, 0.5)
    mul_178: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_16, 0.7071067811865476)
    erf_14: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_178);  mul_178 = None
    add_131: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_179: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_177, add_131);  mul_177 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_132: "i64[]" = torch.ops.aten.add.Tensor(primals_658, 1)
    var_mean_29 = torch.ops.aten.var_mean.correction(mul_179, [0, 2, 3], correction = 0, keepdim = True)
    getitem_79: "f32[1, 768, 1, 1]" = var_mean_29[0]
    getitem_80: "f32[1, 768, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_133: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_79, 1e-05)
    rsqrt_29: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    sub_36: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_179, getitem_80);  mul_179 = None
    mul_180: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_29);  sub_36 = None
    squeeze_27: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    squeeze_28: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_181: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_182: "f32[768]" = torch.ops.aten.mul.Tensor(primals_656, 0.9)
    add_134: "f32[768]" = torch.ops.aten.add.Tensor(mul_181, mul_182);  mul_181 = mul_182 = None
    squeeze_29: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    mul_183: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0001594642002871);  squeeze_29 = None
    mul_184: "f32[768]" = torch.ops.aten.mul.Tensor(mul_183, 0.1);  mul_183 = None
    mul_185: "f32[768]" = torch.ops.aten.mul.Tensor(primals_657, 0.9)
    add_135: "f32[768]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
    unsqueeze_43: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_243, -1)
    unsqueeze_44: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, -1);  unsqueeze_43 = None
    mul_186: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_44);  mul_180 = unsqueeze_44 = None
    unsqueeze_45: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_244, -1);  primals_244 = None
    unsqueeze_46: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_45, -1);  unsqueeze_45 = None
    add_136: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_186, unsqueeze_46);  mul_186 = unsqueeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_17: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_136, primals_245, primals_246, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_125: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_17, [8, 768, 784])
    permute_63: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_125, [0, 2, 1]);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_187: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_27, permute_63);  permute_63 = None
    add_137: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_128, mul_187);  add_128 = mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_75: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_137, memory_format = torch.contiguous_format)
    var_mean_30 = torch.ops.aten.var_mean.correction(clone_75, [2], correction = 0, keepdim = True)
    getitem_81: "f32[8, 784, 1]" = var_mean_30[0]
    getitem_82: "f32[8, 784, 1]" = var_mean_30[1];  var_mean_30 = None
    add_138: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_81, 1e-06);  getitem_81 = None
    rsqrt_30: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    sub_37: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_75, getitem_82);  clone_75 = getitem_82 = None
    mul_188: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_30);  sub_37 = None
    mul_189: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_188, primals_247)
    add_139: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_189, primals_248);  mul_189 = primals_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_126: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_139, [6272, 768]);  add_139 = None
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
    addmm_19: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_250, view_126, permute_64);  primals_250 = None
    view_127: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_19, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_190: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_127, 0.5)
    mul_191: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_127, 0.7071067811865476);  view_127 = None
    erf_15: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_191);  mul_191 = None
    add_140: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_192: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_190, add_140);  mul_190 = add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_128: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_192, [6272, 3072]);  mul_192 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
    addmm_20: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_252, view_128, permute_65);  primals_252 = None
    view_129: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_20, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_193: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_28, view_129);  view_129 = None
    add_141: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_137, mul_193);  add_137 = mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_78: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_141, memory_format = torch.contiguous_format)
    var_mean_31 = torch.ops.aten.var_mean.correction(clone_78, [2], correction = 0, keepdim = True)
    getitem_83: "f32[8, 784, 1]" = var_mean_31[0]
    getitem_84: "f32[8, 784, 1]" = var_mean_31[1];  var_mean_31 = None
    add_142: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_83, 1e-06);  getitem_83 = None
    rsqrt_31: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    sub_38: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_78, getitem_84);  clone_78 = getitem_84 = None
    mul_194: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_31);  sub_38 = None
    mul_195: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_194, primals_253)
    add_143: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_195, primals_254);  mul_195 = primals_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_130: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_143, [6272, 768]);  add_143 = None
    permute_66: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_21: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_256, view_130, permute_66);  primals_256 = None
    view_131: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_21, [8, 784, 2304]);  addmm_21 = None
    view_132: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_131, [8, 784, 3, 16, 48]);  view_131 = None
    permute_67: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_132, [2, 0, 3, 4, 1]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_7 = torch.ops.aten.unbind.int(permute_67);  permute_67 = None
    getitem_85: "f32[8, 16, 48, 784]" = unbind_7[0]
    getitem_86: "f32[8, 16, 48, 784]" = unbind_7[1]
    getitem_87: "f32[8, 16, 48, 784]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_30: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_85, 2.0)
    sum_22: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_30, [-1], True);  pow_30 = None
    pow_31: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_22, 0.5);  sum_22 = None
    clamp_min_14: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_31, 1e-12)
    expand_42: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_14, [8, 16, 48, 784]);  clamp_min_14 = None
    div_27: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_85, expand_42);  expand_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_32: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_86, 2.0)
    sum_23: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_32, [-1], True);  pow_32 = None
    pow_33: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_23, 0.5);  sum_23 = None
    clamp_min_15: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_33, 1e-12)
    expand_43: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_15, [8, 16, 48, 784]);  clamp_min_15 = None
    div_28: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_86, expand_43);  expand_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_68: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_28, [0, 1, 3, 2]);  div_28 = None
    expand_44: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_27, [8, 16, 48, 784]);  div_27 = None
    clone_79: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_133: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_79, [128, 48, 784]);  clone_79 = None
    expand_45: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_68, [8, 16, 784, 48]);  permute_68 = None
    clone_80: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_134: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_80, [128, 784, 48]);  clone_80 = None
    bmm_14: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_133, view_134)
    view_135: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_14, [8, 16, 48, 48])
    mul_196: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_135, primals_30);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_7: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_196, [-1], True)
    sub_39: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_196, amax_7);  mul_196 = amax_7 = None
    exp_7: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_24: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_29: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_7, sum_24);  exp_7 = sum_24 = None
    alias_23: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_81: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_29);  div_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_46: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_81, [8, 16, 48, 48]);  clone_81 = None
    view_136: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_46, [128, 48, 48]);  expand_46 = None
    expand_47: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_87, [8, 16, 48, 784]);  getitem_87 = None
    clone_82: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_137: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_82, [128, 48, 784]);  clone_82 = None
    bmm_15: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_136, view_137)
    view_138: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_15, [8, 16, 48, 784]);  bmm_15 = None
    permute_69: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_138, [0, 3, 1, 2]);  view_138 = None
    view_139: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_69, [8, 784, 768]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(primals_257, [1, 0]);  primals_257 = None
    clone_83: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_139, memory_format = torch.contiguous_format);  view_139 = None
    view_140: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_83, [6272, 768]);  clone_83 = None
    mm_7: "f32[6272, 768]" = torch.ops.aten.mm.default(view_140, permute_70)
    view_141: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_7, [8, 784, 768])
    add_144: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_141, primals_258);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_197: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_29, add_144);  add_144 = None
    add_145: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_141, mul_197);  add_141 = mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_85: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_145, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_85, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 784, 1]" = var_mean_32[0]
    getitem_89: "f32[8, 784, 1]" = var_mean_32[1];  var_mean_32 = None
    add_146: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_32: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    sub_40: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_85, getitem_89);  clone_85 = getitem_89 = None
    mul_198: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_32);  sub_40 = None
    mul_199: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_198, primals_259)
    add_147: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_199, primals_260);  mul_199 = primals_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_71: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_147, [0, 2, 1]);  add_147 = None
    view_142: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_71, [8, 768, 28, 28]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_18: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_142, primals_261, primals_262, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_200: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.5)
    mul_201: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476)
    erf_16: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
    add_148: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_202: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_200, add_148);  mul_200 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_149: "i64[]" = torch.ops.aten.add.Tensor(primals_661, 1)
    var_mean_33 = torch.ops.aten.var_mean.correction(mul_202, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 768, 1, 1]" = var_mean_33[0]
    getitem_91: "f32[1, 768, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_150: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
    rsqrt_33: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
    sub_41: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_202, getitem_91);  mul_202 = None
    mul_203: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_33);  sub_41 = None
    squeeze_30: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_31: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_204: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_205: "f32[768]" = torch.ops.aten.mul.Tensor(primals_659, 0.9)
    add_151: "f32[768]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    squeeze_32: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_206: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0001594642002871);  squeeze_32 = None
    mul_207: "f32[768]" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
    mul_208: "f32[768]" = torch.ops.aten.mul.Tensor(primals_660, 0.9)
    add_152: "f32[768]" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
    unsqueeze_47: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_263, -1)
    unsqueeze_48: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_47, -1);  unsqueeze_47 = None
    mul_209: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_48);  mul_203 = unsqueeze_48 = None
    unsqueeze_49: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_264, -1);  primals_264 = None
    unsqueeze_50: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, -1);  unsqueeze_49 = None
    add_153: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_50);  mul_209 = unsqueeze_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_19: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_153, primals_265, primals_266, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_143: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_19, [8, 768, 784])
    permute_72: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_210: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_31, permute_72);  permute_72 = None
    add_154: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_145, mul_210);  add_145 = mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_86: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_154, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_86, [2], correction = 0, keepdim = True)
    getitem_92: "f32[8, 784, 1]" = var_mean_34[0]
    getitem_93: "f32[8, 784, 1]" = var_mean_34[1];  var_mean_34 = None
    add_155: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
    rsqrt_34: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_42: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_86, getitem_93);  clone_86 = getitem_93 = None
    mul_211: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_34);  sub_42 = None
    mul_212: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_211, primals_267)
    add_156: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_212, primals_268);  mul_212 = primals_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_144: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_156, [6272, 768]);  add_156 = None
    permute_73: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
    addmm_22: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_270, view_144, permute_73);  primals_270 = None
    view_145: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_213: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_145, 0.5)
    mul_214: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_145, 0.7071067811865476);  view_145 = None
    erf_17: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_214);  mul_214 = None
    add_157: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_215: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_213, add_157);  mul_213 = add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_146: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_215, [6272, 3072]);  mul_215 = None
    permute_74: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
    addmm_23: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_272, view_146, permute_74);  primals_272 = None
    view_147: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_23, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_216: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_32, view_147);  view_147 = None
    add_158: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_154, mul_216);  add_154 = mul_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_89: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_158, memory_format = torch.contiguous_format)
    var_mean_35 = torch.ops.aten.var_mean.correction(clone_89, [2], correction = 0, keepdim = True)
    getitem_94: "f32[8, 784, 1]" = var_mean_35[0]
    getitem_95: "f32[8, 784, 1]" = var_mean_35[1];  var_mean_35 = None
    add_159: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
    rsqrt_35: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    sub_43: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_89, getitem_95);  clone_89 = getitem_95 = None
    mul_217: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_35);  sub_43 = None
    mul_218: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_217, primals_273)
    add_160: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_218, primals_274);  mul_218 = primals_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_148: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_160, [6272, 768]);  add_160 = None
    permute_75: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_275, [1, 0]);  primals_275 = None
    addmm_24: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_276, view_148, permute_75);  primals_276 = None
    view_149: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_24, [8, 784, 2304]);  addmm_24 = None
    view_150: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_149, [8, 784, 3, 16, 48]);  view_149 = None
    permute_76: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_150, [2, 0, 3, 4, 1]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_8 = torch.ops.aten.unbind.int(permute_76);  permute_76 = None
    getitem_96: "f32[8, 16, 48, 784]" = unbind_8[0]
    getitem_97: "f32[8, 16, 48, 784]" = unbind_8[1]
    getitem_98: "f32[8, 16, 48, 784]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_34: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_96, 2.0)
    sum_25: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_34, [-1], True);  pow_34 = None
    pow_35: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_25, 0.5);  sum_25 = None
    clamp_min_16: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_35, 1e-12)
    expand_48: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_16, [8, 16, 48, 784]);  clamp_min_16 = None
    div_30: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_96, expand_48);  expand_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_36: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_97, 2.0)
    sum_26: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_36, [-1], True);  pow_36 = None
    pow_37: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_26, 0.5);  sum_26 = None
    clamp_min_17: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_37, 1e-12)
    expand_49: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_17, [8, 16, 48, 784]);  clamp_min_17 = None
    div_31: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_97, expand_49);  expand_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_77: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_31, [0, 1, 3, 2]);  div_31 = None
    expand_50: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_30, [8, 16, 48, 784]);  div_30 = None
    clone_90: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
    view_151: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_90, [128, 48, 784]);  clone_90 = None
    expand_51: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_77, [8, 16, 784, 48]);  permute_77 = None
    clone_91: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_152: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_91, [128, 784, 48]);  clone_91 = None
    bmm_16: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_151, view_152)
    view_153: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_16, [8, 16, 48, 48])
    mul_219: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_153, primals_34);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_8: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_219, [-1], True)
    sub_44: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_219, amax_8);  mul_219 = amax_8 = None
    exp_8: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
    sum_27: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_32: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_8, sum_27);  exp_8 = sum_27 = None
    alias_26: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_92: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_32);  div_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_52: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_92, [8, 16, 48, 48]);  clone_92 = None
    view_154: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_52, [128, 48, 48]);  expand_52 = None
    expand_53: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_98, [8, 16, 48, 784]);  getitem_98 = None
    clone_93: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_155: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_93, [128, 48, 784]);  clone_93 = None
    bmm_17: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_154, view_155)
    view_156: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_17, [8, 16, 48, 784]);  bmm_17 = None
    permute_78: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_156, [0, 3, 1, 2]);  view_156 = None
    view_157: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_78, [8, 784, 768]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(primals_277, [1, 0]);  primals_277 = None
    clone_94: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_157, memory_format = torch.contiguous_format);  view_157 = None
    view_158: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_94, [6272, 768]);  clone_94 = None
    mm_8: "f32[6272, 768]" = torch.ops.aten.mm.default(view_158, permute_79)
    view_159: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_8, [8, 784, 768])
    add_161: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_159, primals_278);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_220: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_33, add_161);  add_161 = None
    add_162: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_158, mul_220);  add_158 = mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_96: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_162, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_96, [2], correction = 0, keepdim = True)
    getitem_99: "f32[8, 784, 1]" = var_mean_36[0]
    getitem_100: "f32[8, 784, 1]" = var_mean_36[1];  var_mean_36 = None
    add_163: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-06);  getitem_99 = None
    rsqrt_36: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_45: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_96, getitem_100);  clone_96 = getitem_100 = None
    mul_221: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_36);  sub_45 = None
    mul_222: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_221, primals_279)
    add_164: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_222, primals_280);  mul_222 = primals_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_80: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_164, [0, 2, 1]);  add_164 = None
    view_160: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_80, [8, 768, 28, 28]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_20: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_160, primals_281, primals_282, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_223: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.5)
    mul_224: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.7071067811865476)
    erf_18: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_224);  mul_224 = None
    add_165: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_225: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_223, add_165);  mul_223 = add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_166: "i64[]" = torch.ops.aten.add.Tensor(primals_664, 1)
    var_mean_37 = torch.ops.aten.var_mean.correction(mul_225, [0, 2, 3], correction = 0, keepdim = True)
    getitem_101: "f32[1, 768, 1, 1]" = var_mean_37[0]
    getitem_102: "f32[1, 768, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_167: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_101, 1e-05)
    rsqrt_37: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    sub_46: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_225, getitem_102);  mul_225 = None
    mul_226: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_37);  sub_46 = None
    squeeze_33: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    squeeze_34: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_227: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_228: "f32[768]" = torch.ops.aten.mul.Tensor(primals_662, 0.9)
    add_168: "f32[768]" = torch.ops.aten.add.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
    squeeze_35: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    mul_229: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001594642002871);  squeeze_35 = None
    mul_230: "f32[768]" = torch.ops.aten.mul.Tensor(mul_229, 0.1);  mul_229 = None
    mul_231: "f32[768]" = torch.ops.aten.mul.Tensor(primals_663, 0.9)
    add_169: "f32[768]" = torch.ops.aten.add.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None
    unsqueeze_51: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_283, -1)
    unsqueeze_52: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_51, -1);  unsqueeze_51 = None
    mul_232: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_52);  mul_226 = unsqueeze_52 = None
    unsqueeze_53: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_284, -1);  primals_284 = None
    unsqueeze_54: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_53, -1);  unsqueeze_53 = None
    add_170: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_232, unsqueeze_54);  mul_232 = unsqueeze_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_21: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_170, primals_285, primals_286, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_161: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_21, [8, 768, 784])
    permute_81: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_233: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_35, permute_81);  permute_81 = None
    add_171: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_162, mul_233);  add_162 = mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_97: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_171, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_97, [2], correction = 0, keepdim = True)
    getitem_103: "f32[8, 784, 1]" = var_mean_38[0]
    getitem_104: "f32[8, 784, 1]" = var_mean_38[1];  var_mean_38 = None
    add_172: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_103, 1e-06);  getitem_103 = None
    rsqrt_38: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_47: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_97, getitem_104);  clone_97 = getitem_104 = None
    mul_234: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_38);  sub_47 = None
    mul_235: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_234, primals_287)
    add_173: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_235, primals_288);  mul_235 = primals_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_162: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_173, [6272, 768]);  add_173 = None
    permute_82: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
    addmm_25: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_290, view_162, permute_82);  primals_290 = None
    view_163: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_25, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_236: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_163, 0.5)
    mul_237: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_163, 0.7071067811865476);  view_163 = None
    erf_19: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_237);  mul_237 = None
    add_174: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_238: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_236, add_174);  mul_236 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_164: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_238, [6272, 3072]);  mul_238 = None
    permute_83: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
    addmm_26: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_292, view_164, permute_83);  primals_292 = None
    view_165: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_26, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_239: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_36, view_165);  view_165 = None
    add_175: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_171, mul_239);  add_171 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_100: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_175, memory_format = torch.contiguous_format)
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_100, [2], correction = 0, keepdim = True)
    getitem_105: "f32[8, 784, 1]" = var_mean_39[0]
    getitem_106: "f32[8, 784, 1]" = var_mean_39[1];  var_mean_39 = None
    add_176: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_105, 1e-06);  getitem_105 = None
    rsqrt_39: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    sub_48: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_100, getitem_106);  clone_100 = getitem_106 = None
    mul_240: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_39);  sub_48 = None
    mul_241: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_240, primals_293)
    add_177: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_241, primals_294);  mul_241 = primals_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_166: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_177, [6272, 768]);  add_177 = None
    permute_84: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_295, [1, 0]);  primals_295 = None
    addmm_27: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_296, view_166, permute_84);  primals_296 = None
    view_167: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_27, [8, 784, 2304]);  addmm_27 = None
    view_168: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_167, [8, 784, 3, 16, 48]);  view_167 = None
    permute_85: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_168, [2, 0, 3, 4, 1]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_9 = torch.ops.aten.unbind.int(permute_85);  permute_85 = None
    getitem_107: "f32[8, 16, 48, 784]" = unbind_9[0]
    getitem_108: "f32[8, 16, 48, 784]" = unbind_9[1]
    getitem_109: "f32[8, 16, 48, 784]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_38: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_107, 2.0)
    sum_28: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_38, [-1], True);  pow_38 = None
    pow_39: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_28, 0.5);  sum_28 = None
    clamp_min_18: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_39, 1e-12)
    expand_54: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_18, [8, 16, 48, 784]);  clamp_min_18 = None
    div_33: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_107, expand_54);  expand_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_40: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_108, 2.0)
    sum_29: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_40, [-1], True);  pow_40 = None
    pow_41: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_29, 0.5);  sum_29 = None
    clamp_min_19: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_41, 1e-12)
    expand_55: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_19, [8, 16, 48, 784]);  clamp_min_19 = None
    div_34: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_108, expand_55);  expand_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_86: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_34, [0, 1, 3, 2]);  div_34 = None
    expand_56: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_33, [8, 16, 48, 784]);  div_33 = None
    clone_101: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_169: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_101, [128, 48, 784]);  clone_101 = None
    expand_57: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_86, [8, 16, 784, 48]);  permute_86 = None
    clone_102: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_170: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_102, [128, 784, 48]);  clone_102 = None
    bmm_18: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_169, view_170)
    view_171: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_18, [8, 16, 48, 48])
    mul_242: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_171, primals_38);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_9: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_242, [-1], True)
    sub_49: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_242, amax_9);  mul_242 = amax_9 = None
    exp_9: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_30: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_35: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_9, sum_30);  exp_9 = sum_30 = None
    alias_29: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_103: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_35);  div_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_58: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_103, [8, 16, 48, 48]);  clone_103 = None
    view_172: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_58, [128, 48, 48]);  expand_58 = None
    expand_59: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_109, [8, 16, 48, 784]);  getitem_109 = None
    clone_104: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_173: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_104, [128, 48, 784]);  clone_104 = None
    bmm_19: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_172, view_173)
    view_174: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_19, [8, 16, 48, 784]);  bmm_19 = None
    permute_87: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_174, [0, 3, 1, 2]);  view_174 = None
    view_175: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_87, [8, 784, 768]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(primals_297, [1, 0]);  primals_297 = None
    clone_105: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_175, memory_format = torch.contiguous_format);  view_175 = None
    view_176: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_105, [6272, 768]);  clone_105 = None
    mm_9: "f32[6272, 768]" = torch.ops.aten.mm.default(view_176, permute_88)
    view_177: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_9, [8, 784, 768])
    add_178: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_177, primals_298);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_243: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_37, add_178);  add_178 = None
    add_179: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_175, mul_243);  add_175 = mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_107: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_179, memory_format = torch.contiguous_format)
    var_mean_40 = torch.ops.aten.var_mean.correction(clone_107, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 784, 1]" = var_mean_40[0]
    getitem_111: "f32[8, 784, 1]" = var_mean_40[1];  var_mean_40 = None
    add_180: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
    rsqrt_40: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
    sub_50: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_107, getitem_111);  clone_107 = getitem_111 = None
    mul_244: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_40);  sub_50 = None
    mul_245: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_244, primals_299)
    add_181: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_245, primals_300);  mul_245 = primals_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_89: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_181, [0, 2, 1]);  add_181 = None
    view_178: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_89, [8, 768, 28, 28]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_22: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_178, primals_301, primals_302, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_246: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, 0.5)
    mul_247: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_22, 0.7071067811865476)
    erf_20: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_247);  mul_247 = None
    add_182: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_248: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_246, add_182);  mul_246 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_183: "i64[]" = torch.ops.aten.add.Tensor(primals_667, 1)
    var_mean_41 = torch.ops.aten.var_mean.correction(mul_248, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 768, 1, 1]" = var_mean_41[0]
    getitem_113: "f32[1, 768, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_184: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_41: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    sub_51: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_248, getitem_113);  mul_248 = None
    mul_249: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_41);  sub_51 = None
    squeeze_36: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_37: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_250: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_251: "f32[768]" = torch.ops.aten.mul.Tensor(primals_665, 0.9)
    add_185: "f32[768]" = torch.ops.aten.add.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
    squeeze_38: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_252: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001594642002871);  squeeze_38 = None
    mul_253: "f32[768]" = torch.ops.aten.mul.Tensor(mul_252, 0.1);  mul_252 = None
    mul_254: "f32[768]" = torch.ops.aten.mul.Tensor(primals_666, 0.9)
    add_186: "f32[768]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
    unsqueeze_55: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_303, -1)
    unsqueeze_56: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, -1);  unsqueeze_55 = None
    mul_255: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_249, unsqueeze_56);  mul_249 = unsqueeze_56 = None
    unsqueeze_57: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_304, -1);  primals_304 = None
    unsqueeze_58: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_57, -1);  unsqueeze_57 = None
    add_187: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_255, unsqueeze_58);  mul_255 = unsqueeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_23: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_187, primals_305, primals_306, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_179: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_23, [8, 768, 784])
    permute_90: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_256: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_39, permute_90);  permute_90 = None
    add_188: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_179, mul_256);  add_179 = mul_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_108: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_188, memory_format = torch.contiguous_format)
    var_mean_42 = torch.ops.aten.var_mean.correction(clone_108, [2], correction = 0, keepdim = True)
    getitem_114: "f32[8, 784, 1]" = var_mean_42[0]
    getitem_115: "f32[8, 784, 1]" = var_mean_42[1];  var_mean_42 = None
    add_189: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-06);  getitem_114 = None
    rsqrt_42: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_52: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_108, getitem_115);  clone_108 = getitem_115 = None
    mul_257: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_42);  sub_52 = None
    mul_258: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_257, primals_307)
    add_190: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_258, primals_308);  mul_258 = primals_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_180: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_190, [6272, 768]);  add_190 = None
    permute_91: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_309, [1, 0]);  primals_309 = None
    addmm_28: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_310, view_180, permute_91);  primals_310 = None
    view_181: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_28, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_259: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_181, 0.5)
    mul_260: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476);  view_181 = None
    erf_21: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_260);  mul_260 = None
    add_191: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_261: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_259, add_191);  mul_259 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_182: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_261, [6272, 3072]);  mul_261 = None
    permute_92: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_311, [1, 0]);  primals_311 = None
    addmm_29: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_312, view_182, permute_92);  primals_312 = None
    view_183: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_29, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_262: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_40, view_183);  view_183 = None
    add_192: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_188, mul_262);  add_188 = mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_111: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_192, memory_format = torch.contiguous_format)
    var_mean_43 = torch.ops.aten.var_mean.correction(clone_111, [2], correction = 0, keepdim = True)
    getitem_116: "f32[8, 784, 1]" = var_mean_43[0]
    getitem_117: "f32[8, 784, 1]" = var_mean_43[1];  var_mean_43 = None
    add_193: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-06);  getitem_116 = None
    rsqrt_43: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    sub_53: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_111, getitem_117);  clone_111 = getitem_117 = None
    mul_263: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_43);  sub_53 = None
    mul_264: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_263, primals_313)
    add_194: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_264, primals_314);  mul_264 = primals_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_184: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_194, [6272, 768]);  add_194 = None
    permute_93: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_315, [1, 0]);  primals_315 = None
    addmm_30: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_316, view_184, permute_93);  primals_316 = None
    view_185: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_30, [8, 784, 2304]);  addmm_30 = None
    view_186: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_185, [8, 784, 3, 16, 48]);  view_185 = None
    permute_94: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_186, [2, 0, 3, 4, 1]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_10 = torch.ops.aten.unbind.int(permute_94);  permute_94 = None
    getitem_118: "f32[8, 16, 48, 784]" = unbind_10[0]
    getitem_119: "f32[8, 16, 48, 784]" = unbind_10[1]
    getitem_120: "f32[8, 16, 48, 784]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_42: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_118, 2.0)
    sum_31: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_42, [-1], True);  pow_42 = None
    pow_43: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_31, 0.5);  sum_31 = None
    clamp_min_20: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_43, 1e-12)
    expand_60: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_20, [8, 16, 48, 784]);  clamp_min_20 = None
    div_36: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_118, expand_60);  expand_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_44: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_119, 2.0)
    sum_32: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_44, [-1], True);  pow_44 = None
    pow_45: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_32, 0.5);  sum_32 = None
    clamp_min_21: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_45, 1e-12)
    expand_61: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_21, [8, 16, 48, 784]);  clamp_min_21 = None
    div_37: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_119, expand_61);  expand_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_95: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_37, [0, 1, 3, 2]);  div_37 = None
    expand_62: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_36, [8, 16, 48, 784]);  div_36 = None
    clone_112: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
    view_187: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_112, [128, 48, 784]);  clone_112 = None
    expand_63: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_95, [8, 16, 784, 48]);  permute_95 = None
    clone_113: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    view_188: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_113, [128, 784, 48]);  clone_113 = None
    bmm_20: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_187, view_188)
    view_189: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_20, [8, 16, 48, 48])
    mul_265: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_189, primals_42);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_10: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_265, [-1], True)
    sub_54: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_265, amax_10);  mul_265 = amax_10 = None
    exp_10: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_54);  sub_54 = None
    sum_33: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_38: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_10, sum_33);  exp_10 = sum_33 = None
    alias_32: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_114: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_38);  div_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_64: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_114, [8, 16, 48, 48]);  clone_114 = None
    view_190: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_64, [128, 48, 48]);  expand_64 = None
    expand_65: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_120, [8, 16, 48, 784]);  getitem_120 = None
    clone_115: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_191: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_115, [128, 48, 784]);  clone_115 = None
    bmm_21: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_190, view_191)
    view_192: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_21, [8, 16, 48, 784]);  bmm_21 = None
    permute_96: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_192, [0, 3, 1, 2]);  view_192 = None
    view_193: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_96, [8, 784, 768]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(primals_317, [1, 0]);  primals_317 = None
    clone_116: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_193, memory_format = torch.contiguous_format);  view_193 = None
    view_194: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_116, [6272, 768]);  clone_116 = None
    mm_10: "f32[6272, 768]" = torch.ops.aten.mm.default(view_194, permute_97)
    view_195: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_10, [8, 784, 768])
    add_195: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_195, primals_318);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_266: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_41, add_195);  add_195 = None
    add_196: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_192, mul_266);  add_192 = mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_118: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_196, memory_format = torch.contiguous_format)
    var_mean_44 = torch.ops.aten.var_mean.correction(clone_118, [2], correction = 0, keepdim = True)
    getitem_121: "f32[8, 784, 1]" = var_mean_44[0]
    getitem_122: "f32[8, 784, 1]" = var_mean_44[1];  var_mean_44 = None
    add_197: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-06);  getitem_121 = None
    rsqrt_44: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    sub_55: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_118, getitem_122);  clone_118 = getitem_122 = None
    mul_267: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_44);  sub_55 = None
    mul_268: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_267, primals_319)
    add_198: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_268, primals_320);  mul_268 = primals_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_98: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_198, [0, 2, 1]);  add_198 = None
    view_196: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_98, [8, 768, 28, 28]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_24: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_196, primals_321, primals_322, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_269: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, 0.5)
    mul_270: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_24, 0.7071067811865476)
    erf_22: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_270);  mul_270 = None
    add_199: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_271: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_269, add_199);  mul_269 = add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_200: "i64[]" = torch.ops.aten.add.Tensor(primals_670, 1)
    var_mean_45 = torch.ops.aten.var_mean.correction(mul_271, [0, 2, 3], correction = 0, keepdim = True)
    getitem_123: "f32[1, 768, 1, 1]" = var_mean_45[0]
    getitem_124: "f32[1, 768, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_201: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_123, 1e-05)
    rsqrt_45: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
    sub_56: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_271, getitem_124);  mul_271 = None
    mul_272: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_45);  sub_56 = None
    squeeze_39: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    squeeze_40: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_273: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_274: "f32[768]" = torch.ops.aten.mul.Tensor(primals_668, 0.9)
    add_202: "f32[768]" = torch.ops.aten.add.Tensor(mul_273, mul_274);  mul_273 = mul_274 = None
    squeeze_41: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    mul_275: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001594642002871);  squeeze_41 = None
    mul_276: "f32[768]" = torch.ops.aten.mul.Tensor(mul_275, 0.1);  mul_275 = None
    mul_277: "f32[768]" = torch.ops.aten.mul.Tensor(primals_669, 0.9)
    add_203: "f32[768]" = torch.ops.aten.add.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
    unsqueeze_59: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_323, -1)
    unsqueeze_60: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_59, -1);  unsqueeze_59 = None
    mul_278: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_272, unsqueeze_60);  mul_272 = unsqueeze_60 = None
    unsqueeze_61: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_324, -1);  primals_324 = None
    unsqueeze_62: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, -1);  unsqueeze_61 = None
    add_204: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_278, unsqueeze_62);  mul_278 = unsqueeze_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_25: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_204, primals_325, primals_326, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_197: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_25, [8, 768, 784])
    permute_99: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_197, [0, 2, 1]);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_279: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_43, permute_99);  permute_99 = None
    add_205: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_196, mul_279);  add_196 = mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_119: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_205, memory_format = torch.contiguous_format)
    var_mean_46 = torch.ops.aten.var_mean.correction(clone_119, [2], correction = 0, keepdim = True)
    getitem_125: "f32[8, 784, 1]" = var_mean_46[0]
    getitem_126: "f32[8, 784, 1]" = var_mean_46[1];  var_mean_46 = None
    add_206: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_125, 1e-06);  getitem_125 = None
    rsqrt_46: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
    sub_57: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_119, getitem_126);  clone_119 = getitem_126 = None
    mul_280: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_46);  sub_57 = None
    mul_281: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_280, primals_327)
    add_207: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_281, primals_328);  mul_281 = primals_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_198: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_207, [6272, 768]);  add_207 = None
    permute_100: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_329, [1, 0]);  primals_329 = None
    addmm_31: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_330, view_198, permute_100);  primals_330 = None
    view_199: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_31, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_282: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_199, 0.5)
    mul_283: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_199, 0.7071067811865476);  view_199 = None
    erf_23: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_283);  mul_283 = None
    add_208: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_284: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_282, add_208);  mul_282 = add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_200: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_284, [6272, 3072]);  mul_284 = None
    permute_101: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_331, [1, 0]);  primals_331 = None
    addmm_32: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_332, view_200, permute_101);  primals_332 = None
    view_201: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_32, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_285: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_44, view_201);  view_201 = None
    add_209: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_205, mul_285);  add_205 = mul_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_122: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_209, memory_format = torch.contiguous_format)
    var_mean_47 = torch.ops.aten.var_mean.correction(clone_122, [2], correction = 0, keepdim = True)
    getitem_127: "f32[8, 784, 1]" = var_mean_47[0]
    getitem_128: "f32[8, 784, 1]" = var_mean_47[1];  var_mean_47 = None
    add_210: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_127, 1e-06);  getitem_127 = None
    rsqrt_47: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    sub_58: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_122, getitem_128);  clone_122 = getitem_128 = None
    mul_286: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_47);  sub_58 = None
    mul_287: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_286, primals_333)
    add_211: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_287, primals_334);  mul_287 = primals_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_202: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_211, [6272, 768]);  add_211 = None
    permute_102: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_335, [1, 0]);  primals_335 = None
    addmm_33: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_336, view_202, permute_102);  primals_336 = None
    view_203: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_33, [8, 784, 2304]);  addmm_33 = None
    view_204: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_203, [8, 784, 3, 16, 48]);  view_203 = None
    permute_103: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_204, [2, 0, 3, 4, 1]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_11 = torch.ops.aten.unbind.int(permute_103);  permute_103 = None
    getitem_129: "f32[8, 16, 48, 784]" = unbind_11[0]
    getitem_130: "f32[8, 16, 48, 784]" = unbind_11[1]
    getitem_131: "f32[8, 16, 48, 784]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_46: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_129, 2.0)
    sum_34: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_46, [-1], True);  pow_46 = None
    pow_47: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_34, 0.5);  sum_34 = None
    clamp_min_22: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_47, 1e-12)
    expand_66: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_22, [8, 16, 48, 784]);  clamp_min_22 = None
    div_39: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_129, expand_66);  expand_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_48: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_130, 2.0)
    sum_35: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_48, [-1], True);  pow_48 = None
    pow_49: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_35, 0.5);  sum_35 = None
    clamp_min_23: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_49, 1e-12)
    expand_67: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_23, [8, 16, 48, 784]);  clamp_min_23 = None
    div_40: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_130, expand_67);  expand_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_104: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_40, [0, 1, 3, 2]);  div_40 = None
    expand_68: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_39, [8, 16, 48, 784]);  div_39 = None
    clone_123: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_205: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_123, [128, 48, 784]);  clone_123 = None
    expand_69: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_104, [8, 16, 784, 48]);  permute_104 = None
    clone_124: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    view_206: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_124, [128, 784, 48]);  clone_124 = None
    bmm_22: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_205, view_206)
    view_207: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_22, [8, 16, 48, 48])
    mul_288: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_207, primals_46);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_11: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_288, [-1], True)
    sub_59: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_288, amax_11);  mul_288 = amax_11 = None
    exp_11: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_59);  sub_59 = None
    sum_36: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_41: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_11, sum_36);  exp_11 = sum_36 = None
    alias_35: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_125: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_41);  div_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_70: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_125, [8, 16, 48, 48]);  clone_125 = None
    view_208: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_70, [128, 48, 48]);  expand_70 = None
    expand_71: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_131, [8, 16, 48, 784]);  getitem_131 = None
    clone_126: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    view_209: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_126, [128, 48, 784]);  clone_126 = None
    bmm_23: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_208, view_209)
    view_210: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_23, [8, 16, 48, 784]);  bmm_23 = None
    permute_105: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_210, [0, 3, 1, 2]);  view_210 = None
    view_211: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_105, [8, 784, 768]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_106: "f32[768, 768]" = torch.ops.aten.permute.default(primals_337, [1, 0]);  primals_337 = None
    clone_127: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_211, memory_format = torch.contiguous_format);  view_211 = None
    view_212: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_127, [6272, 768]);  clone_127 = None
    mm_11: "f32[6272, 768]" = torch.ops.aten.mm.default(view_212, permute_106)
    view_213: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_11, [8, 784, 768])
    add_212: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_213, primals_338);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_289: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_45, add_212);  add_212 = None
    add_213: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_209, mul_289);  add_209 = mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_129: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_213, memory_format = torch.contiguous_format)
    var_mean_48 = torch.ops.aten.var_mean.correction(clone_129, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 784, 1]" = var_mean_48[0]
    getitem_133: "f32[8, 784, 1]" = var_mean_48[1];  var_mean_48 = None
    add_214: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_48: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
    sub_60: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_129, getitem_133);  clone_129 = getitem_133 = None
    mul_290: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_48);  sub_60 = None
    mul_291: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_290, primals_339)
    add_215: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_291, primals_340);  mul_291 = primals_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_107: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_215, [0, 2, 1]);  add_215 = None
    view_214: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_107, [8, 768, 28, 28]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_26: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_214, primals_341, primals_342, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_292: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_26, 0.5)
    mul_293: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_26, 0.7071067811865476)
    erf_24: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_293);  mul_293 = None
    add_216: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_294: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_292, add_216);  mul_292 = add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_217: "i64[]" = torch.ops.aten.add.Tensor(primals_673, 1)
    var_mean_49 = torch.ops.aten.var_mean.correction(mul_294, [0, 2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[1, 768, 1, 1]" = var_mean_49[0]
    getitem_135: "f32[1, 768, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_218: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05)
    rsqrt_49: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    sub_61: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_294, getitem_135);  mul_294 = None
    mul_295: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_49);  sub_61 = None
    squeeze_42: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
    squeeze_43: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_296: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_297: "f32[768]" = torch.ops.aten.mul.Tensor(primals_671, 0.9)
    add_219: "f32[768]" = torch.ops.aten.add.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
    squeeze_44: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
    mul_298: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001594642002871);  squeeze_44 = None
    mul_299: "f32[768]" = torch.ops.aten.mul.Tensor(mul_298, 0.1);  mul_298 = None
    mul_300: "f32[768]" = torch.ops.aten.mul.Tensor(primals_672, 0.9)
    add_220: "f32[768]" = torch.ops.aten.add.Tensor(mul_299, mul_300);  mul_299 = mul_300 = None
    unsqueeze_63: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_343, -1)
    unsqueeze_64: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, -1);  unsqueeze_63 = None
    mul_301: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_295, unsqueeze_64);  mul_295 = unsqueeze_64 = None
    unsqueeze_65: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_344, -1);  primals_344 = None
    unsqueeze_66: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_65, -1);  unsqueeze_65 = None
    add_221: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_301, unsqueeze_66);  mul_301 = unsqueeze_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_27: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_221, primals_345, primals_346, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_215: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_27, [8, 768, 784])
    permute_108: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_215, [0, 2, 1]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_302: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_47, permute_108);  permute_108 = None
    add_222: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_213, mul_302);  add_213 = mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_130: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_222, memory_format = torch.contiguous_format)
    var_mean_50 = torch.ops.aten.var_mean.correction(clone_130, [2], correction = 0, keepdim = True)
    getitem_136: "f32[8, 784, 1]" = var_mean_50[0]
    getitem_137: "f32[8, 784, 1]" = var_mean_50[1];  var_mean_50 = None
    add_223: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-06);  getitem_136 = None
    rsqrt_50: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
    sub_62: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_130, getitem_137);  clone_130 = getitem_137 = None
    mul_303: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_50);  sub_62 = None
    mul_304: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_303, primals_347)
    add_224: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_304, primals_348);  mul_304 = primals_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_216: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_224, [6272, 768]);  add_224 = None
    permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_349, [1, 0]);  primals_349 = None
    addmm_34: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_350, view_216, permute_109);  primals_350 = None
    view_217: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_305: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_306: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
    erf_25: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_306);  mul_306 = None
    add_225: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_307: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_305, add_225);  mul_305 = add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_218: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_307, [6272, 3072]);  mul_307 = None
    permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_351, [1, 0]);  primals_351 = None
    addmm_35: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_352, view_218, permute_110);  primals_352 = None
    view_219: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_35, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_308: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_48, view_219);  view_219 = None
    add_226: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_222, mul_308);  add_222 = mul_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_133: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_226, memory_format = torch.contiguous_format)
    var_mean_51 = torch.ops.aten.var_mean.correction(clone_133, [2], correction = 0, keepdim = True)
    getitem_138: "f32[8, 784, 1]" = var_mean_51[0]
    getitem_139: "f32[8, 784, 1]" = var_mean_51[1];  var_mean_51 = None
    add_227: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-06);  getitem_138 = None
    rsqrt_51: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
    sub_63: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_133, getitem_139);  clone_133 = getitem_139 = None
    mul_309: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_51);  sub_63 = None
    mul_310: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_309, primals_353)
    add_228: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_310, primals_354);  mul_310 = primals_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_220: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_228, [6272, 768]);  add_228 = None
    permute_111: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_355, [1, 0]);  primals_355 = None
    addmm_36: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_356, view_220, permute_111);  primals_356 = None
    view_221: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_36, [8, 784, 2304]);  addmm_36 = None
    view_222: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_221, [8, 784, 3, 16, 48]);  view_221 = None
    permute_112: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_222, [2, 0, 3, 4, 1]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_12 = torch.ops.aten.unbind.int(permute_112);  permute_112 = None
    getitem_140: "f32[8, 16, 48, 784]" = unbind_12[0]
    getitem_141: "f32[8, 16, 48, 784]" = unbind_12[1]
    getitem_142: "f32[8, 16, 48, 784]" = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_50: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_140, 2.0)
    sum_37: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_50, [-1], True);  pow_50 = None
    pow_51: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_37, 0.5);  sum_37 = None
    clamp_min_24: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_51, 1e-12)
    expand_72: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_24, [8, 16, 48, 784]);  clamp_min_24 = None
    div_42: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_140, expand_72);  expand_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_52: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_141, 2.0)
    sum_38: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_52, [-1], True);  pow_52 = None
    pow_53: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_38, 0.5);  sum_38 = None
    clamp_min_25: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_53, 1e-12)
    expand_73: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_25, [8, 16, 48, 784]);  clamp_min_25 = None
    div_43: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_141, expand_73);  expand_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_113: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_43, [0, 1, 3, 2]);  div_43 = None
    expand_74: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_42, [8, 16, 48, 784]);  div_42 = None
    clone_134: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    view_223: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_134, [128, 48, 784]);  clone_134 = None
    expand_75: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_113, [8, 16, 784, 48]);  permute_113 = None
    clone_135: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    view_224: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_135, [128, 784, 48]);  clone_135 = None
    bmm_24: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_223, view_224)
    view_225: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_24, [8, 16, 48, 48])
    mul_311: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_225, primals_50);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_12: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_311, [-1], True)
    sub_64: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_311, amax_12);  mul_311 = amax_12 = None
    exp_12: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_39: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_44: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_12, sum_39);  exp_12 = sum_39 = None
    alias_38: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_136: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_44);  div_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_76: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_136, [8, 16, 48, 48]);  clone_136 = None
    view_226: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_76, [128, 48, 48]);  expand_76 = None
    expand_77: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_142, [8, 16, 48, 784]);  getitem_142 = None
    clone_137: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
    view_227: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_137, [128, 48, 784]);  clone_137 = None
    bmm_25: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_226, view_227)
    view_228: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_25, [8, 16, 48, 784]);  bmm_25 = None
    permute_114: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_228, [0, 3, 1, 2]);  view_228 = None
    view_229: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_114, [8, 784, 768]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(primals_357, [1, 0]);  primals_357 = None
    clone_138: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_229, memory_format = torch.contiguous_format);  view_229 = None
    view_230: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_138, [6272, 768]);  clone_138 = None
    mm_12: "f32[6272, 768]" = torch.ops.aten.mm.default(view_230, permute_115)
    view_231: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_12, [8, 784, 768])
    add_229: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_231, primals_358);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_312: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_49, add_229);  add_229 = None
    add_230: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_226, mul_312);  add_226 = mul_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_140: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_230, memory_format = torch.contiguous_format)
    var_mean_52 = torch.ops.aten.var_mean.correction(clone_140, [2], correction = 0, keepdim = True)
    getitem_143: "f32[8, 784, 1]" = var_mean_52[0]
    getitem_144: "f32[8, 784, 1]" = var_mean_52[1];  var_mean_52 = None
    add_231: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_143, 1e-06);  getitem_143 = None
    rsqrt_52: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
    sub_65: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_140, getitem_144);  clone_140 = getitem_144 = None
    mul_313: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_52);  sub_65 = None
    mul_314: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_313, primals_359)
    add_232: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_314, primals_360);  mul_314 = primals_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_116: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_232, [0, 2, 1]);  add_232 = None
    view_232: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_116, [8, 768, 28, 28]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_28: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_232, primals_361, primals_362, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_315: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_28, 0.5)
    mul_316: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_28, 0.7071067811865476)
    erf_26: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_316);  mul_316 = None
    add_233: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_317: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_315, add_233);  mul_315 = add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_234: "i64[]" = torch.ops.aten.add.Tensor(primals_676, 1)
    var_mean_53 = torch.ops.aten.var_mean.correction(mul_317, [0, 2, 3], correction = 0, keepdim = True)
    getitem_145: "f32[1, 768, 1, 1]" = var_mean_53[0]
    getitem_146: "f32[1, 768, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_235: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_145, 1e-05)
    rsqrt_53: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
    sub_66: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_317, getitem_146);  mul_317 = None
    mul_318: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_53);  sub_66 = None
    squeeze_45: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
    squeeze_46: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_319: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_320: "f32[768]" = torch.ops.aten.mul.Tensor(primals_674, 0.9)
    add_236: "f32[768]" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
    squeeze_47: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
    mul_321: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001594642002871);  squeeze_47 = None
    mul_322: "f32[768]" = torch.ops.aten.mul.Tensor(mul_321, 0.1);  mul_321 = None
    mul_323: "f32[768]" = torch.ops.aten.mul.Tensor(primals_675, 0.9)
    add_237: "f32[768]" = torch.ops.aten.add.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None
    unsqueeze_67: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_363, -1)
    unsqueeze_68: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, -1);  unsqueeze_67 = None
    mul_324: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_318, unsqueeze_68);  mul_318 = unsqueeze_68 = None
    unsqueeze_69: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_364, -1);  primals_364 = None
    unsqueeze_70: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, -1);  unsqueeze_69 = None
    add_238: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_324, unsqueeze_70);  mul_324 = unsqueeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_29: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_238, primals_365, primals_366, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_233: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_29, [8, 768, 784])
    permute_117: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_325: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_51, permute_117);  permute_117 = None
    add_239: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_230, mul_325);  add_230 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_141: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_239, memory_format = torch.contiguous_format)
    var_mean_54 = torch.ops.aten.var_mean.correction(clone_141, [2], correction = 0, keepdim = True)
    getitem_147: "f32[8, 784, 1]" = var_mean_54[0]
    getitem_148: "f32[8, 784, 1]" = var_mean_54[1];  var_mean_54 = None
    add_240: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_147, 1e-06);  getitem_147 = None
    rsqrt_54: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
    sub_67: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_141, getitem_148);  clone_141 = getitem_148 = None
    mul_326: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_54);  sub_67 = None
    mul_327: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_326, primals_367)
    add_241: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_327, primals_368);  mul_327 = primals_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_234: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_241, [6272, 768]);  add_241 = None
    permute_118: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_369, [1, 0]);  primals_369 = None
    addmm_37: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_370, view_234, permute_118);  primals_370 = None
    view_235: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_37, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_328: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_235, 0.5)
    mul_329: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_235, 0.7071067811865476);  view_235 = None
    erf_27: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_329);  mul_329 = None
    add_242: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_330: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_328, add_242);  mul_328 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_236: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_330, [6272, 3072]);  mul_330 = None
    permute_119: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_371, [1, 0]);  primals_371 = None
    addmm_38: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_372, view_236, permute_119);  primals_372 = None
    view_237: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_38, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_331: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_52, view_237);  view_237 = None
    add_243: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_239, mul_331);  add_239 = mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_144: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_243, memory_format = torch.contiguous_format)
    var_mean_55 = torch.ops.aten.var_mean.correction(clone_144, [2], correction = 0, keepdim = True)
    getitem_149: "f32[8, 784, 1]" = var_mean_55[0]
    getitem_150: "f32[8, 784, 1]" = var_mean_55[1];  var_mean_55 = None
    add_244: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_149, 1e-06);  getitem_149 = None
    rsqrt_55: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
    sub_68: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_144, getitem_150);  clone_144 = getitem_150 = None
    mul_332: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_55);  sub_68 = None
    mul_333: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_332, primals_373)
    add_245: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_333, primals_374);  mul_333 = primals_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_238: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_245, [6272, 768]);  add_245 = None
    permute_120: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_375, [1, 0]);  primals_375 = None
    addmm_39: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_376, view_238, permute_120);  primals_376 = None
    view_239: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_39, [8, 784, 2304]);  addmm_39 = None
    view_240: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_239, [8, 784, 3, 16, 48]);  view_239 = None
    permute_121: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_240, [2, 0, 3, 4, 1]);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_13 = torch.ops.aten.unbind.int(permute_121);  permute_121 = None
    getitem_151: "f32[8, 16, 48, 784]" = unbind_13[0]
    getitem_152: "f32[8, 16, 48, 784]" = unbind_13[1]
    getitem_153: "f32[8, 16, 48, 784]" = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_54: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_151, 2.0)
    sum_40: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_54, [-1], True);  pow_54 = None
    pow_55: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_40, 0.5);  sum_40 = None
    clamp_min_26: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_55, 1e-12)
    expand_78: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_26, [8, 16, 48, 784]);  clamp_min_26 = None
    div_45: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_151, expand_78);  expand_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_56: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_152, 2.0)
    sum_41: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_56, [-1], True);  pow_56 = None
    pow_57: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_41, 0.5);  sum_41 = None
    clamp_min_27: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_57, 1e-12)
    expand_79: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_27, [8, 16, 48, 784]);  clamp_min_27 = None
    div_46: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_152, expand_79);  expand_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_122: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_46, [0, 1, 3, 2]);  div_46 = None
    expand_80: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_45, [8, 16, 48, 784]);  div_45 = None
    clone_145: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    view_241: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_145, [128, 48, 784]);  clone_145 = None
    expand_81: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_122, [8, 16, 784, 48]);  permute_122 = None
    clone_146: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    view_242: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_146, [128, 784, 48]);  clone_146 = None
    bmm_26: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_241, view_242)
    view_243: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_26, [8, 16, 48, 48])
    mul_334: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_243, primals_54);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_13: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_334, [-1], True)
    sub_69: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_334, amax_13);  mul_334 = amax_13 = None
    exp_13: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_69);  sub_69 = None
    sum_42: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_47: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_13, sum_42);  exp_13 = sum_42 = None
    alias_41: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_47)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_147: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_47);  div_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_82: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_147, [8, 16, 48, 48]);  clone_147 = None
    view_244: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_82, [128, 48, 48]);  expand_82 = None
    expand_83: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_153, [8, 16, 48, 784]);  getitem_153 = None
    clone_148: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
    view_245: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_148, [128, 48, 784]);  clone_148 = None
    bmm_27: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_244, view_245)
    view_246: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_27, [8, 16, 48, 784]);  bmm_27 = None
    permute_123: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_246, [0, 3, 1, 2]);  view_246 = None
    view_247: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_123, [8, 784, 768]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(primals_377, [1, 0]);  primals_377 = None
    clone_149: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_247, memory_format = torch.contiguous_format);  view_247 = None
    view_248: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_149, [6272, 768]);  clone_149 = None
    mm_13: "f32[6272, 768]" = torch.ops.aten.mm.default(view_248, permute_124)
    view_249: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_13, [8, 784, 768])
    add_246: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_249, primals_378);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_335: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_53, add_246);  add_246 = None
    add_247: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_243, mul_335);  add_243 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_151: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_247, memory_format = torch.contiguous_format)
    var_mean_56 = torch.ops.aten.var_mean.correction(clone_151, [2], correction = 0, keepdim = True)
    getitem_154: "f32[8, 784, 1]" = var_mean_56[0]
    getitem_155: "f32[8, 784, 1]" = var_mean_56[1];  var_mean_56 = None
    add_248: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-06);  getitem_154 = None
    rsqrt_56: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_248);  add_248 = None
    sub_70: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_151, getitem_155);  clone_151 = getitem_155 = None
    mul_336: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_56);  sub_70 = None
    mul_337: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_336, primals_379)
    add_249: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_337, primals_380);  mul_337 = primals_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_125: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_249, [0, 2, 1]);  add_249 = None
    view_250: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_125, [8, 768, 28, 28]);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_30: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_250, primals_381, primals_382, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_338: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_30, 0.5)
    mul_339: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_30, 0.7071067811865476)
    erf_28: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_339);  mul_339 = None
    add_250: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_340: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_338, add_250);  mul_338 = add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_251: "i64[]" = torch.ops.aten.add.Tensor(primals_679, 1)
    var_mean_57 = torch.ops.aten.var_mean.correction(mul_340, [0, 2, 3], correction = 0, keepdim = True)
    getitem_156: "f32[1, 768, 1, 1]" = var_mean_57[0]
    getitem_157: "f32[1, 768, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_252: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
    rsqrt_57: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_252);  add_252 = None
    sub_71: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_340, getitem_157);  mul_340 = None
    mul_341: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_57);  sub_71 = None
    squeeze_48: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
    squeeze_49: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_342: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_343: "f32[768]" = torch.ops.aten.mul.Tensor(primals_677, 0.9)
    add_253: "f32[768]" = torch.ops.aten.add.Tensor(mul_342, mul_343);  mul_342 = mul_343 = None
    squeeze_50: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
    mul_344: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
    mul_345: "f32[768]" = torch.ops.aten.mul.Tensor(mul_344, 0.1);  mul_344 = None
    mul_346: "f32[768]" = torch.ops.aten.mul.Tensor(primals_678, 0.9)
    add_254: "f32[768]" = torch.ops.aten.add.Tensor(mul_345, mul_346);  mul_345 = mul_346 = None
    unsqueeze_71: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_383, -1)
    unsqueeze_72: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_71, -1);  unsqueeze_71 = None
    mul_347: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_341, unsqueeze_72);  mul_341 = unsqueeze_72 = None
    unsqueeze_73: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_384, -1);  primals_384 = None
    unsqueeze_74: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_73, -1);  unsqueeze_73 = None
    add_255: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_347, unsqueeze_74);  mul_347 = unsqueeze_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_31: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_255, primals_385, primals_386, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_251: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_31, [8, 768, 784])
    permute_126: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_348: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_55, permute_126);  permute_126 = None
    add_256: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_247, mul_348);  add_247 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_152: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_256, memory_format = torch.contiguous_format)
    var_mean_58 = torch.ops.aten.var_mean.correction(clone_152, [2], correction = 0, keepdim = True)
    getitem_158: "f32[8, 784, 1]" = var_mean_58[0]
    getitem_159: "f32[8, 784, 1]" = var_mean_58[1];  var_mean_58 = None
    add_257: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
    rsqrt_58: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
    sub_72: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_152, getitem_159);  clone_152 = getitem_159 = None
    mul_349: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_58);  sub_72 = None
    mul_350: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_349, primals_387)
    add_258: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_350, primals_388);  mul_350 = primals_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_252: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_258, [6272, 768]);  add_258 = None
    permute_127: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_389, [1, 0]);  primals_389 = None
    addmm_40: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_390, view_252, permute_127);  primals_390 = None
    view_253: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_40, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_351: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_253, 0.5)
    mul_352: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_253, 0.7071067811865476);  view_253 = None
    erf_29: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_352);  mul_352 = None
    add_259: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_353: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_351, add_259);  mul_351 = add_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_254: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_353, [6272, 3072]);  mul_353 = None
    permute_128: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_391, [1, 0]);  primals_391 = None
    addmm_41: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_392, view_254, permute_128);  primals_392 = None
    view_255: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_41, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_354: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_56, view_255);  view_255 = None
    add_260: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_256, mul_354);  add_256 = mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_155: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_260, memory_format = torch.contiguous_format)
    var_mean_59 = torch.ops.aten.var_mean.correction(clone_155, [2], correction = 0, keepdim = True)
    getitem_160: "f32[8, 784, 1]" = var_mean_59[0]
    getitem_161: "f32[8, 784, 1]" = var_mean_59[1];  var_mean_59 = None
    add_261: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
    rsqrt_59: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
    sub_73: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_155, getitem_161);  clone_155 = getitem_161 = None
    mul_355: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_59);  sub_73 = None
    mul_356: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_355, primals_393)
    add_262: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_356, primals_394);  mul_356 = primals_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_256: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_262, [6272, 768]);  add_262 = None
    permute_129: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_395, [1, 0]);  primals_395 = None
    addmm_42: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_396, view_256, permute_129);  primals_396 = None
    view_257: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_42, [8, 784, 2304]);  addmm_42 = None
    view_258: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_257, [8, 784, 3, 16, 48]);  view_257 = None
    permute_130: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_258, [2, 0, 3, 4, 1]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_14 = torch.ops.aten.unbind.int(permute_130);  permute_130 = None
    getitem_162: "f32[8, 16, 48, 784]" = unbind_14[0]
    getitem_163: "f32[8, 16, 48, 784]" = unbind_14[1]
    getitem_164: "f32[8, 16, 48, 784]" = unbind_14[2];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_58: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_162, 2.0)
    sum_43: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_58, [-1], True);  pow_58 = None
    pow_59: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_43, 0.5);  sum_43 = None
    clamp_min_28: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_59, 1e-12)
    expand_84: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_28, [8, 16, 48, 784]);  clamp_min_28 = None
    div_48: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_162, expand_84);  expand_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_60: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_163, 2.0)
    sum_44: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_60, [-1], True);  pow_60 = None
    pow_61: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_44, 0.5);  sum_44 = None
    clamp_min_29: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_61, 1e-12)
    expand_85: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_29, [8, 16, 48, 784]);  clamp_min_29 = None
    div_49: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_163, expand_85);  expand_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_131: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_49, [0, 1, 3, 2]);  div_49 = None
    expand_86: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_48, [8, 16, 48, 784]);  div_48 = None
    clone_156: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
    view_259: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_156, [128, 48, 784]);  clone_156 = None
    expand_87: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_131, [8, 16, 784, 48]);  permute_131 = None
    clone_157: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
    view_260: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_157, [128, 784, 48]);  clone_157 = None
    bmm_28: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_259, view_260)
    view_261: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_28, [8, 16, 48, 48])
    mul_357: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_261, primals_58);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_14: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_357, [-1], True)
    sub_74: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_357, amax_14);  mul_357 = amax_14 = None
    exp_14: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_74);  sub_74 = None
    sum_45: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_50: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_14, sum_45);  exp_14 = sum_45 = None
    alias_44: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_158: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_50);  div_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_88: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_158, [8, 16, 48, 48]);  clone_158 = None
    view_262: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_88, [128, 48, 48]);  expand_88 = None
    expand_89: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_164, [8, 16, 48, 784]);  getitem_164 = None
    clone_159: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    view_263: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_159, [128, 48, 784]);  clone_159 = None
    bmm_29: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_262, view_263)
    view_264: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_29, [8, 16, 48, 784]);  bmm_29 = None
    permute_132: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_264, [0, 3, 1, 2]);  view_264 = None
    view_265: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_132, [8, 784, 768]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_133: "f32[768, 768]" = torch.ops.aten.permute.default(primals_397, [1, 0]);  primals_397 = None
    clone_160: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_265, memory_format = torch.contiguous_format);  view_265 = None
    view_266: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_160, [6272, 768]);  clone_160 = None
    mm_14: "f32[6272, 768]" = torch.ops.aten.mm.default(view_266, permute_133)
    view_267: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_14, [8, 784, 768])
    add_263: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_267, primals_398);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_358: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_57, add_263);  add_263 = None
    add_264: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_260, mul_358);  add_260 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_162: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_264, memory_format = torch.contiguous_format)
    var_mean_60 = torch.ops.aten.var_mean.correction(clone_162, [2], correction = 0, keepdim = True)
    getitem_165: "f32[8, 784, 1]" = var_mean_60[0]
    getitem_166: "f32[8, 784, 1]" = var_mean_60[1];  var_mean_60 = None
    add_265: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_165, 1e-06);  getitem_165 = None
    rsqrt_60: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_265);  add_265 = None
    sub_75: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_162, getitem_166);  clone_162 = getitem_166 = None
    mul_359: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_60);  sub_75 = None
    mul_360: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_359, primals_399)
    add_266: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_360, primals_400);  mul_360 = primals_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_134: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_266, [0, 2, 1]);  add_266 = None
    view_268: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_134, [8, 768, 28, 28]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_32: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_268, primals_401, primals_402, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_361: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_32, 0.5)
    mul_362: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_32, 0.7071067811865476)
    erf_30: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_362);  mul_362 = None
    add_267: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_363: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_361, add_267);  mul_361 = add_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_268: "i64[]" = torch.ops.aten.add.Tensor(primals_682, 1)
    var_mean_61 = torch.ops.aten.var_mean.correction(mul_363, [0, 2, 3], correction = 0, keepdim = True)
    getitem_167: "f32[1, 768, 1, 1]" = var_mean_61[0]
    getitem_168: "f32[1, 768, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_269: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_167, 1e-05)
    rsqrt_61: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_269);  add_269 = None
    sub_76: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_363, getitem_168);  mul_363 = None
    mul_364: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_61);  sub_76 = None
    squeeze_51: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
    squeeze_52: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_365: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_366: "f32[768]" = torch.ops.aten.mul.Tensor(primals_680, 0.9)
    add_270: "f32[768]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
    squeeze_53: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
    mul_367: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0001594642002871);  squeeze_53 = None
    mul_368: "f32[768]" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
    mul_369: "f32[768]" = torch.ops.aten.mul.Tensor(primals_681, 0.9)
    add_271: "f32[768]" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
    unsqueeze_75: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_403, -1)
    unsqueeze_76: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, -1);  unsqueeze_75 = None
    mul_370: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_76);  mul_364 = unsqueeze_76 = None
    unsqueeze_77: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_404, -1);  primals_404 = None
    unsqueeze_78: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_77, -1);  unsqueeze_77 = None
    add_272: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_78);  mul_370 = unsqueeze_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_33: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_272, primals_405, primals_406, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_269: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_33, [8, 768, 784])
    permute_135: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_371: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_59, permute_135);  permute_135 = None
    add_273: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_264, mul_371);  add_264 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_163: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_273, memory_format = torch.contiguous_format)
    var_mean_62 = torch.ops.aten.var_mean.correction(clone_163, [2], correction = 0, keepdim = True)
    getitem_169: "f32[8, 784, 1]" = var_mean_62[0]
    getitem_170: "f32[8, 784, 1]" = var_mean_62[1];  var_mean_62 = None
    add_274: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_169, 1e-06);  getitem_169 = None
    rsqrt_62: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_274);  add_274 = None
    sub_77: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_163, getitem_170);  clone_163 = getitem_170 = None
    mul_372: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_62);  sub_77 = None
    mul_373: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_372, primals_407)
    add_275: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_373, primals_408);  mul_373 = primals_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_270: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_275, [6272, 768]);  add_275 = None
    permute_136: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_409, [1, 0]);  primals_409 = None
    addmm_43: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_410, view_270, permute_136);  primals_410 = None
    view_271: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_43, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_374: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_271, 0.5)
    mul_375: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_271, 0.7071067811865476);  view_271 = None
    erf_31: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_375);  mul_375 = None
    add_276: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_376: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_374, add_276);  mul_374 = add_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_272: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_376, [6272, 3072]);  mul_376 = None
    permute_137: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_411, [1, 0]);  primals_411 = None
    addmm_44: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_412, view_272, permute_137);  primals_412 = None
    view_273: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_44, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_377: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_60, view_273);  view_273 = None
    add_277: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_273, mul_377);  add_273 = mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_166: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_277, memory_format = torch.contiguous_format)
    var_mean_63 = torch.ops.aten.var_mean.correction(clone_166, [2], correction = 0, keepdim = True)
    getitem_171: "f32[8, 784, 1]" = var_mean_63[0]
    getitem_172: "f32[8, 784, 1]" = var_mean_63[1];  var_mean_63 = None
    add_278: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_171, 1e-06);  getitem_171 = None
    rsqrt_63: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
    sub_78: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_166, getitem_172);  clone_166 = getitem_172 = None
    mul_378: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_63);  sub_78 = None
    mul_379: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_378, primals_413)
    add_279: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_379, primals_414);  mul_379 = primals_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_274: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_279, [6272, 768]);  add_279 = None
    permute_138: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_415, [1, 0]);  primals_415 = None
    addmm_45: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_416, view_274, permute_138);  primals_416 = None
    view_275: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_45, [8, 784, 2304]);  addmm_45 = None
    view_276: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_275, [8, 784, 3, 16, 48]);  view_275 = None
    permute_139: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_276, [2, 0, 3, 4, 1]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_15 = torch.ops.aten.unbind.int(permute_139);  permute_139 = None
    getitem_173: "f32[8, 16, 48, 784]" = unbind_15[0]
    getitem_174: "f32[8, 16, 48, 784]" = unbind_15[1]
    getitem_175: "f32[8, 16, 48, 784]" = unbind_15[2];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_62: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_173, 2.0)
    sum_46: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_62, [-1], True);  pow_62 = None
    pow_63: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_46, 0.5);  sum_46 = None
    clamp_min_30: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_63, 1e-12)
    expand_90: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_30, [8, 16, 48, 784]);  clamp_min_30 = None
    div_51: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_173, expand_90);  expand_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_64: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_174, 2.0)
    sum_47: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_64, [-1], True);  pow_64 = None
    pow_65: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_47, 0.5);  sum_47 = None
    clamp_min_31: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_65, 1e-12)
    expand_91: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_31, [8, 16, 48, 784]);  clamp_min_31 = None
    div_52: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_174, expand_91);  expand_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_140: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_52, [0, 1, 3, 2]);  div_52 = None
    expand_92: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_51, [8, 16, 48, 784]);  div_51 = None
    clone_167: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
    view_277: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_167, [128, 48, 784]);  clone_167 = None
    expand_93: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_140, [8, 16, 784, 48]);  permute_140 = None
    clone_168: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
    view_278: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_168, [128, 784, 48]);  clone_168 = None
    bmm_30: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_277, view_278)
    view_279: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_30, [8, 16, 48, 48])
    mul_380: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_279, primals_62);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_15: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_380, [-1], True)
    sub_79: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_380, amax_15);  mul_380 = amax_15 = None
    exp_15: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_79);  sub_79 = None
    sum_48: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_53: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_15, sum_48);  exp_15 = sum_48 = None
    alias_47: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_169: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_53);  div_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_94: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_169, [8, 16, 48, 48]);  clone_169 = None
    view_280: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_94, [128, 48, 48]);  expand_94 = None
    expand_95: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_175, [8, 16, 48, 784]);  getitem_175 = None
    clone_170: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
    view_281: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_170, [128, 48, 784]);  clone_170 = None
    bmm_31: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_280, view_281)
    view_282: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_31, [8, 16, 48, 784]);  bmm_31 = None
    permute_141: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_282, [0, 3, 1, 2]);  view_282 = None
    view_283: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_141, [8, 784, 768]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_142: "f32[768, 768]" = torch.ops.aten.permute.default(primals_417, [1, 0]);  primals_417 = None
    clone_171: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_283, memory_format = torch.contiguous_format);  view_283 = None
    view_284: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_171, [6272, 768]);  clone_171 = None
    mm_15: "f32[6272, 768]" = torch.ops.aten.mm.default(view_284, permute_142)
    view_285: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_15, [8, 784, 768])
    add_280: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_285, primals_418);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_381: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_61, add_280);  add_280 = None
    add_281: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_277, mul_381);  add_277 = mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_173: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_281, memory_format = torch.contiguous_format)
    var_mean_64 = torch.ops.aten.var_mean.correction(clone_173, [2], correction = 0, keepdim = True)
    getitem_176: "f32[8, 784, 1]" = var_mean_64[0]
    getitem_177: "f32[8, 784, 1]" = var_mean_64[1];  var_mean_64 = None
    add_282: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
    rsqrt_64: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
    sub_80: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_173, getitem_177);  clone_173 = getitem_177 = None
    mul_382: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_64);  sub_80 = None
    mul_383: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_382, primals_419)
    add_283: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_383, primals_420);  mul_383 = primals_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_143: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_283, [0, 2, 1]);  add_283 = None
    view_286: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_143, [8, 768, 28, 28]);  permute_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_34: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_286, primals_421, primals_422, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_384: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_34, 0.5)
    mul_385: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_34, 0.7071067811865476)
    erf_32: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_385);  mul_385 = None
    add_284: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_386: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_384, add_284);  mul_384 = add_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_285: "i64[]" = torch.ops.aten.add.Tensor(primals_685, 1)
    var_mean_65 = torch.ops.aten.var_mean.correction(mul_386, [0, 2, 3], correction = 0, keepdim = True)
    getitem_178: "f32[1, 768, 1, 1]" = var_mean_65[0]
    getitem_179: "f32[1, 768, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_286: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05)
    rsqrt_65: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
    sub_81: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_386, getitem_179);  mul_386 = None
    mul_387: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_65);  sub_81 = None
    squeeze_54: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
    squeeze_55: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
    mul_388: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_389: "f32[768]" = torch.ops.aten.mul.Tensor(primals_683, 0.9)
    add_287: "f32[768]" = torch.ops.aten.add.Tensor(mul_388, mul_389);  mul_388 = mul_389 = None
    squeeze_56: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_178, [0, 2, 3]);  getitem_178 = None
    mul_390: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
    mul_391: "f32[768]" = torch.ops.aten.mul.Tensor(mul_390, 0.1);  mul_390 = None
    mul_392: "f32[768]" = torch.ops.aten.mul.Tensor(primals_684, 0.9)
    add_288: "f32[768]" = torch.ops.aten.add.Tensor(mul_391, mul_392);  mul_391 = mul_392 = None
    unsqueeze_79: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_423, -1)
    unsqueeze_80: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, -1);  unsqueeze_79 = None
    mul_393: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_387, unsqueeze_80);  mul_387 = unsqueeze_80 = None
    unsqueeze_81: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_424, -1);  primals_424 = None
    unsqueeze_82: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_81, -1);  unsqueeze_81 = None
    add_289: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_393, unsqueeze_82);  mul_393 = unsqueeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_35: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_289, primals_425, primals_426, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_287: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_35, [8, 768, 784])
    permute_144: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_394: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_63, permute_144);  permute_144 = None
    add_290: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_281, mul_394);  add_281 = mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_174: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_290, memory_format = torch.contiguous_format)
    var_mean_66 = torch.ops.aten.var_mean.correction(clone_174, [2], correction = 0, keepdim = True)
    getitem_180: "f32[8, 784, 1]" = var_mean_66[0]
    getitem_181: "f32[8, 784, 1]" = var_mean_66[1];  var_mean_66 = None
    add_291: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-06);  getitem_180 = None
    rsqrt_66: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_291);  add_291 = None
    sub_82: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_174, getitem_181);  clone_174 = getitem_181 = None
    mul_395: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_66);  sub_82 = None
    mul_396: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_395, primals_427)
    add_292: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_396, primals_428);  mul_396 = primals_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_288: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_292, [6272, 768]);  add_292 = None
    permute_145: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_429, [1, 0]);  primals_429 = None
    addmm_46: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_430, view_288, permute_145);  primals_430 = None
    view_289: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_46, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_397: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_289, 0.5)
    mul_398: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_289, 0.7071067811865476);  view_289 = None
    erf_33: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_398);  mul_398 = None
    add_293: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_399: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_397, add_293);  mul_397 = add_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_290: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_399, [6272, 3072]);  mul_399 = None
    permute_146: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_431, [1, 0]);  primals_431 = None
    addmm_47: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_432, view_290, permute_146);  primals_432 = None
    view_291: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_47, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_400: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_64, view_291);  view_291 = None
    add_294: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_290, mul_400);  add_290 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_177: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_294, memory_format = torch.contiguous_format)
    var_mean_67 = torch.ops.aten.var_mean.correction(clone_177, [2], correction = 0, keepdim = True)
    getitem_182: "f32[8, 784, 1]" = var_mean_67[0]
    getitem_183: "f32[8, 784, 1]" = var_mean_67[1];  var_mean_67 = None
    add_295: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-06);  getitem_182 = None
    rsqrt_67: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_295);  add_295 = None
    sub_83: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_177, getitem_183);  clone_177 = getitem_183 = None
    mul_401: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_67);  sub_83 = None
    mul_402: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_401, primals_433)
    add_296: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_402, primals_434);  mul_402 = primals_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_292: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_296, [6272, 768]);  add_296 = None
    permute_147: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_435, [1, 0]);  primals_435 = None
    addmm_48: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_436, view_292, permute_147);  primals_436 = None
    view_293: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_48, [8, 784, 2304]);  addmm_48 = None
    view_294: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_293, [8, 784, 3, 16, 48]);  view_293 = None
    permute_148: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_294, [2, 0, 3, 4, 1]);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_16 = torch.ops.aten.unbind.int(permute_148);  permute_148 = None
    getitem_184: "f32[8, 16, 48, 784]" = unbind_16[0]
    getitem_185: "f32[8, 16, 48, 784]" = unbind_16[1]
    getitem_186: "f32[8, 16, 48, 784]" = unbind_16[2];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_66: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_184, 2.0)
    sum_49: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_66, [-1], True);  pow_66 = None
    pow_67: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_49, 0.5);  sum_49 = None
    clamp_min_32: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_67, 1e-12)
    expand_96: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_32, [8, 16, 48, 784]);  clamp_min_32 = None
    div_54: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_184, expand_96);  expand_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_68: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_185, 2.0)
    sum_50: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_68, [-1], True);  pow_68 = None
    pow_69: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_50, 0.5);  sum_50 = None
    clamp_min_33: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_69, 1e-12)
    expand_97: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_33, [8, 16, 48, 784]);  clamp_min_33 = None
    div_55: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_185, expand_97);  expand_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_149: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_55, [0, 1, 3, 2]);  div_55 = None
    expand_98: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_54, [8, 16, 48, 784]);  div_54 = None
    clone_178: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
    view_295: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_178, [128, 48, 784]);  clone_178 = None
    expand_99: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_149, [8, 16, 784, 48]);  permute_149 = None
    clone_179: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
    view_296: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_179, [128, 784, 48]);  clone_179 = None
    bmm_32: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_295, view_296)
    view_297: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_32, [8, 16, 48, 48])
    mul_403: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_297, primals_66);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_16: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_403, [-1], True)
    sub_84: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_403, amax_16);  mul_403 = amax_16 = None
    exp_16: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_84);  sub_84 = None
    sum_51: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_56: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_16, sum_51);  exp_16 = sum_51 = None
    alias_50: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_180: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_56);  div_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_100: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_180, [8, 16, 48, 48]);  clone_180 = None
    view_298: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_100, [128, 48, 48]);  expand_100 = None
    expand_101: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_186, [8, 16, 48, 784]);  getitem_186 = None
    clone_181: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_101, memory_format = torch.contiguous_format);  expand_101 = None
    view_299: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_181, [128, 48, 784]);  clone_181 = None
    bmm_33: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_298, view_299)
    view_300: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_33, [8, 16, 48, 784]);  bmm_33 = None
    permute_150: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_300, [0, 3, 1, 2]);  view_300 = None
    view_301: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_150, [8, 784, 768]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_151: "f32[768, 768]" = torch.ops.aten.permute.default(primals_437, [1, 0]);  primals_437 = None
    clone_182: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_301, memory_format = torch.contiguous_format);  view_301 = None
    view_302: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_182, [6272, 768]);  clone_182 = None
    mm_16: "f32[6272, 768]" = torch.ops.aten.mm.default(view_302, permute_151)
    view_303: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_16, [8, 784, 768])
    add_297: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_303, primals_438);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_404: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_65, add_297);  add_297 = None
    add_298: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_294, mul_404);  add_294 = mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_184: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_298, memory_format = torch.contiguous_format)
    var_mean_68 = torch.ops.aten.var_mean.correction(clone_184, [2], correction = 0, keepdim = True)
    getitem_187: "f32[8, 784, 1]" = var_mean_68[0]
    getitem_188: "f32[8, 784, 1]" = var_mean_68[1];  var_mean_68 = None
    add_299: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_187, 1e-06);  getitem_187 = None
    rsqrt_68: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_299);  add_299 = None
    sub_85: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_184, getitem_188);  clone_184 = getitem_188 = None
    mul_405: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_68);  sub_85 = None
    mul_406: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_405, primals_439)
    add_300: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_406, primals_440);  mul_406 = primals_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_152: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_300, [0, 2, 1]);  add_300 = None
    view_304: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_152, [8, 768, 28, 28]);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_36: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_304, primals_441, primals_442, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_407: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_36, 0.5)
    mul_408: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_36, 0.7071067811865476)
    erf_34: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_408);  mul_408 = None
    add_301: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_409: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_407, add_301);  mul_407 = add_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_302: "i64[]" = torch.ops.aten.add.Tensor(primals_688, 1)
    var_mean_69 = torch.ops.aten.var_mean.correction(mul_409, [0, 2, 3], correction = 0, keepdim = True)
    getitem_189: "f32[1, 768, 1, 1]" = var_mean_69[0]
    getitem_190: "f32[1, 768, 1, 1]" = var_mean_69[1];  var_mean_69 = None
    add_303: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_189, 1e-05)
    rsqrt_69: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
    sub_86: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_409, getitem_190);  mul_409 = None
    mul_410: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_69);  sub_86 = None
    squeeze_57: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_190, [0, 2, 3]);  getitem_190 = None
    squeeze_58: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
    mul_411: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_412: "f32[768]" = torch.ops.aten.mul.Tensor(primals_686, 0.9)
    add_304: "f32[768]" = torch.ops.aten.add.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
    squeeze_59: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
    mul_413: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
    mul_414: "f32[768]" = torch.ops.aten.mul.Tensor(mul_413, 0.1);  mul_413 = None
    mul_415: "f32[768]" = torch.ops.aten.mul.Tensor(primals_687, 0.9)
    add_305: "f32[768]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    unsqueeze_83: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_443, -1)
    unsqueeze_84: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_83, -1);  unsqueeze_83 = None
    mul_416: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_410, unsqueeze_84);  mul_410 = unsqueeze_84 = None
    unsqueeze_85: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_444, -1);  primals_444 = None
    unsqueeze_86: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, -1);  unsqueeze_85 = None
    add_306: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_416, unsqueeze_86);  mul_416 = unsqueeze_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_37: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_306, primals_445, primals_446, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_305: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_37, [8, 768, 784])
    permute_153: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_305, [0, 2, 1]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_417: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_67, permute_153);  permute_153 = None
    add_307: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_298, mul_417);  add_298 = mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_185: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_307, memory_format = torch.contiguous_format)
    var_mean_70 = torch.ops.aten.var_mean.correction(clone_185, [2], correction = 0, keepdim = True)
    getitem_191: "f32[8, 784, 1]" = var_mean_70[0]
    getitem_192: "f32[8, 784, 1]" = var_mean_70[1];  var_mean_70 = None
    add_308: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_191, 1e-06);  getitem_191 = None
    rsqrt_70: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_308);  add_308 = None
    sub_87: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_185, getitem_192);  clone_185 = getitem_192 = None
    mul_418: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_70);  sub_87 = None
    mul_419: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_418, primals_447)
    add_309: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_419, primals_448);  mul_419 = primals_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_306: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_309, [6272, 768]);  add_309 = None
    permute_154: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_449, [1, 0]);  primals_449 = None
    addmm_49: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_450, view_306, permute_154);  primals_450 = None
    view_307: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_49, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_420: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_307, 0.5)
    mul_421: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_307, 0.7071067811865476);  view_307 = None
    erf_35: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_421);  mul_421 = None
    add_310: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_422: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_420, add_310);  mul_420 = add_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_308: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_422, [6272, 3072]);  mul_422 = None
    permute_155: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_451, [1, 0]);  primals_451 = None
    addmm_50: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_452, view_308, permute_155);  primals_452 = None
    view_309: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_50, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_423: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_68, view_309);  view_309 = None
    add_311: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_307, mul_423);  add_307 = mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_188: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_311, memory_format = torch.contiguous_format)
    var_mean_71 = torch.ops.aten.var_mean.correction(clone_188, [2], correction = 0, keepdim = True)
    getitem_193: "f32[8, 784, 1]" = var_mean_71[0]
    getitem_194: "f32[8, 784, 1]" = var_mean_71[1];  var_mean_71 = None
    add_312: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_193, 1e-06);  getitem_193 = None
    rsqrt_71: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_312);  add_312 = None
    sub_88: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_188, getitem_194);  clone_188 = getitem_194 = None
    mul_424: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_71);  sub_88 = None
    mul_425: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_424, primals_453)
    add_313: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_425, primals_454);  mul_425 = primals_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_310: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_313, [6272, 768]);  add_313 = None
    permute_156: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_455, [1, 0]);  primals_455 = None
    addmm_51: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_456, view_310, permute_156);  primals_456 = None
    view_311: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_51, [8, 784, 2304]);  addmm_51 = None
    view_312: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_311, [8, 784, 3, 16, 48]);  view_311 = None
    permute_157: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_312, [2, 0, 3, 4, 1]);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_17 = torch.ops.aten.unbind.int(permute_157);  permute_157 = None
    getitem_195: "f32[8, 16, 48, 784]" = unbind_17[0]
    getitem_196: "f32[8, 16, 48, 784]" = unbind_17[1]
    getitem_197: "f32[8, 16, 48, 784]" = unbind_17[2];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_70: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_195, 2.0)
    sum_52: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_70, [-1], True);  pow_70 = None
    pow_71: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_52, 0.5);  sum_52 = None
    clamp_min_34: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_71, 1e-12)
    expand_102: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_34, [8, 16, 48, 784]);  clamp_min_34 = None
    div_57: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_195, expand_102);  expand_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_72: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_196, 2.0)
    sum_53: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_72, [-1], True);  pow_72 = None
    pow_73: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_53, 0.5);  sum_53 = None
    clamp_min_35: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_73, 1e-12)
    expand_103: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_35, [8, 16, 48, 784]);  clamp_min_35 = None
    div_58: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_196, expand_103);  expand_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_158: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_58, [0, 1, 3, 2]);  div_58 = None
    expand_104: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_57, [8, 16, 48, 784]);  div_57 = None
    clone_189: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
    view_313: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_189, [128, 48, 784]);  clone_189 = None
    expand_105: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_158, [8, 16, 784, 48]);  permute_158 = None
    clone_190: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
    view_314: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_190, [128, 784, 48]);  clone_190 = None
    bmm_34: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_313, view_314)
    view_315: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_34, [8, 16, 48, 48])
    mul_426: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_315, primals_70);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_17: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_426, [-1], True)
    sub_89: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_426, amax_17);  mul_426 = amax_17 = None
    exp_17: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_89);  sub_89 = None
    sum_54: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_59: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_17, sum_54);  exp_17 = sum_54 = None
    alias_53: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_191: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_59);  div_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_106: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_191, [8, 16, 48, 48]);  clone_191 = None
    view_316: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_106, [128, 48, 48]);  expand_106 = None
    expand_107: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_197, [8, 16, 48, 784]);  getitem_197 = None
    clone_192: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
    view_317: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_192, [128, 48, 784]);  clone_192 = None
    bmm_35: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_316, view_317)
    view_318: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_35, [8, 16, 48, 784]);  bmm_35 = None
    permute_159: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_318, [0, 3, 1, 2]);  view_318 = None
    view_319: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_159, [8, 784, 768]);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_160: "f32[768, 768]" = torch.ops.aten.permute.default(primals_457, [1, 0]);  primals_457 = None
    clone_193: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_319, memory_format = torch.contiguous_format);  view_319 = None
    view_320: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_193, [6272, 768]);  clone_193 = None
    mm_17: "f32[6272, 768]" = torch.ops.aten.mm.default(view_320, permute_160)
    view_321: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_17, [8, 784, 768])
    add_314: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_321, primals_458);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_427: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_69, add_314);  add_314 = None
    add_315: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_311, mul_427);  add_311 = mul_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_195: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_315, memory_format = torch.contiguous_format)
    var_mean_72 = torch.ops.aten.var_mean.correction(clone_195, [2], correction = 0, keepdim = True)
    getitem_198: "f32[8, 784, 1]" = var_mean_72[0]
    getitem_199: "f32[8, 784, 1]" = var_mean_72[1];  var_mean_72 = None
    add_316: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-06);  getitem_198 = None
    rsqrt_72: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_316);  add_316 = None
    sub_90: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_195, getitem_199);  clone_195 = getitem_199 = None
    mul_428: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_72);  sub_90 = None
    mul_429: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_428, primals_459)
    add_317: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_429, primals_460);  mul_429 = primals_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_161: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_317, [0, 2, 1]);  add_317 = None
    view_322: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_161, [8, 768, 28, 28]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_38: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_322, primals_461, primals_462, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_430: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_38, 0.5)
    mul_431: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_38, 0.7071067811865476)
    erf_36: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_431);  mul_431 = None
    add_318: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_432: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_430, add_318);  mul_430 = add_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_319: "i64[]" = torch.ops.aten.add.Tensor(primals_691, 1)
    var_mean_73 = torch.ops.aten.var_mean.correction(mul_432, [0, 2, 3], correction = 0, keepdim = True)
    getitem_200: "f32[1, 768, 1, 1]" = var_mean_73[0]
    getitem_201: "f32[1, 768, 1, 1]" = var_mean_73[1];  var_mean_73 = None
    add_320: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-05)
    rsqrt_73: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_320);  add_320 = None
    sub_91: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_432, getitem_201);  mul_432 = None
    mul_433: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_73);  sub_91 = None
    squeeze_60: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_201, [0, 2, 3]);  getitem_201 = None
    squeeze_61: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
    mul_434: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_435: "f32[768]" = torch.ops.aten.mul.Tensor(primals_689, 0.9)
    add_321: "f32[768]" = torch.ops.aten.add.Tensor(mul_434, mul_435);  mul_434 = mul_435 = None
    squeeze_62: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_200, [0, 2, 3]);  getitem_200 = None
    mul_436: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
    mul_437: "f32[768]" = torch.ops.aten.mul.Tensor(mul_436, 0.1);  mul_436 = None
    mul_438: "f32[768]" = torch.ops.aten.mul.Tensor(primals_690, 0.9)
    add_322: "f32[768]" = torch.ops.aten.add.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
    unsqueeze_87: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_463, -1)
    unsqueeze_88: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_87, -1);  unsqueeze_87 = None
    mul_439: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_88);  mul_433 = unsqueeze_88 = None
    unsqueeze_89: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_464, -1);  primals_464 = None
    unsqueeze_90: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_89, -1);  unsqueeze_89 = None
    add_323: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_439, unsqueeze_90);  mul_439 = unsqueeze_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_39: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_323, primals_465, primals_466, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_323: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_39, [8, 768, 784])
    permute_162: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_440: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_71, permute_162);  permute_162 = None
    add_324: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_315, mul_440);  add_315 = mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_196: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_324, memory_format = torch.contiguous_format)
    var_mean_74 = torch.ops.aten.var_mean.correction(clone_196, [2], correction = 0, keepdim = True)
    getitem_202: "f32[8, 784, 1]" = var_mean_74[0]
    getitem_203: "f32[8, 784, 1]" = var_mean_74[1];  var_mean_74 = None
    add_325: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_202, 1e-06);  getitem_202 = None
    rsqrt_74: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_325);  add_325 = None
    sub_92: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_196, getitem_203);  clone_196 = getitem_203 = None
    mul_441: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_74);  sub_92 = None
    mul_442: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_441, primals_467)
    add_326: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_442, primals_468);  mul_442 = primals_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_324: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_326, [6272, 768]);  add_326 = None
    permute_163: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_469, [1, 0]);  primals_469 = None
    addmm_52: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_470, view_324, permute_163);  primals_470 = None
    view_325: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_52, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_443: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_325, 0.5)
    mul_444: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_325, 0.7071067811865476);  view_325 = None
    erf_37: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_444);  mul_444 = None
    add_327: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_445: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_443, add_327);  mul_443 = add_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_326: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_445, [6272, 3072]);  mul_445 = None
    permute_164: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_471, [1, 0]);  primals_471 = None
    addmm_53: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_472, view_326, permute_164);  primals_472 = None
    view_327: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_53, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_446: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_72, view_327);  view_327 = None
    add_328: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_324, mul_446);  add_324 = mul_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_199: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_328, memory_format = torch.contiguous_format)
    var_mean_75 = torch.ops.aten.var_mean.correction(clone_199, [2], correction = 0, keepdim = True)
    getitem_204: "f32[8, 784, 1]" = var_mean_75[0]
    getitem_205: "f32[8, 784, 1]" = var_mean_75[1];  var_mean_75 = None
    add_329: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_204, 1e-06);  getitem_204 = None
    rsqrt_75: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_329);  add_329 = None
    sub_93: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_199, getitem_205);  clone_199 = getitem_205 = None
    mul_447: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_75);  sub_93 = None
    mul_448: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_447, primals_473)
    add_330: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_448, primals_474);  mul_448 = primals_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_328: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_330, [6272, 768]);  add_330 = None
    permute_165: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_475, [1, 0]);  primals_475 = None
    addmm_54: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_476, view_328, permute_165);  primals_476 = None
    view_329: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_54, [8, 784, 2304]);  addmm_54 = None
    view_330: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_329, [8, 784, 3, 16, 48]);  view_329 = None
    permute_166: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_330, [2, 0, 3, 4, 1]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_18 = torch.ops.aten.unbind.int(permute_166);  permute_166 = None
    getitem_206: "f32[8, 16, 48, 784]" = unbind_18[0]
    getitem_207: "f32[8, 16, 48, 784]" = unbind_18[1]
    getitem_208: "f32[8, 16, 48, 784]" = unbind_18[2];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_74: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_206, 2.0)
    sum_55: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_74, [-1], True);  pow_74 = None
    pow_75: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_55, 0.5);  sum_55 = None
    clamp_min_36: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_75, 1e-12)
    expand_108: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_36, [8, 16, 48, 784]);  clamp_min_36 = None
    div_60: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_206, expand_108);  expand_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_76: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_207, 2.0)
    sum_56: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_76, [-1], True);  pow_76 = None
    pow_77: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_56, 0.5);  sum_56 = None
    clamp_min_37: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_77, 1e-12)
    expand_109: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_37, [8, 16, 48, 784]);  clamp_min_37 = None
    div_61: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_207, expand_109);  expand_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_167: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_61, [0, 1, 3, 2]);  div_61 = None
    expand_110: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_60, [8, 16, 48, 784]);  div_60 = None
    clone_200: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_110, memory_format = torch.contiguous_format);  expand_110 = None
    view_331: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_200, [128, 48, 784]);  clone_200 = None
    expand_111: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_167, [8, 16, 784, 48]);  permute_167 = None
    clone_201: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_111, memory_format = torch.contiguous_format);  expand_111 = None
    view_332: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_201, [128, 784, 48]);  clone_201 = None
    bmm_36: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_331, view_332)
    view_333: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_36, [8, 16, 48, 48])
    mul_449: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_333, primals_74);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_18: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_449, [-1], True)
    sub_94: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_449, amax_18);  mul_449 = amax_18 = None
    exp_18: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_94);  sub_94 = None
    sum_57: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_62: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_18, sum_57);  exp_18 = sum_57 = None
    alias_56: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_202: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_62);  div_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_112: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_202, [8, 16, 48, 48]);  clone_202 = None
    view_334: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_112, [128, 48, 48]);  expand_112 = None
    expand_113: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_208, [8, 16, 48, 784]);  getitem_208 = None
    clone_203: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
    view_335: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_203, [128, 48, 784]);  clone_203 = None
    bmm_37: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_334, view_335)
    view_336: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_37, [8, 16, 48, 784]);  bmm_37 = None
    permute_168: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_336, [0, 3, 1, 2]);  view_336 = None
    view_337: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_168, [8, 784, 768]);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(primals_477, [1, 0]);  primals_477 = None
    clone_204: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_337, memory_format = torch.contiguous_format);  view_337 = None
    view_338: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_204, [6272, 768]);  clone_204 = None
    mm_18: "f32[6272, 768]" = torch.ops.aten.mm.default(view_338, permute_169)
    view_339: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_18, [8, 784, 768])
    add_331: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_339, primals_478);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_450: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_73, add_331);  add_331 = None
    add_332: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_328, mul_450);  add_328 = mul_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_206: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_332, memory_format = torch.contiguous_format)
    var_mean_76 = torch.ops.aten.var_mean.correction(clone_206, [2], correction = 0, keepdim = True)
    getitem_209: "f32[8, 784, 1]" = var_mean_76[0]
    getitem_210: "f32[8, 784, 1]" = var_mean_76[1];  var_mean_76 = None
    add_333: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_209, 1e-06);  getitem_209 = None
    rsqrt_76: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_333);  add_333 = None
    sub_95: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_206, getitem_210);  clone_206 = getitem_210 = None
    mul_451: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_76);  sub_95 = None
    mul_452: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_451, primals_479)
    add_334: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_452, primals_480);  mul_452 = primals_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_170: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_334, [0, 2, 1]);  add_334 = None
    view_340: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_170, [8, 768, 28, 28]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_40: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_340, primals_481, primals_482, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_453: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_40, 0.5)
    mul_454: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_40, 0.7071067811865476)
    erf_38: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_454);  mul_454 = None
    add_335: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    mul_455: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_453, add_335);  mul_453 = add_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_336: "i64[]" = torch.ops.aten.add.Tensor(primals_694, 1)
    var_mean_77 = torch.ops.aten.var_mean.correction(mul_455, [0, 2, 3], correction = 0, keepdim = True)
    getitem_211: "f32[1, 768, 1, 1]" = var_mean_77[0]
    getitem_212: "f32[1, 768, 1, 1]" = var_mean_77[1];  var_mean_77 = None
    add_337: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_211, 1e-05)
    rsqrt_77: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_337);  add_337 = None
    sub_96: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_455, getitem_212);  mul_455 = None
    mul_456: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_77);  sub_96 = None
    squeeze_63: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_212, [0, 2, 3]);  getitem_212 = None
    squeeze_64: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
    mul_457: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_458: "f32[768]" = torch.ops.aten.mul.Tensor(primals_692, 0.9)
    add_338: "f32[768]" = torch.ops.aten.add.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
    squeeze_65: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_211, [0, 2, 3]);  getitem_211 = None
    mul_459: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
    mul_460: "f32[768]" = torch.ops.aten.mul.Tensor(mul_459, 0.1);  mul_459 = None
    mul_461: "f32[768]" = torch.ops.aten.mul.Tensor(primals_693, 0.9)
    add_339: "f32[768]" = torch.ops.aten.add.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    unsqueeze_91: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_483, -1)
    unsqueeze_92: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_91, -1);  unsqueeze_91 = None
    mul_462: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_456, unsqueeze_92);  mul_456 = unsqueeze_92 = None
    unsqueeze_93: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_484, -1);  primals_484 = None
    unsqueeze_94: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, -1);  unsqueeze_93 = None
    add_340: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_462, unsqueeze_94);  mul_462 = unsqueeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_41: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_340, primals_485, primals_486, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_341: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_41, [8, 768, 784])
    permute_171: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_341, [0, 2, 1]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_463: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_75, permute_171);  permute_171 = None
    add_341: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_332, mul_463);  add_332 = mul_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_207: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_341, memory_format = torch.contiguous_format)
    var_mean_78 = torch.ops.aten.var_mean.correction(clone_207, [2], correction = 0, keepdim = True)
    getitem_213: "f32[8, 784, 1]" = var_mean_78[0]
    getitem_214: "f32[8, 784, 1]" = var_mean_78[1];  var_mean_78 = None
    add_342: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_213, 1e-06);  getitem_213 = None
    rsqrt_78: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_342);  add_342 = None
    sub_97: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_207, getitem_214);  clone_207 = getitem_214 = None
    mul_464: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_78);  sub_97 = None
    mul_465: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_464, primals_487)
    add_343: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_465, primals_488);  mul_465 = primals_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_342: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_343, [6272, 768]);  add_343 = None
    permute_172: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_489, [1, 0]);  primals_489 = None
    addmm_55: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_490, view_342, permute_172);  primals_490 = None
    view_343: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_55, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_466: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_343, 0.5)
    mul_467: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_343, 0.7071067811865476);  view_343 = None
    erf_39: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_467);  mul_467 = None
    add_344: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    mul_468: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_466, add_344);  mul_466 = add_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_344: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_468, [6272, 3072]);  mul_468 = None
    permute_173: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_491, [1, 0]);  primals_491 = None
    addmm_56: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_492, view_344, permute_173);  primals_492 = None
    view_345: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_56, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_469: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_76, view_345);  view_345 = None
    add_345: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_341, mul_469);  add_341 = mul_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_210: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_345, memory_format = torch.contiguous_format)
    var_mean_79 = torch.ops.aten.var_mean.correction(clone_210, [2], correction = 0, keepdim = True)
    getitem_215: "f32[8, 784, 1]" = var_mean_79[0]
    getitem_216: "f32[8, 784, 1]" = var_mean_79[1];  var_mean_79 = None
    add_346: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_215, 1e-06);  getitem_215 = None
    rsqrt_79: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_346);  add_346 = None
    sub_98: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_210, getitem_216);  clone_210 = getitem_216 = None
    mul_470: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_79);  sub_98 = None
    mul_471: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_470, primals_493)
    add_347: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_471, primals_494);  mul_471 = primals_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_346: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_347, [6272, 768]);  add_347 = None
    permute_174: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_495, [1, 0]);  primals_495 = None
    addmm_57: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_496, view_346, permute_174);  primals_496 = None
    view_347: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_57, [8, 784, 2304]);  addmm_57 = None
    view_348: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_347, [8, 784, 3, 16, 48]);  view_347 = None
    permute_175: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_348, [2, 0, 3, 4, 1]);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_19 = torch.ops.aten.unbind.int(permute_175);  permute_175 = None
    getitem_217: "f32[8, 16, 48, 784]" = unbind_19[0]
    getitem_218: "f32[8, 16, 48, 784]" = unbind_19[1]
    getitem_219: "f32[8, 16, 48, 784]" = unbind_19[2];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_78: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_217, 2.0)
    sum_58: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_78, [-1], True);  pow_78 = None
    pow_79: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_58, 0.5);  sum_58 = None
    clamp_min_38: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_79, 1e-12)
    expand_114: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_38, [8, 16, 48, 784]);  clamp_min_38 = None
    div_63: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_217, expand_114);  expand_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_80: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_218, 2.0)
    sum_59: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_80, [-1], True);  pow_80 = None
    pow_81: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_59, 0.5);  sum_59 = None
    clamp_min_39: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_81, 1e-12)
    expand_115: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_39, [8, 16, 48, 784]);  clamp_min_39 = None
    div_64: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_218, expand_115);  expand_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_176: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_64, [0, 1, 3, 2]);  div_64 = None
    expand_116: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_63, [8, 16, 48, 784]);  div_63 = None
    clone_211: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_116, memory_format = torch.contiguous_format);  expand_116 = None
    view_349: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_211, [128, 48, 784]);  clone_211 = None
    expand_117: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_176, [8, 16, 784, 48]);  permute_176 = None
    clone_212: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
    view_350: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_212, [128, 784, 48]);  clone_212 = None
    bmm_38: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_349, view_350)
    view_351: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_38, [8, 16, 48, 48])
    mul_472: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_351, primals_78);  view_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_19: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_472, [-1], True)
    sub_99: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_472, amax_19);  mul_472 = amax_19 = None
    exp_19: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_99);  sub_99 = None
    sum_60: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_65: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_19, sum_60);  exp_19 = sum_60 = None
    alias_59: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_213: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_65);  div_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_118: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_213, [8, 16, 48, 48]);  clone_213 = None
    view_352: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_118, [128, 48, 48]);  expand_118 = None
    expand_119: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_219, [8, 16, 48, 784]);  getitem_219 = None
    clone_214: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_119, memory_format = torch.contiguous_format);  expand_119 = None
    view_353: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_214, [128, 48, 784]);  clone_214 = None
    bmm_39: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_352, view_353)
    view_354: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_39, [8, 16, 48, 784]);  bmm_39 = None
    permute_177: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_354, [0, 3, 1, 2]);  view_354 = None
    view_355: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_177, [8, 784, 768]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_178: "f32[768, 768]" = torch.ops.aten.permute.default(primals_497, [1, 0]);  primals_497 = None
    clone_215: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_355, memory_format = torch.contiguous_format);  view_355 = None
    view_356: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_215, [6272, 768]);  clone_215 = None
    mm_19: "f32[6272, 768]" = torch.ops.aten.mm.default(view_356, permute_178)
    view_357: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_19, [8, 784, 768])
    add_348: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_357, primals_498);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_473: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_77, add_348);  add_348 = None
    add_349: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_345, mul_473);  add_345 = mul_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_217: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_349, memory_format = torch.contiguous_format)
    var_mean_80 = torch.ops.aten.var_mean.correction(clone_217, [2], correction = 0, keepdim = True)
    getitem_220: "f32[8, 784, 1]" = var_mean_80[0]
    getitem_221: "f32[8, 784, 1]" = var_mean_80[1];  var_mean_80 = None
    add_350: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-06);  getitem_220 = None
    rsqrt_80: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_350);  add_350 = None
    sub_100: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_217, getitem_221);  clone_217 = getitem_221 = None
    mul_474: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_80);  sub_100 = None
    mul_475: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_474, primals_499)
    add_351: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_475, primals_500);  mul_475 = primals_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_179: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_351, [0, 2, 1]);  add_351 = None
    view_358: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_179, [8, 768, 28, 28]);  permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_42: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_358, primals_501, primals_502, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_476: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_42, 0.5)
    mul_477: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_42, 0.7071067811865476)
    erf_40: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_477);  mul_477 = None
    add_352: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
    mul_478: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_476, add_352);  mul_476 = add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_353: "i64[]" = torch.ops.aten.add.Tensor(primals_697, 1)
    var_mean_81 = torch.ops.aten.var_mean.correction(mul_478, [0, 2, 3], correction = 0, keepdim = True)
    getitem_222: "f32[1, 768, 1, 1]" = var_mean_81[0]
    getitem_223: "f32[1, 768, 1, 1]" = var_mean_81[1];  var_mean_81 = None
    add_354: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-05)
    rsqrt_81: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_354);  add_354 = None
    sub_101: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_478, getitem_223);  mul_478 = None
    mul_479: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_81);  sub_101 = None
    squeeze_66: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_223, [0, 2, 3]);  getitem_223 = None
    squeeze_67: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
    mul_480: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_481: "f32[768]" = torch.ops.aten.mul.Tensor(primals_695, 0.9)
    add_355: "f32[768]" = torch.ops.aten.add.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
    squeeze_68: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_222, [0, 2, 3]);  getitem_222 = None
    mul_482: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001594642002871);  squeeze_68 = None
    mul_483: "f32[768]" = torch.ops.aten.mul.Tensor(mul_482, 0.1);  mul_482 = None
    mul_484: "f32[768]" = torch.ops.aten.mul.Tensor(primals_696, 0.9)
    add_356: "f32[768]" = torch.ops.aten.add.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
    unsqueeze_95: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_503, -1)
    unsqueeze_96: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, -1);  unsqueeze_95 = None
    mul_485: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_479, unsqueeze_96);  mul_479 = unsqueeze_96 = None
    unsqueeze_97: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_504, -1);  primals_504 = None
    unsqueeze_98: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_97, -1);  unsqueeze_97 = None
    add_357: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_485, unsqueeze_98);  mul_485 = unsqueeze_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_43: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_357, primals_505, primals_506, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_359: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_43, [8, 768, 784])
    permute_180: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_359, [0, 2, 1]);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_486: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_79, permute_180);  permute_180 = None
    add_358: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_349, mul_486);  add_349 = mul_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_218: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_358, memory_format = torch.contiguous_format)
    var_mean_82 = torch.ops.aten.var_mean.correction(clone_218, [2], correction = 0, keepdim = True)
    getitem_224: "f32[8, 784, 1]" = var_mean_82[0]
    getitem_225: "f32[8, 784, 1]" = var_mean_82[1];  var_mean_82 = None
    add_359: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-06);  getitem_224 = None
    rsqrt_82: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_359);  add_359 = None
    sub_102: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_218, getitem_225);  clone_218 = getitem_225 = None
    mul_487: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_82);  sub_102 = None
    mul_488: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_487, primals_507)
    add_360: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_488, primals_508);  mul_488 = primals_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_360: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_360, [6272, 768]);  add_360 = None
    permute_181: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_509, [1, 0]);  primals_509 = None
    addmm_58: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_510, view_360, permute_181);  primals_510 = None
    view_361: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_58, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_489: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_361, 0.5)
    mul_490: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_361, 0.7071067811865476);  view_361 = None
    erf_41: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_490);  mul_490 = None
    add_361: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
    mul_491: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_489, add_361);  mul_489 = add_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_362: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_491, [6272, 3072]);  mul_491 = None
    permute_182: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_511, [1, 0]);  primals_511 = None
    addmm_59: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_512, view_362, permute_182);  primals_512 = None
    view_363: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_59, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_492: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_80, view_363);  view_363 = None
    add_362: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_358, mul_492);  add_358 = mul_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_221: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_362, memory_format = torch.contiguous_format)
    var_mean_83 = torch.ops.aten.var_mean.correction(clone_221, [2], correction = 0, keepdim = True)
    getitem_226: "f32[8, 784, 1]" = var_mean_83[0]
    getitem_227: "f32[8, 784, 1]" = var_mean_83[1];  var_mean_83 = None
    add_363: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-06);  getitem_226 = None
    rsqrt_83: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_363);  add_363 = None
    sub_103: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_221, getitem_227);  clone_221 = getitem_227 = None
    mul_493: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_83);  sub_103 = None
    mul_494: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_493, primals_513)
    add_364: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_494, primals_514);  mul_494 = primals_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_364: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_364, [6272, 768]);  add_364 = None
    permute_183: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_515, [1, 0]);  primals_515 = None
    addmm_60: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_516, view_364, permute_183);  primals_516 = None
    view_365: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_60, [8, 784, 2304]);  addmm_60 = None
    view_366: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_365, [8, 784, 3, 16, 48]);  view_365 = None
    permute_184: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_366, [2, 0, 3, 4, 1]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_20 = torch.ops.aten.unbind.int(permute_184);  permute_184 = None
    getitem_228: "f32[8, 16, 48, 784]" = unbind_20[0]
    getitem_229: "f32[8, 16, 48, 784]" = unbind_20[1]
    getitem_230: "f32[8, 16, 48, 784]" = unbind_20[2];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_82: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_228, 2.0)
    sum_61: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_82, [-1], True);  pow_82 = None
    pow_83: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_61, 0.5);  sum_61 = None
    clamp_min_40: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_83, 1e-12)
    expand_120: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_40, [8, 16, 48, 784]);  clamp_min_40 = None
    div_66: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_228, expand_120);  expand_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_84: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_229, 2.0)
    sum_62: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_84, [-1], True);  pow_84 = None
    pow_85: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_62, 0.5);  sum_62 = None
    clamp_min_41: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_85, 1e-12)
    expand_121: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_41, [8, 16, 48, 784]);  clamp_min_41 = None
    div_67: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_229, expand_121);  expand_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_185: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_67, [0, 1, 3, 2]);  div_67 = None
    expand_122: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_66, [8, 16, 48, 784]);  div_66 = None
    clone_222: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
    view_367: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_222, [128, 48, 784]);  clone_222 = None
    expand_123: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_185, [8, 16, 784, 48]);  permute_185 = None
    clone_223: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
    view_368: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_223, [128, 784, 48]);  clone_223 = None
    bmm_40: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_367, view_368)
    view_369: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_40, [8, 16, 48, 48])
    mul_495: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_369, primals_82);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_20: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_495, [-1], True)
    sub_104: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_495, amax_20);  mul_495 = amax_20 = None
    exp_20: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_104);  sub_104 = None
    sum_63: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_68: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_20, sum_63);  exp_20 = sum_63 = None
    alias_62: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_224: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_68);  div_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_124: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_224, [8, 16, 48, 48]);  clone_224 = None
    view_370: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_124, [128, 48, 48]);  expand_124 = None
    expand_125: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_230, [8, 16, 48, 784]);  getitem_230 = None
    clone_225: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
    view_371: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_225, [128, 48, 784]);  clone_225 = None
    bmm_41: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_370, view_371)
    view_372: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_41, [8, 16, 48, 784]);  bmm_41 = None
    permute_186: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_372, [0, 3, 1, 2]);  view_372 = None
    view_373: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_186, [8, 784, 768]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(primals_517, [1, 0]);  primals_517 = None
    clone_226: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_373, memory_format = torch.contiguous_format);  view_373 = None
    view_374: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_226, [6272, 768]);  clone_226 = None
    mm_20: "f32[6272, 768]" = torch.ops.aten.mm.default(view_374, permute_187)
    view_375: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_20, [8, 784, 768])
    add_365: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_375, primals_518);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_496: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_81, add_365);  add_365 = None
    add_366: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_362, mul_496);  add_362 = mul_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_228: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_366, memory_format = torch.contiguous_format)
    var_mean_84 = torch.ops.aten.var_mean.correction(clone_228, [2], correction = 0, keepdim = True)
    getitem_231: "f32[8, 784, 1]" = var_mean_84[0]
    getitem_232: "f32[8, 784, 1]" = var_mean_84[1];  var_mean_84 = None
    add_367: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_231, 1e-06);  getitem_231 = None
    rsqrt_84: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_367);  add_367 = None
    sub_105: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_228, getitem_232);  clone_228 = getitem_232 = None
    mul_497: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_84);  sub_105 = None
    mul_498: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_497, primals_519)
    add_368: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_498, primals_520);  mul_498 = primals_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_188: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_368, [0, 2, 1]);  add_368 = None
    view_376: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_188, [8, 768, 28, 28]);  permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_44: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_376, primals_521, primals_522, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_499: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_44, 0.5)
    mul_500: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_44, 0.7071067811865476)
    erf_42: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_500);  mul_500 = None
    add_369: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
    mul_501: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_499, add_369);  mul_499 = add_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_370: "i64[]" = torch.ops.aten.add.Tensor(primals_700, 1)
    var_mean_85 = torch.ops.aten.var_mean.correction(mul_501, [0, 2, 3], correction = 0, keepdim = True)
    getitem_233: "f32[1, 768, 1, 1]" = var_mean_85[0]
    getitem_234: "f32[1, 768, 1, 1]" = var_mean_85[1];  var_mean_85 = None
    add_371: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_233, 1e-05)
    rsqrt_85: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_371);  add_371 = None
    sub_106: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_501, getitem_234);  mul_501 = None
    mul_502: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_85);  sub_106 = None
    squeeze_69: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_234, [0, 2, 3]);  getitem_234 = None
    squeeze_70: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
    mul_503: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_504: "f32[768]" = torch.ops.aten.mul.Tensor(primals_698, 0.9)
    add_372: "f32[768]" = torch.ops.aten.add.Tensor(mul_503, mul_504);  mul_503 = mul_504 = None
    squeeze_71: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_233, [0, 2, 3]);  getitem_233 = None
    mul_505: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001594642002871);  squeeze_71 = None
    mul_506: "f32[768]" = torch.ops.aten.mul.Tensor(mul_505, 0.1);  mul_505 = None
    mul_507: "f32[768]" = torch.ops.aten.mul.Tensor(primals_699, 0.9)
    add_373: "f32[768]" = torch.ops.aten.add.Tensor(mul_506, mul_507);  mul_506 = mul_507 = None
    unsqueeze_99: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_523, -1)
    unsqueeze_100: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, -1);  unsqueeze_99 = None
    mul_508: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_502, unsqueeze_100);  mul_502 = unsqueeze_100 = None
    unsqueeze_101: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_524, -1);  primals_524 = None
    unsqueeze_102: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, -1);  unsqueeze_101 = None
    add_374: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_508, unsqueeze_102);  mul_508 = unsqueeze_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_45: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_374, primals_525, primals_526, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_377: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_45, [8, 768, 784])
    permute_189: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_377, [0, 2, 1]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_509: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_83, permute_189);  permute_189 = None
    add_375: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_366, mul_509);  add_366 = mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_229: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_375, memory_format = torch.contiguous_format)
    var_mean_86 = torch.ops.aten.var_mean.correction(clone_229, [2], correction = 0, keepdim = True)
    getitem_235: "f32[8, 784, 1]" = var_mean_86[0]
    getitem_236: "f32[8, 784, 1]" = var_mean_86[1];  var_mean_86 = None
    add_376: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_235, 1e-06);  getitem_235 = None
    rsqrt_86: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
    sub_107: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_229, getitem_236);  clone_229 = getitem_236 = None
    mul_510: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_86);  sub_107 = None
    mul_511: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_510, primals_527)
    add_377: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_511, primals_528);  mul_511 = primals_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_378: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_377, [6272, 768]);  add_377 = None
    permute_190: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_529, [1, 0]);  primals_529 = None
    addmm_61: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_530, view_378, permute_190);  primals_530 = None
    view_379: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_61, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_512: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_379, 0.5)
    mul_513: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_379, 0.7071067811865476);  view_379 = None
    erf_43: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_513);  mul_513 = None
    add_378: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
    mul_514: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_512, add_378);  mul_512 = add_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_380: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_514, [6272, 3072]);  mul_514 = None
    permute_191: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_531, [1, 0]);  primals_531 = None
    addmm_62: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_532, view_380, permute_191);  primals_532 = None
    view_381: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_62, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_515: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_84, view_381);  view_381 = None
    add_379: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_375, mul_515);  add_375 = mul_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_232: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_379, memory_format = torch.contiguous_format)
    var_mean_87 = torch.ops.aten.var_mean.correction(clone_232, [2], correction = 0, keepdim = True)
    getitem_237: "f32[8, 784, 1]" = var_mean_87[0]
    getitem_238: "f32[8, 784, 1]" = var_mean_87[1];  var_mean_87 = None
    add_380: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_237, 1e-06);  getitem_237 = None
    rsqrt_87: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_380);  add_380 = None
    sub_108: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_232, getitem_238);  clone_232 = getitem_238 = None
    mul_516: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_87);  sub_108 = None
    mul_517: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_516, primals_533)
    add_381: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_517, primals_534);  mul_517 = primals_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_382: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_381, [6272, 768]);  add_381 = None
    permute_192: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_535, [1, 0]);  primals_535 = None
    addmm_63: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_536, view_382, permute_192);  primals_536 = None
    view_383: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_63, [8, 784, 2304]);  addmm_63 = None
    view_384: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_383, [8, 784, 3, 16, 48]);  view_383 = None
    permute_193: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_384, [2, 0, 3, 4, 1]);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_21 = torch.ops.aten.unbind.int(permute_193);  permute_193 = None
    getitem_239: "f32[8, 16, 48, 784]" = unbind_21[0]
    getitem_240: "f32[8, 16, 48, 784]" = unbind_21[1]
    getitem_241: "f32[8, 16, 48, 784]" = unbind_21[2];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_86: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_239, 2.0)
    sum_64: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_86, [-1], True);  pow_86 = None
    pow_87: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_64, 0.5);  sum_64 = None
    clamp_min_42: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_87, 1e-12)
    expand_126: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_42, [8, 16, 48, 784]);  clamp_min_42 = None
    div_69: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_239, expand_126);  expand_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_88: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_240, 2.0)
    sum_65: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_88, [-1], True);  pow_88 = None
    pow_89: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_65, 0.5);  sum_65 = None
    clamp_min_43: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_89, 1e-12)
    expand_127: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_43, [8, 16, 48, 784]);  clamp_min_43 = None
    div_70: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_240, expand_127);  expand_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_194: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_70, [0, 1, 3, 2]);  div_70 = None
    expand_128: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_69, [8, 16, 48, 784]);  div_69 = None
    clone_233: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
    view_385: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_233, [128, 48, 784]);  clone_233 = None
    expand_129: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_194, [8, 16, 784, 48]);  permute_194 = None
    clone_234: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
    view_386: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_234, [128, 784, 48]);  clone_234 = None
    bmm_42: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_385, view_386)
    view_387: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_42, [8, 16, 48, 48])
    mul_518: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_387, primals_86);  view_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_21: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_518, [-1], True)
    sub_109: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_518, amax_21);  mul_518 = amax_21 = None
    exp_21: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_109);  sub_109 = None
    sum_66: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_71: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_21, sum_66);  exp_21 = sum_66 = None
    alias_65: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_235: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_71);  div_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_130: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_235, [8, 16, 48, 48]);  clone_235 = None
    view_388: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_130, [128, 48, 48]);  expand_130 = None
    expand_131: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_241, [8, 16, 48, 784]);  getitem_241 = None
    clone_236: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
    view_389: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_236, [128, 48, 784]);  clone_236 = None
    bmm_43: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_388, view_389)
    view_390: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_43, [8, 16, 48, 784]);  bmm_43 = None
    permute_195: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_390, [0, 3, 1, 2]);  view_390 = None
    view_391: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_195, [8, 784, 768]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_196: "f32[768, 768]" = torch.ops.aten.permute.default(primals_537, [1, 0]);  primals_537 = None
    clone_237: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_391, memory_format = torch.contiguous_format);  view_391 = None
    view_392: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_237, [6272, 768]);  clone_237 = None
    mm_21: "f32[6272, 768]" = torch.ops.aten.mm.default(view_392, permute_196)
    view_393: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_21, [8, 784, 768])
    add_382: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_393, primals_538);  view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_519: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_85, add_382);  add_382 = None
    add_383: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_379, mul_519);  add_379 = mul_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_239: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_383, memory_format = torch.contiguous_format)
    var_mean_88 = torch.ops.aten.var_mean.correction(clone_239, [2], correction = 0, keepdim = True)
    getitem_242: "f32[8, 784, 1]" = var_mean_88[0]
    getitem_243: "f32[8, 784, 1]" = var_mean_88[1];  var_mean_88 = None
    add_384: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-06);  getitem_242 = None
    rsqrt_88: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_384);  add_384 = None
    sub_110: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_239, getitem_243);  clone_239 = getitem_243 = None
    mul_520: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_88);  sub_110 = None
    mul_521: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_520, primals_539)
    add_385: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_521, primals_540);  mul_521 = primals_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_197: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_385, [0, 2, 1]);  add_385 = None
    view_394: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_197, [8, 768, 28, 28]);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_46: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_394, primals_541, primals_542, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_522: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_46, 0.5)
    mul_523: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_46, 0.7071067811865476)
    erf_44: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_523);  mul_523 = None
    add_386: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
    mul_524: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_522, add_386);  mul_522 = add_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_387: "i64[]" = torch.ops.aten.add.Tensor(primals_703, 1)
    var_mean_89 = torch.ops.aten.var_mean.correction(mul_524, [0, 2, 3], correction = 0, keepdim = True)
    getitem_244: "f32[1, 768, 1, 1]" = var_mean_89[0]
    getitem_245: "f32[1, 768, 1, 1]" = var_mean_89[1];  var_mean_89 = None
    add_388: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_244, 1e-05)
    rsqrt_89: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_388);  add_388 = None
    sub_111: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_524, getitem_245);  mul_524 = None
    mul_525: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_89);  sub_111 = None
    squeeze_72: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_245, [0, 2, 3]);  getitem_245 = None
    squeeze_73: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_89, [0, 2, 3]);  rsqrt_89 = None
    mul_526: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_527: "f32[768]" = torch.ops.aten.mul.Tensor(primals_701, 0.9)
    add_389: "f32[768]" = torch.ops.aten.add.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
    squeeze_74: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_244, [0, 2, 3]);  getitem_244 = None
    mul_528: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001594642002871);  squeeze_74 = None
    mul_529: "f32[768]" = torch.ops.aten.mul.Tensor(mul_528, 0.1);  mul_528 = None
    mul_530: "f32[768]" = torch.ops.aten.mul.Tensor(primals_702, 0.9)
    add_390: "f32[768]" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    unsqueeze_103: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_543, -1)
    unsqueeze_104: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, -1);  unsqueeze_103 = None
    mul_531: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_104);  mul_525 = unsqueeze_104 = None
    unsqueeze_105: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_544, -1);  primals_544 = None
    unsqueeze_106: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_105, -1);  unsqueeze_105 = None
    add_391: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_531, unsqueeze_106);  mul_531 = unsqueeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_47: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_391, primals_545, primals_546, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_395: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_47, [8, 768, 784])
    permute_198: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_395, [0, 2, 1]);  view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_532: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_87, permute_198);  permute_198 = None
    add_392: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_383, mul_532);  add_383 = mul_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_240: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_392, memory_format = torch.contiguous_format)
    var_mean_90 = torch.ops.aten.var_mean.correction(clone_240, [2], correction = 0, keepdim = True)
    getitem_246: "f32[8, 784, 1]" = var_mean_90[0]
    getitem_247: "f32[8, 784, 1]" = var_mean_90[1];  var_mean_90 = None
    add_393: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-06);  getitem_246 = None
    rsqrt_90: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_393);  add_393 = None
    sub_112: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_240, getitem_247);  clone_240 = getitem_247 = None
    mul_533: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_90);  sub_112 = None
    mul_534: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_533, primals_547)
    add_394: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_534, primals_548);  mul_534 = primals_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_396: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_394, [6272, 768]);  add_394 = None
    permute_199: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_549, [1, 0]);  primals_549 = None
    addmm_64: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_550, view_396, permute_199);  primals_550 = None
    view_397: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_64, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_535: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_397, 0.5)
    mul_536: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_397, 0.7071067811865476);  view_397 = None
    erf_45: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_536);  mul_536 = None
    add_395: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
    mul_537: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_535, add_395);  mul_535 = add_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_398: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_537, [6272, 3072]);  mul_537 = None
    permute_200: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_551, [1, 0]);  primals_551 = None
    addmm_65: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_552, view_398, permute_200);  primals_552 = None
    view_399: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_65, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_538: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_88, view_399);  view_399 = None
    add_396: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_392, mul_538);  add_392 = mul_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_243: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_396, memory_format = torch.contiguous_format)
    var_mean_91 = torch.ops.aten.var_mean.correction(clone_243, [2], correction = 0, keepdim = True)
    getitem_248: "f32[8, 784, 1]" = var_mean_91[0]
    getitem_249: "f32[8, 784, 1]" = var_mean_91[1];  var_mean_91 = None
    add_397: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_248, 1e-06);  getitem_248 = None
    rsqrt_91: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_397);  add_397 = None
    sub_113: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_243, getitem_249);  clone_243 = getitem_249 = None
    mul_539: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_91);  sub_113 = None
    mul_540: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_539, primals_553)
    add_398: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_540, primals_554);  mul_540 = primals_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_400: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_398, [6272, 768]);  add_398 = None
    permute_201: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_555, [1, 0]);  primals_555 = None
    addmm_66: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_556, view_400, permute_201);  primals_556 = None
    view_401: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_66, [8, 784, 2304]);  addmm_66 = None
    view_402: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_401, [8, 784, 3, 16, 48]);  view_401 = None
    permute_202: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_402, [2, 0, 3, 4, 1]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_22 = torch.ops.aten.unbind.int(permute_202);  permute_202 = None
    getitem_250: "f32[8, 16, 48, 784]" = unbind_22[0]
    getitem_251: "f32[8, 16, 48, 784]" = unbind_22[1]
    getitem_252: "f32[8, 16, 48, 784]" = unbind_22[2];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_90: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_250, 2.0)
    sum_67: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_90, [-1], True);  pow_90 = None
    pow_91: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_67, 0.5);  sum_67 = None
    clamp_min_44: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_91, 1e-12)
    expand_132: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_44, [8, 16, 48, 784]);  clamp_min_44 = None
    div_72: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_250, expand_132);  expand_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_92: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_251, 2.0)
    sum_68: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_92, [-1], True);  pow_92 = None
    pow_93: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_68, 0.5);  sum_68 = None
    clamp_min_45: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_93, 1e-12)
    expand_133: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_45, [8, 16, 48, 784]);  clamp_min_45 = None
    div_73: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_251, expand_133);  expand_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_203: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_73, [0, 1, 3, 2]);  div_73 = None
    expand_134: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_72, [8, 16, 48, 784]);  div_72 = None
    clone_244: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_134, memory_format = torch.contiguous_format);  expand_134 = None
    view_403: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_244, [128, 48, 784]);  clone_244 = None
    expand_135: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_203, [8, 16, 784, 48]);  permute_203 = None
    clone_245: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_135, memory_format = torch.contiguous_format);  expand_135 = None
    view_404: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_245, [128, 784, 48]);  clone_245 = None
    bmm_44: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_403, view_404)
    view_405: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_44, [8, 16, 48, 48])
    mul_541: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_405, primals_90);  view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_22: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_541, [-1], True)
    sub_114: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_541, amax_22);  mul_541 = amax_22 = None
    exp_22: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_114);  sub_114 = None
    sum_69: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_74: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_22, sum_69);  exp_22 = sum_69 = None
    alias_68: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_74)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_246: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_74);  div_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_136: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_246, [8, 16, 48, 48]);  clone_246 = None
    view_406: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_136, [128, 48, 48]);  expand_136 = None
    expand_137: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_252, [8, 16, 48, 784]);  getitem_252 = None
    clone_247: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
    view_407: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_247, [128, 48, 784]);  clone_247 = None
    bmm_45: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_406, view_407)
    view_408: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_45, [8, 16, 48, 784]);  bmm_45 = None
    permute_204: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_408, [0, 3, 1, 2]);  view_408 = None
    view_409: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_204, [8, 784, 768]);  permute_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_205: "f32[768, 768]" = torch.ops.aten.permute.default(primals_557, [1, 0]);  primals_557 = None
    clone_248: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_409, memory_format = torch.contiguous_format);  view_409 = None
    view_410: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_248, [6272, 768]);  clone_248 = None
    mm_22: "f32[6272, 768]" = torch.ops.aten.mm.default(view_410, permute_205)
    view_411: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_22, [8, 784, 768])
    add_399: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_411, primals_558);  view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_542: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_89, add_399);  add_399 = None
    add_400: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_396, mul_542);  add_396 = mul_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_250: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_400, memory_format = torch.contiguous_format)
    var_mean_92 = torch.ops.aten.var_mean.correction(clone_250, [2], correction = 0, keepdim = True)
    getitem_253: "f32[8, 784, 1]" = var_mean_92[0]
    getitem_254: "f32[8, 784, 1]" = var_mean_92[1];  var_mean_92 = None
    add_401: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_253, 1e-06);  getitem_253 = None
    rsqrt_92: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_401);  add_401 = None
    sub_115: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_250, getitem_254);  clone_250 = getitem_254 = None
    mul_543: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_92);  sub_115 = None
    mul_544: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_543, primals_559)
    add_402: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_544, primals_560);  mul_544 = primals_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_206: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_402, [0, 2, 1]);  add_402 = None
    view_412: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_206, [8, 768, 28, 28]);  permute_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_48: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_412, primals_561, primals_562, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_545: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_48, 0.5)
    mul_546: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_48, 0.7071067811865476)
    erf_46: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_546);  mul_546 = None
    add_403: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
    mul_547: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_545, add_403);  mul_545 = add_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_404: "i64[]" = torch.ops.aten.add.Tensor(primals_706, 1)
    var_mean_93 = torch.ops.aten.var_mean.correction(mul_547, [0, 2, 3], correction = 0, keepdim = True)
    getitem_255: "f32[1, 768, 1, 1]" = var_mean_93[0]
    getitem_256: "f32[1, 768, 1, 1]" = var_mean_93[1];  var_mean_93 = None
    add_405: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_255, 1e-05)
    rsqrt_93: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_405);  add_405 = None
    sub_116: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_547, getitem_256);  mul_547 = None
    mul_548: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_93);  sub_116 = None
    squeeze_75: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_256, [0, 2, 3]);  getitem_256 = None
    squeeze_76: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
    mul_549: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_550: "f32[768]" = torch.ops.aten.mul.Tensor(primals_704, 0.9)
    add_406: "f32[768]" = torch.ops.aten.add.Tensor(mul_549, mul_550);  mul_549 = mul_550 = None
    squeeze_77: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_255, [0, 2, 3]);  getitem_255 = None
    mul_551: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001594642002871);  squeeze_77 = None
    mul_552: "f32[768]" = torch.ops.aten.mul.Tensor(mul_551, 0.1);  mul_551 = None
    mul_553: "f32[768]" = torch.ops.aten.mul.Tensor(primals_705, 0.9)
    add_407: "f32[768]" = torch.ops.aten.add.Tensor(mul_552, mul_553);  mul_552 = mul_553 = None
    unsqueeze_107: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_563, -1)
    unsqueeze_108: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_107, -1);  unsqueeze_107 = None
    mul_554: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_548, unsqueeze_108);  mul_548 = unsqueeze_108 = None
    unsqueeze_109: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_564, -1);  primals_564 = None
    unsqueeze_110: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, -1);  unsqueeze_109 = None
    add_408: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_554, unsqueeze_110);  mul_554 = unsqueeze_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_49: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_408, primals_565, primals_566, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_413: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_49, [8, 768, 784])
    permute_207: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_413, [0, 2, 1]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_555: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_91, permute_207);  permute_207 = None
    add_409: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_400, mul_555);  add_400 = mul_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_251: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_409, memory_format = torch.contiguous_format)
    var_mean_94 = torch.ops.aten.var_mean.correction(clone_251, [2], correction = 0, keepdim = True)
    getitem_257: "f32[8, 784, 1]" = var_mean_94[0]
    getitem_258: "f32[8, 784, 1]" = var_mean_94[1];  var_mean_94 = None
    add_410: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_257, 1e-06);  getitem_257 = None
    rsqrt_94: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_410);  add_410 = None
    sub_117: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_251, getitem_258);  clone_251 = getitem_258 = None
    mul_556: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_94);  sub_117 = None
    mul_557: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_556, primals_567)
    add_411: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_557, primals_568);  mul_557 = primals_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_414: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_411, [6272, 768]);  add_411 = None
    permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_569, [1, 0]);  primals_569 = None
    addmm_67: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_570, view_414, permute_208);  primals_570 = None
    view_415: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_67, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_558: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
    mul_559: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476);  view_415 = None
    erf_47: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_559);  mul_559 = None
    add_412: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
    mul_560: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_558, add_412);  mul_558 = add_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_416: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_560, [6272, 3072]);  mul_560 = None
    permute_209: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_571, [1, 0]);  primals_571 = None
    addmm_68: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_572, view_416, permute_209);  primals_572 = None
    view_417: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_68, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_561: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_92, view_417);  view_417 = None
    add_413: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_409, mul_561);  add_409 = mul_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    clone_254: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_413, memory_format = torch.contiguous_format)
    var_mean_95 = torch.ops.aten.var_mean.correction(clone_254, [2], correction = 0, keepdim = True)
    getitem_259: "f32[8, 784, 1]" = var_mean_95[0]
    getitem_260: "f32[8, 784, 1]" = var_mean_95[1];  var_mean_95 = None
    add_414: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_259, 1e-06);  getitem_259 = None
    rsqrt_95: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_414);  add_414 = None
    sub_118: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_254, getitem_260);  clone_254 = getitem_260 = None
    mul_562: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_95);  sub_118 = None
    mul_563: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_562, primals_573)
    add_415: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_563, primals_574);  mul_563 = primals_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    view_418: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_415, [6272, 768]);  add_415 = None
    permute_210: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_575, [1, 0]);  primals_575 = None
    addmm_69: "f32[6272, 2304]" = torch.ops.aten.addmm.default(primals_576, view_418, permute_210);  primals_576 = None
    view_419: "f32[8, 784, 2304]" = torch.ops.aten.reshape.default(addmm_69, [8, 784, 2304]);  addmm_69 = None
    view_420: "f32[8, 784, 3, 16, 48]" = torch.ops.aten.reshape.default(view_419, [8, 784, 3, 16, 48]);  view_419 = None
    permute_211: "f32[3, 8, 16, 48, 784]" = torch.ops.aten.permute.default(view_420, [2, 0, 3, 4, 1]);  view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:215, code: q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
    unbind_23 = torch.ops.aten.unbind.int(permute_211);  permute_211 = None
    getitem_261: "f32[8, 16, 48, 784]" = unbind_23[0]
    getitem_262: "f32[8, 16, 48, 784]" = unbind_23[1]
    getitem_263: "f32[8, 16, 48, 784]" = unbind_23[2];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:218, code: q = torch.nn.functional.normalize(q, dim=-1)
    pow_94: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_261, 2.0)
    sum_70: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_94, [-1], True);  pow_94 = None
    pow_95: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_70, 0.5);  sum_70 = None
    clamp_min_46: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_95, 1e-12)
    expand_138: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_46, [8, 16, 48, 784]);  clamp_min_46 = None
    div_75: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_261, expand_138);  expand_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:219, code: k = torch.nn.functional.normalize(k, dim=-1)
    pow_96: "f32[8, 16, 48, 784]" = torch.ops.aten.pow.Tensor_Scalar(getitem_262, 2.0)
    sum_71: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(pow_96, [-1], True);  pow_96 = None
    pow_97: "f32[8, 16, 48, 1]" = torch.ops.aten.pow.Tensor_Scalar(sum_71, 0.5);  sum_71 = None
    clamp_min_47: "f32[8, 16, 48, 1]" = torch.ops.aten.clamp_min.default(pow_97, 1e-12)
    expand_139: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(clamp_min_47, [8, 16, 48, 784]);  clamp_min_47 = None
    div_76: "f32[8, 16, 48, 784]" = torch.ops.aten.div.Tensor(getitem_262, expand_139);  expand_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_212: "f32[8, 16, 784, 48]" = torch.ops.aten.permute.default(div_76, [0, 1, 3, 2]);  div_76 = None
    expand_140: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(div_75, [8, 16, 48, 784]);  div_75 = None
    clone_255: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_140, memory_format = torch.contiguous_format);  expand_140 = None
    view_421: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_255, [128, 48, 784]);  clone_255 = None
    expand_141: "f32[8, 16, 784, 48]" = torch.ops.aten.expand.default(permute_212, [8, 16, 784, 48]);  permute_212 = None
    clone_256: "f32[8, 16, 784, 48]" = torch.ops.aten.clone.default(expand_141, memory_format = torch.contiguous_format);  expand_141 = None
    view_422: "f32[128, 784, 48]" = torch.ops.aten.reshape.default(clone_256, [128, 784, 48]);  clone_256 = None
    bmm_46: "f32[128, 48, 48]" = torch.ops.aten.bmm.default(view_421, view_422)
    view_423: "f32[8, 16, 48, 48]" = torch.ops.aten.reshape.default(bmm_46, [8, 16, 48, 48])
    mul_564: "f32[8, 16, 48, 48]" = torch.ops.aten.mul.Tensor(view_423, primals_94);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    amax_23: "f32[8, 16, 48, 1]" = torch.ops.aten.amax.default(mul_564, [-1], True)
    sub_119: "f32[8, 16, 48, 48]" = torch.ops.aten.sub.Tensor(mul_564, amax_23);  mul_564 = amax_23 = None
    exp_23: "f32[8, 16, 48, 48]" = torch.ops.aten.exp.default(sub_119);  sub_119 = None
    sum_72: "f32[8, 16, 48, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_77: "f32[8, 16, 48, 48]" = torch.ops.aten.div.Tensor(exp_23, sum_72);  exp_23 = sum_72 = None
    alias_71: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(div_77)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:222, code: attn = self.attn_drop(attn)
    clone_257: "f32[8, 16, 48, 48]" = torch.ops.aten.clone.default(div_77);  div_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    expand_142: "f32[8, 16, 48, 48]" = torch.ops.aten.expand.default(clone_257, [8, 16, 48, 48]);  clone_257 = None
    view_424: "f32[128, 48, 48]" = torch.ops.aten.reshape.default(expand_142, [128, 48, 48]);  expand_142 = None
    expand_143: "f32[8, 16, 48, 784]" = torch.ops.aten.expand.default(getitem_263, [8, 16, 48, 784]);  getitem_263 = None
    clone_258: "f32[8, 16, 48, 784]" = torch.ops.aten.clone.default(expand_143, memory_format = torch.contiguous_format);  expand_143 = None
    view_425: "f32[128, 48, 784]" = torch.ops.aten.reshape.default(clone_258, [128, 48, 784]);  clone_258 = None
    bmm_47: "f32[128, 48, 784]" = torch.ops.aten.bmm.default(view_424, view_425)
    view_426: "f32[8, 16, 48, 784]" = torch.ops.aten.reshape.default(bmm_47, [8, 16, 48, 784]);  bmm_47 = None
    permute_213: "f32[8, 784, 16, 48]" = torch.ops.aten.permute.default(view_426, [0, 3, 1, 2]);  view_426 = None
    view_427: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(permute_213, [8, 784, 768]);  permute_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_214: "f32[768, 768]" = torch.ops.aten.permute.default(primals_577, [1, 0]);  primals_577 = None
    clone_259: "f32[8, 784, 768]" = torch.ops.aten.clone.default(view_427, memory_format = torch.contiguous_format);  view_427 = None
    view_428: "f32[6272, 768]" = torch.ops.aten.reshape.default(clone_259, [6272, 768]);  clone_259 = None
    mm_23: "f32[6272, 768]" = torch.ops.aten.mm.default(view_428, permute_214)
    view_429: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(mm_23, [8, 784, 768])
    add_416: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(view_429, primals_578);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    mul_565: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_93, add_416);  add_416 = None
    add_417: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_413, mul_565);  add_413 = mul_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    clone_261: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_417, memory_format = torch.contiguous_format)
    var_mean_96 = torch.ops.aten.var_mean.correction(clone_261, [2], correction = 0, keepdim = True)
    getitem_264: "f32[8, 784, 1]" = var_mean_96[0]
    getitem_265: "f32[8, 784, 1]" = var_mean_96[1];  var_mean_96 = None
    add_418: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_264, 1e-06);  getitem_264 = None
    rsqrt_96: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_418);  add_418 = None
    sub_120: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_261, getitem_265);  clone_261 = getitem_265 = None
    mul_566: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_96);  sub_120 = None
    mul_567: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_566, primals_579)
    add_419: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_567, primals_580);  mul_567 = primals_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:135, code: x = x.permute(0, 2, 1).reshape(B, C, H, W)
    permute_215: "f32[8, 768, 784]" = torch.ops.aten.permute.default(add_419, [0, 2, 1]);  add_419 = None
    view_430: "f32[8, 768, 28, 28]" = torch.ops.aten.reshape.default(permute_215, [8, 768, 28, 28]);  permute_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:136, code: x = self.conv1(x)
    convolution_50: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(view_430, primals_581, primals_582, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:137, code: x = self.act(x)
    mul_568: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_50, 0.5)
    mul_569: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_50, 0.7071067811865476)
    erf_48: "f32[8, 768, 28, 28]" = torch.ops.aten.erf.default(mul_569);  mul_569 = None
    add_420: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(erf_48, 1);  erf_48 = None
    mul_570: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_568, add_420);  mul_568 = add_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    add_421: "i64[]" = torch.ops.aten.add.Tensor(primals_709, 1)
    var_mean_97 = torch.ops.aten.var_mean.correction(mul_570, [0, 2, 3], correction = 0, keepdim = True)
    getitem_266: "f32[1, 768, 1, 1]" = var_mean_97[0]
    getitem_267: "f32[1, 768, 1, 1]" = var_mean_97[1];  var_mean_97 = None
    add_422: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_266, 1e-05)
    rsqrt_97: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_422);  add_422 = None
    sub_121: "f32[8, 768, 28, 28]" = torch.ops.aten.sub.Tensor(mul_570, getitem_267);  mul_570 = None
    mul_571: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt_97);  sub_121 = None
    squeeze_78: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_267, [0, 2, 3]);  getitem_267 = None
    squeeze_79: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_97, [0, 2, 3]);  rsqrt_97 = None
    mul_572: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_573: "f32[768]" = torch.ops.aten.mul.Tensor(primals_707, 0.9)
    add_423: "f32[768]" = torch.ops.aten.add.Tensor(mul_572, mul_573);  mul_572 = mul_573 = None
    squeeze_80: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_266, [0, 2, 3]);  getitem_266 = None
    mul_574: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001594642002871);  squeeze_80 = None
    mul_575: "f32[768]" = torch.ops.aten.mul.Tensor(mul_574, 0.1);  mul_574 = None
    mul_576: "f32[768]" = torch.ops.aten.mul.Tensor(primals_708, 0.9)
    add_424: "f32[768]" = torch.ops.aten.add.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
    unsqueeze_111: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_583, -1)
    unsqueeze_112: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_111, -1);  unsqueeze_111 = None
    mul_577: "f32[8, 768, 28, 28]" = torch.ops.aten.mul.Tensor(mul_571, unsqueeze_112);  mul_571 = unsqueeze_112 = None
    unsqueeze_113: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_584, -1);  primals_584 = None
    unsqueeze_114: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, -1);  unsqueeze_113 = None
    add_425: "f32[8, 768, 28, 28]" = torch.ops.aten.add.Tensor(mul_577, unsqueeze_114);  mul_577 = unsqueeze_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:139, code: x = self.conv2(x)
    convolution_51: "f32[8, 768, 28, 28]" = torch.ops.aten.convolution.default(add_425, primals_585, primals_586, [1, 1], [1, 1], [1, 1], False, [0, 0], 768);  primals_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:140, code: x = x.reshape(B, C, N).permute(0, 2, 1)
    view_431: "f32[8, 768, 784]" = torch.ops.aten.reshape.default(convolution_51, [8, 768, 784])
    permute_216: "f32[8, 784, 768]" = torch.ops.aten.permute.default(view_431, [0, 2, 1]);  view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    mul_578: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_95, permute_216);  permute_216 = None
    add_426: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_417, mul_578);  add_417 = mul_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    clone_262: "f32[8, 784, 768]" = torch.ops.aten.clone.default(add_426, memory_format = torch.contiguous_format)
    var_mean_98 = torch.ops.aten.var_mean.correction(clone_262, [2], correction = 0, keepdim = True)
    getitem_268: "f32[8, 784, 1]" = var_mean_98[0]
    getitem_269: "f32[8, 784, 1]" = var_mean_98[1];  var_mean_98 = None
    add_427: "f32[8, 784, 1]" = torch.ops.aten.add.Tensor(getitem_268, 1e-06);  getitem_268 = None
    rsqrt_98: "f32[8, 784, 1]" = torch.ops.aten.rsqrt.default(add_427);  add_427 = None
    sub_122: "f32[8, 784, 768]" = torch.ops.aten.sub.Tensor(clone_262, getitem_269);  clone_262 = getitem_269 = None
    mul_579: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_98);  sub_122 = None
    mul_580: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(mul_579, primals_587)
    add_428: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(mul_580, primals_588);  mul_580 = primals_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_432: "f32[6272, 768]" = torch.ops.aten.reshape.default(add_428, [6272, 768]);  add_428 = None
    permute_217: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_589, [1, 0]);  primals_589 = None
    addmm_70: "f32[6272, 3072]" = torch.ops.aten.addmm.default(primals_590, view_432, permute_217);  primals_590 = None
    view_433: "f32[8, 784, 3072]" = torch.ops.aten.reshape.default(addmm_70, [8, 784, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_581: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_433, 0.5)
    mul_582: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(view_433, 0.7071067811865476);  view_433 = None
    erf_49: "f32[8, 784, 3072]" = torch.ops.aten.erf.default(mul_582);  mul_582 = None
    add_429: "f32[8, 784, 3072]" = torch.ops.aten.add.Tensor(erf_49, 1);  erf_49 = None
    mul_583: "f32[8, 784, 3072]" = torch.ops.aten.mul.Tensor(mul_581, add_429);  mul_581 = add_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_434: "f32[6272, 3072]" = torch.ops.aten.reshape.default(mul_583, [6272, 3072]);  mul_583 = None
    permute_218: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_591, [1, 0]);  primals_591 = None
    addmm_71: "f32[6272, 768]" = torch.ops.aten.addmm.default(primals_592, view_434, permute_218);  primals_592 = None
    view_435: "f32[8, 784, 768]" = torch.ops.aten.reshape.default(addmm_71, [8, 784, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    mul_584: "f32[8, 784, 768]" = torch.ops.aten.mul.Tensor(primals_96, view_435);  view_435 = None
    add_430: "f32[8, 784, 768]" = torch.ops.aten.add.Tensor(add_426, mul_584);  add_426 = mul_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:447, code: x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
    expand_144: "f32[8, 1, 768]" = torch.ops.aten.expand.default(primals_97, [8, -1, -1]);  primals_97 = None
    cat_3: "f32[8, 785, 768]" = torch.ops.aten.cat.default([expand_144, add_430], 1);  expand_144 = add_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    var_mean_99 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
    getitem_270: "f32[8, 785, 1]" = var_mean_99[0]
    getitem_271: "f32[8, 785, 1]" = var_mean_99[1];  var_mean_99 = None
    add_431: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_270, 1e-06);  getitem_270 = None
    rsqrt_99: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_431);  add_431 = None
    sub_123: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(cat_3, getitem_271)
    mul_585: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(sub_123, rsqrt_99);  sub_123 = None
    mul_586: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_585, primals_593);  mul_585 = None
    add_432: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(mul_586, primals_594);  mul_586 = primals_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_29: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(add_432, 0, 0, 9223372036854775807)
    select: "f32[8, 768]" = torch.ops.aten.select.int(slice_29, 1, 0)
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(primals_595, [1, 0]);  primals_595 = None
    addmm_72: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_596, select, permute_219);  primals_596 = None
    unsqueeze_115: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_72, 1);  addmm_72 = None
    view_436: "f32[8, 1, 16, 48]" = torch.ops.aten.reshape.default(unsqueeze_115, [8, 1, 16, 48]);  unsqueeze_115 = None
    permute_220: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_436, [0, 2, 1, 3]);  view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_437: "f32[6280, 768]" = torch.ops.aten.reshape.default(add_432, [6280, 768]);  add_432 = None
    permute_221: "f32[768, 768]" = torch.ops.aten.permute.default(primals_597, [1, 0]);  primals_597 = None
    addmm_73: "f32[6280, 768]" = torch.ops.aten.addmm.default(primals_598, view_437, permute_221);  primals_598 = None
    view_438: "f32[8, 785, 768]" = torch.ops.aten.reshape.default(addmm_73, [8, 785, 768]);  addmm_73 = None
    view_439: "f32[8, 785, 16, 48]" = torch.ops.aten.reshape.default(view_438, [8, 785, 16, 48]);  view_438 = None
    permute_222: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_439, [0, 2, 1, 3]);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_223: "f32[768, 768]" = torch.ops.aten.permute.default(primals_599, [1, 0]);  primals_599 = None
    addmm_74: "f32[6280, 768]" = torch.ops.aten.addmm.default(primals_600, view_437, permute_223);  primals_600 = None
    view_441: "f32[8, 785, 768]" = torch.ops.aten.reshape.default(addmm_74, [8, 785, 768]);  addmm_74 = None
    view_442: "f32[8, 785, 16, 48]" = torch.ops.aten.reshape.default(view_441, [8, 785, 16, 48]);  view_441 = None
    permute_224: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_220, permute_222, permute_224, None, True)
    getitem_272: "f32[8, 16, 1, 48]" = _scaled_dot_product_efficient_attention[0]
    getitem_273: "f32[8, 16, 32]" = _scaled_dot_product_efficient_attention[1]
    getitem_274: "i64[]" = _scaled_dot_product_efficient_attention[2]
    getitem_275: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
    alias_72: "f32[8, 16, 1, 48]" = torch.ops.aten.alias.default(getitem_272)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    permute_225: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_272, [0, 2, 1, 3]);  getitem_272 = None
    view_443: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(permute_225, [8, 1, 768]);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_444: "f32[8, 768]" = torch.ops.aten.reshape.default(view_443, [8, 768]);  view_443 = None
    permute_226: "f32[768, 768]" = torch.ops.aten.permute.default(primals_601, [1, 0]);  primals_601 = None
    addmm_75: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_602, view_444, permute_226);  primals_602 = None
    view_445: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(addmm_75, [8, 1, 768]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:182, code: x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
    slice_31: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_29, 1, 1, 9223372036854775807);  slice_29 = None
    cat_4: "f32[8, 785, 768]" = torch.ops.aten.cat.default([view_445, slice_31], 1);  view_445 = slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:183, code: x = x + self.drop_path(self.gamma1 * x_attn)
    mul_587: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(primals_98, cat_4)
    add_433: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(cat_3, mul_587);  mul_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    var_mean_100 = torch.ops.aten.var_mean.correction(add_433, [2], correction = 0, keepdim = True)
    getitem_276: "f32[8, 785, 1]" = var_mean_100[0]
    getitem_277: "f32[8, 785, 1]" = var_mean_100[1];  var_mean_100 = None
    add_434: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_276, 1e-06);  getitem_276 = None
    rsqrt_100: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_434);  add_434 = None
    sub_124: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(add_433, getitem_277);  add_433 = getitem_277 = None
    mul_588: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(sub_124, rsqrt_100);  sub_124 = None
    mul_589: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_588, primals_603)
    add_435: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(mul_589, primals_604);  mul_589 = primals_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    slice_32: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(add_435, 0, 0, 9223372036854775807)
    slice_33: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(slice_32, 1, 0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_227: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_605, [1, 0]);  primals_605 = None
    view_446: "f32[8, 768]" = torch.ops.aten.reshape.default(slice_33, [8, 768]);  slice_33 = None
    mm_24: "f32[8, 3072]" = torch.ops.aten.mm.default(view_446, permute_227)
    view_447: "f32[8, 1, 3072]" = torch.ops.aten.reshape.default(mm_24, [8, 1, 3072])
    add_436: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(view_447, primals_606);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_590: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_436, 0.5)
    mul_591: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_436, 0.7071067811865476);  add_436 = None
    erf_50: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_591);  mul_591 = None
    add_437: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_50, 1);  erf_50 = None
    mul_592: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_590, add_437);  mul_590 = add_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_448: "f32[8, 3072]" = torch.ops.aten.reshape.default(mul_592, [8, 3072]);  mul_592 = None
    permute_228: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_607, [1, 0]);  primals_607 = None
    addmm_76: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_608, view_448, permute_228);  primals_608 = None
    view_449: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(addmm_76, [8, 1, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    mul_593: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(primals_99, view_449);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    slice_35: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_32, 1, 1, 9223372036854775807);  slice_32 = None
    cat_5: "f32[8, 785, 768]" = torch.ops.aten.cat.default([mul_593, slice_35], 1);  mul_593 = slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:192, code: x = x_res + self.drop_path(x)
    add_438: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_435, cat_5);  add_435 = cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    var_mean_101 = torch.ops.aten.var_mean.correction(add_438, [2], correction = 0, keepdim = True)
    getitem_278: "f32[8, 785, 1]" = var_mean_101[0]
    getitem_279: "f32[8, 785, 1]" = var_mean_101[1];  var_mean_101 = None
    add_439: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_278, 1e-06);  getitem_278 = None
    rsqrt_101: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_439);  add_439 = None
    sub_125: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(add_438, getitem_279);  getitem_279 = None
    mul_594: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_101);  sub_125 = None
    mul_595: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_594, primals_609)
    add_440: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(mul_595, primals_610);  mul_595 = primals_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_36: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(add_440, 0, 0, 9223372036854775807)
    select_1: "f32[8, 768]" = torch.ops.aten.select.int(slice_36, 1, 0)
    permute_229: "f32[768, 768]" = torch.ops.aten.permute.default(primals_611, [1, 0]);  primals_611 = None
    addmm_77: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_612, select_1, permute_229);  primals_612 = None
    unsqueeze_116: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(addmm_77, 1);  addmm_77 = None
    view_450: "f32[8, 1, 16, 48]" = torch.ops.aten.reshape.default(unsqueeze_116, [8, 1, 16, 48]);  unsqueeze_116 = None
    permute_230: "f32[8, 16, 1, 48]" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_451: "f32[6280, 768]" = torch.ops.aten.reshape.default(add_440, [6280, 768]);  add_440 = None
    permute_231: "f32[768, 768]" = torch.ops.aten.permute.default(primals_613, [1, 0]);  primals_613 = None
    addmm_78: "f32[6280, 768]" = torch.ops.aten.addmm.default(primals_614, view_451, permute_231);  primals_614 = None
    view_452: "f32[8, 785, 768]" = torch.ops.aten.reshape.default(addmm_78, [8, 785, 768]);  addmm_78 = None
    view_453: "f32[8, 785, 16, 48]" = torch.ops.aten.reshape.default(view_452, [8, 785, 16, 48]);  view_452 = None
    permute_232: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(primals_615, [1, 0]);  primals_615 = None
    addmm_79: "f32[6280, 768]" = torch.ops.aten.addmm.default(primals_616, view_451, permute_233);  primals_616 = None
    view_455: "f32[8, 785, 768]" = torch.ops.aten.reshape.default(addmm_79, [8, 785, 768]);  addmm_79 = None
    view_456: "f32[8, 785, 16, 48]" = torch.ops.aten.reshape.default(view_455, [8, 785, 16, 48]);  view_455 = None
    permute_234: "f32[8, 16, 785, 48]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_230, permute_232, permute_234, None, True)
    getitem_280: "f32[8, 16, 1, 48]" = _scaled_dot_product_efficient_attention_1[0]
    getitem_281: "f32[8, 16, 32]" = _scaled_dot_product_efficient_attention_1[1]
    getitem_282: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
    getitem_283: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
    alias_73: "f32[8, 16, 1, 48]" = torch.ops.aten.alias.default(getitem_280)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    permute_235: "f32[8, 1, 16, 48]" = torch.ops.aten.permute.default(getitem_280, [0, 2, 1, 3]);  getitem_280 = None
    view_457: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(permute_235, [8, 1, 768]);  permute_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    view_458: "f32[8, 768]" = torch.ops.aten.reshape.default(view_457, [8, 768]);  view_457 = None
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(primals_617, [1, 0]);  primals_617 = None
    addmm_80: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_618, view_458, permute_236);  primals_618 = None
    view_459: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(addmm_80, [8, 1, 768]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:182, code: x_attn = torch.cat([self.attn(x_norm1), x_norm1[:, 1:]], dim=1)
    slice_38: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_36, 1, 1, 9223372036854775807);  slice_36 = None
    cat_6: "f32[8, 785, 768]" = torch.ops.aten.cat.default([view_459, slice_38], 1);  view_459 = slice_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:183, code: x = x + self.drop_path(self.gamma1 * x_attn)
    mul_596: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(primals_100, cat_6)
    add_441: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_438, mul_596);  add_438 = mul_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    var_mean_102 = torch.ops.aten.var_mean.correction(add_441, [2], correction = 0, keepdim = True)
    getitem_284: "f32[8, 785, 1]" = var_mean_102[0]
    getitem_285: "f32[8, 785, 1]" = var_mean_102[1];  var_mean_102 = None
    add_442: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_284, 1e-06);  getitem_284 = None
    rsqrt_102: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_442);  add_442 = None
    sub_126: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(add_441, getitem_285);  add_441 = getitem_285 = None
    mul_597: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(sub_126, rsqrt_102);  sub_126 = None
    mul_598: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_597, primals_619)
    add_443: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(mul_598, primals_620);  mul_598 = primals_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:189, code: cls_token = x[:, 0:1]
    slice_39: "f32[8, 785, 768]" = torch.ops.aten.slice.Tensor(add_443, 0, 0, 9223372036854775807)
    slice_40: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(slice_39, 1, 0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_237: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_621, [1, 0]);  primals_621 = None
    view_460: "f32[8, 768]" = torch.ops.aten.reshape.default(slice_40, [8, 768]);  slice_40 = None
    mm_25: "f32[8, 3072]" = torch.ops.aten.mm.default(view_460, permute_237)
    view_461: "f32[8, 1, 3072]" = torch.ops.aten.reshape.default(mm_25, [8, 1, 3072])
    add_444: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(view_461, primals_622);  view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_599: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_444, 0.5)
    mul_600: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(add_444, 0.7071067811865476);  add_444 = None
    erf_51: "f32[8, 1, 3072]" = torch.ops.aten.erf.default(mul_600);  mul_600 = None
    add_445: "f32[8, 1, 3072]" = torch.ops.aten.add.Tensor(erf_51, 1);  erf_51 = None
    mul_601: "f32[8, 1, 3072]" = torch.ops.aten.mul.Tensor(mul_599, add_445);  mul_599 = add_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_462: "f32[8, 3072]" = torch.ops.aten.reshape.default(mul_601, [8, 3072]);  mul_601 = None
    permute_238: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_623, [1, 0]);  primals_623 = None
    addmm_81: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_624, view_462, permute_238);  primals_624 = None
    view_463: "f32[8, 1, 768]" = torch.ops.aten.reshape.default(addmm_81, [8, 1, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:190, code: cls_token = self.gamma2 * self.mlp(cls_token)
    mul_602: "f32[8, 1, 768]" = torch.ops.aten.mul.Tensor(primals_101, view_463);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:191, code: x = torch.cat([cls_token, x[:, 1:]], dim=1)
    slice_42: "f32[8, 784, 768]" = torch.ops.aten.slice.Tensor(slice_39, 1, 1, 9223372036854775807);  slice_39 = None
    cat_7: "f32[8, 785, 768]" = torch.ops.aten.cat.default([mul_602, slice_42], 1);  mul_602 = slice_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:192, code: x = x_res + self.drop_path(x)
    add_446: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(add_443, cat_7);  add_443 = cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:455, code: x = self.norm(x)
    var_mean_103 = torch.ops.aten.var_mean.correction(add_446, [2], correction = 0, keepdim = True)
    getitem_286: "f32[8, 785, 1]" = var_mean_103[0]
    getitem_287: "f32[8, 785, 1]" = var_mean_103[1];  var_mean_103 = None
    add_447: "f32[8, 785, 1]" = torch.ops.aten.add.Tensor(getitem_286, 1e-06);  getitem_286 = None
    rsqrt_103: "f32[8, 785, 1]" = torch.ops.aten.rsqrt.default(add_447);  add_447 = None
    sub_127: "f32[8, 785, 768]" = torch.ops.aten.sub.Tensor(add_446, getitem_287);  add_446 = getitem_287 = None
    mul_603: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(sub_127, rsqrt_103);  sub_127 = None
    mul_604: "f32[8, 785, 768]" = torch.ops.aten.mul.Tensor(mul_603, primals_625)
    add_448: "f32[8, 785, 768]" = torch.ops.aten.add.Tensor(mul_604, primals_626);  mul_604 = primals_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:460, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    select_2: "f32[8, 768]" = torch.ops.aten.select.int(add_448, 1, 0);  add_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:461, code: x = self.head_drop(x)
    clone_271: "f32[8, 768]" = torch.ops.aten.clone.default(select_2);  select_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:462, code: return x if pre_logits else self.head(x)
    permute_239: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_627, [1, 0]);  primals_627 = None
    addmm_82: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_628, clone_271, permute_239);  primals_628 = None
    permute_240: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:455, code: x = self.norm(x)
    div_78: "f32[8, 785, 1]" = torch.ops.aten.div.Tensor(rsqrt_103, 768);  rsqrt_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_244: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_250: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    div_79: "f32[8, 785, 1]" = torch.ops.aten.div.Tensor(rsqrt_102, 768);  rsqrt_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    permute_252: "f32[768, 768]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    alias_74: "f32[8, 16, 1, 48]" = torch.ops.aten.alias.default(alias_73);  alias_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_268: "f32[768, 768]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:181, code: x_norm1 = self.norm1(x)
    div_80: "f32[8, 785, 1]" = torch.ops.aten.div.Tensor(rsqrt_101, 768);  rsqrt_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_272: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_278: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:185, code: x = self.norm2(x)
    div_81: "f32[8, 785, 1]" = torch.ops.aten.div.Tensor(rsqrt_100, 768);  rsqrt_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    permute_280: "f32[768, 768]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    alias_75: "f32[8, 16, 1, 48]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_286: "f32[768, 768]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_300: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_304: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_83: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_98, 768);  rsqrt_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_117: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_118: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None
    unsqueeze_119: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 3);  unsqueeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_84: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_96, 768);  rsqrt_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_312: "f32[768, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_315: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_424, [0, 2, 1]);  view_424 = None
    permute_316: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_76: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_71);  alias_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_317: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_421, [0, 2, 1]);  view_421 = None
    permute_318: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_422, [0, 2, 1]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_321: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_93: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_95, 768);  rsqrt_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_325: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_329: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_94: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_94, 768);  rsqrt_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_129: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_130: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 2);  unsqueeze_129 = None
    unsqueeze_131: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 3);  unsqueeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_95: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_92, 768);  rsqrt_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_337: "f32[768, 768]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_340: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None
    permute_341: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_407, [0, 2, 1]);  view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_79: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_68);  alias_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_342: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_403, [0, 2, 1]);  view_403 = None
    permute_343: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_404, [0, 2, 1]);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_346: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_104: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_91, 768);  rsqrt_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_350: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_354: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_105: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_90, 768);  rsqrt_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_141: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_142: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None
    unsqueeze_143: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 3);  unsqueeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_106: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_88, 768);  rsqrt_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_365: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_388, [0, 2, 1]);  view_388 = None
    permute_366: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_389, [0, 2, 1]);  view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_82: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_65);  alias_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_367: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    permute_368: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_386, [0, 2, 1]);  view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_371: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_115: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_87, 768);  rsqrt_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_375: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_379: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_116: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_86, 768);  rsqrt_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_153: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_154: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 2);  unsqueeze_153 = None
    unsqueeze_155: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 3);  unsqueeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_117: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_84, 768);  rsqrt_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_387: "f32[768, 768]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_390: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_370, [0, 2, 1]);  view_370 = None
    permute_391: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_371, [0, 2, 1]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_85: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_392: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_367, [0, 2, 1]);  view_367 = None
    permute_393: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_368, [0, 2, 1]);  view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_396: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_126: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_83, 768);  rsqrt_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_400: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_404: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_127: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_82, 768);  rsqrt_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_165: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_166: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
    unsqueeze_167: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 3);  unsqueeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_128: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_80, 768);  rsqrt_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_412: "f32[768, 768]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_415: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None
    permute_416: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_353, [0, 2, 1]);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_88: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_417: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
    permute_418: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_350, [0, 2, 1]);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_421: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_137: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_79, 768);  rsqrt_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_425: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_429: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_138: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_78, 768);  rsqrt_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_177: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_178: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 2);  unsqueeze_177 = None
    unsqueeze_179: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 3);  unsqueeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_139: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_76, 768);  rsqrt_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_437: "f32[768, 768]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_440: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_334, [0, 2, 1]);  view_334 = None
    permute_441: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_335, [0, 2, 1]);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_91: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_442: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_331, [0, 2, 1]);  view_331 = None
    permute_443: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_332, [0, 2, 1]);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_446: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_148: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_75, 768);  rsqrt_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_450: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_454: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_149: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_74, 768);  rsqrt_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_189: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_190: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
    unsqueeze_191: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 3);  unsqueeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_150: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_72, 768);  rsqrt_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_462: "f32[768, 768]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_465: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_316, [0, 2, 1]);  view_316 = None
    permute_466: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_94: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_53);  alias_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_467: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
    permute_468: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_314, [0, 2, 1]);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_471: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_159: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_71, 768);  rsqrt_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_475: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_479: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_160: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_70, 768);  rsqrt_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_201: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_202: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
    unsqueeze_203: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 3);  unsqueeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_161: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_68, 768);  rsqrt_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_487: "f32[768, 768]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_490: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_298, [0, 2, 1]);  view_298 = None
    permute_491: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_97: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_492: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
    permute_493: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_496: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_170: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_67, 768);  rsqrt_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_500: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_504: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_171: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_66, 768);  rsqrt_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_213: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_214: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
    unsqueeze_215: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 3);  unsqueeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_172: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_64, 768);  rsqrt_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_512: "f32[768, 768]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_515: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_280, [0, 2, 1]);  view_280 = None
    permute_516: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_281, [0, 2, 1]);  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_100: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_517: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None
    permute_518: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_278, [0, 2, 1]);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_521: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_181: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_63, 768);  rsqrt_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_525: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_529: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_182: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_62, 768);  rsqrt_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_225: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_226: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
    unsqueeze_227: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 3);  unsqueeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_183: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_60, 768);  rsqrt_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_537: "f32[768, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_540: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_262, [0, 2, 1]);  view_262 = None
    permute_541: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_103: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_542: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_259, [0, 2, 1]);  view_259 = None
    permute_543: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_546: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_192: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_59, 768);  rsqrt_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_550: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_554: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_193: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_58, 768);  rsqrt_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_237: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_238: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
    unsqueeze_239: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_194: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_56, 768);  rsqrt_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_562: "f32[768, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_565: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
    permute_566: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_106: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_567: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_241, [0, 2, 1]);  view_241 = None
    permute_568: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_242, [0, 2, 1]);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_571: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_203: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_55, 768);  rsqrt_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_575: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_579: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_204: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_54, 768);  rsqrt_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_249: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_250: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
    unsqueeze_251: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 3);  unsqueeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_205: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_52, 768);  rsqrt_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_587: "f32[768, 768]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_590: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_226, [0, 2, 1]);  view_226 = None
    permute_591: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_227, [0, 2, 1]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_109: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_592: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_223, [0, 2, 1]);  view_223 = None
    permute_593: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_224, [0, 2, 1]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_596: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_214: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_51, 768);  rsqrt_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_600: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_604: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_215: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_50, 768);  rsqrt_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_261: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_262: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
    unsqueeze_263: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 3);  unsqueeze_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_216: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 768);  rsqrt_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_612: "f32[768, 768]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_615: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    permute_616: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_112: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_617: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    permute_618: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_621: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_225: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 768);  rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_625: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_629: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_226: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 768);  rsqrt_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_273: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_274: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
    unsqueeze_275: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_227: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 768);  rsqrt_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_637: "f32[768, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_640: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    permute_641: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_115: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_642: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    permute_643: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_646: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_236: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 768);  rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_650: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_654: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_237: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 768);  rsqrt_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_285: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_286: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
    unsqueeze_287: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_238: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 768);  rsqrt_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_662: "f32[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_665: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    permute_666: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_118: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_667: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    permute_668: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_170, [0, 2, 1]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_671: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_247: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 768);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_675: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_679: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_248: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 768);  rsqrt_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_297: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_298: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
    unsqueeze_299: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_249: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 768);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_687: "f32[768, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_690: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_154, [0, 2, 1]);  view_154 = None
    permute_691: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_121: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_692: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
    permute_693: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_696: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_258: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 768);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_700: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_704: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_259: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 768);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_309: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_310: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
    unsqueeze_311: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 3);  unsqueeze_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_260: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 768);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_712: "f32[768, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_715: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
    permute_716: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_137, [0, 2, 1]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_124: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_717: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    permute_718: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_721: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_269: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 768);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_725: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_729: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_270: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 768);  rsqrt_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_321: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_322: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
    unsqueeze_323: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 3);  unsqueeze_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_271: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 768);  rsqrt_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_737: "f32[768, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_740: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_118, [0, 2, 1]);  view_118 = None
    permute_741: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_127: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_742: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_115, [0, 2, 1]);  view_115 = None
    permute_743: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_116, [0, 2, 1]);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_746: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_280: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 768);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_750: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_754: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_281: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 768);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_333: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_334: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
    unsqueeze_335: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_282: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_762: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_765: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    permute_766: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_130: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_767: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    permute_768: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_771: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_291: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_775: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_779: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_292: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_345: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_346: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
    unsqueeze_347: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 3);  unsqueeze_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_293: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_787: "f32[768, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_790: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    permute_791: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_133: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_792: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    permute_793: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_796: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_302: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_800: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_804: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_303: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_357: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_358: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    unsqueeze_359: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_304: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_812: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_815: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    permute_816: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_136: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_817: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    permute_818: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_821: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_313: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_825: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_829: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_314: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_369: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_370: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
    unsqueeze_371: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 3);  unsqueeze_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_315: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_837: "f32[768, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_840: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_46, [0, 2, 1]);  view_46 = None
    permute_841: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_139: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_842: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
    permute_843: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_44, [0, 2, 1]);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_846: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_324: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_850: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_854: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_325: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_381: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_382: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
    unsqueeze_383: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 3);  unsqueeze_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_326: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_862: "f32[768, 768]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_865: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    permute_866: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_142: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_867: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    permute_868: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_871: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_335: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_875: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_879: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:269, code: x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
    div_336: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:138, code: x = self.bn(x)
    unsqueeze_393: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_394: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
    unsqueeze_395: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:268, code: x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
    div_337: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:226, code: x = self.proj(x)
    permute_887: "f32[768, 768]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:225, code: x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    permute_890: "f32[128, 48, 48]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    permute_891: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:221, code: attn = attn.softmax(dim=-1)
    alias_145: "f32[8, 16, 48, 48]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:220, code: attn = (q @ k.transpose(-2, -1)) * self.temperature
    permute_892: "f32[128, 784, 48]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    permute_893: "f32[128, 48, 784]" = torch.ops.aten.permute.default(view_8, [0, 2, 1]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:214, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
    permute_896: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:265, code: x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
    div_346: "f32[8, 784, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/xcit.py:107, code: x = self.proj(x)
    unsqueeze_405: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_406: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    unsqueeze_407: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
    mul_2064: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(add_11, 0.5);  add_11 = None
    mul_2065: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(add_10, add_10)
    mul_2066: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(mul_2065, -0.5);  mul_2065 = None
    exp_74: "f32[8, 384, 56, 56]" = torch.ops.aten.exp.default(mul_2066);  mul_2066 = None
    mul_2067: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(exp_74, 0.3989422804014327);  exp_74 = None
    mul_2068: "f32[8, 384, 56, 56]" = torch.ops.aten.mul.Tensor(add_10, mul_2067);  add_10 = mul_2067 = None
    add_682: "f32[8, 384, 56, 56]" = torch.ops.aten.add.Tensor(mul_2064, mul_2068);  mul_2064 = mul_2068 = None
    unsqueeze_417: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_418: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    unsqueeze_419: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
    mul_2080: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(add_5, 0.5);  add_5 = None
    mul_2081: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, add_4)
    mul_2082: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(mul_2081, -0.5);  mul_2081 = None
    exp_75: "f32[8, 192, 112, 112]" = torch.ops.aten.exp.default(mul_2082);  mul_2082 = None
    mul_2083: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(exp_75, 0.3989422804014327);  exp_75 = None
    mul_2084: "f32[8, 192, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, mul_2083);  add_4 = mul_2083 = None
    add_684: "f32[8, 192, 112, 112]" = torch.ops.aten.add.Tensor(mul_2080, mul_2084);  mul_2080 = mul_2084 = None
    unsqueeze_429: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_430: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    
    # No stacktrace found for following nodes
    copy_: "f32[192]" = torch.ops.aten.copy_.default(primals_629, add_2);  primals_629 = add_2 = None
    copy__1: "f32[192]" = torch.ops.aten.copy_.default(primals_630, add_3);  primals_630 = add_3 = None
    copy__2: "i64[]" = torch.ops.aten.copy_.default(primals_631, add);  primals_631 = add = None
    copy__3: "f32[384]" = torch.ops.aten.copy_.default(primals_632, add_8);  primals_632 = add_8 = None
    copy__4: "f32[384]" = torch.ops.aten.copy_.default(primals_633, add_9);  primals_633 = add_9 = None
    copy__5: "i64[]" = torch.ops.aten.copy_.default(primals_634, add_6);  primals_634 = add_6 = None
    copy__6: "f32[768]" = torch.ops.aten.copy_.default(primals_635, add_14);  primals_635 = add_14 = None
    copy__7: "f32[768]" = torch.ops.aten.copy_.default(primals_636, add_15);  primals_636 = add_15 = None
    copy__8: "i64[]" = torch.ops.aten.copy_.default(primals_637, add_12);  primals_637 = add_12 = None
    copy__9: "f32[768]" = torch.ops.aten.copy_.default(primals_638, add_32);  primals_638 = add_32 = None
    copy__10: "f32[768]" = torch.ops.aten.copy_.default(primals_639, add_33);  primals_639 = add_33 = None
    copy__11: "i64[]" = torch.ops.aten.copy_.default(primals_640, add_30);  primals_640 = add_30 = None
    copy__12: "f32[768]" = torch.ops.aten.copy_.default(primals_641, add_49);  primals_641 = add_49 = None
    copy__13: "f32[768]" = torch.ops.aten.copy_.default(primals_642, add_50);  primals_642 = add_50 = None
    copy__14: "i64[]" = torch.ops.aten.copy_.default(primals_643, add_47);  primals_643 = add_47 = None
    copy__15: "f32[768]" = torch.ops.aten.copy_.default(primals_644, add_66);  primals_644 = add_66 = None
    copy__16: "f32[768]" = torch.ops.aten.copy_.default(primals_645, add_67);  primals_645 = add_67 = None
    copy__17: "i64[]" = torch.ops.aten.copy_.default(primals_646, add_64);  primals_646 = add_64 = None
    copy__18: "f32[768]" = torch.ops.aten.copy_.default(primals_647, add_83);  primals_647 = add_83 = None
    copy__19: "f32[768]" = torch.ops.aten.copy_.default(primals_648, add_84);  primals_648 = add_84 = None
    copy__20: "i64[]" = torch.ops.aten.copy_.default(primals_649, add_81);  primals_649 = add_81 = None
    copy__21: "f32[768]" = torch.ops.aten.copy_.default(primals_650, add_100);  primals_650 = add_100 = None
    copy__22: "f32[768]" = torch.ops.aten.copy_.default(primals_651, add_101);  primals_651 = add_101 = None
    copy__23: "i64[]" = torch.ops.aten.copy_.default(primals_652, add_98);  primals_652 = add_98 = None
    copy__24: "f32[768]" = torch.ops.aten.copy_.default(primals_653, add_117);  primals_653 = add_117 = None
    copy__25: "f32[768]" = torch.ops.aten.copy_.default(primals_654, add_118);  primals_654 = add_118 = None
    copy__26: "i64[]" = torch.ops.aten.copy_.default(primals_655, add_115);  primals_655 = add_115 = None
    copy__27: "f32[768]" = torch.ops.aten.copy_.default(primals_656, add_134);  primals_656 = add_134 = None
    copy__28: "f32[768]" = torch.ops.aten.copy_.default(primals_657, add_135);  primals_657 = add_135 = None
    copy__29: "i64[]" = torch.ops.aten.copy_.default(primals_658, add_132);  primals_658 = add_132 = None
    copy__30: "f32[768]" = torch.ops.aten.copy_.default(primals_659, add_151);  primals_659 = add_151 = None
    copy__31: "f32[768]" = torch.ops.aten.copy_.default(primals_660, add_152);  primals_660 = add_152 = None
    copy__32: "i64[]" = torch.ops.aten.copy_.default(primals_661, add_149);  primals_661 = add_149 = None
    copy__33: "f32[768]" = torch.ops.aten.copy_.default(primals_662, add_168);  primals_662 = add_168 = None
    copy__34: "f32[768]" = torch.ops.aten.copy_.default(primals_663, add_169);  primals_663 = add_169 = None
    copy__35: "i64[]" = torch.ops.aten.copy_.default(primals_664, add_166);  primals_664 = add_166 = None
    copy__36: "f32[768]" = torch.ops.aten.copy_.default(primals_665, add_185);  primals_665 = add_185 = None
    copy__37: "f32[768]" = torch.ops.aten.copy_.default(primals_666, add_186);  primals_666 = add_186 = None
    copy__38: "i64[]" = torch.ops.aten.copy_.default(primals_667, add_183);  primals_667 = add_183 = None
    copy__39: "f32[768]" = torch.ops.aten.copy_.default(primals_668, add_202);  primals_668 = add_202 = None
    copy__40: "f32[768]" = torch.ops.aten.copy_.default(primals_669, add_203);  primals_669 = add_203 = None
    copy__41: "i64[]" = torch.ops.aten.copy_.default(primals_670, add_200);  primals_670 = add_200 = None
    copy__42: "f32[768]" = torch.ops.aten.copy_.default(primals_671, add_219);  primals_671 = add_219 = None
    copy__43: "f32[768]" = torch.ops.aten.copy_.default(primals_672, add_220);  primals_672 = add_220 = None
    copy__44: "i64[]" = torch.ops.aten.copy_.default(primals_673, add_217);  primals_673 = add_217 = None
    copy__45: "f32[768]" = torch.ops.aten.copy_.default(primals_674, add_236);  primals_674 = add_236 = None
    copy__46: "f32[768]" = torch.ops.aten.copy_.default(primals_675, add_237);  primals_675 = add_237 = None
    copy__47: "i64[]" = torch.ops.aten.copy_.default(primals_676, add_234);  primals_676 = add_234 = None
    copy__48: "f32[768]" = torch.ops.aten.copy_.default(primals_677, add_253);  primals_677 = add_253 = None
    copy__49: "f32[768]" = torch.ops.aten.copy_.default(primals_678, add_254);  primals_678 = add_254 = None
    copy__50: "i64[]" = torch.ops.aten.copy_.default(primals_679, add_251);  primals_679 = add_251 = None
    copy__51: "f32[768]" = torch.ops.aten.copy_.default(primals_680, add_270);  primals_680 = add_270 = None
    copy__52: "f32[768]" = torch.ops.aten.copy_.default(primals_681, add_271);  primals_681 = add_271 = None
    copy__53: "i64[]" = torch.ops.aten.copy_.default(primals_682, add_268);  primals_682 = add_268 = None
    copy__54: "f32[768]" = torch.ops.aten.copy_.default(primals_683, add_287);  primals_683 = add_287 = None
    copy__55: "f32[768]" = torch.ops.aten.copy_.default(primals_684, add_288);  primals_684 = add_288 = None
    copy__56: "i64[]" = torch.ops.aten.copy_.default(primals_685, add_285);  primals_685 = add_285 = None
    copy__57: "f32[768]" = torch.ops.aten.copy_.default(primals_686, add_304);  primals_686 = add_304 = None
    copy__58: "f32[768]" = torch.ops.aten.copy_.default(primals_687, add_305);  primals_687 = add_305 = None
    copy__59: "i64[]" = torch.ops.aten.copy_.default(primals_688, add_302);  primals_688 = add_302 = None
    copy__60: "f32[768]" = torch.ops.aten.copy_.default(primals_689, add_321);  primals_689 = add_321 = None
    copy__61: "f32[768]" = torch.ops.aten.copy_.default(primals_690, add_322);  primals_690 = add_322 = None
    copy__62: "i64[]" = torch.ops.aten.copy_.default(primals_691, add_319);  primals_691 = add_319 = None
    copy__63: "f32[768]" = torch.ops.aten.copy_.default(primals_692, add_338);  primals_692 = add_338 = None
    copy__64: "f32[768]" = torch.ops.aten.copy_.default(primals_693, add_339);  primals_693 = add_339 = None
    copy__65: "i64[]" = torch.ops.aten.copy_.default(primals_694, add_336);  primals_694 = add_336 = None
    copy__66: "f32[768]" = torch.ops.aten.copy_.default(primals_695, add_355);  primals_695 = add_355 = None
    copy__67: "f32[768]" = torch.ops.aten.copy_.default(primals_696, add_356);  primals_696 = add_356 = None
    copy__68: "i64[]" = torch.ops.aten.copy_.default(primals_697, add_353);  primals_697 = add_353 = None
    copy__69: "f32[768]" = torch.ops.aten.copy_.default(primals_698, add_372);  primals_698 = add_372 = None
    copy__70: "f32[768]" = torch.ops.aten.copy_.default(primals_699, add_373);  primals_699 = add_373 = None
    copy__71: "i64[]" = torch.ops.aten.copy_.default(primals_700, add_370);  primals_700 = add_370 = None
    copy__72: "f32[768]" = torch.ops.aten.copy_.default(primals_701, add_389);  primals_701 = add_389 = None
    copy__73: "f32[768]" = torch.ops.aten.copy_.default(primals_702, add_390);  primals_702 = add_390 = None
    copy__74: "i64[]" = torch.ops.aten.copy_.default(primals_703, add_387);  primals_703 = add_387 = None
    copy__75: "f32[768]" = torch.ops.aten.copy_.default(primals_704, add_406);  primals_704 = add_406 = None
    copy__76: "f32[768]" = torch.ops.aten.copy_.default(primals_705, add_407);  primals_705 = add_407 = None
    copy__77: "i64[]" = torch.ops.aten.copy_.default(primals_706, add_404);  primals_706 = add_404 = None
    copy__78: "f32[768]" = torch.ops.aten.copy_.default(primals_707, add_423);  primals_707 = add_423 = None
    copy__79: "f32[768]" = torch.ops.aten.copy_.default(primals_708, add_424);  primals_708 = add_424 = None
    copy__80: "i64[]" = torch.ops.aten.copy_.default(primals_709, add_421);  primals_709 = add_421 = None
    return [addmm_82, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_113, primals_118, primals_119, primals_121, primals_123, primals_125, primals_127, primals_133, primals_138, primals_139, primals_141, primals_143, primals_145, primals_147, primals_153, primals_158, primals_159, primals_161, primals_163, primals_165, primals_167, primals_173, primals_178, primals_179, primals_181, primals_183, primals_185, primals_187, primals_193, primals_198, primals_199, primals_201, primals_203, primals_205, primals_207, primals_213, primals_218, primals_219, primals_221, primals_223, primals_225, primals_227, primals_233, primals_238, primals_239, primals_241, primals_243, primals_245, primals_247, primals_253, primals_258, primals_259, primals_261, primals_263, primals_265, primals_267, primals_273, primals_278, primals_279, primals_281, primals_283, primals_285, primals_287, primals_293, primals_298, primals_299, primals_301, primals_303, primals_305, primals_307, primals_313, primals_318, primals_319, primals_321, primals_323, primals_325, primals_327, primals_333, primals_338, primals_339, primals_341, primals_343, primals_345, primals_347, primals_353, primals_358, primals_359, primals_361, primals_363, primals_365, primals_367, primals_373, primals_378, primals_379, primals_381, primals_383, primals_385, primals_387, primals_393, primals_398, primals_399, primals_401, primals_403, primals_405, primals_407, primals_413, primals_418, primals_419, primals_421, primals_423, primals_425, primals_427, primals_433, primals_438, primals_439, primals_441, primals_443, primals_445, primals_447, primals_453, primals_458, primals_459, primals_461, primals_463, primals_465, primals_467, primals_473, primals_478, primals_479, primals_481, primals_483, primals_485, primals_487, primals_493, primals_498, primals_499, primals_501, primals_503, primals_505, primals_507, primals_513, primals_518, primals_519, primals_521, primals_523, primals_525, primals_527, primals_533, primals_538, primals_539, primals_541, primals_543, primals_545, primals_547, primals_553, primals_558, primals_559, primals_561, primals_563, primals_565, primals_567, primals_573, primals_578, primals_579, primals_581, primals_583, primals_585, primals_587, primals_593, primals_603, primals_606, primals_609, primals_619, primals_622, primals_625, primals_710, convolution, squeeze_1, mul_9, convolution_1, squeeze_4, mul_19, convolution_2, squeeze_7, permute_1, mul_33, view_4, getitem_8, getitem_9, pow_3, pow_5, bmm, view_14, mm, mul_37, view_16, convolution_4, squeeze_10, add_34, convolution_5, mul_50, view_18, addmm_1, view_20, addmm_2, mul_56, view_22, getitem_19, getitem_20, pow_7, pow_9, bmm_2, view_32, mm_1, mul_60, view_34, convolution_6, squeeze_13, add_51, convolution_7, mul_73, view_36, addmm_4, view_38, addmm_5, mul_79, view_40, getitem_30, getitem_31, pow_11, pow_13, bmm_4, view_50, mm_2, mul_83, view_52, convolution_8, squeeze_16, add_68, convolution_9, mul_96, view_54, addmm_7, view_56, addmm_8, mul_102, view_58, getitem_41, getitem_42, pow_15, pow_17, bmm_6, view_68, mm_3, mul_106, view_70, convolution_10, squeeze_19, add_85, convolution_11, mul_119, view_72, addmm_10, view_74, addmm_11, mul_125, view_76, getitem_52, getitem_53, pow_19, pow_21, bmm_8, view_86, mm_4, mul_129, view_88, convolution_12, squeeze_22, add_102, convolution_13, mul_142, view_90, addmm_13, view_92, addmm_14, mul_148, view_94, getitem_63, getitem_64, pow_23, pow_25, bmm_10, view_104, mm_5, mul_152, view_106, convolution_14, squeeze_25, add_119, convolution_15, mul_165, view_108, addmm_16, view_110, addmm_17, mul_171, view_112, getitem_74, getitem_75, pow_27, pow_29, bmm_12, view_122, mm_6, mul_175, view_124, convolution_16, squeeze_28, add_136, convolution_17, mul_188, view_126, addmm_19, view_128, addmm_20, mul_194, view_130, getitem_85, getitem_86, pow_31, pow_33, bmm_14, view_140, mm_7, mul_198, view_142, convolution_18, squeeze_31, add_153, convolution_19, mul_211, view_144, addmm_22, view_146, addmm_23, mul_217, view_148, getitem_96, getitem_97, pow_35, pow_37, bmm_16, view_158, mm_8, mul_221, view_160, convolution_20, squeeze_34, add_170, convolution_21, mul_234, view_162, addmm_25, view_164, addmm_26, mul_240, view_166, getitem_107, getitem_108, pow_39, pow_41, bmm_18, view_176, mm_9, mul_244, view_178, convolution_22, squeeze_37, add_187, convolution_23, mul_257, view_180, addmm_28, view_182, addmm_29, mul_263, view_184, getitem_118, getitem_119, pow_43, pow_45, bmm_20, view_194, mm_10, mul_267, view_196, convolution_24, squeeze_40, add_204, convolution_25, mul_280, view_198, addmm_31, view_200, addmm_32, mul_286, view_202, getitem_129, getitem_130, pow_47, pow_49, bmm_22, view_212, mm_11, mul_290, view_214, convolution_26, squeeze_43, add_221, convolution_27, mul_303, view_216, addmm_34, view_218, addmm_35, mul_309, view_220, getitem_140, getitem_141, pow_51, pow_53, bmm_24, view_230, mm_12, mul_313, view_232, convolution_28, squeeze_46, add_238, convolution_29, mul_326, view_234, addmm_37, view_236, addmm_38, mul_332, view_238, getitem_151, getitem_152, pow_55, pow_57, bmm_26, view_248, mm_13, mul_336, view_250, convolution_30, squeeze_49, add_255, convolution_31, mul_349, view_252, addmm_40, view_254, addmm_41, mul_355, view_256, getitem_162, getitem_163, pow_59, pow_61, bmm_28, view_266, mm_14, mul_359, view_268, convolution_32, squeeze_52, add_272, convolution_33, mul_372, view_270, addmm_43, view_272, addmm_44, mul_378, view_274, getitem_173, getitem_174, pow_63, pow_65, bmm_30, view_284, mm_15, mul_382, view_286, convolution_34, squeeze_55, add_289, convolution_35, mul_395, view_288, addmm_46, view_290, addmm_47, mul_401, view_292, getitem_184, getitem_185, pow_67, pow_69, bmm_32, view_302, mm_16, mul_405, view_304, convolution_36, squeeze_58, add_306, convolution_37, mul_418, view_306, addmm_49, view_308, addmm_50, mul_424, view_310, getitem_195, getitem_196, pow_71, pow_73, bmm_34, view_320, mm_17, mul_428, view_322, convolution_38, squeeze_61, add_323, convolution_39, mul_441, view_324, addmm_52, view_326, addmm_53, mul_447, view_328, getitem_206, getitem_207, pow_75, pow_77, bmm_36, view_338, mm_18, mul_451, view_340, convolution_40, squeeze_64, add_340, convolution_41, mul_464, view_342, addmm_55, view_344, addmm_56, mul_470, view_346, getitem_217, getitem_218, pow_79, pow_81, bmm_38, view_356, mm_19, mul_474, view_358, convolution_42, squeeze_67, add_357, convolution_43, mul_487, view_360, addmm_58, view_362, addmm_59, mul_493, view_364, getitem_228, getitem_229, pow_83, pow_85, bmm_40, view_374, mm_20, mul_497, view_376, convolution_44, squeeze_70, add_374, convolution_45, mul_510, view_378, addmm_61, view_380, addmm_62, mul_516, view_382, getitem_239, getitem_240, pow_87, pow_89, bmm_42, view_392, mm_21, mul_520, view_394, convolution_46, squeeze_73, add_391, convolution_47, mul_533, view_396, addmm_64, view_398, addmm_65, mul_539, view_400, getitem_250, getitem_251, pow_91, pow_93, bmm_44, view_410, mm_22, mul_543, view_412, convolution_48, squeeze_76, add_408, convolution_49, mul_556, view_414, addmm_67, view_416, addmm_68, mul_562, view_418, getitem_261, getitem_262, pow_95, pow_97, bmm_46, view_428, mm_23, mul_566, view_430, convolution_50, squeeze_79, add_425, convolution_51, mul_579, view_432, addmm_70, view_434, addmm_71, cat_3, getitem_271, rsqrt_99, select, permute_220, view_437, permute_222, permute_224, getitem_273, getitem_274, getitem_275, view_444, cat_4, mul_588, view_446, mm_24, view_448, addmm_76, mul_594, select_1, permute_230, view_451, permute_232, permute_234, getitem_281, getitem_282, getitem_283, view_458, cat_6, mul_597, view_460, mm_25, view_462, addmm_81, mul_603, clone_271, permute_240, div_78, permute_244, permute_250, div_79, permute_252, alias_74, permute_258, permute_263, permute_268, div_80, permute_272, permute_278, div_81, permute_280, alias_75, permute_286, permute_291, permute_296, permute_300, permute_304, div_83, unsqueeze_119, div_84, permute_312, permute_315, permute_316, alias_76, permute_317, permute_318, permute_321, div_93, permute_325, permute_329, div_94, unsqueeze_131, div_95, permute_337, permute_340, permute_341, alias_79, permute_342, permute_343, permute_346, div_104, permute_350, permute_354, div_105, unsqueeze_143, div_106, permute_362, permute_365, permute_366, alias_82, permute_367, permute_368, permute_371, div_115, permute_375, permute_379, div_116, unsqueeze_155, div_117, permute_387, permute_390, permute_391, alias_85, permute_392, permute_393, permute_396, div_126, permute_400, permute_404, div_127, unsqueeze_167, div_128, permute_412, permute_415, permute_416, alias_88, permute_417, permute_418, permute_421, div_137, permute_425, permute_429, div_138, unsqueeze_179, div_139, permute_437, permute_440, permute_441, alias_91, permute_442, permute_443, permute_446, div_148, permute_450, permute_454, div_149, unsqueeze_191, div_150, permute_462, permute_465, permute_466, alias_94, permute_467, permute_468, permute_471, div_159, permute_475, permute_479, div_160, unsqueeze_203, div_161, permute_487, permute_490, permute_491, alias_97, permute_492, permute_493, permute_496, div_170, permute_500, permute_504, div_171, unsqueeze_215, div_172, permute_512, permute_515, permute_516, alias_100, permute_517, permute_518, permute_521, div_181, permute_525, permute_529, div_182, unsqueeze_227, div_183, permute_537, permute_540, permute_541, alias_103, permute_542, permute_543, permute_546, div_192, permute_550, permute_554, div_193, unsqueeze_239, div_194, permute_562, permute_565, permute_566, alias_106, permute_567, permute_568, permute_571, div_203, permute_575, permute_579, div_204, unsqueeze_251, div_205, permute_587, permute_590, permute_591, alias_109, permute_592, permute_593, permute_596, div_214, permute_600, permute_604, div_215, unsqueeze_263, div_216, permute_612, permute_615, permute_616, alias_112, permute_617, permute_618, permute_621, div_225, permute_625, permute_629, div_226, unsqueeze_275, div_227, permute_637, permute_640, permute_641, alias_115, permute_642, permute_643, permute_646, div_236, permute_650, permute_654, div_237, unsqueeze_287, div_238, permute_662, permute_665, permute_666, alias_118, permute_667, permute_668, permute_671, div_247, permute_675, permute_679, div_248, unsqueeze_299, div_249, permute_687, permute_690, permute_691, alias_121, permute_692, permute_693, permute_696, div_258, permute_700, permute_704, div_259, unsqueeze_311, div_260, permute_712, permute_715, permute_716, alias_124, permute_717, permute_718, permute_721, div_269, permute_725, permute_729, div_270, unsqueeze_323, div_271, permute_737, permute_740, permute_741, alias_127, permute_742, permute_743, permute_746, div_280, permute_750, permute_754, div_281, unsqueeze_335, div_282, permute_762, permute_765, permute_766, alias_130, permute_767, permute_768, permute_771, div_291, permute_775, permute_779, div_292, unsqueeze_347, div_293, permute_787, permute_790, permute_791, alias_133, permute_792, permute_793, permute_796, div_302, permute_800, permute_804, div_303, unsqueeze_359, div_304, permute_812, permute_815, permute_816, alias_136, permute_817, permute_818, permute_821, div_313, permute_825, permute_829, div_314, unsqueeze_371, div_315, permute_837, permute_840, permute_841, alias_139, permute_842, permute_843, permute_846, div_324, permute_850, permute_854, div_325, unsqueeze_383, div_326, permute_862, permute_865, permute_866, alias_142, permute_867, permute_868, permute_871, div_335, permute_875, permute_879, div_336, unsqueeze_395, div_337, permute_887, permute_890, permute_891, alias_145, permute_892, permute_893, permute_896, div_346, unsqueeze_407, add_682, unsqueeze_419, add_684, unsqueeze_431]
    