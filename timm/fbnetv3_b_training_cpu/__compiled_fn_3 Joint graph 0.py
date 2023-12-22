from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[16]"; primals_2: "f32[16]"; primals_3: "f32[16]"; primals_4: "f32[16]"; primals_5: "f32[16]"; primals_6: "f32[16]"; primals_7: "f32[16]"; primals_8: "f32[16]"; primals_9: "f32[16]"; primals_10: "f32[16]"; primals_11: "f32[64]"; primals_12: "f32[64]"; primals_13: "f32[64]"; primals_14: "f32[64]"; primals_15: "f32[24]"; primals_16: "f32[24]"; primals_17: "f32[48]"; primals_18: "f32[48]"; primals_19: "f32[48]"; primals_20: "f32[48]"; primals_21: "f32[24]"; primals_22: "f32[24]"; primals_23: "f32[48]"; primals_24: "f32[48]"; primals_25: "f32[48]"; primals_26: "f32[48]"; primals_27: "f32[24]"; primals_28: "f32[24]"; primals_29: "f32[48]"; primals_30: "f32[48]"; primals_31: "f32[48]"; primals_32: "f32[48]"; primals_33: "f32[24]"; primals_34: "f32[24]"; primals_35: "f32[120]"; primals_36: "f32[120]"; primals_37: "f32[120]"; primals_38: "f32[120]"; primals_39: "f32[40]"; primals_40: "f32[40]"; primals_41: "f32[120]"; primals_42: "f32[120]"; primals_43: "f32[120]"; primals_44: "f32[120]"; primals_45: "f32[40]"; primals_46: "f32[40]"; primals_47: "f32[120]"; primals_48: "f32[120]"; primals_49: "f32[120]"; primals_50: "f32[120]"; primals_51: "f32[40]"; primals_52: "f32[40]"; primals_53: "f32[120]"; primals_54: "f32[120]"; primals_55: "f32[120]"; primals_56: "f32[120]"; primals_57: "f32[40]"; primals_58: "f32[40]"; primals_59: "f32[120]"; primals_60: "f32[120]"; primals_61: "f32[120]"; primals_62: "f32[120]"; primals_63: "f32[40]"; primals_64: "f32[40]"; primals_65: "f32[200]"; primals_66: "f32[200]"; primals_67: "f32[200]"; primals_68: "f32[200]"; primals_69: "f32[72]"; primals_70: "f32[72]"; primals_71: "f32[216]"; primals_72: "f32[216]"; primals_73: "f32[216]"; primals_74: "f32[216]"; primals_75: "f32[72]"; primals_76: "f32[72]"; primals_77: "f32[216]"; primals_78: "f32[216]"; primals_79: "f32[216]"; primals_80: "f32[216]"; primals_81: "f32[72]"; primals_82: "f32[72]"; primals_83: "f32[216]"; primals_84: "f32[216]"; primals_85: "f32[216]"; primals_86: "f32[216]"; primals_87: "f32[72]"; primals_88: "f32[72]"; primals_89: "f32[216]"; primals_90: "f32[216]"; primals_91: "f32[216]"; primals_92: "f32[216]"; primals_93: "f32[72]"; primals_94: "f32[72]"; primals_95: "f32[360]"; primals_96: "f32[360]"; primals_97: "f32[360]"; primals_98: "f32[360]"; primals_99: "f32[120]"; primals_100: "f32[120]"; primals_101: "f32[360]"; primals_102: "f32[360]"; primals_103: "f32[360]"; primals_104: "f32[360]"; primals_105: "f32[120]"; primals_106: "f32[120]"; primals_107: "f32[360]"; primals_108: "f32[360]"; primals_109: "f32[360]"; primals_110: "f32[360]"; primals_111: "f32[120]"; primals_112: "f32[120]"; primals_113: "f32[360]"; primals_114: "f32[360]"; primals_115: "f32[360]"; primals_116: "f32[360]"; primals_117: "f32[120]"; primals_118: "f32[120]"; primals_119: "f32[360]"; primals_120: "f32[360]"; primals_121: "f32[360]"; primals_122: "f32[360]"; primals_123: "f32[120]"; primals_124: "f32[120]"; primals_125: "f32[360]"; primals_126: "f32[360]"; primals_127: "f32[360]"; primals_128: "f32[360]"; primals_129: "f32[120]"; primals_130: "f32[120]"; primals_131: "f32[720]"; primals_132: "f32[720]"; primals_133: "f32[720]"; primals_134: "f32[720]"; primals_135: "f32[184]"; primals_136: "f32[184]"; primals_137: "f32[736]"; primals_138: "f32[736]"; primals_139: "f32[736]"; primals_140: "f32[736]"; primals_141: "f32[184]"; primals_142: "f32[184]"; primals_143: "f32[736]"; primals_144: "f32[736]"; primals_145: "f32[736]"; primals_146: "f32[736]"; primals_147: "f32[184]"; primals_148: "f32[184]"; primals_149: "f32[736]"; primals_150: "f32[736]"; primals_151: "f32[736]"; primals_152: "f32[736]"; primals_153: "f32[184]"; primals_154: "f32[184]"; primals_155: "f32[736]"; primals_156: "f32[736]"; primals_157: "f32[736]"; primals_158: "f32[736]"; primals_159: "f32[184]"; primals_160: "f32[184]"; primals_161: "f32[736]"; primals_162: "f32[736]"; primals_163: "f32[736]"; primals_164: "f32[736]"; primals_165: "f32[184]"; primals_166: "f32[184]"; primals_167: "f32[1104]"; primals_168: "f32[1104]"; primals_169: "f32[1104]"; primals_170: "f32[1104]"; primals_171: "f32[224]"; primals_172: "f32[224]"; primals_173: "f32[1344]"; primals_174: "f32[1344]"; primals_175: "f32[1000, 1984]"; primals_176: "f32[1000]"; primals_177: "f32[16, 3, 3, 3]"; primals_178: "f32[16, 1, 3, 3]"; primals_179: "f32[16, 16, 1, 1]"; primals_180: "f32[16, 1, 3, 3]"; primals_181: "f32[16, 16, 1, 1]"; primals_182: "f32[64, 16, 1, 1]"; primals_183: "f32[64, 1, 5, 5]"; primals_184: "f32[24, 64, 1, 1]"; primals_185: "f32[48, 24, 1, 1]"; primals_186: "f32[48, 1, 5, 5]"; primals_187: "f32[24, 48, 1, 1]"; primals_188: "f32[48, 24, 1, 1]"; primals_189: "f32[48, 1, 5, 5]"; primals_190: "f32[24, 48, 1, 1]"; primals_191: "f32[48, 24, 1, 1]"; primals_192: "f32[48, 1, 5, 5]"; primals_193: "f32[24, 48, 1, 1]"; primals_194: "f32[120, 24, 1, 1]"; primals_195: "f32[120, 1, 5, 5]"; primals_196: "f32[8, 120, 1, 1]"; primals_197: "f32[8]"; primals_198: "f32[120, 8, 1, 1]"; primals_199: "f32[120]"; primals_200: "f32[40, 120, 1, 1]"; primals_201: "f32[120, 40, 1, 1]"; primals_202: "f32[120, 1, 5, 5]"; primals_203: "f32[16, 120, 1, 1]"; primals_204: "f32[16]"; primals_205: "f32[120, 16, 1, 1]"; primals_206: "f32[120]"; primals_207: "f32[40, 120, 1, 1]"; primals_208: "f32[120, 40, 1, 1]"; primals_209: "f32[120, 1, 5, 5]"; primals_210: "f32[16, 120, 1, 1]"; primals_211: "f32[16]"; primals_212: "f32[120, 16, 1, 1]"; primals_213: "f32[120]"; primals_214: "f32[40, 120, 1, 1]"; primals_215: "f32[120, 40, 1, 1]"; primals_216: "f32[120, 1, 5, 5]"; primals_217: "f32[16, 120, 1, 1]"; primals_218: "f32[16]"; primals_219: "f32[120, 16, 1, 1]"; primals_220: "f32[120]"; primals_221: "f32[40, 120, 1, 1]"; primals_222: "f32[120, 40, 1, 1]"; primals_223: "f32[120, 1, 5, 5]"; primals_224: "f32[16, 120, 1, 1]"; primals_225: "f32[16]"; primals_226: "f32[120, 16, 1, 1]"; primals_227: "f32[120]"; primals_228: "f32[40, 120, 1, 1]"; primals_229: "f32[200, 40, 1, 1]"; primals_230: "f32[200, 1, 5, 5]"; primals_231: "f32[72, 200, 1, 1]"; primals_232: "f32[216, 72, 1, 1]"; primals_233: "f32[216, 1, 3, 3]"; primals_234: "f32[72, 216, 1, 1]"; primals_235: "f32[216, 72, 1, 1]"; primals_236: "f32[216, 1, 3, 3]"; primals_237: "f32[72, 216, 1, 1]"; primals_238: "f32[216, 72, 1, 1]"; primals_239: "f32[216, 1, 3, 3]"; primals_240: "f32[72, 216, 1, 1]"; primals_241: "f32[216, 72, 1, 1]"; primals_242: "f32[216, 1, 3, 3]"; primals_243: "f32[72, 216, 1, 1]"; primals_244: "f32[360, 72, 1, 1]"; primals_245: "f32[360, 1, 3, 3]"; primals_246: "f32[24, 360, 1, 1]"; primals_247: "f32[24]"; primals_248: "f32[360, 24, 1, 1]"; primals_249: "f32[360]"; primals_250: "f32[120, 360, 1, 1]"; primals_251: "f32[360, 120, 1, 1]"; primals_252: "f32[360, 1, 5, 5]"; primals_253: "f32[32, 360, 1, 1]"; primals_254: "f32[32]"; primals_255: "f32[360, 32, 1, 1]"; primals_256: "f32[360]"; primals_257: "f32[120, 360, 1, 1]"; primals_258: "f32[360, 120, 1, 1]"; primals_259: "f32[360, 1, 5, 5]"; primals_260: "f32[32, 360, 1, 1]"; primals_261: "f32[32]"; primals_262: "f32[360, 32, 1, 1]"; primals_263: "f32[360]"; primals_264: "f32[120, 360, 1, 1]"; primals_265: "f32[360, 120, 1, 1]"; primals_266: "f32[360, 1, 5, 5]"; primals_267: "f32[32, 360, 1, 1]"; primals_268: "f32[32]"; primals_269: "f32[360, 32, 1, 1]"; primals_270: "f32[360]"; primals_271: "f32[120, 360, 1, 1]"; primals_272: "f32[360, 120, 1, 1]"; primals_273: "f32[360, 1, 5, 5]"; primals_274: "f32[32, 360, 1, 1]"; primals_275: "f32[32]"; primals_276: "f32[360, 32, 1, 1]"; primals_277: "f32[360]"; primals_278: "f32[120, 360, 1, 1]"; primals_279: "f32[360, 120, 1, 1]"; primals_280: "f32[360, 1, 5, 5]"; primals_281: "f32[32, 360, 1, 1]"; primals_282: "f32[32]"; primals_283: "f32[360, 32, 1, 1]"; primals_284: "f32[360]"; primals_285: "f32[120, 360, 1, 1]"; primals_286: "f32[720, 120, 1, 1]"; primals_287: "f32[720, 1, 3, 3]"; primals_288: "f32[32, 720, 1, 1]"; primals_289: "f32[32]"; primals_290: "f32[720, 32, 1, 1]"; primals_291: "f32[720]"; primals_292: "f32[184, 720, 1, 1]"; primals_293: "f32[736, 184, 1, 1]"; primals_294: "f32[736, 1, 5, 5]"; primals_295: "f32[48, 736, 1, 1]"; primals_296: "f32[48]"; primals_297: "f32[736, 48, 1, 1]"; primals_298: "f32[736]"; primals_299: "f32[184, 736, 1, 1]"; primals_300: "f32[736, 184, 1, 1]"; primals_301: "f32[736, 1, 5, 5]"; primals_302: "f32[48, 736, 1, 1]"; primals_303: "f32[48]"; primals_304: "f32[736, 48, 1, 1]"; primals_305: "f32[736]"; primals_306: "f32[184, 736, 1, 1]"; primals_307: "f32[736, 184, 1, 1]"; primals_308: "f32[736, 1, 5, 5]"; primals_309: "f32[48, 736, 1, 1]"; primals_310: "f32[48]"; primals_311: "f32[736, 48, 1, 1]"; primals_312: "f32[736]"; primals_313: "f32[184, 736, 1, 1]"; primals_314: "f32[736, 184, 1, 1]"; primals_315: "f32[736, 1, 5, 5]"; primals_316: "f32[48, 736, 1, 1]"; primals_317: "f32[48]"; primals_318: "f32[736, 48, 1, 1]"; primals_319: "f32[736]"; primals_320: "f32[184, 736, 1, 1]"; primals_321: "f32[736, 184, 1, 1]"; primals_322: "f32[736, 1, 5, 5]"; primals_323: "f32[48, 736, 1, 1]"; primals_324: "f32[48]"; primals_325: "f32[736, 48, 1, 1]"; primals_326: "f32[736]"; primals_327: "f32[184, 736, 1, 1]"; primals_328: "f32[1104, 184, 1, 1]"; primals_329: "f32[1104, 1, 5, 5]"; primals_330: "f32[48, 1104, 1, 1]"; primals_331: "f32[48]"; primals_332: "f32[1104, 48, 1, 1]"; primals_333: "f32[1104]"; primals_334: "f32[224, 1104, 1, 1]"; primals_335: "f32[1344, 224, 1, 1]"; primals_336: "f32[1984, 1344, 1, 1]"; primals_337: "i64[]"; primals_338: "f32[16]"; primals_339: "f32[16]"; primals_340: "i64[]"; primals_341: "f32[16]"; primals_342: "f32[16]"; primals_343: "i64[]"; primals_344: "f32[16]"; primals_345: "f32[16]"; primals_346: "i64[]"; primals_347: "f32[16]"; primals_348: "f32[16]"; primals_349: "i64[]"; primals_350: "f32[16]"; primals_351: "f32[16]"; primals_352: "i64[]"; primals_353: "f32[64]"; primals_354: "f32[64]"; primals_355: "i64[]"; primals_356: "f32[64]"; primals_357: "f32[64]"; primals_358: "i64[]"; primals_359: "f32[24]"; primals_360: "f32[24]"; primals_361: "i64[]"; primals_362: "f32[48]"; primals_363: "f32[48]"; primals_364: "i64[]"; primals_365: "f32[48]"; primals_366: "f32[48]"; primals_367: "i64[]"; primals_368: "f32[24]"; primals_369: "f32[24]"; primals_370: "i64[]"; primals_371: "f32[48]"; primals_372: "f32[48]"; primals_373: "i64[]"; primals_374: "f32[48]"; primals_375: "f32[48]"; primals_376: "i64[]"; primals_377: "f32[24]"; primals_378: "f32[24]"; primals_379: "i64[]"; primals_380: "f32[48]"; primals_381: "f32[48]"; primals_382: "i64[]"; primals_383: "f32[48]"; primals_384: "f32[48]"; primals_385: "i64[]"; primals_386: "f32[24]"; primals_387: "f32[24]"; primals_388: "i64[]"; primals_389: "f32[120]"; primals_390: "f32[120]"; primals_391: "i64[]"; primals_392: "f32[120]"; primals_393: "f32[120]"; primals_394: "i64[]"; primals_395: "f32[40]"; primals_396: "f32[40]"; primals_397: "i64[]"; primals_398: "f32[120]"; primals_399: "f32[120]"; primals_400: "i64[]"; primals_401: "f32[120]"; primals_402: "f32[120]"; primals_403: "i64[]"; primals_404: "f32[40]"; primals_405: "f32[40]"; primals_406: "i64[]"; primals_407: "f32[120]"; primals_408: "f32[120]"; primals_409: "i64[]"; primals_410: "f32[120]"; primals_411: "f32[120]"; primals_412: "i64[]"; primals_413: "f32[40]"; primals_414: "f32[40]"; primals_415: "i64[]"; primals_416: "f32[120]"; primals_417: "f32[120]"; primals_418: "i64[]"; primals_419: "f32[120]"; primals_420: "f32[120]"; primals_421: "i64[]"; primals_422: "f32[40]"; primals_423: "f32[40]"; primals_424: "i64[]"; primals_425: "f32[120]"; primals_426: "f32[120]"; primals_427: "i64[]"; primals_428: "f32[120]"; primals_429: "f32[120]"; primals_430: "i64[]"; primals_431: "f32[40]"; primals_432: "f32[40]"; primals_433: "i64[]"; primals_434: "f32[200]"; primals_435: "f32[200]"; primals_436: "i64[]"; primals_437: "f32[200]"; primals_438: "f32[200]"; primals_439: "i64[]"; primals_440: "f32[72]"; primals_441: "f32[72]"; primals_442: "i64[]"; primals_443: "f32[216]"; primals_444: "f32[216]"; primals_445: "i64[]"; primals_446: "f32[216]"; primals_447: "f32[216]"; primals_448: "i64[]"; primals_449: "f32[72]"; primals_450: "f32[72]"; primals_451: "i64[]"; primals_452: "f32[216]"; primals_453: "f32[216]"; primals_454: "i64[]"; primals_455: "f32[216]"; primals_456: "f32[216]"; primals_457: "i64[]"; primals_458: "f32[72]"; primals_459: "f32[72]"; primals_460: "i64[]"; primals_461: "f32[216]"; primals_462: "f32[216]"; primals_463: "i64[]"; primals_464: "f32[216]"; primals_465: "f32[216]"; primals_466: "i64[]"; primals_467: "f32[72]"; primals_468: "f32[72]"; primals_469: "i64[]"; primals_470: "f32[216]"; primals_471: "f32[216]"; primals_472: "i64[]"; primals_473: "f32[216]"; primals_474: "f32[216]"; primals_475: "i64[]"; primals_476: "f32[72]"; primals_477: "f32[72]"; primals_478: "i64[]"; primals_479: "f32[360]"; primals_480: "f32[360]"; primals_481: "i64[]"; primals_482: "f32[360]"; primals_483: "f32[360]"; primals_484: "i64[]"; primals_485: "f32[120]"; primals_486: "f32[120]"; primals_487: "i64[]"; primals_488: "f32[360]"; primals_489: "f32[360]"; primals_490: "i64[]"; primals_491: "f32[360]"; primals_492: "f32[360]"; primals_493: "i64[]"; primals_494: "f32[120]"; primals_495: "f32[120]"; primals_496: "i64[]"; primals_497: "f32[360]"; primals_498: "f32[360]"; primals_499: "i64[]"; primals_500: "f32[360]"; primals_501: "f32[360]"; primals_502: "i64[]"; primals_503: "f32[120]"; primals_504: "f32[120]"; primals_505: "i64[]"; primals_506: "f32[360]"; primals_507: "f32[360]"; primals_508: "i64[]"; primals_509: "f32[360]"; primals_510: "f32[360]"; primals_511: "i64[]"; primals_512: "f32[120]"; primals_513: "f32[120]"; primals_514: "i64[]"; primals_515: "f32[360]"; primals_516: "f32[360]"; primals_517: "i64[]"; primals_518: "f32[360]"; primals_519: "f32[360]"; primals_520: "i64[]"; primals_521: "f32[120]"; primals_522: "f32[120]"; primals_523: "i64[]"; primals_524: "f32[360]"; primals_525: "f32[360]"; primals_526: "i64[]"; primals_527: "f32[360]"; primals_528: "f32[360]"; primals_529: "i64[]"; primals_530: "f32[120]"; primals_531: "f32[120]"; primals_532: "i64[]"; primals_533: "f32[720]"; primals_534: "f32[720]"; primals_535: "i64[]"; primals_536: "f32[720]"; primals_537: "f32[720]"; primals_538: "i64[]"; primals_539: "f32[184]"; primals_540: "f32[184]"; primals_541: "i64[]"; primals_542: "f32[736]"; primals_543: "f32[736]"; primals_544: "i64[]"; primals_545: "f32[736]"; primals_546: "f32[736]"; primals_547: "i64[]"; primals_548: "f32[184]"; primals_549: "f32[184]"; primals_550: "i64[]"; primals_551: "f32[736]"; primals_552: "f32[736]"; primals_553: "i64[]"; primals_554: "f32[736]"; primals_555: "f32[736]"; primals_556: "i64[]"; primals_557: "f32[184]"; primals_558: "f32[184]"; primals_559: "i64[]"; primals_560: "f32[736]"; primals_561: "f32[736]"; primals_562: "i64[]"; primals_563: "f32[736]"; primals_564: "f32[736]"; primals_565: "i64[]"; primals_566: "f32[184]"; primals_567: "f32[184]"; primals_568: "i64[]"; primals_569: "f32[736]"; primals_570: "f32[736]"; primals_571: "i64[]"; primals_572: "f32[736]"; primals_573: "f32[736]"; primals_574: "i64[]"; primals_575: "f32[184]"; primals_576: "f32[184]"; primals_577: "i64[]"; primals_578: "f32[736]"; primals_579: "f32[736]"; primals_580: "i64[]"; primals_581: "f32[736]"; primals_582: "f32[736]"; primals_583: "i64[]"; primals_584: "f32[184]"; primals_585: "f32[184]"; primals_586: "i64[]"; primals_587: "f32[1104]"; primals_588: "f32[1104]"; primals_589: "i64[]"; primals_590: "f32[1104]"; primals_591: "f32[1104]"; primals_592: "i64[]"; primals_593: "f32[224]"; primals_594: "f32[224]"; primals_595: "i64[]"; primals_596: "f32[1344]"; primals_597: "f32[1344]"; primals_598: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:135, code: x = self.conv_stem(x)
    convolution: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(primals_598, primals_177, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add: "i64[]" = torch.ops.aten.add.Tensor(primals_337, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
    getitem: "f32[1, 16, 1, 1]" = var_mean[0]
    getitem_1: "f32[1, 16, 1, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
    rsqrt: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
    mul: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    squeeze: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
    squeeze_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
    mul_1: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
    mul_2: "f32[16]" = torch.ops.aten.mul.Tensor(primals_338, 0.9)
    add_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
    squeeze_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
    mul_3: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.00000996502277);  squeeze_2 = None
    mul_4: "f32[16]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
    mul_5: "f32[16]" = torch.ops.aten.mul.Tensor(primals_339, 0.9)
    add_3: "f32[16]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    unsqueeze: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_1, -1)
    unsqueeze_1: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
    mul_6: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
    unsqueeze_2: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_2, -1);  primals_2 = None
    unsqueeze_3: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
    add_4: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone: "f32[8, 16, 112, 112]" = torch.ops.aten.clone.default(add_4)
    add_5: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_4, 3)
    clamp_min: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min, 6);  clamp_min = None
    mul_7: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, clamp_max);  add_4 = clamp_max = None
    div: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_7, 6);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_1: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(div, primals_178, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_6: "i64[]" = torch.ops.aten.add.Tensor(primals_340, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
    getitem_2: "f32[1, 16, 1, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 16, 1, 1]" = var_mean_1[1];  var_mean_1 = None
    add_7: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
    rsqrt_1: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_1: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
    mul_8: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    squeeze_3: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
    squeeze_4: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
    mul_9: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
    mul_10: "f32[16]" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
    add_8: "f32[16]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
    squeeze_5: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
    mul_11: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.00000996502277);  squeeze_5 = None
    mul_12: "f32[16]" = torch.ops.aten.mul.Tensor(mul_11, 0.1);  mul_11 = None
    mul_13: "f32[16]" = torch.ops.aten.mul.Tensor(primals_342, 0.9)
    add_9: "f32[16]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    unsqueeze_4: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_3, -1)
    unsqueeze_5: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    mul_14: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_8, unsqueeze_5);  mul_8 = unsqueeze_5 = None
    unsqueeze_6: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_4, -1);  primals_4 = None
    unsqueeze_7: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
    add_10: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_7);  mul_14 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_1: "f32[8, 16, 112, 112]" = torch.ops.aten.clone.default(add_10)
    add_11: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_10, 3)
    clamp_min_1: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_11, 0);  add_11 = None
    clamp_max_1: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
    mul_15: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_10, clamp_max_1);  add_10 = clamp_max_1 = None
    div_1: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_15, 6);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_2: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(div_1, primals_179, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_12: "i64[]" = torch.ops.aten.add.Tensor(primals_343, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
    getitem_4: "f32[1, 16, 1, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 16, 1, 1]" = var_mean_2[1];  var_mean_2 = None
    add_13: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
    rsqrt_2: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
    sub_2: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
    mul_16: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    squeeze_6: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
    squeeze_7: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
    mul_17: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
    mul_18: "f32[16]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
    add_14: "f32[16]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
    squeeze_8: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
    mul_19: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
    mul_20: "f32[16]" = torch.ops.aten.mul.Tensor(mul_19, 0.1);  mul_19 = None
    mul_21: "f32[16]" = torch.ops.aten.mul.Tensor(primals_345, 0.9)
    add_15: "f32[16]" = torch.ops.aten.add.Tensor(mul_20, mul_21);  mul_20 = mul_21 = None
    unsqueeze_8: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_5, -1)
    unsqueeze_9: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
    mul_22: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_9);  mul_16 = unsqueeze_9 = None
    unsqueeze_10: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
    unsqueeze_11: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
    add_16: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_22, unsqueeze_11);  mul_22 = unsqueeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:129, code: x = self.drop_path(x) + shortcut
    add_17: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_16, div);  add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_3: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(add_17, primals_180, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_18: "i64[]" = torch.ops.aten.add.Tensor(primals_346, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
    getitem_6: "f32[1, 16, 1, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 16, 1, 1]" = var_mean_3[1];  var_mean_3 = None
    add_19: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
    rsqrt_3: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_3: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
    mul_23: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    squeeze_9: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
    squeeze_10: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
    mul_24: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
    mul_25: "f32[16]" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
    add_20: "f32[16]" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None
    squeeze_11: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
    mul_26: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
    mul_27: "f32[16]" = torch.ops.aten.mul.Tensor(mul_26, 0.1);  mul_26 = None
    mul_28: "f32[16]" = torch.ops.aten.mul.Tensor(primals_348, 0.9)
    add_21: "f32[16]" = torch.ops.aten.add.Tensor(mul_27, mul_28);  mul_27 = mul_28 = None
    unsqueeze_12: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1)
    unsqueeze_13: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
    mul_29: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_23, unsqueeze_13);  mul_23 = unsqueeze_13 = None
    unsqueeze_14: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
    unsqueeze_15: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
    add_22: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_15);  mul_29 = unsqueeze_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_2: "f32[8, 16, 112, 112]" = torch.ops.aten.clone.default(add_22)
    add_23: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_22, 3)
    clamp_min_2: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_min.default(add_23, 0);  add_23 = None
    clamp_max_2: "f32[8, 16, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
    mul_30: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_22, clamp_max_2);  add_22 = clamp_max_2 = None
    div_2: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(mul_30, 6);  mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_4: "f32[8, 16, 112, 112]" = torch.ops.aten.convolution.default(div_2, primals_181, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_24: "i64[]" = torch.ops.aten.add.Tensor(primals_349, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
    getitem_8: "f32[1, 16, 1, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 16, 1, 1]" = var_mean_4[1];  var_mean_4 = None
    add_25: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
    rsqrt_4: "f32[1, 16, 1, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_4: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
    mul_31: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    squeeze_12: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
    squeeze_13: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
    mul_32: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
    mul_33: "f32[16]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
    add_26: "f32[16]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    squeeze_14: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
    mul_34: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.00000996502277);  squeeze_14 = None
    mul_35: "f32[16]" = torch.ops.aten.mul.Tensor(mul_34, 0.1);  mul_34 = None
    mul_36: "f32[16]" = torch.ops.aten.mul.Tensor(primals_351, 0.9)
    add_27: "f32[16]" = torch.ops.aten.add.Tensor(mul_35, mul_36);  mul_35 = mul_36 = None
    unsqueeze_16: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_9, -1)
    unsqueeze_17: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
    mul_37: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_17);  mul_31 = unsqueeze_17 = None
    unsqueeze_18: "f32[16, 1]" = torch.ops.aten.unsqueeze.default(primals_10, -1);  primals_10 = None
    unsqueeze_19: "f32[16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
    add_28: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(mul_37, unsqueeze_19);  mul_37 = unsqueeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:129, code: x = self.drop_path(x) + shortcut
    add_29: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_28, add_17);  add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_5: "f32[8, 64, 112, 112]" = torch.ops.aten.convolution.default(add_29, primals_182, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_30: "i64[]" = torch.ops.aten.add.Tensor(primals_352, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
    getitem_10: "f32[1, 64, 1, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 64, 1, 1]" = var_mean_5[1];  var_mean_5 = None
    add_31: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
    rsqrt_5: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_5: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
    mul_38: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    squeeze_15: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
    squeeze_16: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
    mul_39: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
    mul_40: "f32[64]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
    add_32: "f32[64]" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
    squeeze_17: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
    mul_41: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.00000996502277);  squeeze_17 = None
    mul_42: "f32[64]" = torch.ops.aten.mul.Tensor(mul_41, 0.1);  mul_41 = None
    mul_43: "f32[64]" = torch.ops.aten.mul.Tensor(primals_354, 0.9)
    add_33: "f32[64]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
    unsqueeze_20: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_11, -1)
    unsqueeze_21: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
    mul_44: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_38, unsqueeze_21);  mul_38 = unsqueeze_21 = None
    unsqueeze_22: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
    unsqueeze_23: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
    add_34: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_23);  mul_44 = unsqueeze_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_3: "f32[8, 64, 112, 112]" = torch.ops.aten.clone.default(add_34)
    add_35: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(add_34, 3)
    clamp_min_3: "f32[8, 64, 112, 112]" = torch.ops.aten.clamp_min.default(add_35, 0);  add_35 = None
    clamp_max_3: "f32[8, 64, 112, 112]" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
    mul_45: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(add_34, clamp_max_3);  add_34 = clamp_max_3 = None
    div_3: "f32[8, 64, 112, 112]" = torch.ops.aten.div.Tensor(mul_45, 6);  mul_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_6: "f32[8, 64, 56, 56]" = torch.ops.aten.convolution.default(div_3, primals_183, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_355, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
    getitem_12: "f32[1, 64, 1, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 64, 1, 1]" = var_mean_6[1];  var_mean_6 = None
    add_37: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
    rsqrt_6: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_6: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
    mul_46: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    squeeze_18: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
    squeeze_19: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
    mul_47: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
    mul_48: "f32[64]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
    add_38: "f32[64]" = torch.ops.aten.add.Tensor(mul_47, mul_48);  mul_47 = mul_48 = None
    squeeze_20: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
    mul_49: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000398612827361);  squeeze_20 = None
    mul_50: "f32[64]" = torch.ops.aten.mul.Tensor(mul_49, 0.1);  mul_49 = None
    mul_51: "f32[64]" = torch.ops.aten.mul.Tensor(primals_357, 0.9)
    add_39: "f32[64]" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
    unsqueeze_24: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1)
    unsqueeze_25: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
    mul_52: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_25);  mul_46 = unsqueeze_25 = None
    unsqueeze_26: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
    unsqueeze_27: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
    add_40: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_52, unsqueeze_27);  mul_52 = unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_4: "f32[8, 64, 56, 56]" = torch.ops.aten.clone.default(add_40)
    add_41: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(add_40, 3)
    clamp_min_4: "f32[8, 64, 56, 56]" = torch.ops.aten.clamp_min.default(add_41, 0);  add_41 = None
    clamp_max_4: "f32[8, 64, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_4, 6);  clamp_min_4 = None
    mul_53: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(add_40, clamp_max_4);  add_40 = clamp_max_4 = None
    div_4: "f32[8, 64, 56, 56]" = torch.ops.aten.div.Tensor(mul_53, 6);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_7: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(div_4, primals_184, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_42: "i64[]" = torch.ops.aten.add.Tensor(primals_358, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
    getitem_14: "f32[1, 24, 1, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 24, 1, 1]" = var_mean_7[1];  var_mean_7 = None
    add_43: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
    rsqrt_7: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_7: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
    mul_54: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    squeeze_21: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
    squeeze_22: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
    mul_55: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
    mul_56: "f32[24]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
    add_44: "f32[24]" = torch.ops.aten.add.Tensor(mul_55, mul_56);  mul_55 = mul_56 = None
    squeeze_23: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
    mul_57: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000398612827361);  squeeze_23 = None
    mul_58: "f32[24]" = torch.ops.aten.mul.Tensor(mul_57, 0.1);  mul_57 = None
    mul_59: "f32[24]" = torch.ops.aten.mul.Tensor(primals_360, 0.9)
    add_45: "f32[24]" = torch.ops.aten.add.Tensor(mul_58, mul_59);  mul_58 = mul_59 = None
    unsqueeze_28: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_15, -1)
    unsqueeze_29: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
    mul_60: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_54, unsqueeze_29);  mul_54 = unsqueeze_29 = None
    unsqueeze_30: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_16, -1);  primals_16 = None
    unsqueeze_31: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
    add_46: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_60, unsqueeze_31);  mul_60 = unsqueeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_8: "f32[8, 48, 56, 56]" = torch.ops.aten.convolution.default(add_46, primals_185, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_361, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
    getitem_16: "f32[1, 48, 1, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 48, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_48: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
    rsqrt_8: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_8: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
    mul_61: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    squeeze_24: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
    squeeze_25: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
    mul_62: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
    mul_63: "f32[48]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
    add_49: "f32[48]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
    squeeze_26: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
    mul_64: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000398612827361);  squeeze_26 = None
    mul_65: "f32[48]" = torch.ops.aten.mul.Tensor(mul_64, 0.1);  mul_64 = None
    mul_66: "f32[48]" = torch.ops.aten.mul.Tensor(primals_363, 0.9)
    add_50: "f32[48]" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
    unsqueeze_32: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_17, -1)
    unsqueeze_33: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
    mul_67: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_33);  mul_61 = unsqueeze_33 = None
    unsqueeze_34: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
    unsqueeze_35: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
    add_51: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(mul_67, unsqueeze_35);  mul_67 = unsqueeze_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_5: "f32[8, 48, 56, 56]" = torch.ops.aten.clone.default(add_51)
    add_52: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(add_51, 3)
    clamp_min_5: "f32[8, 48, 56, 56]" = torch.ops.aten.clamp_min.default(add_52, 0);  add_52 = None
    clamp_max_5: "f32[8, 48, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_5, 6);  clamp_min_5 = None
    mul_68: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(add_51, clamp_max_5);  add_51 = clamp_max_5 = None
    div_5: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(mul_68, 6);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_9: "f32[8, 48, 56, 56]" = torch.ops.aten.convolution.default(div_5, primals_186, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_53: "i64[]" = torch.ops.aten.add.Tensor(primals_364, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
    getitem_18: "f32[1, 48, 1, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 48, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_54: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
    rsqrt_9: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_9: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
    mul_69: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    squeeze_27: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
    squeeze_28: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
    mul_70: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
    mul_71: "f32[48]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
    add_55: "f32[48]" = torch.ops.aten.add.Tensor(mul_70, mul_71);  mul_70 = mul_71 = None
    squeeze_29: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
    mul_72: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
    mul_73: "f32[48]" = torch.ops.aten.mul.Tensor(mul_72, 0.1);  mul_72 = None
    mul_74: "f32[48]" = torch.ops.aten.mul.Tensor(primals_366, 0.9)
    add_56: "f32[48]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
    unsqueeze_36: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
    unsqueeze_37: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
    mul_75: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(mul_69, unsqueeze_37);  mul_69 = unsqueeze_37 = None
    unsqueeze_38: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
    unsqueeze_39: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
    add_57: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(mul_75, unsqueeze_39);  mul_75 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_6: "f32[8, 48, 56, 56]" = torch.ops.aten.clone.default(add_57)
    add_58: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(add_57, 3)
    clamp_min_6: "f32[8, 48, 56, 56]" = torch.ops.aten.clamp_min.default(add_58, 0);  add_58 = None
    clamp_max_6: "f32[8, 48, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_6, 6);  clamp_min_6 = None
    mul_76: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(add_57, clamp_max_6);  add_57 = clamp_max_6 = None
    div_6: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(mul_76, 6);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_10: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(div_6, primals_187, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_59: "i64[]" = torch.ops.aten.add.Tensor(primals_367, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
    getitem_20: "f32[1, 24, 1, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 24, 1, 1]" = var_mean_10[1];  var_mean_10 = None
    add_60: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
    rsqrt_10: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_10: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
    mul_77: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    squeeze_30: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
    squeeze_31: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
    mul_78: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
    mul_79: "f32[24]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
    add_61: "f32[24]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    squeeze_32: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
    mul_80: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000398612827361);  squeeze_32 = None
    mul_81: "f32[24]" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
    mul_82: "f32[24]" = torch.ops.aten.mul.Tensor(primals_369, 0.9)
    add_62: "f32[24]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
    unsqueeze_40: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_21, -1)
    unsqueeze_41: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
    mul_83: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_41);  mul_77 = unsqueeze_41 = None
    unsqueeze_42: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_22, -1);  primals_22 = None
    unsqueeze_43: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
    add_63: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_43);  mul_83 = unsqueeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_64: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_63, add_46);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_11: "f32[8, 48, 56, 56]" = torch.ops.aten.convolution.default(add_64, primals_188, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_65: "i64[]" = torch.ops.aten.add.Tensor(primals_370, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 48, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 48, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_66: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
    rsqrt_11: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_11: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
    mul_84: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    squeeze_33: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    mul_85: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
    mul_86: "f32[48]" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
    add_67: "f32[48]" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
    squeeze_35: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
    mul_87: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000398612827361);  squeeze_35 = None
    mul_88: "f32[48]" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
    mul_89: "f32[48]" = torch.ops.aten.mul.Tensor(primals_372, 0.9)
    add_68: "f32[48]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
    unsqueeze_44: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_23, -1)
    unsqueeze_45: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
    mul_90: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_45);  mul_84 = unsqueeze_45 = None
    unsqueeze_46: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
    unsqueeze_47: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
    add_69: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_47);  mul_90 = unsqueeze_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_7: "f32[8, 48, 56, 56]" = torch.ops.aten.clone.default(add_69)
    add_70: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(add_69, 3)
    clamp_min_7: "f32[8, 48, 56, 56]" = torch.ops.aten.clamp_min.default(add_70, 0);  add_70 = None
    clamp_max_7: "f32[8, 48, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_7, 6);  clamp_min_7 = None
    mul_91: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(add_69, clamp_max_7);  add_69 = clamp_max_7 = None
    div_7: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(mul_91, 6);  mul_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_12: "f32[8, 48, 56, 56]" = torch.ops.aten.convolution.default(div_7, primals_189, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_71: "i64[]" = torch.ops.aten.add.Tensor(primals_373, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
    getitem_24: "f32[1, 48, 1, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 48, 1, 1]" = var_mean_12[1];  var_mean_12 = None
    add_72: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
    rsqrt_12: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_12: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
    mul_92: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    squeeze_36: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
    squeeze_37: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
    mul_93: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
    mul_94: "f32[48]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
    add_73: "f32[48]" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
    squeeze_38: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
    mul_95: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000398612827361);  squeeze_38 = None
    mul_96: "f32[48]" = torch.ops.aten.mul.Tensor(mul_95, 0.1);  mul_95 = None
    mul_97: "f32[48]" = torch.ops.aten.mul.Tensor(primals_375, 0.9)
    add_74: "f32[48]" = torch.ops.aten.add.Tensor(mul_96, mul_97);  mul_96 = mul_97 = None
    unsqueeze_48: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_25, -1)
    unsqueeze_49: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
    mul_98: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(mul_92, unsqueeze_49);  mul_92 = unsqueeze_49 = None
    unsqueeze_50: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_26, -1);  primals_26 = None
    unsqueeze_51: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
    add_75: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_51);  mul_98 = unsqueeze_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_8: "f32[8, 48, 56, 56]" = torch.ops.aten.clone.default(add_75)
    add_76: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(add_75, 3)
    clamp_min_8: "f32[8, 48, 56, 56]" = torch.ops.aten.clamp_min.default(add_76, 0);  add_76 = None
    clamp_max_8: "f32[8, 48, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_8, 6);  clamp_min_8 = None
    mul_99: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(add_75, clamp_max_8);  add_75 = clamp_max_8 = None
    div_8: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(mul_99, 6);  mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_13: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(div_8, primals_190, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_77: "i64[]" = torch.ops.aten.add.Tensor(primals_376, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
    getitem_26: "f32[1, 24, 1, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 24, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_78: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
    rsqrt_13: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_13: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
    mul_100: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    squeeze_39: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
    squeeze_40: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
    mul_101: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
    mul_102: "f32[24]" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
    add_79: "f32[24]" = torch.ops.aten.add.Tensor(mul_101, mul_102);  mul_101 = mul_102 = None
    squeeze_41: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
    mul_103: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000398612827361);  squeeze_41 = None
    mul_104: "f32[24]" = torch.ops.aten.mul.Tensor(mul_103, 0.1);  mul_103 = None
    mul_105: "f32[24]" = torch.ops.aten.mul.Tensor(primals_378, 0.9)
    add_80: "f32[24]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
    unsqueeze_52: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_27, -1)
    unsqueeze_53: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
    mul_106: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_53);  mul_100 = unsqueeze_53 = None
    unsqueeze_54: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1);  primals_28 = None
    unsqueeze_55: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
    add_81: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_106, unsqueeze_55);  mul_106 = unsqueeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_82: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_81, add_64);  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_14: "f32[8, 48, 56, 56]" = torch.ops.aten.convolution.default(add_82, primals_191, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_83: "i64[]" = torch.ops.aten.add.Tensor(primals_379, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
    getitem_28: "f32[1, 48, 1, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 48, 1, 1]" = var_mean_14[1];  var_mean_14 = None
    add_84: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
    rsqrt_14: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_14: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
    mul_107: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    squeeze_42: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
    squeeze_43: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
    mul_108: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
    mul_109: "f32[48]" = torch.ops.aten.mul.Tensor(primals_380, 0.9)
    add_85: "f32[48]" = torch.ops.aten.add.Tensor(mul_108, mul_109);  mul_108 = mul_109 = None
    squeeze_44: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
    mul_110: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000398612827361);  squeeze_44 = None
    mul_111: "f32[48]" = torch.ops.aten.mul.Tensor(mul_110, 0.1);  mul_110 = None
    mul_112: "f32[48]" = torch.ops.aten.mul.Tensor(primals_381, 0.9)
    add_86: "f32[48]" = torch.ops.aten.add.Tensor(mul_111, mul_112);  mul_111 = mul_112 = None
    unsqueeze_56: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1)
    unsqueeze_57: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
    mul_113: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(mul_107, unsqueeze_57);  mul_107 = unsqueeze_57 = None
    unsqueeze_58: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
    unsqueeze_59: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
    add_87: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_59);  mul_113 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_9: "f32[8, 48, 56, 56]" = torch.ops.aten.clone.default(add_87)
    add_88: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(add_87, 3)
    clamp_min_9: "f32[8, 48, 56, 56]" = torch.ops.aten.clamp_min.default(add_88, 0);  add_88 = None
    clamp_max_9: "f32[8, 48, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_9, 6);  clamp_min_9 = None
    mul_114: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(add_87, clamp_max_9);  add_87 = clamp_max_9 = None
    div_9: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(mul_114, 6);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_15: "f32[8, 48, 56, 56]" = torch.ops.aten.convolution.default(div_9, primals_192, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_89: "i64[]" = torch.ops.aten.add.Tensor(primals_382, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 48, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 48, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_90: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
    rsqrt_15: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_15: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
    mul_115: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    squeeze_45: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    mul_116: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
    mul_117: "f32[48]" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
    add_91: "f32[48]" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
    squeeze_47: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
    mul_118: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000398612827361);  squeeze_47 = None
    mul_119: "f32[48]" = torch.ops.aten.mul.Tensor(mul_118, 0.1);  mul_118 = None
    mul_120: "f32[48]" = torch.ops.aten.mul.Tensor(primals_384, 0.9)
    add_92: "f32[48]" = torch.ops.aten.add.Tensor(mul_119, mul_120);  mul_119 = mul_120 = None
    unsqueeze_60: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_31, -1)
    unsqueeze_61: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
    mul_121: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_61);  mul_115 = unsqueeze_61 = None
    unsqueeze_62: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_32, -1);  primals_32 = None
    unsqueeze_63: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
    add_93: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_63);  mul_121 = unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_10: "f32[8, 48, 56, 56]" = torch.ops.aten.clone.default(add_93)
    add_94: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(add_93, 3)
    clamp_min_10: "f32[8, 48, 56, 56]" = torch.ops.aten.clamp_min.default(add_94, 0);  add_94 = None
    clamp_max_10: "f32[8, 48, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_10, 6);  clamp_min_10 = None
    mul_122: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(add_93, clamp_max_10);  add_93 = clamp_max_10 = None
    div_10: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(mul_122, 6);  mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_16: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(div_10, primals_193, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_95: "i64[]" = torch.ops.aten.add.Tensor(primals_385, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_32: "f32[1, 24, 1, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 24, 1, 1]" = var_mean_16[1];  var_mean_16 = None
    add_96: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
    rsqrt_16: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_16: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
    mul_123: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    squeeze_48: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
    squeeze_49: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
    mul_124: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
    mul_125: "f32[24]" = torch.ops.aten.mul.Tensor(primals_386, 0.9)
    add_97: "f32[24]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
    squeeze_50: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
    mul_126: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000398612827361);  squeeze_50 = None
    mul_127: "f32[24]" = torch.ops.aten.mul.Tensor(mul_126, 0.1);  mul_126 = None
    mul_128: "f32[24]" = torch.ops.aten.mul.Tensor(primals_387, 0.9)
    add_98: "f32[24]" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    unsqueeze_64: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_33, -1)
    unsqueeze_65: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
    mul_129: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_123, unsqueeze_65);  mul_123 = unsqueeze_65 = None
    unsqueeze_66: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1);  primals_34 = None
    unsqueeze_67: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
    add_99: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_129, unsqueeze_67);  mul_129 = unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_100: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_99, add_82);  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_17: "f32[8, 120, 56, 56]" = torch.ops.aten.convolution.default(add_100, primals_194, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_101: "i64[]" = torch.ops.aten.add.Tensor(primals_388, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_17 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
    getitem_34: "f32[1, 120, 1, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 120, 1, 1]" = var_mean_17[1];  var_mean_17 = None
    add_102: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
    rsqrt_17: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
    sub_17: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_35)
    mul_130: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    squeeze_51: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
    squeeze_52: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
    mul_131: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
    mul_132: "f32[120]" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
    add_103: "f32[120]" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
    squeeze_53: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
    mul_133: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000398612827361);  squeeze_53 = None
    mul_134: "f32[120]" = torch.ops.aten.mul.Tensor(mul_133, 0.1);  mul_133 = None
    mul_135: "f32[120]" = torch.ops.aten.mul.Tensor(primals_390, 0.9)
    add_104: "f32[120]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    unsqueeze_68: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1)
    unsqueeze_69: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
    mul_136: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_69);  mul_130 = unsqueeze_69 = None
    unsqueeze_70: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
    unsqueeze_71: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
    add_105: "f32[8, 120, 56, 56]" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_71);  mul_136 = unsqueeze_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_11: "f32[8, 120, 56, 56]" = torch.ops.aten.clone.default(add_105)
    add_106: "f32[8, 120, 56, 56]" = torch.ops.aten.add.Tensor(add_105, 3)
    clamp_min_11: "f32[8, 120, 56, 56]" = torch.ops.aten.clamp_min.default(add_106, 0);  add_106 = None
    clamp_max_11: "f32[8, 120, 56, 56]" = torch.ops.aten.clamp_max.default(clamp_min_11, 6);  clamp_min_11 = None
    mul_137: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(add_105, clamp_max_11);  add_105 = clamp_max_11 = None
    div_11: "f32[8, 120, 56, 56]" = torch.ops.aten.div.Tensor(mul_137, 6);  mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_18: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(div_11, primals_195, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_107: "i64[]" = torch.ops.aten.add.Tensor(primals_391, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_18 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
    getitem_36: "f32[1, 120, 1, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 120, 1, 1]" = var_mean_18[1];  var_mean_18 = None
    add_108: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
    rsqrt_18: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    sub_18: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_37)
    mul_138: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    squeeze_54: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
    squeeze_55: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
    mul_139: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
    mul_140: "f32[120]" = torch.ops.aten.mul.Tensor(primals_392, 0.9)
    add_109: "f32[120]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
    squeeze_56: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
    mul_141: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0001594642002871);  squeeze_56 = None
    mul_142: "f32[120]" = torch.ops.aten.mul.Tensor(mul_141, 0.1);  mul_141 = None
    mul_143: "f32[120]" = torch.ops.aten.mul.Tensor(primals_393, 0.9)
    add_110: "f32[120]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    unsqueeze_72: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1)
    unsqueeze_73: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
    mul_144: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_73);  mul_138 = unsqueeze_73 = None
    unsqueeze_74: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_38, -1);  primals_38 = None
    unsqueeze_75: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
    add_111: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_75);  mul_144 = unsqueeze_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_12: "f32[8, 120, 28, 28]" = torch.ops.aten.clone.default(add_111)
    add_112: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(add_111, 3)
    clamp_min_12: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_min.default(add_112, 0);  add_112 = None
    clamp_max_12: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_12, 6);  clamp_min_12 = None
    mul_145: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_111, clamp_max_12);  add_111 = clamp_max_12 = None
    div_12: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(mul_145, 6);  mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_12, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_19: "f32[8, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_196, primals_197, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_13: "f32[8, 8, 1, 1]" = torch.ops.aten.clone.default(convolution_19)
    add_113: "f32[8, 8, 1, 1]" = torch.ops.aten.add.Tensor(convolution_19, 3)
    clamp_min_13: "f32[8, 8, 1, 1]" = torch.ops.aten.clamp_min.default(add_113, 0);  add_113 = None
    clamp_max_13: "f32[8, 8, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
    mul_146: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_19, clamp_max_13);  convolution_19 = clamp_max_13 = None
    div_13: "f32[8, 8, 1, 1]" = torch.ops.aten.div.Tensor(mul_146, 6);  mul_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_20: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_13, primals_198, primals_199, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_114: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_20, 3)
    clamp_min_14: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_114, 0);  add_114 = None
    clamp_max_14: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
    div_14: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_14, 6);  clamp_max_14 = None
    mul_147: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(div_12, div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_21: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_147, primals_200, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_115: "i64[]" = torch.ops.aten.add.Tensor(primals_394, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 40, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 40, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_116: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
    rsqrt_19: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    sub_19: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_39)
    mul_148: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    squeeze_57: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    mul_149: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
    mul_150: "f32[40]" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
    add_117: "f32[40]" = torch.ops.aten.add.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
    squeeze_59: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
    mul_151: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0001594642002871);  squeeze_59 = None
    mul_152: "f32[40]" = torch.ops.aten.mul.Tensor(mul_151, 0.1);  mul_151 = None
    mul_153: "f32[40]" = torch.ops.aten.mul.Tensor(primals_396, 0.9)
    add_118: "f32[40]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
    unsqueeze_76: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_39, -1)
    unsqueeze_77: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
    mul_154: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_77);  mul_148 = unsqueeze_77 = None
    unsqueeze_78: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1);  primals_40 = None
    unsqueeze_79: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
    add_119: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_154, unsqueeze_79);  mul_154 = unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_22: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(add_119, primals_201, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_120: "i64[]" = torch.ops.aten.add.Tensor(primals_397, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_20 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
    getitem_40: "f32[1, 120, 1, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 120, 1, 1]" = var_mean_20[1];  var_mean_20 = None
    add_121: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
    rsqrt_20: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    sub_20: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_41)
    mul_155: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    squeeze_60: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
    squeeze_61: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
    mul_156: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
    mul_157: "f32[120]" = torch.ops.aten.mul.Tensor(primals_398, 0.9)
    add_122: "f32[120]" = torch.ops.aten.add.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
    squeeze_62: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
    mul_158: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0001594642002871);  squeeze_62 = None
    mul_159: "f32[120]" = torch.ops.aten.mul.Tensor(mul_158, 0.1);  mul_158 = None
    mul_160: "f32[120]" = torch.ops.aten.mul.Tensor(primals_399, 0.9)
    add_123: "f32[120]" = torch.ops.aten.add.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    unsqueeze_80: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1)
    unsqueeze_81: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
    mul_161: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_155, unsqueeze_81);  mul_155 = unsqueeze_81 = None
    unsqueeze_82: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
    unsqueeze_83: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
    add_124: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_83);  mul_161 = unsqueeze_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_14: "f32[8, 120, 28, 28]" = torch.ops.aten.clone.default(add_124)
    add_125: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(add_124, 3)
    clamp_min_15: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_min.default(add_125, 0);  add_125 = None
    clamp_max_15: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_15, 6);  clamp_min_15 = None
    mul_162: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_124, clamp_max_15);  add_124 = clamp_max_15 = None
    div_15: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(mul_162, 6);  mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_23: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(div_15, primals_202, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_126: "i64[]" = torch.ops.aten.add.Tensor(primals_400, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_21 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
    getitem_42: "f32[1, 120, 1, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 120, 1, 1]" = var_mean_21[1];  var_mean_21 = None
    add_127: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
    rsqrt_21: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_21: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_43)
    mul_163: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    squeeze_63: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
    squeeze_64: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
    mul_164: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
    mul_165: "f32[120]" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
    add_128: "f32[120]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    squeeze_65: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
    mul_166: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0001594642002871);  squeeze_65 = None
    mul_167: "f32[120]" = torch.ops.aten.mul.Tensor(mul_166, 0.1);  mul_166 = None
    mul_168: "f32[120]" = torch.ops.aten.mul.Tensor(primals_402, 0.9)
    add_129: "f32[120]" = torch.ops.aten.add.Tensor(mul_167, mul_168);  mul_167 = mul_168 = None
    unsqueeze_84: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_43, -1)
    unsqueeze_85: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
    mul_169: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_85);  mul_163 = unsqueeze_85 = None
    unsqueeze_86: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_44, -1);  primals_44 = None
    unsqueeze_87: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
    add_130: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_169, unsqueeze_87);  mul_169 = unsqueeze_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_15: "f32[8, 120, 28, 28]" = torch.ops.aten.clone.default(add_130)
    add_131: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(add_130, 3)
    clamp_min_16: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_min.default(add_131, 0);  add_131 = None
    clamp_max_16: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
    mul_170: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_130, clamp_max_16);  add_130 = clamp_max_16 = None
    div_16: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(mul_170, 6);  mul_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_1: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_16, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_24: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_203, primals_204, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_16: "f32[8, 16, 1, 1]" = torch.ops.aten.clone.default(convolution_24)
    add_132: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(convolution_24, 3)
    clamp_min_17: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_min.default(add_132, 0);  add_132 = None
    clamp_max_17: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
    mul_171: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_24, clamp_max_17);  convolution_24 = clamp_max_17 = None
    div_17: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(mul_171, 6);  mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_25: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_17, primals_205, primals_206, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_133: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_25, 3)
    clamp_min_18: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_133, 0);  add_133 = None
    clamp_max_18: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_18, 6);  clamp_min_18 = None
    div_18: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_18, 6);  clamp_max_18 = None
    mul_172: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(div_16, div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_26: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_172, primals_207, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_134: "i64[]" = torch.ops.aten.add.Tensor(primals_403, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_22 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_44: "f32[1, 40, 1, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 40, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_135: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
    rsqrt_22: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    sub_22: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_45)
    mul_173: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    squeeze_66: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
    squeeze_67: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
    mul_174: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
    mul_175: "f32[40]" = torch.ops.aten.mul.Tensor(primals_404, 0.9)
    add_136: "f32[40]" = torch.ops.aten.add.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None
    squeeze_68: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
    mul_176: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0001594642002871);  squeeze_68 = None
    mul_177: "f32[40]" = torch.ops.aten.mul.Tensor(mul_176, 0.1);  mul_176 = None
    mul_178: "f32[40]" = torch.ops.aten.mul.Tensor(primals_405, 0.9)
    add_137: "f32[40]" = torch.ops.aten.add.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None
    unsqueeze_88: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_45, -1)
    unsqueeze_89: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
    mul_179: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_173, unsqueeze_89);  mul_173 = unsqueeze_89 = None
    unsqueeze_90: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1);  primals_46 = None
    unsqueeze_91: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
    add_138: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_179, unsqueeze_91);  mul_179 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_139: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_138, add_119);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_27: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(add_139, primals_208, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_140: "i64[]" = torch.ops.aten.add.Tensor(primals_406, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 120, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 120, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_141: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
    rsqrt_23: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_23: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_47)
    mul_180: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    squeeze_69: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    mul_181: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
    mul_182: "f32[120]" = torch.ops.aten.mul.Tensor(primals_407, 0.9)
    add_142: "f32[120]" = torch.ops.aten.add.Tensor(mul_181, mul_182);  mul_181 = mul_182 = None
    squeeze_71: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
    mul_183: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0001594642002871);  squeeze_71 = None
    mul_184: "f32[120]" = torch.ops.aten.mul.Tensor(mul_183, 0.1);  mul_183 = None
    mul_185: "f32[120]" = torch.ops.aten.mul.Tensor(primals_408, 0.9)
    add_143: "f32[120]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
    unsqueeze_92: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
    unsqueeze_93: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
    mul_186: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_93);  mul_180 = unsqueeze_93 = None
    unsqueeze_94: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
    unsqueeze_95: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
    add_144: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_186, unsqueeze_95);  mul_186 = unsqueeze_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_17: "f32[8, 120, 28, 28]" = torch.ops.aten.clone.default(add_144)
    add_145: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(add_144, 3)
    clamp_min_19: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_min.default(add_145, 0);  add_145 = None
    clamp_max_19: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
    mul_187: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_144, clamp_max_19);  add_144 = clamp_max_19 = None
    div_19: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(mul_187, 6);  mul_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_28: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(div_19, primals_209, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_146: "i64[]" = torch.ops.aten.add.Tensor(primals_409, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_24 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
    getitem_48: "f32[1, 120, 1, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 120, 1, 1]" = var_mean_24[1];  var_mean_24 = None
    add_147: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
    rsqrt_24: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_24: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_49)
    mul_188: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    squeeze_72: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
    squeeze_73: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
    mul_189: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
    mul_190: "f32[120]" = torch.ops.aten.mul.Tensor(primals_410, 0.9)
    add_148: "f32[120]" = torch.ops.aten.add.Tensor(mul_189, mul_190);  mul_189 = mul_190 = None
    squeeze_74: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
    mul_191: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0001594642002871);  squeeze_74 = None
    mul_192: "f32[120]" = torch.ops.aten.mul.Tensor(mul_191, 0.1);  mul_191 = None
    mul_193: "f32[120]" = torch.ops.aten.mul.Tensor(primals_411, 0.9)
    add_149: "f32[120]" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
    unsqueeze_96: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1)
    unsqueeze_97: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
    mul_194: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_188, unsqueeze_97);  mul_188 = unsqueeze_97 = None
    unsqueeze_98: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1);  primals_50 = None
    unsqueeze_99: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
    add_150: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_99);  mul_194 = unsqueeze_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_18: "f32[8, 120, 28, 28]" = torch.ops.aten.clone.default(add_150)
    add_151: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(add_150, 3)
    clamp_min_20: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_min.default(add_151, 0);  add_151 = None
    clamp_max_20: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
    mul_195: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_150, clamp_max_20);  add_150 = clamp_max_20 = None
    div_20: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(mul_195, 6);  mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_2: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_20, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_29: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_210, primals_211, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_19: "f32[8, 16, 1, 1]" = torch.ops.aten.clone.default(convolution_29)
    add_152: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(convolution_29, 3)
    clamp_min_21: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_min.default(add_152, 0);  add_152 = None
    clamp_max_21: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_21, 6);  clamp_min_21 = None
    mul_196: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_29, clamp_max_21);  convolution_29 = clamp_max_21 = None
    div_21: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(mul_196, 6);  mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_30: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_21, primals_212, primals_213, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_153: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_30, 3)
    clamp_min_22: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_153, 0);  add_153 = None
    clamp_max_22: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
    div_22: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_22, 6);  clamp_max_22 = None
    mul_197: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(div_20, div_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_31: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_197, primals_214, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_154: "i64[]" = torch.ops.aten.add.Tensor(primals_412, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_25 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_50: "f32[1, 40, 1, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 40, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_155: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
    rsqrt_25: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_25: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_51)
    mul_198: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    squeeze_75: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
    squeeze_76: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
    mul_199: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
    mul_200: "f32[40]" = torch.ops.aten.mul.Tensor(primals_413, 0.9)
    add_156: "f32[40]" = torch.ops.aten.add.Tensor(mul_199, mul_200);  mul_199 = mul_200 = None
    squeeze_77: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
    mul_201: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001594642002871);  squeeze_77 = None
    mul_202: "f32[40]" = torch.ops.aten.mul.Tensor(mul_201, 0.1);  mul_201 = None
    mul_203: "f32[40]" = torch.ops.aten.mul.Tensor(primals_414, 0.9)
    add_157: "f32[40]" = torch.ops.aten.add.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None
    unsqueeze_100: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1)
    unsqueeze_101: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
    mul_204: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_198, unsqueeze_101);  mul_198 = unsqueeze_101 = None
    unsqueeze_102: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_52, -1);  primals_52 = None
    unsqueeze_103: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
    add_158: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_204, unsqueeze_103);  mul_204 = unsqueeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_159: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_158, add_139);  add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_32: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(add_159, primals_215, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_160: "i64[]" = torch.ops.aten.add.Tensor(primals_415, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_26 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
    getitem_52: "f32[1, 120, 1, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 120, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_161: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
    rsqrt_26: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_26: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_53)
    mul_205: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    squeeze_78: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
    squeeze_79: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
    mul_206: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
    mul_207: "f32[120]" = torch.ops.aten.mul.Tensor(primals_416, 0.9)
    add_162: "f32[120]" = torch.ops.aten.add.Tensor(mul_206, mul_207);  mul_206 = mul_207 = None
    squeeze_80: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
    mul_208: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001594642002871);  squeeze_80 = None
    mul_209: "f32[120]" = torch.ops.aten.mul.Tensor(mul_208, 0.1);  mul_208 = None
    mul_210: "f32[120]" = torch.ops.aten.mul.Tensor(primals_417, 0.9)
    add_163: "f32[120]" = torch.ops.aten.add.Tensor(mul_209, mul_210);  mul_209 = mul_210 = None
    unsqueeze_104: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_53, -1)
    unsqueeze_105: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
    mul_211: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_105);  mul_205 = unsqueeze_105 = None
    unsqueeze_106: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
    unsqueeze_107: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
    add_164: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_211, unsqueeze_107);  mul_211 = unsqueeze_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_20: "f32[8, 120, 28, 28]" = torch.ops.aten.clone.default(add_164)
    add_165: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(add_164, 3)
    clamp_min_23: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_min.default(add_165, 0);  add_165 = None
    clamp_max_23: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
    mul_212: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_164, clamp_max_23);  add_164 = clamp_max_23 = None
    div_23: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(mul_212, 6);  mul_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_33: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(div_23, primals_216, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_166: "i64[]" = torch.ops.aten.add.Tensor(primals_418, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 120, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 120, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_167: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
    rsqrt_27: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    sub_27: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_55)
    mul_213: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    squeeze_81: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_82: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    mul_214: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
    mul_215: "f32[120]" = torch.ops.aten.mul.Tensor(primals_419, 0.9)
    add_168: "f32[120]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    squeeze_83: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
    mul_216: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0001594642002871);  squeeze_83 = None
    mul_217: "f32[120]" = torch.ops.aten.mul.Tensor(mul_216, 0.1);  mul_216 = None
    mul_218: "f32[120]" = torch.ops.aten.mul.Tensor(primals_420, 0.9)
    add_169: "f32[120]" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
    unsqueeze_108: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1)
    unsqueeze_109: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
    mul_219: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_213, unsqueeze_109);  mul_213 = unsqueeze_109 = None
    unsqueeze_110: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1);  primals_56 = None
    unsqueeze_111: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
    add_170: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_219, unsqueeze_111);  mul_219 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_21: "f32[8, 120, 28, 28]" = torch.ops.aten.clone.default(add_170)
    add_171: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(add_170, 3)
    clamp_min_24: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_min.default(add_171, 0);  add_171 = None
    clamp_max_24: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_24, 6);  clamp_min_24 = None
    mul_220: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_170, clamp_max_24);  add_170 = clamp_max_24 = None
    div_24: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(mul_220, 6);  mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_3: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_24, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_34: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_217, primals_218, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_22: "f32[8, 16, 1, 1]" = torch.ops.aten.clone.default(convolution_34)
    add_172: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(convolution_34, 3)
    clamp_min_25: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_min.default(add_172, 0);  add_172 = None
    clamp_max_25: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
    mul_221: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_34, clamp_max_25);  convolution_34 = clamp_max_25 = None
    div_25: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(mul_221, 6);  mul_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_35: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_25, primals_219, primals_220, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_173: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_35, 3)
    clamp_min_26: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_173, 0);  add_173 = None
    clamp_max_26: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
    div_26: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_26, 6);  clamp_max_26 = None
    mul_222: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(div_24, div_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_36: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_222, primals_221, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_174: "i64[]" = torch.ops.aten.add.Tensor(primals_421, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_28 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_56: "f32[1, 40, 1, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 40, 1, 1]" = var_mean_28[1];  var_mean_28 = None
    add_175: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
    rsqrt_28: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_28: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_57)
    mul_223: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    squeeze_84: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
    squeeze_85: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
    mul_224: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
    mul_225: "f32[40]" = torch.ops.aten.mul.Tensor(primals_422, 0.9)
    add_176: "f32[40]" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
    squeeze_86: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
    mul_226: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0001594642002871);  squeeze_86 = None
    mul_227: "f32[40]" = torch.ops.aten.mul.Tensor(mul_226, 0.1);  mul_226 = None
    mul_228: "f32[40]" = torch.ops.aten.mul.Tensor(primals_423, 0.9)
    add_177: "f32[40]" = torch.ops.aten.add.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
    unsqueeze_112: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1)
    unsqueeze_113: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
    mul_229: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_113);  mul_223 = unsqueeze_113 = None
    unsqueeze_114: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_58, -1);  primals_58 = None
    unsqueeze_115: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
    add_178: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_229, unsqueeze_115);  mul_229 = unsqueeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_179: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_178, add_159);  add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_37: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(add_179, primals_222, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_424, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_29 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
    getitem_58: "f32[1, 120, 1, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 120, 1, 1]" = var_mean_29[1];  var_mean_29 = None
    add_181: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
    rsqrt_29: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    sub_29: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_59)
    mul_230: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    squeeze_87: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
    squeeze_88: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
    mul_231: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
    mul_232: "f32[120]" = torch.ops.aten.mul.Tensor(primals_425, 0.9)
    add_182: "f32[120]" = torch.ops.aten.add.Tensor(mul_231, mul_232);  mul_231 = mul_232 = None
    squeeze_89: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
    mul_233: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0001594642002871);  squeeze_89 = None
    mul_234: "f32[120]" = torch.ops.aten.mul.Tensor(mul_233, 0.1);  mul_233 = None
    mul_235: "f32[120]" = torch.ops.aten.mul.Tensor(primals_426, 0.9)
    add_183: "f32[120]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    unsqueeze_116: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_59, -1)
    unsqueeze_117: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
    mul_236: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_230, unsqueeze_117);  mul_230 = unsqueeze_117 = None
    unsqueeze_118: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
    unsqueeze_119: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
    add_184: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_119);  mul_236 = unsqueeze_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_23: "f32[8, 120, 28, 28]" = torch.ops.aten.clone.default(add_184)
    add_185: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(add_184, 3)
    clamp_min_27: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_min.default(add_185, 0);  add_185 = None
    clamp_max_27: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_27, 6);  clamp_min_27 = None
    mul_237: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_184, clamp_max_27);  add_184 = clamp_max_27 = None
    div_27: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(mul_237, 6);  mul_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_38: "f32[8, 120, 28, 28]" = torch.ops.aten.convolution.default(div_27, primals_223, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 120)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_186: "i64[]" = torch.ops.aten.add.Tensor(primals_427, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_30 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
    getitem_60: "f32[1, 120, 1, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 120, 1, 1]" = var_mean_30[1];  var_mean_30 = None
    add_187: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
    rsqrt_30: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    sub_30: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_61)
    mul_238: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    squeeze_90: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
    squeeze_91: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
    mul_239: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
    mul_240: "f32[120]" = torch.ops.aten.mul.Tensor(primals_428, 0.9)
    add_188: "f32[120]" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    squeeze_92: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
    mul_241: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0001594642002871);  squeeze_92 = None
    mul_242: "f32[120]" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
    mul_243: "f32[120]" = torch.ops.aten.mul.Tensor(primals_429, 0.9)
    add_189: "f32[120]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    unsqueeze_120: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_61, -1)
    unsqueeze_121: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
    mul_244: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_121);  mul_238 = unsqueeze_121 = None
    unsqueeze_122: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1);  primals_62 = None
    unsqueeze_123: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
    add_190: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_123);  mul_244 = unsqueeze_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_24: "f32[8, 120, 28, 28]" = torch.ops.aten.clone.default(add_190)
    add_191: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(add_190, 3)
    clamp_min_28: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_min.default(add_191, 0);  add_191 = None
    clamp_max_28: "f32[8, 120, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_28, 6);  clamp_min_28 = None
    mul_245: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_190, clamp_max_28);  add_190 = clamp_max_28 = None
    div_28: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(mul_245, 6);  mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_4: "f32[8, 120, 1, 1]" = torch.ops.aten.mean.dim(div_28, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_39: "f32[8, 16, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_224, primals_225, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_25: "f32[8, 16, 1, 1]" = torch.ops.aten.clone.default(convolution_39)
    add_192: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(convolution_39, 3)
    clamp_min_29: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_min.default(add_192, 0);  add_192 = None
    clamp_max_29: "f32[8, 16, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_29, 6);  clamp_min_29 = None
    mul_246: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_39, clamp_max_29);  convolution_39 = clamp_max_29 = None
    div_29: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(mul_246, 6);  mul_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_40: "f32[8, 120, 1, 1]" = torch.ops.aten.convolution.default(div_29, primals_226, primals_227, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_193: "f32[8, 120, 1, 1]" = torch.ops.aten.add.Tensor(convolution_40, 3)
    clamp_min_30: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_min.default(add_193, 0);  add_193 = None
    clamp_max_30: "f32[8, 120, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_30, 6);  clamp_min_30 = None
    div_30: "f32[8, 120, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_30, 6);  clamp_max_30 = None
    mul_247: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(div_28, div_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_41: "f32[8, 40, 28, 28]" = torch.ops.aten.convolution.default(mul_247, primals_228, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_194: "i64[]" = torch.ops.aten.add.Tensor(primals_430, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 40, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 40, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_195: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
    rsqrt_31: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
    sub_31: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_63)
    mul_248: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    squeeze_93: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_94: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    mul_249: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
    mul_250: "f32[40]" = torch.ops.aten.mul.Tensor(primals_431, 0.9)
    add_196: "f32[40]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
    squeeze_95: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
    mul_251: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0001594642002871);  squeeze_95 = None
    mul_252: "f32[40]" = torch.ops.aten.mul.Tensor(mul_251, 0.1);  mul_251 = None
    mul_253: "f32[40]" = torch.ops.aten.mul.Tensor(primals_432, 0.9)
    add_197: "f32[40]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    unsqueeze_124: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1)
    unsqueeze_125: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
    mul_254: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_125);  mul_248 = unsqueeze_125 = None
    unsqueeze_126: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_64, -1);  primals_64 = None
    unsqueeze_127: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
    add_198: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_127);  mul_254 = unsqueeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_199: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_198, add_179);  add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_42: "f32[8, 200, 28, 28]" = torch.ops.aten.convolution.default(add_199, primals_229, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_200: "i64[]" = torch.ops.aten.add.Tensor(primals_433, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_32 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
    getitem_64: "f32[1, 200, 1, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 200, 1, 1]" = var_mean_32[1];  var_mean_32 = None
    add_201: "f32[1, 200, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
    rsqrt_32: "f32[1, 200, 1, 1]" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
    sub_32: "f32[8, 200, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_65)
    mul_255: "f32[8, 200, 28, 28]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    squeeze_96: "f32[200]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
    squeeze_97: "f32[200]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
    mul_256: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
    mul_257: "f32[200]" = torch.ops.aten.mul.Tensor(primals_434, 0.9)
    add_202: "f32[200]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
    squeeze_98: "f32[200]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
    mul_258: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0001594642002871);  squeeze_98 = None
    mul_259: "f32[200]" = torch.ops.aten.mul.Tensor(mul_258, 0.1);  mul_258 = None
    mul_260: "f32[200]" = torch.ops.aten.mul.Tensor(primals_435, 0.9)
    add_203: "f32[200]" = torch.ops.aten.add.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
    unsqueeze_128: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_65, -1)
    unsqueeze_129: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
    mul_261: "f32[8, 200, 28, 28]" = torch.ops.aten.mul.Tensor(mul_255, unsqueeze_129);  mul_255 = unsqueeze_129 = None
    unsqueeze_130: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
    unsqueeze_131: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
    add_204: "f32[8, 200, 28, 28]" = torch.ops.aten.add.Tensor(mul_261, unsqueeze_131);  mul_261 = unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_26: "f32[8, 200, 28, 28]" = torch.ops.aten.clone.default(add_204)
    add_205: "f32[8, 200, 28, 28]" = torch.ops.aten.add.Tensor(add_204, 3)
    clamp_min_31: "f32[8, 200, 28, 28]" = torch.ops.aten.clamp_min.default(add_205, 0);  add_205 = None
    clamp_max_31: "f32[8, 200, 28, 28]" = torch.ops.aten.clamp_max.default(clamp_min_31, 6);  clamp_min_31 = None
    mul_262: "f32[8, 200, 28, 28]" = torch.ops.aten.mul.Tensor(add_204, clamp_max_31);  add_204 = clamp_max_31 = None
    div_31: "f32[8, 200, 28, 28]" = torch.ops.aten.div.Tensor(mul_262, 6);  mul_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_43: "f32[8, 200, 14, 14]" = torch.ops.aten.convolution.default(div_31, primals_230, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 200)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_206: "i64[]" = torch.ops.aten.add.Tensor(primals_436, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_33 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
    getitem_66: "f32[1, 200, 1, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 200, 1, 1]" = var_mean_33[1];  var_mean_33 = None
    add_207: "f32[1, 200, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
    rsqrt_33: "f32[1, 200, 1, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
    sub_33: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_67)
    mul_263: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    squeeze_99: "f32[200]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
    squeeze_100: "f32[200]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
    mul_264: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
    mul_265: "f32[200]" = torch.ops.aten.mul.Tensor(primals_437, 0.9)
    add_208: "f32[200]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
    squeeze_101: "f32[200]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
    mul_266: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0006381620931717);  squeeze_101 = None
    mul_267: "f32[200]" = torch.ops.aten.mul.Tensor(mul_266, 0.1);  mul_266 = None
    mul_268: "f32[200]" = torch.ops.aten.mul.Tensor(primals_438, 0.9)
    add_209: "f32[200]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
    unsqueeze_132: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_67, -1)
    unsqueeze_133: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
    mul_269: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_133);  mul_263 = unsqueeze_133 = None
    unsqueeze_134: "f32[200, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1);  primals_68 = None
    unsqueeze_135: "f32[200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
    add_210: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_135);  mul_269 = unsqueeze_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_27: "f32[8, 200, 14, 14]" = torch.ops.aten.clone.default(add_210)
    add_211: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(add_210, 3)
    clamp_min_32: "f32[8, 200, 14, 14]" = torch.ops.aten.clamp_min.default(add_211, 0);  add_211 = None
    clamp_max_32: "f32[8, 200, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_32, 6);  clamp_min_32 = None
    mul_270: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(add_210, clamp_max_32);  add_210 = clamp_max_32 = None
    div_32: "f32[8, 200, 14, 14]" = torch.ops.aten.div.Tensor(mul_270, 6);  mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_44: "f32[8, 72, 14, 14]" = torch.ops.aten.convolution.default(div_32, primals_231, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_439, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_34 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
    getitem_68: "f32[1, 72, 1, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 72, 1, 1]" = var_mean_34[1];  var_mean_34 = None
    add_213: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
    rsqrt_34: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
    sub_34: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_69)
    mul_271: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    squeeze_102: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
    squeeze_103: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
    mul_272: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
    mul_273: "f32[72]" = torch.ops.aten.mul.Tensor(primals_440, 0.9)
    add_214: "f32[72]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
    squeeze_104: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
    mul_274: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0006381620931717);  squeeze_104 = None
    mul_275: "f32[72]" = torch.ops.aten.mul.Tensor(mul_274, 0.1);  mul_274 = None
    mul_276: "f32[72]" = torch.ops.aten.mul.Tensor(primals_441, 0.9)
    add_215: "f32[72]" = torch.ops.aten.add.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    unsqueeze_136: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
    unsqueeze_137: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
    mul_277: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_137);  mul_271 = unsqueeze_137 = None
    unsqueeze_138: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_70, -1);  primals_70 = None
    unsqueeze_139: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
    add_216: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(mul_277, unsqueeze_139);  mul_277 = unsqueeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_45: "f32[8, 216, 14, 14]" = torch.ops.aten.convolution.default(add_216, primals_232, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_217: "i64[]" = torch.ops.aten.add.Tensor(primals_442, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 216, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 216, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_218: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
    rsqrt_35: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
    sub_35: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_71)
    mul_278: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    squeeze_105: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_106: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    mul_279: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
    mul_280: "f32[216]" = torch.ops.aten.mul.Tensor(primals_443, 0.9)
    add_219: "f32[216]" = torch.ops.aten.add.Tensor(mul_279, mul_280);  mul_279 = mul_280 = None
    squeeze_107: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
    mul_281: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0006381620931717);  squeeze_107 = None
    mul_282: "f32[216]" = torch.ops.aten.mul.Tensor(mul_281, 0.1);  mul_281 = None
    mul_283: "f32[216]" = torch.ops.aten.mul.Tensor(primals_444, 0.9)
    add_220: "f32[216]" = torch.ops.aten.add.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
    unsqueeze_140: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_71, -1)
    unsqueeze_141: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
    mul_284: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_141);  mul_278 = unsqueeze_141 = None
    unsqueeze_142: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
    unsqueeze_143: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
    add_221: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(mul_284, unsqueeze_143);  mul_284 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_28: "f32[8, 216, 14, 14]" = torch.ops.aten.clone.default(add_221)
    add_222: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(add_221, 3)
    clamp_min_33: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_min.default(add_222, 0);  add_222 = None
    clamp_max_33: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_33, 6);  clamp_min_33 = None
    mul_285: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(add_221, clamp_max_33);  add_221 = clamp_max_33 = None
    div_33: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(mul_285, 6);  mul_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_46: "f32[8, 216, 14, 14]" = torch.ops.aten.convolution.default(div_33, primals_233, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_223: "i64[]" = torch.ops.aten.add.Tensor(primals_445, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_36 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_72: "f32[1, 216, 1, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 216, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_224: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
    rsqrt_36: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    sub_36: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_73)
    mul_286: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
    squeeze_108: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
    squeeze_109: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
    mul_287: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
    mul_288: "f32[216]" = torch.ops.aten.mul.Tensor(primals_446, 0.9)
    add_225: "f32[216]" = torch.ops.aten.add.Tensor(mul_287, mul_288);  mul_287 = mul_288 = None
    squeeze_110: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
    mul_289: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0006381620931717);  squeeze_110 = None
    mul_290: "f32[216]" = torch.ops.aten.mul.Tensor(mul_289, 0.1);  mul_289 = None
    mul_291: "f32[216]" = torch.ops.aten.mul.Tensor(primals_447, 0.9)
    add_226: "f32[216]" = torch.ops.aten.add.Tensor(mul_290, mul_291);  mul_290 = mul_291 = None
    unsqueeze_144: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1)
    unsqueeze_145: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
    mul_292: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_145);  mul_286 = unsqueeze_145 = None
    unsqueeze_146: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_74, -1);  primals_74 = None
    unsqueeze_147: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
    add_227: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(mul_292, unsqueeze_147);  mul_292 = unsqueeze_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_29: "f32[8, 216, 14, 14]" = torch.ops.aten.clone.default(add_227)
    add_228: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(add_227, 3)
    clamp_min_34: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_min.default(add_228, 0);  add_228 = None
    clamp_max_34: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_34, 6);  clamp_min_34 = None
    mul_293: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(add_227, clamp_max_34);  add_227 = clamp_max_34 = None
    div_34: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(mul_293, 6);  mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_47: "f32[8, 72, 14, 14]" = torch.ops.aten.convolution.default(div_34, primals_234, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_229: "i64[]" = torch.ops.aten.add.Tensor(primals_448, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_37 = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
    getitem_74: "f32[1, 72, 1, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 72, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_230: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
    rsqrt_37: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_230);  add_230 = None
    sub_37: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_75)
    mul_294: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    squeeze_111: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
    squeeze_112: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
    mul_295: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
    mul_296: "f32[72]" = torch.ops.aten.mul.Tensor(primals_449, 0.9)
    add_231: "f32[72]" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    squeeze_113: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
    mul_297: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0006381620931717);  squeeze_113 = None
    mul_298: "f32[72]" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
    mul_299: "f32[72]" = torch.ops.aten.mul.Tensor(primals_450, 0.9)
    add_232: "f32[72]" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_148: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_75, -1)
    unsqueeze_149: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
    mul_300: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_149);  mul_294 = unsqueeze_149 = None
    unsqueeze_150: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_76, -1);  primals_76 = None
    unsqueeze_151: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
    add_233: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_151);  mul_300 = unsqueeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_234: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(add_233, add_216);  add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_48: "f32[8, 216, 14, 14]" = torch.ops.aten.convolution.default(add_234, primals_235, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_235: "i64[]" = torch.ops.aten.add.Tensor(primals_451, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_38 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
    getitem_76: "f32[1, 216, 1, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 216, 1, 1]" = var_mean_38[1];  var_mean_38 = None
    add_236: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
    rsqrt_38: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
    sub_38: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_77)
    mul_301: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    squeeze_114: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
    squeeze_115: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
    mul_302: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
    mul_303: "f32[216]" = torch.ops.aten.mul.Tensor(primals_452, 0.9)
    add_237: "f32[216]" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
    squeeze_116: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
    mul_304: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0006381620931717);  squeeze_116 = None
    mul_305: "f32[216]" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
    mul_306: "f32[216]" = torch.ops.aten.mul.Tensor(primals_453, 0.9)
    add_238: "f32[216]" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_152: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_77, -1)
    unsqueeze_153: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
    mul_307: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_153);  mul_301 = unsqueeze_153 = None
    unsqueeze_154: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
    unsqueeze_155: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
    add_239: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_155);  mul_307 = unsqueeze_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_30: "f32[8, 216, 14, 14]" = torch.ops.aten.clone.default(add_239)
    add_240: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(add_239, 3)
    clamp_min_35: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_min.default(add_240, 0);  add_240 = None
    clamp_max_35: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_35, 6);  clamp_min_35 = None
    mul_308: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(add_239, clamp_max_35);  add_239 = clamp_max_35 = None
    div_35: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(mul_308, 6);  mul_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_49: "f32[8, 216, 14, 14]" = torch.ops.aten.convolution.default(div_35, primals_236, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_241: "i64[]" = torch.ops.aten.add.Tensor(primals_454, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 216, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 216, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_242: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
    rsqrt_39: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
    sub_39: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_79)
    mul_309: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    squeeze_117: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_118: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    mul_310: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
    mul_311: "f32[216]" = torch.ops.aten.mul.Tensor(primals_455, 0.9)
    add_243: "f32[216]" = torch.ops.aten.add.Tensor(mul_310, mul_311);  mul_310 = mul_311 = None
    squeeze_119: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
    mul_312: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0006381620931717);  squeeze_119 = None
    mul_313: "f32[216]" = torch.ops.aten.mul.Tensor(mul_312, 0.1);  mul_312 = None
    mul_314: "f32[216]" = torch.ops.aten.mul.Tensor(primals_456, 0.9)
    add_244: "f32[216]" = torch.ops.aten.add.Tensor(mul_313, mul_314);  mul_313 = mul_314 = None
    unsqueeze_156: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1)
    unsqueeze_157: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
    mul_315: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(mul_309, unsqueeze_157);  mul_309 = unsqueeze_157 = None
    unsqueeze_158: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_80, -1);  primals_80 = None
    unsqueeze_159: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
    add_245: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(mul_315, unsqueeze_159);  mul_315 = unsqueeze_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_31: "f32[8, 216, 14, 14]" = torch.ops.aten.clone.default(add_245)
    add_246: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(add_245, 3)
    clamp_min_36: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_min.default(add_246, 0);  add_246 = None
    clamp_max_36: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_36, 6);  clamp_min_36 = None
    mul_316: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(add_245, clamp_max_36);  add_245 = clamp_max_36 = None
    div_36: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(mul_316, 6);  mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_50: "f32[8, 72, 14, 14]" = torch.ops.aten.convolution.default(div_36, primals_237, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_247: "i64[]" = torch.ops.aten.add.Tensor(primals_457, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_40 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
    getitem_80: "f32[1, 72, 1, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 72, 1, 1]" = var_mean_40[1];  var_mean_40 = None
    add_248: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
    rsqrt_40: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_248);  add_248 = None
    sub_40: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_81)
    mul_317: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
    squeeze_120: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
    squeeze_121: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
    mul_318: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
    mul_319: "f32[72]" = torch.ops.aten.mul.Tensor(primals_458, 0.9)
    add_249: "f32[72]" = torch.ops.aten.add.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
    squeeze_122: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
    mul_320: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0006381620931717);  squeeze_122 = None
    mul_321: "f32[72]" = torch.ops.aten.mul.Tensor(mul_320, 0.1);  mul_320 = None
    mul_322: "f32[72]" = torch.ops.aten.mul.Tensor(primals_459, 0.9)
    add_250: "f32[72]" = torch.ops.aten.add.Tensor(mul_321, mul_322);  mul_321 = mul_322 = None
    unsqueeze_160: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_81, -1)
    unsqueeze_161: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
    mul_323: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(mul_317, unsqueeze_161);  mul_317 = unsqueeze_161 = None
    unsqueeze_162: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_82, -1);  primals_82 = None
    unsqueeze_163: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
    add_251: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(mul_323, unsqueeze_163);  mul_323 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_252: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(add_251, add_234);  add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_51: "f32[8, 216, 14, 14]" = torch.ops.aten.convolution.default(add_252, primals_238, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_253: "i64[]" = torch.ops.aten.add.Tensor(primals_460, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_41 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_82: "f32[1, 216, 1, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 216, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_254: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
    rsqrt_41: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
    sub_41: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_83)
    mul_324: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
    squeeze_123: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
    squeeze_124: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
    mul_325: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
    mul_326: "f32[216]" = torch.ops.aten.mul.Tensor(primals_461, 0.9)
    add_255: "f32[216]" = torch.ops.aten.add.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
    squeeze_125: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
    mul_327: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0006381620931717);  squeeze_125 = None
    mul_328: "f32[216]" = torch.ops.aten.mul.Tensor(mul_327, 0.1);  mul_327 = None
    mul_329: "f32[216]" = torch.ops.aten.mul.Tensor(primals_462, 0.9)
    add_256: "f32[216]" = torch.ops.aten.add.Tensor(mul_328, mul_329);  mul_328 = mul_329 = None
    unsqueeze_164: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_83, -1)
    unsqueeze_165: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
    mul_330: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(mul_324, unsqueeze_165);  mul_324 = unsqueeze_165 = None
    unsqueeze_166: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
    unsqueeze_167: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
    add_257: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(mul_330, unsqueeze_167);  mul_330 = unsqueeze_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_32: "f32[8, 216, 14, 14]" = torch.ops.aten.clone.default(add_257)
    add_258: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(add_257, 3)
    clamp_min_37: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_min.default(add_258, 0);  add_258 = None
    clamp_max_37: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_37, 6);  clamp_min_37 = None
    mul_331: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(add_257, clamp_max_37);  add_257 = clamp_max_37 = None
    div_37: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(mul_331, 6);  mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_52: "f32[8, 216, 14, 14]" = torch.ops.aten.convolution.default(div_37, primals_239, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_259: "i64[]" = torch.ops.aten.add.Tensor(primals_463, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_42 = torch.ops.aten.var_mean.correction(convolution_52, [0, 2, 3], correction = 0, keepdim = True)
    getitem_84: "f32[1, 216, 1, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 216, 1, 1]" = var_mean_42[1];  var_mean_42 = None
    add_260: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
    rsqrt_42: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
    sub_42: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, getitem_85)
    mul_332: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
    squeeze_126: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
    squeeze_127: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
    mul_333: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
    mul_334: "f32[216]" = torch.ops.aten.mul.Tensor(primals_464, 0.9)
    add_261: "f32[216]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    squeeze_128: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
    mul_335: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0006381620931717);  squeeze_128 = None
    mul_336: "f32[216]" = torch.ops.aten.mul.Tensor(mul_335, 0.1);  mul_335 = None
    mul_337: "f32[216]" = torch.ops.aten.mul.Tensor(primals_465, 0.9)
    add_262: "f32[216]" = torch.ops.aten.add.Tensor(mul_336, mul_337);  mul_336 = mul_337 = None
    unsqueeze_168: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1)
    unsqueeze_169: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
    mul_338: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(mul_332, unsqueeze_169);  mul_332 = unsqueeze_169 = None
    unsqueeze_170: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_86, -1);  primals_86 = None
    unsqueeze_171: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
    add_263: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(mul_338, unsqueeze_171);  mul_338 = unsqueeze_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_33: "f32[8, 216, 14, 14]" = torch.ops.aten.clone.default(add_263)
    add_264: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(add_263, 3)
    clamp_min_38: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_min.default(add_264, 0);  add_264 = None
    clamp_max_38: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_38, 6);  clamp_min_38 = None
    mul_339: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(add_263, clamp_max_38);  add_263 = clamp_max_38 = None
    div_38: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(mul_339, 6);  mul_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_53: "f32[8, 72, 14, 14]" = torch.ops.aten.convolution.default(div_38, primals_240, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_265: "i64[]" = torch.ops.aten.add.Tensor(primals_466, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 72, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 72, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_266: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
    rsqrt_43: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
    sub_43: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_87)
    mul_340: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
    squeeze_129: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_130: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    mul_341: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
    mul_342: "f32[72]" = torch.ops.aten.mul.Tensor(primals_467, 0.9)
    add_267: "f32[72]" = torch.ops.aten.add.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
    squeeze_131: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
    mul_343: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0006381620931717);  squeeze_131 = None
    mul_344: "f32[72]" = torch.ops.aten.mul.Tensor(mul_343, 0.1);  mul_343 = None
    mul_345: "f32[72]" = torch.ops.aten.mul.Tensor(primals_468, 0.9)
    add_268: "f32[72]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    unsqueeze_172: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_87, -1)
    unsqueeze_173: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
    mul_346: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_173);  mul_340 = unsqueeze_173 = None
    unsqueeze_174: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_88, -1);  primals_88 = None
    unsqueeze_175: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
    add_269: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(mul_346, unsqueeze_175);  mul_346 = unsqueeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_270: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(add_269, add_252);  add_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_54: "f32[8, 216, 14, 14]" = torch.ops.aten.convolution.default(add_270, primals_241, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_271: "i64[]" = torch.ops.aten.add.Tensor(primals_469, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_44 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
    getitem_88: "f32[1, 216, 1, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 216, 1, 1]" = var_mean_44[1];  var_mean_44 = None
    add_272: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
    rsqrt_44: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
    sub_44: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_89)
    mul_347: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
    squeeze_132: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
    squeeze_133: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
    mul_348: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
    mul_349: "f32[216]" = torch.ops.aten.mul.Tensor(primals_470, 0.9)
    add_273: "f32[216]" = torch.ops.aten.add.Tensor(mul_348, mul_349);  mul_348 = mul_349 = None
    squeeze_134: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
    mul_350: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0006381620931717);  squeeze_134 = None
    mul_351: "f32[216]" = torch.ops.aten.mul.Tensor(mul_350, 0.1);  mul_350 = None
    mul_352: "f32[216]" = torch.ops.aten.mul.Tensor(primals_471, 0.9)
    add_274: "f32[216]" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    unsqueeze_176: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_89, -1)
    unsqueeze_177: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
    mul_353: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(mul_347, unsqueeze_177);  mul_347 = unsqueeze_177 = None
    unsqueeze_178: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
    unsqueeze_179: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
    add_275: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(mul_353, unsqueeze_179);  mul_353 = unsqueeze_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_34: "f32[8, 216, 14, 14]" = torch.ops.aten.clone.default(add_275)
    add_276: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(add_275, 3)
    clamp_min_39: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_min.default(add_276, 0);  add_276 = None
    clamp_max_39: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_39, 6);  clamp_min_39 = None
    mul_354: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(add_275, clamp_max_39);  add_275 = clamp_max_39 = None
    div_39: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(mul_354, 6);  mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_55: "f32[8, 216, 14, 14]" = torch.ops.aten.convolution.default(div_39, primals_242, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 216)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_277: "i64[]" = torch.ops.aten.add.Tensor(primals_472, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_45 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
    getitem_90: "f32[1, 216, 1, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 216, 1, 1]" = var_mean_45[1];  var_mean_45 = None
    add_278: "f32[1, 216, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
    rsqrt_45: "f32[1, 216, 1, 1]" = torch.ops.aten.rsqrt.default(add_278);  add_278 = None
    sub_45: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_91)
    mul_355: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
    squeeze_135: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
    squeeze_136: "f32[216]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
    mul_356: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
    mul_357: "f32[216]" = torch.ops.aten.mul.Tensor(primals_473, 0.9)
    add_279: "f32[216]" = torch.ops.aten.add.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
    squeeze_137: "f32[216]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
    mul_358: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0006381620931717);  squeeze_137 = None
    mul_359: "f32[216]" = torch.ops.aten.mul.Tensor(mul_358, 0.1);  mul_358 = None
    mul_360: "f32[216]" = torch.ops.aten.mul.Tensor(primals_474, 0.9)
    add_280: "f32[216]" = torch.ops.aten.add.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
    unsqueeze_180: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1)
    unsqueeze_181: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
    mul_361: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(mul_355, unsqueeze_181);  mul_355 = unsqueeze_181 = None
    unsqueeze_182: "f32[216, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
    unsqueeze_183: "f32[216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
    add_281: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(mul_361, unsqueeze_183);  mul_361 = unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_35: "f32[8, 216, 14, 14]" = torch.ops.aten.clone.default(add_281)
    add_282: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(add_281, 3)
    clamp_min_40: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_min.default(add_282, 0);  add_282 = None
    clamp_max_40: "f32[8, 216, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_40, 6);  clamp_min_40 = None
    mul_362: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(add_281, clamp_max_40);  add_281 = clamp_max_40 = None
    div_40: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(mul_362, 6);  mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_56: "f32[8, 72, 14, 14]" = torch.ops.aten.convolution.default(div_40, primals_243, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_283: "i64[]" = torch.ops.aten.add.Tensor(primals_475, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_46 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_92: "f32[1, 72, 1, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 72, 1, 1]" = var_mean_46[1];  var_mean_46 = None
    add_284: "f32[1, 72, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
    rsqrt_46: "f32[1, 72, 1, 1]" = torch.ops.aten.rsqrt.default(add_284);  add_284 = None
    sub_46: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_93)
    mul_363: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
    squeeze_138: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
    squeeze_139: "f32[72]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
    mul_364: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
    mul_365: "f32[72]" = torch.ops.aten.mul.Tensor(primals_476, 0.9)
    add_285: "f32[72]" = torch.ops.aten.add.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
    squeeze_140: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
    mul_366: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0006381620931717);  squeeze_140 = None
    mul_367: "f32[72]" = torch.ops.aten.mul.Tensor(mul_366, 0.1);  mul_366 = None
    mul_368: "f32[72]" = torch.ops.aten.mul.Tensor(primals_477, 0.9)
    add_286: "f32[72]" = torch.ops.aten.add.Tensor(mul_367, mul_368);  mul_367 = mul_368 = None
    unsqueeze_184: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1)
    unsqueeze_185: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
    mul_369: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(mul_363, unsqueeze_185);  mul_363 = unsqueeze_185 = None
    unsqueeze_186: "f32[72, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1);  primals_94 = None
    unsqueeze_187: "f32[72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
    add_287: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(mul_369, unsqueeze_187);  mul_369 = unsqueeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_288: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(add_287, add_270);  add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_57: "f32[8, 360, 14, 14]" = torch.ops.aten.convolution.default(add_288, primals_244, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_289: "i64[]" = torch.ops.aten.add.Tensor(primals_478, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 360, 1, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 360, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_290: "f32[1, 360, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
    rsqrt_47: "f32[1, 360, 1, 1]" = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
    sub_47: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_95)
    mul_370: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
    squeeze_141: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_142: "f32[360]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    mul_371: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
    mul_372: "f32[360]" = torch.ops.aten.mul.Tensor(primals_479, 0.9)
    add_291: "f32[360]" = torch.ops.aten.add.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
    squeeze_143: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
    mul_373: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0006381620931717);  squeeze_143 = None
    mul_374: "f32[360]" = torch.ops.aten.mul.Tensor(mul_373, 0.1);  mul_373 = None
    mul_375: "f32[360]" = torch.ops.aten.mul.Tensor(primals_480, 0.9)
    add_292: "f32[360]" = torch.ops.aten.add.Tensor(mul_374, mul_375);  mul_374 = mul_375 = None
    unsqueeze_188: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1)
    unsqueeze_189: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
    mul_376: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_189);  mul_370 = unsqueeze_189 = None
    unsqueeze_190: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
    unsqueeze_191: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
    add_293: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_376, unsqueeze_191);  mul_376 = unsqueeze_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_36: "f32[8, 360, 14, 14]" = torch.ops.aten.clone.default(add_293)
    add_294: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(add_293, 3)
    clamp_min_41: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_min.default(add_294, 0);  add_294 = None
    clamp_max_41: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_41, 6);  clamp_min_41 = None
    mul_377: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_293, clamp_max_41);  add_293 = clamp_max_41 = None
    div_41: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(mul_377, 6);  mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_58: "f32[8, 360, 14, 14]" = torch.ops.aten.convolution.default(div_41, primals_245, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 360)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_295: "i64[]" = torch.ops.aten.add.Tensor(primals_481, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_48 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
    getitem_96: "f32[1, 360, 1, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 360, 1, 1]" = var_mean_48[1];  var_mean_48 = None
    add_296: "f32[1, 360, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
    rsqrt_48: "f32[1, 360, 1, 1]" = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
    sub_48: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_97)
    mul_378: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
    squeeze_144: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
    squeeze_145: "f32[360]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
    mul_379: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
    mul_380: "f32[360]" = torch.ops.aten.mul.Tensor(primals_482, 0.9)
    add_297: "f32[360]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    squeeze_146: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
    mul_381: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0006381620931717);  squeeze_146 = None
    mul_382: "f32[360]" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
    mul_383: "f32[360]" = torch.ops.aten.mul.Tensor(primals_483, 0.9)
    add_298: "f32[360]" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
    unsqueeze_192: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
    unsqueeze_193: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
    mul_384: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_193);  mul_378 = unsqueeze_193 = None
    unsqueeze_194: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
    unsqueeze_195: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
    add_299: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_195);  mul_384 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_37: "f32[8, 360, 14, 14]" = torch.ops.aten.clone.default(add_299)
    add_300: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(add_299, 3)
    clamp_min_42: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_min.default(add_300, 0);  add_300 = None
    clamp_max_42: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_42, 6);  clamp_min_42 = None
    mul_385: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_299, clamp_max_42);  add_299 = clamp_max_42 = None
    div_42: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(mul_385, 6);  mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_5: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_42, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_59: "f32[8, 24, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_246, primals_247, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_38: "f32[8, 24, 1, 1]" = torch.ops.aten.clone.default(convolution_59)
    add_301: "f32[8, 24, 1, 1]" = torch.ops.aten.add.Tensor(convolution_59, 3)
    clamp_min_43: "f32[8, 24, 1, 1]" = torch.ops.aten.clamp_min.default(add_301, 0);  add_301 = None
    clamp_max_43: "f32[8, 24, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_43, 6);  clamp_min_43 = None
    mul_386: "f32[8, 24, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_59, clamp_max_43);  convolution_59 = clamp_max_43 = None
    div_43: "f32[8, 24, 1, 1]" = torch.ops.aten.div.Tensor(mul_386, 6);  mul_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_60: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_43, primals_248, primals_249, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_302: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_60, 3)
    clamp_min_44: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_302, 0);  add_302 = None
    clamp_max_44: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_44, 6);  clamp_min_44 = None
    div_44: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_44, 6);  clamp_max_44 = None
    mul_387: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(div_42, div_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_61: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(mul_387, primals_250, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_303: "i64[]" = torch.ops.aten.add.Tensor(primals_484, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_49 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_98: "f32[1, 120, 1, 1]" = var_mean_49[0]
    getitem_99: "f32[1, 120, 1, 1]" = var_mean_49[1];  var_mean_49 = None
    add_304: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
    rsqrt_49: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_304);  add_304 = None
    sub_49: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_99)
    mul_388: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
    squeeze_147: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
    squeeze_148: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
    mul_389: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
    mul_390: "f32[120]" = torch.ops.aten.mul.Tensor(primals_485, 0.9)
    add_305: "f32[120]" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
    squeeze_149: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
    mul_391: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0006381620931717);  squeeze_149 = None
    mul_392: "f32[120]" = torch.ops.aten.mul.Tensor(mul_391, 0.1);  mul_391 = None
    mul_393: "f32[120]" = torch.ops.aten.mul.Tensor(primals_486, 0.9)
    add_306: "f32[120]" = torch.ops.aten.add.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
    unsqueeze_196: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1)
    unsqueeze_197: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
    mul_394: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_197);  mul_388 = unsqueeze_197 = None
    unsqueeze_198: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1);  primals_100 = None
    unsqueeze_199: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
    add_307: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(mul_394, unsqueeze_199);  mul_394 = unsqueeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_62: "f32[8, 360, 14, 14]" = torch.ops.aten.convolution.default(add_307, primals_251, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_308: "i64[]" = torch.ops.aten.add.Tensor(primals_487, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_50 = torch.ops.aten.var_mean.correction(convolution_62, [0, 2, 3], correction = 0, keepdim = True)
    getitem_100: "f32[1, 360, 1, 1]" = var_mean_50[0]
    getitem_101: "f32[1, 360, 1, 1]" = var_mean_50[1];  var_mean_50 = None
    add_309: "f32[1, 360, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
    rsqrt_50: "f32[1, 360, 1, 1]" = torch.ops.aten.rsqrt.default(add_309);  add_309 = None
    sub_50: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, getitem_101)
    mul_395: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
    squeeze_150: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
    squeeze_151: "f32[360]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
    mul_396: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
    mul_397: "f32[360]" = torch.ops.aten.mul.Tensor(primals_488, 0.9)
    add_310: "f32[360]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
    squeeze_152: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
    mul_398: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0006381620931717);  squeeze_152 = None
    mul_399: "f32[360]" = torch.ops.aten.mul.Tensor(mul_398, 0.1);  mul_398 = None
    mul_400: "f32[360]" = torch.ops.aten.mul.Tensor(primals_489, 0.9)
    add_311: "f32[360]" = torch.ops.aten.add.Tensor(mul_399, mul_400);  mul_399 = mul_400 = None
    unsqueeze_200: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1)
    unsqueeze_201: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
    mul_401: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(mul_395, unsqueeze_201);  mul_395 = unsqueeze_201 = None
    unsqueeze_202: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
    unsqueeze_203: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
    add_312: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_401, unsqueeze_203);  mul_401 = unsqueeze_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_39: "f32[8, 360, 14, 14]" = torch.ops.aten.clone.default(add_312)
    add_313: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(add_312, 3)
    clamp_min_45: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_min.default(add_313, 0);  add_313 = None
    clamp_max_45: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_45, 6);  clamp_min_45 = None
    mul_402: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_312, clamp_max_45);  add_312 = clamp_max_45 = None
    div_45: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(mul_402, 6);  mul_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_63: "f32[8, 360, 14, 14]" = torch.ops.aten.convolution.default(div_45, primals_252, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_314: "i64[]" = torch.ops.aten.add.Tensor(primals_490, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 360, 1, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 360, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_315: "f32[1, 360, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
    rsqrt_51: "f32[1, 360, 1, 1]" = torch.ops.aten.rsqrt.default(add_315);  add_315 = None
    sub_51: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_103)
    mul_403: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
    squeeze_153: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_154: "f32[360]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    mul_404: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
    mul_405: "f32[360]" = torch.ops.aten.mul.Tensor(primals_491, 0.9)
    add_316: "f32[360]" = torch.ops.aten.add.Tensor(mul_404, mul_405);  mul_404 = mul_405 = None
    squeeze_155: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
    mul_406: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0006381620931717);  squeeze_155 = None
    mul_407: "f32[360]" = torch.ops.aten.mul.Tensor(mul_406, 0.1);  mul_406 = None
    mul_408: "f32[360]" = torch.ops.aten.mul.Tensor(primals_492, 0.9)
    add_317: "f32[360]" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    unsqueeze_204: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1)
    unsqueeze_205: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
    mul_409: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(mul_403, unsqueeze_205);  mul_403 = unsqueeze_205 = None
    unsqueeze_206: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
    unsqueeze_207: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
    add_318: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_409, unsqueeze_207);  mul_409 = unsqueeze_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_40: "f32[8, 360, 14, 14]" = torch.ops.aten.clone.default(add_318)
    add_319: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(add_318, 3)
    clamp_min_46: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_min.default(add_319, 0);  add_319 = None
    clamp_max_46: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_46, 6);  clamp_min_46 = None
    mul_410: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_318, clamp_max_46);  add_318 = clamp_max_46 = None
    div_46: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(mul_410, 6);  mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_6: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_46, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_64: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_253, primals_254, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_41: "f32[8, 32, 1, 1]" = torch.ops.aten.clone.default(convolution_64)
    add_320: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_64, 3)
    clamp_min_47: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_320, 0);  add_320 = None
    clamp_max_47: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_47, 6);  clamp_min_47 = None
    mul_411: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_64, clamp_max_47);  convolution_64 = clamp_max_47 = None
    div_47: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_411, 6);  mul_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_65: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_47, primals_255, primals_256, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_321: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_65, 3)
    clamp_min_48: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_321, 0);  add_321 = None
    clamp_max_48: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_48, 6);  clamp_min_48 = None
    div_48: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_48, 6);  clamp_max_48 = None
    mul_412: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(div_46, div_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_66: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(mul_412, primals_257, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_322: "i64[]" = torch.ops.aten.add.Tensor(primals_493, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_52 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_104: "f32[1, 120, 1, 1]" = var_mean_52[0]
    getitem_105: "f32[1, 120, 1, 1]" = var_mean_52[1];  var_mean_52 = None
    add_323: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
    rsqrt_52: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_323);  add_323 = None
    sub_52: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_105)
    mul_413: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
    squeeze_156: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
    squeeze_157: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
    mul_414: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
    mul_415: "f32[120]" = torch.ops.aten.mul.Tensor(primals_494, 0.9)
    add_324: "f32[120]" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
    squeeze_158: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
    mul_416: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0006381620931717);  squeeze_158 = None
    mul_417: "f32[120]" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
    mul_418: "f32[120]" = torch.ops.aten.mul.Tensor(primals_495, 0.9)
    add_325: "f32[120]" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
    unsqueeze_208: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1)
    unsqueeze_209: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
    mul_419: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_209);  mul_413 = unsqueeze_209 = None
    unsqueeze_210: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1);  primals_106 = None
    unsqueeze_211: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
    add_326: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_211);  mul_419 = unsqueeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_327: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_326, add_307);  add_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_67: "f32[8, 360, 14, 14]" = torch.ops.aten.convolution.default(add_327, primals_258, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_328: "i64[]" = torch.ops.aten.add.Tensor(primals_496, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_53 = torch.ops.aten.var_mean.correction(convolution_67, [0, 2, 3], correction = 0, keepdim = True)
    getitem_106: "f32[1, 360, 1, 1]" = var_mean_53[0]
    getitem_107: "f32[1, 360, 1, 1]" = var_mean_53[1];  var_mean_53 = None
    add_329: "f32[1, 360, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
    rsqrt_53: "f32[1, 360, 1, 1]" = torch.ops.aten.rsqrt.default(add_329);  add_329 = None
    sub_53: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, getitem_107)
    mul_420: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
    squeeze_159: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
    squeeze_160: "f32[360]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
    mul_421: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
    mul_422: "f32[360]" = torch.ops.aten.mul.Tensor(primals_497, 0.9)
    add_330: "f32[360]" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    squeeze_161: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
    mul_423: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0006381620931717);  squeeze_161 = None
    mul_424: "f32[360]" = torch.ops.aten.mul.Tensor(mul_423, 0.1);  mul_423 = None
    mul_425: "f32[360]" = torch.ops.aten.mul.Tensor(primals_498, 0.9)
    add_331: "f32[360]" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
    unsqueeze_212: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1)
    unsqueeze_213: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
    mul_426: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_213);  mul_420 = unsqueeze_213 = None
    unsqueeze_214: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
    unsqueeze_215: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
    add_332: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_215);  mul_426 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_42: "f32[8, 360, 14, 14]" = torch.ops.aten.clone.default(add_332)
    add_333: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(add_332, 3)
    clamp_min_49: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_min.default(add_333, 0);  add_333 = None
    clamp_max_49: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_49, 6);  clamp_min_49 = None
    mul_427: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_332, clamp_max_49);  add_332 = clamp_max_49 = None
    div_49: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(mul_427, 6);  mul_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_68: "f32[8, 360, 14, 14]" = torch.ops.aten.convolution.default(div_49, primals_259, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_334: "i64[]" = torch.ops.aten.add.Tensor(primals_499, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_54 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True)
    getitem_108: "f32[1, 360, 1, 1]" = var_mean_54[0]
    getitem_109: "f32[1, 360, 1, 1]" = var_mean_54[1];  var_mean_54 = None
    add_335: "f32[1, 360, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
    rsqrt_54: "f32[1, 360, 1, 1]" = torch.ops.aten.rsqrt.default(add_335);  add_335 = None
    sub_54: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_109)
    mul_428: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
    squeeze_162: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
    squeeze_163: "f32[360]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
    mul_429: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
    mul_430: "f32[360]" = torch.ops.aten.mul.Tensor(primals_500, 0.9)
    add_336: "f32[360]" = torch.ops.aten.add.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
    squeeze_164: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
    mul_431: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0006381620931717);  squeeze_164 = None
    mul_432: "f32[360]" = torch.ops.aten.mul.Tensor(mul_431, 0.1);  mul_431 = None
    mul_433: "f32[360]" = torch.ops.aten.mul.Tensor(primals_501, 0.9)
    add_337: "f32[360]" = torch.ops.aten.add.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
    unsqueeze_216: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1)
    unsqueeze_217: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
    mul_434: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(mul_428, unsqueeze_217);  mul_428 = unsqueeze_217 = None
    unsqueeze_218: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_110, -1);  primals_110 = None
    unsqueeze_219: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
    add_338: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_434, unsqueeze_219);  mul_434 = unsqueeze_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_43: "f32[8, 360, 14, 14]" = torch.ops.aten.clone.default(add_338)
    add_339: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(add_338, 3)
    clamp_min_50: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_min.default(add_339, 0);  add_339 = None
    clamp_max_50: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_50, 6);  clamp_min_50 = None
    mul_435: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_338, clamp_max_50);  add_338 = clamp_max_50 = None
    div_50: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(mul_435, 6);  mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_7: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_50, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_69: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_260, primals_261, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_44: "f32[8, 32, 1, 1]" = torch.ops.aten.clone.default(convolution_69)
    add_340: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_69, 3)
    clamp_min_51: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_340, 0);  add_340 = None
    clamp_max_51: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_51, 6);  clamp_min_51 = None
    mul_436: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_69, clamp_max_51);  convolution_69 = clamp_max_51 = None
    div_51: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_436, 6);  mul_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_70: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_51, primals_262, primals_263, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_341: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_70, 3)
    clamp_min_52: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_341, 0);  add_341 = None
    clamp_max_52: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_52, 6);  clamp_min_52 = None
    div_52: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_52, 6);  clamp_max_52 = None
    mul_437: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(div_50, div_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_71: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(mul_437, primals_264, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_342: "i64[]" = torch.ops.aten.add.Tensor(primals_502, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 120, 1, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 120, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_343: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
    rsqrt_55: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_343);  add_343 = None
    sub_55: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_111)
    mul_438: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
    squeeze_165: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_166: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    mul_439: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
    mul_440: "f32[120]" = torch.ops.aten.mul.Tensor(primals_503, 0.9)
    add_344: "f32[120]" = torch.ops.aten.add.Tensor(mul_439, mul_440);  mul_439 = mul_440 = None
    squeeze_167: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
    mul_441: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0006381620931717);  squeeze_167 = None
    mul_442: "f32[120]" = torch.ops.aten.mul.Tensor(mul_441, 0.1);  mul_441 = None
    mul_443: "f32[120]" = torch.ops.aten.mul.Tensor(primals_504, 0.9)
    add_345: "f32[120]" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    unsqueeze_220: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_111, -1)
    unsqueeze_221: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
    mul_444: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(mul_438, unsqueeze_221);  mul_438 = unsqueeze_221 = None
    unsqueeze_222: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1);  primals_112 = None
    unsqueeze_223: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
    add_346: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(mul_444, unsqueeze_223);  mul_444 = unsqueeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_347: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_346, add_327);  add_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_72: "f32[8, 360, 14, 14]" = torch.ops.aten.convolution.default(add_347, primals_265, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_348: "i64[]" = torch.ops.aten.add.Tensor(primals_505, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_56 = torch.ops.aten.var_mean.correction(convolution_72, [0, 2, 3], correction = 0, keepdim = True)
    getitem_112: "f32[1, 360, 1, 1]" = var_mean_56[0]
    getitem_113: "f32[1, 360, 1, 1]" = var_mean_56[1];  var_mean_56 = None
    add_349: "f32[1, 360, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
    rsqrt_56: "f32[1, 360, 1, 1]" = torch.ops.aten.rsqrt.default(add_349);  add_349 = None
    sub_56: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, getitem_113)
    mul_445: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
    squeeze_168: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
    squeeze_169: "f32[360]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
    mul_446: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
    mul_447: "f32[360]" = torch.ops.aten.mul.Tensor(primals_506, 0.9)
    add_350: "f32[360]" = torch.ops.aten.add.Tensor(mul_446, mul_447);  mul_446 = mul_447 = None
    squeeze_170: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
    mul_448: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0006381620931717);  squeeze_170 = None
    mul_449: "f32[360]" = torch.ops.aten.mul.Tensor(mul_448, 0.1);  mul_448 = None
    mul_450: "f32[360]" = torch.ops.aten.mul.Tensor(primals_507, 0.9)
    add_351: "f32[360]" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    unsqueeze_224: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1)
    unsqueeze_225: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
    mul_451: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_225);  mul_445 = unsqueeze_225 = None
    unsqueeze_226: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
    unsqueeze_227: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
    add_352: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_451, unsqueeze_227);  mul_451 = unsqueeze_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_45: "f32[8, 360, 14, 14]" = torch.ops.aten.clone.default(add_352)
    add_353: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(add_352, 3)
    clamp_min_53: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_min.default(add_353, 0);  add_353 = None
    clamp_max_53: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_53, 6);  clamp_min_53 = None
    mul_452: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_352, clamp_max_53);  add_352 = clamp_max_53 = None
    div_53: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(mul_452, 6);  mul_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_73: "f32[8, 360, 14, 14]" = torch.ops.aten.convolution.default(div_53, primals_266, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_354: "i64[]" = torch.ops.aten.add.Tensor(primals_508, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_57 = torch.ops.aten.var_mean.correction(convolution_73, [0, 2, 3], correction = 0, keepdim = True)
    getitem_114: "f32[1, 360, 1, 1]" = var_mean_57[0]
    getitem_115: "f32[1, 360, 1, 1]" = var_mean_57[1];  var_mean_57 = None
    add_355: "f32[1, 360, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
    rsqrt_57: "f32[1, 360, 1, 1]" = torch.ops.aten.rsqrt.default(add_355);  add_355 = None
    sub_57: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, getitem_115)
    mul_453: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
    squeeze_171: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
    squeeze_172: "f32[360]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
    mul_454: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
    mul_455: "f32[360]" = torch.ops.aten.mul.Tensor(primals_509, 0.9)
    add_356: "f32[360]" = torch.ops.aten.add.Tensor(mul_454, mul_455);  mul_454 = mul_455 = None
    squeeze_173: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
    mul_456: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0006381620931717);  squeeze_173 = None
    mul_457: "f32[360]" = torch.ops.aten.mul.Tensor(mul_456, 0.1);  mul_456 = None
    mul_458: "f32[360]" = torch.ops.aten.mul.Tensor(primals_510, 0.9)
    add_357: "f32[360]" = torch.ops.aten.add.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
    unsqueeze_228: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_115, -1)
    unsqueeze_229: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
    mul_459: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(mul_453, unsqueeze_229);  mul_453 = unsqueeze_229 = None
    unsqueeze_230: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1);  primals_116 = None
    unsqueeze_231: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
    add_358: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_459, unsqueeze_231);  mul_459 = unsqueeze_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_46: "f32[8, 360, 14, 14]" = torch.ops.aten.clone.default(add_358)
    add_359: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(add_358, 3)
    clamp_min_54: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_min.default(add_359, 0);  add_359 = None
    clamp_max_54: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_54, 6);  clamp_min_54 = None
    mul_460: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_358, clamp_max_54);  add_358 = clamp_max_54 = None
    div_54: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(mul_460, 6);  mul_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_8: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_54, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_74: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_267, primals_268, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_47: "f32[8, 32, 1, 1]" = torch.ops.aten.clone.default(convolution_74)
    add_360: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_74, 3)
    clamp_min_55: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_360, 0);  add_360 = None
    clamp_max_55: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_55, 6);  clamp_min_55 = None
    mul_461: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_74, clamp_max_55);  convolution_74 = clamp_max_55 = None
    div_55: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_461, 6);  mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_75: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_55, primals_269, primals_270, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_361: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_75, 3)
    clamp_min_56: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_361, 0);  add_361 = None
    clamp_max_56: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_56, 6);  clamp_min_56 = None
    div_56: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_56, 6);  clamp_max_56 = None
    mul_462: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(div_54, div_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_76: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(mul_462, primals_271, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_362: "i64[]" = torch.ops.aten.add.Tensor(primals_511, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_58 = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
    getitem_116: "f32[1, 120, 1, 1]" = var_mean_58[0]
    getitem_117: "f32[1, 120, 1, 1]" = var_mean_58[1];  var_mean_58 = None
    add_363: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
    rsqrt_58: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_363);  add_363 = None
    sub_58: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_117)
    mul_463: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
    squeeze_174: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
    squeeze_175: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
    mul_464: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
    mul_465: "f32[120]" = torch.ops.aten.mul.Tensor(primals_512, 0.9)
    add_364: "f32[120]" = torch.ops.aten.add.Tensor(mul_464, mul_465);  mul_464 = mul_465 = None
    squeeze_176: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
    mul_466: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0006381620931717);  squeeze_176 = None
    mul_467: "f32[120]" = torch.ops.aten.mul.Tensor(mul_466, 0.1);  mul_466 = None
    mul_468: "f32[120]" = torch.ops.aten.mul.Tensor(primals_513, 0.9)
    add_365: "f32[120]" = torch.ops.aten.add.Tensor(mul_467, mul_468);  mul_467 = mul_468 = None
    unsqueeze_232: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1)
    unsqueeze_233: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
    mul_469: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(mul_463, unsqueeze_233);  mul_463 = unsqueeze_233 = None
    unsqueeze_234: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1);  primals_118 = None
    unsqueeze_235: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
    add_366: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(mul_469, unsqueeze_235);  mul_469 = unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_367: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_366, add_347);  add_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_77: "f32[8, 360, 14, 14]" = torch.ops.aten.convolution.default(add_367, primals_272, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_368: "i64[]" = torch.ops.aten.add.Tensor(primals_514, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_77, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 360, 1, 1]" = var_mean_59[0]
    getitem_119: "f32[1, 360, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_369: "f32[1, 360, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
    rsqrt_59: "f32[1, 360, 1, 1]" = torch.ops.aten.rsqrt.default(add_369);  add_369 = None
    sub_59: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, getitem_119)
    mul_470: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
    squeeze_177: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_178: "f32[360]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    mul_471: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
    mul_472: "f32[360]" = torch.ops.aten.mul.Tensor(primals_515, 0.9)
    add_370: "f32[360]" = torch.ops.aten.add.Tensor(mul_471, mul_472);  mul_471 = mul_472 = None
    squeeze_179: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
    mul_473: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0006381620931717);  squeeze_179 = None
    mul_474: "f32[360]" = torch.ops.aten.mul.Tensor(mul_473, 0.1);  mul_473 = None
    mul_475: "f32[360]" = torch.ops.aten.mul.Tensor(primals_516, 0.9)
    add_371: "f32[360]" = torch.ops.aten.add.Tensor(mul_474, mul_475);  mul_474 = mul_475 = None
    unsqueeze_236: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
    unsqueeze_237: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
    mul_476: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(mul_470, unsqueeze_237);  mul_470 = unsqueeze_237 = None
    unsqueeze_238: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
    unsqueeze_239: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
    add_372: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_476, unsqueeze_239);  mul_476 = unsqueeze_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_48: "f32[8, 360, 14, 14]" = torch.ops.aten.clone.default(add_372)
    add_373: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(add_372, 3)
    clamp_min_57: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_min.default(add_373, 0);  add_373 = None
    clamp_max_57: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_57, 6);  clamp_min_57 = None
    mul_477: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_372, clamp_max_57);  add_372 = clamp_max_57 = None
    div_57: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(mul_477, 6);  mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_78: "f32[8, 360, 14, 14]" = torch.ops.aten.convolution.default(div_57, primals_273, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_374: "i64[]" = torch.ops.aten.add.Tensor(primals_517, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_60 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
    getitem_120: "f32[1, 360, 1, 1]" = var_mean_60[0]
    getitem_121: "f32[1, 360, 1, 1]" = var_mean_60[1];  var_mean_60 = None
    add_375: "f32[1, 360, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
    rsqrt_60: "f32[1, 360, 1, 1]" = torch.ops.aten.rsqrt.default(add_375);  add_375 = None
    sub_60: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_121)
    mul_478: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
    squeeze_180: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
    squeeze_181: "f32[360]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
    mul_479: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
    mul_480: "f32[360]" = torch.ops.aten.mul.Tensor(primals_518, 0.9)
    add_376: "f32[360]" = torch.ops.aten.add.Tensor(mul_479, mul_480);  mul_479 = mul_480 = None
    squeeze_182: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
    mul_481: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0006381620931717);  squeeze_182 = None
    mul_482: "f32[360]" = torch.ops.aten.mul.Tensor(mul_481, 0.1);  mul_481 = None
    mul_483: "f32[360]" = torch.ops.aten.mul.Tensor(primals_519, 0.9)
    add_377: "f32[360]" = torch.ops.aten.add.Tensor(mul_482, mul_483);  mul_482 = mul_483 = None
    unsqueeze_240: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1)
    unsqueeze_241: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
    mul_484: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(mul_478, unsqueeze_241);  mul_478 = unsqueeze_241 = None
    unsqueeze_242: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1);  primals_122 = None
    unsqueeze_243: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
    add_378: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_484, unsqueeze_243);  mul_484 = unsqueeze_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_49: "f32[8, 360, 14, 14]" = torch.ops.aten.clone.default(add_378)
    add_379: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(add_378, 3)
    clamp_min_58: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_min.default(add_379, 0);  add_379 = None
    clamp_max_58: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_58, 6);  clamp_min_58 = None
    mul_485: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_378, clamp_max_58);  add_378 = clamp_max_58 = None
    div_58: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(mul_485, 6);  mul_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_9: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_58, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_79: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_274, primals_275, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_50: "f32[8, 32, 1, 1]" = torch.ops.aten.clone.default(convolution_79)
    add_380: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_79, 3)
    clamp_min_59: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_380, 0);  add_380 = None
    clamp_max_59: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_59, 6);  clamp_min_59 = None
    mul_486: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_79, clamp_max_59);  convolution_79 = clamp_max_59 = None
    div_59: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_486, 6);  mul_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_80: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_59, primals_276, primals_277, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_381: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_80, 3)
    clamp_min_60: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_381, 0);  add_381 = None
    clamp_max_60: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_60, 6);  clamp_min_60 = None
    div_60: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_60, 6);  clamp_max_60 = None
    mul_487: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(div_58, div_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_81: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(mul_487, primals_278, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_382: "i64[]" = torch.ops.aten.add.Tensor(primals_520, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_61 = torch.ops.aten.var_mean.correction(convolution_81, [0, 2, 3], correction = 0, keepdim = True)
    getitem_122: "f32[1, 120, 1, 1]" = var_mean_61[0]
    getitem_123: "f32[1, 120, 1, 1]" = var_mean_61[1];  var_mean_61 = None
    add_383: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
    rsqrt_61: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_383);  add_383 = None
    sub_61: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, getitem_123)
    mul_488: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
    squeeze_183: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
    squeeze_184: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
    mul_489: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
    mul_490: "f32[120]" = torch.ops.aten.mul.Tensor(primals_521, 0.9)
    add_384: "f32[120]" = torch.ops.aten.add.Tensor(mul_489, mul_490);  mul_489 = mul_490 = None
    squeeze_185: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
    mul_491: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0006381620931717);  squeeze_185 = None
    mul_492: "f32[120]" = torch.ops.aten.mul.Tensor(mul_491, 0.1);  mul_491 = None
    mul_493: "f32[120]" = torch.ops.aten.mul.Tensor(primals_522, 0.9)
    add_385: "f32[120]" = torch.ops.aten.add.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
    unsqueeze_244: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1)
    unsqueeze_245: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
    mul_494: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(mul_488, unsqueeze_245);  mul_488 = unsqueeze_245 = None
    unsqueeze_246: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_124, -1);  primals_124 = None
    unsqueeze_247: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
    add_386: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(mul_494, unsqueeze_247);  mul_494 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_387: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_386, add_367);  add_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_82: "f32[8, 360, 14, 14]" = torch.ops.aten.convolution.default(add_387, primals_279, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_388: "i64[]" = torch.ops.aten.add.Tensor(primals_523, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_62 = torch.ops.aten.var_mean.correction(convolution_82, [0, 2, 3], correction = 0, keepdim = True)
    getitem_124: "f32[1, 360, 1, 1]" = var_mean_62[0]
    getitem_125: "f32[1, 360, 1, 1]" = var_mean_62[1];  var_mean_62 = None
    add_389: "f32[1, 360, 1, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
    rsqrt_62: "f32[1, 360, 1, 1]" = torch.ops.aten.rsqrt.default(add_389);  add_389 = None
    sub_62: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, getitem_125)
    mul_495: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
    squeeze_186: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
    squeeze_187: "f32[360]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
    mul_496: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
    mul_497: "f32[360]" = torch.ops.aten.mul.Tensor(primals_524, 0.9)
    add_390: "f32[360]" = torch.ops.aten.add.Tensor(mul_496, mul_497);  mul_496 = mul_497 = None
    squeeze_188: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
    mul_498: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0006381620931717);  squeeze_188 = None
    mul_499: "f32[360]" = torch.ops.aten.mul.Tensor(mul_498, 0.1);  mul_498 = None
    mul_500: "f32[360]" = torch.ops.aten.mul.Tensor(primals_525, 0.9)
    add_391: "f32[360]" = torch.ops.aten.add.Tensor(mul_499, mul_500);  mul_499 = mul_500 = None
    unsqueeze_248: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_125, -1)
    unsqueeze_249: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
    mul_501: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(mul_495, unsqueeze_249);  mul_495 = unsqueeze_249 = None
    unsqueeze_250: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
    unsqueeze_251: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
    add_392: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_501, unsqueeze_251);  mul_501 = unsqueeze_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_51: "f32[8, 360, 14, 14]" = torch.ops.aten.clone.default(add_392)
    add_393: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(add_392, 3)
    clamp_min_61: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_min.default(add_393, 0);  add_393 = None
    clamp_max_61: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_61, 6);  clamp_min_61 = None
    mul_502: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_392, clamp_max_61);  add_392 = clamp_max_61 = None
    div_61: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(mul_502, 6);  mul_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_83: "f32[8, 360, 14, 14]" = torch.ops.aten.convolution.default(div_61, primals_280, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 360)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_394: "i64[]" = torch.ops.aten.add.Tensor(primals_526, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_63 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
    getitem_126: "f32[1, 360, 1, 1]" = var_mean_63[0]
    getitem_127: "f32[1, 360, 1, 1]" = var_mean_63[1];  var_mean_63 = None
    add_395: "f32[1, 360, 1, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
    rsqrt_63: "f32[1, 360, 1, 1]" = torch.ops.aten.rsqrt.default(add_395);  add_395 = None
    sub_63: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_127)
    mul_503: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
    squeeze_189: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
    squeeze_190: "f32[360]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
    mul_504: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
    mul_505: "f32[360]" = torch.ops.aten.mul.Tensor(primals_527, 0.9)
    add_396: "f32[360]" = torch.ops.aten.add.Tensor(mul_504, mul_505);  mul_504 = mul_505 = None
    squeeze_191: "f32[360]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
    mul_506: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0006381620931717);  squeeze_191 = None
    mul_507: "f32[360]" = torch.ops.aten.mul.Tensor(mul_506, 0.1);  mul_506 = None
    mul_508: "f32[360]" = torch.ops.aten.mul.Tensor(primals_528, 0.9)
    add_397: "f32[360]" = torch.ops.aten.add.Tensor(mul_507, mul_508);  mul_507 = mul_508 = None
    unsqueeze_252: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1)
    unsqueeze_253: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
    mul_509: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(mul_503, unsqueeze_253);  mul_503 = unsqueeze_253 = None
    unsqueeze_254: "f32[360, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1);  primals_128 = None
    unsqueeze_255: "f32[360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
    add_398: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_509, unsqueeze_255);  mul_509 = unsqueeze_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_52: "f32[8, 360, 14, 14]" = torch.ops.aten.clone.default(add_398)
    add_399: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(add_398, 3)
    clamp_min_62: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_min.default(add_399, 0);  add_399 = None
    clamp_max_62: "f32[8, 360, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_62, 6);  clamp_min_62 = None
    mul_510: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_398, clamp_max_62);  add_398 = clamp_max_62 = None
    div_62: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(mul_510, 6);  mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_10: "f32[8, 360, 1, 1]" = torch.ops.aten.mean.dim(div_62, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_84: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_281, primals_282, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_53: "f32[8, 32, 1, 1]" = torch.ops.aten.clone.default(convolution_84)
    add_400: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_84, 3)
    clamp_min_63: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_400, 0);  add_400 = None
    clamp_max_63: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_63, 6);  clamp_min_63 = None
    mul_511: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_84, clamp_max_63);  convolution_84 = clamp_max_63 = None
    div_63: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_511, 6);  mul_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_85: "f32[8, 360, 1, 1]" = torch.ops.aten.convolution.default(div_63, primals_283, primals_284, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_401: "f32[8, 360, 1, 1]" = torch.ops.aten.add.Tensor(convolution_85, 3)
    clamp_min_64: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_min.default(add_401, 0);  add_401 = None
    clamp_max_64: "f32[8, 360, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_64, 6);  clamp_min_64 = None
    div_64: "f32[8, 360, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_64, 6);  clamp_max_64 = None
    mul_512: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(div_62, div_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_86: "f32[8, 120, 14, 14]" = torch.ops.aten.convolution.default(mul_512, primals_285, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_402: "i64[]" = torch.ops.aten.add.Tensor(primals_529, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_64 = torch.ops.aten.var_mean.correction(convolution_86, [0, 2, 3], correction = 0, keepdim = True)
    getitem_128: "f32[1, 120, 1, 1]" = var_mean_64[0]
    getitem_129: "f32[1, 120, 1, 1]" = var_mean_64[1];  var_mean_64 = None
    add_403: "f32[1, 120, 1, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
    rsqrt_64: "f32[1, 120, 1, 1]" = torch.ops.aten.rsqrt.default(add_403);  add_403 = None
    sub_64: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, getitem_129)
    mul_513: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
    squeeze_192: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
    squeeze_193: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
    mul_514: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
    mul_515: "f32[120]" = torch.ops.aten.mul.Tensor(primals_530, 0.9)
    add_404: "f32[120]" = torch.ops.aten.add.Tensor(mul_514, mul_515);  mul_514 = mul_515 = None
    squeeze_194: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
    mul_516: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0006381620931717);  squeeze_194 = None
    mul_517: "f32[120]" = torch.ops.aten.mul.Tensor(mul_516, 0.1);  mul_516 = None
    mul_518: "f32[120]" = torch.ops.aten.mul.Tensor(primals_531, 0.9)
    add_405: "f32[120]" = torch.ops.aten.add.Tensor(mul_517, mul_518);  mul_517 = mul_518 = None
    unsqueeze_256: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1)
    unsqueeze_257: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
    mul_519: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(mul_513, unsqueeze_257);  mul_513 = unsqueeze_257 = None
    unsqueeze_258: "f32[120, 1]" = torch.ops.aten.unsqueeze.default(primals_130, -1);  primals_130 = None
    unsqueeze_259: "f32[120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
    add_406: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(mul_519, unsqueeze_259);  mul_519 = unsqueeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_407: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_406, add_387);  add_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_87: "f32[8, 720, 14, 14]" = torch.ops.aten.convolution.default(add_407, primals_286, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_408: "i64[]" = torch.ops.aten.add.Tensor(primals_532, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_65 = torch.ops.aten.var_mean.correction(convolution_87, [0, 2, 3], correction = 0, keepdim = True)
    getitem_130: "f32[1, 720, 1, 1]" = var_mean_65[0]
    getitem_131: "f32[1, 720, 1, 1]" = var_mean_65[1];  var_mean_65 = None
    add_409: "f32[1, 720, 1, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05)
    rsqrt_65: "f32[1, 720, 1, 1]" = torch.ops.aten.rsqrt.default(add_409);  add_409 = None
    sub_65: "f32[8, 720, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, getitem_131)
    mul_520: "f32[8, 720, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
    squeeze_195: "f32[720]" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
    squeeze_196: "f32[720]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
    mul_521: "f32[720]" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
    mul_522: "f32[720]" = torch.ops.aten.mul.Tensor(primals_533, 0.9)
    add_410: "f32[720]" = torch.ops.aten.add.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
    squeeze_197: "f32[720]" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
    mul_523: "f32[720]" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0006381620931717);  squeeze_197 = None
    mul_524: "f32[720]" = torch.ops.aten.mul.Tensor(mul_523, 0.1);  mul_523 = None
    mul_525: "f32[720]" = torch.ops.aten.mul.Tensor(primals_534, 0.9)
    add_411: "f32[720]" = torch.ops.aten.add.Tensor(mul_524, mul_525);  mul_524 = mul_525 = None
    unsqueeze_260: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(primals_131, -1)
    unsqueeze_261: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
    mul_526: "f32[8, 720, 14, 14]" = torch.ops.aten.mul.Tensor(mul_520, unsqueeze_261);  mul_520 = unsqueeze_261 = None
    unsqueeze_262: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
    unsqueeze_263: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
    add_412: "f32[8, 720, 14, 14]" = torch.ops.aten.add.Tensor(mul_526, unsqueeze_263);  mul_526 = unsqueeze_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_54: "f32[8, 720, 14, 14]" = torch.ops.aten.clone.default(add_412)
    add_413: "f32[8, 720, 14, 14]" = torch.ops.aten.add.Tensor(add_412, 3)
    clamp_min_65: "f32[8, 720, 14, 14]" = torch.ops.aten.clamp_min.default(add_413, 0);  add_413 = None
    clamp_max_65: "f32[8, 720, 14, 14]" = torch.ops.aten.clamp_max.default(clamp_min_65, 6);  clamp_min_65 = None
    mul_527: "f32[8, 720, 14, 14]" = torch.ops.aten.mul.Tensor(add_412, clamp_max_65);  add_412 = clamp_max_65 = None
    div_65: "f32[8, 720, 14, 14]" = torch.ops.aten.div.Tensor(mul_527, 6);  mul_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_88: "f32[8, 720, 7, 7]" = torch.ops.aten.convolution.default(div_65, primals_287, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 720)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_414: "i64[]" = torch.ops.aten.add.Tensor(primals_535, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_66 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
    getitem_132: "f32[1, 720, 1, 1]" = var_mean_66[0]
    getitem_133: "f32[1, 720, 1, 1]" = var_mean_66[1];  var_mean_66 = None
    add_415: "f32[1, 720, 1, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-05)
    rsqrt_66: "f32[1, 720, 1, 1]" = torch.ops.aten.rsqrt.default(add_415);  add_415 = None
    sub_66: "f32[8, 720, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_133)
    mul_528: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
    squeeze_198: "f32[720]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
    squeeze_199: "f32[720]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
    mul_529: "f32[720]" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
    mul_530: "f32[720]" = torch.ops.aten.mul.Tensor(primals_536, 0.9)
    add_416: "f32[720]" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    squeeze_200: "f32[720]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
    mul_531: "f32[720]" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0025575447570332);  squeeze_200 = None
    mul_532: "f32[720]" = torch.ops.aten.mul.Tensor(mul_531, 0.1);  mul_531 = None
    mul_533: "f32[720]" = torch.ops.aten.mul.Tensor(primals_537, 0.9)
    add_417: "f32[720]" = torch.ops.aten.add.Tensor(mul_532, mul_533);  mul_532 = mul_533 = None
    unsqueeze_264: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(primals_133, -1)
    unsqueeze_265: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
    mul_534: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(mul_528, unsqueeze_265);  mul_528 = unsqueeze_265 = None
    unsqueeze_266: "f32[720, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1);  primals_134 = None
    unsqueeze_267: "f32[720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
    add_418: "f32[8, 720, 7, 7]" = torch.ops.aten.add.Tensor(mul_534, unsqueeze_267);  mul_534 = unsqueeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_55: "f32[8, 720, 7, 7]" = torch.ops.aten.clone.default(add_418)
    add_419: "f32[8, 720, 7, 7]" = torch.ops.aten.add.Tensor(add_418, 3)
    clamp_min_66: "f32[8, 720, 7, 7]" = torch.ops.aten.clamp_min.default(add_419, 0);  add_419 = None
    clamp_max_66: "f32[8, 720, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_66, 6);  clamp_min_66 = None
    mul_535: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(add_418, clamp_max_66);  add_418 = clamp_max_66 = None
    div_66: "f32[8, 720, 7, 7]" = torch.ops.aten.div.Tensor(mul_535, 6);  mul_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_11: "f32[8, 720, 1, 1]" = torch.ops.aten.mean.dim(div_66, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_89: "f32[8, 32, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_288, primals_289, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_56: "f32[8, 32, 1, 1]" = torch.ops.aten.clone.default(convolution_89)
    add_420: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(convolution_89, 3)
    clamp_min_67: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_min.default(add_420, 0);  add_420 = None
    clamp_max_67: "f32[8, 32, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_67, 6);  clamp_min_67 = None
    mul_536: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_89, clamp_max_67);  convolution_89 = clamp_max_67 = None
    div_67: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(mul_536, 6);  mul_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_90: "f32[8, 720, 1, 1]" = torch.ops.aten.convolution.default(div_67, primals_290, primals_291, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_421: "f32[8, 720, 1, 1]" = torch.ops.aten.add.Tensor(convolution_90, 3)
    clamp_min_68: "f32[8, 720, 1, 1]" = torch.ops.aten.clamp_min.default(add_421, 0);  add_421 = None
    clamp_max_68: "f32[8, 720, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_68, 6);  clamp_min_68 = None
    div_68: "f32[8, 720, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_68, 6);  clamp_max_68 = None
    mul_537: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(div_66, div_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_91: "f32[8, 184, 7, 7]" = torch.ops.aten.convolution.default(mul_537, primals_292, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_422: "i64[]" = torch.ops.aten.add.Tensor(primals_538, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_67 = torch.ops.aten.var_mean.correction(convolution_91, [0, 2, 3], correction = 0, keepdim = True)
    getitem_134: "f32[1, 184, 1, 1]" = var_mean_67[0]
    getitem_135: "f32[1, 184, 1, 1]" = var_mean_67[1];  var_mean_67 = None
    add_423: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05)
    rsqrt_67: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_423);  add_423 = None
    sub_67: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_91, getitem_135)
    mul_538: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
    squeeze_201: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
    squeeze_202: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
    mul_539: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
    mul_540: "f32[184]" = torch.ops.aten.mul.Tensor(primals_539, 0.9)
    add_424: "f32[184]" = torch.ops.aten.add.Tensor(mul_539, mul_540);  mul_539 = mul_540 = None
    squeeze_203: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
    mul_541: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_203, 1.0025575447570332);  squeeze_203 = None
    mul_542: "f32[184]" = torch.ops.aten.mul.Tensor(mul_541, 0.1);  mul_541 = None
    mul_543: "f32[184]" = torch.ops.aten.mul.Tensor(primals_540, 0.9)
    add_425: "f32[184]" = torch.ops.aten.add.Tensor(mul_542, mul_543);  mul_542 = mul_543 = None
    unsqueeze_268: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1)
    unsqueeze_269: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
    mul_544: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(mul_538, unsqueeze_269);  mul_538 = unsqueeze_269 = None
    unsqueeze_270: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_136, -1);  primals_136 = None
    unsqueeze_271: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
    add_426: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(mul_544, unsqueeze_271);  mul_544 = unsqueeze_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_92: "f32[8, 736, 7, 7]" = torch.ops.aten.convolution.default(add_426, primals_293, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_427: "i64[]" = torch.ops.aten.add.Tensor(primals_541, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_68 = torch.ops.aten.var_mean.correction(convolution_92, [0, 2, 3], correction = 0, keepdim = True)
    getitem_136: "f32[1, 736, 1, 1]" = var_mean_68[0]
    getitem_137: "f32[1, 736, 1, 1]" = var_mean_68[1];  var_mean_68 = None
    add_428: "f32[1, 736, 1, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05)
    rsqrt_68: "f32[1, 736, 1, 1]" = torch.ops.aten.rsqrt.default(add_428);  add_428 = None
    sub_68: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_92, getitem_137)
    mul_545: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
    squeeze_204: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
    squeeze_205: "f32[736]" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
    mul_546: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1)
    mul_547: "f32[736]" = torch.ops.aten.mul.Tensor(primals_542, 0.9)
    add_429: "f32[736]" = torch.ops.aten.add.Tensor(mul_546, mul_547);  mul_546 = mul_547 = None
    squeeze_206: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
    mul_548: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_206, 1.0025575447570332);  squeeze_206 = None
    mul_549: "f32[736]" = torch.ops.aten.mul.Tensor(mul_548, 0.1);  mul_548 = None
    mul_550: "f32[736]" = torch.ops.aten.mul.Tensor(primals_543, 0.9)
    add_430: "f32[736]" = torch.ops.aten.add.Tensor(mul_549, mul_550);  mul_549 = mul_550 = None
    unsqueeze_272: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_137, -1)
    unsqueeze_273: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
    mul_551: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(mul_545, unsqueeze_273);  mul_545 = unsqueeze_273 = None
    unsqueeze_274: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
    unsqueeze_275: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
    add_431: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_551, unsqueeze_275);  mul_551 = unsqueeze_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_57: "f32[8, 736, 7, 7]" = torch.ops.aten.clone.default(add_431)
    add_432: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(add_431, 3)
    clamp_min_69: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_min.default(add_432, 0);  add_432 = None
    clamp_max_69: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_69, 6);  clamp_min_69 = None
    mul_552: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_431, clamp_max_69);  add_431 = clamp_max_69 = None
    div_69: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(mul_552, 6);  mul_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_93: "f32[8, 736, 7, 7]" = torch.ops.aten.convolution.default(div_69, primals_294, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_433: "i64[]" = torch.ops.aten.add.Tensor(primals_544, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_69 = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True)
    getitem_138: "f32[1, 736, 1, 1]" = var_mean_69[0]
    getitem_139: "f32[1, 736, 1, 1]" = var_mean_69[1];  var_mean_69 = None
    add_434: "f32[1, 736, 1, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05)
    rsqrt_69: "f32[1, 736, 1, 1]" = torch.ops.aten.rsqrt.default(add_434);  add_434 = None
    sub_69: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_139)
    mul_553: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
    squeeze_207: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
    squeeze_208: "f32[736]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
    mul_554: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1)
    mul_555: "f32[736]" = torch.ops.aten.mul.Tensor(primals_545, 0.9)
    add_435: "f32[736]" = torch.ops.aten.add.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
    squeeze_209: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
    mul_556: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_209, 1.0025575447570332);  squeeze_209 = None
    mul_557: "f32[736]" = torch.ops.aten.mul.Tensor(mul_556, 0.1);  mul_556 = None
    mul_558: "f32[736]" = torch.ops.aten.mul.Tensor(primals_546, 0.9)
    add_436: "f32[736]" = torch.ops.aten.add.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
    unsqueeze_276: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_139, -1)
    unsqueeze_277: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
    mul_559: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_277);  mul_553 = unsqueeze_277 = None
    unsqueeze_278: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1);  primals_140 = None
    unsqueeze_279: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
    add_437: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_559, unsqueeze_279);  mul_559 = unsqueeze_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_58: "f32[8, 736, 7, 7]" = torch.ops.aten.clone.default(add_437)
    add_438: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(add_437, 3)
    clamp_min_70: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_min.default(add_438, 0);  add_438 = None
    clamp_max_70: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_70, 6);  clamp_min_70 = None
    mul_560: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_437, clamp_max_70);  add_437 = clamp_max_70 = None
    div_70: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(mul_560, 6);  mul_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_12: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_70, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_94: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_295, primals_296, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_59: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_94)
    add_439: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_94, 3)
    clamp_min_71: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_439, 0);  add_439 = None
    clamp_max_71: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_71, 6);  clamp_min_71 = None
    mul_561: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_94, clamp_max_71);  convolution_94 = clamp_max_71 = None
    div_71: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_561, 6);  mul_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_95: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_71, primals_297, primals_298, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_440: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_95, 3)
    clamp_min_72: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_440, 0);  add_440 = None
    clamp_max_72: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_72, 6);  clamp_min_72 = None
    div_72: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_72, 6);  clamp_max_72 = None
    mul_562: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(div_70, div_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_96: "f32[8, 184, 7, 7]" = torch.ops.aten.convolution.default(mul_562, primals_299, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_441: "i64[]" = torch.ops.aten.add.Tensor(primals_547, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_70 = torch.ops.aten.var_mean.correction(convolution_96, [0, 2, 3], correction = 0, keepdim = True)
    getitem_140: "f32[1, 184, 1, 1]" = var_mean_70[0]
    getitem_141: "f32[1, 184, 1, 1]" = var_mean_70[1];  var_mean_70 = None
    add_442: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05)
    rsqrt_70: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_442);  add_442 = None
    sub_70: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_96, getitem_141)
    mul_563: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
    squeeze_210: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
    squeeze_211: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
    mul_564: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
    mul_565: "f32[184]" = torch.ops.aten.mul.Tensor(primals_548, 0.9)
    add_443: "f32[184]" = torch.ops.aten.add.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    squeeze_212: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
    mul_566: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_212, 1.0025575447570332);  squeeze_212 = None
    mul_567: "f32[184]" = torch.ops.aten.mul.Tensor(mul_566, 0.1);  mul_566 = None
    mul_568: "f32[184]" = torch.ops.aten.mul.Tensor(primals_549, 0.9)
    add_444: "f32[184]" = torch.ops.aten.add.Tensor(mul_567, mul_568);  mul_567 = mul_568 = None
    unsqueeze_280: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1)
    unsqueeze_281: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
    mul_569: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(mul_563, unsqueeze_281);  mul_563 = unsqueeze_281 = None
    unsqueeze_282: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_142, -1);  primals_142 = None
    unsqueeze_283: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
    add_445: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(mul_569, unsqueeze_283);  mul_569 = unsqueeze_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_446: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_445, add_426);  add_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_97: "f32[8, 736, 7, 7]" = torch.ops.aten.convolution.default(add_446, primals_300, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_447: "i64[]" = torch.ops.aten.add.Tensor(primals_550, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_71 = torch.ops.aten.var_mean.correction(convolution_97, [0, 2, 3], correction = 0, keepdim = True)
    getitem_142: "f32[1, 736, 1, 1]" = var_mean_71[0]
    getitem_143: "f32[1, 736, 1, 1]" = var_mean_71[1];  var_mean_71 = None
    add_448: "f32[1, 736, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05)
    rsqrt_71: "f32[1, 736, 1, 1]" = torch.ops.aten.rsqrt.default(add_448);  add_448 = None
    sub_71: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_97, getitem_143)
    mul_570: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
    squeeze_213: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
    squeeze_214: "f32[736]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
    mul_571: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1)
    mul_572: "f32[736]" = torch.ops.aten.mul.Tensor(primals_551, 0.9)
    add_449: "f32[736]" = torch.ops.aten.add.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
    squeeze_215: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
    mul_573: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_215, 1.0025575447570332);  squeeze_215 = None
    mul_574: "f32[736]" = torch.ops.aten.mul.Tensor(mul_573, 0.1);  mul_573 = None
    mul_575: "f32[736]" = torch.ops.aten.mul.Tensor(primals_552, 0.9)
    add_450: "f32[736]" = torch.ops.aten.add.Tensor(mul_574, mul_575);  mul_574 = mul_575 = None
    unsqueeze_284: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_143, -1)
    unsqueeze_285: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
    mul_576: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(mul_570, unsqueeze_285);  mul_570 = unsqueeze_285 = None
    unsqueeze_286: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
    unsqueeze_287: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
    add_451: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_576, unsqueeze_287);  mul_576 = unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_60: "f32[8, 736, 7, 7]" = torch.ops.aten.clone.default(add_451)
    add_452: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(add_451, 3)
    clamp_min_73: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_min.default(add_452, 0);  add_452 = None
    clamp_max_73: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_73, 6);  clamp_min_73 = None
    mul_577: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_451, clamp_max_73);  add_451 = clamp_max_73 = None
    div_73: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(mul_577, 6);  mul_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_98: "f32[8, 736, 7, 7]" = torch.ops.aten.convolution.default(div_73, primals_301, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_453: "i64[]" = torch.ops.aten.add.Tensor(primals_553, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_72 = torch.ops.aten.var_mean.correction(convolution_98, [0, 2, 3], correction = 0, keepdim = True)
    getitem_144: "f32[1, 736, 1, 1]" = var_mean_72[0]
    getitem_145: "f32[1, 736, 1, 1]" = var_mean_72[1];  var_mean_72 = None
    add_454: "f32[1, 736, 1, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05)
    rsqrt_72: "f32[1, 736, 1, 1]" = torch.ops.aten.rsqrt.default(add_454);  add_454 = None
    sub_72: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, getitem_145)
    mul_578: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
    squeeze_216: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
    squeeze_217: "f32[736]" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
    mul_579: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1)
    mul_580: "f32[736]" = torch.ops.aten.mul.Tensor(primals_554, 0.9)
    add_455: "f32[736]" = torch.ops.aten.add.Tensor(mul_579, mul_580);  mul_579 = mul_580 = None
    squeeze_218: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
    mul_581: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_218, 1.0025575447570332);  squeeze_218 = None
    mul_582: "f32[736]" = torch.ops.aten.mul.Tensor(mul_581, 0.1);  mul_581 = None
    mul_583: "f32[736]" = torch.ops.aten.mul.Tensor(primals_555, 0.9)
    add_456: "f32[736]" = torch.ops.aten.add.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    unsqueeze_288: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_145, -1)
    unsqueeze_289: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
    mul_584: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(mul_578, unsqueeze_289);  mul_578 = unsqueeze_289 = None
    unsqueeze_290: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_146, -1);  primals_146 = None
    unsqueeze_291: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
    add_457: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_584, unsqueeze_291);  mul_584 = unsqueeze_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_61: "f32[8, 736, 7, 7]" = torch.ops.aten.clone.default(add_457)
    add_458: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(add_457, 3)
    clamp_min_74: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_min.default(add_458, 0);  add_458 = None
    clamp_max_74: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_74, 6);  clamp_min_74 = None
    mul_585: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_457, clamp_max_74);  add_457 = clamp_max_74 = None
    div_74: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(mul_585, 6);  mul_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_13: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_74, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_99: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_13, primals_302, primals_303, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_62: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_99)
    add_459: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_99, 3)
    clamp_min_75: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_459, 0);  add_459 = None
    clamp_max_75: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_75, 6);  clamp_min_75 = None
    mul_586: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_99, clamp_max_75);  convolution_99 = clamp_max_75 = None
    div_75: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_586, 6);  mul_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_100: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_75, primals_304, primals_305, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_460: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_100, 3)
    clamp_min_76: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_460, 0);  add_460 = None
    clamp_max_76: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_76, 6);  clamp_min_76 = None
    div_76: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_76, 6);  clamp_max_76 = None
    mul_587: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(div_74, div_76)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_101: "f32[8, 184, 7, 7]" = torch.ops.aten.convolution.default(mul_587, primals_306, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_461: "i64[]" = torch.ops.aten.add.Tensor(primals_556, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_73 = torch.ops.aten.var_mean.correction(convolution_101, [0, 2, 3], correction = 0, keepdim = True)
    getitem_146: "f32[1, 184, 1, 1]" = var_mean_73[0]
    getitem_147: "f32[1, 184, 1, 1]" = var_mean_73[1];  var_mean_73 = None
    add_462: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05)
    rsqrt_73: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_462);  add_462 = None
    sub_73: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_101, getitem_147)
    mul_588: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
    squeeze_219: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
    squeeze_220: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
    mul_589: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
    mul_590: "f32[184]" = torch.ops.aten.mul.Tensor(primals_557, 0.9)
    add_463: "f32[184]" = torch.ops.aten.add.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
    squeeze_221: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
    mul_591: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_221, 1.0025575447570332);  squeeze_221 = None
    mul_592: "f32[184]" = torch.ops.aten.mul.Tensor(mul_591, 0.1);  mul_591 = None
    mul_593: "f32[184]" = torch.ops.aten.mul.Tensor(primals_558, 0.9)
    add_464: "f32[184]" = torch.ops.aten.add.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
    unsqueeze_292: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_147, -1)
    unsqueeze_293: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
    mul_594: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(mul_588, unsqueeze_293);  mul_588 = unsqueeze_293 = None
    unsqueeze_294: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_148, -1);  primals_148 = None
    unsqueeze_295: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
    add_465: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(mul_594, unsqueeze_295);  mul_594 = unsqueeze_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_466: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_465, add_446);  add_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_102: "f32[8, 736, 7, 7]" = torch.ops.aten.convolution.default(add_466, primals_307, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_467: "i64[]" = torch.ops.aten.add.Tensor(primals_559, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_74 = torch.ops.aten.var_mean.correction(convolution_102, [0, 2, 3], correction = 0, keepdim = True)
    getitem_148: "f32[1, 736, 1, 1]" = var_mean_74[0]
    getitem_149: "f32[1, 736, 1, 1]" = var_mean_74[1];  var_mean_74 = None
    add_468: "f32[1, 736, 1, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05)
    rsqrt_74: "f32[1, 736, 1, 1]" = torch.ops.aten.rsqrt.default(add_468);  add_468 = None
    sub_74: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, getitem_149)
    mul_595: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
    squeeze_222: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
    squeeze_223: "f32[736]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
    mul_596: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
    mul_597: "f32[736]" = torch.ops.aten.mul.Tensor(primals_560, 0.9)
    add_469: "f32[736]" = torch.ops.aten.add.Tensor(mul_596, mul_597);  mul_596 = mul_597 = None
    squeeze_224: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
    mul_598: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_224, 1.0025575447570332);  squeeze_224 = None
    mul_599: "f32[736]" = torch.ops.aten.mul.Tensor(mul_598, 0.1);  mul_598 = None
    mul_600: "f32[736]" = torch.ops.aten.mul.Tensor(primals_561, 0.9)
    add_470: "f32[736]" = torch.ops.aten.add.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
    unsqueeze_296: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_149, -1)
    unsqueeze_297: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
    mul_601: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_297);  mul_595 = unsqueeze_297 = None
    unsqueeze_298: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
    unsqueeze_299: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
    add_471: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_601, unsqueeze_299);  mul_601 = unsqueeze_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_63: "f32[8, 736, 7, 7]" = torch.ops.aten.clone.default(add_471)
    add_472: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(add_471, 3)
    clamp_min_77: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_min.default(add_472, 0);  add_472 = None
    clamp_max_77: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_77, 6);  clamp_min_77 = None
    mul_602: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_471, clamp_max_77);  add_471 = clamp_max_77 = None
    div_77: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(mul_602, 6);  mul_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_103: "f32[8, 736, 7, 7]" = torch.ops.aten.convolution.default(div_77, primals_308, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_473: "i64[]" = torch.ops.aten.add.Tensor(primals_562, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_75 = torch.ops.aten.var_mean.correction(convolution_103, [0, 2, 3], correction = 0, keepdim = True)
    getitem_150: "f32[1, 736, 1, 1]" = var_mean_75[0]
    getitem_151: "f32[1, 736, 1, 1]" = var_mean_75[1];  var_mean_75 = None
    add_474: "f32[1, 736, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
    rsqrt_75: "f32[1, 736, 1, 1]" = torch.ops.aten.rsqrt.default(add_474);  add_474 = None
    sub_75: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_103, getitem_151)
    mul_603: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
    squeeze_225: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
    squeeze_226: "f32[736]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
    mul_604: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1)
    mul_605: "f32[736]" = torch.ops.aten.mul.Tensor(primals_563, 0.9)
    add_475: "f32[736]" = torch.ops.aten.add.Tensor(mul_604, mul_605);  mul_604 = mul_605 = None
    squeeze_227: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
    mul_606: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_227, 1.0025575447570332);  squeeze_227 = None
    mul_607: "f32[736]" = torch.ops.aten.mul.Tensor(mul_606, 0.1);  mul_606 = None
    mul_608: "f32[736]" = torch.ops.aten.mul.Tensor(primals_564, 0.9)
    add_476: "f32[736]" = torch.ops.aten.add.Tensor(mul_607, mul_608);  mul_607 = mul_608 = None
    unsqueeze_300: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_151, -1)
    unsqueeze_301: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
    mul_609: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(mul_603, unsqueeze_301);  mul_603 = unsqueeze_301 = None
    unsqueeze_302: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_152, -1);  primals_152 = None
    unsqueeze_303: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
    add_477: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_609, unsqueeze_303);  mul_609 = unsqueeze_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_64: "f32[8, 736, 7, 7]" = torch.ops.aten.clone.default(add_477)
    add_478: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(add_477, 3)
    clamp_min_78: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_min.default(add_478, 0);  add_478 = None
    clamp_max_78: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_78, 6);  clamp_min_78 = None
    mul_610: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_477, clamp_max_78);  add_477 = clamp_max_78 = None
    div_78: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(mul_610, 6);  mul_610 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_14: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_78, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_104: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_14, primals_309, primals_310, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_65: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_104)
    add_479: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_104, 3)
    clamp_min_79: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_479, 0);  add_479 = None
    clamp_max_79: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_79, 6);  clamp_min_79 = None
    mul_611: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_104, clamp_max_79);  convolution_104 = clamp_max_79 = None
    div_79: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_611, 6);  mul_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_105: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_79, primals_311, primals_312, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_480: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_105, 3)
    clamp_min_80: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_480, 0);  add_480 = None
    clamp_max_80: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_80, 6);  clamp_min_80 = None
    div_80: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_80, 6);  clamp_max_80 = None
    mul_612: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(div_78, div_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_106: "f32[8, 184, 7, 7]" = torch.ops.aten.convolution.default(mul_612, primals_313, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_481: "i64[]" = torch.ops.aten.add.Tensor(primals_565, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_76 = torch.ops.aten.var_mean.correction(convolution_106, [0, 2, 3], correction = 0, keepdim = True)
    getitem_152: "f32[1, 184, 1, 1]" = var_mean_76[0]
    getitem_153: "f32[1, 184, 1, 1]" = var_mean_76[1];  var_mean_76 = None
    add_482: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05)
    rsqrt_76: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_482);  add_482 = None
    sub_76: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_106, getitem_153)
    mul_613: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
    squeeze_228: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
    squeeze_229: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
    mul_614: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1)
    mul_615: "f32[184]" = torch.ops.aten.mul.Tensor(primals_566, 0.9)
    add_483: "f32[184]" = torch.ops.aten.add.Tensor(mul_614, mul_615);  mul_614 = mul_615 = None
    squeeze_230: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
    mul_616: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_230, 1.0025575447570332);  squeeze_230 = None
    mul_617: "f32[184]" = torch.ops.aten.mul.Tensor(mul_616, 0.1);  mul_616 = None
    mul_618: "f32[184]" = torch.ops.aten.mul.Tensor(primals_567, 0.9)
    add_484: "f32[184]" = torch.ops.aten.add.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
    unsqueeze_304: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_153, -1)
    unsqueeze_305: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
    mul_619: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(mul_613, unsqueeze_305);  mul_613 = unsqueeze_305 = None
    unsqueeze_306: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_154, -1);  primals_154 = None
    unsqueeze_307: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
    add_485: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(mul_619, unsqueeze_307);  mul_619 = unsqueeze_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_486: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_485, add_466);  add_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_107: "f32[8, 736, 7, 7]" = torch.ops.aten.convolution.default(add_486, primals_314, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_487: "i64[]" = torch.ops.aten.add.Tensor(primals_568, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_77 = torch.ops.aten.var_mean.correction(convolution_107, [0, 2, 3], correction = 0, keepdim = True)
    getitem_154: "f32[1, 736, 1, 1]" = var_mean_77[0]
    getitem_155: "f32[1, 736, 1, 1]" = var_mean_77[1];  var_mean_77 = None
    add_488: "f32[1, 736, 1, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-05)
    rsqrt_77: "f32[1, 736, 1, 1]" = torch.ops.aten.rsqrt.default(add_488);  add_488 = None
    sub_77: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_107, getitem_155)
    mul_620: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
    squeeze_231: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
    squeeze_232: "f32[736]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
    mul_621: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
    mul_622: "f32[736]" = torch.ops.aten.mul.Tensor(primals_569, 0.9)
    add_489: "f32[736]" = torch.ops.aten.add.Tensor(mul_621, mul_622);  mul_621 = mul_622 = None
    squeeze_233: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
    mul_623: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_233, 1.0025575447570332);  squeeze_233 = None
    mul_624: "f32[736]" = torch.ops.aten.mul.Tensor(mul_623, 0.1);  mul_623 = None
    mul_625: "f32[736]" = torch.ops.aten.mul.Tensor(primals_570, 0.9)
    add_490: "f32[736]" = torch.ops.aten.add.Tensor(mul_624, mul_625);  mul_624 = mul_625 = None
    unsqueeze_308: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_155, -1)
    unsqueeze_309: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
    mul_626: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(mul_620, unsqueeze_309);  mul_620 = unsqueeze_309 = None
    unsqueeze_310: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
    unsqueeze_311: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
    add_491: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_626, unsqueeze_311);  mul_626 = unsqueeze_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_66: "f32[8, 736, 7, 7]" = torch.ops.aten.clone.default(add_491)
    add_492: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(add_491, 3)
    clamp_min_81: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_min.default(add_492, 0);  add_492 = None
    clamp_max_81: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_81, 6);  clamp_min_81 = None
    mul_627: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_491, clamp_max_81);  add_491 = clamp_max_81 = None
    div_81: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(mul_627, 6);  mul_627 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_108: "f32[8, 736, 7, 7]" = torch.ops.aten.convolution.default(div_81, primals_315, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_493: "i64[]" = torch.ops.aten.add.Tensor(primals_571, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_78 = torch.ops.aten.var_mean.correction(convolution_108, [0, 2, 3], correction = 0, keepdim = True)
    getitem_156: "f32[1, 736, 1, 1]" = var_mean_78[0]
    getitem_157: "f32[1, 736, 1, 1]" = var_mean_78[1];  var_mean_78 = None
    add_494: "f32[1, 736, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
    rsqrt_78: "f32[1, 736, 1, 1]" = torch.ops.aten.rsqrt.default(add_494);  add_494 = None
    sub_78: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_108, getitem_157)
    mul_628: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
    squeeze_234: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
    squeeze_235: "f32[736]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
    mul_629: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1)
    mul_630: "f32[736]" = torch.ops.aten.mul.Tensor(primals_572, 0.9)
    add_495: "f32[736]" = torch.ops.aten.add.Tensor(mul_629, mul_630);  mul_629 = mul_630 = None
    squeeze_236: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
    mul_631: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_236, 1.0025575447570332);  squeeze_236 = None
    mul_632: "f32[736]" = torch.ops.aten.mul.Tensor(mul_631, 0.1);  mul_631 = None
    mul_633: "f32[736]" = torch.ops.aten.mul.Tensor(primals_573, 0.9)
    add_496: "f32[736]" = torch.ops.aten.add.Tensor(mul_632, mul_633);  mul_632 = mul_633 = None
    unsqueeze_312: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_157, -1)
    unsqueeze_313: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
    mul_634: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(mul_628, unsqueeze_313);  mul_628 = unsqueeze_313 = None
    unsqueeze_314: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_158, -1);  primals_158 = None
    unsqueeze_315: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
    add_497: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_634, unsqueeze_315);  mul_634 = unsqueeze_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_67: "f32[8, 736, 7, 7]" = torch.ops.aten.clone.default(add_497)
    add_498: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(add_497, 3)
    clamp_min_82: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_min.default(add_498, 0);  add_498 = None
    clamp_max_82: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_82, 6);  clamp_min_82 = None
    mul_635: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_497, clamp_max_82);  add_497 = clamp_max_82 = None
    div_82: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(mul_635, 6);  mul_635 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_15: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_82, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_109: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_15, primals_316, primals_317, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_68: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_109)
    add_499: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_109, 3)
    clamp_min_83: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_499, 0);  add_499 = None
    clamp_max_83: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_83, 6);  clamp_min_83 = None
    mul_636: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_109, clamp_max_83);  convolution_109 = clamp_max_83 = None
    div_83: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_636, 6);  mul_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_110: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_83, primals_318, primals_319, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_500: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_110, 3)
    clamp_min_84: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_500, 0);  add_500 = None
    clamp_max_84: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_84, 6);  clamp_min_84 = None
    div_84: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_84, 6);  clamp_max_84 = None
    mul_637: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(div_82, div_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_111: "f32[8, 184, 7, 7]" = torch.ops.aten.convolution.default(mul_637, primals_320, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_501: "i64[]" = torch.ops.aten.add.Tensor(primals_574, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_79 = torch.ops.aten.var_mean.correction(convolution_111, [0, 2, 3], correction = 0, keepdim = True)
    getitem_158: "f32[1, 184, 1, 1]" = var_mean_79[0]
    getitem_159: "f32[1, 184, 1, 1]" = var_mean_79[1];  var_mean_79 = None
    add_502: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
    rsqrt_79: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_502);  add_502 = None
    sub_79: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_111, getitem_159)
    mul_638: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
    squeeze_237: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
    squeeze_238: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
    mul_639: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1)
    mul_640: "f32[184]" = torch.ops.aten.mul.Tensor(primals_575, 0.9)
    add_503: "f32[184]" = torch.ops.aten.add.Tensor(mul_639, mul_640);  mul_639 = mul_640 = None
    squeeze_239: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
    mul_641: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_239, 1.0025575447570332);  squeeze_239 = None
    mul_642: "f32[184]" = torch.ops.aten.mul.Tensor(mul_641, 0.1);  mul_641 = None
    mul_643: "f32[184]" = torch.ops.aten.mul.Tensor(primals_576, 0.9)
    add_504: "f32[184]" = torch.ops.aten.add.Tensor(mul_642, mul_643);  mul_642 = mul_643 = None
    unsqueeze_316: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_159, -1)
    unsqueeze_317: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
    mul_644: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(mul_638, unsqueeze_317);  mul_638 = unsqueeze_317 = None
    unsqueeze_318: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_160, -1);  primals_160 = None
    unsqueeze_319: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
    add_505: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(mul_644, unsqueeze_319);  mul_644 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_506: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_505, add_486);  add_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_112: "f32[8, 736, 7, 7]" = torch.ops.aten.convolution.default(add_506, primals_321, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_507: "i64[]" = torch.ops.aten.add.Tensor(primals_577, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_80 = torch.ops.aten.var_mean.correction(convolution_112, [0, 2, 3], correction = 0, keepdim = True)
    getitem_160: "f32[1, 736, 1, 1]" = var_mean_80[0]
    getitem_161: "f32[1, 736, 1, 1]" = var_mean_80[1];  var_mean_80 = None
    add_508: "f32[1, 736, 1, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05)
    rsqrt_80: "f32[1, 736, 1, 1]" = torch.ops.aten.rsqrt.default(add_508);  add_508 = None
    sub_80: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_112, getitem_161)
    mul_645: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = None
    squeeze_240: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3]);  getitem_161 = None
    squeeze_241: "f32[736]" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
    mul_646: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_240, 0.1)
    mul_647: "f32[736]" = torch.ops.aten.mul.Tensor(primals_578, 0.9)
    add_509: "f32[736]" = torch.ops.aten.add.Tensor(mul_646, mul_647);  mul_646 = mul_647 = None
    squeeze_242: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_160, [0, 2, 3]);  getitem_160 = None
    mul_648: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_242, 1.0025575447570332);  squeeze_242 = None
    mul_649: "f32[736]" = torch.ops.aten.mul.Tensor(mul_648, 0.1);  mul_648 = None
    mul_650: "f32[736]" = torch.ops.aten.mul.Tensor(primals_579, 0.9)
    add_510: "f32[736]" = torch.ops.aten.add.Tensor(mul_649, mul_650);  mul_649 = mul_650 = None
    unsqueeze_320: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_161, -1)
    unsqueeze_321: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
    mul_651: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(mul_645, unsqueeze_321);  mul_645 = unsqueeze_321 = None
    unsqueeze_322: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1);  primals_162 = None
    unsqueeze_323: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
    add_511: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_651, unsqueeze_323);  mul_651 = unsqueeze_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_69: "f32[8, 736, 7, 7]" = torch.ops.aten.clone.default(add_511)
    add_512: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(add_511, 3)
    clamp_min_85: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_min.default(add_512, 0);  add_512 = None
    clamp_max_85: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_85, 6);  clamp_min_85 = None
    mul_652: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_511, clamp_max_85);  add_511 = clamp_max_85 = None
    div_85: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(mul_652, 6);  mul_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_113: "f32[8, 736, 7, 7]" = torch.ops.aten.convolution.default(div_85, primals_322, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 736)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_513: "i64[]" = torch.ops.aten.add.Tensor(primals_580, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_81 = torch.ops.aten.var_mean.correction(convolution_113, [0, 2, 3], correction = 0, keepdim = True)
    getitem_162: "f32[1, 736, 1, 1]" = var_mean_81[0]
    getitem_163: "f32[1, 736, 1, 1]" = var_mean_81[1];  var_mean_81 = None
    add_514: "f32[1, 736, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05)
    rsqrt_81: "f32[1, 736, 1, 1]" = torch.ops.aten.rsqrt.default(add_514);  add_514 = None
    sub_81: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_113, getitem_163)
    mul_653: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = None
    squeeze_243: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
    squeeze_244: "f32[736]" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
    mul_654: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_243, 0.1)
    mul_655: "f32[736]" = torch.ops.aten.mul.Tensor(primals_581, 0.9)
    add_515: "f32[736]" = torch.ops.aten.add.Tensor(mul_654, mul_655);  mul_654 = mul_655 = None
    squeeze_245: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
    mul_656: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_245, 1.0025575447570332);  squeeze_245 = None
    mul_657: "f32[736]" = torch.ops.aten.mul.Tensor(mul_656, 0.1);  mul_656 = None
    mul_658: "f32[736]" = torch.ops.aten.mul.Tensor(primals_582, 0.9)
    add_516: "f32[736]" = torch.ops.aten.add.Tensor(mul_657, mul_658);  mul_657 = mul_658 = None
    unsqueeze_324: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1)
    unsqueeze_325: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
    mul_659: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(mul_653, unsqueeze_325);  mul_653 = unsqueeze_325 = None
    unsqueeze_326: "f32[736, 1]" = torch.ops.aten.unsqueeze.default(primals_164, -1);  primals_164 = None
    unsqueeze_327: "f32[736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
    add_517: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_659, unsqueeze_327);  mul_659 = unsqueeze_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_70: "f32[8, 736, 7, 7]" = torch.ops.aten.clone.default(add_517)
    add_518: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(add_517, 3)
    clamp_min_86: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_min.default(add_518, 0);  add_518 = None
    clamp_max_86: "f32[8, 736, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_86, 6);  clamp_min_86 = None
    mul_660: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_517, clamp_max_86);  add_517 = clamp_max_86 = None
    div_86: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(mul_660, 6);  mul_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_16: "f32[8, 736, 1, 1]" = torch.ops.aten.mean.dim(div_86, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_114: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_16, primals_323, primals_324, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_71: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_114)
    add_519: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_114, 3)
    clamp_min_87: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_519, 0);  add_519 = None
    clamp_max_87: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_87, 6);  clamp_min_87 = None
    mul_661: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_114, clamp_max_87);  convolution_114 = clamp_max_87 = None
    div_87: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_661, 6);  mul_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_115: "f32[8, 736, 1, 1]" = torch.ops.aten.convolution.default(div_87, primals_325, primals_326, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_520: "f32[8, 736, 1, 1]" = torch.ops.aten.add.Tensor(convolution_115, 3)
    clamp_min_88: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_min.default(add_520, 0);  add_520 = None
    clamp_max_88: "f32[8, 736, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_88, 6);  clamp_min_88 = None
    div_88: "f32[8, 736, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_88, 6);  clamp_max_88 = None
    mul_662: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(div_86, div_88)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_116: "f32[8, 184, 7, 7]" = torch.ops.aten.convolution.default(mul_662, primals_327, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_521: "i64[]" = torch.ops.aten.add.Tensor(primals_583, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_82 = torch.ops.aten.var_mean.correction(convolution_116, [0, 2, 3], correction = 0, keepdim = True)
    getitem_164: "f32[1, 184, 1, 1]" = var_mean_82[0]
    getitem_165: "f32[1, 184, 1, 1]" = var_mean_82[1];  var_mean_82 = None
    add_522: "f32[1, 184, 1, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-05)
    rsqrt_82: "f32[1, 184, 1, 1]" = torch.ops.aten.rsqrt.default(add_522);  add_522 = None
    sub_82: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_116, getitem_165)
    mul_663: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
    squeeze_246: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
    squeeze_247: "f32[184]" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
    mul_664: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_246, 0.1)
    mul_665: "f32[184]" = torch.ops.aten.mul.Tensor(primals_584, 0.9)
    add_523: "f32[184]" = torch.ops.aten.add.Tensor(mul_664, mul_665);  mul_664 = mul_665 = None
    squeeze_248: "f32[184]" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
    mul_666: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_248, 1.0025575447570332);  squeeze_248 = None
    mul_667: "f32[184]" = torch.ops.aten.mul.Tensor(mul_666, 0.1);  mul_666 = None
    mul_668: "f32[184]" = torch.ops.aten.mul.Tensor(primals_585, 0.9)
    add_524: "f32[184]" = torch.ops.aten.add.Tensor(mul_667, mul_668);  mul_667 = mul_668 = None
    unsqueeze_328: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_165, -1)
    unsqueeze_329: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
    mul_669: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(mul_663, unsqueeze_329);  mul_663 = unsqueeze_329 = None
    unsqueeze_330: "f32[184, 1]" = torch.ops.aten.unsqueeze.default(primals_166, -1);  primals_166 = None
    unsqueeze_331: "f32[184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
    add_525: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(mul_669, unsqueeze_331);  mul_669 = unsqueeze_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    add_526: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_525, add_506);  add_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_117: "f32[8, 1104, 7, 7]" = torch.ops.aten.convolution.default(add_526, primals_328, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_527: "i64[]" = torch.ops.aten.add.Tensor(primals_586, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_83 = torch.ops.aten.var_mean.correction(convolution_117, [0, 2, 3], correction = 0, keepdim = True)
    getitem_166: "f32[1, 1104, 1, 1]" = var_mean_83[0]
    getitem_167: "f32[1, 1104, 1, 1]" = var_mean_83[1];  var_mean_83 = None
    add_528: "f32[1, 1104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05)
    rsqrt_83: "f32[1, 1104, 1, 1]" = torch.ops.aten.rsqrt.default(add_528);  add_528 = None
    sub_83: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_117, getitem_167)
    mul_670: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
    squeeze_249: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
    squeeze_250: "f32[1104]" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
    mul_671: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_249, 0.1)
    mul_672: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_587, 0.9)
    add_529: "f32[1104]" = torch.ops.aten.add.Tensor(mul_671, mul_672);  mul_671 = mul_672 = None
    squeeze_251: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
    mul_673: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_251, 1.0025575447570332);  squeeze_251 = None
    mul_674: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_673, 0.1);  mul_673 = None
    mul_675: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_588, 0.9)
    add_530: "f32[1104]" = torch.ops.aten.add.Tensor(mul_674, mul_675);  mul_674 = mul_675 = None
    unsqueeze_332: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1)
    unsqueeze_333: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
    mul_676: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(mul_670, unsqueeze_333);  mul_670 = unsqueeze_333 = None
    unsqueeze_334: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_168, -1);  primals_168 = None
    unsqueeze_335: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
    add_531: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(mul_676, unsqueeze_335);  mul_676 = unsqueeze_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_72: "f32[8, 1104, 7, 7]" = torch.ops.aten.clone.default(add_531)
    add_532: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(add_531, 3)
    clamp_min_89: "f32[8, 1104, 7, 7]" = torch.ops.aten.clamp_min.default(add_532, 0);  add_532 = None
    clamp_max_89: "f32[8, 1104, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_89, 6);  clamp_min_89 = None
    mul_677: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(add_531, clamp_max_89);  add_531 = clamp_max_89 = None
    div_89: "f32[8, 1104, 7, 7]" = torch.ops.aten.div.Tensor(mul_677, 6);  mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_118: "f32[8, 1104, 7, 7]" = torch.ops.aten.convolution.default(div_89, primals_329, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_533: "i64[]" = torch.ops.aten.add.Tensor(primals_589, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_84 = torch.ops.aten.var_mean.correction(convolution_118, [0, 2, 3], correction = 0, keepdim = True)
    getitem_168: "f32[1, 1104, 1, 1]" = var_mean_84[0]
    getitem_169: "f32[1, 1104, 1, 1]" = var_mean_84[1];  var_mean_84 = None
    add_534: "f32[1, 1104, 1, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-05)
    rsqrt_84: "f32[1, 1104, 1, 1]" = torch.ops.aten.rsqrt.default(add_534);  add_534 = None
    sub_84: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_118, getitem_169)
    mul_678: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
    squeeze_252: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
    squeeze_253: "f32[1104]" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
    mul_679: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_252, 0.1)
    mul_680: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_590, 0.9)
    add_535: "f32[1104]" = torch.ops.aten.add.Tensor(mul_679, mul_680);  mul_679 = mul_680 = None
    squeeze_254: "f32[1104]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
    mul_681: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_254, 1.0025575447570332);  squeeze_254 = None
    mul_682: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_681, 0.1);  mul_681 = None
    mul_683: "f32[1104]" = torch.ops.aten.mul.Tensor(primals_591, 0.9)
    add_536: "f32[1104]" = torch.ops.aten.add.Tensor(mul_682, mul_683);  mul_682 = mul_683 = None
    unsqueeze_336: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_169, -1)
    unsqueeze_337: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
    mul_684: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(mul_678, unsqueeze_337);  mul_678 = unsqueeze_337 = None
    unsqueeze_338: "f32[1104, 1]" = torch.ops.aten.unsqueeze.default(primals_170, -1);  primals_170 = None
    unsqueeze_339: "f32[1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
    add_537: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(mul_684, unsqueeze_339);  mul_684 = unsqueeze_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_73: "f32[8, 1104, 7, 7]" = torch.ops.aten.clone.default(add_537)
    add_538: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(add_537, 3)
    clamp_min_90: "f32[8, 1104, 7, 7]" = torch.ops.aten.clamp_min.default(add_538, 0);  add_538 = None
    clamp_max_90: "f32[8, 1104, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_90, 6);  clamp_min_90 = None
    mul_685: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(add_537, clamp_max_90);  add_537 = clamp_max_90 = None
    div_90: "f32[8, 1104, 7, 7]" = torch.ops.aten.div.Tensor(mul_685, 6);  mul_685 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    mean_17: "f32[8, 1104, 1, 1]" = torch.ops.aten.mean.dim(div_90, [2, 3], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_119: "f32[8, 48, 1, 1]" = torch.ops.aten.convolution.default(mean_17, primals_330, primals_331, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    clone_74: "f32[8, 48, 1, 1]" = torch.ops.aten.clone.default(convolution_119)
    add_539: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(convolution_119, 3)
    clamp_min_91: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_min.default(add_539, 0);  add_539 = None
    clamp_max_91: "f32[8, 48, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_91, 6);  clamp_min_91 = None
    mul_686: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_119, clamp_max_91);  convolution_119 = clamp_max_91 = None
    div_91: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(mul_686, 6);  mul_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_120: "f32[8, 1104, 1, 1]" = torch.ops.aten.convolution.default(div_91, primals_332, primals_333, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    add_540: "f32[8, 1104, 1, 1]" = torch.ops.aten.add.Tensor(convolution_120, 3)
    clamp_min_92: "f32[8, 1104, 1, 1]" = torch.ops.aten.clamp_min.default(add_540, 0);  add_540 = None
    clamp_max_92: "f32[8, 1104, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_92, 6);  clamp_min_92 = None
    div_92: "f32[8, 1104, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_92, 6);  clamp_max_92 = None
    mul_687: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(div_90, div_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_121: "f32[8, 224, 7, 7]" = torch.ops.aten.convolution.default(mul_687, primals_334, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_541: "i64[]" = torch.ops.aten.add.Tensor(primals_592, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_85 = torch.ops.aten.var_mean.correction(convolution_121, [0, 2, 3], correction = 0, keepdim = True)
    getitem_170: "f32[1, 224, 1, 1]" = var_mean_85[0]
    getitem_171: "f32[1, 224, 1, 1]" = var_mean_85[1];  var_mean_85 = None
    add_542: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-05)
    rsqrt_85: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_542);  add_542 = None
    sub_85: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_121, getitem_171)
    mul_688: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = None
    squeeze_255: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
    squeeze_256: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
    mul_689: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_255, 0.1)
    mul_690: "f32[224]" = torch.ops.aten.mul.Tensor(primals_593, 0.9)
    add_543: "f32[224]" = torch.ops.aten.add.Tensor(mul_689, mul_690);  mul_689 = mul_690 = None
    squeeze_257: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_170, [0, 2, 3]);  getitem_170 = None
    mul_691: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_257, 1.0025575447570332);  squeeze_257 = None
    mul_692: "f32[224]" = torch.ops.aten.mul.Tensor(mul_691, 0.1);  mul_691 = None
    mul_693: "f32[224]" = torch.ops.aten.mul.Tensor(primals_594, 0.9)
    add_544: "f32[224]" = torch.ops.aten.add.Tensor(mul_692, mul_693);  mul_692 = mul_693 = None
    unsqueeze_340: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_171, -1)
    unsqueeze_341: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
    mul_694: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_688, unsqueeze_341);  mul_688 = unsqueeze_341 = None
    unsqueeze_342: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_172, -1);  primals_172 = None
    unsqueeze_343: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
    add_545: "f32[8, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_694, unsqueeze_343);  mul_694 = unsqueeze_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:82, code: x = self.conv(x)
    convolution_122: "f32[8, 1344, 7, 7]" = torch.ops.aten.convolution.default(add_545, primals_335, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:98, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
    add_546: "i64[]" = torch.ops.aten.add.Tensor(primals_595, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    var_mean_86 = torch.ops.aten.var_mean.correction(convolution_122, [0, 2, 3], correction = 0, keepdim = True)
    getitem_172: "f32[1, 1344, 1, 1]" = var_mean_86[0]
    getitem_173: "f32[1, 1344, 1, 1]" = var_mean_86[1];  var_mean_86 = None
    add_547: "f32[1, 1344, 1, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05)
    rsqrt_86: "f32[1, 1344, 1, 1]" = torch.ops.aten.rsqrt.default(add_547);  add_547 = None
    sub_86: "f32[8, 1344, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_122, getitem_173)
    mul_695: "f32[8, 1344, 7, 7]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = None
    squeeze_258: "f32[1344]" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3]);  getitem_173 = None
    squeeze_259: "f32[1344]" = torch.ops.aten.squeeze.dims(rsqrt_86, [0, 2, 3]);  rsqrt_86 = None
    mul_696: "f32[1344]" = torch.ops.aten.mul.Tensor(squeeze_258, 0.1)
    mul_697: "f32[1344]" = torch.ops.aten.mul.Tensor(primals_596, 0.9)
    add_548: "f32[1344]" = torch.ops.aten.add.Tensor(mul_696, mul_697);  mul_696 = mul_697 = None
    squeeze_260: "f32[1344]" = torch.ops.aten.squeeze.dims(getitem_172, [0, 2, 3]);  getitem_172 = None
    mul_698: "f32[1344]" = torch.ops.aten.mul.Tensor(squeeze_260, 1.0025575447570332);  squeeze_260 = None
    mul_699: "f32[1344]" = torch.ops.aten.mul.Tensor(mul_698, 0.1);  mul_698 = None
    mul_700: "f32[1344]" = torch.ops.aten.mul.Tensor(primals_597, 0.9)
    add_549: "f32[1344]" = torch.ops.aten.add.Tensor(mul_699, mul_700);  mul_699 = mul_700 = None
    unsqueeze_344: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(primals_173, -1)
    unsqueeze_345: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
    mul_701: "f32[8, 1344, 7, 7]" = torch.ops.aten.mul.Tensor(mul_695, unsqueeze_345);  mul_695 = unsqueeze_345 = None
    unsqueeze_346: "f32[1344, 1]" = torch.ops.aten.unsqueeze.default(primals_174, -1);  primals_174 = None
    unsqueeze_347: "f32[1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
    add_550: "f32[8, 1344, 7, 7]" = torch.ops.aten.add.Tensor(mul_701, unsqueeze_347);  mul_701 = unsqueeze_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    clone_75: "f32[8, 1344, 7, 7]" = torch.ops.aten.clone.default(add_550)
    add_551: "f32[8, 1344, 7, 7]" = torch.ops.aten.add.Tensor(add_550, 3)
    clamp_min_93: "f32[8, 1344, 7, 7]" = torch.ops.aten.clamp_min.default(add_551, 0);  add_551 = None
    clamp_max_93: "f32[8, 1344, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_93, 6);  clamp_min_93 = None
    mul_702: "f32[8, 1344, 7, 7]" = torch.ops.aten.mul.Tensor(add_550, clamp_max_93);  add_550 = clamp_max_93 = None
    div_93: "f32[8, 1344, 7, 7]" = torch.ops.aten.div.Tensor(mul_702, 6);  mul_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    mean_18: "f32[8, 1344, 1, 1]" = torch.ops.aten.mean.dim(div_93, [-1, -2], True);  div_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:145, code: x = self.conv_head(x)
    convolution_123: "f32[8, 1984, 1, 1]" = torch.ops.aten.convolution.default(mean_18, primals_336, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:146, code: x = self.act2(x)
    clone_76: "f32[8, 1984, 1, 1]" = torch.ops.aten.clone.default(convolution_123)
    add_552: "f32[8, 1984, 1, 1]" = torch.ops.aten.add.Tensor(convolution_123, 3)
    clamp_min_94: "f32[8, 1984, 1, 1]" = torch.ops.aten.clamp_min.default(add_552, 0);  add_552 = None
    clamp_max_94: "f32[8, 1984, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_94, 6);  clamp_min_94 = None
    mul_703: "f32[8, 1984, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_123, clamp_max_94);  convolution_123 = clamp_max_94 = None
    div_94: "f32[8, 1984, 1, 1]" = torch.ops.aten.div.Tensor(mul_703, 6);  mul_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/linear.py:19, code: return F.linear(input, self.weight, self.bias)
    permute: "f32[1984, 1000]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    view_1: "f32[8, 1984]" = torch.ops.aten.view.default(div_94, [8, 1984]);  div_94 = None
    addmm: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_176, view_1, permute);  primals_176 = None
    permute_1: "f32[1000, 1984]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm: "f32[8, 1984]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1984]" = torch.ops.aten.mm.default(permute_2, view_1);  permute_2 = view_1 = None
    permute_3: "f32[1984, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_2: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1984]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:147, code: x = self.flatten(x)
    view_3: "f32[8, 1984, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1984, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:146, code: x = self.act2(x)
    lt: "b8[8, 1984, 1, 1]" = torch.ops.aten.lt.Scalar(clone_76, -3)
    le: "b8[8, 1984, 1, 1]" = torch.ops.aten.le.Scalar(clone_76, 3)
    div_95: "f32[8, 1984, 1, 1]" = torch.ops.aten.div.Tensor(clone_76, 3);  clone_76 = None
    add_553: "f32[8, 1984, 1, 1]" = torch.ops.aten.add.Tensor(div_95, 0.5);  div_95 = None
    mul_704: "f32[8, 1984, 1, 1]" = torch.ops.aten.mul.Tensor(view_3, add_553);  add_553 = None
    where: "f32[8, 1984, 1, 1]" = torch.ops.aten.where.self(le, mul_704, view_3);  le = mul_704 = view_3 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[8, 1984, 1, 1]" = torch.ops.aten.where.self(lt, scalar_tensor, where);  lt = scalar_tensor = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:145, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(where_1, mean_18, primals_336, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = mean_18 = primals_336 = None
    getitem_174: "f32[8, 1344, 1, 1]" = convolution_backward[0]
    getitem_175: "f32[1984, 1344, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1344, 7, 7]" = torch.ops.aten.expand.default(getitem_174, [8, 1344, 7, 7]);  getitem_174 = None
    div_96: "f32[8, 1344, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_1: "b8[8, 1344, 7, 7]" = torch.ops.aten.lt.Scalar(clone_75, -3)
    le_1: "b8[8, 1344, 7, 7]" = torch.ops.aten.le.Scalar(clone_75, 3)
    div_97: "f32[8, 1344, 7, 7]" = torch.ops.aten.div.Tensor(clone_75, 3);  clone_75 = None
    add_554: "f32[8, 1344, 7, 7]" = torch.ops.aten.add.Tensor(div_97, 0.5);  div_97 = None
    mul_705: "f32[8, 1344, 7, 7]" = torch.ops.aten.mul.Tensor(div_96, add_554);  add_554 = None
    where_2: "f32[8, 1344, 7, 7]" = torch.ops.aten.where.self(le_1, mul_705, div_96);  le_1 = mul_705 = div_96 = None
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[8, 1344, 7, 7]" = torch.ops.aten.where.self(lt_1, scalar_tensor_1, where_2);  lt_1 = scalar_tensor_1 = where_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_348: "f32[1, 1344]" = torch.ops.aten.unsqueeze.default(squeeze_258, 0);  squeeze_258 = None
    unsqueeze_349: "f32[1, 1344, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 2);  unsqueeze_348 = None
    unsqueeze_350: "f32[1, 1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 3);  unsqueeze_349 = None
    sum_2: "f32[1344]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_87: "f32[8, 1344, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_350)
    mul_706: "f32[8, 1344, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_87);  sub_87 = None
    sum_3: "f32[1344]" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 2, 3]);  mul_706 = None
    mul_707: "f32[1344]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_351: "f32[1, 1344]" = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
    unsqueeze_352: "f32[1, 1344, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
    unsqueeze_353: "f32[1, 1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 3);  unsqueeze_352 = None
    mul_708: "f32[1344]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_709: "f32[1344]" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
    mul_710: "f32[1344]" = torch.ops.aten.mul.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
    unsqueeze_354: "f32[1, 1344]" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
    unsqueeze_355: "f32[1, 1344, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 2);  unsqueeze_354 = None
    unsqueeze_356: "f32[1, 1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 3);  unsqueeze_355 = None
    mul_711: "f32[1344]" = torch.ops.aten.mul.Tensor(squeeze_259, primals_173);  primals_173 = None
    unsqueeze_357: "f32[1, 1344]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    unsqueeze_358: "f32[1, 1344, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    unsqueeze_359: "f32[1, 1344, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
    sub_88: "f32[8, 1344, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_350);  convolution_122 = unsqueeze_350 = None
    mul_712: "f32[8, 1344, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_356);  sub_88 = unsqueeze_356 = None
    sub_89: "f32[8, 1344, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_712);  where_3 = mul_712 = None
    sub_90: "f32[8, 1344, 7, 7]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_353);  sub_89 = unsqueeze_353 = None
    mul_713: "f32[8, 1344, 7, 7]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_359);  sub_90 = unsqueeze_359 = None
    mul_714: "f32[1344]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_259);  sum_3 = squeeze_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:82, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_713, add_545, primals_335, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_713 = add_545 = primals_335 = None
    getitem_177: "f32[8, 224, 7, 7]" = convolution_backward_1[0]
    getitem_178: "f32[1344, 224, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_360: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_255, 0);  squeeze_255 = None
    unsqueeze_361: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 2);  unsqueeze_360 = None
    unsqueeze_362: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 3);  unsqueeze_361 = None
    sum_4: "f32[224]" = torch.ops.aten.sum.dim_IntList(getitem_177, [0, 2, 3])
    sub_91: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_362)
    mul_715: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_177, sub_91);  sub_91 = None
    sum_5: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_715, [0, 2, 3]);  mul_715 = None
    mul_716: "f32[224]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_363: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
    unsqueeze_364: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
    unsqueeze_365: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 3);  unsqueeze_364 = None
    mul_717: "f32[224]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_718: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
    mul_719: "f32[224]" = torch.ops.aten.mul.Tensor(mul_717, mul_718);  mul_717 = mul_718 = None
    unsqueeze_366: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
    unsqueeze_367: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 2);  unsqueeze_366 = None
    unsqueeze_368: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 3);  unsqueeze_367 = None
    mul_720: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_256, primals_171);  primals_171 = None
    unsqueeze_369: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    unsqueeze_370: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
    unsqueeze_371: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 3);  unsqueeze_370 = None
    sub_92: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_362);  convolution_121 = unsqueeze_362 = None
    mul_721: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_368);  sub_92 = unsqueeze_368 = None
    sub_93: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_177, mul_721);  getitem_177 = mul_721 = None
    sub_94: "f32[8, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_365);  sub_93 = unsqueeze_365 = None
    mul_722: "f32[8, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_371);  sub_94 = unsqueeze_371 = None
    mul_723: "f32[224]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_256);  sum_5 = squeeze_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_722, mul_687, primals_334, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_722 = mul_687 = primals_334 = None
    getitem_180: "f32[8, 1104, 7, 7]" = convolution_backward_2[0]
    getitem_181: "f32[224, 1104, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_724: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_180, div_90);  div_90 = None
    mul_725: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_180, div_92);  getitem_180 = div_92 = None
    sum_6: "f32[8, 1104, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_724, [2, 3], True);  mul_724 = None
    gt: "b8[8, 1104, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_120, -3.0)
    lt_2: "b8[8, 1104, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_120, 3.0);  convolution_120 = None
    bitwise_and: "b8[8, 1104, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt, lt_2);  gt = lt_2 = None
    mul_726: "f32[8, 1104, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, 0.16666666666666666);  sum_6 = None
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[8, 1104, 1, 1]" = torch.ops.aten.where.self(bitwise_and, mul_726, scalar_tensor_2);  bitwise_and = mul_726 = scalar_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_4, div_91, primals_332, [1104], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_4 = div_91 = primals_332 = None
    getitem_183: "f32[8, 48, 1, 1]" = convolution_backward_3[0]
    getitem_184: "f32[1104, 48, 1, 1]" = convolution_backward_3[1]
    getitem_185: "f32[1104]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_3: "b8[8, 48, 1, 1]" = torch.ops.aten.lt.Scalar(clone_74, -3)
    le_2: "b8[8, 48, 1, 1]" = torch.ops.aten.le.Scalar(clone_74, 3)
    div_98: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(clone_74, 3);  clone_74 = None
    add_555: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(div_98, 0.5);  div_98 = None
    mul_727: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_183, add_555);  add_555 = None
    where_5: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(le_2, mul_727, getitem_183);  le_2 = mul_727 = getitem_183 = None
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(lt_3, scalar_tensor_3, where_5);  lt_3 = scalar_tensor_3 = where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(where_6, mean_17, primals_330, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_6 = mean_17 = primals_330 = None
    getitem_186: "f32[8, 1104, 1, 1]" = convolution_backward_4[0]
    getitem_187: "f32[48, 1104, 1, 1]" = convolution_backward_4[1]
    getitem_188: "f32[48]" = convolution_backward_4[2];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 1104, 7, 7]" = torch.ops.aten.expand.default(getitem_186, [8, 1104, 7, 7]);  getitem_186 = None
    div_99: "f32[8, 1104, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_556: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(mul_725, div_99);  mul_725 = div_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_4: "b8[8, 1104, 7, 7]" = torch.ops.aten.lt.Scalar(clone_73, -3)
    le_3: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(clone_73, 3)
    div_100: "f32[8, 1104, 7, 7]" = torch.ops.aten.div.Tensor(clone_73, 3);  clone_73 = None
    add_557: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(div_100, 0.5);  div_100 = None
    mul_728: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(add_556, add_557);  add_557 = None
    where_7: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_3, mul_728, add_556);  le_3 = mul_728 = add_556 = None
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_8: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(lt_4, scalar_tensor_4, where_7);  lt_4 = scalar_tensor_4 = where_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_372: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
    unsqueeze_373: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 2);  unsqueeze_372 = None
    unsqueeze_374: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 3);  unsqueeze_373 = None
    sum_7: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_95: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_374)
    mul_729: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_95);  sub_95 = None
    sum_8: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_729, [0, 2, 3]);  mul_729 = None
    mul_730: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    unsqueeze_375: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_376: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
    unsqueeze_377: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 3);  unsqueeze_376 = None
    mul_731: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    mul_732: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
    mul_733: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_731, mul_732);  mul_731 = mul_732 = None
    unsqueeze_378: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_733, 0);  mul_733 = None
    unsqueeze_379: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 2);  unsqueeze_378 = None
    unsqueeze_380: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 3);  unsqueeze_379 = None
    mul_734: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_253, primals_169);  primals_169 = None
    unsqueeze_381: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
    unsqueeze_382: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
    unsqueeze_383: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 3);  unsqueeze_382 = None
    sub_96: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_374);  convolution_118 = unsqueeze_374 = None
    mul_735: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_380);  sub_96 = unsqueeze_380 = None
    sub_97: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_735);  where_8 = mul_735 = None
    sub_98: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_377);  sub_97 = unsqueeze_377 = None
    mul_736: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_383);  sub_98 = unsqueeze_383 = None
    mul_737: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_253);  sum_8 = squeeze_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_736, div_89, primals_329, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1104, [True, True, False]);  mul_736 = div_89 = primals_329 = None
    getitem_189: "f32[8, 1104, 7, 7]" = convolution_backward_5[0]
    getitem_190: "f32[1104, 1, 5, 5]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_5: "b8[8, 1104, 7, 7]" = torch.ops.aten.lt.Scalar(clone_72, -3)
    le_4: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(clone_72, 3)
    div_101: "f32[8, 1104, 7, 7]" = torch.ops.aten.div.Tensor(clone_72, 3);  clone_72 = None
    add_558: "f32[8, 1104, 7, 7]" = torch.ops.aten.add.Tensor(div_101, 0.5);  div_101 = None
    mul_738: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_189, add_558);  add_558 = None
    where_9: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_4, mul_738, getitem_189);  le_4 = mul_738 = getitem_189 = None
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(lt_5, scalar_tensor_5, where_9);  lt_5 = scalar_tensor_5 = where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_384: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
    unsqueeze_385: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 2);  unsqueeze_384 = None
    unsqueeze_386: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 3);  unsqueeze_385 = None
    sum_9: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_99: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_386)
    mul_739: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_99);  sub_99 = None
    sum_10: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_739, [0, 2, 3]);  mul_739 = None
    mul_740: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    unsqueeze_387: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    unsqueeze_388: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
    unsqueeze_389: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 3);  unsqueeze_388 = None
    mul_741: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    mul_742: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
    mul_743: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_741, mul_742);  mul_741 = mul_742 = None
    unsqueeze_390: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_391: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 2);  unsqueeze_390 = None
    unsqueeze_392: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 3);  unsqueeze_391 = None
    mul_744: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_250, primals_167);  primals_167 = None
    unsqueeze_393: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
    unsqueeze_394: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
    unsqueeze_395: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
    sub_100: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_386);  convolution_117 = unsqueeze_386 = None
    mul_745: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_392);  sub_100 = unsqueeze_392 = None
    sub_101: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_10, mul_745);  where_10 = mul_745 = None
    sub_102: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_389);  sub_101 = unsqueeze_389 = None
    mul_746: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_395);  sub_102 = unsqueeze_395 = None
    mul_747: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_250);  sum_10 = squeeze_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_746, add_526, primals_328, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_746 = add_526 = primals_328 = None
    getitem_192: "f32[8, 184, 7, 7]" = convolution_backward_6[0]
    getitem_193: "f32[1104, 184, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_396: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
    unsqueeze_397: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
    unsqueeze_398: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
    sum_11: "f32[184]" = torch.ops.aten.sum.dim_IntList(getitem_192, [0, 2, 3])
    sub_103: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_398)
    mul_748: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_192, sub_103);  sub_103 = None
    sum_12: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_748, [0, 2, 3]);  mul_748 = None
    mul_749: "f32[184]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    unsqueeze_399: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
    unsqueeze_400: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    unsqueeze_401: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
    mul_750: "f32[184]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    mul_751: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
    mul_752: "f32[184]" = torch.ops.aten.mul.Tensor(mul_750, mul_751);  mul_750 = mul_751 = None
    unsqueeze_402: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
    unsqueeze_403: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
    unsqueeze_404: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
    mul_753: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_247, primals_165);  primals_165 = None
    unsqueeze_405: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
    unsqueeze_406: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    unsqueeze_407: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
    sub_104: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_398);  convolution_116 = unsqueeze_398 = None
    mul_754: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_404);  sub_104 = unsqueeze_404 = None
    sub_105: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_192, mul_754);  mul_754 = None
    sub_106: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_401);  sub_105 = unsqueeze_401 = None
    mul_755: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_407);  sub_106 = unsqueeze_407 = None
    mul_756: "f32[184]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_247);  sum_12 = squeeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_755, mul_662, primals_327, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_755 = mul_662 = primals_327 = None
    getitem_195: "f32[8, 736, 7, 7]" = convolution_backward_7[0]
    getitem_196: "f32[184, 736, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_757: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_195, div_86);  div_86 = None
    mul_758: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_195, div_88);  getitem_195 = div_88 = None
    sum_13: "f32[8, 736, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_757, [2, 3], True);  mul_757 = None
    gt_1: "b8[8, 736, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_115, -3.0)
    lt_6: "b8[8, 736, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_115, 3.0);  convolution_115 = None
    bitwise_and_1: "b8[8, 736, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_1, lt_6);  gt_1 = lt_6 = None
    mul_759: "f32[8, 736, 1, 1]" = torch.ops.aten.mul.Tensor(sum_13, 0.16666666666666666);  sum_13 = None
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_11: "f32[8, 736, 1, 1]" = torch.ops.aten.where.self(bitwise_and_1, mul_759, scalar_tensor_6);  bitwise_and_1 = mul_759 = scalar_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_11, div_87, primals_325, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_11 = div_87 = primals_325 = None
    getitem_198: "f32[8, 48, 1, 1]" = convolution_backward_8[0]
    getitem_199: "f32[736, 48, 1, 1]" = convolution_backward_8[1]
    getitem_200: "f32[736]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_7: "b8[8, 48, 1, 1]" = torch.ops.aten.lt.Scalar(clone_71, -3)
    le_5: "b8[8, 48, 1, 1]" = torch.ops.aten.le.Scalar(clone_71, 3)
    div_102: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(clone_71, 3);  clone_71 = None
    add_559: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(div_102, 0.5);  div_102 = None
    mul_760: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_198, add_559);  add_559 = None
    where_12: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(le_5, mul_760, getitem_198);  le_5 = mul_760 = getitem_198 = None
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(lt_7, scalar_tensor_7, where_12);  lt_7 = scalar_tensor_7 = where_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(where_13, mean_16, primals_323, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_13 = mean_16 = primals_323 = None
    getitem_201: "f32[8, 736, 1, 1]" = convolution_backward_9[0]
    getitem_202: "f32[48, 736, 1, 1]" = convolution_backward_9[1]
    getitem_203: "f32[48]" = convolution_backward_9[2];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 736, 7, 7]" = torch.ops.aten.expand.default(getitem_201, [8, 736, 7, 7]);  getitem_201 = None
    div_103: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_560: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_758, div_103);  mul_758 = div_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_8: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_70, -3)
    le_6: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_70, 3)
    div_104: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_70, 3);  clone_70 = None
    add_561: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_104, 0.5);  div_104 = None
    mul_761: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_560, add_561);  add_561 = None
    where_14: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_6, mul_761, add_560);  le_6 = mul_761 = add_560 = None
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_8, scalar_tensor_8, where_14);  lt_8 = scalar_tensor_8 = where_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_408: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
    unsqueeze_409: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
    unsqueeze_410: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
    sum_14: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_107: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_410)
    mul_762: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_15, sub_107);  sub_107 = None
    sum_15: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_762, [0, 2, 3]);  mul_762 = None
    mul_763: "f32[736]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_411: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
    unsqueeze_412: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    unsqueeze_413: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
    mul_764: "f32[736]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_765: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
    mul_766: "f32[736]" = torch.ops.aten.mul.Tensor(mul_764, mul_765);  mul_764 = mul_765 = None
    unsqueeze_414: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
    unsqueeze_415: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
    unsqueeze_416: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
    mul_767: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_244, primals_163);  primals_163 = None
    unsqueeze_417: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
    unsqueeze_418: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    unsqueeze_419: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
    sub_108: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_410);  convolution_113 = unsqueeze_410 = None
    mul_768: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_416);  sub_108 = unsqueeze_416 = None
    sub_109: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_15, mul_768);  where_15 = mul_768 = None
    sub_110: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_413);  sub_109 = unsqueeze_413 = None
    mul_769: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_419);  sub_110 = unsqueeze_419 = None
    mul_770: "f32[736]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_244);  sum_15 = squeeze_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_769, div_85, primals_322, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False]);  mul_769 = div_85 = primals_322 = None
    getitem_204: "f32[8, 736, 7, 7]" = convolution_backward_10[0]
    getitem_205: "f32[736, 1, 5, 5]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_9: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_69, -3)
    le_7: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_69, 3)
    div_105: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_69, 3);  clone_69 = None
    add_562: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_105, 0.5);  div_105 = None
    mul_771: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_204, add_562);  add_562 = None
    where_16: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_7, mul_771, getitem_204);  le_7 = mul_771 = getitem_204 = None
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_9, scalar_tensor_9, where_16);  lt_9 = scalar_tensor_9 = where_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_420: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
    unsqueeze_421: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
    unsqueeze_422: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
    sum_16: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_111: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_422)
    mul_772: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_17, sub_111);  sub_111 = None
    sum_17: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_772, [0, 2, 3]);  mul_772 = None
    mul_773: "f32[736]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_423: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_424: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    unsqueeze_425: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
    mul_774: "f32[736]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_775: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
    mul_776: "f32[736]" = torch.ops.aten.mul.Tensor(mul_774, mul_775);  mul_774 = mul_775 = None
    unsqueeze_426: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_427: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
    unsqueeze_428: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
    mul_777: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_241, primals_161);  primals_161 = None
    unsqueeze_429: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
    unsqueeze_430: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    sub_112: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_422);  convolution_112 = unsqueeze_422 = None
    mul_778: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_428);  sub_112 = unsqueeze_428 = None
    sub_113: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_17, mul_778);  where_17 = mul_778 = None
    sub_114: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_425);  sub_113 = unsqueeze_425 = None
    mul_779: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_431);  sub_114 = unsqueeze_431 = None
    mul_780: "f32[736]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_241);  sum_17 = squeeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_779, add_506, primals_321, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_779 = add_506 = primals_321 = None
    getitem_207: "f32[8, 184, 7, 7]" = convolution_backward_11[0]
    getitem_208: "f32[736, 184, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_563: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(getitem_192, getitem_207);  getitem_192 = getitem_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_432: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
    unsqueeze_433: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
    unsqueeze_434: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
    sum_18: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_563, [0, 2, 3])
    sub_115: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_434)
    mul_781: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_563, sub_115);  sub_115 = None
    sum_19: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_781, [0, 2, 3]);  mul_781 = None
    mul_782: "f32[184]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_435: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_436: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
    unsqueeze_437: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 3);  unsqueeze_436 = None
    mul_783: "f32[184]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_784: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
    mul_785: "f32[184]" = torch.ops.aten.mul.Tensor(mul_783, mul_784);  mul_783 = mul_784 = None
    unsqueeze_438: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
    unsqueeze_439: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
    unsqueeze_440: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
    mul_786: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_238, primals_159);  primals_159 = None
    unsqueeze_441: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
    unsqueeze_442: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
    unsqueeze_443: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
    sub_116: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_434);  convolution_111 = unsqueeze_434 = None
    mul_787: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_440);  sub_116 = unsqueeze_440 = None
    sub_117: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_563, mul_787);  mul_787 = None
    sub_118: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_117, unsqueeze_437);  sub_117 = unsqueeze_437 = None
    mul_788: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_443);  sub_118 = unsqueeze_443 = None
    mul_789: "f32[184]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_238);  sum_19 = squeeze_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_788, mul_637, primals_320, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_788 = mul_637 = primals_320 = None
    getitem_210: "f32[8, 736, 7, 7]" = convolution_backward_12[0]
    getitem_211: "f32[184, 736, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_790: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_210, div_82);  div_82 = None
    mul_791: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_210, div_84);  getitem_210 = div_84 = None
    sum_20: "f32[8, 736, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_790, [2, 3], True);  mul_790 = None
    gt_2: "b8[8, 736, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_110, -3.0)
    lt_10: "b8[8, 736, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_110, 3.0);  convolution_110 = None
    bitwise_and_2: "b8[8, 736, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_2, lt_10);  gt_2 = lt_10 = None
    mul_792: "f32[8, 736, 1, 1]" = torch.ops.aten.mul.Tensor(sum_20, 0.16666666666666666);  sum_20 = None
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_18: "f32[8, 736, 1, 1]" = torch.ops.aten.where.self(bitwise_and_2, mul_792, scalar_tensor_10);  bitwise_and_2 = mul_792 = scalar_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_18, div_83, primals_318, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_18 = div_83 = primals_318 = None
    getitem_213: "f32[8, 48, 1, 1]" = convolution_backward_13[0]
    getitem_214: "f32[736, 48, 1, 1]" = convolution_backward_13[1]
    getitem_215: "f32[736]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_11: "b8[8, 48, 1, 1]" = torch.ops.aten.lt.Scalar(clone_68, -3)
    le_8: "b8[8, 48, 1, 1]" = torch.ops.aten.le.Scalar(clone_68, 3)
    div_106: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(clone_68, 3);  clone_68 = None
    add_564: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(div_106, 0.5);  div_106 = None
    mul_793: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_213, add_564);  add_564 = None
    where_19: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(le_8, mul_793, getitem_213);  le_8 = mul_793 = getitem_213 = None
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_20: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(lt_11, scalar_tensor_11, where_19);  lt_11 = scalar_tensor_11 = where_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_20, mean_15, primals_316, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_20 = mean_15 = primals_316 = None
    getitem_216: "f32[8, 736, 1, 1]" = convolution_backward_14[0]
    getitem_217: "f32[48, 736, 1, 1]" = convolution_backward_14[1]
    getitem_218: "f32[48]" = convolution_backward_14[2];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 736, 7, 7]" = torch.ops.aten.expand.default(getitem_216, [8, 736, 7, 7]);  getitem_216 = None
    div_107: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_565: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_791, div_107);  mul_791 = div_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_12: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_67, -3)
    le_9: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_67, 3)
    div_108: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_67, 3);  clone_67 = None
    add_566: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_108, 0.5);  div_108 = None
    mul_794: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_565, add_566);  add_566 = None
    where_21: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_9, mul_794, add_565);  le_9 = mul_794 = add_565 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_22: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_12, scalar_tensor_12, where_21);  lt_12 = scalar_tensor_12 = where_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_444: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
    unsqueeze_445: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 2);  unsqueeze_444 = None
    unsqueeze_446: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 3);  unsqueeze_445 = None
    sum_21: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_119: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_446)
    mul_795: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_22, sub_119);  sub_119 = None
    sum_22: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_795, [0, 2, 3]);  mul_795 = None
    mul_796: "f32[736]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    unsqueeze_447: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_796, 0);  mul_796 = None
    unsqueeze_448: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
    unsqueeze_449: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 3);  unsqueeze_448 = None
    mul_797: "f32[736]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    mul_798: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
    mul_799: "f32[736]" = torch.ops.aten.mul.Tensor(mul_797, mul_798);  mul_797 = mul_798 = None
    unsqueeze_450: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
    unsqueeze_451: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 2);  unsqueeze_450 = None
    unsqueeze_452: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 3);  unsqueeze_451 = None
    mul_800: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_235, primals_157);  primals_157 = None
    unsqueeze_453: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_454: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
    unsqueeze_455: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
    sub_120: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_446);  convolution_108 = unsqueeze_446 = None
    mul_801: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_452);  sub_120 = unsqueeze_452 = None
    sub_121: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_22, mul_801);  where_22 = mul_801 = None
    sub_122: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_121, unsqueeze_449);  sub_121 = unsqueeze_449 = None
    mul_802: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_455);  sub_122 = unsqueeze_455 = None
    mul_803: "f32[736]" = torch.ops.aten.mul.Tensor(sum_22, squeeze_235);  sum_22 = squeeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_802, div_81, primals_315, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False]);  mul_802 = div_81 = primals_315 = None
    getitem_219: "f32[8, 736, 7, 7]" = convolution_backward_15[0]
    getitem_220: "f32[736, 1, 5, 5]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_13: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_66, -3)
    le_10: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_66, 3)
    div_109: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_66, 3);  clone_66 = None
    add_567: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_109, 0.5);  div_109 = None
    mul_804: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_219, add_567);  add_567 = None
    where_23: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_10, mul_804, getitem_219);  le_10 = mul_804 = getitem_219 = None
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_24: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_13, scalar_tensor_13, where_23);  lt_13 = scalar_tensor_13 = where_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_456: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
    unsqueeze_457: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 2);  unsqueeze_456 = None
    unsqueeze_458: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 3);  unsqueeze_457 = None
    sum_23: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_123: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_458)
    mul_805: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_24, sub_123);  sub_123 = None
    sum_24: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_805, [0, 2, 3]);  mul_805 = None
    mul_806: "f32[736]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    unsqueeze_459: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
    unsqueeze_460: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
    unsqueeze_461: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 3);  unsqueeze_460 = None
    mul_807: "f32[736]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    mul_808: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
    mul_809: "f32[736]" = torch.ops.aten.mul.Tensor(mul_807, mul_808);  mul_807 = mul_808 = None
    unsqueeze_462: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
    unsqueeze_463: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 2);  unsqueeze_462 = None
    unsqueeze_464: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 3);  unsqueeze_463 = None
    mul_810: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_232, primals_155);  primals_155 = None
    unsqueeze_465: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
    unsqueeze_466: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
    unsqueeze_467: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
    sub_124: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_458);  convolution_107 = unsqueeze_458 = None
    mul_811: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_464);  sub_124 = unsqueeze_464 = None
    sub_125: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_24, mul_811);  where_24 = mul_811 = None
    sub_126: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_461);  sub_125 = unsqueeze_461 = None
    mul_812: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_467);  sub_126 = unsqueeze_467 = None
    mul_813: "f32[736]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_232);  sum_24 = squeeze_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_812, add_486, primals_314, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_812 = add_486 = primals_314 = None
    getitem_222: "f32[8, 184, 7, 7]" = convolution_backward_16[0]
    getitem_223: "f32[736, 184, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_568: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_563, getitem_222);  add_563 = getitem_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_468: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
    unsqueeze_469: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 2);  unsqueeze_468 = None
    unsqueeze_470: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 3);  unsqueeze_469 = None
    sum_25: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_568, [0, 2, 3])
    sub_127: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_470)
    mul_814: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_568, sub_127);  sub_127 = None
    sum_26: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3]);  mul_814 = None
    mul_815: "f32[184]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    unsqueeze_471: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
    unsqueeze_472: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
    unsqueeze_473: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 3);  unsqueeze_472 = None
    mul_816: "f32[184]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    mul_817: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
    mul_818: "f32[184]" = torch.ops.aten.mul.Tensor(mul_816, mul_817);  mul_816 = mul_817 = None
    unsqueeze_474: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    unsqueeze_475: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 2);  unsqueeze_474 = None
    unsqueeze_476: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 3);  unsqueeze_475 = None
    mul_819: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_229, primals_153);  primals_153 = None
    unsqueeze_477: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
    unsqueeze_478: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
    unsqueeze_479: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 3);  unsqueeze_478 = None
    sub_128: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_470);  convolution_106 = unsqueeze_470 = None
    mul_820: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_476);  sub_128 = unsqueeze_476 = None
    sub_129: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_568, mul_820);  mul_820 = None
    sub_130: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_473);  sub_129 = unsqueeze_473 = None
    mul_821: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_479);  sub_130 = unsqueeze_479 = None
    mul_822: "f32[184]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_229);  sum_26 = squeeze_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_821, mul_612, primals_313, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_821 = mul_612 = primals_313 = None
    getitem_225: "f32[8, 736, 7, 7]" = convolution_backward_17[0]
    getitem_226: "f32[184, 736, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_823: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_225, div_78);  div_78 = None
    mul_824: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_225, div_80);  getitem_225 = div_80 = None
    sum_27: "f32[8, 736, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_823, [2, 3], True);  mul_823 = None
    gt_3: "b8[8, 736, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_105, -3.0)
    lt_14: "b8[8, 736, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_105, 3.0);  convolution_105 = None
    bitwise_and_3: "b8[8, 736, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_3, lt_14);  gt_3 = lt_14 = None
    mul_825: "f32[8, 736, 1, 1]" = torch.ops.aten.mul.Tensor(sum_27, 0.16666666666666666);  sum_27 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_25: "f32[8, 736, 1, 1]" = torch.ops.aten.where.self(bitwise_and_3, mul_825, scalar_tensor_14);  bitwise_and_3 = mul_825 = scalar_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_25, div_79, primals_311, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_25 = div_79 = primals_311 = None
    getitem_228: "f32[8, 48, 1, 1]" = convolution_backward_18[0]
    getitem_229: "f32[736, 48, 1, 1]" = convolution_backward_18[1]
    getitem_230: "f32[736]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_15: "b8[8, 48, 1, 1]" = torch.ops.aten.lt.Scalar(clone_65, -3)
    le_11: "b8[8, 48, 1, 1]" = torch.ops.aten.le.Scalar(clone_65, 3)
    div_110: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(clone_65, 3);  clone_65 = None
    add_569: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(div_110, 0.5);  div_110 = None
    mul_826: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_228, add_569);  add_569 = None
    where_26: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(le_11, mul_826, getitem_228);  le_11 = mul_826 = getitem_228 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_27: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(lt_15, scalar_tensor_15, where_26);  lt_15 = scalar_tensor_15 = where_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(where_27, mean_14, primals_309, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_27 = mean_14 = primals_309 = None
    getitem_231: "f32[8, 736, 1, 1]" = convolution_backward_19[0]
    getitem_232: "f32[48, 736, 1, 1]" = convolution_backward_19[1]
    getitem_233: "f32[48]" = convolution_backward_19[2];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 736, 7, 7]" = torch.ops.aten.expand.default(getitem_231, [8, 736, 7, 7]);  getitem_231 = None
    div_111: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_570: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_824, div_111);  mul_824 = div_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_16: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_64, -3)
    le_12: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_64, 3)
    div_112: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_64, 3);  clone_64 = None
    add_571: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_112, 0.5);  div_112 = None
    mul_827: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_570, add_571);  add_571 = None
    where_28: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_12, mul_827, add_570);  le_12 = mul_827 = add_570 = None
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_29: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_16, scalar_tensor_16, where_28);  lt_16 = scalar_tensor_16 = where_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_480: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
    unsqueeze_481: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 2);  unsqueeze_480 = None
    unsqueeze_482: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 3);  unsqueeze_481 = None
    sum_28: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_131: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_482)
    mul_828: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_29, sub_131);  sub_131 = None
    sum_29: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_828, [0, 2, 3]);  mul_828 = None
    mul_829: "f32[736]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_483: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
    unsqueeze_484: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
    unsqueeze_485: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 3);  unsqueeze_484 = None
    mul_830: "f32[736]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_831: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
    mul_832: "f32[736]" = torch.ops.aten.mul.Tensor(mul_830, mul_831);  mul_830 = mul_831 = None
    unsqueeze_486: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
    unsqueeze_487: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 2);  unsqueeze_486 = None
    unsqueeze_488: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 3);  unsqueeze_487 = None
    mul_833: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_226, primals_151);  primals_151 = None
    unsqueeze_489: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_833, 0);  mul_833 = None
    unsqueeze_490: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
    unsqueeze_491: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 3);  unsqueeze_490 = None
    sub_132: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_482);  convolution_103 = unsqueeze_482 = None
    mul_834: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_488);  sub_132 = unsqueeze_488 = None
    sub_133: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_29, mul_834);  where_29 = mul_834 = None
    sub_134: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_133, unsqueeze_485);  sub_133 = unsqueeze_485 = None
    mul_835: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_491);  sub_134 = unsqueeze_491 = None
    mul_836: "f32[736]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_226);  sum_29 = squeeze_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_835, div_77, primals_308, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False]);  mul_835 = div_77 = primals_308 = None
    getitem_234: "f32[8, 736, 7, 7]" = convolution_backward_20[0]
    getitem_235: "f32[736, 1, 5, 5]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_17: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_63, -3)
    le_13: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_63, 3)
    div_113: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_63, 3);  clone_63 = None
    add_572: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_113, 0.5);  div_113 = None
    mul_837: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_234, add_572);  add_572 = None
    where_30: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_13, mul_837, getitem_234);  le_13 = mul_837 = getitem_234 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_31: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_17, scalar_tensor_17, where_30);  lt_17 = scalar_tensor_17 = where_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_492: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
    unsqueeze_493: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 2);  unsqueeze_492 = None
    unsqueeze_494: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 3);  unsqueeze_493 = None
    sum_30: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_135: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_494)
    mul_838: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_31, sub_135);  sub_135 = None
    sum_31: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_838, [0, 2, 3]);  mul_838 = None
    mul_839: "f32[736]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_495: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_839, 0);  mul_839 = None
    unsqueeze_496: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
    unsqueeze_497: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 3);  unsqueeze_496 = None
    mul_840: "f32[736]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_841: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
    mul_842: "f32[736]" = torch.ops.aten.mul.Tensor(mul_840, mul_841);  mul_840 = mul_841 = None
    unsqueeze_498: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_842, 0);  mul_842 = None
    unsqueeze_499: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 2);  unsqueeze_498 = None
    unsqueeze_500: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 3);  unsqueeze_499 = None
    mul_843: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_223, primals_149);  primals_149 = None
    unsqueeze_501: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
    unsqueeze_502: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
    unsqueeze_503: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 3);  unsqueeze_502 = None
    sub_136: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_494);  convolution_102 = unsqueeze_494 = None
    mul_844: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_500);  sub_136 = unsqueeze_500 = None
    sub_137: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_31, mul_844);  where_31 = mul_844 = None
    sub_138: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_137, unsqueeze_497);  sub_137 = unsqueeze_497 = None
    mul_845: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_503);  sub_138 = unsqueeze_503 = None
    mul_846: "f32[736]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_223);  sum_31 = squeeze_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_845, add_466, primals_307, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_845 = add_466 = primals_307 = None
    getitem_237: "f32[8, 184, 7, 7]" = convolution_backward_21[0]
    getitem_238: "f32[736, 184, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_573: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_568, getitem_237);  add_568 = getitem_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_504: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
    unsqueeze_505: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 2);  unsqueeze_504 = None
    unsqueeze_506: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 3);  unsqueeze_505 = None
    sum_32: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_573, [0, 2, 3])
    sub_139: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_506)
    mul_847: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_573, sub_139);  sub_139 = None
    sum_33: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_847, [0, 2, 3]);  mul_847 = None
    mul_848: "f32[184]" = torch.ops.aten.mul.Tensor(sum_32, 0.002551020408163265)
    unsqueeze_507: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
    unsqueeze_508: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
    unsqueeze_509: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 3);  unsqueeze_508 = None
    mul_849: "f32[184]" = torch.ops.aten.mul.Tensor(sum_33, 0.002551020408163265)
    mul_850: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
    mul_851: "f32[184]" = torch.ops.aten.mul.Tensor(mul_849, mul_850);  mul_849 = mul_850 = None
    unsqueeze_510: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_851, 0);  mul_851 = None
    unsqueeze_511: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 2);  unsqueeze_510 = None
    unsqueeze_512: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 3);  unsqueeze_511 = None
    mul_852: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_220, primals_147);  primals_147 = None
    unsqueeze_513: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
    unsqueeze_514: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
    unsqueeze_515: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
    sub_140: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_506);  convolution_101 = unsqueeze_506 = None
    mul_853: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_512);  sub_140 = unsqueeze_512 = None
    sub_141: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_573, mul_853);  mul_853 = None
    sub_142: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_141, unsqueeze_509);  sub_141 = unsqueeze_509 = None
    mul_854: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_515);  sub_142 = unsqueeze_515 = None
    mul_855: "f32[184]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_220);  sum_33 = squeeze_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_854, mul_587, primals_306, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_854 = mul_587 = primals_306 = None
    getitem_240: "f32[8, 736, 7, 7]" = convolution_backward_22[0]
    getitem_241: "f32[184, 736, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_856: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_240, div_74);  div_74 = None
    mul_857: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_240, div_76);  getitem_240 = div_76 = None
    sum_34: "f32[8, 736, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_856, [2, 3], True);  mul_856 = None
    gt_4: "b8[8, 736, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_100, -3.0)
    lt_18: "b8[8, 736, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_100, 3.0);  convolution_100 = None
    bitwise_and_4: "b8[8, 736, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_4, lt_18);  gt_4 = lt_18 = None
    mul_858: "f32[8, 736, 1, 1]" = torch.ops.aten.mul.Tensor(sum_34, 0.16666666666666666);  sum_34 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_32: "f32[8, 736, 1, 1]" = torch.ops.aten.where.self(bitwise_and_4, mul_858, scalar_tensor_18);  bitwise_and_4 = mul_858 = scalar_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_32, div_75, primals_304, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_32 = div_75 = primals_304 = None
    getitem_243: "f32[8, 48, 1, 1]" = convolution_backward_23[0]
    getitem_244: "f32[736, 48, 1, 1]" = convolution_backward_23[1]
    getitem_245: "f32[736]" = convolution_backward_23[2];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_19: "b8[8, 48, 1, 1]" = torch.ops.aten.lt.Scalar(clone_62, -3)
    le_14: "b8[8, 48, 1, 1]" = torch.ops.aten.le.Scalar(clone_62, 3)
    div_114: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(clone_62, 3);  clone_62 = None
    add_574: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(div_114, 0.5);  div_114 = None
    mul_859: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_243, add_574);  add_574 = None
    where_33: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(le_14, mul_859, getitem_243);  le_14 = mul_859 = getitem_243 = None
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_34: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(lt_19, scalar_tensor_19, where_33);  lt_19 = scalar_tensor_19 = where_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(where_34, mean_13, primals_302, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_34 = mean_13 = primals_302 = None
    getitem_246: "f32[8, 736, 1, 1]" = convolution_backward_24[0]
    getitem_247: "f32[48, 736, 1, 1]" = convolution_backward_24[1]
    getitem_248: "f32[48]" = convolution_backward_24[2];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[8, 736, 7, 7]" = torch.ops.aten.expand.default(getitem_246, [8, 736, 7, 7]);  getitem_246 = None
    div_115: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Scalar(expand_5, 49);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_575: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_857, div_115);  mul_857 = div_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_20: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_61, -3)
    le_15: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_61, 3)
    div_116: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_61, 3);  clone_61 = None
    add_576: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_116, 0.5);  div_116 = None
    mul_860: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_575, add_576);  add_576 = None
    where_35: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_15, mul_860, add_575);  le_15 = mul_860 = add_575 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_36: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_20, scalar_tensor_20, where_35);  lt_20 = scalar_tensor_20 = where_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_516: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
    unsqueeze_517: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 2);  unsqueeze_516 = None
    unsqueeze_518: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 3);  unsqueeze_517 = None
    sum_35: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_143: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_518)
    mul_861: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_36, sub_143);  sub_143 = None
    sum_36: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_861, [0, 2, 3]);  mul_861 = None
    mul_862: "f32[736]" = torch.ops.aten.mul.Tensor(sum_35, 0.002551020408163265)
    unsqueeze_519: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
    unsqueeze_520: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 2);  unsqueeze_519 = None
    unsqueeze_521: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 3);  unsqueeze_520 = None
    mul_863: "f32[736]" = torch.ops.aten.mul.Tensor(sum_36, 0.002551020408163265)
    mul_864: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
    mul_865: "f32[736]" = torch.ops.aten.mul.Tensor(mul_863, mul_864);  mul_863 = mul_864 = None
    unsqueeze_522: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_865, 0);  mul_865 = None
    unsqueeze_523: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 2);  unsqueeze_522 = None
    unsqueeze_524: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 3);  unsqueeze_523 = None
    mul_866: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_217, primals_145);  primals_145 = None
    unsqueeze_525: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_866, 0);  mul_866 = None
    unsqueeze_526: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 2);  unsqueeze_525 = None
    unsqueeze_527: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 3);  unsqueeze_526 = None
    sub_144: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_518);  convolution_98 = unsqueeze_518 = None
    mul_867: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_524);  sub_144 = unsqueeze_524 = None
    sub_145: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_36, mul_867);  where_36 = mul_867 = None
    sub_146: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_145, unsqueeze_521);  sub_145 = unsqueeze_521 = None
    mul_868: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_527);  sub_146 = unsqueeze_527 = None
    mul_869: "f32[736]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_217);  sum_36 = squeeze_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_868, div_73, primals_301, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False]);  mul_868 = div_73 = primals_301 = None
    getitem_249: "f32[8, 736, 7, 7]" = convolution_backward_25[0]
    getitem_250: "f32[736, 1, 5, 5]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_21: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_60, -3)
    le_16: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_60, 3)
    div_117: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_60, 3);  clone_60 = None
    add_577: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_117, 0.5);  div_117 = None
    mul_870: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_249, add_577);  add_577 = None
    where_37: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_16, mul_870, getitem_249);  le_16 = mul_870 = getitem_249 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_38: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_21, scalar_tensor_21, where_37);  lt_21 = scalar_tensor_21 = where_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_528: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
    unsqueeze_529: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 2);  unsqueeze_528 = None
    unsqueeze_530: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 3);  unsqueeze_529 = None
    sum_37: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_147: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_530)
    mul_871: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_38, sub_147);  sub_147 = None
    sum_38: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_871, [0, 2, 3]);  mul_871 = None
    mul_872: "f32[736]" = torch.ops.aten.mul.Tensor(sum_37, 0.002551020408163265)
    unsqueeze_531: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
    unsqueeze_532: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 2);  unsqueeze_531 = None
    unsqueeze_533: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 3);  unsqueeze_532 = None
    mul_873: "f32[736]" = torch.ops.aten.mul.Tensor(sum_38, 0.002551020408163265)
    mul_874: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
    mul_875: "f32[736]" = torch.ops.aten.mul.Tensor(mul_873, mul_874);  mul_873 = mul_874 = None
    unsqueeze_534: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
    unsqueeze_535: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 2);  unsqueeze_534 = None
    unsqueeze_536: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 3);  unsqueeze_535 = None
    mul_876: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_214, primals_143);  primals_143 = None
    unsqueeze_537: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_876, 0);  mul_876 = None
    unsqueeze_538: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 2);  unsqueeze_537 = None
    unsqueeze_539: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 3);  unsqueeze_538 = None
    sub_148: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_530);  convolution_97 = unsqueeze_530 = None
    mul_877: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_536);  sub_148 = unsqueeze_536 = None
    sub_149: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_38, mul_877);  where_38 = mul_877 = None
    sub_150: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_149, unsqueeze_533);  sub_149 = unsqueeze_533 = None
    mul_878: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_539);  sub_150 = unsqueeze_539 = None
    mul_879: "f32[736]" = torch.ops.aten.mul.Tensor(sum_38, squeeze_214);  sum_38 = squeeze_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_878, add_446, primals_300, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_878 = add_446 = primals_300 = None
    getitem_252: "f32[8, 184, 7, 7]" = convolution_backward_26[0]
    getitem_253: "f32[736, 184, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_578: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_573, getitem_252);  add_573 = getitem_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_540: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
    unsqueeze_541: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 2);  unsqueeze_540 = None
    unsqueeze_542: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 3);  unsqueeze_541 = None
    sum_39: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_578, [0, 2, 3])
    sub_151: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_542)
    mul_880: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_578, sub_151);  sub_151 = None
    sum_40: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_880, [0, 2, 3]);  mul_880 = None
    mul_881: "f32[184]" = torch.ops.aten.mul.Tensor(sum_39, 0.002551020408163265)
    unsqueeze_543: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    unsqueeze_544: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 2);  unsqueeze_543 = None
    unsqueeze_545: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 3);  unsqueeze_544 = None
    mul_882: "f32[184]" = torch.ops.aten.mul.Tensor(sum_40, 0.002551020408163265)
    mul_883: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
    mul_884: "f32[184]" = torch.ops.aten.mul.Tensor(mul_882, mul_883);  mul_882 = mul_883 = None
    unsqueeze_546: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_547: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 2);  unsqueeze_546 = None
    unsqueeze_548: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 3);  unsqueeze_547 = None
    mul_885: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_211, primals_141);  primals_141 = None
    unsqueeze_549: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_885, 0);  mul_885 = None
    unsqueeze_550: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 2);  unsqueeze_549 = None
    unsqueeze_551: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 3);  unsqueeze_550 = None
    sub_152: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_542);  convolution_96 = unsqueeze_542 = None
    mul_886: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_548);  sub_152 = unsqueeze_548 = None
    sub_153: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_578, mul_886);  mul_886 = None
    sub_154: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_153, unsqueeze_545);  sub_153 = unsqueeze_545 = None
    mul_887: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_551);  sub_154 = unsqueeze_551 = None
    mul_888: "f32[184]" = torch.ops.aten.mul.Tensor(sum_40, squeeze_211);  sum_40 = squeeze_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_887, mul_562, primals_299, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_887 = mul_562 = primals_299 = None
    getitem_255: "f32[8, 736, 7, 7]" = convolution_backward_27[0]
    getitem_256: "f32[184, 736, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_889: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_255, div_70);  div_70 = None
    mul_890: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_255, div_72);  getitem_255 = div_72 = None
    sum_41: "f32[8, 736, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_889, [2, 3], True);  mul_889 = None
    gt_5: "b8[8, 736, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_95, -3.0)
    lt_22: "b8[8, 736, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_95, 3.0);  convolution_95 = None
    bitwise_and_5: "b8[8, 736, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_5, lt_22);  gt_5 = lt_22 = None
    mul_891: "f32[8, 736, 1, 1]" = torch.ops.aten.mul.Tensor(sum_41, 0.16666666666666666);  sum_41 = None
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_39: "f32[8, 736, 1, 1]" = torch.ops.aten.where.self(bitwise_and_5, mul_891, scalar_tensor_22);  bitwise_and_5 = mul_891 = scalar_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(where_39, div_71, primals_297, [736], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_39 = div_71 = primals_297 = None
    getitem_258: "f32[8, 48, 1, 1]" = convolution_backward_28[0]
    getitem_259: "f32[736, 48, 1, 1]" = convolution_backward_28[1]
    getitem_260: "f32[736]" = convolution_backward_28[2];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_23: "b8[8, 48, 1, 1]" = torch.ops.aten.lt.Scalar(clone_59, -3)
    le_17: "b8[8, 48, 1, 1]" = torch.ops.aten.le.Scalar(clone_59, 3)
    div_118: "f32[8, 48, 1, 1]" = torch.ops.aten.div.Tensor(clone_59, 3);  clone_59 = None
    add_579: "f32[8, 48, 1, 1]" = torch.ops.aten.add.Tensor(div_118, 0.5);  div_118 = None
    mul_892: "f32[8, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_258, add_579);  add_579 = None
    where_40: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(le_17, mul_892, getitem_258);  le_17 = mul_892 = getitem_258 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_41: "f32[8, 48, 1, 1]" = torch.ops.aten.where.self(lt_23, scalar_tensor_23, where_40);  lt_23 = scalar_tensor_23 = where_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(where_41, mean_12, primals_295, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_41 = mean_12 = primals_295 = None
    getitem_261: "f32[8, 736, 1, 1]" = convolution_backward_29[0]
    getitem_262: "f32[48, 736, 1, 1]" = convolution_backward_29[1]
    getitem_263: "f32[48]" = convolution_backward_29[2];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[8, 736, 7, 7]" = torch.ops.aten.expand.default(getitem_261, [8, 736, 7, 7]);  getitem_261 = None
    div_119: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Scalar(expand_6, 49);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_580: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(mul_890, div_119);  mul_890 = div_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_24: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_58, -3)
    le_18: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_58, 3)
    div_120: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_58, 3);  clone_58 = None
    add_581: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_120, 0.5);  div_120 = None
    mul_893: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(add_580, add_581);  add_581 = None
    where_42: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_18, mul_893, add_580);  le_18 = mul_893 = add_580 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_43: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_24, scalar_tensor_24, where_42);  lt_24 = scalar_tensor_24 = where_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_552: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
    unsqueeze_553: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 2);  unsqueeze_552 = None
    unsqueeze_554: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 3);  unsqueeze_553 = None
    sum_42: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_155: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_554)
    mul_894: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_43, sub_155);  sub_155 = None
    sum_43: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_894, [0, 2, 3]);  mul_894 = None
    mul_895: "f32[736]" = torch.ops.aten.mul.Tensor(sum_42, 0.002551020408163265)
    unsqueeze_555: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_895, 0);  mul_895 = None
    unsqueeze_556: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 2);  unsqueeze_555 = None
    unsqueeze_557: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 3);  unsqueeze_556 = None
    mul_896: "f32[736]" = torch.ops.aten.mul.Tensor(sum_43, 0.002551020408163265)
    mul_897: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
    mul_898: "f32[736]" = torch.ops.aten.mul.Tensor(mul_896, mul_897);  mul_896 = mul_897 = None
    unsqueeze_558: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_898, 0);  mul_898 = None
    unsqueeze_559: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 2);  unsqueeze_558 = None
    unsqueeze_560: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 3);  unsqueeze_559 = None
    mul_899: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_208, primals_139);  primals_139 = None
    unsqueeze_561: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
    unsqueeze_562: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 2);  unsqueeze_561 = None
    unsqueeze_563: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 3);  unsqueeze_562 = None
    sub_156: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_554);  convolution_93 = unsqueeze_554 = None
    mul_900: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_560);  sub_156 = unsqueeze_560 = None
    sub_157: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_43, mul_900);  where_43 = mul_900 = None
    sub_158: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_157, unsqueeze_557);  sub_157 = unsqueeze_557 = None
    mul_901: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_563);  sub_158 = unsqueeze_563 = None
    mul_902: "f32[736]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_208);  sum_43 = squeeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_901, div_69, primals_294, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 736, [True, True, False]);  mul_901 = div_69 = primals_294 = None
    getitem_264: "f32[8, 736, 7, 7]" = convolution_backward_30[0]
    getitem_265: "f32[736, 1, 5, 5]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_25: "b8[8, 736, 7, 7]" = torch.ops.aten.lt.Scalar(clone_57, -3)
    le_19: "b8[8, 736, 7, 7]" = torch.ops.aten.le.Scalar(clone_57, 3)
    div_121: "f32[8, 736, 7, 7]" = torch.ops.aten.div.Tensor(clone_57, 3);  clone_57 = None
    add_582: "f32[8, 736, 7, 7]" = torch.ops.aten.add.Tensor(div_121, 0.5);  div_121 = None
    mul_903: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_264, add_582);  add_582 = None
    where_44: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(le_19, mul_903, getitem_264);  le_19 = mul_903 = getitem_264 = None
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_45: "f32[8, 736, 7, 7]" = torch.ops.aten.where.self(lt_25, scalar_tensor_25, where_44);  lt_25 = scalar_tensor_25 = where_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_564: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
    unsqueeze_565: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 2);  unsqueeze_564 = None
    unsqueeze_566: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 3);  unsqueeze_565 = None
    sum_44: "f32[736]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_159: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_566)
    mul_904: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(where_45, sub_159);  sub_159 = None
    sum_45: "f32[736]" = torch.ops.aten.sum.dim_IntList(mul_904, [0, 2, 3]);  mul_904 = None
    mul_905: "f32[736]" = torch.ops.aten.mul.Tensor(sum_44, 0.002551020408163265)
    unsqueeze_567: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_905, 0);  mul_905 = None
    unsqueeze_568: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 2);  unsqueeze_567 = None
    unsqueeze_569: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 3);  unsqueeze_568 = None
    mul_906: "f32[736]" = torch.ops.aten.mul.Tensor(sum_45, 0.002551020408163265)
    mul_907: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
    mul_908: "f32[736]" = torch.ops.aten.mul.Tensor(mul_906, mul_907);  mul_906 = mul_907 = None
    unsqueeze_570: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_908, 0);  mul_908 = None
    unsqueeze_571: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 2);  unsqueeze_570 = None
    unsqueeze_572: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 3);  unsqueeze_571 = None
    mul_909: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_205, primals_137);  primals_137 = None
    unsqueeze_573: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
    unsqueeze_574: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 2);  unsqueeze_573 = None
    unsqueeze_575: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 3);  unsqueeze_574 = None
    sub_160: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_566);  convolution_92 = unsqueeze_566 = None
    mul_910: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_572);  sub_160 = unsqueeze_572 = None
    sub_161: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(where_45, mul_910);  where_45 = mul_910 = None
    sub_162: "f32[8, 736, 7, 7]" = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_569);  sub_161 = unsqueeze_569 = None
    mul_911: "f32[8, 736, 7, 7]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_575);  sub_162 = unsqueeze_575 = None
    mul_912: "f32[736]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_205);  sum_45 = squeeze_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_911, add_426, primals_293, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_911 = add_426 = primals_293 = None
    getitem_267: "f32[8, 184, 7, 7]" = convolution_backward_31[0]
    getitem_268: "f32[736, 184, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_583: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_578, getitem_267);  add_578 = getitem_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_576: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
    unsqueeze_577: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 2);  unsqueeze_576 = None
    unsqueeze_578: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 3);  unsqueeze_577 = None
    sum_46: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_583, [0, 2, 3])
    sub_163: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_578)
    mul_913: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_583, sub_163);  sub_163 = None
    sum_47: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_913, [0, 2, 3]);  mul_913 = None
    mul_914: "f32[184]" = torch.ops.aten.mul.Tensor(sum_46, 0.002551020408163265)
    unsqueeze_579: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
    unsqueeze_580: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 2);  unsqueeze_579 = None
    unsqueeze_581: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 3);  unsqueeze_580 = None
    mul_915: "f32[184]" = torch.ops.aten.mul.Tensor(sum_47, 0.002551020408163265)
    mul_916: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
    mul_917: "f32[184]" = torch.ops.aten.mul.Tensor(mul_915, mul_916);  mul_915 = mul_916 = None
    unsqueeze_582: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_917, 0);  mul_917 = None
    unsqueeze_583: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 2);  unsqueeze_582 = None
    unsqueeze_584: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 3);  unsqueeze_583 = None
    mul_918: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_202, primals_135);  primals_135 = None
    unsqueeze_585: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
    unsqueeze_586: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
    unsqueeze_587: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
    sub_164: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_578);  convolution_91 = unsqueeze_578 = None
    mul_919: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_584);  sub_164 = unsqueeze_584 = None
    sub_165: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_583, mul_919);  add_583 = mul_919 = None
    sub_166: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_581);  sub_165 = unsqueeze_581 = None
    mul_920: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_587);  sub_166 = unsqueeze_587 = None
    mul_921: "f32[184]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_202);  sum_47 = squeeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_920, mul_537, primals_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_920 = mul_537 = primals_292 = None
    getitem_270: "f32[8, 720, 7, 7]" = convolution_backward_32[0]
    getitem_271: "f32[184, 720, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_922: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_270, div_66);  div_66 = None
    mul_923: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_270, div_68);  getitem_270 = div_68 = None
    sum_48: "f32[8, 720, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_922, [2, 3], True);  mul_922 = None
    gt_6: "b8[8, 720, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_90, -3.0)
    lt_26: "b8[8, 720, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_90, 3.0);  convolution_90 = None
    bitwise_and_6: "b8[8, 720, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_6, lt_26);  gt_6 = lt_26 = None
    mul_924: "f32[8, 720, 1, 1]" = torch.ops.aten.mul.Tensor(sum_48, 0.16666666666666666);  sum_48 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_46: "f32[8, 720, 1, 1]" = torch.ops.aten.where.self(bitwise_and_6, mul_924, scalar_tensor_26);  bitwise_and_6 = mul_924 = scalar_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_46, div_67, primals_290, [720], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_46 = div_67 = primals_290 = None
    getitem_273: "f32[8, 32, 1, 1]" = convolution_backward_33[0]
    getitem_274: "f32[720, 32, 1, 1]" = convolution_backward_33[1]
    getitem_275: "f32[720]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_27: "b8[8, 32, 1, 1]" = torch.ops.aten.lt.Scalar(clone_56, -3)
    le_20: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(clone_56, 3)
    div_122: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(clone_56, 3);  clone_56 = None
    add_584: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(div_122, 0.5);  div_122 = None
    mul_925: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_273, add_584);  add_584 = None
    where_47: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_20, mul_925, getitem_273);  le_20 = mul_925 = getitem_273 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_48: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(lt_27, scalar_tensor_27, where_47);  lt_27 = scalar_tensor_27 = where_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(where_48, mean_11, primals_288, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_48 = mean_11 = primals_288 = None
    getitem_276: "f32[8, 720, 1, 1]" = convolution_backward_34[0]
    getitem_277: "f32[32, 720, 1, 1]" = convolution_backward_34[1]
    getitem_278: "f32[32]" = convolution_backward_34[2];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[8, 720, 7, 7]" = torch.ops.aten.expand.default(getitem_276, [8, 720, 7, 7]);  getitem_276 = None
    div_123: "f32[8, 720, 7, 7]" = torch.ops.aten.div.Scalar(expand_7, 49);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_585: "f32[8, 720, 7, 7]" = torch.ops.aten.add.Tensor(mul_923, div_123);  mul_923 = div_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_28: "b8[8, 720, 7, 7]" = torch.ops.aten.lt.Scalar(clone_55, -3)
    le_21: "b8[8, 720, 7, 7]" = torch.ops.aten.le.Scalar(clone_55, 3)
    div_124: "f32[8, 720, 7, 7]" = torch.ops.aten.div.Tensor(clone_55, 3);  clone_55 = None
    add_586: "f32[8, 720, 7, 7]" = torch.ops.aten.add.Tensor(div_124, 0.5);  div_124 = None
    mul_926: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(add_585, add_586);  add_586 = None
    where_49: "f32[8, 720, 7, 7]" = torch.ops.aten.where.self(le_21, mul_926, add_585);  le_21 = mul_926 = add_585 = None
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_50: "f32[8, 720, 7, 7]" = torch.ops.aten.where.self(lt_28, scalar_tensor_28, where_49);  lt_28 = scalar_tensor_28 = where_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_588: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
    unsqueeze_589: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 2);  unsqueeze_588 = None
    unsqueeze_590: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 3);  unsqueeze_589 = None
    sum_49: "f32[720]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_167: "f32[8, 720, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_590)
    mul_927: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(where_50, sub_167);  sub_167 = None
    sum_50: "f32[720]" = torch.ops.aten.sum.dim_IntList(mul_927, [0, 2, 3]);  mul_927 = None
    mul_928: "f32[720]" = torch.ops.aten.mul.Tensor(sum_49, 0.002551020408163265)
    unsqueeze_591: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(mul_928, 0);  mul_928 = None
    unsqueeze_592: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 2);  unsqueeze_591 = None
    unsqueeze_593: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 3);  unsqueeze_592 = None
    mul_929: "f32[720]" = torch.ops.aten.mul.Tensor(sum_50, 0.002551020408163265)
    mul_930: "f32[720]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_931: "f32[720]" = torch.ops.aten.mul.Tensor(mul_929, mul_930);  mul_929 = mul_930 = None
    unsqueeze_594: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
    unsqueeze_595: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 2);  unsqueeze_594 = None
    unsqueeze_596: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 3);  unsqueeze_595 = None
    mul_932: "f32[720]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_133);  primals_133 = None
    unsqueeze_597: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
    unsqueeze_598: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
    unsqueeze_599: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
    sub_168: "f32[8, 720, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_590);  convolution_88 = unsqueeze_590 = None
    mul_933: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_596);  sub_168 = unsqueeze_596 = None
    sub_169: "f32[8, 720, 7, 7]" = torch.ops.aten.sub.Tensor(where_50, mul_933);  where_50 = mul_933 = None
    sub_170: "f32[8, 720, 7, 7]" = torch.ops.aten.sub.Tensor(sub_169, unsqueeze_593);  sub_169 = unsqueeze_593 = None
    mul_934: "f32[8, 720, 7, 7]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_599);  sub_170 = unsqueeze_599 = None
    mul_935: "f32[720]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_199);  sum_50 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_934, div_65, primals_287, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 720, [True, True, False]);  mul_934 = div_65 = primals_287 = None
    getitem_279: "f32[8, 720, 14, 14]" = convolution_backward_35[0]
    getitem_280: "f32[720, 1, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_29: "b8[8, 720, 14, 14]" = torch.ops.aten.lt.Scalar(clone_54, -3)
    le_22: "b8[8, 720, 14, 14]" = torch.ops.aten.le.Scalar(clone_54, 3)
    div_125: "f32[8, 720, 14, 14]" = torch.ops.aten.div.Tensor(clone_54, 3);  clone_54 = None
    add_587: "f32[8, 720, 14, 14]" = torch.ops.aten.add.Tensor(div_125, 0.5);  div_125 = None
    mul_936: "f32[8, 720, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_279, add_587);  add_587 = None
    where_51: "f32[8, 720, 14, 14]" = torch.ops.aten.where.self(le_22, mul_936, getitem_279);  le_22 = mul_936 = getitem_279 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_52: "f32[8, 720, 14, 14]" = torch.ops.aten.where.self(lt_29, scalar_tensor_29, where_51);  lt_29 = scalar_tensor_29 = where_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_600: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
    unsqueeze_601: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 2);  unsqueeze_600 = None
    unsqueeze_602: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 3);  unsqueeze_601 = None
    sum_51: "f32[720]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_171: "f32[8, 720, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_602)
    mul_937: "f32[8, 720, 14, 14]" = torch.ops.aten.mul.Tensor(where_52, sub_171);  sub_171 = None
    sum_52: "f32[720]" = torch.ops.aten.sum.dim_IntList(mul_937, [0, 2, 3]);  mul_937 = None
    mul_938: "f32[720]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    unsqueeze_603: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
    unsqueeze_604: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 2);  unsqueeze_603 = None
    unsqueeze_605: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 3);  unsqueeze_604 = None
    mul_939: "f32[720]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    mul_940: "f32[720]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_941: "f32[720]" = torch.ops.aten.mul.Tensor(mul_939, mul_940);  mul_939 = mul_940 = None
    unsqueeze_606: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
    unsqueeze_607: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 2);  unsqueeze_606 = None
    unsqueeze_608: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 3);  unsqueeze_607 = None
    mul_942: "f32[720]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_131);  primals_131 = None
    unsqueeze_609: "f32[1, 720]" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
    unsqueeze_610: "f32[1, 720, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
    unsqueeze_611: "f32[1, 720, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
    sub_172: "f32[8, 720, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_602);  convolution_87 = unsqueeze_602 = None
    mul_943: "f32[8, 720, 14, 14]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_608);  sub_172 = unsqueeze_608 = None
    sub_173: "f32[8, 720, 14, 14]" = torch.ops.aten.sub.Tensor(where_52, mul_943);  where_52 = mul_943 = None
    sub_174: "f32[8, 720, 14, 14]" = torch.ops.aten.sub.Tensor(sub_173, unsqueeze_605);  sub_173 = unsqueeze_605 = None
    mul_944: "f32[8, 720, 14, 14]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_611);  sub_174 = unsqueeze_611 = None
    mul_945: "f32[720]" = torch.ops.aten.mul.Tensor(sum_52, squeeze_196);  sum_52 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_944, add_407, primals_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_944 = add_407 = primals_286 = None
    getitem_282: "f32[8, 120, 14, 14]" = convolution_backward_36[0]
    getitem_283: "f32[720, 120, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_612: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
    unsqueeze_613: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 2);  unsqueeze_612 = None
    unsqueeze_614: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 3);  unsqueeze_613 = None
    sum_53: "f32[120]" = torch.ops.aten.sum.dim_IntList(getitem_282, [0, 2, 3])
    sub_175: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_614)
    mul_946: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_282, sub_175);  sub_175 = None
    sum_54: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_946, [0, 2, 3]);  mul_946 = None
    mul_947: "f32[120]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    unsqueeze_615: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_947, 0);  mul_947 = None
    unsqueeze_616: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 2);  unsqueeze_615 = None
    unsqueeze_617: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 3);  unsqueeze_616 = None
    mul_948: "f32[120]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    mul_949: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_950: "f32[120]" = torch.ops.aten.mul.Tensor(mul_948, mul_949);  mul_948 = mul_949 = None
    unsqueeze_618: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
    unsqueeze_619: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 2);  unsqueeze_618 = None
    unsqueeze_620: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 3);  unsqueeze_619 = None
    mul_951: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_129);  primals_129 = None
    unsqueeze_621: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
    unsqueeze_622: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 2);  unsqueeze_621 = None
    unsqueeze_623: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 3);  unsqueeze_622 = None
    sub_176: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_614);  convolution_86 = unsqueeze_614 = None
    mul_952: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_620);  sub_176 = unsqueeze_620 = None
    sub_177: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_282, mul_952);  mul_952 = None
    sub_178: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_177, unsqueeze_617);  sub_177 = unsqueeze_617 = None
    mul_953: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_623);  sub_178 = unsqueeze_623 = None
    mul_954: "f32[120]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_193);  sum_54 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_953, mul_512, primals_285, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_953 = mul_512 = primals_285 = None
    getitem_285: "f32[8, 360, 14, 14]" = convolution_backward_37[0]
    getitem_286: "f32[120, 360, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_955: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_285, div_62);  div_62 = None
    mul_956: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_285, div_64);  getitem_285 = div_64 = None
    sum_55: "f32[8, 360, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_955, [2, 3], True);  mul_955 = None
    gt_7: "b8[8, 360, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_85, -3.0)
    lt_30: "b8[8, 360, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_85, 3.0);  convolution_85 = None
    bitwise_and_7: "b8[8, 360, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_7, lt_30);  gt_7 = lt_30 = None
    mul_957: "f32[8, 360, 1, 1]" = torch.ops.aten.mul.Tensor(sum_55, 0.16666666666666666);  sum_55 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_53: "f32[8, 360, 1, 1]" = torch.ops.aten.where.self(bitwise_and_7, mul_957, scalar_tensor_30);  bitwise_and_7 = mul_957 = scalar_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(where_53, div_63, primals_283, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_53 = div_63 = primals_283 = None
    getitem_288: "f32[8, 32, 1, 1]" = convolution_backward_38[0]
    getitem_289: "f32[360, 32, 1, 1]" = convolution_backward_38[1]
    getitem_290: "f32[360]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_31: "b8[8, 32, 1, 1]" = torch.ops.aten.lt.Scalar(clone_53, -3)
    le_23: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(clone_53, 3)
    div_126: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(clone_53, 3);  clone_53 = None
    add_588: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(div_126, 0.5);  div_126 = None
    mul_958: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_288, add_588);  add_588 = None
    where_54: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_23, mul_958, getitem_288);  le_23 = mul_958 = getitem_288 = None
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_55: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(lt_31, scalar_tensor_31, where_54);  lt_31 = scalar_tensor_31 = where_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(where_55, mean_10, primals_281, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_55 = mean_10 = primals_281 = None
    getitem_291: "f32[8, 360, 1, 1]" = convolution_backward_39[0]
    getitem_292: "f32[32, 360, 1, 1]" = convolution_backward_39[1]
    getitem_293: "f32[32]" = convolution_backward_39[2];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[8, 360, 14, 14]" = torch.ops.aten.expand.default(getitem_291, [8, 360, 14, 14]);  getitem_291 = None
    div_127: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_589: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_956, div_127);  mul_956 = div_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_32: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_52, -3)
    le_24: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_52, 3)
    div_128: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_52, 3);  clone_52 = None
    add_590: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_128, 0.5);  div_128 = None
    mul_959: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_589, add_590);  add_590 = None
    where_56: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_24, mul_959, add_589);  le_24 = mul_959 = add_589 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_57: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_32, scalar_tensor_32, where_56);  lt_32 = scalar_tensor_32 = where_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_624: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
    unsqueeze_625: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 2);  unsqueeze_624 = None
    unsqueeze_626: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 3);  unsqueeze_625 = None
    sum_56: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_179: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_626)
    mul_960: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_57, sub_179);  sub_179 = None
    sum_57: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_960, [0, 2, 3]);  mul_960 = None
    mul_961: "f32[360]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_627: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_961, 0);  mul_961 = None
    unsqueeze_628: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 2);  unsqueeze_627 = None
    unsqueeze_629: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 3);  unsqueeze_628 = None
    mul_962: "f32[360]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_963: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_964: "f32[360]" = torch.ops.aten.mul.Tensor(mul_962, mul_963);  mul_962 = mul_963 = None
    unsqueeze_630: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_964, 0);  mul_964 = None
    unsqueeze_631: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 2);  unsqueeze_630 = None
    unsqueeze_632: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 3);  unsqueeze_631 = None
    mul_965: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_127);  primals_127 = None
    unsqueeze_633: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_965, 0);  mul_965 = None
    unsqueeze_634: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 2);  unsqueeze_633 = None
    unsqueeze_635: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 3);  unsqueeze_634 = None
    sub_180: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_626);  convolution_83 = unsqueeze_626 = None
    mul_966: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_632);  sub_180 = unsqueeze_632 = None
    sub_181: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_57, mul_966);  where_57 = mul_966 = None
    sub_182: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_629);  sub_181 = unsqueeze_629 = None
    mul_967: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_635);  sub_182 = unsqueeze_635 = None
    mul_968: "f32[360]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_190);  sum_57 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_967, div_61, primals_280, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False]);  mul_967 = div_61 = primals_280 = None
    getitem_294: "f32[8, 360, 14, 14]" = convolution_backward_40[0]
    getitem_295: "f32[360, 1, 5, 5]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_33: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_51, -3)
    le_25: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_51, 3)
    div_129: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_51, 3);  clone_51 = None
    add_591: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_129, 0.5);  div_129 = None
    mul_969: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_294, add_591);  add_591 = None
    where_58: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_25, mul_969, getitem_294);  le_25 = mul_969 = getitem_294 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_59: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_33, scalar_tensor_33, where_58);  lt_33 = scalar_tensor_33 = where_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_636: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
    unsqueeze_637: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 2);  unsqueeze_636 = None
    unsqueeze_638: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 3);  unsqueeze_637 = None
    sum_58: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    sub_183: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_638)
    mul_970: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_59, sub_183);  sub_183 = None
    sum_59: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_970, [0, 2, 3]);  mul_970 = None
    mul_971: "f32[360]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_639: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
    unsqueeze_640: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 2);  unsqueeze_639 = None
    unsqueeze_641: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 3);  unsqueeze_640 = None
    mul_972: "f32[360]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_973: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_974: "f32[360]" = torch.ops.aten.mul.Tensor(mul_972, mul_973);  mul_972 = mul_973 = None
    unsqueeze_642: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
    unsqueeze_643: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 2);  unsqueeze_642 = None
    unsqueeze_644: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 3);  unsqueeze_643 = None
    mul_975: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_125);  primals_125 = None
    unsqueeze_645: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_975, 0);  mul_975 = None
    unsqueeze_646: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 2);  unsqueeze_645 = None
    unsqueeze_647: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 3);  unsqueeze_646 = None
    sub_184: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_638);  convolution_82 = unsqueeze_638 = None
    mul_976: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_644);  sub_184 = unsqueeze_644 = None
    sub_185: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_59, mul_976);  where_59 = mul_976 = None
    sub_186: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_185, unsqueeze_641);  sub_185 = unsqueeze_641 = None
    mul_977: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_647);  sub_186 = unsqueeze_647 = None
    mul_978: "f32[360]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_187);  sum_59 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_977, add_387, primals_279, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_977 = add_387 = primals_279 = None
    getitem_297: "f32[8, 120, 14, 14]" = convolution_backward_41[0]
    getitem_298: "f32[360, 120, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_592: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(getitem_282, getitem_297);  getitem_282 = getitem_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_648: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
    unsqueeze_649: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 2);  unsqueeze_648 = None
    unsqueeze_650: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 3);  unsqueeze_649 = None
    sum_60: "f32[120]" = torch.ops.aten.sum.dim_IntList(add_592, [0, 2, 3])
    sub_187: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_650)
    mul_979: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(add_592, sub_187);  sub_187 = None
    sum_61: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_979, [0, 2, 3]);  mul_979 = None
    mul_980: "f32[120]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_651: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_980, 0);  mul_980 = None
    unsqueeze_652: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 2);  unsqueeze_651 = None
    unsqueeze_653: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 3);  unsqueeze_652 = None
    mul_981: "f32[120]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_982: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_983: "f32[120]" = torch.ops.aten.mul.Tensor(mul_981, mul_982);  mul_981 = mul_982 = None
    unsqueeze_654: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_655: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 2);  unsqueeze_654 = None
    unsqueeze_656: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 3);  unsqueeze_655 = None
    mul_984: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_123);  primals_123 = None
    unsqueeze_657: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_984, 0);  mul_984 = None
    unsqueeze_658: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 2);  unsqueeze_657 = None
    unsqueeze_659: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 3);  unsqueeze_658 = None
    sub_188: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_650);  convolution_81 = unsqueeze_650 = None
    mul_985: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_656);  sub_188 = unsqueeze_656 = None
    sub_189: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(add_592, mul_985);  mul_985 = None
    sub_190: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_653);  sub_189 = unsqueeze_653 = None
    mul_986: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_659);  sub_190 = unsqueeze_659 = None
    mul_987: "f32[120]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_184);  sum_61 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_986, mul_487, primals_278, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_986 = mul_487 = primals_278 = None
    getitem_300: "f32[8, 360, 14, 14]" = convolution_backward_42[0]
    getitem_301: "f32[120, 360, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_988: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_300, div_58);  div_58 = None
    mul_989: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_300, div_60);  getitem_300 = div_60 = None
    sum_62: "f32[8, 360, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_988, [2, 3], True);  mul_988 = None
    gt_8: "b8[8, 360, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_80, -3.0)
    lt_34: "b8[8, 360, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_80, 3.0);  convolution_80 = None
    bitwise_and_8: "b8[8, 360, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_8, lt_34);  gt_8 = lt_34 = None
    mul_990: "f32[8, 360, 1, 1]" = torch.ops.aten.mul.Tensor(sum_62, 0.16666666666666666);  sum_62 = None
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_60: "f32[8, 360, 1, 1]" = torch.ops.aten.where.self(bitwise_and_8, mul_990, scalar_tensor_34);  bitwise_and_8 = mul_990 = scalar_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(where_60, div_59, primals_276, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_60 = div_59 = primals_276 = None
    getitem_303: "f32[8, 32, 1, 1]" = convolution_backward_43[0]
    getitem_304: "f32[360, 32, 1, 1]" = convolution_backward_43[1]
    getitem_305: "f32[360]" = convolution_backward_43[2];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_35: "b8[8, 32, 1, 1]" = torch.ops.aten.lt.Scalar(clone_50, -3)
    le_26: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(clone_50, 3)
    div_130: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(clone_50, 3);  clone_50 = None
    add_593: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(div_130, 0.5);  div_130 = None
    mul_991: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_303, add_593);  add_593 = None
    where_61: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_26, mul_991, getitem_303);  le_26 = mul_991 = getitem_303 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_62: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(lt_35, scalar_tensor_35, where_61);  lt_35 = scalar_tensor_35 = where_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(where_62, mean_9, primals_274, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_62 = mean_9 = primals_274 = None
    getitem_306: "f32[8, 360, 1, 1]" = convolution_backward_44[0]
    getitem_307: "f32[32, 360, 1, 1]" = convolution_backward_44[1]
    getitem_308: "f32[32]" = convolution_backward_44[2];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[8, 360, 14, 14]" = torch.ops.aten.expand.default(getitem_306, [8, 360, 14, 14]);  getitem_306 = None
    div_131: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_594: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_989, div_131);  mul_989 = div_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_36: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_49, -3)
    le_27: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_49, 3)
    div_132: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_49, 3);  clone_49 = None
    add_595: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_132, 0.5);  div_132 = None
    mul_992: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_594, add_595);  add_595 = None
    where_63: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_27, mul_992, add_594);  le_27 = mul_992 = add_594 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_64: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_36, scalar_tensor_36, where_63);  lt_36 = scalar_tensor_36 = where_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_660: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
    unsqueeze_661: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 2);  unsqueeze_660 = None
    unsqueeze_662: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 3);  unsqueeze_661 = None
    sum_63: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_191: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_662)
    mul_993: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_64, sub_191);  sub_191 = None
    sum_64: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_993, [0, 2, 3]);  mul_993 = None
    mul_994: "f32[360]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    unsqueeze_663: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_994, 0);  mul_994 = None
    unsqueeze_664: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 2);  unsqueeze_663 = None
    unsqueeze_665: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 3);  unsqueeze_664 = None
    mul_995: "f32[360]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    mul_996: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_997: "f32[360]" = torch.ops.aten.mul.Tensor(mul_995, mul_996);  mul_995 = mul_996 = None
    unsqueeze_666: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_997, 0);  mul_997 = None
    unsqueeze_667: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 2);  unsqueeze_666 = None
    unsqueeze_668: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 3);  unsqueeze_667 = None
    mul_998: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_121);  primals_121 = None
    unsqueeze_669: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_998, 0);  mul_998 = None
    unsqueeze_670: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 2);  unsqueeze_669 = None
    unsqueeze_671: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 3);  unsqueeze_670 = None
    sub_192: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_662);  convolution_78 = unsqueeze_662 = None
    mul_999: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_668);  sub_192 = unsqueeze_668 = None
    sub_193: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_64, mul_999);  where_64 = mul_999 = None
    sub_194: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_193, unsqueeze_665);  sub_193 = unsqueeze_665 = None
    mul_1000: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_671);  sub_194 = unsqueeze_671 = None
    mul_1001: "f32[360]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_181);  sum_64 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_1000, div_57, primals_273, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False]);  mul_1000 = div_57 = primals_273 = None
    getitem_309: "f32[8, 360, 14, 14]" = convolution_backward_45[0]
    getitem_310: "f32[360, 1, 5, 5]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_37: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_48, -3)
    le_28: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_48, 3)
    div_133: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_48, 3);  clone_48 = None
    add_596: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_133, 0.5);  div_133 = None
    mul_1002: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_309, add_596);  add_596 = None
    where_65: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_28, mul_1002, getitem_309);  le_28 = mul_1002 = getitem_309 = None
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_66: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_37, scalar_tensor_37, where_65);  lt_37 = scalar_tensor_37 = where_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_672: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_673: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 2);  unsqueeze_672 = None
    unsqueeze_674: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 3);  unsqueeze_673 = None
    sum_65: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_195: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_674)
    mul_1003: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_66, sub_195);  sub_195 = None
    sum_66: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1003, [0, 2, 3]);  mul_1003 = None
    mul_1004: "f32[360]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    unsqueeze_675: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
    unsqueeze_676: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 2);  unsqueeze_675 = None
    unsqueeze_677: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 3);  unsqueeze_676 = None
    mul_1005: "f32[360]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    mul_1006: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_1007: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1005, mul_1006);  mul_1005 = mul_1006 = None
    unsqueeze_678: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1007, 0);  mul_1007 = None
    unsqueeze_679: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 2);  unsqueeze_678 = None
    unsqueeze_680: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 3);  unsqueeze_679 = None
    mul_1008: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_119);  primals_119 = None
    unsqueeze_681: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
    unsqueeze_682: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 2);  unsqueeze_681 = None
    unsqueeze_683: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 3);  unsqueeze_682 = None
    sub_196: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_674);  convolution_77 = unsqueeze_674 = None
    mul_1009: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_680);  sub_196 = unsqueeze_680 = None
    sub_197: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_66, mul_1009);  where_66 = mul_1009 = None
    sub_198: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_197, unsqueeze_677);  sub_197 = unsqueeze_677 = None
    mul_1010: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_683);  sub_198 = unsqueeze_683 = None
    mul_1011: "f32[360]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_178);  sum_66 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1010, add_367, primals_272, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1010 = add_367 = primals_272 = None
    getitem_312: "f32[8, 120, 14, 14]" = convolution_backward_46[0]
    getitem_313: "f32[360, 120, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_597: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_592, getitem_312);  add_592 = getitem_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_684: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
    unsqueeze_685: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 2);  unsqueeze_684 = None
    unsqueeze_686: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 3);  unsqueeze_685 = None
    sum_67: "f32[120]" = torch.ops.aten.sum.dim_IntList(add_597, [0, 2, 3])
    sub_199: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_686)
    mul_1012: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(add_597, sub_199);  sub_199 = None
    sum_68: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1012, [0, 2, 3]);  mul_1012 = None
    mul_1013: "f32[120]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    unsqueeze_687: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
    unsqueeze_688: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 2);  unsqueeze_687 = None
    unsqueeze_689: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 3);  unsqueeze_688 = None
    mul_1014: "f32[120]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    mul_1015: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_1016: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1014, mul_1015);  mul_1014 = mul_1015 = None
    unsqueeze_690: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1016, 0);  mul_1016 = None
    unsqueeze_691: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 2);  unsqueeze_690 = None
    unsqueeze_692: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 3);  unsqueeze_691 = None
    mul_1017: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_117);  primals_117 = None
    unsqueeze_693: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1017, 0);  mul_1017 = None
    unsqueeze_694: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 2);  unsqueeze_693 = None
    unsqueeze_695: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 3);  unsqueeze_694 = None
    sub_200: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_686);  convolution_76 = unsqueeze_686 = None
    mul_1018: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_692);  sub_200 = unsqueeze_692 = None
    sub_201: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(add_597, mul_1018);  mul_1018 = None
    sub_202: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_689);  sub_201 = unsqueeze_689 = None
    mul_1019: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_695);  sub_202 = unsqueeze_695 = None
    mul_1020: "f32[120]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_175);  sum_68 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1019, mul_462, primals_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1019 = mul_462 = primals_271 = None
    getitem_315: "f32[8, 360, 14, 14]" = convolution_backward_47[0]
    getitem_316: "f32[120, 360, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1021: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_315, div_54);  div_54 = None
    mul_1022: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_315, div_56);  getitem_315 = div_56 = None
    sum_69: "f32[8, 360, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1021, [2, 3], True);  mul_1021 = None
    gt_9: "b8[8, 360, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_75, -3.0)
    lt_38: "b8[8, 360, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_75, 3.0);  convolution_75 = None
    bitwise_and_9: "b8[8, 360, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_9, lt_38);  gt_9 = lt_38 = None
    mul_1023: "f32[8, 360, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, 0.16666666666666666);  sum_69 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_67: "f32[8, 360, 1, 1]" = torch.ops.aten.where.self(bitwise_and_9, mul_1023, scalar_tensor_38);  bitwise_and_9 = mul_1023 = scalar_tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(where_67, div_55, primals_269, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_67 = div_55 = primals_269 = None
    getitem_318: "f32[8, 32, 1, 1]" = convolution_backward_48[0]
    getitem_319: "f32[360, 32, 1, 1]" = convolution_backward_48[1]
    getitem_320: "f32[360]" = convolution_backward_48[2];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_39: "b8[8, 32, 1, 1]" = torch.ops.aten.lt.Scalar(clone_47, -3)
    le_29: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(clone_47, 3)
    div_134: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(clone_47, 3);  clone_47 = None
    add_598: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(div_134, 0.5);  div_134 = None
    mul_1024: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_318, add_598);  add_598 = None
    where_68: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_29, mul_1024, getitem_318);  le_29 = mul_1024 = getitem_318 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_69: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(lt_39, scalar_tensor_39, where_68);  lt_39 = scalar_tensor_39 = where_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(where_69, mean_8, primals_267, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_69 = mean_8 = primals_267 = None
    getitem_321: "f32[8, 360, 1, 1]" = convolution_backward_49[0]
    getitem_322: "f32[32, 360, 1, 1]" = convolution_backward_49[1]
    getitem_323: "f32[32]" = convolution_backward_49[2];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[8, 360, 14, 14]" = torch.ops.aten.expand.default(getitem_321, [8, 360, 14, 14]);  getitem_321 = None
    div_135: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_599: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_1022, div_135);  mul_1022 = div_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_40: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_46, -3)
    le_30: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_46, 3)
    div_136: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_46, 3);  clone_46 = None
    add_600: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_136, 0.5);  div_136 = None
    mul_1025: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_599, add_600);  add_600 = None
    where_70: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_30, mul_1025, add_599);  le_30 = mul_1025 = add_599 = None
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_71: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_40, scalar_tensor_40, where_70);  lt_40 = scalar_tensor_40 = where_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_696: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
    unsqueeze_697: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 2);  unsqueeze_696 = None
    unsqueeze_698: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 3);  unsqueeze_697 = None
    sum_70: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_203: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_698)
    mul_1026: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_71, sub_203);  sub_203 = None
    sum_71: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1026, [0, 2, 3]);  mul_1026 = None
    mul_1027: "f32[360]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_699: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1027, 0);  mul_1027 = None
    unsqueeze_700: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 2);  unsqueeze_699 = None
    unsqueeze_701: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 3);  unsqueeze_700 = None
    mul_1028: "f32[360]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_1029: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_1030: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1028, mul_1029);  mul_1028 = mul_1029 = None
    unsqueeze_702: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1030, 0);  mul_1030 = None
    unsqueeze_703: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 2);  unsqueeze_702 = None
    unsqueeze_704: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 3);  unsqueeze_703 = None
    mul_1031: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_115);  primals_115 = None
    unsqueeze_705: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1031, 0);  mul_1031 = None
    unsqueeze_706: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 2);  unsqueeze_705 = None
    unsqueeze_707: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 3);  unsqueeze_706 = None
    sub_204: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_698);  convolution_73 = unsqueeze_698 = None
    mul_1032: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_704);  sub_204 = unsqueeze_704 = None
    sub_205: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_71, mul_1032);  where_71 = mul_1032 = None
    sub_206: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_701);  sub_205 = unsqueeze_701 = None
    mul_1033: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_707);  sub_206 = unsqueeze_707 = None
    mul_1034: "f32[360]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_172);  sum_71 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1033, div_53, primals_266, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False]);  mul_1033 = div_53 = primals_266 = None
    getitem_324: "f32[8, 360, 14, 14]" = convolution_backward_50[0]
    getitem_325: "f32[360, 1, 5, 5]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_41: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_45, -3)
    le_31: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_45, 3)
    div_137: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_45, 3);  clone_45 = None
    add_601: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_137, 0.5);  div_137 = None
    mul_1035: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_324, add_601);  add_601 = None
    where_72: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_31, mul_1035, getitem_324);  le_31 = mul_1035 = getitem_324 = None
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_73: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_41, scalar_tensor_41, where_72);  lt_41 = scalar_tensor_41 = where_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_708: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
    unsqueeze_709: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 2);  unsqueeze_708 = None
    unsqueeze_710: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 3);  unsqueeze_709 = None
    sum_72: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_207: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_710)
    mul_1036: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_73, sub_207);  sub_207 = None
    sum_73: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1036, [0, 2, 3]);  mul_1036 = None
    mul_1037: "f32[360]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_711: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1037, 0);  mul_1037 = None
    unsqueeze_712: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 2);  unsqueeze_711 = None
    unsqueeze_713: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 3);  unsqueeze_712 = None
    mul_1038: "f32[360]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_1039: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_1040: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1038, mul_1039);  mul_1038 = mul_1039 = None
    unsqueeze_714: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1040, 0);  mul_1040 = None
    unsqueeze_715: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 2);  unsqueeze_714 = None
    unsqueeze_716: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 3);  unsqueeze_715 = None
    mul_1041: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_113);  primals_113 = None
    unsqueeze_717: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1041, 0);  mul_1041 = None
    unsqueeze_718: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 2);  unsqueeze_717 = None
    unsqueeze_719: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 3);  unsqueeze_718 = None
    sub_208: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_710);  convolution_72 = unsqueeze_710 = None
    mul_1042: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_716);  sub_208 = unsqueeze_716 = None
    sub_209: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_73, mul_1042);  where_73 = mul_1042 = None
    sub_210: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_209, unsqueeze_713);  sub_209 = unsqueeze_713 = None
    mul_1043: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_719);  sub_210 = unsqueeze_719 = None
    mul_1044: "f32[360]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_169);  sum_73 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1043, add_347, primals_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1043 = add_347 = primals_265 = None
    getitem_327: "f32[8, 120, 14, 14]" = convolution_backward_51[0]
    getitem_328: "f32[360, 120, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_602: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_597, getitem_327);  add_597 = getitem_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_720: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_721: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 2);  unsqueeze_720 = None
    unsqueeze_722: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 3);  unsqueeze_721 = None
    sum_74: "f32[120]" = torch.ops.aten.sum.dim_IntList(add_602, [0, 2, 3])
    sub_211: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_722)
    mul_1045: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(add_602, sub_211);  sub_211 = None
    sum_75: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1045, [0, 2, 3]);  mul_1045 = None
    mul_1046: "f32[120]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_723: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
    unsqueeze_724: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 2);  unsqueeze_723 = None
    unsqueeze_725: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 3);  unsqueeze_724 = None
    mul_1047: "f32[120]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_1048: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_1049: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1047, mul_1048);  mul_1047 = mul_1048 = None
    unsqueeze_726: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1049, 0);  mul_1049 = None
    unsqueeze_727: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 2);  unsqueeze_726 = None
    unsqueeze_728: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 3);  unsqueeze_727 = None
    mul_1050: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_111);  primals_111 = None
    unsqueeze_729: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1050, 0);  mul_1050 = None
    unsqueeze_730: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 2);  unsqueeze_729 = None
    unsqueeze_731: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 3);  unsqueeze_730 = None
    sub_212: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_722);  convolution_71 = unsqueeze_722 = None
    mul_1051: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_728);  sub_212 = unsqueeze_728 = None
    sub_213: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(add_602, mul_1051);  mul_1051 = None
    sub_214: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_213, unsqueeze_725);  sub_213 = unsqueeze_725 = None
    mul_1052: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_731);  sub_214 = unsqueeze_731 = None
    mul_1053: "f32[120]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_166);  sum_75 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1052, mul_437, primals_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1052 = mul_437 = primals_264 = None
    getitem_330: "f32[8, 360, 14, 14]" = convolution_backward_52[0]
    getitem_331: "f32[120, 360, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1054: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_330, div_50);  div_50 = None
    mul_1055: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_330, div_52);  getitem_330 = div_52 = None
    sum_76: "f32[8, 360, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1054, [2, 3], True);  mul_1054 = None
    gt_10: "b8[8, 360, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_70, -3.0)
    lt_42: "b8[8, 360, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_70, 3.0);  convolution_70 = None
    bitwise_and_10: "b8[8, 360, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_10, lt_42);  gt_10 = lt_42 = None
    mul_1056: "f32[8, 360, 1, 1]" = torch.ops.aten.mul.Tensor(sum_76, 0.16666666666666666);  sum_76 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_74: "f32[8, 360, 1, 1]" = torch.ops.aten.where.self(bitwise_and_10, mul_1056, scalar_tensor_42);  bitwise_and_10 = mul_1056 = scalar_tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(where_74, div_51, primals_262, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_74 = div_51 = primals_262 = None
    getitem_333: "f32[8, 32, 1, 1]" = convolution_backward_53[0]
    getitem_334: "f32[360, 32, 1, 1]" = convolution_backward_53[1]
    getitem_335: "f32[360]" = convolution_backward_53[2];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_43: "b8[8, 32, 1, 1]" = torch.ops.aten.lt.Scalar(clone_44, -3)
    le_32: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(clone_44, 3)
    div_138: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(clone_44, 3);  clone_44 = None
    add_603: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(div_138, 0.5);  div_138 = None
    mul_1057: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_333, add_603);  add_603 = None
    where_75: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_32, mul_1057, getitem_333);  le_32 = mul_1057 = getitem_333 = None
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_76: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(lt_43, scalar_tensor_43, where_75);  lt_43 = scalar_tensor_43 = where_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(where_76, mean_7, primals_260, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_76 = mean_7 = primals_260 = None
    getitem_336: "f32[8, 360, 1, 1]" = convolution_backward_54[0]
    getitem_337: "f32[32, 360, 1, 1]" = convolution_backward_54[1]
    getitem_338: "f32[32]" = convolution_backward_54[2];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[8, 360, 14, 14]" = torch.ops.aten.expand.default(getitem_336, [8, 360, 14, 14]);  getitem_336 = None
    div_139: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_604: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_1055, div_139);  mul_1055 = div_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_44: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_43, -3)
    le_33: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_43, 3)
    div_140: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_43, 3);  clone_43 = None
    add_605: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_140, 0.5);  div_140 = None
    mul_1058: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_604, add_605);  add_605 = None
    where_77: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_33, mul_1058, add_604);  le_33 = mul_1058 = add_604 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_78: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_44, scalar_tensor_44, where_77);  lt_44 = scalar_tensor_44 = where_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_732: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
    unsqueeze_733: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 2);  unsqueeze_732 = None
    unsqueeze_734: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 3);  unsqueeze_733 = None
    sum_77: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_215: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_734)
    mul_1059: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_78, sub_215);  sub_215 = None
    sum_78: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1059, [0, 2, 3]);  mul_1059 = None
    mul_1060: "f32[360]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    unsqueeze_735: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1060, 0);  mul_1060 = None
    unsqueeze_736: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 2);  unsqueeze_735 = None
    unsqueeze_737: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 3);  unsqueeze_736 = None
    mul_1061: "f32[360]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    mul_1062: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_1063: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1061, mul_1062);  mul_1061 = mul_1062 = None
    unsqueeze_738: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1063, 0);  mul_1063 = None
    unsqueeze_739: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 2);  unsqueeze_738 = None
    unsqueeze_740: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 3);  unsqueeze_739 = None
    mul_1064: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_109);  primals_109 = None
    unsqueeze_741: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
    unsqueeze_742: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 2);  unsqueeze_741 = None
    unsqueeze_743: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 3);  unsqueeze_742 = None
    sub_216: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_734);  convolution_68 = unsqueeze_734 = None
    mul_1065: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_740);  sub_216 = unsqueeze_740 = None
    sub_217: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_78, mul_1065);  where_78 = mul_1065 = None
    sub_218: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_217, unsqueeze_737);  sub_217 = unsqueeze_737 = None
    mul_1066: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_743);  sub_218 = unsqueeze_743 = None
    mul_1067: "f32[360]" = torch.ops.aten.mul.Tensor(sum_78, squeeze_163);  sum_78 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1066, div_49, primals_259, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False]);  mul_1066 = div_49 = primals_259 = None
    getitem_339: "f32[8, 360, 14, 14]" = convolution_backward_55[0]
    getitem_340: "f32[360, 1, 5, 5]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_45: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_42, -3)
    le_34: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_42, 3)
    div_141: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_42, 3);  clone_42 = None
    add_606: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_141, 0.5);  div_141 = None
    mul_1068: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_339, add_606);  add_606 = None
    where_79: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_34, mul_1068, getitem_339);  le_34 = mul_1068 = getitem_339 = None
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_80: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_45, scalar_tensor_45, where_79);  lt_45 = scalar_tensor_45 = where_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_744: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
    unsqueeze_745: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 2);  unsqueeze_744 = None
    unsqueeze_746: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 3);  unsqueeze_745 = None
    sum_79: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_219: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_746)
    mul_1069: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_80, sub_219);  sub_219 = None
    sum_80: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1069, [0, 2, 3]);  mul_1069 = None
    mul_1070: "f32[360]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    unsqueeze_747: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1070, 0);  mul_1070 = None
    unsqueeze_748: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 2);  unsqueeze_747 = None
    unsqueeze_749: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 3);  unsqueeze_748 = None
    mul_1071: "f32[360]" = torch.ops.aten.mul.Tensor(sum_80, 0.0006377551020408163)
    mul_1072: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_1073: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1071, mul_1072);  mul_1071 = mul_1072 = None
    unsqueeze_750: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
    unsqueeze_751: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 2);  unsqueeze_750 = None
    unsqueeze_752: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 3);  unsqueeze_751 = None
    mul_1074: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_107);  primals_107 = None
    unsqueeze_753: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1074, 0);  mul_1074 = None
    unsqueeze_754: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 2);  unsqueeze_753 = None
    unsqueeze_755: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 3);  unsqueeze_754 = None
    sub_220: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_746);  convolution_67 = unsqueeze_746 = None
    mul_1075: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_752);  sub_220 = unsqueeze_752 = None
    sub_221: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_80, mul_1075);  where_80 = mul_1075 = None
    sub_222: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_749);  sub_221 = unsqueeze_749 = None
    mul_1076: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_755);  sub_222 = unsqueeze_755 = None
    mul_1077: "f32[360]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_160);  sum_80 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1076, add_327, primals_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1076 = add_327 = primals_258 = None
    getitem_342: "f32[8, 120, 14, 14]" = convolution_backward_56[0]
    getitem_343: "f32[360, 120, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_607: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_602, getitem_342);  add_602 = getitem_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_756: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
    unsqueeze_757: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 2);  unsqueeze_756 = None
    unsqueeze_758: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 3);  unsqueeze_757 = None
    sum_81: "f32[120]" = torch.ops.aten.sum.dim_IntList(add_607, [0, 2, 3])
    sub_223: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_758)
    mul_1078: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(add_607, sub_223);  sub_223 = None
    sum_82: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1078, [0, 2, 3]);  mul_1078 = None
    mul_1079: "f32[120]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    unsqueeze_759: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1079, 0);  mul_1079 = None
    unsqueeze_760: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 2);  unsqueeze_759 = None
    unsqueeze_761: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 3);  unsqueeze_760 = None
    mul_1080: "f32[120]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    mul_1081: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_1082: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1080, mul_1081);  mul_1080 = mul_1081 = None
    unsqueeze_762: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
    unsqueeze_763: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 2);  unsqueeze_762 = None
    unsqueeze_764: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 3);  unsqueeze_763 = None
    mul_1083: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_105);  primals_105 = None
    unsqueeze_765: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1083, 0);  mul_1083 = None
    unsqueeze_766: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 2);  unsqueeze_765 = None
    unsqueeze_767: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 3);  unsqueeze_766 = None
    sub_224: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_758);  convolution_66 = unsqueeze_758 = None
    mul_1084: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_764);  sub_224 = unsqueeze_764 = None
    sub_225: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(add_607, mul_1084);  mul_1084 = None
    sub_226: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_225, unsqueeze_761);  sub_225 = unsqueeze_761 = None
    mul_1085: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_767);  sub_226 = unsqueeze_767 = None
    mul_1086: "f32[120]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_157);  sum_82 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1085, mul_412, primals_257, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1085 = mul_412 = primals_257 = None
    getitem_345: "f32[8, 360, 14, 14]" = convolution_backward_57[0]
    getitem_346: "f32[120, 360, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1087: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_345, div_46);  div_46 = None
    mul_1088: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_345, div_48);  getitem_345 = div_48 = None
    sum_83: "f32[8, 360, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1087, [2, 3], True);  mul_1087 = None
    gt_11: "b8[8, 360, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_65, -3.0)
    lt_46: "b8[8, 360, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_65, 3.0);  convolution_65 = None
    bitwise_and_11: "b8[8, 360, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_11, lt_46);  gt_11 = lt_46 = None
    mul_1089: "f32[8, 360, 1, 1]" = torch.ops.aten.mul.Tensor(sum_83, 0.16666666666666666);  sum_83 = None
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_81: "f32[8, 360, 1, 1]" = torch.ops.aten.where.self(bitwise_and_11, mul_1089, scalar_tensor_46);  bitwise_and_11 = mul_1089 = scalar_tensor_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(where_81, div_47, primals_255, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_81 = div_47 = primals_255 = None
    getitem_348: "f32[8, 32, 1, 1]" = convolution_backward_58[0]
    getitem_349: "f32[360, 32, 1, 1]" = convolution_backward_58[1]
    getitem_350: "f32[360]" = convolution_backward_58[2];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_47: "b8[8, 32, 1, 1]" = torch.ops.aten.lt.Scalar(clone_41, -3)
    le_35: "b8[8, 32, 1, 1]" = torch.ops.aten.le.Scalar(clone_41, 3)
    div_142: "f32[8, 32, 1, 1]" = torch.ops.aten.div.Tensor(clone_41, 3);  clone_41 = None
    add_608: "f32[8, 32, 1, 1]" = torch.ops.aten.add.Tensor(div_142, 0.5);  div_142 = None
    mul_1090: "f32[8, 32, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_348, add_608);  add_608 = None
    where_82: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(le_35, mul_1090, getitem_348);  le_35 = mul_1090 = getitem_348 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_83: "f32[8, 32, 1, 1]" = torch.ops.aten.where.self(lt_47, scalar_tensor_47, where_82);  lt_47 = scalar_tensor_47 = where_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(where_83, mean_6, primals_253, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_83 = mean_6 = primals_253 = None
    getitem_351: "f32[8, 360, 1, 1]" = convolution_backward_59[0]
    getitem_352: "f32[32, 360, 1, 1]" = convolution_backward_59[1]
    getitem_353: "f32[32]" = convolution_backward_59[2];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[8, 360, 14, 14]" = torch.ops.aten.expand.default(getitem_351, [8, 360, 14, 14]);  getitem_351 = None
    div_143: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Scalar(expand_12, 196);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_609: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_1088, div_143);  mul_1088 = div_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_48: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_40, -3)
    le_36: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_40, 3)
    div_144: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_40, 3);  clone_40 = None
    add_610: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_144, 0.5);  div_144 = None
    mul_1091: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_609, add_610);  add_610 = None
    where_84: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_36, mul_1091, add_609);  le_36 = mul_1091 = add_609 = None
    scalar_tensor_48: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_85: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_48, scalar_tensor_48, where_84);  lt_48 = scalar_tensor_48 = where_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_768: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_769: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 2);  unsqueeze_768 = None
    unsqueeze_770: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 3);  unsqueeze_769 = None
    sum_84: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_227: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_770)
    mul_1092: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_85, sub_227);  sub_227 = None
    sum_85: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1092, [0, 2, 3]);  mul_1092 = None
    mul_1093: "f32[360]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    unsqueeze_771: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1093, 0);  mul_1093 = None
    unsqueeze_772: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 2);  unsqueeze_771 = None
    unsqueeze_773: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 3);  unsqueeze_772 = None
    mul_1094: "f32[360]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    mul_1095: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_1096: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1094, mul_1095);  mul_1094 = mul_1095 = None
    unsqueeze_774: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1096, 0);  mul_1096 = None
    unsqueeze_775: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 2);  unsqueeze_774 = None
    unsqueeze_776: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 3);  unsqueeze_775 = None
    mul_1097: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_777: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1097, 0);  mul_1097 = None
    unsqueeze_778: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 2);  unsqueeze_777 = None
    unsqueeze_779: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 3);  unsqueeze_778 = None
    sub_228: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_770);  convolution_63 = unsqueeze_770 = None
    mul_1098: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_776);  sub_228 = unsqueeze_776 = None
    sub_229: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_85, mul_1098);  where_85 = mul_1098 = None
    sub_230: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_229, unsqueeze_773);  sub_229 = unsqueeze_773 = None
    mul_1099: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_779);  sub_230 = unsqueeze_779 = None
    mul_1100: "f32[360]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_154);  sum_85 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1099, div_45, primals_252, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 360, [True, True, False]);  mul_1099 = div_45 = primals_252 = None
    getitem_354: "f32[8, 360, 14, 14]" = convolution_backward_60[0]
    getitem_355: "f32[360, 1, 5, 5]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_49: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_39, -3)
    le_37: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_39, 3)
    div_145: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_39, 3);  clone_39 = None
    add_611: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_145, 0.5);  div_145 = None
    mul_1101: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_354, add_611);  add_611 = None
    where_86: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_37, mul_1101, getitem_354);  le_37 = mul_1101 = getitem_354 = None
    scalar_tensor_49: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_87: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_49, scalar_tensor_49, where_86);  lt_49 = scalar_tensor_49 = where_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_780: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
    unsqueeze_781: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 2);  unsqueeze_780 = None
    unsqueeze_782: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 3);  unsqueeze_781 = None
    sum_86: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_231: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_782)
    mul_1102: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_87, sub_231);  sub_231 = None
    sum_87: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1102, [0, 2, 3]);  mul_1102 = None
    mul_1103: "f32[360]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006377551020408163)
    unsqueeze_783: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1103, 0);  mul_1103 = None
    unsqueeze_784: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 2);  unsqueeze_783 = None
    unsqueeze_785: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 3);  unsqueeze_784 = None
    mul_1104: "f32[360]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006377551020408163)
    mul_1105: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_1106: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1104, mul_1105);  mul_1104 = mul_1105 = None
    unsqueeze_786: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1106, 0);  mul_1106 = None
    unsqueeze_787: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 2);  unsqueeze_786 = None
    unsqueeze_788: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 3);  unsqueeze_787 = None
    mul_1107: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_789: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1107, 0);  mul_1107 = None
    unsqueeze_790: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 2);  unsqueeze_789 = None
    unsqueeze_791: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 3);  unsqueeze_790 = None
    sub_232: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_782);  convolution_62 = unsqueeze_782 = None
    mul_1108: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_788);  sub_232 = unsqueeze_788 = None
    sub_233: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_87, mul_1108);  where_87 = mul_1108 = None
    sub_234: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_233, unsqueeze_785);  sub_233 = unsqueeze_785 = None
    mul_1109: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_791);  sub_234 = unsqueeze_791 = None
    mul_1110: "f32[360]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_151);  sum_87 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1109, add_307, primals_251, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1109 = add_307 = primals_251 = None
    getitem_357: "f32[8, 120, 14, 14]" = convolution_backward_61[0]
    getitem_358: "f32[360, 120, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_612: "f32[8, 120, 14, 14]" = torch.ops.aten.add.Tensor(add_607, getitem_357);  add_607 = getitem_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_792: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
    unsqueeze_793: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 2);  unsqueeze_792 = None
    unsqueeze_794: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 3);  unsqueeze_793 = None
    sum_88: "f32[120]" = torch.ops.aten.sum.dim_IntList(add_612, [0, 2, 3])
    sub_235: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_794)
    mul_1111: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(add_612, sub_235);  sub_235 = None
    sum_89: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1111, [0, 2, 3]);  mul_1111 = None
    mul_1112: "f32[120]" = torch.ops.aten.mul.Tensor(sum_88, 0.0006377551020408163)
    unsqueeze_795: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
    unsqueeze_796: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 2);  unsqueeze_795 = None
    unsqueeze_797: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 3);  unsqueeze_796 = None
    mul_1113: "f32[120]" = torch.ops.aten.mul.Tensor(sum_89, 0.0006377551020408163)
    mul_1114: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_1115: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1113, mul_1114);  mul_1113 = mul_1114 = None
    unsqueeze_798: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1115, 0);  mul_1115 = None
    unsqueeze_799: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 2);  unsqueeze_798 = None
    unsqueeze_800: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 3);  unsqueeze_799 = None
    mul_1116: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_801: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1116, 0);  mul_1116 = None
    unsqueeze_802: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 2);  unsqueeze_801 = None
    unsqueeze_803: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 3);  unsqueeze_802 = None
    sub_236: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_794);  convolution_61 = unsqueeze_794 = None
    mul_1117: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_800);  sub_236 = unsqueeze_800 = None
    sub_237: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(add_612, mul_1117);  add_612 = mul_1117 = None
    sub_238: "f32[8, 120, 14, 14]" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_797);  sub_237 = unsqueeze_797 = None
    mul_1118: "f32[8, 120, 14, 14]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_803);  sub_238 = unsqueeze_803 = None
    mul_1119: "f32[120]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_148);  sum_89 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1118, mul_387, primals_250, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1118 = mul_387 = primals_250 = None
    getitem_360: "f32[8, 360, 14, 14]" = convolution_backward_62[0]
    getitem_361: "f32[120, 360, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1120: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_360, div_42);  div_42 = None
    mul_1121: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_360, div_44);  getitem_360 = div_44 = None
    sum_90: "f32[8, 360, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1120, [2, 3], True);  mul_1120 = None
    gt_12: "b8[8, 360, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_60, -3.0)
    lt_50: "b8[8, 360, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_60, 3.0);  convolution_60 = None
    bitwise_and_12: "b8[8, 360, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_12, lt_50);  gt_12 = lt_50 = None
    mul_1122: "f32[8, 360, 1, 1]" = torch.ops.aten.mul.Tensor(sum_90, 0.16666666666666666);  sum_90 = None
    scalar_tensor_50: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_88: "f32[8, 360, 1, 1]" = torch.ops.aten.where.self(bitwise_and_12, mul_1122, scalar_tensor_50);  bitwise_and_12 = mul_1122 = scalar_tensor_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(where_88, div_43, primals_248, [360], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_88 = div_43 = primals_248 = None
    getitem_363: "f32[8, 24, 1, 1]" = convolution_backward_63[0]
    getitem_364: "f32[360, 24, 1, 1]" = convolution_backward_63[1]
    getitem_365: "f32[360]" = convolution_backward_63[2];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_51: "b8[8, 24, 1, 1]" = torch.ops.aten.lt.Scalar(clone_38, -3)
    le_38: "b8[8, 24, 1, 1]" = torch.ops.aten.le.Scalar(clone_38, 3)
    div_146: "f32[8, 24, 1, 1]" = torch.ops.aten.div.Tensor(clone_38, 3);  clone_38 = None
    add_613: "f32[8, 24, 1, 1]" = torch.ops.aten.add.Tensor(div_146, 0.5);  div_146 = None
    mul_1123: "f32[8, 24, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_363, add_613);  add_613 = None
    where_89: "f32[8, 24, 1, 1]" = torch.ops.aten.where.self(le_38, mul_1123, getitem_363);  le_38 = mul_1123 = getitem_363 = None
    scalar_tensor_51: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_90: "f32[8, 24, 1, 1]" = torch.ops.aten.where.self(lt_51, scalar_tensor_51, where_89);  lt_51 = scalar_tensor_51 = where_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(where_90, mean_5, primals_246, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_90 = mean_5 = primals_246 = None
    getitem_366: "f32[8, 360, 1, 1]" = convolution_backward_64[0]
    getitem_367: "f32[24, 360, 1, 1]" = convolution_backward_64[1]
    getitem_368: "f32[24]" = convolution_backward_64[2];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_13: "f32[8, 360, 14, 14]" = torch.ops.aten.expand.default(getitem_366, [8, 360, 14, 14]);  getitem_366 = None
    div_147: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Scalar(expand_13, 196);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_614: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(mul_1121, div_147);  mul_1121 = div_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_52: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_37, -3)
    le_39: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_37, 3)
    div_148: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_37, 3);  clone_37 = None
    add_615: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_148, 0.5);  div_148 = None
    mul_1124: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(add_614, add_615);  add_615 = None
    where_91: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_39, mul_1124, add_614);  le_39 = mul_1124 = add_614 = None
    scalar_tensor_52: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_92: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_52, scalar_tensor_52, where_91);  lt_52 = scalar_tensor_52 = where_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_804: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
    unsqueeze_805: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 2);  unsqueeze_804 = None
    unsqueeze_806: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 3);  unsqueeze_805 = None
    sum_91: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_239: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_806)
    mul_1125: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_92, sub_239);  sub_239 = None
    sum_92: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1125, [0, 2, 3]);  mul_1125 = None
    mul_1126: "f32[360]" = torch.ops.aten.mul.Tensor(sum_91, 0.0006377551020408163)
    unsqueeze_807: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1126, 0);  mul_1126 = None
    unsqueeze_808: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 2);  unsqueeze_807 = None
    unsqueeze_809: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 3);  unsqueeze_808 = None
    mul_1127: "f32[360]" = torch.ops.aten.mul.Tensor(sum_92, 0.0006377551020408163)
    mul_1128: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_1129: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1127, mul_1128);  mul_1127 = mul_1128 = None
    unsqueeze_810: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1129, 0);  mul_1129 = None
    unsqueeze_811: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 2);  unsqueeze_810 = None
    unsqueeze_812: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 3);  unsqueeze_811 = None
    mul_1130: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_813: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1130, 0);  mul_1130 = None
    unsqueeze_814: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 2);  unsqueeze_813 = None
    unsqueeze_815: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 3);  unsqueeze_814 = None
    sub_240: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_806);  convolution_58 = unsqueeze_806 = None
    mul_1131: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_812);  sub_240 = unsqueeze_812 = None
    sub_241: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_92, mul_1131);  where_92 = mul_1131 = None
    sub_242: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_241, unsqueeze_809);  sub_241 = unsqueeze_809 = None
    mul_1132: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_815);  sub_242 = unsqueeze_815 = None
    mul_1133: "f32[360]" = torch.ops.aten.mul.Tensor(sum_92, squeeze_145);  sum_92 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1132, div_41, primals_245, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 360, [True, True, False]);  mul_1132 = div_41 = primals_245 = None
    getitem_369: "f32[8, 360, 14, 14]" = convolution_backward_65[0]
    getitem_370: "f32[360, 1, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_53: "b8[8, 360, 14, 14]" = torch.ops.aten.lt.Scalar(clone_36, -3)
    le_40: "b8[8, 360, 14, 14]" = torch.ops.aten.le.Scalar(clone_36, 3)
    div_149: "f32[8, 360, 14, 14]" = torch.ops.aten.div.Tensor(clone_36, 3);  clone_36 = None
    add_616: "f32[8, 360, 14, 14]" = torch.ops.aten.add.Tensor(div_149, 0.5);  div_149 = None
    mul_1134: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_369, add_616);  add_616 = None
    where_93: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(le_40, mul_1134, getitem_369);  le_40 = mul_1134 = getitem_369 = None
    scalar_tensor_53: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_94: "f32[8, 360, 14, 14]" = torch.ops.aten.where.self(lt_53, scalar_tensor_53, where_93);  lt_53 = scalar_tensor_53 = where_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_816: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_817: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 2);  unsqueeze_816 = None
    unsqueeze_818: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 3);  unsqueeze_817 = None
    sum_93: "f32[360]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_243: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_818)
    mul_1135: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(where_94, sub_243);  sub_243 = None
    sum_94: "f32[360]" = torch.ops.aten.sum.dim_IntList(mul_1135, [0, 2, 3]);  mul_1135 = None
    mul_1136: "f32[360]" = torch.ops.aten.mul.Tensor(sum_93, 0.0006377551020408163)
    unsqueeze_819: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1136, 0);  mul_1136 = None
    unsqueeze_820: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 2);  unsqueeze_819 = None
    unsqueeze_821: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 3);  unsqueeze_820 = None
    mul_1137: "f32[360]" = torch.ops.aten.mul.Tensor(sum_94, 0.0006377551020408163)
    mul_1138: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_1139: "f32[360]" = torch.ops.aten.mul.Tensor(mul_1137, mul_1138);  mul_1137 = mul_1138 = None
    unsqueeze_822: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1139, 0);  mul_1139 = None
    unsqueeze_823: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 2);  unsqueeze_822 = None
    unsqueeze_824: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 3);  unsqueeze_823 = None
    mul_1140: "f32[360]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_825: "f32[1, 360]" = torch.ops.aten.unsqueeze.default(mul_1140, 0);  mul_1140 = None
    unsqueeze_826: "f32[1, 360, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 2);  unsqueeze_825 = None
    unsqueeze_827: "f32[1, 360, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 3);  unsqueeze_826 = None
    sub_244: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_818);  convolution_57 = unsqueeze_818 = None
    mul_1141: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_824);  sub_244 = unsqueeze_824 = None
    sub_245: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(where_94, mul_1141);  where_94 = mul_1141 = None
    sub_246: "f32[8, 360, 14, 14]" = torch.ops.aten.sub.Tensor(sub_245, unsqueeze_821);  sub_245 = unsqueeze_821 = None
    mul_1142: "f32[8, 360, 14, 14]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_827);  sub_246 = unsqueeze_827 = None
    mul_1143: "f32[360]" = torch.ops.aten.mul.Tensor(sum_94, squeeze_142);  sum_94 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1142, add_288, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1142 = add_288 = primals_244 = None
    getitem_372: "f32[8, 72, 14, 14]" = convolution_backward_66[0]
    getitem_373: "f32[360, 72, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_828: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
    unsqueeze_829: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 2);  unsqueeze_828 = None
    unsqueeze_830: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 3);  unsqueeze_829 = None
    sum_95: "f32[72]" = torch.ops.aten.sum.dim_IntList(getitem_372, [0, 2, 3])
    sub_247: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_830)
    mul_1144: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_372, sub_247);  sub_247 = None
    sum_96: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_1144, [0, 2, 3]);  mul_1144 = None
    mul_1145: "f32[72]" = torch.ops.aten.mul.Tensor(sum_95, 0.0006377551020408163)
    unsqueeze_831: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1145, 0);  mul_1145 = None
    unsqueeze_832: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 2);  unsqueeze_831 = None
    unsqueeze_833: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 3);  unsqueeze_832 = None
    mul_1146: "f32[72]" = torch.ops.aten.mul.Tensor(sum_96, 0.0006377551020408163)
    mul_1147: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_1148: "f32[72]" = torch.ops.aten.mul.Tensor(mul_1146, mul_1147);  mul_1146 = mul_1147 = None
    unsqueeze_834: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1148, 0);  mul_1148 = None
    unsqueeze_835: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 2);  unsqueeze_834 = None
    unsqueeze_836: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 3);  unsqueeze_835 = None
    mul_1149: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_837: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1149, 0);  mul_1149 = None
    unsqueeze_838: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 2);  unsqueeze_837 = None
    unsqueeze_839: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 3);  unsqueeze_838 = None
    sub_248: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_830);  convolution_56 = unsqueeze_830 = None
    mul_1150: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_836);  sub_248 = unsqueeze_836 = None
    sub_249: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_372, mul_1150);  mul_1150 = None
    sub_250: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(sub_249, unsqueeze_833);  sub_249 = unsqueeze_833 = None
    mul_1151: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_839);  sub_250 = unsqueeze_839 = None
    mul_1152: "f32[72]" = torch.ops.aten.mul.Tensor(sum_96, squeeze_139);  sum_96 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1151, div_40, primals_243, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1151 = div_40 = primals_243 = None
    getitem_375: "f32[8, 216, 14, 14]" = convolution_backward_67[0]
    getitem_376: "f32[72, 216, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_54: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_35, -3)
    le_41: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_35, 3)
    div_150: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_35, 3);  clone_35 = None
    add_617: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_150, 0.5);  div_150 = None
    mul_1153: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_375, add_617);  add_617 = None
    where_95: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_41, mul_1153, getitem_375);  le_41 = mul_1153 = getitem_375 = None
    scalar_tensor_54: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_96: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_54, scalar_tensor_54, where_95);  lt_54 = scalar_tensor_54 = where_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_840: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
    unsqueeze_841: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 2);  unsqueeze_840 = None
    unsqueeze_842: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 3);  unsqueeze_841 = None
    sum_97: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_251: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_842)
    mul_1154: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_96, sub_251);  sub_251 = None
    sum_98: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1154, [0, 2, 3]);  mul_1154 = None
    mul_1155: "f32[216]" = torch.ops.aten.mul.Tensor(sum_97, 0.0006377551020408163)
    unsqueeze_843: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1155, 0);  mul_1155 = None
    unsqueeze_844: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 2);  unsqueeze_843 = None
    unsqueeze_845: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 3);  unsqueeze_844 = None
    mul_1156: "f32[216]" = torch.ops.aten.mul.Tensor(sum_98, 0.0006377551020408163)
    mul_1157: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_1158: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1156, mul_1157);  mul_1156 = mul_1157 = None
    unsqueeze_846: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1158, 0);  mul_1158 = None
    unsqueeze_847: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 2);  unsqueeze_846 = None
    unsqueeze_848: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 3);  unsqueeze_847 = None
    mul_1159: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_849: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1159, 0);  mul_1159 = None
    unsqueeze_850: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 2);  unsqueeze_849 = None
    unsqueeze_851: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 3);  unsqueeze_850 = None
    sub_252: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_842);  convolution_55 = unsqueeze_842 = None
    mul_1160: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_848);  sub_252 = unsqueeze_848 = None
    sub_253: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_96, mul_1160);  where_96 = mul_1160 = None
    sub_254: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_253, unsqueeze_845);  sub_253 = unsqueeze_845 = None
    mul_1161: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_851);  sub_254 = unsqueeze_851 = None
    mul_1162: "f32[216]" = torch.ops.aten.mul.Tensor(sum_98, squeeze_136);  sum_98 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1161, div_39, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  mul_1161 = div_39 = primals_242 = None
    getitem_378: "f32[8, 216, 14, 14]" = convolution_backward_68[0]
    getitem_379: "f32[216, 1, 3, 3]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_55: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_34, -3)
    le_42: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_34, 3)
    div_151: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_34, 3);  clone_34 = None
    add_618: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_151, 0.5);  div_151 = None
    mul_1163: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_378, add_618);  add_618 = None
    where_97: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_42, mul_1163, getitem_378);  le_42 = mul_1163 = getitem_378 = None
    scalar_tensor_55: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_98: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_55, scalar_tensor_55, where_97);  lt_55 = scalar_tensor_55 = where_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_852: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
    unsqueeze_853: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 2);  unsqueeze_852 = None
    unsqueeze_854: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 3);  unsqueeze_853 = None
    sum_99: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_98, [0, 2, 3])
    sub_255: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_854)
    mul_1164: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_98, sub_255);  sub_255 = None
    sum_100: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1164, [0, 2, 3]);  mul_1164 = None
    mul_1165: "f32[216]" = torch.ops.aten.mul.Tensor(sum_99, 0.0006377551020408163)
    unsqueeze_855: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1165, 0);  mul_1165 = None
    unsqueeze_856: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 2);  unsqueeze_855 = None
    unsqueeze_857: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 3);  unsqueeze_856 = None
    mul_1166: "f32[216]" = torch.ops.aten.mul.Tensor(sum_100, 0.0006377551020408163)
    mul_1167: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_1168: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1166, mul_1167);  mul_1166 = mul_1167 = None
    unsqueeze_858: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1168, 0);  mul_1168 = None
    unsqueeze_859: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 2);  unsqueeze_858 = None
    unsqueeze_860: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 3);  unsqueeze_859 = None
    mul_1169: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_861: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1169, 0);  mul_1169 = None
    unsqueeze_862: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 2);  unsqueeze_861 = None
    unsqueeze_863: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 3);  unsqueeze_862 = None
    sub_256: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_854);  convolution_54 = unsqueeze_854 = None
    mul_1170: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_860);  sub_256 = unsqueeze_860 = None
    sub_257: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_98, mul_1170);  where_98 = mul_1170 = None
    sub_258: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_257, unsqueeze_857);  sub_257 = unsqueeze_857 = None
    mul_1171: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_863);  sub_258 = unsqueeze_863 = None
    mul_1172: "f32[216]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_133);  sum_100 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1171, add_270, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1171 = add_270 = primals_241 = None
    getitem_381: "f32[8, 72, 14, 14]" = convolution_backward_69[0]
    getitem_382: "f32[216, 72, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_619: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(getitem_372, getitem_381);  getitem_372 = getitem_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_864: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_865: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 2);  unsqueeze_864 = None
    unsqueeze_866: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 3);  unsqueeze_865 = None
    sum_101: "f32[72]" = torch.ops.aten.sum.dim_IntList(add_619, [0, 2, 3])
    sub_259: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_866)
    mul_1173: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(add_619, sub_259);  sub_259 = None
    sum_102: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_1173, [0, 2, 3]);  mul_1173 = None
    mul_1174: "f32[72]" = torch.ops.aten.mul.Tensor(sum_101, 0.0006377551020408163)
    unsqueeze_867: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1174, 0);  mul_1174 = None
    unsqueeze_868: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 2);  unsqueeze_867 = None
    unsqueeze_869: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 3);  unsqueeze_868 = None
    mul_1175: "f32[72]" = torch.ops.aten.mul.Tensor(sum_102, 0.0006377551020408163)
    mul_1176: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_1177: "f32[72]" = torch.ops.aten.mul.Tensor(mul_1175, mul_1176);  mul_1175 = mul_1176 = None
    unsqueeze_870: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1177, 0);  mul_1177 = None
    unsqueeze_871: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 2);  unsqueeze_870 = None
    unsqueeze_872: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 3);  unsqueeze_871 = None
    mul_1178: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_873: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1178, 0);  mul_1178 = None
    unsqueeze_874: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 2);  unsqueeze_873 = None
    unsqueeze_875: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 3);  unsqueeze_874 = None
    sub_260: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_866);  convolution_53 = unsqueeze_866 = None
    mul_1179: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_872);  sub_260 = unsqueeze_872 = None
    sub_261: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(add_619, mul_1179);  mul_1179 = None
    sub_262: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(sub_261, unsqueeze_869);  sub_261 = unsqueeze_869 = None
    mul_1180: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_875);  sub_262 = unsqueeze_875 = None
    mul_1181: "f32[72]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_130);  sum_102 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1180, div_38, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1180 = div_38 = primals_240 = None
    getitem_384: "f32[8, 216, 14, 14]" = convolution_backward_70[0]
    getitem_385: "f32[72, 216, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_56: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_33, -3)
    le_43: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_33, 3)
    div_152: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_33, 3);  clone_33 = None
    add_620: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_152, 0.5);  div_152 = None
    mul_1182: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_384, add_620);  add_620 = None
    where_99: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_43, mul_1182, getitem_384);  le_43 = mul_1182 = getitem_384 = None
    scalar_tensor_56: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_100: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_56, scalar_tensor_56, where_99);  lt_56 = scalar_tensor_56 = where_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_876: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
    unsqueeze_877: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 2);  unsqueeze_876 = None
    unsqueeze_878: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 3);  unsqueeze_877 = None
    sum_103: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_100, [0, 2, 3])
    sub_263: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_878)
    mul_1183: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_100, sub_263);  sub_263 = None
    sum_104: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1183, [0, 2, 3]);  mul_1183 = None
    mul_1184: "f32[216]" = torch.ops.aten.mul.Tensor(sum_103, 0.0006377551020408163)
    unsqueeze_879: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1184, 0);  mul_1184 = None
    unsqueeze_880: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 2);  unsqueeze_879 = None
    unsqueeze_881: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 3);  unsqueeze_880 = None
    mul_1185: "f32[216]" = torch.ops.aten.mul.Tensor(sum_104, 0.0006377551020408163)
    mul_1186: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_1187: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1185, mul_1186);  mul_1185 = mul_1186 = None
    unsqueeze_882: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1187, 0);  mul_1187 = None
    unsqueeze_883: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 2);  unsqueeze_882 = None
    unsqueeze_884: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 3);  unsqueeze_883 = None
    mul_1188: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_885: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1188, 0);  mul_1188 = None
    unsqueeze_886: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 2);  unsqueeze_885 = None
    unsqueeze_887: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 3);  unsqueeze_886 = None
    sub_264: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_878);  convolution_52 = unsqueeze_878 = None
    mul_1189: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_884);  sub_264 = unsqueeze_884 = None
    sub_265: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_100, mul_1189);  where_100 = mul_1189 = None
    sub_266: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_265, unsqueeze_881);  sub_265 = unsqueeze_881 = None
    mul_1190: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_887);  sub_266 = unsqueeze_887 = None
    mul_1191: "f32[216]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_127);  sum_104 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1190, div_37, primals_239, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  mul_1190 = div_37 = primals_239 = None
    getitem_387: "f32[8, 216, 14, 14]" = convolution_backward_71[0]
    getitem_388: "f32[216, 1, 3, 3]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_57: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_32, -3)
    le_44: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_32, 3)
    div_153: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_32, 3);  clone_32 = None
    add_621: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_153, 0.5);  div_153 = None
    mul_1192: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_387, add_621);  add_621 = None
    where_101: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_44, mul_1192, getitem_387);  le_44 = mul_1192 = getitem_387 = None
    scalar_tensor_57: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_102: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_57, scalar_tensor_57, where_101);  lt_57 = scalar_tensor_57 = where_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_888: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
    unsqueeze_889: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 2);  unsqueeze_888 = None
    unsqueeze_890: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 3);  unsqueeze_889 = None
    sum_105: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_102, [0, 2, 3])
    sub_267: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_890)
    mul_1193: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_102, sub_267);  sub_267 = None
    sum_106: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1193, [0, 2, 3]);  mul_1193 = None
    mul_1194: "f32[216]" = torch.ops.aten.mul.Tensor(sum_105, 0.0006377551020408163)
    unsqueeze_891: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1194, 0);  mul_1194 = None
    unsqueeze_892: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 2);  unsqueeze_891 = None
    unsqueeze_893: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 3);  unsqueeze_892 = None
    mul_1195: "f32[216]" = torch.ops.aten.mul.Tensor(sum_106, 0.0006377551020408163)
    mul_1196: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_1197: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1195, mul_1196);  mul_1195 = mul_1196 = None
    unsqueeze_894: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1197, 0);  mul_1197 = None
    unsqueeze_895: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 2);  unsqueeze_894 = None
    unsqueeze_896: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 3);  unsqueeze_895 = None
    mul_1198: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_897: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1198, 0);  mul_1198 = None
    unsqueeze_898: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 2);  unsqueeze_897 = None
    unsqueeze_899: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 3);  unsqueeze_898 = None
    sub_268: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_890);  convolution_51 = unsqueeze_890 = None
    mul_1199: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_896);  sub_268 = unsqueeze_896 = None
    sub_269: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_102, mul_1199);  where_102 = mul_1199 = None
    sub_270: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_269, unsqueeze_893);  sub_269 = unsqueeze_893 = None
    mul_1200: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_899);  sub_270 = unsqueeze_899 = None
    mul_1201: "f32[216]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_124);  sum_106 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1200, add_252, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1200 = add_252 = primals_238 = None
    getitem_390: "f32[8, 72, 14, 14]" = convolution_backward_72[0]
    getitem_391: "f32[216, 72, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_622: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(add_619, getitem_390);  add_619 = getitem_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_900: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
    unsqueeze_901: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 2);  unsqueeze_900 = None
    unsqueeze_902: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 3);  unsqueeze_901 = None
    sum_107: "f32[72]" = torch.ops.aten.sum.dim_IntList(add_622, [0, 2, 3])
    sub_271: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_902)
    mul_1202: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(add_622, sub_271);  sub_271 = None
    sum_108: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_1202, [0, 2, 3]);  mul_1202 = None
    mul_1203: "f32[72]" = torch.ops.aten.mul.Tensor(sum_107, 0.0006377551020408163)
    unsqueeze_903: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1203, 0);  mul_1203 = None
    unsqueeze_904: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 2);  unsqueeze_903 = None
    unsqueeze_905: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 3);  unsqueeze_904 = None
    mul_1204: "f32[72]" = torch.ops.aten.mul.Tensor(sum_108, 0.0006377551020408163)
    mul_1205: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_1206: "f32[72]" = torch.ops.aten.mul.Tensor(mul_1204, mul_1205);  mul_1204 = mul_1205 = None
    unsqueeze_906: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1206, 0);  mul_1206 = None
    unsqueeze_907: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 2);  unsqueeze_906 = None
    unsqueeze_908: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 3);  unsqueeze_907 = None
    mul_1207: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_909: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1207, 0);  mul_1207 = None
    unsqueeze_910: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 2);  unsqueeze_909 = None
    unsqueeze_911: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 3);  unsqueeze_910 = None
    sub_272: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_902);  convolution_50 = unsqueeze_902 = None
    mul_1208: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_908);  sub_272 = unsqueeze_908 = None
    sub_273: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(add_622, mul_1208);  mul_1208 = None
    sub_274: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(sub_273, unsqueeze_905);  sub_273 = unsqueeze_905 = None
    mul_1209: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_911);  sub_274 = unsqueeze_911 = None
    mul_1210: "f32[72]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_121);  sum_108 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1209, div_36, primals_237, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1209 = div_36 = primals_237 = None
    getitem_393: "f32[8, 216, 14, 14]" = convolution_backward_73[0]
    getitem_394: "f32[72, 216, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_58: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_31, -3)
    le_45: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_31, 3)
    div_154: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_31, 3);  clone_31 = None
    add_623: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_154, 0.5);  div_154 = None
    mul_1211: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_393, add_623);  add_623 = None
    where_103: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_45, mul_1211, getitem_393);  le_45 = mul_1211 = getitem_393 = None
    scalar_tensor_58: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_104: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_58, scalar_tensor_58, where_103);  lt_58 = scalar_tensor_58 = where_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_912: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_913: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 2);  unsqueeze_912 = None
    unsqueeze_914: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 3);  unsqueeze_913 = None
    sum_109: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_104, [0, 2, 3])
    sub_275: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_914)
    mul_1212: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_104, sub_275);  sub_275 = None
    sum_110: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1212, [0, 2, 3]);  mul_1212 = None
    mul_1213: "f32[216]" = torch.ops.aten.mul.Tensor(sum_109, 0.0006377551020408163)
    unsqueeze_915: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1213, 0);  mul_1213 = None
    unsqueeze_916: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 2);  unsqueeze_915 = None
    unsqueeze_917: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 3);  unsqueeze_916 = None
    mul_1214: "f32[216]" = torch.ops.aten.mul.Tensor(sum_110, 0.0006377551020408163)
    mul_1215: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_1216: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1214, mul_1215);  mul_1214 = mul_1215 = None
    unsqueeze_918: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1216, 0);  mul_1216 = None
    unsqueeze_919: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 2);  unsqueeze_918 = None
    unsqueeze_920: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 3);  unsqueeze_919 = None
    mul_1217: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_921: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1217, 0);  mul_1217 = None
    unsqueeze_922: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 2);  unsqueeze_921 = None
    unsqueeze_923: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 3);  unsqueeze_922 = None
    sub_276: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_914);  convolution_49 = unsqueeze_914 = None
    mul_1218: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_920);  sub_276 = unsqueeze_920 = None
    sub_277: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_104, mul_1218);  where_104 = mul_1218 = None
    sub_278: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_277, unsqueeze_917);  sub_277 = unsqueeze_917 = None
    mul_1219: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_923);  sub_278 = unsqueeze_923 = None
    mul_1220: "f32[216]" = torch.ops.aten.mul.Tensor(sum_110, squeeze_118);  sum_110 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1219, div_35, primals_236, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  mul_1219 = div_35 = primals_236 = None
    getitem_396: "f32[8, 216, 14, 14]" = convolution_backward_74[0]
    getitem_397: "f32[216, 1, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_59: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_30, -3)
    le_46: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_30, 3)
    div_155: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_30, 3);  clone_30 = None
    add_624: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_155, 0.5);  div_155 = None
    mul_1221: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_396, add_624);  add_624 = None
    where_105: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_46, mul_1221, getitem_396);  le_46 = mul_1221 = getitem_396 = None
    scalar_tensor_59: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_106: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_59, scalar_tensor_59, where_105);  lt_59 = scalar_tensor_59 = where_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_924: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
    unsqueeze_925: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 2);  unsqueeze_924 = None
    unsqueeze_926: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 3);  unsqueeze_925 = None
    sum_111: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_106, [0, 2, 3])
    sub_279: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_926)
    mul_1222: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_106, sub_279);  sub_279 = None
    sum_112: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1222, [0, 2, 3]);  mul_1222 = None
    mul_1223: "f32[216]" = torch.ops.aten.mul.Tensor(sum_111, 0.0006377551020408163)
    unsqueeze_927: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1223, 0);  mul_1223 = None
    unsqueeze_928: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 2);  unsqueeze_927 = None
    unsqueeze_929: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 3);  unsqueeze_928 = None
    mul_1224: "f32[216]" = torch.ops.aten.mul.Tensor(sum_112, 0.0006377551020408163)
    mul_1225: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_1226: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1224, mul_1225);  mul_1224 = mul_1225 = None
    unsqueeze_930: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1226, 0);  mul_1226 = None
    unsqueeze_931: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 2);  unsqueeze_930 = None
    unsqueeze_932: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 3);  unsqueeze_931 = None
    mul_1227: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_933: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1227, 0);  mul_1227 = None
    unsqueeze_934: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 2);  unsqueeze_933 = None
    unsqueeze_935: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 3);  unsqueeze_934 = None
    sub_280: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_926);  convolution_48 = unsqueeze_926 = None
    mul_1228: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_932);  sub_280 = unsqueeze_932 = None
    sub_281: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_106, mul_1228);  where_106 = mul_1228 = None
    sub_282: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_281, unsqueeze_929);  sub_281 = unsqueeze_929 = None
    mul_1229: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_935);  sub_282 = unsqueeze_935 = None
    mul_1230: "f32[216]" = torch.ops.aten.mul.Tensor(sum_112, squeeze_115);  sum_112 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1229, add_234, primals_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1229 = add_234 = primals_235 = None
    getitem_399: "f32[8, 72, 14, 14]" = convolution_backward_75[0]
    getitem_400: "f32[216, 72, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_625: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(add_622, getitem_399);  add_622 = getitem_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_936: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
    unsqueeze_937: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 2);  unsqueeze_936 = None
    unsqueeze_938: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 3);  unsqueeze_937 = None
    sum_113: "f32[72]" = torch.ops.aten.sum.dim_IntList(add_625, [0, 2, 3])
    sub_283: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_938)
    mul_1231: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(add_625, sub_283);  sub_283 = None
    sum_114: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_1231, [0, 2, 3]);  mul_1231 = None
    mul_1232: "f32[72]" = torch.ops.aten.mul.Tensor(sum_113, 0.0006377551020408163)
    unsqueeze_939: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1232, 0);  mul_1232 = None
    unsqueeze_940: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 2);  unsqueeze_939 = None
    unsqueeze_941: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 3);  unsqueeze_940 = None
    mul_1233: "f32[72]" = torch.ops.aten.mul.Tensor(sum_114, 0.0006377551020408163)
    mul_1234: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_1235: "f32[72]" = torch.ops.aten.mul.Tensor(mul_1233, mul_1234);  mul_1233 = mul_1234 = None
    unsqueeze_942: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1235, 0);  mul_1235 = None
    unsqueeze_943: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 2);  unsqueeze_942 = None
    unsqueeze_944: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 3);  unsqueeze_943 = None
    mul_1236: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_945: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1236, 0);  mul_1236 = None
    unsqueeze_946: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 2);  unsqueeze_945 = None
    unsqueeze_947: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 3);  unsqueeze_946 = None
    sub_284: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_938);  convolution_47 = unsqueeze_938 = None
    mul_1237: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_944);  sub_284 = unsqueeze_944 = None
    sub_285: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(add_625, mul_1237);  mul_1237 = None
    sub_286: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(sub_285, unsqueeze_941);  sub_285 = unsqueeze_941 = None
    mul_1238: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_947);  sub_286 = unsqueeze_947 = None
    mul_1239: "f32[72]" = torch.ops.aten.mul.Tensor(sum_114, squeeze_112);  sum_114 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1238, div_34, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1238 = div_34 = primals_234 = None
    getitem_402: "f32[8, 216, 14, 14]" = convolution_backward_76[0]
    getitem_403: "f32[72, 216, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_60: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_29, -3)
    le_47: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_29, 3)
    div_156: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_29, 3);  clone_29 = None
    add_626: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_156, 0.5);  div_156 = None
    mul_1240: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_402, add_626);  add_626 = None
    where_107: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_47, mul_1240, getitem_402);  le_47 = mul_1240 = getitem_402 = None
    scalar_tensor_60: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_108: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_60, scalar_tensor_60, where_107);  lt_60 = scalar_tensor_60 = where_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_948: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
    unsqueeze_949: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 2);  unsqueeze_948 = None
    unsqueeze_950: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 3);  unsqueeze_949 = None
    sum_115: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_108, [0, 2, 3])
    sub_287: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_950)
    mul_1241: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_108, sub_287);  sub_287 = None
    sum_116: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1241, [0, 2, 3]);  mul_1241 = None
    mul_1242: "f32[216]" = torch.ops.aten.mul.Tensor(sum_115, 0.0006377551020408163)
    unsqueeze_951: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1242, 0);  mul_1242 = None
    unsqueeze_952: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 2);  unsqueeze_951 = None
    unsqueeze_953: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 3);  unsqueeze_952 = None
    mul_1243: "f32[216]" = torch.ops.aten.mul.Tensor(sum_116, 0.0006377551020408163)
    mul_1244: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_1245: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1243, mul_1244);  mul_1243 = mul_1244 = None
    unsqueeze_954: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1245, 0);  mul_1245 = None
    unsqueeze_955: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 2);  unsqueeze_954 = None
    unsqueeze_956: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 3);  unsqueeze_955 = None
    mul_1246: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_957: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1246, 0);  mul_1246 = None
    unsqueeze_958: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_957, 2);  unsqueeze_957 = None
    unsqueeze_959: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 3);  unsqueeze_958 = None
    sub_288: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_950);  convolution_46 = unsqueeze_950 = None
    mul_1247: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_956);  sub_288 = unsqueeze_956 = None
    sub_289: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_108, mul_1247);  where_108 = mul_1247 = None
    sub_290: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_289, unsqueeze_953);  sub_289 = unsqueeze_953 = None
    mul_1248: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_959);  sub_290 = unsqueeze_959 = None
    mul_1249: "f32[216]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_109);  sum_116 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1248, div_33, primals_233, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 216, [True, True, False]);  mul_1248 = div_33 = primals_233 = None
    getitem_405: "f32[8, 216, 14, 14]" = convolution_backward_77[0]
    getitem_406: "f32[216, 1, 3, 3]" = convolution_backward_77[1];  convolution_backward_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_61: "b8[8, 216, 14, 14]" = torch.ops.aten.lt.Scalar(clone_28, -3)
    le_48: "b8[8, 216, 14, 14]" = torch.ops.aten.le.Scalar(clone_28, 3)
    div_157: "f32[8, 216, 14, 14]" = torch.ops.aten.div.Tensor(clone_28, 3);  clone_28 = None
    add_627: "f32[8, 216, 14, 14]" = torch.ops.aten.add.Tensor(div_157, 0.5);  div_157 = None
    mul_1250: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_405, add_627);  add_627 = None
    where_109: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(le_48, mul_1250, getitem_405);  le_48 = mul_1250 = getitem_405 = None
    scalar_tensor_61: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_110: "f32[8, 216, 14, 14]" = torch.ops.aten.where.self(lt_61, scalar_tensor_61, where_109);  lt_61 = scalar_tensor_61 = where_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_960: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_961: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 2);  unsqueeze_960 = None
    unsqueeze_962: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 3);  unsqueeze_961 = None
    sum_117: "f32[216]" = torch.ops.aten.sum.dim_IntList(where_110, [0, 2, 3])
    sub_291: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_962)
    mul_1251: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(where_110, sub_291);  sub_291 = None
    sum_118: "f32[216]" = torch.ops.aten.sum.dim_IntList(mul_1251, [0, 2, 3]);  mul_1251 = None
    mul_1252: "f32[216]" = torch.ops.aten.mul.Tensor(sum_117, 0.0006377551020408163)
    unsqueeze_963: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1252, 0);  mul_1252 = None
    unsqueeze_964: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 2);  unsqueeze_963 = None
    unsqueeze_965: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 3);  unsqueeze_964 = None
    mul_1253: "f32[216]" = torch.ops.aten.mul.Tensor(sum_118, 0.0006377551020408163)
    mul_1254: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_1255: "f32[216]" = torch.ops.aten.mul.Tensor(mul_1253, mul_1254);  mul_1253 = mul_1254 = None
    unsqueeze_966: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1255, 0);  mul_1255 = None
    unsqueeze_967: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 2);  unsqueeze_966 = None
    unsqueeze_968: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 3);  unsqueeze_967 = None
    mul_1256: "f32[216]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_969: "f32[1, 216]" = torch.ops.aten.unsqueeze.default(mul_1256, 0);  mul_1256 = None
    unsqueeze_970: "f32[1, 216, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 2);  unsqueeze_969 = None
    unsqueeze_971: "f32[1, 216, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 3);  unsqueeze_970 = None
    sub_292: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_962);  convolution_45 = unsqueeze_962 = None
    mul_1257: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_968);  sub_292 = unsqueeze_968 = None
    sub_293: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(where_110, mul_1257);  where_110 = mul_1257 = None
    sub_294: "f32[8, 216, 14, 14]" = torch.ops.aten.sub.Tensor(sub_293, unsqueeze_965);  sub_293 = unsqueeze_965 = None
    mul_1258: "f32[8, 216, 14, 14]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_971);  sub_294 = unsqueeze_971 = None
    mul_1259: "f32[216]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_106);  sum_118 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1258, add_216, primals_232, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1258 = add_216 = primals_232 = None
    getitem_408: "f32[8, 72, 14, 14]" = convolution_backward_78[0]
    getitem_409: "f32[216, 72, 1, 1]" = convolution_backward_78[1];  convolution_backward_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_628: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(add_625, getitem_408);  add_625 = getitem_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_972: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
    unsqueeze_973: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 2);  unsqueeze_972 = None
    unsqueeze_974: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 3);  unsqueeze_973 = None
    sum_119: "f32[72]" = torch.ops.aten.sum.dim_IntList(add_628, [0, 2, 3])
    sub_295: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_974)
    mul_1260: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(add_628, sub_295);  sub_295 = None
    sum_120: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_1260, [0, 2, 3]);  mul_1260 = None
    mul_1261: "f32[72]" = torch.ops.aten.mul.Tensor(sum_119, 0.0006377551020408163)
    unsqueeze_975: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1261, 0);  mul_1261 = None
    unsqueeze_976: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 2);  unsqueeze_975 = None
    unsqueeze_977: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 3);  unsqueeze_976 = None
    mul_1262: "f32[72]" = torch.ops.aten.mul.Tensor(sum_120, 0.0006377551020408163)
    mul_1263: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_1264: "f32[72]" = torch.ops.aten.mul.Tensor(mul_1262, mul_1263);  mul_1262 = mul_1263 = None
    unsqueeze_978: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1264, 0);  mul_1264 = None
    unsqueeze_979: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 2);  unsqueeze_978 = None
    unsqueeze_980: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 3);  unsqueeze_979 = None
    mul_1265: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_981: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_1265, 0);  mul_1265 = None
    unsqueeze_982: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 2);  unsqueeze_981 = None
    unsqueeze_983: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 3);  unsqueeze_982 = None
    sub_296: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_974);  convolution_44 = unsqueeze_974 = None
    mul_1266: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_980);  sub_296 = unsqueeze_980 = None
    sub_297: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(add_628, mul_1266);  add_628 = mul_1266 = None
    sub_298: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(sub_297, unsqueeze_977);  sub_297 = unsqueeze_977 = None
    mul_1267: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_983);  sub_298 = unsqueeze_983 = None
    mul_1268: "f32[72]" = torch.ops.aten.mul.Tensor(sum_120, squeeze_103);  sum_120 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1267, div_32, primals_231, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1267 = div_32 = primals_231 = None
    getitem_411: "f32[8, 200, 14, 14]" = convolution_backward_79[0]
    getitem_412: "f32[72, 200, 1, 1]" = convolution_backward_79[1];  convolution_backward_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_62: "b8[8, 200, 14, 14]" = torch.ops.aten.lt.Scalar(clone_27, -3)
    le_49: "b8[8, 200, 14, 14]" = torch.ops.aten.le.Scalar(clone_27, 3)
    div_158: "f32[8, 200, 14, 14]" = torch.ops.aten.div.Tensor(clone_27, 3);  clone_27 = None
    add_629: "f32[8, 200, 14, 14]" = torch.ops.aten.add.Tensor(div_158, 0.5);  div_158 = None
    mul_1269: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_411, add_629);  add_629 = None
    where_111: "f32[8, 200, 14, 14]" = torch.ops.aten.where.self(le_49, mul_1269, getitem_411);  le_49 = mul_1269 = getitem_411 = None
    scalar_tensor_62: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_112: "f32[8, 200, 14, 14]" = torch.ops.aten.where.self(lt_62, scalar_tensor_62, where_111);  lt_62 = scalar_tensor_62 = where_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_984: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
    unsqueeze_985: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 2);  unsqueeze_984 = None
    unsqueeze_986: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 3);  unsqueeze_985 = None
    sum_121: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_112, [0, 2, 3])
    sub_299: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_986)
    mul_1270: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(where_112, sub_299);  sub_299 = None
    sum_122: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_1270, [0, 2, 3]);  mul_1270 = None
    mul_1271: "f32[200]" = torch.ops.aten.mul.Tensor(sum_121, 0.0006377551020408163)
    unsqueeze_987: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1271, 0);  mul_1271 = None
    unsqueeze_988: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 2);  unsqueeze_987 = None
    unsqueeze_989: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 3);  unsqueeze_988 = None
    mul_1272: "f32[200]" = torch.ops.aten.mul.Tensor(sum_122, 0.0006377551020408163)
    mul_1273: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_1274: "f32[200]" = torch.ops.aten.mul.Tensor(mul_1272, mul_1273);  mul_1272 = mul_1273 = None
    unsqueeze_990: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1274, 0);  mul_1274 = None
    unsqueeze_991: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 2);  unsqueeze_990 = None
    unsqueeze_992: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 3);  unsqueeze_991 = None
    mul_1275: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_993: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1275, 0);  mul_1275 = None
    unsqueeze_994: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_993, 2);  unsqueeze_993 = None
    unsqueeze_995: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 3);  unsqueeze_994 = None
    sub_300: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_986);  convolution_43 = unsqueeze_986 = None
    mul_1276: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_992);  sub_300 = unsqueeze_992 = None
    sub_301: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(where_112, mul_1276);  where_112 = mul_1276 = None
    sub_302: "f32[8, 200, 14, 14]" = torch.ops.aten.sub.Tensor(sub_301, unsqueeze_989);  sub_301 = unsqueeze_989 = None
    mul_1277: "f32[8, 200, 14, 14]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_995);  sub_302 = unsqueeze_995 = None
    mul_1278: "f32[200]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_100);  sum_122 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1277, div_31, primals_230, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 200, [True, True, False]);  mul_1277 = div_31 = primals_230 = None
    getitem_414: "f32[8, 200, 28, 28]" = convolution_backward_80[0]
    getitem_415: "f32[200, 1, 5, 5]" = convolution_backward_80[1];  convolution_backward_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_63: "b8[8, 200, 28, 28]" = torch.ops.aten.lt.Scalar(clone_26, -3)
    le_50: "b8[8, 200, 28, 28]" = torch.ops.aten.le.Scalar(clone_26, 3)
    div_159: "f32[8, 200, 28, 28]" = torch.ops.aten.div.Tensor(clone_26, 3);  clone_26 = None
    add_630: "f32[8, 200, 28, 28]" = torch.ops.aten.add.Tensor(div_159, 0.5);  div_159 = None
    mul_1279: "f32[8, 200, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_414, add_630);  add_630 = None
    where_113: "f32[8, 200, 28, 28]" = torch.ops.aten.where.self(le_50, mul_1279, getitem_414);  le_50 = mul_1279 = getitem_414 = None
    scalar_tensor_63: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_114: "f32[8, 200, 28, 28]" = torch.ops.aten.where.self(lt_63, scalar_tensor_63, where_113);  lt_63 = scalar_tensor_63 = where_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_996: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
    unsqueeze_997: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 2);  unsqueeze_996 = None
    unsqueeze_998: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 3);  unsqueeze_997 = None
    sum_123: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_114, [0, 2, 3])
    sub_303: "f32[8, 200, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_998)
    mul_1280: "f32[8, 200, 28, 28]" = torch.ops.aten.mul.Tensor(where_114, sub_303);  sub_303 = None
    sum_124: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_1280, [0, 2, 3]);  mul_1280 = None
    mul_1281: "f32[200]" = torch.ops.aten.mul.Tensor(sum_123, 0.00015943877551020407)
    unsqueeze_999: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1281, 0);  mul_1281 = None
    unsqueeze_1000: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 2);  unsqueeze_999 = None
    unsqueeze_1001: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 3);  unsqueeze_1000 = None
    mul_1282: "f32[200]" = torch.ops.aten.mul.Tensor(sum_124, 0.00015943877551020407)
    mul_1283: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_1284: "f32[200]" = torch.ops.aten.mul.Tensor(mul_1282, mul_1283);  mul_1282 = mul_1283 = None
    unsqueeze_1002: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1284, 0);  mul_1284 = None
    unsqueeze_1003: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 2);  unsqueeze_1002 = None
    unsqueeze_1004: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 3);  unsqueeze_1003 = None
    mul_1285: "f32[200]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_1005: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_1285, 0);  mul_1285 = None
    unsqueeze_1006: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1005, 2);  unsqueeze_1005 = None
    unsqueeze_1007: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 3);  unsqueeze_1006 = None
    sub_304: "f32[8, 200, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_998);  convolution_42 = unsqueeze_998 = None
    mul_1286: "f32[8, 200, 28, 28]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_1004);  sub_304 = unsqueeze_1004 = None
    sub_305: "f32[8, 200, 28, 28]" = torch.ops.aten.sub.Tensor(where_114, mul_1286);  where_114 = mul_1286 = None
    sub_306: "f32[8, 200, 28, 28]" = torch.ops.aten.sub.Tensor(sub_305, unsqueeze_1001);  sub_305 = unsqueeze_1001 = None
    mul_1287: "f32[8, 200, 28, 28]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_1007);  sub_306 = unsqueeze_1007 = None
    mul_1288: "f32[200]" = torch.ops.aten.mul.Tensor(sum_124, squeeze_97);  sum_124 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_81 = torch.ops.aten.convolution_backward.default(mul_1287, add_199, primals_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1287 = add_199 = primals_229 = None
    getitem_417: "f32[8, 40, 28, 28]" = convolution_backward_81[0]
    getitem_418: "f32[200, 40, 1, 1]" = convolution_backward_81[1];  convolution_backward_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1008: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_1009: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 2);  unsqueeze_1008 = None
    unsqueeze_1010: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 3);  unsqueeze_1009 = None
    sum_125: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_417, [0, 2, 3])
    sub_307: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1010)
    mul_1289: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_417, sub_307);  sub_307 = None
    sum_126: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1289, [0, 2, 3]);  mul_1289 = None
    mul_1290: "f32[40]" = torch.ops.aten.mul.Tensor(sum_125, 0.00015943877551020407)
    unsqueeze_1011: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1290, 0);  mul_1290 = None
    unsqueeze_1012: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 2);  unsqueeze_1011 = None
    unsqueeze_1013: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 3);  unsqueeze_1012 = None
    mul_1291: "f32[40]" = torch.ops.aten.mul.Tensor(sum_126, 0.00015943877551020407)
    mul_1292: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_1293: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1291, mul_1292);  mul_1291 = mul_1292 = None
    unsqueeze_1014: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1293, 0);  mul_1293 = None
    unsqueeze_1015: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 2);  unsqueeze_1014 = None
    unsqueeze_1016: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 3);  unsqueeze_1015 = None
    mul_1294: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_1017: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1294, 0);  mul_1294 = None
    unsqueeze_1018: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1017, 2);  unsqueeze_1017 = None
    unsqueeze_1019: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 3);  unsqueeze_1018 = None
    sub_308: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_1010);  convolution_41 = unsqueeze_1010 = None
    mul_1295: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1016);  sub_308 = unsqueeze_1016 = None
    sub_309: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_417, mul_1295);  mul_1295 = None
    sub_310: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_309, unsqueeze_1013);  sub_309 = unsqueeze_1013 = None
    mul_1296: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_1019);  sub_310 = unsqueeze_1019 = None
    mul_1297: "f32[40]" = torch.ops.aten.mul.Tensor(sum_126, squeeze_94);  sum_126 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_82 = torch.ops.aten.convolution_backward.default(mul_1296, mul_247, primals_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1296 = mul_247 = primals_228 = None
    getitem_420: "f32[8, 120, 28, 28]" = convolution_backward_82[0]
    getitem_421: "f32[40, 120, 1, 1]" = convolution_backward_82[1];  convolution_backward_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1298: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_420, div_28);  div_28 = None
    mul_1299: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_420, div_30);  getitem_420 = div_30 = None
    sum_127: "f32[8, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1298, [2, 3], True);  mul_1298 = None
    gt_13: "b8[8, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_40, -3.0)
    lt_64: "b8[8, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_40, 3.0);  convolution_40 = None
    bitwise_and_13: "b8[8, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_13, lt_64);  gt_13 = lt_64 = None
    mul_1300: "f32[8, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_127, 0.16666666666666666);  sum_127 = None
    scalar_tensor_64: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_115: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_13, mul_1300, scalar_tensor_64);  bitwise_and_13 = mul_1300 = scalar_tensor_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_83 = torch.ops.aten.convolution_backward.default(where_115, div_29, primals_226, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_115 = div_29 = primals_226 = None
    getitem_423: "f32[8, 16, 1, 1]" = convolution_backward_83[0]
    getitem_424: "f32[120, 16, 1, 1]" = convolution_backward_83[1]
    getitem_425: "f32[120]" = convolution_backward_83[2];  convolution_backward_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_65: "b8[8, 16, 1, 1]" = torch.ops.aten.lt.Scalar(clone_25, -3)
    le_51: "b8[8, 16, 1, 1]" = torch.ops.aten.le.Scalar(clone_25, 3)
    div_160: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(clone_25, 3);  clone_25 = None
    add_631: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(div_160, 0.5);  div_160 = None
    mul_1301: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_423, add_631);  add_631 = None
    where_116: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(le_51, mul_1301, getitem_423);  le_51 = mul_1301 = getitem_423 = None
    scalar_tensor_65: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_117: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(lt_65, scalar_tensor_65, where_116);  lt_65 = scalar_tensor_65 = where_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_84 = torch.ops.aten.convolution_backward.default(where_117, mean_4, primals_224, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_117 = mean_4 = primals_224 = None
    getitem_426: "f32[8, 120, 1, 1]" = convolution_backward_84[0]
    getitem_427: "f32[16, 120, 1, 1]" = convolution_backward_84[1]
    getitem_428: "f32[16]" = convolution_backward_84[2];  convolution_backward_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_14: "f32[8, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_426, [8, 120, 28, 28]);  getitem_426 = None
    div_161: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_14, 784);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_632: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_1299, div_161);  mul_1299 = div_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_66: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_24, -3)
    le_52: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_24, 3)
    div_162: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_24, 3);  clone_24 = None
    add_633: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_162, 0.5);  div_162 = None
    mul_1302: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_632, add_633);  add_633 = None
    where_118: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_52, mul_1302, add_632);  le_52 = mul_1302 = add_632 = None
    scalar_tensor_66: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_119: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_66, scalar_tensor_66, where_118);  lt_66 = scalar_tensor_66 = where_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1020: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
    unsqueeze_1021: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 2);  unsqueeze_1020 = None
    unsqueeze_1022: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 3);  unsqueeze_1021 = None
    sum_128: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_119, [0, 2, 3])
    sub_311: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_1022)
    mul_1303: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_119, sub_311);  sub_311 = None
    sum_129: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1303, [0, 2, 3]);  mul_1303 = None
    mul_1304: "f32[120]" = torch.ops.aten.mul.Tensor(sum_128, 0.00015943877551020407)
    unsqueeze_1023: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1304, 0);  mul_1304 = None
    unsqueeze_1024: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 2);  unsqueeze_1023 = None
    unsqueeze_1025: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 3);  unsqueeze_1024 = None
    mul_1305: "f32[120]" = torch.ops.aten.mul.Tensor(sum_129, 0.00015943877551020407)
    mul_1306: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_1307: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1305, mul_1306);  mul_1305 = mul_1306 = None
    unsqueeze_1026: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1307, 0);  mul_1307 = None
    unsqueeze_1027: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 2);  unsqueeze_1026 = None
    unsqueeze_1028: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 3);  unsqueeze_1027 = None
    mul_1308: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_1029: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1308, 0);  mul_1308 = None
    unsqueeze_1030: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1029, 2);  unsqueeze_1029 = None
    unsqueeze_1031: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 3);  unsqueeze_1030 = None
    sub_312: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_1022);  convolution_38 = unsqueeze_1022 = None
    mul_1309: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1028);  sub_312 = unsqueeze_1028 = None
    sub_313: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_119, mul_1309);  where_119 = mul_1309 = None
    sub_314: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_313, unsqueeze_1025);  sub_313 = unsqueeze_1025 = None
    mul_1310: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_1031);  sub_314 = unsqueeze_1031 = None
    mul_1311: "f32[120]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_91);  sum_129 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_85 = torch.ops.aten.convolution_backward.default(mul_1310, div_27, primals_223, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1310 = div_27 = primals_223 = None
    getitem_429: "f32[8, 120, 28, 28]" = convolution_backward_85[0]
    getitem_430: "f32[120, 1, 5, 5]" = convolution_backward_85[1];  convolution_backward_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_67: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_23, -3)
    le_53: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_23, 3)
    div_163: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_23, 3);  clone_23 = None
    add_634: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_163, 0.5);  div_163 = None
    mul_1312: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_429, add_634);  add_634 = None
    where_120: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_53, mul_1312, getitem_429);  le_53 = mul_1312 = getitem_429 = None
    scalar_tensor_67: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_121: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_67, scalar_tensor_67, where_120);  lt_67 = scalar_tensor_67 = where_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1032: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
    unsqueeze_1033: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 2);  unsqueeze_1032 = None
    unsqueeze_1034: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 3);  unsqueeze_1033 = None
    sum_130: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_121, [0, 2, 3])
    sub_315: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1034)
    mul_1313: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_121, sub_315);  sub_315 = None
    sum_131: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1313, [0, 2, 3]);  mul_1313 = None
    mul_1314: "f32[120]" = torch.ops.aten.mul.Tensor(sum_130, 0.00015943877551020407)
    unsqueeze_1035: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1314, 0);  mul_1314 = None
    unsqueeze_1036: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 2);  unsqueeze_1035 = None
    unsqueeze_1037: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 3);  unsqueeze_1036 = None
    mul_1315: "f32[120]" = torch.ops.aten.mul.Tensor(sum_131, 0.00015943877551020407)
    mul_1316: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_1317: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1315, mul_1316);  mul_1315 = mul_1316 = None
    unsqueeze_1038: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1317, 0);  mul_1317 = None
    unsqueeze_1039: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 2);  unsqueeze_1038 = None
    unsqueeze_1040: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 3);  unsqueeze_1039 = None
    mul_1318: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_1041: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1318, 0);  mul_1318 = None
    unsqueeze_1042: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1041, 2);  unsqueeze_1041 = None
    unsqueeze_1043: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 3);  unsqueeze_1042 = None
    sub_316: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_1034);  convolution_37 = unsqueeze_1034 = None
    mul_1319: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1040);  sub_316 = unsqueeze_1040 = None
    sub_317: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_121, mul_1319);  where_121 = mul_1319 = None
    sub_318: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_317, unsqueeze_1037);  sub_317 = unsqueeze_1037 = None
    mul_1320: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_1043);  sub_318 = unsqueeze_1043 = None
    mul_1321: "f32[120]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_88);  sum_131 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_86 = torch.ops.aten.convolution_backward.default(mul_1320, add_179, primals_222, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1320 = add_179 = primals_222 = None
    getitem_432: "f32[8, 40, 28, 28]" = convolution_backward_86[0]
    getitem_433: "f32[120, 40, 1, 1]" = convolution_backward_86[1];  convolution_backward_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_635: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_417, getitem_432);  getitem_417 = getitem_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1044: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
    unsqueeze_1045: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, 2);  unsqueeze_1044 = None
    unsqueeze_1046: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 3);  unsqueeze_1045 = None
    sum_132: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_635, [0, 2, 3])
    sub_319: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1046)
    mul_1322: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_635, sub_319);  sub_319 = None
    sum_133: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1322, [0, 2, 3]);  mul_1322 = None
    mul_1323: "f32[40]" = torch.ops.aten.mul.Tensor(sum_132, 0.00015943877551020407)
    unsqueeze_1047: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1323, 0);  mul_1323 = None
    unsqueeze_1048: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 2);  unsqueeze_1047 = None
    unsqueeze_1049: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 3);  unsqueeze_1048 = None
    mul_1324: "f32[40]" = torch.ops.aten.mul.Tensor(sum_133, 0.00015943877551020407)
    mul_1325: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_1326: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1324, mul_1325);  mul_1324 = mul_1325 = None
    unsqueeze_1050: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1326, 0);  mul_1326 = None
    unsqueeze_1051: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 2);  unsqueeze_1050 = None
    unsqueeze_1052: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 3);  unsqueeze_1051 = None
    mul_1327: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_1053: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1327, 0);  mul_1327 = None
    unsqueeze_1054: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1053, 2);  unsqueeze_1053 = None
    unsqueeze_1055: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 3);  unsqueeze_1054 = None
    sub_320: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_1046);  convolution_36 = unsqueeze_1046 = None
    mul_1328: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1052);  sub_320 = unsqueeze_1052 = None
    sub_321: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_635, mul_1328);  mul_1328 = None
    sub_322: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_321, unsqueeze_1049);  sub_321 = unsqueeze_1049 = None
    mul_1329: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_1055);  sub_322 = unsqueeze_1055 = None
    mul_1330: "f32[40]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_85);  sum_133 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_87 = torch.ops.aten.convolution_backward.default(mul_1329, mul_222, primals_221, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1329 = mul_222 = primals_221 = None
    getitem_435: "f32[8, 120, 28, 28]" = convolution_backward_87[0]
    getitem_436: "f32[40, 120, 1, 1]" = convolution_backward_87[1];  convolution_backward_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1331: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_435, div_24);  div_24 = None
    mul_1332: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_435, div_26);  getitem_435 = div_26 = None
    sum_134: "f32[8, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1331, [2, 3], True);  mul_1331 = None
    gt_14: "b8[8, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_35, -3.0)
    lt_68: "b8[8, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_35, 3.0);  convolution_35 = None
    bitwise_and_14: "b8[8, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_14, lt_68);  gt_14 = lt_68 = None
    mul_1333: "f32[8, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_134, 0.16666666666666666);  sum_134 = None
    scalar_tensor_68: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_122: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_14, mul_1333, scalar_tensor_68);  bitwise_and_14 = mul_1333 = scalar_tensor_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_88 = torch.ops.aten.convolution_backward.default(where_122, div_25, primals_219, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_122 = div_25 = primals_219 = None
    getitem_438: "f32[8, 16, 1, 1]" = convolution_backward_88[0]
    getitem_439: "f32[120, 16, 1, 1]" = convolution_backward_88[1]
    getitem_440: "f32[120]" = convolution_backward_88[2];  convolution_backward_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_69: "b8[8, 16, 1, 1]" = torch.ops.aten.lt.Scalar(clone_22, -3)
    le_54: "b8[8, 16, 1, 1]" = torch.ops.aten.le.Scalar(clone_22, 3)
    div_164: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(clone_22, 3);  clone_22 = None
    add_636: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(div_164, 0.5);  div_164 = None
    mul_1334: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_438, add_636);  add_636 = None
    where_123: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(le_54, mul_1334, getitem_438);  le_54 = mul_1334 = getitem_438 = None
    scalar_tensor_69: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_124: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(lt_69, scalar_tensor_69, where_123);  lt_69 = scalar_tensor_69 = where_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_89 = torch.ops.aten.convolution_backward.default(where_124, mean_3, primals_217, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_124 = mean_3 = primals_217 = None
    getitem_441: "f32[8, 120, 1, 1]" = convolution_backward_89[0]
    getitem_442: "f32[16, 120, 1, 1]" = convolution_backward_89[1]
    getitem_443: "f32[16]" = convolution_backward_89[2];  convolution_backward_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_15: "f32[8, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_441, [8, 120, 28, 28]);  getitem_441 = None
    div_165: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_15, 784);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_637: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_1332, div_165);  mul_1332 = div_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_70: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_21, -3)
    le_55: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_21, 3)
    div_166: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_21, 3);  clone_21 = None
    add_638: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_166, 0.5);  div_166 = None
    mul_1335: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_637, add_638);  add_638 = None
    where_125: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_55, mul_1335, add_637);  le_55 = mul_1335 = add_637 = None
    scalar_tensor_70: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_126: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_70, scalar_tensor_70, where_125);  lt_70 = scalar_tensor_70 = where_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1056: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_1057: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, 2);  unsqueeze_1056 = None
    unsqueeze_1058: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 3);  unsqueeze_1057 = None
    sum_135: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_126, [0, 2, 3])
    sub_323: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1058)
    mul_1336: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_126, sub_323);  sub_323 = None
    sum_136: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1336, [0, 2, 3]);  mul_1336 = None
    mul_1337: "f32[120]" = torch.ops.aten.mul.Tensor(sum_135, 0.00015943877551020407)
    unsqueeze_1059: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1337, 0);  mul_1337 = None
    unsqueeze_1060: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 2);  unsqueeze_1059 = None
    unsqueeze_1061: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 3);  unsqueeze_1060 = None
    mul_1338: "f32[120]" = torch.ops.aten.mul.Tensor(sum_136, 0.00015943877551020407)
    mul_1339: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_1340: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1338, mul_1339);  mul_1338 = mul_1339 = None
    unsqueeze_1062: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1340, 0);  mul_1340 = None
    unsqueeze_1063: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 2);  unsqueeze_1062 = None
    unsqueeze_1064: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 3);  unsqueeze_1063 = None
    mul_1341: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_1065: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1341, 0);  mul_1341 = None
    unsqueeze_1066: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1065, 2);  unsqueeze_1065 = None
    unsqueeze_1067: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 3);  unsqueeze_1066 = None
    sub_324: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_1058);  convolution_33 = unsqueeze_1058 = None
    mul_1342: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1064);  sub_324 = unsqueeze_1064 = None
    sub_325: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_126, mul_1342);  where_126 = mul_1342 = None
    sub_326: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_325, unsqueeze_1061);  sub_325 = unsqueeze_1061 = None
    mul_1343: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_1067);  sub_326 = unsqueeze_1067 = None
    mul_1344: "f32[120]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_82);  sum_136 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_90 = torch.ops.aten.convolution_backward.default(mul_1343, div_23, primals_216, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1343 = div_23 = primals_216 = None
    getitem_444: "f32[8, 120, 28, 28]" = convolution_backward_90[0]
    getitem_445: "f32[120, 1, 5, 5]" = convolution_backward_90[1];  convolution_backward_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_71: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_20, -3)
    le_56: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_20, 3)
    div_167: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_20, 3);  clone_20 = None
    add_639: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_167, 0.5);  div_167 = None
    mul_1345: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_444, add_639);  add_639 = None
    where_127: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_56, mul_1345, getitem_444);  le_56 = mul_1345 = getitem_444 = None
    scalar_tensor_71: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_128: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_71, scalar_tensor_71, where_127);  lt_71 = scalar_tensor_71 = where_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1068: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
    unsqueeze_1069: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, 2);  unsqueeze_1068 = None
    unsqueeze_1070: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 3);  unsqueeze_1069 = None
    sum_137: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_128, [0, 2, 3])
    sub_327: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1070)
    mul_1346: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_128, sub_327);  sub_327 = None
    sum_138: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1346, [0, 2, 3]);  mul_1346 = None
    mul_1347: "f32[120]" = torch.ops.aten.mul.Tensor(sum_137, 0.00015943877551020407)
    unsqueeze_1071: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1347, 0);  mul_1347 = None
    unsqueeze_1072: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 2);  unsqueeze_1071 = None
    unsqueeze_1073: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 3);  unsqueeze_1072 = None
    mul_1348: "f32[120]" = torch.ops.aten.mul.Tensor(sum_138, 0.00015943877551020407)
    mul_1349: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_1350: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1348, mul_1349);  mul_1348 = mul_1349 = None
    unsqueeze_1074: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1350, 0);  mul_1350 = None
    unsqueeze_1075: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 2);  unsqueeze_1074 = None
    unsqueeze_1076: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 3);  unsqueeze_1075 = None
    mul_1351: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_1077: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1351, 0);  mul_1351 = None
    unsqueeze_1078: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1077, 2);  unsqueeze_1077 = None
    unsqueeze_1079: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 3);  unsqueeze_1078 = None
    sub_328: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_1070);  convolution_32 = unsqueeze_1070 = None
    mul_1352: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1076);  sub_328 = unsqueeze_1076 = None
    sub_329: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_128, mul_1352);  where_128 = mul_1352 = None
    sub_330: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_329, unsqueeze_1073);  sub_329 = unsqueeze_1073 = None
    mul_1353: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_1079);  sub_330 = unsqueeze_1079 = None
    mul_1354: "f32[120]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_79);  sum_138 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_91 = torch.ops.aten.convolution_backward.default(mul_1353, add_159, primals_215, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1353 = add_159 = primals_215 = None
    getitem_447: "f32[8, 40, 28, 28]" = convolution_backward_91[0]
    getitem_448: "f32[120, 40, 1, 1]" = convolution_backward_91[1];  convolution_backward_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_640: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_635, getitem_447);  add_635 = getitem_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1080: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
    unsqueeze_1081: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, 2);  unsqueeze_1080 = None
    unsqueeze_1082: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1081, 3);  unsqueeze_1081 = None
    sum_139: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_640, [0, 2, 3])
    sub_331: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1082)
    mul_1355: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_640, sub_331);  sub_331 = None
    sum_140: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1355, [0, 2, 3]);  mul_1355 = None
    mul_1356: "f32[40]" = torch.ops.aten.mul.Tensor(sum_139, 0.00015943877551020407)
    unsqueeze_1083: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1356, 0);  mul_1356 = None
    unsqueeze_1084: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 2);  unsqueeze_1083 = None
    unsqueeze_1085: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 3);  unsqueeze_1084 = None
    mul_1357: "f32[40]" = torch.ops.aten.mul.Tensor(sum_140, 0.00015943877551020407)
    mul_1358: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_1359: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1357, mul_1358);  mul_1357 = mul_1358 = None
    unsqueeze_1086: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1359, 0);  mul_1359 = None
    unsqueeze_1087: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 2);  unsqueeze_1086 = None
    unsqueeze_1088: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 3);  unsqueeze_1087 = None
    mul_1360: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_1089: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1360, 0);  mul_1360 = None
    unsqueeze_1090: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1089, 2);  unsqueeze_1089 = None
    unsqueeze_1091: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 3);  unsqueeze_1090 = None
    sub_332: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_1082);  convolution_31 = unsqueeze_1082 = None
    mul_1361: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1088);  sub_332 = unsqueeze_1088 = None
    sub_333: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_640, mul_1361);  mul_1361 = None
    sub_334: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_333, unsqueeze_1085);  sub_333 = unsqueeze_1085 = None
    mul_1362: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_1091);  sub_334 = unsqueeze_1091 = None
    mul_1363: "f32[40]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_76);  sum_140 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_92 = torch.ops.aten.convolution_backward.default(mul_1362, mul_197, primals_214, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1362 = mul_197 = primals_214 = None
    getitem_450: "f32[8, 120, 28, 28]" = convolution_backward_92[0]
    getitem_451: "f32[40, 120, 1, 1]" = convolution_backward_92[1];  convolution_backward_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1364: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_450, div_20);  div_20 = None
    mul_1365: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_450, div_22);  getitem_450 = div_22 = None
    sum_141: "f32[8, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1364, [2, 3], True);  mul_1364 = None
    gt_15: "b8[8, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_30, -3.0)
    lt_72: "b8[8, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_30, 3.0);  convolution_30 = None
    bitwise_and_15: "b8[8, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_15, lt_72);  gt_15 = lt_72 = None
    mul_1366: "f32[8, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_141, 0.16666666666666666);  sum_141 = None
    scalar_tensor_72: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_129: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_15, mul_1366, scalar_tensor_72);  bitwise_and_15 = mul_1366 = scalar_tensor_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_93 = torch.ops.aten.convolution_backward.default(where_129, div_21, primals_212, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_129 = div_21 = primals_212 = None
    getitem_453: "f32[8, 16, 1, 1]" = convolution_backward_93[0]
    getitem_454: "f32[120, 16, 1, 1]" = convolution_backward_93[1]
    getitem_455: "f32[120]" = convolution_backward_93[2];  convolution_backward_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_73: "b8[8, 16, 1, 1]" = torch.ops.aten.lt.Scalar(clone_19, -3)
    le_57: "b8[8, 16, 1, 1]" = torch.ops.aten.le.Scalar(clone_19, 3)
    div_168: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(clone_19, 3);  clone_19 = None
    add_641: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(div_168, 0.5);  div_168 = None
    mul_1367: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_453, add_641);  add_641 = None
    where_130: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(le_57, mul_1367, getitem_453);  le_57 = mul_1367 = getitem_453 = None
    scalar_tensor_73: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_131: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(lt_73, scalar_tensor_73, where_130);  lt_73 = scalar_tensor_73 = where_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_94 = torch.ops.aten.convolution_backward.default(where_131, mean_2, primals_210, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_131 = mean_2 = primals_210 = None
    getitem_456: "f32[8, 120, 1, 1]" = convolution_backward_94[0]
    getitem_457: "f32[16, 120, 1, 1]" = convolution_backward_94[1]
    getitem_458: "f32[16]" = convolution_backward_94[2];  convolution_backward_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_16: "f32[8, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_456, [8, 120, 28, 28]);  getitem_456 = None
    div_169: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_16, 784);  expand_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_642: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_1365, div_169);  mul_1365 = div_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_74: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_18, -3)
    le_58: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_18, 3)
    div_170: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_18, 3);  clone_18 = None
    add_643: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_170, 0.5);  div_170 = None
    mul_1368: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_642, add_643);  add_643 = None
    where_132: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_58, mul_1368, add_642);  le_58 = mul_1368 = add_642 = None
    scalar_tensor_74: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_133: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_74, scalar_tensor_74, where_132);  lt_74 = scalar_tensor_74 = where_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1092: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
    unsqueeze_1093: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, 2);  unsqueeze_1092 = None
    unsqueeze_1094: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1093, 3);  unsqueeze_1093 = None
    sum_142: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_133, [0, 2, 3])
    sub_335: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1094)
    mul_1369: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_133, sub_335);  sub_335 = None
    sum_143: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1369, [0, 2, 3]);  mul_1369 = None
    mul_1370: "f32[120]" = torch.ops.aten.mul.Tensor(sum_142, 0.00015943877551020407)
    unsqueeze_1095: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1370, 0);  mul_1370 = None
    unsqueeze_1096: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 2);  unsqueeze_1095 = None
    unsqueeze_1097: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 3);  unsqueeze_1096 = None
    mul_1371: "f32[120]" = torch.ops.aten.mul.Tensor(sum_143, 0.00015943877551020407)
    mul_1372: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_1373: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1371, mul_1372);  mul_1371 = mul_1372 = None
    unsqueeze_1098: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1373, 0);  mul_1373 = None
    unsqueeze_1099: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 2);  unsqueeze_1098 = None
    unsqueeze_1100: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 3);  unsqueeze_1099 = None
    mul_1374: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_1101: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1374, 0);  mul_1374 = None
    unsqueeze_1102: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1101, 2);  unsqueeze_1101 = None
    unsqueeze_1103: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, 3);  unsqueeze_1102 = None
    sub_336: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_1094);  convolution_28 = unsqueeze_1094 = None
    mul_1375: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_1100);  sub_336 = unsqueeze_1100 = None
    sub_337: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_133, mul_1375);  where_133 = mul_1375 = None
    sub_338: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_337, unsqueeze_1097);  sub_337 = unsqueeze_1097 = None
    mul_1376: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_1103);  sub_338 = unsqueeze_1103 = None
    mul_1377: "f32[120]" = torch.ops.aten.mul.Tensor(sum_143, squeeze_73);  sum_143 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_95 = torch.ops.aten.convolution_backward.default(mul_1376, div_19, primals_209, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1376 = div_19 = primals_209 = None
    getitem_459: "f32[8, 120, 28, 28]" = convolution_backward_95[0]
    getitem_460: "f32[120, 1, 5, 5]" = convolution_backward_95[1];  convolution_backward_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_75: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_17, -3)
    le_59: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_17, 3)
    div_171: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_17, 3);  clone_17 = None
    add_644: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_171, 0.5);  div_171 = None
    mul_1378: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_459, add_644);  add_644 = None
    where_134: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_59, mul_1378, getitem_459);  le_59 = mul_1378 = getitem_459 = None
    scalar_tensor_75: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_135: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_75, scalar_tensor_75, where_134);  lt_75 = scalar_tensor_75 = where_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1104: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_1105: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, 2);  unsqueeze_1104 = None
    unsqueeze_1106: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1105, 3);  unsqueeze_1105 = None
    sum_144: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_135, [0, 2, 3])
    sub_339: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1106)
    mul_1379: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_135, sub_339);  sub_339 = None
    sum_145: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1379, [0, 2, 3]);  mul_1379 = None
    mul_1380: "f32[120]" = torch.ops.aten.mul.Tensor(sum_144, 0.00015943877551020407)
    unsqueeze_1107: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1380, 0);  mul_1380 = None
    unsqueeze_1108: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 2);  unsqueeze_1107 = None
    unsqueeze_1109: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 3);  unsqueeze_1108 = None
    mul_1381: "f32[120]" = torch.ops.aten.mul.Tensor(sum_145, 0.00015943877551020407)
    mul_1382: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_1383: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1381, mul_1382);  mul_1381 = mul_1382 = None
    unsqueeze_1110: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1383, 0);  mul_1383 = None
    unsqueeze_1111: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 2);  unsqueeze_1110 = None
    unsqueeze_1112: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 3);  unsqueeze_1111 = None
    mul_1384: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_1113: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1384, 0);  mul_1384 = None
    unsqueeze_1114: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1113, 2);  unsqueeze_1113 = None
    unsqueeze_1115: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, 3);  unsqueeze_1114 = None
    sub_340: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_1106);  convolution_27 = unsqueeze_1106 = None
    mul_1385: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_1112);  sub_340 = unsqueeze_1112 = None
    sub_341: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_135, mul_1385);  where_135 = mul_1385 = None
    sub_342: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_341, unsqueeze_1109);  sub_341 = unsqueeze_1109 = None
    mul_1386: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_1115);  sub_342 = unsqueeze_1115 = None
    mul_1387: "f32[120]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_70);  sum_145 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_96 = torch.ops.aten.convolution_backward.default(mul_1386, add_139, primals_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1386 = add_139 = primals_208 = None
    getitem_462: "f32[8, 40, 28, 28]" = convolution_backward_96[0]
    getitem_463: "f32[120, 40, 1, 1]" = convolution_backward_96[1];  convolution_backward_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_645: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_640, getitem_462);  add_640 = getitem_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1116: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
    unsqueeze_1117: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, 2);  unsqueeze_1116 = None
    unsqueeze_1118: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1117, 3);  unsqueeze_1117 = None
    sum_146: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_645, [0, 2, 3])
    sub_343: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1118)
    mul_1388: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_645, sub_343);  sub_343 = None
    sum_147: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1388, [0, 2, 3]);  mul_1388 = None
    mul_1389: "f32[40]" = torch.ops.aten.mul.Tensor(sum_146, 0.00015943877551020407)
    unsqueeze_1119: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1389, 0);  mul_1389 = None
    unsqueeze_1120: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 2);  unsqueeze_1119 = None
    unsqueeze_1121: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 3);  unsqueeze_1120 = None
    mul_1390: "f32[40]" = torch.ops.aten.mul.Tensor(sum_147, 0.00015943877551020407)
    mul_1391: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_1392: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1390, mul_1391);  mul_1390 = mul_1391 = None
    unsqueeze_1122: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1392, 0);  mul_1392 = None
    unsqueeze_1123: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 2);  unsqueeze_1122 = None
    unsqueeze_1124: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1123, 3);  unsqueeze_1123 = None
    mul_1393: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_1125: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1393, 0);  mul_1393 = None
    unsqueeze_1126: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1125, 2);  unsqueeze_1125 = None
    unsqueeze_1127: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, 3);  unsqueeze_1126 = None
    sub_344: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_1118);  convolution_26 = unsqueeze_1118 = None
    mul_1394: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_1124);  sub_344 = unsqueeze_1124 = None
    sub_345: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_645, mul_1394);  mul_1394 = None
    sub_346: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_345, unsqueeze_1121);  sub_345 = unsqueeze_1121 = None
    mul_1395: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_1127);  sub_346 = unsqueeze_1127 = None
    mul_1396: "f32[40]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_67);  sum_147 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_97 = torch.ops.aten.convolution_backward.default(mul_1395, mul_172, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1395 = mul_172 = primals_207 = None
    getitem_465: "f32[8, 120, 28, 28]" = convolution_backward_97[0]
    getitem_466: "f32[40, 120, 1, 1]" = convolution_backward_97[1];  convolution_backward_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1397: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_465, div_16);  div_16 = None
    mul_1398: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_465, div_18);  getitem_465 = div_18 = None
    sum_148: "f32[8, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1397, [2, 3], True);  mul_1397 = None
    gt_16: "b8[8, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_25, -3.0)
    lt_76: "b8[8, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_25, 3.0);  convolution_25 = None
    bitwise_and_16: "b8[8, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_16, lt_76);  gt_16 = lt_76 = None
    mul_1399: "f32[8, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_148, 0.16666666666666666);  sum_148 = None
    scalar_tensor_76: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_136: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_16, mul_1399, scalar_tensor_76);  bitwise_and_16 = mul_1399 = scalar_tensor_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_98 = torch.ops.aten.convolution_backward.default(where_136, div_17, primals_205, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_136 = div_17 = primals_205 = None
    getitem_468: "f32[8, 16, 1, 1]" = convolution_backward_98[0]
    getitem_469: "f32[120, 16, 1, 1]" = convolution_backward_98[1]
    getitem_470: "f32[120]" = convolution_backward_98[2];  convolution_backward_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_77: "b8[8, 16, 1, 1]" = torch.ops.aten.lt.Scalar(clone_16, -3)
    le_60: "b8[8, 16, 1, 1]" = torch.ops.aten.le.Scalar(clone_16, 3)
    div_172: "f32[8, 16, 1, 1]" = torch.ops.aten.div.Tensor(clone_16, 3);  clone_16 = None
    add_646: "f32[8, 16, 1, 1]" = torch.ops.aten.add.Tensor(div_172, 0.5);  div_172 = None
    mul_1400: "f32[8, 16, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_468, add_646);  add_646 = None
    where_137: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(le_60, mul_1400, getitem_468);  le_60 = mul_1400 = getitem_468 = None
    scalar_tensor_77: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_138: "f32[8, 16, 1, 1]" = torch.ops.aten.where.self(lt_77, scalar_tensor_77, where_137);  lt_77 = scalar_tensor_77 = where_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_99 = torch.ops.aten.convolution_backward.default(where_138, mean_1, primals_203, [16], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_138 = mean_1 = primals_203 = None
    getitem_471: "f32[8, 120, 1, 1]" = convolution_backward_99[0]
    getitem_472: "f32[16, 120, 1, 1]" = convolution_backward_99[1]
    getitem_473: "f32[16]" = convolution_backward_99[2];  convolution_backward_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_17: "f32[8, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_471, [8, 120, 28, 28]);  getitem_471 = None
    div_173: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_17, 784);  expand_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_647: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_1398, div_173);  mul_1398 = div_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_78: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_15, -3)
    le_61: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_15, 3)
    div_174: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_15, 3);  clone_15 = None
    add_648: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_174, 0.5);  div_174 = None
    mul_1401: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_647, add_648);  add_648 = None
    where_139: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_61, mul_1401, add_647);  le_61 = mul_1401 = add_647 = None
    scalar_tensor_78: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_140: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_78, scalar_tensor_78, where_139);  lt_78 = scalar_tensor_78 = where_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1128: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
    unsqueeze_1129: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, 2);  unsqueeze_1128 = None
    unsqueeze_1130: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1129, 3);  unsqueeze_1129 = None
    sum_149: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_140, [0, 2, 3])
    sub_347: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1130)
    mul_1402: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_140, sub_347);  sub_347 = None
    sum_150: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1402, [0, 2, 3]);  mul_1402 = None
    mul_1403: "f32[120]" = torch.ops.aten.mul.Tensor(sum_149, 0.00015943877551020407)
    unsqueeze_1131: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1403, 0);  mul_1403 = None
    unsqueeze_1132: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 2);  unsqueeze_1131 = None
    unsqueeze_1133: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 3);  unsqueeze_1132 = None
    mul_1404: "f32[120]" = torch.ops.aten.mul.Tensor(sum_150, 0.00015943877551020407)
    mul_1405: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_1406: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1404, mul_1405);  mul_1404 = mul_1405 = None
    unsqueeze_1134: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1406, 0);  mul_1406 = None
    unsqueeze_1135: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 2);  unsqueeze_1134 = None
    unsqueeze_1136: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 3);  unsqueeze_1135 = None
    mul_1407: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_1137: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1407, 0);  mul_1407 = None
    unsqueeze_1138: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1137, 2);  unsqueeze_1137 = None
    unsqueeze_1139: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 3);  unsqueeze_1138 = None
    sub_348: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_1130);  convolution_23 = unsqueeze_1130 = None
    mul_1408: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_1136);  sub_348 = unsqueeze_1136 = None
    sub_349: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_140, mul_1408);  where_140 = mul_1408 = None
    sub_350: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_349, unsqueeze_1133);  sub_349 = unsqueeze_1133 = None
    mul_1409: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_350, unsqueeze_1139);  sub_350 = unsqueeze_1139 = None
    mul_1410: "f32[120]" = torch.ops.aten.mul.Tensor(sum_150, squeeze_64);  sum_150 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_100 = torch.ops.aten.convolution_backward.default(mul_1409, div_15, primals_202, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1409 = div_15 = primals_202 = None
    getitem_474: "f32[8, 120, 28, 28]" = convolution_backward_100[0]
    getitem_475: "f32[120, 1, 5, 5]" = convolution_backward_100[1];  convolution_backward_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_79: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_14, -3)
    le_62: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_14, 3)
    div_175: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_14, 3);  clone_14 = None
    add_649: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_175, 0.5);  div_175 = None
    mul_1411: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_474, add_649);  add_649 = None
    where_141: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_62, mul_1411, getitem_474);  le_62 = mul_1411 = getitem_474 = None
    scalar_tensor_79: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_142: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_79, scalar_tensor_79, where_141);  lt_79 = scalar_tensor_79 = where_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1140: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
    unsqueeze_1141: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, 2);  unsqueeze_1140 = None
    unsqueeze_1142: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1141, 3);  unsqueeze_1141 = None
    sum_151: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_142, [0, 2, 3])
    sub_351: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1142)
    mul_1412: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_142, sub_351);  sub_351 = None
    sum_152: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1412, [0, 2, 3]);  mul_1412 = None
    mul_1413: "f32[120]" = torch.ops.aten.mul.Tensor(sum_151, 0.00015943877551020407)
    unsqueeze_1143: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1413, 0);  mul_1413 = None
    unsqueeze_1144: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 2);  unsqueeze_1143 = None
    unsqueeze_1145: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 3);  unsqueeze_1144 = None
    mul_1414: "f32[120]" = torch.ops.aten.mul.Tensor(sum_152, 0.00015943877551020407)
    mul_1415: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_1416: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1414, mul_1415);  mul_1414 = mul_1415 = None
    unsqueeze_1146: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1416, 0);  mul_1416 = None
    unsqueeze_1147: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 2);  unsqueeze_1146 = None
    unsqueeze_1148: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 3);  unsqueeze_1147 = None
    mul_1417: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_1149: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1417, 0);  mul_1417 = None
    unsqueeze_1150: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1149, 2);  unsqueeze_1149 = None
    unsqueeze_1151: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 3);  unsqueeze_1150 = None
    sub_352: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_1142);  convolution_22 = unsqueeze_1142 = None
    mul_1418: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_1148);  sub_352 = unsqueeze_1148 = None
    sub_353: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_142, mul_1418);  where_142 = mul_1418 = None
    sub_354: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_353, unsqueeze_1145);  sub_353 = unsqueeze_1145 = None
    mul_1419: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_354, unsqueeze_1151);  sub_354 = unsqueeze_1151 = None
    mul_1420: "f32[120]" = torch.ops.aten.mul.Tensor(sum_152, squeeze_61);  sum_152 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_101 = torch.ops.aten.convolution_backward.default(mul_1419, add_119, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1419 = add_119 = primals_201 = None
    getitem_477: "f32[8, 40, 28, 28]" = convolution_backward_101[0]
    getitem_478: "f32[120, 40, 1, 1]" = convolution_backward_101[1];  convolution_backward_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_650: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_645, getitem_477);  add_645 = getitem_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1152: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_1153: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, 2);  unsqueeze_1152 = None
    unsqueeze_1154: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1153, 3);  unsqueeze_1153 = None
    sum_153: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_650, [0, 2, 3])
    sub_355: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1154)
    mul_1421: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_650, sub_355);  sub_355 = None
    sum_154: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_1421, [0, 2, 3]);  mul_1421 = None
    mul_1422: "f32[40]" = torch.ops.aten.mul.Tensor(sum_153, 0.00015943877551020407)
    unsqueeze_1155: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1422, 0);  mul_1422 = None
    unsqueeze_1156: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 2);  unsqueeze_1155 = None
    unsqueeze_1157: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 3);  unsqueeze_1156 = None
    mul_1423: "f32[40]" = torch.ops.aten.mul.Tensor(sum_154, 0.00015943877551020407)
    mul_1424: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_1425: "f32[40]" = torch.ops.aten.mul.Tensor(mul_1423, mul_1424);  mul_1423 = mul_1424 = None
    unsqueeze_1158: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1425, 0);  mul_1425 = None
    unsqueeze_1159: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 2);  unsqueeze_1158 = None
    unsqueeze_1160: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 3);  unsqueeze_1159 = None
    mul_1426: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_1161: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_1426, 0);  mul_1426 = None
    unsqueeze_1162: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1161, 2);  unsqueeze_1161 = None
    unsqueeze_1163: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 3);  unsqueeze_1162 = None
    sub_356: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_1154);  convolution_21 = unsqueeze_1154 = None
    mul_1427: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_1160);  sub_356 = unsqueeze_1160 = None
    sub_357: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_650, mul_1427);  add_650 = mul_1427 = None
    sub_358: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_357, unsqueeze_1157);  sub_357 = unsqueeze_1157 = None
    mul_1428: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_358, unsqueeze_1163);  sub_358 = unsqueeze_1163 = None
    mul_1429: "f32[40]" = torch.ops.aten.mul.Tensor(sum_154, squeeze_58);  sum_154 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_102 = torch.ops.aten.convolution_backward.default(mul_1428, mul_147, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1428 = mul_147 = primals_200 = None
    getitem_480: "f32[8, 120, 28, 28]" = convolution_backward_102[0]
    getitem_481: "f32[40, 120, 1, 1]" = convolution_backward_102[1];  convolution_backward_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    mul_1430: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_480, div_12);  div_12 = None
    mul_1431: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_480, div_14);  getitem_480 = div_14 = None
    sum_155: "f32[8, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1430, [2, 3], True);  mul_1430 = None
    gt_17: "b8[8, 120, 1, 1]" = torch.ops.aten.gt.Scalar(convolution_20, -3.0)
    lt_80: "b8[8, 120, 1, 1]" = torch.ops.aten.lt.Scalar(convolution_20, 3.0);  convolution_20 = None
    bitwise_and_17: "b8[8, 120, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt_17, lt_80);  gt_17 = lt_80 = None
    mul_1432: "f32[8, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_155, 0.16666666666666666);  sum_155 = None
    scalar_tensor_80: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_143: "f32[8, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_17, mul_1432, scalar_tensor_80);  bitwise_and_17 = mul_1432 = scalar_tensor_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    convolution_backward_103 = torch.ops.aten.convolution_backward.default(where_143, div_13, primals_198, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_143 = div_13 = primals_198 = None
    getitem_483: "f32[8, 8, 1, 1]" = convolution_backward_103[0]
    getitem_484: "f32[120, 8, 1, 1]" = convolution_backward_103[1]
    getitem_485: "f32[120]" = convolution_backward_103[2];  convolution_backward_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    lt_81: "b8[8, 8, 1, 1]" = torch.ops.aten.lt.Scalar(clone_13, -3)
    le_63: "b8[8, 8, 1, 1]" = torch.ops.aten.le.Scalar(clone_13, 3)
    div_176: "f32[8, 8, 1, 1]" = torch.ops.aten.div.Tensor(clone_13, 3);  clone_13 = None
    add_651: "f32[8, 8, 1, 1]" = torch.ops.aten.add.Tensor(div_176, 0.5);  div_176 = None
    mul_1433: "f32[8, 8, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_483, add_651);  add_651 = None
    where_144: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(le_63, mul_1433, getitem_483);  le_63 = mul_1433 = getitem_483 = None
    scalar_tensor_81: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_145: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(lt_81, scalar_tensor_81, where_144);  lt_81 = scalar_tensor_81 = where_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    convolution_backward_104 = torch.ops.aten.convolution_backward.default(where_145, mean, primals_196, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_145 = mean = primals_196 = None
    getitem_486: "f32[8, 120, 1, 1]" = convolution_backward_104[0]
    getitem_487: "f32[8, 120, 1, 1]" = convolution_backward_104[1]
    getitem_488: "f32[8]" = convolution_backward_104[2];  convolution_backward_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    expand_18: "f32[8, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_486, [8, 120, 28, 28]);  getitem_486 = None
    div_177: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_18, 784);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    add_652: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_1431, div_177);  mul_1431 = div_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_82: "b8[8, 120, 28, 28]" = torch.ops.aten.lt.Scalar(clone_12, -3)
    le_64: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(clone_12, 3)
    div_178: "f32[8, 120, 28, 28]" = torch.ops.aten.div.Tensor(clone_12, 3);  clone_12 = None
    add_653: "f32[8, 120, 28, 28]" = torch.ops.aten.add.Tensor(div_178, 0.5);  div_178 = None
    mul_1434: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(add_652, add_653);  add_653 = None
    where_146: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_64, mul_1434, add_652);  le_64 = mul_1434 = add_652 = None
    scalar_tensor_82: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_147: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(lt_82, scalar_tensor_82, where_146);  lt_82 = scalar_tensor_82 = where_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1164: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
    unsqueeze_1165: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, 2);  unsqueeze_1164 = None
    unsqueeze_1166: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1165, 3);  unsqueeze_1165 = None
    sum_156: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_147, [0, 2, 3])
    sub_359: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1166)
    mul_1435: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_147, sub_359);  sub_359 = None
    sum_157: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1435, [0, 2, 3]);  mul_1435 = None
    mul_1436: "f32[120]" = torch.ops.aten.mul.Tensor(sum_156, 0.00015943877551020407)
    unsqueeze_1167: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1436, 0);  mul_1436 = None
    unsqueeze_1168: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 2);  unsqueeze_1167 = None
    unsqueeze_1169: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 3);  unsqueeze_1168 = None
    mul_1437: "f32[120]" = torch.ops.aten.mul.Tensor(sum_157, 0.00015943877551020407)
    mul_1438: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1439: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1437, mul_1438);  mul_1437 = mul_1438 = None
    unsqueeze_1170: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1439, 0);  mul_1439 = None
    unsqueeze_1171: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 2);  unsqueeze_1170 = None
    unsqueeze_1172: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 3);  unsqueeze_1171 = None
    mul_1440: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_1173: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1440, 0);  mul_1440 = None
    unsqueeze_1174: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1173, 2);  unsqueeze_1173 = None
    unsqueeze_1175: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 3);  unsqueeze_1174 = None
    sub_360: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_1166);  convolution_18 = unsqueeze_1166 = None
    mul_1441: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_1172);  sub_360 = unsqueeze_1172 = None
    sub_361: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_147, mul_1441);  where_147 = mul_1441 = None
    sub_362: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_361, unsqueeze_1169);  sub_361 = unsqueeze_1169 = None
    mul_1442: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_362, unsqueeze_1175);  sub_362 = unsqueeze_1175 = None
    mul_1443: "f32[120]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_55);  sum_157 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_105 = torch.ops.aten.convolution_backward.default(mul_1442, div_11, primals_195, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_1442 = div_11 = primals_195 = None
    getitem_489: "f32[8, 120, 56, 56]" = convolution_backward_105[0]
    getitem_490: "f32[120, 1, 5, 5]" = convolution_backward_105[1];  convolution_backward_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_83: "b8[8, 120, 56, 56]" = torch.ops.aten.lt.Scalar(clone_11, -3)
    le_65: "b8[8, 120, 56, 56]" = torch.ops.aten.le.Scalar(clone_11, 3)
    div_179: "f32[8, 120, 56, 56]" = torch.ops.aten.div.Tensor(clone_11, 3);  clone_11 = None
    add_654: "f32[8, 120, 56, 56]" = torch.ops.aten.add.Tensor(div_179, 0.5);  div_179 = None
    mul_1444: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_489, add_654);  add_654 = None
    where_148: "f32[8, 120, 56, 56]" = torch.ops.aten.where.self(le_65, mul_1444, getitem_489);  le_65 = mul_1444 = getitem_489 = None
    scalar_tensor_83: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_149: "f32[8, 120, 56, 56]" = torch.ops.aten.where.self(lt_83, scalar_tensor_83, where_148);  lt_83 = scalar_tensor_83 = where_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1176: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
    unsqueeze_1177: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, 2);  unsqueeze_1176 = None
    unsqueeze_1178: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1177, 3);  unsqueeze_1177 = None
    sum_158: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_149, [0, 2, 3])
    sub_363: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1178)
    mul_1445: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(where_149, sub_363);  sub_363 = None
    sum_159: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_1445, [0, 2, 3]);  mul_1445 = None
    mul_1446: "f32[120]" = torch.ops.aten.mul.Tensor(sum_158, 3.985969387755102e-05)
    unsqueeze_1179: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1446, 0);  mul_1446 = None
    unsqueeze_1180: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 2);  unsqueeze_1179 = None
    unsqueeze_1181: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 3);  unsqueeze_1180 = None
    mul_1447: "f32[120]" = torch.ops.aten.mul.Tensor(sum_159, 3.985969387755102e-05)
    mul_1448: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1449: "f32[120]" = torch.ops.aten.mul.Tensor(mul_1447, mul_1448);  mul_1447 = mul_1448 = None
    unsqueeze_1182: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1449, 0);  mul_1449 = None
    unsqueeze_1183: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 2);  unsqueeze_1182 = None
    unsqueeze_1184: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1183, 3);  unsqueeze_1183 = None
    mul_1450: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_1185: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_1450, 0);  mul_1450 = None
    unsqueeze_1186: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1185, 2);  unsqueeze_1185 = None
    unsqueeze_1187: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, 3);  unsqueeze_1186 = None
    sub_364: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_1178);  convolution_17 = unsqueeze_1178 = None
    mul_1451: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_1184);  sub_364 = unsqueeze_1184 = None
    sub_365: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(where_149, mul_1451);  where_149 = mul_1451 = None
    sub_366: "f32[8, 120, 56, 56]" = torch.ops.aten.sub.Tensor(sub_365, unsqueeze_1181);  sub_365 = unsqueeze_1181 = None
    mul_1452: "f32[8, 120, 56, 56]" = torch.ops.aten.mul.Tensor(sub_366, unsqueeze_1187);  sub_366 = unsqueeze_1187 = None
    mul_1453: "f32[120]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_52);  sum_159 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_106 = torch.ops.aten.convolution_backward.default(mul_1452, add_100, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1452 = add_100 = primals_194 = None
    getitem_492: "f32[8, 24, 56, 56]" = convolution_backward_106[0]
    getitem_493: "f32[120, 24, 1, 1]" = convolution_backward_106[1];  convolution_backward_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1188: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
    unsqueeze_1189: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, 2);  unsqueeze_1188 = None
    unsqueeze_1190: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1189, 3);  unsqueeze_1189 = None
    sum_160: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_492, [0, 2, 3])
    sub_367: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1190)
    mul_1454: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_492, sub_367);  sub_367 = None
    sum_161: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1454, [0, 2, 3]);  mul_1454 = None
    mul_1455: "f32[24]" = torch.ops.aten.mul.Tensor(sum_160, 3.985969387755102e-05)
    unsqueeze_1191: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1455, 0);  mul_1455 = None
    unsqueeze_1192: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 2);  unsqueeze_1191 = None
    unsqueeze_1193: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, 3);  unsqueeze_1192 = None
    mul_1456: "f32[24]" = torch.ops.aten.mul.Tensor(sum_161, 3.985969387755102e-05)
    mul_1457: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1458: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1456, mul_1457);  mul_1456 = mul_1457 = None
    unsqueeze_1194: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1458, 0);  mul_1458 = None
    unsqueeze_1195: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 2);  unsqueeze_1194 = None
    unsqueeze_1196: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 3);  unsqueeze_1195 = None
    mul_1459: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_1197: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1459, 0);  mul_1459 = None
    unsqueeze_1198: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1197, 2);  unsqueeze_1197 = None
    unsqueeze_1199: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 3);  unsqueeze_1198 = None
    sub_368: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_1190);  convolution_16 = unsqueeze_1190 = None
    mul_1460: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_1196);  sub_368 = unsqueeze_1196 = None
    sub_369: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_492, mul_1460);  mul_1460 = None
    sub_370: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_369, unsqueeze_1193);  sub_369 = unsqueeze_1193 = None
    mul_1461: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_370, unsqueeze_1199);  sub_370 = unsqueeze_1199 = None
    mul_1462: "f32[24]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_49);  sum_161 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_107 = torch.ops.aten.convolution_backward.default(mul_1461, div_10, primals_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1461 = div_10 = primals_193 = None
    getitem_495: "f32[8, 48, 56, 56]" = convolution_backward_107[0]
    getitem_496: "f32[24, 48, 1, 1]" = convolution_backward_107[1];  convolution_backward_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_84: "b8[8, 48, 56, 56]" = torch.ops.aten.lt.Scalar(clone_10, -3)
    le_66: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(clone_10, 3)
    div_180: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(clone_10, 3);  clone_10 = None
    add_655: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(div_180, 0.5);  div_180 = None
    mul_1463: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_495, add_655);  add_655 = None
    where_150: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_66, mul_1463, getitem_495);  le_66 = mul_1463 = getitem_495 = None
    scalar_tensor_84: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_151: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(lt_84, scalar_tensor_84, where_150);  lt_84 = scalar_tensor_84 = where_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1200: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_1201: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, 2);  unsqueeze_1200 = None
    unsqueeze_1202: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1201, 3);  unsqueeze_1201 = None
    sum_162: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_151, [0, 2, 3])
    sub_371: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1202)
    mul_1464: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_151, sub_371);  sub_371 = None
    sum_163: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1464, [0, 2, 3]);  mul_1464 = None
    mul_1465: "f32[48]" = torch.ops.aten.mul.Tensor(sum_162, 3.985969387755102e-05)
    unsqueeze_1203: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1465, 0);  mul_1465 = None
    unsqueeze_1204: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 2);  unsqueeze_1203 = None
    unsqueeze_1205: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 3);  unsqueeze_1204 = None
    mul_1466: "f32[48]" = torch.ops.aten.mul.Tensor(sum_163, 3.985969387755102e-05)
    mul_1467: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1468: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1466, mul_1467);  mul_1466 = mul_1467 = None
    unsqueeze_1206: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1468, 0);  mul_1468 = None
    unsqueeze_1207: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 2);  unsqueeze_1206 = None
    unsqueeze_1208: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 3);  unsqueeze_1207 = None
    mul_1469: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_1209: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1469, 0);  mul_1469 = None
    unsqueeze_1210: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1209, 2);  unsqueeze_1209 = None
    unsqueeze_1211: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 3);  unsqueeze_1210 = None
    sub_372: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_1202);  convolution_15 = unsqueeze_1202 = None
    mul_1470: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_1208);  sub_372 = unsqueeze_1208 = None
    sub_373: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_151, mul_1470);  where_151 = mul_1470 = None
    sub_374: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_373, unsqueeze_1205);  sub_373 = unsqueeze_1205 = None
    mul_1471: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_374, unsqueeze_1211);  sub_374 = unsqueeze_1211 = None
    mul_1472: "f32[48]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_46);  sum_163 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_108 = torch.ops.aten.convolution_backward.default(mul_1471, div_9, primals_192, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False]);  mul_1471 = div_9 = primals_192 = None
    getitem_498: "f32[8, 48, 56, 56]" = convolution_backward_108[0]
    getitem_499: "f32[48, 1, 5, 5]" = convolution_backward_108[1];  convolution_backward_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_85: "b8[8, 48, 56, 56]" = torch.ops.aten.lt.Scalar(clone_9, -3)
    le_67: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(clone_9, 3)
    div_181: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(clone_9, 3);  clone_9 = None
    add_656: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(div_181, 0.5);  div_181 = None
    mul_1473: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_498, add_656);  add_656 = None
    where_152: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_67, mul_1473, getitem_498);  le_67 = mul_1473 = getitem_498 = None
    scalar_tensor_85: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_153: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(lt_85, scalar_tensor_85, where_152);  lt_85 = scalar_tensor_85 = where_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1212: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
    unsqueeze_1213: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, 2);  unsqueeze_1212 = None
    unsqueeze_1214: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1213, 3);  unsqueeze_1213 = None
    sum_164: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_153, [0, 2, 3])
    sub_375: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1214)
    mul_1474: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_153, sub_375);  sub_375 = None
    sum_165: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1474, [0, 2, 3]);  mul_1474 = None
    mul_1475: "f32[48]" = torch.ops.aten.mul.Tensor(sum_164, 3.985969387755102e-05)
    unsqueeze_1215: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1475, 0);  mul_1475 = None
    unsqueeze_1216: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 2);  unsqueeze_1215 = None
    unsqueeze_1217: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 3);  unsqueeze_1216 = None
    mul_1476: "f32[48]" = torch.ops.aten.mul.Tensor(sum_165, 3.985969387755102e-05)
    mul_1477: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1478: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1476, mul_1477);  mul_1476 = mul_1477 = None
    unsqueeze_1218: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1478, 0);  mul_1478 = None
    unsqueeze_1219: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 2);  unsqueeze_1218 = None
    unsqueeze_1220: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 3);  unsqueeze_1219 = None
    mul_1479: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_1221: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1479, 0);  mul_1479 = None
    unsqueeze_1222: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1221, 2);  unsqueeze_1221 = None
    unsqueeze_1223: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 3);  unsqueeze_1222 = None
    sub_376: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_1214);  convolution_14 = unsqueeze_1214 = None
    mul_1480: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_1220);  sub_376 = unsqueeze_1220 = None
    sub_377: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_153, mul_1480);  where_153 = mul_1480 = None
    sub_378: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_377, unsqueeze_1217);  sub_377 = unsqueeze_1217 = None
    mul_1481: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_378, unsqueeze_1223);  sub_378 = unsqueeze_1223 = None
    mul_1482: "f32[48]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_43);  sum_165 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_109 = torch.ops.aten.convolution_backward.default(mul_1481, add_82, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1481 = add_82 = primals_191 = None
    getitem_501: "f32[8, 24, 56, 56]" = convolution_backward_109[0]
    getitem_502: "f32[48, 24, 1, 1]" = convolution_backward_109[1];  convolution_backward_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_657: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_492, getitem_501);  getitem_492 = getitem_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1224: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
    unsqueeze_1225: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, 2);  unsqueeze_1224 = None
    unsqueeze_1226: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1225, 3);  unsqueeze_1225 = None
    sum_166: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_657, [0, 2, 3])
    sub_379: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1226)
    mul_1483: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_657, sub_379);  sub_379 = None
    sum_167: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1483, [0, 2, 3]);  mul_1483 = None
    mul_1484: "f32[24]" = torch.ops.aten.mul.Tensor(sum_166, 3.985969387755102e-05)
    unsqueeze_1227: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1484, 0);  mul_1484 = None
    unsqueeze_1228: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 2);  unsqueeze_1227 = None
    unsqueeze_1229: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 3);  unsqueeze_1228 = None
    mul_1485: "f32[24]" = torch.ops.aten.mul.Tensor(sum_167, 3.985969387755102e-05)
    mul_1486: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1487: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1485, mul_1486);  mul_1485 = mul_1486 = None
    unsqueeze_1230: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1487, 0);  mul_1487 = None
    unsqueeze_1231: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 2);  unsqueeze_1230 = None
    unsqueeze_1232: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 3);  unsqueeze_1231 = None
    mul_1488: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_1233: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1488, 0);  mul_1488 = None
    unsqueeze_1234: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1233, 2);  unsqueeze_1233 = None
    unsqueeze_1235: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 3);  unsqueeze_1234 = None
    sub_380: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_1226);  convolution_13 = unsqueeze_1226 = None
    mul_1489: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_1232);  sub_380 = unsqueeze_1232 = None
    sub_381: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_657, mul_1489);  mul_1489 = None
    sub_382: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_381, unsqueeze_1229);  sub_381 = unsqueeze_1229 = None
    mul_1490: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_382, unsqueeze_1235);  sub_382 = unsqueeze_1235 = None
    mul_1491: "f32[24]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_40);  sum_167 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_110 = torch.ops.aten.convolution_backward.default(mul_1490, div_8, primals_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1490 = div_8 = primals_190 = None
    getitem_504: "f32[8, 48, 56, 56]" = convolution_backward_110[0]
    getitem_505: "f32[24, 48, 1, 1]" = convolution_backward_110[1];  convolution_backward_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_86: "b8[8, 48, 56, 56]" = torch.ops.aten.lt.Scalar(clone_8, -3)
    le_68: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(clone_8, 3)
    div_182: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(clone_8, 3);  clone_8 = None
    add_658: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(div_182, 0.5);  div_182 = None
    mul_1492: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_504, add_658);  add_658 = None
    where_154: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_68, mul_1492, getitem_504);  le_68 = mul_1492 = getitem_504 = None
    scalar_tensor_86: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_155: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(lt_86, scalar_tensor_86, where_154);  lt_86 = scalar_tensor_86 = where_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1236: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
    unsqueeze_1237: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, 2);  unsqueeze_1236 = None
    unsqueeze_1238: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1237, 3);  unsqueeze_1237 = None
    sum_168: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_155, [0, 2, 3])
    sub_383: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1238)
    mul_1493: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_155, sub_383);  sub_383 = None
    sum_169: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1493, [0, 2, 3]);  mul_1493 = None
    mul_1494: "f32[48]" = torch.ops.aten.mul.Tensor(sum_168, 3.985969387755102e-05)
    unsqueeze_1239: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1494, 0);  mul_1494 = None
    unsqueeze_1240: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 2);  unsqueeze_1239 = None
    unsqueeze_1241: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, 3);  unsqueeze_1240 = None
    mul_1495: "f32[48]" = torch.ops.aten.mul.Tensor(sum_169, 3.985969387755102e-05)
    mul_1496: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1497: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1495, mul_1496);  mul_1495 = mul_1496 = None
    unsqueeze_1242: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1497, 0);  mul_1497 = None
    unsqueeze_1243: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 2);  unsqueeze_1242 = None
    unsqueeze_1244: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1243, 3);  unsqueeze_1243 = None
    mul_1498: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_1245: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1498, 0);  mul_1498 = None
    unsqueeze_1246: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1245, 2);  unsqueeze_1245 = None
    unsqueeze_1247: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, 3);  unsqueeze_1246 = None
    sub_384: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_1238);  convolution_12 = unsqueeze_1238 = None
    mul_1499: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_1244);  sub_384 = unsqueeze_1244 = None
    sub_385: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_155, mul_1499);  where_155 = mul_1499 = None
    sub_386: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_385, unsqueeze_1241);  sub_385 = unsqueeze_1241 = None
    mul_1500: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_386, unsqueeze_1247);  sub_386 = unsqueeze_1247 = None
    mul_1501: "f32[48]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_37);  sum_169 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_111 = torch.ops.aten.convolution_backward.default(mul_1500, div_7, primals_189, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False]);  mul_1500 = div_7 = primals_189 = None
    getitem_507: "f32[8, 48, 56, 56]" = convolution_backward_111[0]
    getitem_508: "f32[48, 1, 5, 5]" = convolution_backward_111[1];  convolution_backward_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_87: "b8[8, 48, 56, 56]" = torch.ops.aten.lt.Scalar(clone_7, -3)
    le_69: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(clone_7, 3)
    div_183: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(clone_7, 3);  clone_7 = None
    add_659: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(div_183, 0.5);  div_183 = None
    mul_1502: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_507, add_659);  add_659 = None
    where_156: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_69, mul_1502, getitem_507);  le_69 = mul_1502 = getitem_507 = None
    scalar_tensor_87: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_157: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(lt_87, scalar_tensor_87, where_156);  lt_87 = scalar_tensor_87 = where_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1248: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_1249: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, 2);  unsqueeze_1248 = None
    unsqueeze_1250: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1249, 3);  unsqueeze_1249 = None
    sum_170: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_157, [0, 2, 3])
    sub_387: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1250)
    mul_1503: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_157, sub_387);  sub_387 = None
    sum_171: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1503, [0, 2, 3]);  mul_1503 = None
    mul_1504: "f32[48]" = torch.ops.aten.mul.Tensor(sum_170, 3.985969387755102e-05)
    unsqueeze_1251: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1504, 0);  mul_1504 = None
    unsqueeze_1252: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 2);  unsqueeze_1251 = None
    unsqueeze_1253: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, 3);  unsqueeze_1252 = None
    mul_1505: "f32[48]" = torch.ops.aten.mul.Tensor(sum_171, 3.985969387755102e-05)
    mul_1506: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1507: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1505, mul_1506);  mul_1505 = mul_1506 = None
    unsqueeze_1254: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1507, 0);  mul_1507 = None
    unsqueeze_1255: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 2);  unsqueeze_1254 = None
    unsqueeze_1256: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1255, 3);  unsqueeze_1255 = None
    mul_1508: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_1257: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1508, 0);  mul_1508 = None
    unsqueeze_1258: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1257, 2);  unsqueeze_1257 = None
    unsqueeze_1259: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, 3);  unsqueeze_1258 = None
    sub_388: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_1250);  convolution_11 = unsqueeze_1250 = None
    mul_1509: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_1256);  sub_388 = unsqueeze_1256 = None
    sub_389: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_157, mul_1509);  where_157 = mul_1509 = None
    sub_390: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_389, unsqueeze_1253);  sub_389 = unsqueeze_1253 = None
    mul_1510: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_390, unsqueeze_1259);  sub_390 = unsqueeze_1259 = None
    mul_1511: "f32[48]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_34);  sum_171 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_112 = torch.ops.aten.convolution_backward.default(mul_1510, add_64, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1510 = add_64 = primals_188 = None
    getitem_510: "f32[8, 24, 56, 56]" = convolution_backward_112[0]
    getitem_511: "f32[48, 24, 1, 1]" = convolution_backward_112[1];  convolution_backward_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_660: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_657, getitem_510);  add_657 = getitem_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1260: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
    unsqueeze_1261: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, 2);  unsqueeze_1260 = None
    unsqueeze_1262: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1261, 3);  unsqueeze_1261 = None
    sum_172: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_660, [0, 2, 3])
    sub_391: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1262)
    mul_1512: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_660, sub_391);  sub_391 = None
    sum_173: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1512, [0, 2, 3]);  mul_1512 = None
    mul_1513: "f32[24]" = torch.ops.aten.mul.Tensor(sum_172, 3.985969387755102e-05)
    unsqueeze_1263: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1513, 0);  mul_1513 = None
    unsqueeze_1264: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 2);  unsqueeze_1263 = None
    unsqueeze_1265: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, 3);  unsqueeze_1264 = None
    mul_1514: "f32[24]" = torch.ops.aten.mul.Tensor(sum_173, 3.985969387755102e-05)
    mul_1515: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1516: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1514, mul_1515);  mul_1514 = mul_1515 = None
    unsqueeze_1266: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1516, 0);  mul_1516 = None
    unsqueeze_1267: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 2);  unsqueeze_1266 = None
    unsqueeze_1268: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1267, 3);  unsqueeze_1267 = None
    mul_1517: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_1269: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1517, 0);  mul_1517 = None
    unsqueeze_1270: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1269, 2);  unsqueeze_1269 = None
    unsqueeze_1271: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, 3);  unsqueeze_1270 = None
    sub_392: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_1262);  convolution_10 = unsqueeze_1262 = None
    mul_1518: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_1268);  sub_392 = unsqueeze_1268 = None
    sub_393: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_660, mul_1518);  mul_1518 = None
    sub_394: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_393, unsqueeze_1265);  sub_393 = unsqueeze_1265 = None
    mul_1519: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_394, unsqueeze_1271);  sub_394 = unsqueeze_1271 = None
    mul_1520: "f32[24]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_31);  sum_173 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_113 = torch.ops.aten.convolution_backward.default(mul_1519, div_6, primals_187, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1519 = div_6 = primals_187 = None
    getitem_513: "f32[8, 48, 56, 56]" = convolution_backward_113[0]
    getitem_514: "f32[24, 48, 1, 1]" = convolution_backward_113[1];  convolution_backward_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_88: "b8[8, 48, 56, 56]" = torch.ops.aten.lt.Scalar(clone_6, -3)
    le_70: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(clone_6, 3)
    div_184: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(clone_6, 3);  clone_6 = None
    add_661: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(div_184, 0.5);  div_184 = None
    mul_1521: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_513, add_661);  add_661 = None
    where_158: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_70, mul_1521, getitem_513);  le_70 = mul_1521 = getitem_513 = None
    scalar_tensor_88: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_159: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(lt_88, scalar_tensor_88, where_158);  lt_88 = scalar_tensor_88 = where_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1272: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
    unsqueeze_1273: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, 2);  unsqueeze_1272 = None
    unsqueeze_1274: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1273, 3);  unsqueeze_1273 = None
    sum_174: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_159, [0, 2, 3])
    sub_395: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1274)
    mul_1522: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_159, sub_395);  sub_395 = None
    sum_175: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1522, [0, 2, 3]);  mul_1522 = None
    mul_1523: "f32[48]" = torch.ops.aten.mul.Tensor(sum_174, 3.985969387755102e-05)
    unsqueeze_1275: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1523, 0);  mul_1523 = None
    unsqueeze_1276: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 2);  unsqueeze_1275 = None
    unsqueeze_1277: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, 3);  unsqueeze_1276 = None
    mul_1524: "f32[48]" = torch.ops.aten.mul.Tensor(sum_175, 3.985969387755102e-05)
    mul_1525: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1526: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1524, mul_1525);  mul_1524 = mul_1525 = None
    unsqueeze_1278: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1526, 0);  mul_1526 = None
    unsqueeze_1279: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 2);  unsqueeze_1278 = None
    unsqueeze_1280: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1279, 3);  unsqueeze_1279 = None
    mul_1527: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_1281: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1527, 0);  mul_1527 = None
    unsqueeze_1282: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1281, 2);  unsqueeze_1281 = None
    unsqueeze_1283: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 3);  unsqueeze_1282 = None
    sub_396: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_1274);  convolution_9 = unsqueeze_1274 = None
    mul_1528: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_1280);  sub_396 = unsqueeze_1280 = None
    sub_397: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_159, mul_1528);  where_159 = mul_1528 = None
    sub_398: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_397, unsqueeze_1277);  sub_397 = unsqueeze_1277 = None
    mul_1529: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_398, unsqueeze_1283);  sub_398 = unsqueeze_1283 = None
    mul_1530: "f32[48]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_28);  sum_175 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_114 = torch.ops.aten.convolution_backward.default(mul_1529, div_5, primals_186, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, False]);  mul_1529 = div_5 = primals_186 = None
    getitem_516: "f32[8, 48, 56, 56]" = convolution_backward_114[0]
    getitem_517: "f32[48, 1, 5, 5]" = convolution_backward_114[1];  convolution_backward_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_89: "b8[8, 48, 56, 56]" = torch.ops.aten.lt.Scalar(clone_5, -3)
    le_71: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(clone_5, 3)
    div_185: "f32[8, 48, 56, 56]" = torch.ops.aten.div.Tensor(clone_5, 3);  clone_5 = None
    add_662: "f32[8, 48, 56, 56]" = torch.ops.aten.add.Tensor(div_185, 0.5);  div_185 = None
    mul_1531: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_516, add_662);  add_662 = None
    where_160: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_71, mul_1531, getitem_516);  le_71 = mul_1531 = getitem_516 = None
    scalar_tensor_89: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_161: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(lt_89, scalar_tensor_89, where_160);  lt_89 = scalar_tensor_89 = where_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1284: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
    unsqueeze_1285: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, 2);  unsqueeze_1284 = None
    unsqueeze_1286: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1285, 3);  unsqueeze_1285 = None
    sum_176: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_161, [0, 2, 3])
    sub_399: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1286)
    mul_1532: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_161, sub_399);  sub_399 = None
    sum_177: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_1532, [0, 2, 3]);  mul_1532 = None
    mul_1533: "f32[48]" = torch.ops.aten.mul.Tensor(sum_176, 3.985969387755102e-05)
    unsqueeze_1287: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1533, 0);  mul_1533 = None
    unsqueeze_1288: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1287, 2);  unsqueeze_1287 = None
    unsqueeze_1289: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, 3);  unsqueeze_1288 = None
    mul_1534: "f32[48]" = torch.ops.aten.mul.Tensor(sum_177, 3.985969387755102e-05)
    mul_1535: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1536: "f32[48]" = torch.ops.aten.mul.Tensor(mul_1534, mul_1535);  mul_1534 = mul_1535 = None
    unsqueeze_1290: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1536, 0);  mul_1536 = None
    unsqueeze_1291: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 2);  unsqueeze_1290 = None
    unsqueeze_1292: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 3);  unsqueeze_1291 = None
    mul_1537: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_1293: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_1537, 0);  mul_1537 = None
    unsqueeze_1294: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1293, 2);  unsqueeze_1293 = None
    unsqueeze_1295: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 3);  unsqueeze_1294 = None
    sub_400: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_1286);  convolution_8 = unsqueeze_1286 = None
    mul_1538: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_1292);  sub_400 = unsqueeze_1292 = None
    sub_401: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_161, mul_1538);  where_161 = mul_1538 = None
    sub_402: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_401, unsqueeze_1289);  sub_401 = unsqueeze_1289 = None
    mul_1539: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_402, unsqueeze_1295);  sub_402 = unsqueeze_1295 = None
    mul_1540: "f32[48]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_25);  sum_177 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_115 = torch.ops.aten.convolution_backward.default(mul_1539, add_46, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1539 = add_46 = primals_185 = None
    getitem_519: "f32[8, 24, 56, 56]" = convolution_backward_115[0]
    getitem_520: "f32[48, 24, 1, 1]" = convolution_backward_115[1];  convolution_backward_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_663: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_660, getitem_519);  add_660 = getitem_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1296: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
    unsqueeze_1297: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, 2);  unsqueeze_1296 = None
    unsqueeze_1298: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1297, 3);  unsqueeze_1297 = None
    sum_178: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_663, [0, 2, 3])
    sub_403: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1298)
    mul_1541: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_663, sub_403);  sub_403 = None
    sum_179: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_1541, [0, 2, 3]);  mul_1541 = None
    mul_1542: "f32[24]" = torch.ops.aten.mul.Tensor(sum_178, 3.985969387755102e-05)
    unsqueeze_1299: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1542, 0);  mul_1542 = None
    unsqueeze_1300: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1299, 2);  unsqueeze_1299 = None
    unsqueeze_1301: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, 3);  unsqueeze_1300 = None
    mul_1543: "f32[24]" = torch.ops.aten.mul.Tensor(sum_179, 3.985969387755102e-05)
    mul_1544: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1545: "f32[24]" = torch.ops.aten.mul.Tensor(mul_1543, mul_1544);  mul_1543 = mul_1544 = None
    unsqueeze_1302: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1545, 0);  mul_1545 = None
    unsqueeze_1303: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 2);  unsqueeze_1302 = None
    unsqueeze_1304: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 3);  unsqueeze_1303 = None
    mul_1546: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_1305: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_1546, 0);  mul_1546 = None
    unsqueeze_1306: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1305, 2);  unsqueeze_1305 = None
    unsqueeze_1307: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 3);  unsqueeze_1306 = None
    sub_404: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_1298);  convolution_7 = unsqueeze_1298 = None
    mul_1547: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_404, unsqueeze_1304);  sub_404 = unsqueeze_1304 = None
    sub_405: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_663, mul_1547);  add_663 = mul_1547 = None
    sub_406: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_405, unsqueeze_1301);  sub_405 = unsqueeze_1301 = None
    mul_1548: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_406, unsqueeze_1307);  sub_406 = unsqueeze_1307 = None
    mul_1549: "f32[24]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_22);  sum_179 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_116 = torch.ops.aten.convolution_backward.default(mul_1548, div_4, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1548 = div_4 = primals_184 = None
    getitem_522: "f32[8, 64, 56, 56]" = convolution_backward_116[0]
    getitem_523: "f32[24, 64, 1, 1]" = convolution_backward_116[1];  convolution_backward_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_90: "b8[8, 64, 56, 56]" = torch.ops.aten.lt.Scalar(clone_4, -3)
    le_72: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(clone_4, 3)
    div_186: "f32[8, 64, 56, 56]" = torch.ops.aten.div.Tensor(clone_4, 3);  clone_4 = None
    add_664: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(div_186, 0.5);  div_186 = None
    mul_1550: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_522, add_664);  add_664 = None
    where_162: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_72, mul_1550, getitem_522);  le_72 = mul_1550 = getitem_522 = None
    scalar_tensor_90: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_163: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(lt_90, scalar_tensor_90, where_162);  lt_90 = scalar_tensor_90 = where_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1308: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
    unsqueeze_1309: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, 2);  unsqueeze_1308 = None
    unsqueeze_1310: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1309, 3);  unsqueeze_1309 = None
    sum_180: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_163, [0, 2, 3])
    sub_407: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1310)
    mul_1551: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_163, sub_407);  sub_407 = None
    sum_181: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1551, [0, 2, 3]);  mul_1551 = None
    mul_1552: "f32[64]" = torch.ops.aten.mul.Tensor(sum_180, 3.985969387755102e-05)
    unsqueeze_1311: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1552, 0);  mul_1552 = None
    unsqueeze_1312: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1311, 2);  unsqueeze_1311 = None
    unsqueeze_1313: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, 3);  unsqueeze_1312 = None
    mul_1553: "f32[64]" = torch.ops.aten.mul.Tensor(sum_181, 3.985969387755102e-05)
    mul_1554: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1555: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1553, mul_1554);  mul_1553 = mul_1554 = None
    unsqueeze_1314: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1555, 0);  mul_1555 = None
    unsqueeze_1315: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 2);  unsqueeze_1314 = None
    unsqueeze_1316: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 3);  unsqueeze_1315 = None
    mul_1556: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_1317: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1556, 0);  mul_1556 = None
    unsqueeze_1318: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1317, 2);  unsqueeze_1317 = None
    unsqueeze_1319: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 3);  unsqueeze_1318 = None
    sub_408: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_1310);  convolution_6 = unsqueeze_1310 = None
    mul_1557: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_408, unsqueeze_1316);  sub_408 = unsqueeze_1316 = None
    sub_409: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_163, mul_1557);  where_163 = mul_1557 = None
    sub_410: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_409, unsqueeze_1313);  sub_409 = unsqueeze_1313 = None
    mul_1558: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_410, unsqueeze_1319);  sub_410 = unsqueeze_1319 = None
    mul_1559: "f32[64]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_19);  sum_181 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_117 = torch.ops.aten.convolution_backward.default(mul_1558, div_3, primals_183, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 64, [True, True, False]);  mul_1558 = div_3 = primals_183 = None
    getitem_525: "f32[8, 64, 112, 112]" = convolution_backward_117[0]
    getitem_526: "f32[64, 1, 5, 5]" = convolution_backward_117[1];  convolution_backward_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_91: "b8[8, 64, 112, 112]" = torch.ops.aten.lt.Scalar(clone_3, -3)
    le_73: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(clone_3, 3)
    div_187: "f32[8, 64, 112, 112]" = torch.ops.aten.div.Tensor(clone_3, 3);  clone_3 = None
    add_665: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(div_187, 0.5);  div_187 = None
    mul_1560: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_525, add_665);  add_665 = None
    where_164: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_73, mul_1560, getitem_525);  le_73 = mul_1560 = getitem_525 = None
    scalar_tensor_91: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_165: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(lt_91, scalar_tensor_91, where_164);  lt_91 = scalar_tensor_91 = where_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1320: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
    unsqueeze_1321: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, 2);  unsqueeze_1320 = None
    unsqueeze_1322: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1321, 3);  unsqueeze_1321 = None
    sum_182: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_165, [0, 2, 3])
    sub_411: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1322)
    mul_1561: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_165, sub_411);  sub_411 = None
    sum_183: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1561, [0, 2, 3]);  mul_1561 = None
    mul_1562: "f32[64]" = torch.ops.aten.mul.Tensor(sum_182, 9.964923469387754e-06)
    unsqueeze_1323: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1562, 0);  mul_1562 = None
    unsqueeze_1324: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1323, 2);  unsqueeze_1323 = None
    unsqueeze_1325: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, 3);  unsqueeze_1324 = None
    mul_1563: "f32[64]" = torch.ops.aten.mul.Tensor(sum_183, 9.964923469387754e-06)
    mul_1564: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1565: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1563, mul_1564);  mul_1563 = mul_1564 = None
    unsqueeze_1326: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1565, 0);  mul_1565 = None
    unsqueeze_1327: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 2);  unsqueeze_1326 = None
    unsqueeze_1328: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 3);  unsqueeze_1327 = None
    mul_1566: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_1329: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1566, 0);  mul_1566 = None
    unsqueeze_1330: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1329, 2);  unsqueeze_1329 = None
    unsqueeze_1331: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, 3);  unsqueeze_1330 = None
    sub_412: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1322);  convolution_5 = unsqueeze_1322 = None
    mul_1567: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_412, unsqueeze_1328);  sub_412 = unsqueeze_1328 = None
    sub_413: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_165, mul_1567);  where_165 = mul_1567 = None
    sub_414: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_413, unsqueeze_1325);  sub_413 = unsqueeze_1325 = None
    mul_1568: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_414, unsqueeze_1331);  sub_414 = unsqueeze_1331 = None
    mul_1569: "f32[64]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_16);  sum_183 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_118 = torch.ops.aten.convolution_backward.default(mul_1568, add_29, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1568 = add_29 = primals_182 = None
    getitem_528: "f32[8, 16, 112, 112]" = convolution_backward_118[0]
    getitem_529: "f32[64, 16, 1, 1]" = convolution_backward_118[1];  convolution_backward_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1332: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
    unsqueeze_1333: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, 2);  unsqueeze_1332 = None
    unsqueeze_1334: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1333, 3);  unsqueeze_1333 = None
    sum_184: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_528, [0, 2, 3])
    sub_415: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1334)
    mul_1570: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_528, sub_415);  sub_415 = None
    sum_185: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1570, [0, 2, 3]);  mul_1570 = None
    mul_1571: "f32[16]" = torch.ops.aten.mul.Tensor(sum_184, 9.964923469387754e-06)
    unsqueeze_1335: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1571, 0);  mul_1571 = None
    unsqueeze_1336: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1335, 2);  unsqueeze_1335 = None
    unsqueeze_1337: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, 3);  unsqueeze_1336 = None
    mul_1572: "f32[16]" = torch.ops.aten.mul.Tensor(sum_185, 9.964923469387754e-06)
    mul_1573: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1574: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1572, mul_1573);  mul_1572 = mul_1573 = None
    unsqueeze_1338: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1574, 0);  mul_1574 = None
    unsqueeze_1339: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 2);  unsqueeze_1338 = None
    unsqueeze_1340: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1339, 3);  unsqueeze_1339 = None
    mul_1575: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_1341: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1575, 0);  mul_1575 = None
    unsqueeze_1342: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1341, 2);  unsqueeze_1341 = None
    unsqueeze_1343: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, 3);  unsqueeze_1342 = None
    sub_416: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1334);  convolution_4 = unsqueeze_1334 = None
    mul_1576: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_416, unsqueeze_1340);  sub_416 = unsqueeze_1340 = None
    sub_417: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(getitem_528, mul_1576);  mul_1576 = None
    sub_418: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_417, unsqueeze_1337);  sub_417 = unsqueeze_1337 = None
    mul_1577: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_418, unsqueeze_1343);  sub_418 = unsqueeze_1343 = None
    mul_1578: "f32[16]" = torch.ops.aten.mul.Tensor(sum_185, squeeze_13);  sum_185 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_119 = torch.ops.aten.convolution_backward.default(mul_1577, div_2, primals_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1577 = div_2 = primals_181 = None
    getitem_531: "f32[8, 16, 112, 112]" = convolution_backward_119[0]
    getitem_532: "f32[16, 16, 1, 1]" = convolution_backward_119[1];  convolution_backward_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_92: "b8[8, 16, 112, 112]" = torch.ops.aten.lt.Scalar(clone_2, -3)
    le_74: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(clone_2, 3)
    div_188: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(clone_2, 3);  clone_2 = None
    add_666: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_188, 0.5);  div_188 = None
    mul_1579: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_531, add_666);  add_666 = None
    where_166: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_74, mul_1579, getitem_531);  le_74 = mul_1579 = getitem_531 = None
    scalar_tensor_92: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_167: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(lt_92, scalar_tensor_92, where_166);  lt_92 = scalar_tensor_92 = where_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1344: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
    unsqueeze_1345: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, 2);  unsqueeze_1344 = None
    unsqueeze_1346: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1345, 3);  unsqueeze_1345 = None
    sum_186: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_167, [0, 2, 3])
    sub_419: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1346)
    mul_1580: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_167, sub_419);  sub_419 = None
    sum_187: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1580, [0, 2, 3]);  mul_1580 = None
    mul_1581: "f32[16]" = torch.ops.aten.mul.Tensor(sum_186, 9.964923469387754e-06)
    unsqueeze_1347: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1581, 0);  mul_1581 = None
    unsqueeze_1348: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1347, 2);  unsqueeze_1347 = None
    unsqueeze_1349: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, 3);  unsqueeze_1348 = None
    mul_1582: "f32[16]" = torch.ops.aten.mul.Tensor(sum_187, 9.964923469387754e-06)
    mul_1583: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1584: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1582, mul_1583);  mul_1582 = mul_1583 = None
    unsqueeze_1350: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1584, 0);  mul_1584 = None
    unsqueeze_1351: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 2);  unsqueeze_1350 = None
    unsqueeze_1352: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 3);  unsqueeze_1351 = None
    mul_1585: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_1353: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1585, 0);  mul_1585 = None
    unsqueeze_1354: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1353, 2);  unsqueeze_1353 = None
    unsqueeze_1355: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 3);  unsqueeze_1354 = None
    sub_420: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1346);  convolution_3 = unsqueeze_1346 = None
    mul_1586: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_420, unsqueeze_1352);  sub_420 = unsqueeze_1352 = None
    sub_421: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_167, mul_1586);  where_167 = mul_1586 = None
    sub_422: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_421, unsqueeze_1349);  sub_421 = unsqueeze_1349 = None
    mul_1587: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_422, unsqueeze_1355);  sub_422 = unsqueeze_1355 = None
    mul_1588: "f32[16]" = torch.ops.aten.mul.Tensor(sum_187, squeeze_10);  sum_187 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_120 = torch.ops.aten.convolution_backward.default(mul_1587, add_17, primals_180, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_1587 = add_17 = primals_180 = None
    getitem_534: "f32[8, 16, 112, 112]" = convolution_backward_120[0]
    getitem_535: "f32[16, 1, 3, 3]" = convolution_backward_120[1];  convolution_backward_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    add_667: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(getitem_528, getitem_534);  getitem_528 = getitem_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1356: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
    unsqueeze_1357: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, 2);  unsqueeze_1356 = None
    unsqueeze_1358: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1357, 3);  unsqueeze_1357 = None
    sum_188: "f32[16]" = torch.ops.aten.sum.dim_IntList(add_667, [0, 2, 3])
    sub_423: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1358)
    mul_1589: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_667, sub_423);  sub_423 = None
    sum_189: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1589, [0, 2, 3]);  mul_1589 = None
    mul_1590: "f32[16]" = torch.ops.aten.mul.Tensor(sum_188, 9.964923469387754e-06)
    unsqueeze_1359: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1590, 0);  mul_1590 = None
    unsqueeze_1360: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1359, 2);  unsqueeze_1359 = None
    unsqueeze_1361: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, 3);  unsqueeze_1360 = None
    mul_1591: "f32[16]" = torch.ops.aten.mul.Tensor(sum_189, 9.964923469387754e-06)
    mul_1592: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1593: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1591, mul_1592);  mul_1591 = mul_1592 = None
    unsqueeze_1362: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1593, 0);  mul_1593 = None
    unsqueeze_1363: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 2);  unsqueeze_1362 = None
    unsqueeze_1364: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1363, 3);  unsqueeze_1363 = None
    mul_1594: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_1365: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1594, 0);  mul_1594 = None
    unsqueeze_1366: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1365, 2);  unsqueeze_1365 = None
    unsqueeze_1367: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, 3);  unsqueeze_1366 = None
    sub_424: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1358);  convolution_2 = unsqueeze_1358 = None
    mul_1595: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_424, unsqueeze_1364);  sub_424 = unsqueeze_1364 = None
    sub_425: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(add_667, mul_1595);  mul_1595 = None
    sub_426: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_425, unsqueeze_1361);  sub_425 = unsqueeze_1361 = None
    mul_1596: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_426, unsqueeze_1367);  sub_426 = unsqueeze_1367 = None
    mul_1597: "f32[16]" = torch.ops.aten.mul.Tensor(sum_189, squeeze_7);  sum_189 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_121 = torch.ops.aten.convolution_backward.default(mul_1596, div_1, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1596 = div_1 = primals_179 = None
    getitem_537: "f32[8, 16, 112, 112]" = convolution_backward_121[0]
    getitem_538: "f32[16, 16, 1, 1]" = convolution_backward_121[1];  convolution_backward_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_93: "b8[8, 16, 112, 112]" = torch.ops.aten.lt.Scalar(clone_1, -3)
    le_75: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(clone_1, 3)
    div_189: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(clone_1, 3);  clone_1 = None
    add_668: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_189, 0.5);  div_189 = None
    mul_1598: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_537, add_668);  add_668 = None
    where_168: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_75, mul_1598, getitem_537);  le_75 = mul_1598 = getitem_537 = None
    scalar_tensor_93: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_169: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(lt_93, scalar_tensor_93, where_168);  lt_93 = scalar_tensor_93 = where_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1368: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
    unsqueeze_1369: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, 2);  unsqueeze_1368 = None
    unsqueeze_1370: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1369, 3);  unsqueeze_1369 = None
    sum_190: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_169, [0, 2, 3])
    sub_427: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1370)
    mul_1599: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_169, sub_427);  sub_427 = None
    sum_191: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1599, [0, 2, 3]);  mul_1599 = None
    mul_1600: "f32[16]" = torch.ops.aten.mul.Tensor(sum_190, 9.964923469387754e-06)
    unsqueeze_1371: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1600, 0);  mul_1600 = None
    unsqueeze_1372: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1371, 2);  unsqueeze_1371 = None
    unsqueeze_1373: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, 3);  unsqueeze_1372 = None
    mul_1601: "f32[16]" = torch.ops.aten.mul.Tensor(sum_191, 9.964923469387754e-06)
    mul_1602: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1603: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1601, mul_1602);  mul_1601 = mul_1602 = None
    unsqueeze_1374: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1603, 0);  mul_1603 = None
    unsqueeze_1375: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 2);  unsqueeze_1374 = None
    unsqueeze_1376: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1375, 3);  unsqueeze_1375 = None
    mul_1604: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_1377: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1604, 0);  mul_1604 = None
    unsqueeze_1378: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1377, 2);  unsqueeze_1377 = None
    unsqueeze_1379: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, 3);  unsqueeze_1378 = None
    sub_428: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1370);  convolution_1 = unsqueeze_1370 = None
    mul_1605: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_428, unsqueeze_1376);  sub_428 = unsqueeze_1376 = None
    sub_429: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_169, mul_1605);  where_169 = mul_1605 = None
    sub_430: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_429, unsqueeze_1373);  sub_429 = unsqueeze_1373 = None
    mul_1606: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_430, unsqueeze_1379);  sub_430 = unsqueeze_1379 = None
    mul_1607: "f32[16]" = torch.ops.aten.mul.Tensor(sum_191, squeeze_4);  sum_191 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_122 = torch.ops.aten.convolution_backward.default(mul_1606, div, primals_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_1606 = div = primals_178 = None
    getitem_540: "f32[8, 16, 112, 112]" = convolution_backward_122[0]
    getitem_541: "f32[16, 1, 3, 3]" = convolution_backward_122[1];  convolution_backward_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    add_669: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(add_667, getitem_540);  add_667 = getitem_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    lt_94: "b8[8, 16, 112, 112]" = torch.ops.aten.lt.Scalar(clone, -3)
    le_76: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(clone, 3)
    div_190: "f32[8, 16, 112, 112]" = torch.ops.aten.div.Tensor(clone, 3);  clone = None
    add_670: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_190, 0.5);  div_190 = None
    mul_1608: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_669, add_670);  add_670 = None
    where_170: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_76, mul_1608, add_669);  le_76 = mul_1608 = add_669 = None
    scalar_tensor_94: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_171: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(lt_94, scalar_tensor_94, where_170);  lt_94 = scalar_tensor_94 = where_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    unsqueeze_1380: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    unsqueeze_1381: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, 2);  unsqueeze_1380 = None
    unsqueeze_1382: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1381, 3);  unsqueeze_1381 = None
    sum_192: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_171, [0, 2, 3])
    sub_431: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1382)
    mul_1609: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_171, sub_431);  sub_431 = None
    sum_193: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1609, [0, 2, 3]);  mul_1609 = None
    mul_1610: "f32[16]" = torch.ops.aten.mul.Tensor(sum_192, 9.964923469387754e-06)
    unsqueeze_1383: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1610, 0);  mul_1610 = None
    unsqueeze_1384: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1383, 2);  unsqueeze_1383 = None
    unsqueeze_1385: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, 3);  unsqueeze_1384 = None
    mul_1611: "f32[16]" = torch.ops.aten.mul.Tensor(sum_193, 9.964923469387754e-06)
    mul_1612: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1613: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1611, mul_1612);  mul_1611 = mul_1612 = None
    unsqueeze_1386: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1613, 0);  mul_1613 = None
    unsqueeze_1387: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 2);  unsqueeze_1386 = None
    unsqueeze_1388: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1387, 3);  unsqueeze_1387 = None
    mul_1614: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_1389: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1614, 0);  mul_1614 = None
    unsqueeze_1390: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1389, 2);  unsqueeze_1389 = None
    unsqueeze_1391: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, 3);  unsqueeze_1390 = None
    sub_432: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1382);  convolution = unsqueeze_1382 = None
    mul_1615: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_432, unsqueeze_1388);  sub_432 = unsqueeze_1388 = None
    sub_433: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_171, mul_1615);  where_171 = mul_1615 = None
    sub_434: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_433, unsqueeze_1385);  sub_433 = unsqueeze_1385 = None
    mul_1616: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_434, unsqueeze_1391);  sub_434 = unsqueeze_1391 = None
    mul_1617: "f32[16]" = torch.ops.aten.mul.Tensor(sum_193, squeeze_1);  sum_193 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilenetv3.py:135, code: x = self.conv_stem(x)
    convolution_backward_123 = torch.ops.aten.convolution_backward.default(mul_1616, primals_598, primals_177, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1616 = primals_598 = primals_177 = None
    getitem_544: "f32[16, 3, 3, 3]" = convolution_backward_123[1];  convolution_backward_123 = None
    
    # No stacktrace found for following nodes
    copy_: "i64[]" = torch.ops.aten.copy_.default(primals_337, add);  primals_337 = add = None
    copy__1: "f32[16]" = torch.ops.aten.copy_.default(primals_338, add_2);  primals_338 = add_2 = None
    copy__2: "f32[16]" = torch.ops.aten.copy_.default(primals_339, add_3);  primals_339 = add_3 = None
    copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_340, add_6);  primals_340 = add_6 = None
    copy__4: "f32[16]" = torch.ops.aten.copy_.default(primals_341, add_8);  primals_341 = add_8 = None
    copy__5: "f32[16]" = torch.ops.aten.copy_.default(primals_342, add_9);  primals_342 = add_9 = None
    copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_343, add_12);  primals_343 = add_12 = None
    copy__7: "f32[16]" = torch.ops.aten.copy_.default(primals_344, add_14);  primals_344 = add_14 = None
    copy__8: "f32[16]" = torch.ops.aten.copy_.default(primals_345, add_15);  primals_345 = add_15 = None
    copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_346, add_18);  primals_346 = add_18 = None
    copy__10: "f32[16]" = torch.ops.aten.copy_.default(primals_347, add_20);  primals_347 = add_20 = None
    copy__11: "f32[16]" = torch.ops.aten.copy_.default(primals_348, add_21);  primals_348 = add_21 = None
    copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_349, add_24);  primals_349 = add_24 = None
    copy__13: "f32[16]" = torch.ops.aten.copy_.default(primals_350, add_26);  primals_350 = add_26 = None
    copy__14: "f32[16]" = torch.ops.aten.copy_.default(primals_351, add_27);  primals_351 = add_27 = None
    copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_352, add_30);  primals_352 = add_30 = None
    copy__16: "f32[64]" = torch.ops.aten.copy_.default(primals_353, add_32);  primals_353 = add_32 = None
    copy__17: "f32[64]" = torch.ops.aten.copy_.default(primals_354, add_33);  primals_354 = add_33 = None
    copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_355, add_36);  primals_355 = add_36 = None
    copy__19: "f32[64]" = torch.ops.aten.copy_.default(primals_356, add_38);  primals_356 = add_38 = None
    copy__20: "f32[64]" = torch.ops.aten.copy_.default(primals_357, add_39);  primals_357 = add_39 = None
    copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_358, add_42);  primals_358 = add_42 = None
    copy__22: "f32[24]" = torch.ops.aten.copy_.default(primals_359, add_44);  primals_359 = add_44 = None
    copy__23: "f32[24]" = torch.ops.aten.copy_.default(primals_360, add_45);  primals_360 = add_45 = None
    copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_361, add_47);  primals_361 = add_47 = None
    copy__25: "f32[48]" = torch.ops.aten.copy_.default(primals_362, add_49);  primals_362 = add_49 = None
    copy__26: "f32[48]" = torch.ops.aten.copy_.default(primals_363, add_50);  primals_363 = add_50 = None
    copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_364, add_53);  primals_364 = add_53 = None
    copy__28: "f32[48]" = torch.ops.aten.copy_.default(primals_365, add_55);  primals_365 = add_55 = None
    copy__29: "f32[48]" = torch.ops.aten.copy_.default(primals_366, add_56);  primals_366 = add_56 = None
    copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_367, add_59);  primals_367 = add_59 = None
    copy__31: "f32[24]" = torch.ops.aten.copy_.default(primals_368, add_61);  primals_368 = add_61 = None
    copy__32: "f32[24]" = torch.ops.aten.copy_.default(primals_369, add_62);  primals_369 = add_62 = None
    copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_370, add_65);  primals_370 = add_65 = None
    copy__34: "f32[48]" = torch.ops.aten.copy_.default(primals_371, add_67);  primals_371 = add_67 = None
    copy__35: "f32[48]" = torch.ops.aten.copy_.default(primals_372, add_68);  primals_372 = add_68 = None
    copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_373, add_71);  primals_373 = add_71 = None
    copy__37: "f32[48]" = torch.ops.aten.copy_.default(primals_374, add_73);  primals_374 = add_73 = None
    copy__38: "f32[48]" = torch.ops.aten.copy_.default(primals_375, add_74);  primals_375 = add_74 = None
    copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_376, add_77);  primals_376 = add_77 = None
    copy__40: "f32[24]" = torch.ops.aten.copy_.default(primals_377, add_79);  primals_377 = add_79 = None
    copy__41: "f32[24]" = torch.ops.aten.copy_.default(primals_378, add_80);  primals_378 = add_80 = None
    copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_379, add_83);  primals_379 = add_83 = None
    copy__43: "f32[48]" = torch.ops.aten.copy_.default(primals_380, add_85);  primals_380 = add_85 = None
    copy__44: "f32[48]" = torch.ops.aten.copy_.default(primals_381, add_86);  primals_381 = add_86 = None
    copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_382, add_89);  primals_382 = add_89 = None
    copy__46: "f32[48]" = torch.ops.aten.copy_.default(primals_383, add_91);  primals_383 = add_91 = None
    copy__47: "f32[48]" = torch.ops.aten.copy_.default(primals_384, add_92);  primals_384 = add_92 = None
    copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_385, add_95);  primals_385 = add_95 = None
    copy__49: "f32[24]" = torch.ops.aten.copy_.default(primals_386, add_97);  primals_386 = add_97 = None
    copy__50: "f32[24]" = torch.ops.aten.copy_.default(primals_387, add_98);  primals_387 = add_98 = None
    copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_388, add_101);  primals_388 = add_101 = None
    copy__52: "f32[120]" = torch.ops.aten.copy_.default(primals_389, add_103);  primals_389 = add_103 = None
    copy__53: "f32[120]" = torch.ops.aten.copy_.default(primals_390, add_104);  primals_390 = add_104 = None
    copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_391, add_107);  primals_391 = add_107 = None
    copy__55: "f32[120]" = torch.ops.aten.copy_.default(primals_392, add_109);  primals_392 = add_109 = None
    copy__56: "f32[120]" = torch.ops.aten.copy_.default(primals_393, add_110);  primals_393 = add_110 = None
    copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_394, add_115);  primals_394 = add_115 = None
    copy__58: "f32[40]" = torch.ops.aten.copy_.default(primals_395, add_117);  primals_395 = add_117 = None
    copy__59: "f32[40]" = torch.ops.aten.copy_.default(primals_396, add_118);  primals_396 = add_118 = None
    copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_397, add_120);  primals_397 = add_120 = None
    copy__61: "f32[120]" = torch.ops.aten.copy_.default(primals_398, add_122);  primals_398 = add_122 = None
    copy__62: "f32[120]" = torch.ops.aten.copy_.default(primals_399, add_123);  primals_399 = add_123 = None
    copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_400, add_126);  primals_400 = add_126 = None
    copy__64: "f32[120]" = torch.ops.aten.copy_.default(primals_401, add_128);  primals_401 = add_128 = None
    copy__65: "f32[120]" = torch.ops.aten.copy_.default(primals_402, add_129);  primals_402 = add_129 = None
    copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_403, add_134);  primals_403 = add_134 = None
    copy__67: "f32[40]" = torch.ops.aten.copy_.default(primals_404, add_136);  primals_404 = add_136 = None
    copy__68: "f32[40]" = torch.ops.aten.copy_.default(primals_405, add_137);  primals_405 = add_137 = None
    copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_406, add_140);  primals_406 = add_140 = None
    copy__70: "f32[120]" = torch.ops.aten.copy_.default(primals_407, add_142);  primals_407 = add_142 = None
    copy__71: "f32[120]" = torch.ops.aten.copy_.default(primals_408, add_143);  primals_408 = add_143 = None
    copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_409, add_146);  primals_409 = add_146 = None
    copy__73: "f32[120]" = torch.ops.aten.copy_.default(primals_410, add_148);  primals_410 = add_148 = None
    copy__74: "f32[120]" = torch.ops.aten.copy_.default(primals_411, add_149);  primals_411 = add_149 = None
    copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_412, add_154);  primals_412 = add_154 = None
    copy__76: "f32[40]" = torch.ops.aten.copy_.default(primals_413, add_156);  primals_413 = add_156 = None
    copy__77: "f32[40]" = torch.ops.aten.copy_.default(primals_414, add_157);  primals_414 = add_157 = None
    copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_415, add_160);  primals_415 = add_160 = None
    copy__79: "f32[120]" = torch.ops.aten.copy_.default(primals_416, add_162);  primals_416 = add_162 = None
    copy__80: "f32[120]" = torch.ops.aten.copy_.default(primals_417, add_163);  primals_417 = add_163 = None
    copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_418, add_166);  primals_418 = add_166 = None
    copy__82: "f32[120]" = torch.ops.aten.copy_.default(primals_419, add_168);  primals_419 = add_168 = None
    copy__83: "f32[120]" = torch.ops.aten.copy_.default(primals_420, add_169);  primals_420 = add_169 = None
    copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_421, add_174);  primals_421 = add_174 = None
    copy__85: "f32[40]" = torch.ops.aten.copy_.default(primals_422, add_176);  primals_422 = add_176 = None
    copy__86: "f32[40]" = torch.ops.aten.copy_.default(primals_423, add_177);  primals_423 = add_177 = None
    copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_424, add_180);  primals_424 = add_180 = None
    copy__88: "f32[120]" = torch.ops.aten.copy_.default(primals_425, add_182);  primals_425 = add_182 = None
    copy__89: "f32[120]" = torch.ops.aten.copy_.default(primals_426, add_183);  primals_426 = add_183 = None
    copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_427, add_186);  primals_427 = add_186 = None
    copy__91: "f32[120]" = torch.ops.aten.copy_.default(primals_428, add_188);  primals_428 = add_188 = None
    copy__92: "f32[120]" = torch.ops.aten.copy_.default(primals_429, add_189);  primals_429 = add_189 = None
    copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_430, add_194);  primals_430 = add_194 = None
    copy__94: "f32[40]" = torch.ops.aten.copy_.default(primals_431, add_196);  primals_431 = add_196 = None
    copy__95: "f32[40]" = torch.ops.aten.copy_.default(primals_432, add_197);  primals_432 = add_197 = None
    copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_433, add_200);  primals_433 = add_200 = None
    copy__97: "f32[200]" = torch.ops.aten.copy_.default(primals_434, add_202);  primals_434 = add_202 = None
    copy__98: "f32[200]" = torch.ops.aten.copy_.default(primals_435, add_203);  primals_435 = add_203 = None
    copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_436, add_206);  primals_436 = add_206 = None
    copy__100: "f32[200]" = torch.ops.aten.copy_.default(primals_437, add_208);  primals_437 = add_208 = None
    copy__101: "f32[200]" = torch.ops.aten.copy_.default(primals_438, add_209);  primals_438 = add_209 = None
    copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_439, add_212);  primals_439 = add_212 = None
    copy__103: "f32[72]" = torch.ops.aten.copy_.default(primals_440, add_214);  primals_440 = add_214 = None
    copy__104: "f32[72]" = torch.ops.aten.copy_.default(primals_441, add_215);  primals_441 = add_215 = None
    copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_442, add_217);  primals_442 = add_217 = None
    copy__106: "f32[216]" = torch.ops.aten.copy_.default(primals_443, add_219);  primals_443 = add_219 = None
    copy__107: "f32[216]" = torch.ops.aten.copy_.default(primals_444, add_220);  primals_444 = add_220 = None
    copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_445, add_223);  primals_445 = add_223 = None
    copy__109: "f32[216]" = torch.ops.aten.copy_.default(primals_446, add_225);  primals_446 = add_225 = None
    copy__110: "f32[216]" = torch.ops.aten.copy_.default(primals_447, add_226);  primals_447 = add_226 = None
    copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_448, add_229);  primals_448 = add_229 = None
    copy__112: "f32[72]" = torch.ops.aten.copy_.default(primals_449, add_231);  primals_449 = add_231 = None
    copy__113: "f32[72]" = torch.ops.aten.copy_.default(primals_450, add_232);  primals_450 = add_232 = None
    copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_451, add_235);  primals_451 = add_235 = None
    copy__115: "f32[216]" = torch.ops.aten.copy_.default(primals_452, add_237);  primals_452 = add_237 = None
    copy__116: "f32[216]" = torch.ops.aten.copy_.default(primals_453, add_238);  primals_453 = add_238 = None
    copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_454, add_241);  primals_454 = add_241 = None
    copy__118: "f32[216]" = torch.ops.aten.copy_.default(primals_455, add_243);  primals_455 = add_243 = None
    copy__119: "f32[216]" = torch.ops.aten.copy_.default(primals_456, add_244);  primals_456 = add_244 = None
    copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_457, add_247);  primals_457 = add_247 = None
    copy__121: "f32[72]" = torch.ops.aten.copy_.default(primals_458, add_249);  primals_458 = add_249 = None
    copy__122: "f32[72]" = torch.ops.aten.copy_.default(primals_459, add_250);  primals_459 = add_250 = None
    copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_460, add_253);  primals_460 = add_253 = None
    copy__124: "f32[216]" = torch.ops.aten.copy_.default(primals_461, add_255);  primals_461 = add_255 = None
    copy__125: "f32[216]" = torch.ops.aten.copy_.default(primals_462, add_256);  primals_462 = add_256 = None
    copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_463, add_259);  primals_463 = add_259 = None
    copy__127: "f32[216]" = torch.ops.aten.copy_.default(primals_464, add_261);  primals_464 = add_261 = None
    copy__128: "f32[216]" = torch.ops.aten.copy_.default(primals_465, add_262);  primals_465 = add_262 = None
    copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_466, add_265);  primals_466 = add_265 = None
    copy__130: "f32[72]" = torch.ops.aten.copy_.default(primals_467, add_267);  primals_467 = add_267 = None
    copy__131: "f32[72]" = torch.ops.aten.copy_.default(primals_468, add_268);  primals_468 = add_268 = None
    copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_469, add_271);  primals_469 = add_271 = None
    copy__133: "f32[216]" = torch.ops.aten.copy_.default(primals_470, add_273);  primals_470 = add_273 = None
    copy__134: "f32[216]" = torch.ops.aten.copy_.default(primals_471, add_274);  primals_471 = add_274 = None
    copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_472, add_277);  primals_472 = add_277 = None
    copy__136: "f32[216]" = torch.ops.aten.copy_.default(primals_473, add_279);  primals_473 = add_279 = None
    copy__137: "f32[216]" = torch.ops.aten.copy_.default(primals_474, add_280);  primals_474 = add_280 = None
    copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_475, add_283);  primals_475 = add_283 = None
    copy__139: "f32[72]" = torch.ops.aten.copy_.default(primals_476, add_285);  primals_476 = add_285 = None
    copy__140: "f32[72]" = torch.ops.aten.copy_.default(primals_477, add_286);  primals_477 = add_286 = None
    copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_478, add_289);  primals_478 = add_289 = None
    copy__142: "f32[360]" = torch.ops.aten.copy_.default(primals_479, add_291);  primals_479 = add_291 = None
    copy__143: "f32[360]" = torch.ops.aten.copy_.default(primals_480, add_292);  primals_480 = add_292 = None
    copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_481, add_295);  primals_481 = add_295 = None
    copy__145: "f32[360]" = torch.ops.aten.copy_.default(primals_482, add_297);  primals_482 = add_297 = None
    copy__146: "f32[360]" = torch.ops.aten.copy_.default(primals_483, add_298);  primals_483 = add_298 = None
    copy__147: "i64[]" = torch.ops.aten.copy_.default(primals_484, add_303);  primals_484 = add_303 = None
    copy__148: "f32[120]" = torch.ops.aten.copy_.default(primals_485, add_305);  primals_485 = add_305 = None
    copy__149: "f32[120]" = torch.ops.aten.copy_.default(primals_486, add_306);  primals_486 = add_306 = None
    copy__150: "i64[]" = torch.ops.aten.copy_.default(primals_487, add_308);  primals_487 = add_308 = None
    copy__151: "f32[360]" = torch.ops.aten.copy_.default(primals_488, add_310);  primals_488 = add_310 = None
    copy__152: "f32[360]" = torch.ops.aten.copy_.default(primals_489, add_311);  primals_489 = add_311 = None
    copy__153: "i64[]" = torch.ops.aten.copy_.default(primals_490, add_314);  primals_490 = add_314 = None
    copy__154: "f32[360]" = torch.ops.aten.copy_.default(primals_491, add_316);  primals_491 = add_316 = None
    copy__155: "f32[360]" = torch.ops.aten.copy_.default(primals_492, add_317);  primals_492 = add_317 = None
    copy__156: "i64[]" = torch.ops.aten.copy_.default(primals_493, add_322);  primals_493 = add_322 = None
    copy__157: "f32[120]" = torch.ops.aten.copy_.default(primals_494, add_324);  primals_494 = add_324 = None
    copy__158: "f32[120]" = torch.ops.aten.copy_.default(primals_495, add_325);  primals_495 = add_325 = None
    copy__159: "i64[]" = torch.ops.aten.copy_.default(primals_496, add_328);  primals_496 = add_328 = None
    copy__160: "f32[360]" = torch.ops.aten.copy_.default(primals_497, add_330);  primals_497 = add_330 = None
    copy__161: "f32[360]" = torch.ops.aten.copy_.default(primals_498, add_331);  primals_498 = add_331 = None
    copy__162: "i64[]" = torch.ops.aten.copy_.default(primals_499, add_334);  primals_499 = add_334 = None
    copy__163: "f32[360]" = torch.ops.aten.copy_.default(primals_500, add_336);  primals_500 = add_336 = None
    copy__164: "f32[360]" = torch.ops.aten.copy_.default(primals_501, add_337);  primals_501 = add_337 = None
    copy__165: "i64[]" = torch.ops.aten.copy_.default(primals_502, add_342);  primals_502 = add_342 = None
    copy__166: "f32[120]" = torch.ops.aten.copy_.default(primals_503, add_344);  primals_503 = add_344 = None
    copy__167: "f32[120]" = torch.ops.aten.copy_.default(primals_504, add_345);  primals_504 = add_345 = None
    copy__168: "i64[]" = torch.ops.aten.copy_.default(primals_505, add_348);  primals_505 = add_348 = None
    copy__169: "f32[360]" = torch.ops.aten.copy_.default(primals_506, add_350);  primals_506 = add_350 = None
    copy__170: "f32[360]" = torch.ops.aten.copy_.default(primals_507, add_351);  primals_507 = add_351 = None
    copy__171: "i64[]" = torch.ops.aten.copy_.default(primals_508, add_354);  primals_508 = add_354 = None
    copy__172: "f32[360]" = torch.ops.aten.copy_.default(primals_509, add_356);  primals_509 = add_356 = None
    copy__173: "f32[360]" = torch.ops.aten.copy_.default(primals_510, add_357);  primals_510 = add_357 = None
    copy__174: "i64[]" = torch.ops.aten.copy_.default(primals_511, add_362);  primals_511 = add_362 = None
    copy__175: "f32[120]" = torch.ops.aten.copy_.default(primals_512, add_364);  primals_512 = add_364 = None
    copy__176: "f32[120]" = torch.ops.aten.copy_.default(primals_513, add_365);  primals_513 = add_365 = None
    copy__177: "i64[]" = torch.ops.aten.copy_.default(primals_514, add_368);  primals_514 = add_368 = None
    copy__178: "f32[360]" = torch.ops.aten.copy_.default(primals_515, add_370);  primals_515 = add_370 = None
    copy__179: "f32[360]" = torch.ops.aten.copy_.default(primals_516, add_371);  primals_516 = add_371 = None
    copy__180: "i64[]" = torch.ops.aten.copy_.default(primals_517, add_374);  primals_517 = add_374 = None
    copy__181: "f32[360]" = torch.ops.aten.copy_.default(primals_518, add_376);  primals_518 = add_376 = None
    copy__182: "f32[360]" = torch.ops.aten.copy_.default(primals_519, add_377);  primals_519 = add_377 = None
    copy__183: "i64[]" = torch.ops.aten.copy_.default(primals_520, add_382);  primals_520 = add_382 = None
    copy__184: "f32[120]" = torch.ops.aten.copy_.default(primals_521, add_384);  primals_521 = add_384 = None
    copy__185: "f32[120]" = torch.ops.aten.copy_.default(primals_522, add_385);  primals_522 = add_385 = None
    copy__186: "i64[]" = torch.ops.aten.copy_.default(primals_523, add_388);  primals_523 = add_388 = None
    copy__187: "f32[360]" = torch.ops.aten.copy_.default(primals_524, add_390);  primals_524 = add_390 = None
    copy__188: "f32[360]" = torch.ops.aten.copy_.default(primals_525, add_391);  primals_525 = add_391 = None
    copy__189: "i64[]" = torch.ops.aten.copy_.default(primals_526, add_394);  primals_526 = add_394 = None
    copy__190: "f32[360]" = torch.ops.aten.copy_.default(primals_527, add_396);  primals_527 = add_396 = None
    copy__191: "f32[360]" = torch.ops.aten.copy_.default(primals_528, add_397);  primals_528 = add_397 = None
    copy__192: "i64[]" = torch.ops.aten.copy_.default(primals_529, add_402);  primals_529 = add_402 = None
    copy__193: "f32[120]" = torch.ops.aten.copy_.default(primals_530, add_404);  primals_530 = add_404 = None
    copy__194: "f32[120]" = torch.ops.aten.copy_.default(primals_531, add_405);  primals_531 = add_405 = None
    copy__195: "i64[]" = torch.ops.aten.copy_.default(primals_532, add_408);  primals_532 = add_408 = None
    copy__196: "f32[720]" = torch.ops.aten.copy_.default(primals_533, add_410);  primals_533 = add_410 = None
    copy__197: "f32[720]" = torch.ops.aten.copy_.default(primals_534, add_411);  primals_534 = add_411 = None
    copy__198: "i64[]" = torch.ops.aten.copy_.default(primals_535, add_414);  primals_535 = add_414 = None
    copy__199: "f32[720]" = torch.ops.aten.copy_.default(primals_536, add_416);  primals_536 = add_416 = None
    copy__200: "f32[720]" = torch.ops.aten.copy_.default(primals_537, add_417);  primals_537 = add_417 = None
    copy__201: "i64[]" = torch.ops.aten.copy_.default(primals_538, add_422);  primals_538 = add_422 = None
    copy__202: "f32[184]" = torch.ops.aten.copy_.default(primals_539, add_424);  primals_539 = add_424 = None
    copy__203: "f32[184]" = torch.ops.aten.copy_.default(primals_540, add_425);  primals_540 = add_425 = None
    copy__204: "i64[]" = torch.ops.aten.copy_.default(primals_541, add_427);  primals_541 = add_427 = None
    copy__205: "f32[736]" = torch.ops.aten.copy_.default(primals_542, add_429);  primals_542 = add_429 = None
    copy__206: "f32[736]" = torch.ops.aten.copy_.default(primals_543, add_430);  primals_543 = add_430 = None
    copy__207: "i64[]" = torch.ops.aten.copy_.default(primals_544, add_433);  primals_544 = add_433 = None
    copy__208: "f32[736]" = torch.ops.aten.copy_.default(primals_545, add_435);  primals_545 = add_435 = None
    copy__209: "f32[736]" = torch.ops.aten.copy_.default(primals_546, add_436);  primals_546 = add_436 = None
    copy__210: "i64[]" = torch.ops.aten.copy_.default(primals_547, add_441);  primals_547 = add_441 = None
    copy__211: "f32[184]" = torch.ops.aten.copy_.default(primals_548, add_443);  primals_548 = add_443 = None
    copy__212: "f32[184]" = torch.ops.aten.copy_.default(primals_549, add_444);  primals_549 = add_444 = None
    copy__213: "i64[]" = torch.ops.aten.copy_.default(primals_550, add_447);  primals_550 = add_447 = None
    copy__214: "f32[736]" = torch.ops.aten.copy_.default(primals_551, add_449);  primals_551 = add_449 = None
    copy__215: "f32[736]" = torch.ops.aten.copy_.default(primals_552, add_450);  primals_552 = add_450 = None
    copy__216: "i64[]" = torch.ops.aten.copy_.default(primals_553, add_453);  primals_553 = add_453 = None
    copy__217: "f32[736]" = torch.ops.aten.copy_.default(primals_554, add_455);  primals_554 = add_455 = None
    copy__218: "f32[736]" = torch.ops.aten.copy_.default(primals_555, add_456);  primals_555 = add_456 = None
    copy__219: "i64[]" = torch.ops.aten.copy_.default(primals_556, add_461);  primals_556 = add_461 = None
    copy__220: "f32[184]" = torch.ops.aten.copy_.default(primals_557, add_463);  primals_557 = add_463 = None
    copy__221: "f32[184]" = torch.ops.aten.copy_.default(primals_558, add_464);  primals_558 = add_464 = None
    copy__222: "i64[]" = torch.ops.aten.copy_.default(primals_559, add_467);  primals_559 = add_467 = None
    copy__223: "f32[736]" = torch.ops.aten.copy_.default(primals_560, add_469);  primals_560 = add_469 = None
    copy__224: "f32[736]" = torch.ops.aten.copy_.default(primals_561, add_470);  primals_561 = add_470 = None
    copy__225: "i64[]" = torch.ops.aten.copy_.default(primals_562, add_473);  primals_562 = add_473 = None
    copy__226: "f32[736]" = torch.ops.aten.copy_.default(primals_563, add_475);  primals_563 = add_475 = None
    copy__227: "f32[736]" = torch.ops.aten.copy_.default(primals_564, add_476);  primals_564 = add_476 = None
    copy__228: "i64[]" = torch.ops.aten.copy_.default(primals_565, add_481);  primals_565 = add_481 = None
    copy__229: "f32[184]" = torch.ops.aten.copy_.default(primals_566, add_483);  primals_566 = add_483 = None
    copy__230: "f32[184]" = torch.ops.aten.copy_.default(primals_567, add_484);  primals_567 = add_484 = None
    copy__231: "i64[]" = torch.ops.aten.copy_.default(primals_568, add_487);  primals_568 = add_487 = None
    copy__232: "f32[736]" = torch.ops.aten.copy_.default(primals_569, add_489);  primals_569 = add_489 = None
    copy__233: "f32[736]" = torch.ops.aten.copy_.default(primals_570, add_490);  primals_570 = add_490 = None
    copy__234: "i64[]" = torch.ops.aten.copy_.default(primals_571, add_493);  primals_571 = add_493 = None
    copy__235: "f32[736]" = torch.ops.aten.copy_.default(primals_572, add_495);  primals_572 = add_495 = None
    copy__236: "f32[736]" = torch.ops.aten.copy_.default(primals_573, add_496);  primals_573 = add_496 = None
    copy__237: "i64[]" = torch.ops.aten.copy_.default(primals_574, add_501);  primals_574 = add_501 = None
    copy__238: "f32[184]" = torch.ops.aten.copy_.default(primals_575, add_503);  primals_575 = add_503 = None
    copy__239: "f32[184]" = torch.ops.aten.copy_.default(primals_576, add_504);  primals_576 = add_504 = None
    copy__240: "i64[]" = torch.ops.aten.copy_.default(primals_577, add_507);  primals_577 = add_507 = None
    copy__241: "f32[736]" = torch.ops.aten.copy_.default(primals_578, add_509);  primals_578 = add_509 = None
    copy__242: "f32[736]" = torch.ops.aten.copy_.default(primals_579, add_510);  primals_579 = add_510 = None
    copy__243: "i64[]" = torch.ops.aten.copy_.default(primals_580, add_513);  primals_580 = add_513 = None
    copy__244: "f32[736]" = torch.ops.aten.copy_.default(primals_581, add_515);  primals_581 = add_515 = None
    copy__245: "f32[736]" = torch.ops.aten.copy_.default(primals_582, add_516);  primals_582 = add_516 = None
    copy__246: "i64[]" = torch.ops.aten.copy_.default(primals_583, add_521);  primals_583 = add_521 = None
    copy__247: "f32[184]" = torch.ops.aten.copy_.default(primals_584, add_523);  primals_584 = add_523 = None
    copy__248: "f32[184]" = torch.ops.aten.copy_.default(primals_585, add_524);  primals_585 = add_524 = None
    copy__249: "i64[]" = torch.ops.aten.copy_.default(primals_586, add_527);  primals_586 = add_527 = None
    copy__250: "f32[1104]" = torch.ops.aten.copy_.default(primals_587, add_529);  primals_587 = add_529 = None
    copy__251: "f32[1104]" = torch.ops.aten.copy_.default(primals_588, add_530);  primals_588 = add_530 = None
    copy__252: "i64[]" = torch.ops.aten.copy_.default(primals_589, add_533);  primals_589 = add_533 = None
    copy__253: "f32[1104]" = torch.ops.aten.copy_.default(primals_590, add_535);  primals_590 = add_535 = None
    copy__254: "f32[1104]" = torch.ops.aten.copy_.default(primals_591, add_536);  primals_591 = add_536 = None
    copy__255: "i64[]" = torch.ops.aten.copy_.default(primals_592, add_541);  primals_592 = add_541 = None
    copy__256: "f32[224]" = torch.ops.aten.copy_.default(primals_593, add_543);  primals_593 = add_543 = None
    copy__257: "f32[224]" = torch.ops.aten.copy_.default(primals_594, add_544);  primals_594 = add_544 = None
    copy__258: "i64[]" = torch.ops.aten.copy_.default(primals_595, add_546);  primals_595 = add_546 = None
    copy__259: "f32[1344]" = torch.ops.aten.copy_.default(primals_596, add_548);  primals_596 = add_548 = None
    copy__260: "f32[1344]" = torch.ops.aten.copy_.default(primals_597, add_549);  primals_597 = add_549 = None
    return pytree.tree_unflatten([addmm, mul_1617, sum_192, mul_1607, sum_190, mul_1597, sum_188, mul_1588, sum_186, mul_1578, sum_184, mul_1569, sum_182, mul_1559, sum_180, mul_1549, sum_178, mul_1540, sum_176, mul_1530, sum_174, mul_1520, sum_172, mul_1511, sum_170, mul_1501, sum_168, mul_1491, sum_166, mul_1482, sum_164, mul_1472, sum_162, mul_1462, sum_160, mul_1453, sum_158, mul_1443, sum_156, mul_1429, sum_153, mul_1420, sum_151, mul_1410, sum_149, mul_1396, sum_146, mul_1387, sum_144, mul_1377, sum_142, mul_1363, sum_139, mul_1354, sum_137, mul_1344, sum_135, mul_1330, sum_132, mul_1321, sum_130, mul_1311, sum_128, mul_1297, sum_125, mul_1288, sum_123, mul_1278, sum_121, mul_1268, sum_119, mul_1259, sum_117, mul_1249, sum_115, mul_1239, sum_113, mul_1230, sum_111, mul_1220, sum_109, mul_1210, sum_107, mul_1201, sum_105, mul_1191, sum_103, mul_1181, sum_101, mul_1172, sum_99, mul_1162, sum_97, mul_1152, sum_95, mul_1143, sum_93, mul_1133, sum_91, mul_1119, sum_88, mul_1110, sum_86, mul_1100, sum_84, mul_1086, sum_81, mul_1077, sum_79, mul_1067, sum_77, mul_1053, sum_74, mul_1044, sum_72, mul_1034, sum_70, mul_1020, sum_67, mul_1011, sum_65, mul_1001, sum_63, mul_987, sum_60, mul_978, sum_58, mul_968, sum_56, mul_954, sum_53, mul_945, sum_51, mul_935, sum_49, mul_921, sum_46, mul_912, sum_44, mul_902, sum_42, mul_888, sum_39, mul_879, sum_37, mul_869, sum_35, mul_855, sum_32, mul_846, sum_30, mul_836, sum_28, mul_822, sum_25, mul_813, sum_23, mul_803, sum_21, mul_789, sum_18, mul_780, sum_16, mul_770, sum_14, mul_756, sum_11, mul_747, sum_9, mul_737, sum_7, mul_723, sum_4, mul_714, sum_2, permute_4, view_2, getitem_544, getitem_541, getitem_538, getitem_535, getitem_532, getitem_529, getitem_526, getitem_523, getitem_520, getitem_517, getitem_514, getitem_511, getitem_508, getitem_505, getitem_502, getitem_499, getitem_496, getitem_493, getitem_490, getitem_487, getitem_488, getitem_484, getitem_485, getitem_481, getitem_478, getitem_475, getitem_472, getitem_473, getitem_469, getitem_470, getitem_466, getitem_463, getitem_460, getitem_457, getitem_458, getitem_454, getitem_455, getitem_451, getitem_448, getitem_445, getitem_442, getitem_443, getitem_439, getitem_440, getitem_436, getitem_433, getitem_430, getitem_427, getitem_428, getitem_424, getitem_425, getitem_421, getitem_418, getitem_415, getitem_412, getitem_409, getitem_406, getitem_403, getitem_400, getitem_397, getitem_394, getitem_391, getitem_388, getitem_385, getitem_382, getitem_379, getitem_376, getitem_373, getitem_370, getitem_367, getitem_368, getitem_364, getitem_365, getitem_361, getitem_358, getitem_355, getitem_352, getitem_353, getitem_349, getitem_350, getitem_346, getitem_343, getitem_340, getitem_337, getitem_338, getitem_334, getitem_335, getitem_331, getitem_328, getitem_325, getitem_322, getitem_323, getitem_319, getitem_320, getitem_316, getitem_313, getitem_310, getitem_307, getitem_308, getitem_304, getitem_305, getitem_301, getitem_298, getitem_295, getitem_292, getitem_293, getitem_289, getitem_290, getitem_286, getitem_283, getitem_280, getitem_277, getitem_278, getitem_274, getitem_275, getitem_271, getitem_268, getitem_265, getitem_262, getitem_263, getitem_259, getitem_260, getitem_256, getitem_253, getitem_250, getitem_247, getitem_248, getitem_244, getitem_245, getitem_241, getitem_238, getitem_235, getitem_232, getitem_233, getitem_229, getitem_230, getitem_226, getitem_223, getitem_220, getitem_217, getitem_218, getitem_214, getitem_215, getitem_211, getitem_208, getitem_205, getitem_202, getitem_203, getitem_199, getitem_200, getitem_196, getitem_193, getitem_190, getitem_187, getitem_188, getitem_184, getitem_185, getitem_181, getitem_178, getitem_175, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    